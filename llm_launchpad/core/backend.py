"""Modal subprocess invocation and streaming wrappers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import shlex
import subprocess
import threading
from typing import Any, Dict, Generator, List, Optional

from ..protocol.enums import BackendType
from ..protocol.events import ErrorEvent, LogEvent, OperationCompleteEvent
from ..protocol.models import DeploymentConfig, EndpointInfo, LaunchpadSettings
from .modal_auth import get_modal_profile
from .modal_cli import _CLI_TIMEOUT_SECONDS as _MODAL_CLI_TIMEOUT_SECONDS
from .modal_cli import resolve_modal_cli_path
from .shutdown import is_shutting_down as _is_shutting_down
from .shutdown import shutdown_event
from .diagnostics import log_exception
from .naming import default_served_model_name
from .naming import infer_backend_from_app_name, infer_instance_from_app_name, legacy_app_name
from .naming import modal_function_name, modal_web_label


@dataclass(frozen=True)
class ModalCliError:
    """Structured diagnostics for a failed Modal CLI call."""

    message: str
    command: tuple[str, ...] = ()
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


@dataclass(frozen=True)
class ModalListAppsResult:
    """Result from ``modal app list --json``."""

    rows: list[EndpointInfo] | None = None
    error: ModalCliError | None = None

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class ModalTextResult:
    """Result from a Modal CLI call that returns text."""

    output: str | None = None
    error: ModalCliError | None = None

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class ModalVolumeListResult:
    """Result from ``modal volume ls --json``."""

    entries: list[dict[str, Any]] | None = None
    error: ModalCliError | None = None

    @property
    def success(self) -> bool:
        return self.error is None


class ModalBackend:
    """Wrapper around Modal CLI subprocess calls."""

    # ------------------------------------------------------------------
    # Subprocess lifecycle tracking
    # ------------------------------------------------------------------

    _active_procs: set[subprocess.Popen] = set()
    _active_procs_lock = threading.Lock()
    _shutdown_event = shutdown_event()
    _CLI_TIMEOUT_SECONDS = _MODAL_CLI_TIMEOUT_SECONDS

    @staticmethod
    def modal_cli_path() -> Optional[str]:
        """Resolve the Modal CLI, preferring the active environment's scripts dir."""
        return resolve_modal_cli_path()

    @classmethod
    def _resolve_command(cls, command: List[str]) -> List[str]:
        if command and command[0] == "modal":
            modal_cli = cls.modal_cli_path()
            if modal_cli:
                return [modal_cli, *command[1:]]
        return command

    @classmethod
    def register_proc(cls, proc: subprocess.Popen) -> None:
        """Track a subprocess so it can be terminated on shutdown."""
        with cls._active_procs_lock:
            cls._active_procs.add(proc)

    @classmethod
    def unregister_proc(cls, proc: subprocess.Popen) -> None:
        """Remove a subprocess from tracking."""
        with cls._active_procs_lock:
            cls._active_procs.discard(proc)

    @classmethod
    def terminate_all(cls) -> None:
        """Signal shutdown and terminate all tracked subprocesses.

        Called on app exit to unblock any worker threads that are waiting
        on subprocess I/O so Python can shut down cleanly.
        """
        from .shutdown import request_shutdown

        request_shutdown()
        import time
        time.sleep(0.05)
        with cls._active_procs_lock:
            procs = list(cls._active_procs)
            cls._active_procs.clear()
        for proc in procs:
            try:
                proc.terminate()
                if proc.stdout:
                    proc.stdout.close()
                if proc.stderr and proc.stderr != subprocess.PIPE:
                    proc.stderr.close()
            except Exception:
                log_exception("Failed to terminate a tracked Modal subprocess")

    # ------------------------------------------------------------------
    # Pre-flight checks
    # ------------------------------------------------------------------

    @staticmethod
    def is_cli_available() -> bool:
        return ModalBackend.modal_cli_path() is not None

    @staticmethod
    def get_username() -> Optional[str]:
        """Return the current Modal profile/workspace slug, or None."""
        return get_modal_profile()

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------

    @staticmethod
    def default_server_url(
        username: str,
        backend: Optional[BackendType] = None,
        app_name: Optional[str] = None,
        function_slug: Optional[str] = None,
    ) -> str:
        resolved_backend = backend or infer_backend_from_app_name(app_name or "")
        if resolved_backend is None:
            resolved_backend = BackendType.LLAMACPP
        resolved = app_name or legacy_app_name(resolved_backend)
        if resolved_backend == BackendType.LLAMACPP and function_slug:
            label = modal_web_label(resolved, function_slug)
            return f"https://{username}--{label}.modal.run"
        serve_name = modal_function_name("serve", function_slug)
        return f"https://{username}--{resolved}-{serve_name}.modal.run"

    @staticmethod
    def extract_modal_web_url(line: str) -> Optional[str]:
        """Extract the first Modal web URL from an output line."""
        match = re.search(r"https://[A-Za-z0-9-]+\.modal\.run", line or "")
        return match.group(0) if match else None

    @staticmethod
    def test_curl_command(
        backend: BackendType,
        server_url: str,
        served_model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> str:
        base = server_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")].rstrip("/")
        content = "Say hello in one short sentence."

        def _curl_json(endpoint: str, payload: Dict[str, Any]) -> str:
            body = shlex.quote(json.dumps(payload, separators=(",", ":")))
            auth_header = ""
            if api_key:
                auth_header = f" -H {shlex.quote(f'Authorization: Bearer {api_key}')}"
            return (
                f"curl -s -X POST {base}{endpoint} "
                "-H 'Content-Type: application/json' "
                f"-d {body}{auth_header}"
            )

        if backend == BackendType.VLLM:
            model = (served_model_name or "").strip() or os.environ.get(
                "SERVED_MODEL_NAME",
                default_served_model_name(os.environ.get("MODEL_NAME")),
            )
            return _curl_json(
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": content}],
                },
            )
        model = (served_model_name or "").strip() or "default"
        return _curl_json(
            "/v1/completions",
            {
                "model": model,
                "prompt": content,
                "max_tokens": 32,
            },
        )

    # ------------------------------------------------------------------
    # Environment builders
    # ------------------------------------------------------------------

    @staticmethod
    def env_for_backend(config: DeploymentConfig) -> Dict[str, str]:
        """Derive backend-specific env vars from a deployment config."""
        env: Dict[str, str] = {}
        if config.app_name:
            env["MODAL_APP_NAME"] = config.app_name
        if config.function_slug:
            env["MODAL_FUNCTION_SLUG"] = config.function_slug
        if config.backend == BackendType.LLAMACPP:
            if config.llamacpp_image_no_cache is not None:
                env["LLAMA_CPP_IMAGE_NO_CACHE"] = "true" if config.llamacpp_image_no_cache else "false"
            return env
        if config.backend != BackendType.VLLM:
            return env
        if config.model_name:
            env["MODEL_NAME"] = config.model_name
        if config.model_revision:
            env["MODEL_REVISION"] = config.model_revision
        if config.served_model_name:
            env["SERVED_MODEL_NAME"] = config.served_model_name
        if config.fast_boot is not None:
            env["FAST_BOOT"] = "true" if config.fast_boot else "false"
        if config.n_gpu is not None and config.n_gpu > 0:
            env["N_GPU"] = str(config.n_gpu)
        if config.trust_remote_code is not None:
            env["TRUST_REMOTE_CODE"] = "true" if config.trust_remote_code else "false"
        if config.reasoning_parser:
            env["REASONING_PARSER"] = config.reasoning_parser
        if config.tool_call_parser:
            env["TOOL_CALL_PARSER"] = config.tool_call_parser
        if config.default_chat_template_kwargs:
            env["DEFAULT_CHAT_TEMPLATE_KWARGS"] = config.default_chat_template_kwargs
        return env

    @staticmethod
    def build_full_env(
        settings: LaunchpadSettings,
        config: DeploymentConfig,
    ) -> Dict[str, str]:
        env = settings.to_env()
        gpu_type = (config.gpu_type or "").strip()
        if gpu_type and config.gpu_count is not None and config.gpu_count > 0:
            env["GPU_CONFIG"] = f"{gpu_type.upper()}:{config.gpu_count}"
        env.update(ModalBackend.env_for_backend(config))
        return env

    # ------------------------------------------------------------------
    # Command builders
    # ------------------------------------------------------------------

    @staticmethod
    def build_deploy_command(backend: BackendType, app_name: Optional[str] = None) -> List[str]:
        cmd = ["modal", "deploy", "-m", backend.script]
        if app_name:
            cmd += ["--name", app_name]
        return cmd

    @staticmethod
    def build_run_command(config: DeploymentConfig) -> List[str]:
        if config.backend == BackendType.VLLM:
            return ["modal", "run", "-m", BackendType.VLLM.script]

        args: List[str] = ["modal", "run", "-m", f"{BackendType.LLAMACPP.script}::main"]
        if config.preset:
            args += ["--preset", config.preset]
        if config.repo_id:
            args += ["--repo-id", config.repo_id]
        if config.quant:
            args += ["--quant", config.quant]
        if config.revision:
            args += ["--revision", config.revision]
        if config.preload:
            args += ["--preload"]
        else:
            args += ["--no-preload"]
        if config.do_deploy:
            args += ["--deploy"]
        if config.server_args:
            args.append(f"--server-args={config.server_args}")
        if config.host:
            args += ["--host", config.host]
        if config.port is not None:
            args += ["--port", str(config.port)]
        if config.n_gpu_layers is not None:
            args += ["--n-gpu-layers", str(config.n_gpu_layers)]
        if config.placement_assessment is not None:
            args += ["--serving-fingerprint", config.placement_assessment.fingerprint]
        return args

    @staticmethod
    def build_modal_entrypoint_command(
        script: str,
        entrypoint: str,
        args: Optional[List[str]] = None,
    ) -> List[str]:
        cmd = ["modal", "run", "-m", f"{script}::{entrypoint}"]
        if args:
            cmd.extend(args)
        return cmd

    # ------------------------------------------------------------------
    # Log follow helper
    # ------------------------------------------------------------------

    @staticmethod
    def logs_follow_args() -> List[str]:
        """Best available follow flag for ``modal app logs``."""
        try:
            res = subprocess.run(
                ModalBackend._resolve_command(["modal", "app", "logs", "--help"]),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=ModalBackend._CLI_TIMEOUT_SECONDS,
            )
            text = (res.stdout or "") + (res.stderr or "")
            if "--follow" in text:
                return ["--follow"]
            if "-f" in text:
                return ["-f"]
        except subprocess.TimeoutExpired:
            return []
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------
    # App listing
    # ------------------------------------------------------------------

    @staticmethod
    def list_apps() -> Optional[List[EndpointInfo]]:
        """Return parsed app list (possibly empty), or None on failure."""
        result = ModalBackend.list_apps_result()
        return result.rows if result.success else None

    @staticmethod
    def list_apps_result() -> ModalListAppsResult:
        """Return parsed app list plus diagnostics on failure."""
        command = ["modal", "app", "list", "--json"]
        try:
            result = subprocess.run(
                ModalBackend._resolve_command(command),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=ModalBackend._CLI_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return ModalListAppsResult(
                rows=None,
                error=ModalCliError(
                    message="Timed out while querying Modal app list.",
                    command=tuple(command),
                    timed_out=True,
                ),
            )
        except Exception as exc:
            return ModalListAppsResult(
                rows=None,
                error=ModalCliError(
                    message=f"Failed to query Modal app list: {exc}",
                    command=tuple(command),
                ),
            )
        if result.returncode != 0:
            return ModalListAppsResult(
                rows=None,
                error=_modal_cli_error_from_completed_process(
                    command,
                    result,
                    fallback_message="Modal app list failed.",
                ),
            )
        try:
            payload: Any = json.loads(result.stdout or "[]")
        except Exception as exc:
            return ModalListAppsResult(
                rows=None,
                error=ModalCliError(
                    message=f"Modal app list returned invalid JSON: {exc}",
                    command=tuple(command),
                    exit_code=result.returncode,
                    stdout=result.stdout or "",
                    stderr=result.stderr or "",
                ),
            )

        return ModalListAppsResult(rows=_extract_modal_app_rows(payload), error=None)

    @staticmethod
    def list_apps_raw_result() -> ModalTextResult:
        """Fallback: return raw app-list text plus diagnostics on failure."""
        command = ["modal", "app", "list"]
        try:
            result = subprocess.run(
                ModalBackend._resolve_command(command),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=ModalBackend._CLI_TIMEOUT_SECONDS,
            )
            if result.returncode == 0:
                return ModalTextResult(output=result.stdout or "", error=None)
            return ModalTextResult(
                output=None,
                error=_modal_cli_error_from_completed_process(
                    command,
                    result,
                    fallback_message="Modal app list failed.",
                ),
            )
        except subprocess.TimeoutExpired:
            return ModalTextResult(
                output=None,
                error=ModalCliError(
                    message="Timed out while querying Modal app list.",
                    command=tuple(command),
                    timed_out=True,
                ),
            )
        except Exception as exc:
            return ModalTextResult(
                output=None,
                error=ModalCliError(
                    message=f"Failed to query Modal app list: {exc}",
                    command=tuple(command),
                ),
            )

    @staticmethod
    def billing_report_json() -> tuple[Optional[Any], Optional[str]]:
        """Return billing report JSON and an optional error message."""
        commands = [
            ["modal", "billing", "report", "--for", "this month", "--json"],
            ["modal", "workspace", "billing", "report", "--for", "this month", "--json"],
        ]

        def _looks_like_unsupported_command(message: str) -> bool:
            normalized = (message or "").lower()
            return (
                "no such command" in normalized
                or "invalid choice" in normalized
                or "unrecognized arguments" in normalized
                or "usage: modal [options] command" in normalized
            )

        last_error: Optional[str] = None
        unsupported_command_seen = False
        for command in commands:
            try:
                result = subprocess.run(
                    ModalBackend._resolve_command(command),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=ModalBackend._CLI_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired:
                return None, "Timed out while requesting billing report."
            except FileNotFoundError:
                return None, "Modal CLI not found in PATH."
            except Exception as exc:
                return None, str(exc)

            if result.returncode == 0:
                try:
                    return json.loads(result.stdout or "{}"), None
                except Exception:
                    return None, "Billing report returned invalid JSON."

            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            details = stderr or stdout or f"Exit code {result.returncode}"
            last_error = details
            if _looks_like_unsupported_command(details):
                unsupported_command_seen = True
                continue
            return None, details

        if unsupported_command_seen:
            return (
                None,
                "Billing report command is unavailable in this Modal CLI version. "
                "Upgrade Modal CLI and retry.",
            )

        return None, last_error or "Could not read billing report."

    @staticmethod
    def list_volume(volume_name: str, path: str = "/") -> Optional[List[Dict[str, Any]]]:
        """List files/directories in a Modal Volume path."""
        result = ModalBackend.list_volume_result(volume_name, path)
        return result.entries if result.success else None

    @staticmethod
    def list_volume_result(volume_name: str, path: str = "/") -> ModalVolumeListResult:
        """List Modal Volume entries plus diagnostics on failure."""
        if _is_shutting_down():
            return ModalVolumeListResult(entries=[], error=None)
        command = ["modal", "volume", "ls", volume_name, path, "--json"]
        try:
            result = subprocess.run(
                ModalBackend._resolve_command(command),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=ModalBackend._CLI_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return ModalVolumeListResult(
                entries=None,
                error=ModalCliError(
                    message=f"Timed out while listing Modal volume '{volume_name}' at {path}.",
                    command=tuple(command),
                    timed_out=True,
                ),
            )
        except Exception as exc:
            return ModalVolumeListResult(
                entries=None,
                error=ModalCliError(
                    message=f"Failed to list Modal volume '{volume_name}' at {path}: {exc}",
                    command=tuple(command),
                ),
            )
        if result.returncode != 0:
            return ModalVolumeListResult(
                entries=None,
                error=_modal_cli_error_from_completed_process(
                    command,
                    result,
                    fallback_message=f"Modal volume list failed for '{volume_name}' at {path}.",
                ),
            )
        try:
            payload = json.loads(result.stdout or "[]")
        except Exception as exc:
            return ModalVolumeListResult(
                entries=None,
                error=ModalCliError(
                    message=f"Modal volume list returned invalid JSON for '{volume_name}' at {path}: {exc}",
                    command=tuple(command),
                    exit_code=result.returncode,
                    stdout=result.stdout or "",
                    stderr=result.stderr or "",
                ),
            )
        if isinstance(payload, list):
            return ModalVolumeListResult(
                entries=[entry for entry in payload if isinstance(entry, dict)],
                error=None,
            )
        if isinstance(payload, dict):
            nested = payload.get("entries") or payload.get("items") or []
            if isinstance(nested, list):
                return ModalVolumeListResult(
                    entries=[entry for entry in nested if isinstance(entry, dict)],
                    error=None,
                )
        return ModalVolumeListResult(
            entries=None,
            error=ModalCliError(
                message=f"Modal volume list returned an unsupported JSON shape for '{volume_name}' at {path}.",
                command=tuple(command),
                exit_code=result.returncode,
                stdout=result.stdout or "",
                stderr=result.stderr or "",
            ),
        )

    # ------------------------------------------------------------------
    # Streaming subprocess execution
    # ------------------------------------------------------------------

    @staticmethod
    def run_streaming(
        command: List[str],
        env: Optional[Dict[str, str]] = None,
    ) -> Generator[LogEvent | OperationCompleteEvent | ErrorEvent, None, None]:
        """Run *command* and yield protocol events for each output line."""
        merged = {**os.environ, **(env or {})}
        try:
            proc = subprocess.Popen(
                ModalBackend._resolve_command(command),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=merged,
                bufsize=1,
            )
        except Exception as exc:
            yield ErrorEvent(message=str(exc), exit_code=1, recoverable=False)
            return

        ModalBackend.register_proc(proc)
        try:
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                yield LogEvent(line=raw_line.rstrip("\n"))

            proc.wait()
            if proc.returncode == 0:
                yield OperationCompleteEvent(success=True, exit_code=0)
            else:
                yield OperationCompleteEvent(
                    success=False,
                    exit_code=proc.returncode,
                    detail=f"Process exited with code {proc.returncode}",
                )
        finally:
            ModalBackend.unregister_proc(proc)

    @staticmethod
    def run_modal_script_entrypoint(
        script: str,
        entrypoint: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> Generator[LogEvent | OperationCompleteEvent | ErrorEvent, None, None]:
        """Run `modal run <script>::<entrypoint>` and stream output."""
        cmd = ModalBackend.build_modal_entrypoint_command(script, entrypoint, args=args)
        yield from ModalBackend.run_streaming(cmd, env=env)

    @staticmethod
    def run_modal_script_entrypoint_capture(
        script: str,
        entrypoint: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> Optional[tuple[int, str, str]]:
        """Run `modal run <script>::<entrypoint>` and capture stdout/stderr."""
        cmd = ModalBackend.build_modal_entrypoint_command(script, entrypoint, args=args)
        merged = {**os.environ, **(env or {})}
        try:
            result = subprocess.run(
                ModalBackend._resolve_command(cmd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=merged,
                timeout=ModalBackend._CLI_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return None
        except Exception:
            return None
        return result.returncode, result.stdout or "", result.stderr or ""

    @staticmethod
    def run_volume_remove(
        volume_name: str,
        remote_path: str,
        recursive: bool = True,
    ) -> Generator[LogEvent | OperationCompleteEvent | ErrorEvent, None, None]:
        """Delete a file/directory from a Modal Volume."""
        cmd = ["modal", "volume", "rm", volume_name, remote_path]
        if recursive:
            cmd.append("--recursive")
        yield from ModalBackend.run_streaming(cmd)

    @staticmethod
    def run_blocking(
        command: List[str],
        env: Optional[Dict[str, str]] = None,
    ) -> int:
        """Run *command* blocking and return exit code."""
        merged = None
        if env:
            merged = {**os.environ, **env}
        result = subprocess.run(ModalBackend._resolve_command(command), text=True, env=merged)
        return result.returncode


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _modal_cli_error_from_completed_process(
    command: list[str],
    result: subprocess.CompletedProcess[str],
    *,
    fallback_message: str,
) -> ModalCliError:
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    details = stderr or stdout or f"Exit code {result.returncode}"
    return ModalCliError(
        message=f"{fallback_message} {details}",
        command=tuple(command),
        exit_code=result.returncode,
        stdout=result.stdout or "",
        stderr=result.stderr or "",
    )


def _non_empty_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _nested_string_for_keys(
    value: Any,
    keys: set[str],
    *,
    max_depth: int = 6,
) -> Optional[str]:
    """Depth-limited search for a string value by key name (case-insensitive)."""
    if max_depth < 0:
        return None

    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).strip().lower() in keys:
                text = _non_empty_text(child)
                if text:
                    return text
        for child in value.values():
            found = _nested_string_for_keys(child, keys, max_depth=max_depth - 1)
            if found:
                return found
        return None

    if isinstance(value, list):
        for child in value:
            found = _nested_string_for_keys(child, keys, max_depth=max_depth - 1)
            if found:
                return found
        return None

    return None


def _collect_modal_web_urls(value: Any, *, max_depth: int = 6) -> List[str]:
    """Collect Modal web URLs found anywhere in a nested JSON-like payload."""
    if max_depth < 0:
        return []

    urls: List[str] = []
    if isinstance(value, dict):
        for child in value.values():
            urls.extend(_collect_modal_web_urls(child, max_depth=max_depth - 1))
        return urls

    if isinstance(value, list):
        for child in value:
            urls.extend(_collect_modal_web_urls(child, max_depth=max_depth - 1))
        return urls

    if isinstance(value, str):
        maybe_url = ModalBackend.extract_modal_web_url(value)
        if maybe_url:
            urls.append(maybe_url)
    return urls


def _extract_item_web_url(item: dict[str, Any]) -> Optional[str]:
    """Best-effort extraction of a Modal web URL from one app-list row."""
    for key in ("web_url", "webUrl", "url", "URL"):
        candidate = item.get(key)
        if isinstance(candidate, str):
            maybe_url = ModalBackend.extract_modal_web_url(candidate)
            if maybe_url:
                return maybe_url

    urls = _collect_modal_web_urls(item)
    if not urls:
        return None

    unique_urls = list(dict.fromkeys(urls))
    unique_urls.sort(key=lambda url: (url.endswith("-dev.modal.run"), len(url)))
    return unique_urls[0]


def _extract_modal_app_rows(payload: Any) -> List[EndpointInfo]:
    rows: List[EndpointInfo] = []
    if isinstance(payload, dict):
        if isinstance(payload.get("apps"), list):
            payload = payload["apps"]
        elif isinstance(payload.get("data"), list):
            payload = payload["data"]
    if not isinstance(payload, list):
        return rows

    for item in payload:
        if not isinstance(item, dict):
            continue
        name = str(
            item.get("name")
            or item.get("app_name")
            or item.get("app")
            or item.get("label")
            or item.get("description")
            or item.get("Description")
            or ""
        ).strip()
        app_id = str(
            item.get("app_id")
            or item.get("id")
            or item.get("appId")
            or item.get("App ID")
            or ""
        ).strip()
        state = str(
            item.get("state")
            or item.get("status")
            or item.get("phase")
            or item.get("State")
            or "unknown"
        ).strip()
        served_model_name = _nested_string_for_keys(item, {"served_model_name"})
        model_name = _nested_string_for_keys(item, {"model_name"})
        repo_id = _nested_string_for_keys(item, {"repo_id", "model_repo_id"})
        quant = _nested_string_for_keys(item, {"quant"})
        backend = infer_backend_from_app_name(name)
        if backend is None:
            if model_name:
                backend = BackendType.VLLM
            elif repo_id or quant:
                backend = BackendType.LLAMACPP
        instance = infer_instance_from_app_name(name, backend)
        rows.append(
            EndpointInfo(
                name=name,
                app_id=app_id,
                state=state,
                backend=backend,
                instance_name=instance,
                web_url=_extract_item_web_url(item),
                served_model_name=served_model_name,
                model_name=model_name,
                repo_id=repo_id,
                quant=quant,
            )
        )
    return rows
