"""Modal subprocess invocation and streaming wrappers."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import threading
from typing import Any, Dict, Generator, List, Optional

from ..protocol.enums import BackendType
from ..protocol.events import ErrorEvent, LogEvent, OperationCompleteEvent
from ..protocol.models import DeploymentConfig, EndpointInfo, LaunchpadSettings
from .naming import default_served_model_name
from .naming import infer_backend_from_app_name, infer_instance_from_app_name, legacy_app_name
from .naming import modal_function_name


class ModalBackend:
    """Wrapper around Modal CLI subprocess calls."""

    # ------------------------------------------------------------------
    # Subprocess lifecycle tracking
    # ------------------------------------------------------------------

    _active_procs: set[subprocess.Popen] = set()
    _active_procs_lock = threading.Lock()
    _shutdown_event = threading.Event()
    _CLI_TIMEOUT_SECONDS = 8.0

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
        cls._shutdown_event.set()
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
                pass

    @classmethod
    def is_shutting_down(cls) -> bool:
        """Return True if a shutdown has been requested."""
        return cls._shutdown_event.is_set()

    # ------------------------------------------------------------------
    # Pre-flight checks
    # ------------------------------------------------------------------

    @staticmethod
    def is_cli_available() -> bool:
        return shutil.which("modal") is not None

    @staticmethod
    def get_username() -> Optional[str]:
        """Return the current Modal profile username, or None."""
        try:
            res = subprocess.run(
                ["modal", "profile", "current"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=ModalBackend._CLI_TIMEOUT_SECONDS,
            )
            username = (res.stdout or "").strip()
            if res.returncode == 0 and username:
                return username
        except subprocess.TimeoutExpired:
            return None
        except Exception:
            pass
        return None

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
        resolved = app_name or legacy_app_name(backend or BackendType.LLAMACPP)
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
    ) -> str:
        base = server_url.rstrip("/")
        content = "Say hello in one short sentence."

        def _curl_json(endpoint: str, payload: Dict[str, Any]) -> str:
            body = shlex.quote(json.dumps(payload, separators=(",", ":")))
            return (
                f"curl -s -X POST {base}{endpoint} "
                "-H 'Content-Type: application/json' "
                f"-d {body}"
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
        cmd = ["modal", "deploy", backend.script]
        if app_name:
            cmd += ["--name", app_name]
        return cmd

    @staticmethod
    def build_run_command(config: DeploymentConfig) -> List[str]:
        if config.backend == BackendType.VLLM:
            return ["modal", "run", BackendType.VLLM.script]

        args: List[str] = ["modal", "run", f"{BackendType.LLAMACPP.script}::main"]
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
            args += ["--server_args", config.server_args]
        if config.host:
            args += ["--host", config.host]
        if config.port is not None:
            args += ["--port", str(config.port)]
        if config.n_gpu_layers is not None:
            args += ["--n_gpu_layers", str(config.n_gpu_layers)]
        return args

    @staticmethod
    def build_modal_entrypoint_command(
        script: str,
        entrypoint: str,
        args: Optional[List[str]] = None,
    ) -> List[str]:
        cmd = ["modal", "run", f"{script}::{entrypoint}"]
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
                ["modal", "app", "logs", "--help"],
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
        try:
            result = subprocess.run(
                ["modal", "app", "list", "--json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=ModalBackend._CLI_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return None
        except Exception:
            return None
        if result.returncode != 0:
            return None
        try:
            payload: Any = json.loads(result.stdout or "[]")
        except Exception:
            return None

        return _extract_modal_app_rows(payload)

    @staticmethod
    def list_apps_raw() -> Optional[str]:
        """Fallback: return raw text from ``modal app list``."""
        try:
            result = subprocess.run(
                ["modal", "app", "list"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=ModalBackend._CLI_TIMEOUT_SECONDS,
            )
            if result.returncode == 0:
                return result.stdout or ""
        except subprocess.TimeoutExpired:
            return None
        except Exception:
            pass
        return None

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
                    command,
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
        if ModalBackend.is_shutting_down():
            return []
        try:
            result = subprocess.run(
                ["modal", "volume", "ls", volume_name, path, "--json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=ModalBackend._CLI_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return None
        except Exception:
            return None
        if result.returncode != 0:
            return None
        try:
            payload = json.loads(result.stdout or "[]")
        except Exception:
            return None
        if isinstance(payload, list):
            return [entry for entry in payload if isinstance(entry, dict)]
        if isinstance(payload, dict):
            nested = payload.get("entries") or payload.get("items") or []
            if isinstance(nested, list):
                return [entry for entry in nested if isinstance(entry, dict)]
        return None

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
                command,
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
                cmd,
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
        result = subprocess.run(command, text=True, env=merged)
        return result.returncode


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

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
        backend = infer_backend_from_app_name(name)
        instance = infer_instance_from_app_name(name, backend)
        rows.append(
            EndpointInfo(
                name=name,
                app_id=app_id,
                state=state,
                backend=backend,
                instance_name=instance,
            )
        )
    return rows
