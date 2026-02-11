"""Modal subprocess invocation and streaming wrappers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any, Dict, Generator, List, Optional

from ..protocol.enums import BackendType
from ..protocol.events import ErrorEvent, LogEvent, OperationCompleteEvent
from ..protocol.models import DeploymentConfig, EndpointInfo, LaunchpadSettings
from ..presets import PRESETS


class ModalBackend:
    """Wrapper around Modal CLI subprocess calls."""

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
            )
            username = (res.stdout or "").strip()
            if res.returncode == 0 and username:
                return username
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------

    @staticmethod
    def default_server_url(username: str, backend: BackendType) -> str:
        return f"https://{username}--{backend.app_name}-serve.modal.run"

    @staticmethod
    def server_example_url(backend: BackendType) -> str:
        return f"https://<user>--{backend.app_name}-serve.modal.run"

    @staticmethod
    def test_curl_command(backend: BackendType, server_url: str) -> str:
        base = server_url.rstrip("/")
        if backend == BackendType.VLLM:
            model = os.environ.get("SERVED_MODEL_NAME", "llm")
            return (
                f"curl -s -X POST {base}/v1/chat/completions "
                "-H 'Content-Type: application/json' "
                f"-d '{{\"model\":\"{model}\",\"messages\":[{{\"role\":\"user\","
                "\"content\":\"Say hello in one short sentence.\"}}]}}'"
            )
        return (
            f"curl -s -X POST {base}/v1/completions "
            "-H 'Content-Type: application/json' "
            "-d '{\"model\":\"default\",\"prompt\":\"Say hello in one short sentence.\","
            "\"max_tokens\":32}'"
        )

    # ------------------------------------------------------------------
    # Environment builders
    # ------------------------------------------------------------------

    @staticmethod
    def env_for_backend(config: DeploymentConfig) -> Dict[str, str]:
        """Derive backend-specific env vars from a deployment config."""
        env: Dict[str, str] = {}
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
        return env

    @staticmethod
    def build_full_env(
        settings: LaunchpadSettings,
        config: DeploymentConfig,
    ) -> Dict[str, str]:
        env = settings.to_env()
        env.update(ModalBackend.env_for_backend(config))
        return env

    # ------------------------------------------------------------------
    # Command builders
    # ------------------------------------------------------------------

    @staticmethod
    def build_deploy_command(backend: BackendType) -> List[str]:
        return ["modal", "deploy", backend.script]

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
            )
            text = (res.stdout or "") + (res.stderr or "")
            if "--follow" in text:
                return ["--follow"]
            if "-f" in text:
                return ["-f"]
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------
    # App listing
    # ------------------------------------------------------------------

    @staticmethod
    def list_apps() -> Optional[List[EndpointInfo]]:
        """Return parsed app list, or None on failure."""
        try:
            result = subprocess.run(
                ["modal", "app", "list", "--json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None
        try:
            payload: Any = json.loads(result.stdout or "[]")
        except Exception:
            return None

        rows = _extract_modal_app_rows(payload)
        return rows if rows else None

    @staticmethod
    def list_apps_raw() -> Optional[str]:
        """Fallback: return raw text from ``modal app list``."""
        try:
            result = subprocess.run(
                ["modal", "app", "list"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout or ""
        except Exception:
            pass
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

_LAUNCHPAD_APPS = {
    BackendType.LLAMACPP: BackendType.LLAMACPP.app_name,
    BackendType.VLLM: BackendType.VLLM.app_name,
}

_APP_NAME_TO_BACKEND = {v: k for k, v in _LAUNCHPAD_APPS.items()}


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
        backend = _APP_NAME_TO_BACKEND.get(name)
        rows.append(EndpointInfo(name=name, app_id=app_id, state=state, backend=backend))
    return rows
