"""High-level workflow orchestrator that yields protocol events.

All long-running operations (deploy, warmup, logs, etc.) are implemented
as generators that yield protocol events. The TUI workers and headless
CLI both consume these generators.
"""

from __future__ import annotations

import subprocess
import time
from typing import Generator, List, Optional, Union

from ..protocol.enums import BackendType, DeploymentState, OperationType
from ..protocol.events import (
    BaseEvent,
    ErrorEvent,
    LogEvent,
    OperationCompleteEvent,
    StateChangeEvent,
)
from ..protocol.models import DeploymentConfig, EndpointInfo, LaunchpadSettings

from .backend import ModalBackend
from .config import ConfigStore
from .naming import legacy_app_name

# Type alias for event generators
EventStream = Generator[BaseEvent, None, None]


class Orchestrator:
    """Coordinate high-level deployment flows.

    Every public method is a generator yielding protocol events so callers
    can stream output in real-time (TUI workers, headless printers, etc.).
    """

    def __init__(
        self,
        config_store: ConfigStore | None = None,
        backend: ModalBackend | None = None,
    ) -> None:
        self.config_store = config_store or ConfigStore()
        self.backend = backend or ModalBackend()

    # ------------------------------------------------------------------
    # Pre-flight
    # ------------------------------------------------------------------

    def preflight(self) -> tuple[bool, str, str]:
        """Check Modal CLI availability and auth.

        Returns ``(ok, username, error_message)``.
        """
        if not ModalBackend.is_cli_available():
            return False, "", "Modal CLI not found. Install with: pip install modal && modal setup"
        username = ModalBackend.get_username()
        if not username:
            return False, "", "Modal authentication missing. Run: modal setup"
        return True, username, ""

    # ------------------------------------------------------------------
    # Deploy
    # ------------------------------------------------------------------

    def deploy(self, config: DeploymentConfig) -> EventStream:
        """Run a full deploy workflow (optional preload + deploy + warmup)."""
        settings = self.config_store.load()
        env = ModalBackend.build_full_env(settings, config)

        if config.backend == BackendType.VLLM:
            yield from self._deploy_vllm(config, env)
        else:
            yield from self._deploy_llamacpp(config, env)

    def _deploy_vllm(
        self, config: DeploymentConfig, env: dict[str, str]
    ) -> EventStream:
        if config.do_deploy:
            cmd = ModalBackend.build_deploy_command(config.backend, app_name=config.app_name)
            yield StateChangeEvent(
                current=DeploymentState.DEPLOYING,
                operation=OperationType.DEPLOY,
                detail=" ".join(cmd),
            )
            yield LogEvent(line=f"Running: {' '.join(cmd)}")
            if env:
                yield LogEvent(line=f"  env: {', '.join(f'{k}={v}' for k, v in env.items())}")
            yield from ModalBackend.run_streaming(cmd, env=env)
        elif config.run_smoke:
            cmd = ["modal", "run", BackendType.VLLM.script]
            yield StateChangeEvent(
                current=DeploymentState.RUNNING,
                operation=OperationType.SMOKE_TEST,
                detail=" ".join(cmd),
            )
            yield LogEvent(line=f"Running: {' '.join(cmd)}")
            yield from ModalBackend.run_streaming(cmd, env=env)

    def _deploy_llamacpp(
        self, config: DeploymentConfig, env: dict[str, str]
    ) -> EventStream:
        cmd = ModalBackend.build_run_command(config)
        yield StateChangeEvent(
            current=DeploymentState.DEPLOYING if config.do_deploy else DeploymentState.RUNNING,
            operation=OperationType.DEPLOY,
            detail=" ".join(cmd),
        )
        yield LogEvent(line=f"Running: {' '.join(cmd)}")
        if env:
            yield LogEvent(line=f"  env: {', '.join(f'{k}={v}' for k, v in env.items())}")
        yield from ModalBackend.run_streaming(cmd, env=env)

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def warmup(
        self,
        backend: BackendType,
        server_url: str,
        timeout: int = 1800,
        tail_logs: bool = True,
        app_name: Optional[str] = None,
        served_model_name: Optional[str] = None,
    ) -> EventStream:
        """Probe endpoint readiness and optionally tail logs."""
        yield StateChangeEvent(
            current=DeploymentState.WARMING_UP,
            operation=OperationType.WARMUP,
            detail=f"Probing {server_url}",
        )

        is_vllm = backend == BackendType.VLLM
        probe_url = server_url.rstrip("/") + ("/health" if is_vllm else "/v1/completions")
        yield LogEvent(line=f"Probing readiness at: {probe_url}")

        # Start log tailing in background
        logs_proc: Optional[subprocess.Popen[str]] = None
        if tail_logs:
            try:
                follow = ModalBackend.logs_follow_args()
                target_app_name = app_name or legacy_app_name(backend)
                logs_proc = subprocess.Popen(
                    ["modal", "app", "logs", *follow, target_app_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except Exception as exc:
                yield LogEvent(line=f"Warning: failed to start log tailing: {exc}")

        try:
            import requests  # type: ignore
        except ImportError:
            yield ErrorEvent(
                message="'requests' is required. Install with: pip install requests",
                operation=OperationType.WARMUP,
                recoverable=False,
            )
            return

        payload = {"model": "default", "prompt": "ping", "max_tokens": 1, "temperature": 0}
        headers = {"Content-Type": "application/json"}
        start = time.time()
        backoff = 2.0
        max_backoff = 30.0
        last_err: Optional[str] = None

        while True:
            # Drain any log lines from the background process
            if logs_proc and logs_proc.stdout:
                import select

                while True:
                    ready, _, _ = select.select([logs_proc.stdout], [], [], 0)
                    if not ready:
                        break
                    raw_line = logs_proc.stdout.readline()
                    if not raw_line:
                        break
                    yield LogEvent(line=raw_line.rstrip("\n"), operation=OperationType.WARMUP)

            elapsed = time.time() - start
            if elapsed > timeout:
                if logs_proc:
                    try:
                        logs_proc.terminate()
                    except Exception:
                        pass
                yield ErrorEvent(
                    message=f"Timed out after {timeout}s. Last error: {last_err}",
                    operation=OperationType.WARMUP,
                )
                yield OperationCompleteEvent(
                    operation=OperationType.WARMUP, success=False, exit_code=1
                )
                return

            try:
                import json as _json

                if is_vllm:
                    resp = requests.get(probe_url, timeout=10)
                else:
                    resp = requests.post(
                        probe_url, headers=headers, data=_json.dumps(payload), timeout=10
                    )
                if 200 <= resp.status_code < 300:
                    if logs_proc:
                        try:
                            logs_proc.terminate()
                        except Exception:
                            pass
                    yield LogEvent(line="Server is ready!")
                    curl_cmd = ModalBackend.test_curl_command(
                        backend,
                        server_url,
                        served_model_name=served_model_name,
                    )
                    yield LogEvent(line=f"Test command:\n{curl_cmd}")
                    yield StateChangeEvent(
                        current=DeploymentState.HEALTHY, operation=OperationType.WARMUP
                    )
                    yield OperationCompleteEvent(
                        operation=OperationType.WARMUP, success=True, data={"url": server_url}
                    )
                    return
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except Exception as exc:
                last_err = str(exc)

            time.sleep(backoff)
            backoff = min(max_backoff, backoff * 1.5)

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------

    def tail_logs(
        self,
        backend: BackendType,
        follow: bool = True,
        app_name: Optional[str] = None,
    ) -> EventStream:
        """Tail Modal logs for the given backend."""
        target_app_name = app_name or legacy_app_name(backend)
        cmd: List[str] = ["modal", "app", "logs"]
        if follow:
            cmd.extend(ModalBackend.logs_follow_args())
        cmd.append(target_app_name)

        yield StateChangeEvent(
            current=DeploymentState.RUNNING,
            operation=OperationType.LOGS,
            detail=" ".join(cmd),
        )
        yield from ModalBackend.run_streaming(cmd)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def check_status(
        self,
        backend: BackendType,
        server_url: str,
        timeout: int = 60,
    ) -> EventStream:
        """Probe endpoint health with backoff."""
        is_vllm = backend == BackendType.VLLM
        probe_url = server_url.rstrip("/") + ("/health" if is_vllm else "/v1/completions")

        yield StateChangeEvent(
            current=DeploymentState.RUNNING,
            operation=OperationType.STATUS,
            detail=f"Checking {probe_url}",
        )
        yield LogEvent(line=f"Checking endpoint status at: {probe_url}")

        try:
            import requests  # type: ignore
        except ImportError:
            yield ErrorEvent(
                message="'requests' required", operation=OperationType.STATUS, recoverable=False
            )
            return

        import json as _json

        payload = {"model": "default", "prompt": "ping", "max_tokens": 1, "temperature": 0}
        headers = {"Content-Type": "application/json"}
        start = time.time()
        backoff = 2.0
        max_backoff = 15.0
        last_err: Optional[str] = None

        while True:
            if time.time() - start > timeout:
                yield ErrorEvent(
                    message=f"Unhealthy (timed out). Last: {last_err}",
                    operation=OperationType.STATUS,
                )
                yield OperationCompleteEvent(
                    operation=OperationType.STATUS, success=False, exit_code=1
                )
                return

            try:
                if is_vllm:
                    resp = requests.get(probe_url, timeout=10)
                else:
                    resp = requests.post(
                        probe_url, headers=headers, data=_json.dumps(payload), timeout=10
                    )
                if 200 <= resp.status_code < 300:
                    curl_cmd = ModalBackend.test_curl_command(backend, server_url)
                    yield LogEvent(
                        line=f"Status: healthy (backend={backend.value}, url={server_url})"
                    )
                    yield LogEvent(line=f"Test command:\n{curl_cmd}")
                    yield StateChangeEvent(
                        current=DeploymentState.HEALTHY, operation=OperationType.STATUS
                    )
                    yield OperationCompleteEvent(
                        operation=OperationType.STATUS, success=True
                    )
                    return
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except Exception as exc:
                last_err = str(exc)

            time.sleep(backoff)
            backoff = min(max_backoff, backoff * 1.5)

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_deployments(self) -> EventStream:
        """List launchpad-related Modal apps."""
        yield StateChangeEvent(
            current=DeploymentState.RUNNING, operation=OperationType.LIST
        )

        apps = ModalBackend.list_apps()
        if apps:
            launchpad = [a for a in apps if a.backend is not None]
            if not launchpad:
                yield LogEvent(line="No launchpad deployments found.")
            else:
                yield LogEvent(line="Launchpad deployments:")
                for info in launchpad:
                    suffix = f" ({info.app_id})" if info.app_id else ""
                    bk = info.backend.value if info.backend else "unknown"
                    inst = info.instance_name or "-"
                    yield LogEvent(
                        line=f"  backend={bk}  instance={inst}  app={info.name}  state={info.state}{suffix}"
                    )
            yield OperationCompleteEvent(
                operation=OperationType.LIST, success=True, data=launchpad
            )
            return

        # Fallback to raw text
        raw = ModalBackend.list_apps_raw()
        if raw:
            names = {legacy_app_name(bt) for bt in BackendType}
            lines = raw.splitlines()
            matches = [l for l in lines if any(n in l for n in names) or "vllm-" in l or "llamacpp-" in l]
            if matches:
                yield LogEvent(line="Launchpad deployments:")
                for line in matches:
                    yield LogEvent(line=f"  {line.strip()}")
            else:
                yield LogEvent(line="No launchpad deployments found.")
            yield OperationCompleteEvent(operation=OperationType.LIST, success=True)
        else:
            yield ErrorEvent(
                message="Failed to query Modal app list.",
                operation=OperationType.LIST,
            )
            yield OperationCompleteEvent(
                operation=OperationType.LIST, success=False, exit_code=1
            )

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    def stop_app(self, backend: BackendType, app_name: Optional[str] = None) -> EventStream:
        """Stop a deployed app."""
        target_app_name = app_name or legacy_app_name(backend)
        cmd = ["modal", "app", "stop", target_app_name]
        yield StateChangeEvent(
            current=DeploymentState.RUNNING,
            operation=OperationType.STOP,
            detail=f"Stopping {target_app_name}",
        )
        yield LogEvent(line=f"Stopping app: {target_app_name}")
        yield from ModalBackend.run_streaming(cmd)
