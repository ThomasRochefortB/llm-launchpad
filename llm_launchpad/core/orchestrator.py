"""High-level workflow orchestrator that yields protocol events.

All long-running operations (deploy, warmup, logs, etc.) are implemented
as generators that yield protocol events. The TUI workers and headless
CLI both consume these generators.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import queue
import re
import subprocess
import threading
import time
from typing import Any, Generator, List, Optional, Union

from ..protocol.enums import BackendType, DeploymentState, OperationType
from ..protocol.events import (
    BaseEvent,
    ErrorEvent,
    LogEvent,
    OperationCompleteEvent,
    StateChangeEvent,
)
from ..protocol.models import DeploymentConfig, EndpointInfo, LaunchpadSettings
from ..protocol.models import StoredModelInfo, StorageSnapshot

from .backend import ModalBackend
from .config import ConfigStore
from .naming import legacy_app_name
from .naming import random_function_slug
from .paths import MODAL_LLAMACPP_SCRIPT, MODAL_VLLM_SCRIPT

# Type alias for event generators
EventStream = Generator[BaseEvent, None, None]

_MODAL_GPU_WAIT_RE = re.compile(
    r"waiting to be scheduled on a (?P<worker>[A-Za-z0-9_:\-]+) worker",
    flags=re.IGNORECASE,
)
_MODAL_RELAX_RE = re.compile(r"Relaxing requirements \((?P<requirements>[^)]+)\)", flags=re.IGNORECASE)
_GGUF_QUANT_RE = re.compile(r"(Q\d(?:_[A-Z0-9]+)+|IQ\d+_[A-Z0-9_]+)", flags=re.IGNORECASE)
_SIZE_TOKEN_RE = re.compile(r"^\s*(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z]+)?\s*$")
_LEGACY_SHARD_RE = re.compile(r"[-_](?:\d{2,6})[-_]of[-_](?:\d{2,6})$", flags=re.IGNORECASE)


def _modal_gpu_scheduling_hint(response_text: str) -> str | None:
    """Extract a concise queueing status from Modal's scheduling message."""
    text = (response_text or "").strip()
    if not text:
        return None
    if "waiting to be scheduled" not in text.lower():
        return None

    worker = ""
    requirements = ""
    worker_match = _MODAL_GPU_WAIT_RE.search(text)
    if worker_match:
        worker = worker_match.group("worker").strip()
    requirements_match = _MODAL_RELAX_RE.search(text)
    if requirements_match:
        requirements = requirements_match.group("requirements").strip()

    parts: list[str] = []
    if worker:
        parts.append(worker)
    if requirements:
        parts.append(requirements)

    if not parts:
        return "Waiting for GPU scheduling"
    return f"Waiting for GPU scheduling ({', '.join(parts)})"


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
        if config.do_deploy and config.backend != BackendType.VLLM and not config.function_slug:
            config.function_slug = random_function_slug()
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

        # Start log tailing in background.
        # We use a dedicated reader thread + queue.Queue so that every
        # line the subprocess emits is captured immediately, regardless
        # of Python's internal TextIOWrapper read-ahead buffering.
        # (select() only checks the kernel pipe buffer, but readline()
        # can consume up to 2 KB into an internal buffer in a single
        # call — a mismatch that causes lines to be "stuck" until the
        # next kernel-buffer read.)
        target_app_name = app_name or legacy_app_name(backend)
        logs_proc: Optional[subprocess.Popen[str]] = None
        log_queue: Optional[queue.Queue[Optional[str]]] = None
        logs_retry_at = 0.0

        def _start_logs_tail() -> subprocess.Popen[str]:
            nonlocal log_queue
            follow = ModalBackend.logs_follow_args()
            # PYTHONUNBUFFERED forces the modal CLI (a Python program) to
            # flush each write immediately when stdout is a pipe.
            unbuf_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            proc = subprocess.Popen(
                ["modal", "app", "logs", *follow, target_app_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=unbuf_env,
            )
            q: queue.Queue[Optional[str]] = queue.Queue()
            log_queue = q

            def _reader() -> None:
                try:
                    assert proc.stdout is not None
                    for raw_line in proc.stdout:
                        q.put(raw_line.rstrip("\n"))
                except Exception:
                    pass
                finally:
                    q.put(None)  # sentinel: reader finished

            threading.Thread(target=_reader, daemon=True).start()
            return proc

        if tail_logs:
            try:
                logs_proc = _start_logs_tail()
            except Exception as exc:
                yield LogEvent(line=f"Warning: failed to start log tailing: {exc}")
                logs_retry_at = time.time() + 5.0

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
        last_scheduling_hint: Optional[str] = None
        # Track displayed log lines so the historical-fetch fallback can
        # avoid duplicating output already shown by the live stream.
        seen_log_lines: set[str] = set()

        def _drain_queue() -> Generator[LogEvent, None, None]:
            """Yield log events from the reader-thread queue (non-blocking)."""
            nonlocal logs_proc, log_queue, logs_retry_at
            if log_queue is None:
                return
            while True:
                try:
                    line = log_queue.get_nowait()
                except queue.Empty:
                    break
                if line is None:
                    # Reader thread finished — subprocess exited.
                    logs_proc = None
                    log_queue = None
                    logs_retry_at = time.time() + 5.0
                    break
                seen_log_lines.add(line)
                yield LogEvent(line=line, operation=OperationType.WARMUP)

        while True:
            # Exit promptly if the app is shutting down.
            if ModalBackend.is_shutting_down():
                if logs_proc:
                    try:
                        logs_proc.terminate()
                    except Exception:
                        pass
                return

            if tail_logs and logs_proc is None and time.time() >= logs_retry_at:
                # ── Historical-fetch fallback ──
                # The live stream (`modal app logs`) may not deliver the
                # final output of a crashing container before the stream
                # closes.  The Modal dashboard shows those logs because it
                # reads *persisted* logs.  Re-running `modal app logs`
                # after a short delay retrieves the same persisted data,
                # letting us display crash tracebacks the live stream
                # missed.
                yield from self._fetch_historical_logs(
                    target_app_name, seen_log_lines
                )
                # Re-attach live stream for further output.
                try:
                    logs_proc = _start_logs_tail()
                except Exception as exc:
                    yield LogEvent(line=f"Warning: failed to start log tailing: {exc}")
                    logs_retry_at = time.time() + 5.0

            # Drain log lines delivered by the reader thread.
            yield from _drain_queue()

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
                body = resp.text or ""
                last_err = f"HTTP {resp.status_code}: {body[:200]}"
                scheduling_hint = _modal_gpu_scheduling_hint(body)
                if scheduling_hint:
                    if scheduling_hint != last_scheduling_hint:
                        yield StateChangeEvent(
                            current=DeploymentState.QUEUED,
                            operation=OperationType.WARMUP,
                            detail=scheduling_hint,
                        )
                        yield LogEvent(line=scheduling_hint, operation=OperationType.WARMUP)
                        last_scheduling_hint = scheduling_hint
                elif last_scheduling_hint is not None:
                    yield StateChangeEvent(
                        current=DeploymentState.WARMING_UP,
                        operation=OperationType.WARMUP,
                        detail="GPU allocated; continuing readiness probe",
                    )
                    yield LogEvent(
                        line="GPU allocated; continuing readiness probe.",
                        operation=OperationType.WARMUP,
                    )
                    last_scheduling_hint = None
            except Exception as exc:
                last_err = str(exc)

            # Sleep in small increments, draining the log queue between
            # each chunk so that lines appear in the TUI promptly.
            sleep_end = time.time() + backoff
            while time.time() < sleep_end:
                if ModalBackend.is_shutting_down():
                    break
                chunk = min(0.5, sleep_end - time.time())
                if chunk > 0:
                    ModalBackend._shutdown_event.wait(timeout=chunk)
                yield from _drain_queue()
            backoff = min(max_backoff, backoff * 1.5)

    # ------------------------------------------------------------------
    # Historical log fetch (fallback for crash output)
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_historical_logs(
        app_name: str,
        seen: set[str],
    ) -> EventStream:
        """Re-run ``modal app logs`` to capture persisted crash output.

        The live log stream may not deliver the tail end of a crashing
        container's output.  Modal persists those logs server-side and
        the dashboard reads them.  By re-running the CLI command after a
        short delay we retrieve the same data, then yield only the lines
        the live stream missed (tracked via *seen*).
        """
        try:
            unbuf_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            hist = subprocess.run(
                ["modal", "app", "logs", app_name],
                capture_output=True,
                text=True,
                timeout=15,
                env=unbuf_env,
            )
            hist_output = hist.stdout or ""
        except subprocess.TimeoutExpired as te:
            # Use whatever output was captured before the timeout.
            hist_output = te.stdout if isinstance(te.stdout, str) else ""
        except Exception:
            hist_output = ""

        if not hist_output:
            return

        new_lines: list[str] = []
        for raw_line in hist_output.splitlines():
            line = raw_line.rstrip()
            if line and line not in seen:
                new_lines.append(line)
                seen.add(line)

        if new_lines:
            yield LogEvent(
                line="── Fetched historical container logs ──",
                operation=OperationType.WARMUP,
            )
            for line in new_lines:
                yield LogEvent(line=line, operation=OperationType.WARMUP)

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
        last_scheduling_hint: Optional[str] = None

        while True:
            if ModalBackend.is_shutting_down():
                return

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
                body = resp.text or ""
                last_err = f"HTTP {resp.status_code}: {body[:200]}"
                scheduling_hint = _modal_gpu_scheduling_hint(body)
                if scheduling_hint:
                    if scheduling_hint != last_scheduling_hint:
                        yield StateChangeEvent(
                            current=DeploymentState.QUEUED,
                            operation=OperationType.STATUS,
                            detail=scheduling_hint,
                        )
                        yield LogEvent(line=scheduling_hint, operation=OperationType.STATUS)
                        last_scheduling_hint = scheduling_hint
                elif last_scheduling_hint is not None:
                    yield StateChangeEvent(
                        current=DeploymentState.RUNNING,
                        operation=OperationType.STATUS,
                        detail="GPU allocated; continuing health check",
                    )
                    yield LogEvent(
                        line="GPU allocated; continuing health check.",
                        operation=OperationType.STATUS,
                    )
                    last_scheduling_hint = None
            except Exception as exc:
                last_err = str(exc)

            # Interruptible sleep: wakes immediately on shutdown.
            if ModalBackend._shutdown_event.wait(timeout=backoff):
                return
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
        if apps is not None:
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
    # Storage
    # ------------------------------------------------------------------

    def list_storage(self) -> EventStream:
        """List cached model artifacts for both backends."""
        yield StateChangeEvent(
            current=DeploymentState.RUNNING,
            operation=OperationType.STORAGE_LIST,
            detail="Scanning Modal volumes for cached models",
        )
        scan_started = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                llamacpp_future = executor.submit(self._list_llamacpp_models)
                vllm_future = executor.submit(self._list_vllm_models)
                snapshot = StorageSnapshot(
                    llamacpp_models=llamacpp_future.result(),
                    vllm_models=vllm_future.result(),
                )
        except Exception as exc:
            yield ErrorEvent(
                message=f"Failed to list storage: {exc}",
                operation=OperationType.STORAGE_LIST,
                recoverable=True,
            )
            yield OperationCompleteEvent(
                operation=OperationType.STORAGE_LIST,
                success=False,
                exit_code=1,
                detail=str(exc),
            )
            return

        scan_elapsed_ms = int((time.perf_counter() - scan_started) * 1000)
        yield LogEvent(
            line=(
                f"Storage snapshot: llama.cpp={len(snapshot.llamacpp_models)} models, "
                f"vLLM={len(snapshot.vllm_models)} models, "
                f"total={snapshot.total_models}, bytes={snapshot.total_size_bytes}, "
                f"scan_ms={scan_elapsed_ms}"
            ),
            operation=OperationType.STORAGE_LIST,
        )
        yield OperationCompleteEvent(
            operation=OperationType.STORAGE_LIST,
            success=True,
            data=snapshot,
        )

    def predownload_model(
        self,
        backend: BackendType,
        model_id: str,
        quant: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> EventStream:
        """Pre-download a model for a backend without deploying."""
        if backend == BackendType.LLAMACPP:
            script = MODAL_LLAMACPP_SCRIPT
            entrypoint = "predownload_model"
            args = ["--repo-id", model_id]
            if quant:
                args.extend(["--quant", quant])
            if revision:
                args.extend(["--revision", revision])
        else:
            script = MODAL_VLLM_SCRIPT
            entrypoint = "predownload_model"
            args = ["--repo-id", model_id]
            if revision:
                args.extend(["--revision", revision])

        cmd = ModalBackend.build_modal_entrypoint_command(script, entrypoint, args)
        yield StateChangeEvent(
            current=DeploymentState.RUNNING,
            operation=OperationType.STORAGE_PREDOWNLOAD,
            detail=" ".join(cmd),
        )
        yield LogEvent(
            line=f"Running: {' '.join(cmd)}",
            operation=OperationType.STORAGE_PREDOWNLOAD,
        )

        for event in ModalBackend.run_modal_script_entrypoint(script, entrypoint, args=args):
            if isinstance(event, OperationCompleteEvent):
                yield OperationCompleteEvent(
                    operation=OperationType.STORAGE_PREDOWNLOAD,
                    success=event.success,
                    exit_code=event.exit_code,
                    detail=event.detail,
                    data=event.data,
                )
            elif isinstance(event, ErrorEvent):
                yield ErrorEvent(
                    message=event.message,
                    operation=OperationType.STORAGE_PREDOWNLOAD,
                    exit_code=event.exit_code,
                    recoverable=event.recoverable,
                )
            elif isinstance(event, LogEvent):
                yield LogEvent(
                    line=event.line,
                    stream=event.stream,
                    operation=OperationType.STORAGE_PREDOWNLOAD,
                )
            else:
                yield event

    def delete_stored_model(self, model: StoredModelInfo) -> EventStream:
        """Delete cached storage artifacts for a selected model."""
        targets = self._storage_delete_targets(model)
        if not targets:
            yield ErrorEvent(
                message=f"No removable paths found for {model.model_id}",
                operation=OperationType.STORAGE_DELETE,
                recoverable=True,
            )
            yield OperationCompleteEvent(
                operation=OperationType.STORAGE_DELETE,
                success=False,
                exit_code=1,
                detail="No paths resolved for deletion.",
            )
            return

        yield StateChangeEvent(
            current=DeploymentState.RUNNING,
            operation=OperationType.STORAGE_DELETE,
            detail=f"Deleting {model.model_id}",
        )
        for volume_name, remote_path, recursive in targets:
            yield LogEvent(
                line=f"Deleting {remote_path} from {volume_name}",
                operation=OperationType.STORAGE_DELETE,
            )
            for event in ModalBackend.run_volume_remove(volume_name, remote_path, recursive=recursive):
                if isinstance(event, OperationCompleteEvent):
                    if not event.success:
                        yield OperationCompleteEvent(
                            operation=OperationType.STORAGE_DELETE,
                            success=False,
                            exit_code=event.exit_code,
                            detail=event.detail or f"Failed removing {remote_path}",
                        )
                        return
                    continue
                if isinstance(event, ErrorEvent):
                    yield ErrorEvent(
                        message=event.message,
                        operation=OperationType.STORAGE_DELETE,
                        exit_code=event.exit_code,
                        recoverable=event.recoverable,
                    )
                    yield OperationCompleteEvent(
                        operation=OperationType.STORAGE_DELETE,
                        success=False,
                        exit_code=event.exit_code or 1,
                        detail=event.message,
                    )
                    return
                if isinstance(event, LogEvent):
                    yield LogEvent(
                        line=event.line,
                        stream=event.stream,
                        operation=OperationType.STORAGE_DELETE,
                    )
                else:
                    yield event

        yield OperationCompleteEvent(
            operation=OperationType.STORAGE_DELETE,
            success=True,
            detail=f"Deleted storage for {model.model_id}",
        )

    def _list_llamacpp_models(self) -> list[StoredModelInfo]:
        files = self._walk_volume_files(self._LLAMACPP_STORAGE_VOLUME, "/models")
        grouped: dict[tuple[str, Optional[str]], StoredModelInfo] = {}
        incomplete_keys: set[tuple[str, Optional[str]]] = set()
        keys_with_any_files: set[tuple[str, Optional[str]]] = set()
        keys_with_gguf_files: set[tuple[str, Optional[str]]] = set()
        for row in files:
            path = str(row.get("path", "")).strip()
            size = int(row.get("size_bytes", 0) or 0)
            rel = path.removeprefix("/models/").strip("/")
            parts = rel.split("/")
            if len(parts) < 2:
                continue
            model_id = parts[0].replace("__", "/")
            revision = parts[1]
            key = (model_id, None if revision == "main" else revision)
            keys_with_any_files.add(key)
            is_incomplete = path.lower().endswith(".incomplete")
            if is_incomplete:
                incomplete_keys.add(key)
                entry = grouped.get(key)
                if entry is None:
                    entry = StoredModelInfo(
                        backend=BackendType.LLAMACPP,
                        model_id=model_id,
                        revision=key[1],
                        quant=None,
                        size_bytes=0,
                        file_count=0,
                        source_volume=self._LLAMACPP_STORAGE_VOLUME,
                        paths=[],
                        incomplete=True,
                    )
                    grouped[key] = entry
                entry.size_bytes += size
                entry.file_count += 1
                if entry.paths is not None and path not in entry.paths:
                    entry.paths.append(path)
                continue
            if not path.lower().endswith(".gguf"):
                continue
            if len(parts) < 3:
                continue
            keys_with_gguf_files.add(key)
            entry = grouped.get(key)
            if entry is None:
                entry = StoredModelInfo(
                    backend=BackendType.LLAMACPP,
                    model_id=model_id,
                    revision=key[1],
                    quant=None,
                    size_bytes=0,
                    file_count=0,
                    source_volume=self._LLAMACPP_STORAGE_VOLUME,
                    paths=[],
                )
                grouped[key] = entry
            entry.size_bytes += size
            entry.file_count += 1
            if entry.paths is not None:
                entry.paths.append(path)
            if entry.quant is None:
                quant_match = _GGUF_QUANT_RE.search(path.upper())
                if quant_match:
                    entry.quant = quant_match.group(1).upper()

        # Compatibility: legacy flat cache files.
        legacy_files = self._walk_volume_files(self._LLAMACPP_STORAGE_VOLUME, "/", recursive=False)
        for row in legacy_files:
            path = str(row.get("path", "")).strip()
            if not path.lower().endswith(".gguf"):
                continue
            if path.startswith("/models/"):
                continue
            size = int(row.get("size_bytes", 0) or 0)
            stem = path.split("/")[-1].rsplit(".", 1)[0]
            stem = _LEGACY_SHARD_RE.sub("", stem)
            model_id = f"legacy:{stem}"
            key = (model_id, None)
            entry = grouped.get(key)
            if entry is None:
                entry = StoredModelInfo(
                    backend=BackendType.LLAMACPP,
                    model_id=model_id,
                    revision=None,
                    quant=None,
                    size_bytes=0,
                    file_count=0,
                    source_volume=self._LLAMACPP_STORAGE_VOLUME,
                    paths=[],
                )
                grouped[key] = entry
            entry.size_bytes += size
            entry.file_count += 1
            if entry.paths is not None:
                entry.paths.append(path)
            if entry.quant is None:
                quant_match = _GGUF_QUANT_RE.search(path.upper())
                if quant_match:
                    entry.quant = quant_match.group(1).upper()

        # If a model revision has cache metadata/files but no GGUF payload yet,
        # treat it as incomplete so it is visible and recoverable in the UI.
        for key in sorted(keys_with_any_files - keys_with_gguf_files):
            entry = grouped.get(key)
            if entry is None:
                entry = StoredModelInfo(
                    backend=BackendType.LLAMACPP,
                    model_id=key[0],
                    revision=key[1],
                    quant=None,
                    size_bytes=0,
                    file_count=0,
                    source_volume=self._LLAMACPP_STORAGE_VOLUME,
                    paths=[],
                    incomplete=True,
                )
                grouped[key] = entry
            entry.incomplete = True

        for key in incomplete_keys:
            entry = grouped.get(key)
            if entry is not None:
                entry.incomplete = True

        return sorted(grouped.values(), key=lambda row: (row.model_id, row.revision or ""))

    def _list_vllm_models(self) -> list[StoredModelInfo]:
        files = self._walk_volume_files(self._VLLM_STORAGE_VOLUME, "/hub")
        grouped: dict[str, StoredModelInfo] = {}
        blob_files: dict[str, set[str]] = {}
        incomplete_blob_files: dict[str, set[str]] = {}
        snapshot_file_counts: dict[str, int] = {}
        ref_file_counts: dict[str, int] = {}
        for row in files:
            path = str(row.get("path", "")).strip()
            size = int(row.get("size_bytes", 0) or 0)
            rel = path.removeprefix("/hub/").strip("/")
            if not rel.startswith("models--"):
                continue
            parts = rel.split("/")
            if not parts:
                continue
            model_dir = parts[0]
            encoded = model_dir[len("models--") :]
            model_id = encoded.replace("--", "/")
            entry = grouped.get(model_id)
            if entry is None:
                entry = StoredModelInfo(
                    backend=BackendType.VLLM,
                    model_id=model_id,
                    revision=None,
                    quant=None,
                    size_bytes=0,
                    file_count=0,
                    source_volume=self._VLLM_STORAGE_VOLUME,
                    paths=[],
                )
                grouped[model_id] = entry
            if row.get("is_file", False):
                entry.size_bytes += size
                entry.file_count += 1
                lower_path = path.lower()
                if "/snapshots/" in lower_path:
                    snapshot_file_counts[model_id] = snapshot_file_counts.get(model_id, 0) + 1
                if "/refs/" in lower_path:
                    ref_file_counts[model_id] = ref_file_counts.get(model_id, 0) + 1
                if "/blobs/" in lower_path:
                    blob_name = path.rsplit("/", 1)[-1].strip()
                    if blob_name.endswith(".incomplete"):
                        final_blob_name = blob_name[: -len(".incomplete")]
                        if final_blob_name:
                            incomplete_blob_files.setdefault(model_id, set()).add(final_blob_name)
                    elif blob_name:
                        blob_files.setdefault(model_id, set()).add(blob_name)
            if entry.paths is not None and path not in entry.paths:
                entry.paths.append(path)

        for model_id, entry in grouped.items():
            known_blobs = blob_files.get(model_id, set())
            pending_blobs = incomplete_blob_files.get(model_id, set())
            snapshot_count = snapshot_file_counts.get(model_id, 0)
            ref_count = ref_file_counts.get(model_id, 0)
            # Align with huggingface_hub cache semantics:
            # incomplete blobs are resumable partial downloads, but once the
            # final blob exists, download logic treats that file as complete.
            # Keep stale orphan .incomplete blobs from old attempts from marking
            # a model incomplete when a usable snapshot already exists.
            unresolved_incomplete = any(blob_name not in known_blobs for blob_name in pending_blobs)
            if unresolved_incomplete and snapshot_count == 0:
                entry.incomplete = True
            # Refs without snapshots indicate metadata without an accessible
            # snapshot payload for this repo.
            if ref_count > 0 and snapshot_count == 0:
                entry.incomplete = True
        return sorted(grouped.values(), key=lambda row: row.model_id)

    def _walk_volume_files(
        self,
        volume_name: str,
        root: str,
        recursive: bool = True,
    ) -> list[dict[str, Any]]:
        pending = [root]
        files: list[dict[str, Any]] = []
        visited: set[str] = set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            rows = ModalBackend.list_volume(volume_name, current) or []
            for row in rows:
                path = self._volume_entry_path(current, row)
                if not path:
                    continue
                is_dir = self._volume_entry_is_dir(row, path)
                if is_dir:
                    if path == current or not recursive:
                        continue
                    pending.append(path)
                else:
                    files.append(
                        {
                            "path": path,
                            "size_bytes": Orchestrator._parse_size_bytes(
                                Orchestrator._entry_value(
                                    row, "size", "size_bytes", "bytes", "filesize"
                                )
                            ),
                            "is_file": True,
                        }
                    )
        return files

    def _storage_delete_targets(
        self, model: StoredModelInfo
    ) -> list[tuple[str, str, bool]]:
        targets: list[tuple[str, str, bool]] = []
        if model.backend == BackendType.LLAMACPP:
            if model.model_id.startswith("legacy:"):
                for path in sorted(set(model.paths or [])):
                    targets.append((model.source_volume or self._LLAMACPP_STORAGE_VOLUME, path, False))
            else:
                revision = model.revision or "main"
                repo_slug = model.model_id.replace("/", "__")
                targets.append(
                    (
                        model.source_volume or self._LLAMACPP_STORAGE_VOLUME,
                        f"/models/{repo_slug}/{revision}",
                        True,
                    )
                )
            return targets

        if model.backend == BackendType.VLLM:
            encoded = model.model_id.replace("/", "--")
            targets.append(
                (
                    model.source_volume or self._VLLM_STORAGE_VOLUME,
                    f"/hub/models--{encoded}",
                    True,
                )
            )
        return targets

    @staticmethod
    def _entry_value(row: dict[str, Any], *keys: str) -> Any:
        """Case-insensitive lookup for JSON fields from Modal CLI."""
        lowered = {str(key).strip().lower(): value for key, value in row.items()}
        for key in keys:
            needle = key.strip().lower()
            if needle in lowered:
                return lowered[needle]
        return None

    @staticmethod
    def _volume_entry_path(parent: str, row: dict[str, Any]) -> str:
        for key in ("path", "name", "file", "filename", "entry"):
            value = Orchestrator._entry_value(row, key)
            if isinstance(value, str) and value.strip():
                candidate = value.strip()
                if candidate.startswith("/"):
                    return candidate
                parent_clean = parent.strip("/")
                if parent_clean and candidate.startswith(f"{parent_clean}/"):
                    return f"/{candidate.lstrip('/')}"
                if parent == "/" and candidate:
                    return f"/{candidate.lstrip('/')}"
                if parent.endswith("/"):
                    return f"{parent}{candidate}"
                return f"{parent}/{candidate}"
        return ""

    @staticmethod
    def _volume_entry_is_dir(row: dict[str, Any], path: str) -> bool:
        is_dir = Orchestrator._entry_value(row, "is_dir", "isdir", "directory")
        if isinstance(is_dir, bool):
            return is_dir
        entry_type = str(Orchestrator._entry_value(row, "type", "kind") or "").strip().lower()
        if entry_type in {"dir", "directory", "folder"}:
            return True
        size_hint = Orchestrator._entry_value(row, "size", "size_bytes", "bytes")
        if size_hint in (None, "", 0, "0"):
            # If no file-size signal is present, treat slash-terminated entries as directories.
            if path.endswith("/"):
                return True
        return path.endswith("/")

    @staticmethod
    def _parse_size_bytes(value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value).strip()
        if not text:
            return 0
        clean = text.replace(",", "")
        if clean.isdigit():
            return int(clean)
        match = _SIZE_TOKEN_RE.match(clean)
        if not match:
            return 0
        number = float(match.group("num"))
        unit = (match.group("unit") or "B").upper()
        factors = {
            "B": 1,
            "KB": 1024,
            "MB": 1024**2,
            "GB": 1024**3,
            "TB": 1024**4,
            "KIB": 1024,
            "MIB": 1024**2,
            "GIB": 1024**3,
            "TIB": 1024**4,
        }
        factor = factors.get(unit, 1)
        return int(number * factor)

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
    _LLAMACPP_STORAGE_VOLUME = "huggingface-cache"
    _VLLM_STORAGE_VOLUME = "huggingface-cache"
