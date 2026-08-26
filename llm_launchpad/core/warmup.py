"""Endpoint warmup and Modal log tailing helpers."""

from __future__ import annotations

from .shutdown import is_shutting_down, shutdown_event

import json
import os
import queue
import re
import subprocess
import threading
import time
from typing import Generator, Optional

from ..protocol.enums import BackendType, ComputeProvider, DeploymentState, OperationType
from ..protocol.events import (
    BaseEvent,
    ErrorEvent,
    LogEvent,
    OperationCompleteEvent,
    StateChangeEvent,
)
from .backend import ModalBackend
from .naming import legacy_app_name
from .prime_backend import PrimeBackend

EventStream = Generator[BaseEvent, None, None]

_MODAL_GPU_WAIT_RE = re.compile(
    r"waiting to be scheduled on a (?P<worker>[A-Za-z0-9_:\-]+) worker",
    flags=re.IGNORECASE,
)
_MODAL_RELAX_RE = re.compile(r"Relaxing requirements \((?P<requirements>[^)]+)\)", flags=re.IGNORECASE)


def modal_gpu_scheduling_hint(response_text: str) -> str | None:
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


def probe_response_is_ready(backend: BackendType, status_code: int, body: str) -> tuple[bool, str | None]:
    """Validate readiness probe success beyond HTTP status for llama.cpp."""
    if not (200 <= status_code < 300):
        return False, None
    if backend == BackendType.VLLM:
        return True, None

    text = (body or "").strip()
    if not text:
        return False, "empty response body"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False, "non-JSON response body"
    if not isinstance(payload, dict):
        return False, "unexpected JSON payload type"
    choices = payload.get("choices")
    if isinstance(choices, list):
        return True, None
    if isinstance(payload.get("error"), dict):
        return False, "error response"
    return False, "missing completions choices"


def endpoint_root_url(server_url: str) -> str:
    """Normalize an OpenAI-compatible URL to the host/function root."""
    root = (server_url or "").strip().rstrip("/")
    if root.endswith("/v1"):
        return root[: -len("/v1")].rstrip("/")
    return root


def status_probe_url(backend: BackendType, server_url: str) -> str:
    """Build the readiness probe URL without duplicating API path segments."""
    root = endpoint_root_url(server_url)
    return root + ("/health" if backend == BackendType.VLLM else "/v1/completions")


class WarmupRunner:
    """Probe endpoint readiness while optionally tailing Modal logs."""

    def run(
        self,
        backend: BackendType,
        server_url: str,
        timeout: int = 1800,
        tail_logs: bool = True,
        app_name: Optional[str] = None,
        served_model_name: Optional[str] = None,
        provider: ComputeProvider = ComputeProvider.MODAL,
        api_key: Optional[str] = None,
        pod_id: Optional[str] = None,
        prime_backend: PrimeBackend | None = None,
    ) -> EventStream:
        """Probe endpoint readiness and optionally tail logs."""
        yield StateChangeEvent(
            current=DeploymentState.WARMING_UP,
            operation=OperationType.WARMUP,
            detail=f"Probing {server_url}",
        )

        is_vllm = backend == BackendType.VLLM
        probe_url = status_probe_url(backend, server_url)
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

        if tail_logs and provider == ComputeProvider.MODAL:
            try:
                logs_proc = _start_logs_tail()
            except Exception as exc:
                yield LogEvent(line=f"Warning: failed to start log tailing: {exc}")
                logs_retry_at = time.time() + 5.0

        try:
            import requests
        except ImportError:
            yield ErrorEvent(
                message="'requests' is required. Install with: pip install requests",
                operation=OperationType.WARMUP,
                recoverable=False,
            )
            yield OperationCompleteEvent(
                operation=OperationType.WARMUP,
                success=False,
                exit_code=1,
            )
            return

        llama_probe_model = (served_model_name or "").strip() or "default"
        payload = {
            "model": llama_probe_model,
            "prompt": "ping",
            "max_tokens": 1,
            "temperature": 0,
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
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
            if is_shutting_down():
                if logs_proc:
                    try:
                        logs_proc.terminate()
                    except Exception:
                        pass
                return

            if tail_logs and logs_proc is None and time.time() >= logs_retry_at:
                # Historical-fetch fallback: the live stream may miss the
                # final output of a crashing container, while Modal's persisted
                # logs still have it.
                yield from fetch_historical_logs(target_app_name, seen_log_lines)
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
                if is_vllm:
                    resp = requests.get(probe_url, headers=headers, timeout=10)
                else:
                    resp = requests.post(
                        probe_url, headers=headers, data=json.dumps(payload), timeout=10
                    )
                body = resp.text or ""
                ready_ok, readiness_reason = probe_response_is_ready(
                    backend, resp.status_code, body
                )
                if ready_ok:
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
                        api_key=api_key,
                    )
                    yield LogEvent(line=f"Test command:\n{curl_cmd}")
                    yield StateChangeEvent(
                        current=DeploymentState.HEALTHY, operation=OperationType.WARMUP
                    )
                    yield OperationCompleteEvent(
                        operation=OperationType.WARMUP, success=True, data={"url": server_url}
                    )
                    return
                if 200 <= resp.status_code < 300 and readiness_reason:
                    last_err = f"HTTP {resp.status_code} (not ready): {readiness_reason}; body={body[:200]}"
                else:
                    last_err = f"HTTP {resp.status_code}: {body[:200]}"
                scheduling_hint = modal_gpu_scheduling_hint(body)
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
                if (
                    tail_logs
                    and provider == ComputeProvider.PRIME
                    and pod_id
                    and prime_backend is not None
                ):
                    try:
                        lines = prime_backend.get_pod_logs(pod_id, tail=200)
                        for line in lines:
                            if line not in seen_log_lines:
                                seen_log_lines.add(line)
                                yield LogEvent(line=line, operation=OperationType.WARMUP)
                    except Exception:
                        pass
            except Exception as exc:
                last_err = str(exc)

            # Sleep in small increments, draining the log queue between
            # each chunk so that lines appear in the TUI promptly.
            sleep_end = time.time() + backoff
            while time.time() < sleep_end:
                if is_shutting_down():
                    break
                chunk = min(0.5, sleep_end - time.time())
                if chunk > 0:
                    shutdown_event().wait(timeout=chunk)
                yield from _drain_queue()
            backoff = min(max_backoff, backoff * 1.5)


def fetch_historical_logs(
    app_name: str,
    seen: set[str],
) -> EventStream:
    """Re-run ``modal app logs`` to capture persisted crash output."""
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
