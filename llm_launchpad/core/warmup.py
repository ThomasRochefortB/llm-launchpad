"""Endpoint warmup and Modal log tailing helpers."""

from __future__ import annotations

from ..protocol.models import VisionCapabilities
# The startup-phase line format lives beside the other milestone line parsers
# in deploy_log_summary, which is also what preserves the lines through
# summarization. Re-exported here because the phase timer is what emits them.
from .deploy_log_summary import (
    parse_startup_phase_line as parse_startup_phase_line,
    startup_phase_summary_line,
)
from .vision_probe import VISION_PROBE_FAILED, verify_image_request

from .shutdown import is_shutting_down, shutdown_event

import json
import os
import queue
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from collections.abc import Generator, Iterable
from urllib.parse import urlsplit

from ..gguf_files import is_target_gguf
from ..protocol.enums import (
    BackendType,
    ComputeProvider,
    DeploymentState,
    EvidenceLevel,
    OperationType,
    ServingObjective,
)
from ..protocol.events import (
    BaseEvent,
    LogEvent,
    OperationCompleteEvent,
    StateChangeEvent,
)
from ..protocol.models import (
    PerformancePoint,
    PlacementAssessment,
    RuntimeAttestation,
    ServingRequirements,
)
from .backend import ModalBackend
from .fit_calibration import FitCalibrationRecorder
from .operation_events import fail_operation
from .diagnostics import log_exception
if TYPE_CHECKING:
    from .runtime_evidence import LlamacppRuntimeEvidence

from .llamacpp_planner import (
    attestation_now,
    load_runtime_attestation,
    save_runtime_attestation,
)
from .naming import legacy_app_name
from .prime_backend import PrimeBackend

EventStream = Generator[BaseEvent, None, None]

_MODAL_GPU_WAIT_RE = re.compile(
    r"waiting to be scheduled on a (?P<worker>[A-Za-z0-9_:\-]+) worker",
    flags=re.IGNORECASE,
)
_MODAL_RELAX_RE = re.compile(r"Relaxing requirements \((?P<requirements>[^)]+)\)", flags=re.IGNORECASE)
_LLAMACPP_CONTEXT_LOAD_FAILURE_RE = re.compile(
    r"\bsrv\s+load_model:\s+failed to create_context with model ['\"](?P<path>.+?\.gguf)['\"]",
    re.IGNORECASE,
)
_LLAMACPP_AUXILIARY_MODEL_FAILURE = (
    "llama.cpp could not load the selected GGUF as the main model. "
    "Choose a main-model quantization."
)

# A serving plan the placement cannot hold is rejected identically on every
# container restart, so waiting out the readiness timeout only replays the same
# crash for half an hour. Matched on the runner's own sentence rather than
# llama.cpp's, which names the guard it aborted at rather than the cause.
_TERMINAL_STARTUP_MARKERS = (
    "llama.cpp rejected the full-context GPU-only serving plan",
    "No target-model GGUF files matched the requested repo/quant for llama.cpp",
)


def terminal_startup_failure(line: str) -> str | None:
    """Return the message when a log line reports a crash that will repeat."""

    text = (line or "").strip()
    for marker in _TERMINAL_STARTUP_MARKERS:
        index = text.find(marker)
        if index >= 0:
            # The marker can arrive wrapped in a traceback or a repr, so the
            # message is cut out of whichever framing carried it.
            return text[index:].strip().rstrip("')\"") or marker
    return None


class StartupPhaseTimer:
    """Emit milestone lines that break one deploy+warmup run into phases.

    The four phases are emitted separately so callers can reconstruct which
    part of the run dominated wall-clock time:

    - ``deploy``: deploy start until the deploy operation completes.
    - ``warmup-wait``: warmup start until first successful readiness probe
      (i.e. provider scheduling, binding, and model load).
    - ``calibration``: certification start until calibration returns.
    - ``total``: deploy start until warmup completion.
    """

    def __init__(self) -> None:
        self._deploy_started: float | None = None
        self._warmup_started: float | None = None
        self._ready_at: float | None = None
        self._calibration_started: float | None = None

    def deploy_started(self, *, at: float | None = None) -> None:
        self._deploy_started = at if at is not None else time.monotonic()

    def warmup_started(self, *, at: float | None = None) -> None:
        self._warmup_started = at if at is not None else time.monotonic()

    def ready(self, *, at: float | None = None) -> None:
        now = at if at is not None else time.monotonic()
        if self._ready_at is None:
            self._ready_at = now

    def calibration_started(self, *, at: float | None = None) -> None:
        now = at if at is not None else time.monotonic()
        if self._calibration_started is None:
            self._calibration_started = now

    def deploy_event(self, operation: OperationType) -> LogEvent | None:
        """Return the ``deploy`` phase line when deploy completes, else None."""
        if self._deploy_started is None:
            return None
        seconds = time.monotonic() - self._deploy_started
        line = startup_phase_summary_line("deploy", seconds)
        return LogEvent(line=line, operation=operation, is_milestone=True)

    def calibration_event(self) -> LogEvent | None:
        """Return the ``calibration`` phase line for a finished calibration."""
        if self._calibration_started is None:
            return None
        seconds = time.monotonic() - self._calibration_started
        line = startup_phase_summary_line("calibration", seconds)
        return LogEvent(
            line=line, operation=OperationType.WARMUP, is_milestone=True
        )

    def finish_events(self) -> list[LogEvent]:
        """Return warmup-wait/total lines when warmup completes."""
        if self._warmup_started is None or self._deploy_started is None:
            return []
        now = time.monotonic()
        ready_at = self._ready_at if self._ready_at is not None else now
        return [
            LogEvent(
                line=startup_phase_summary_line(
                    "warmup-wait", ready_at - self._warmup_started
                ),
                operation=OperationType.WARMUP,
                is_milestone=True,
            ),
            LogEvent(
                line=startup_phase_summary_line("total", now - self._deploy_started),
                operation=OperationType.WARMUP,
                is_milestone=True,
            ),
        ]


_CALIBRATION_BUDGET_SECONDS = 90.0
_CALIBRATION_WARMUP_TIMEOUT_SECONDS = 120.0
_GENERAL_PURPOSE_MIN_OUTPUT_TPS = 8.0
_DEFAULT_PROBE_TIMEOUT_SECONDS = 10.0
# Modal's @modal.web_server proxy holds a cold request open while scheduling
# and booting a container (up to 150s before returning a 303). Disconnecting
# at 10s aborts the demand signal during multi-GPU cold starts, so hold Modal
# probes up to 60s while draining logs concurrently on the main loop.
_MODAL_PROBE_TIMEOUT_SECONDS = 60.0
_MODAL_COLD_RETRY_BACKOFF_SECONDS = 2.0


def _probe_timeout_seconds(
    provider: ComputeProvider,
    timeout: float,
    elapsed: float,
) -> float:
    """Return the per-request HTTP timeout for one readiness probe."""
    if provider != ComputeProvider.MODAL:
        return _DEFAULT_PROBE_TIMEOUT_SECONDS
    remaining = max(_DEFAULT_PROBE_TIMEOUT_SECONDS, float(timeout) - max(0.0, elapsed))
    return min(_MODAL_PROBE_TIMEOUT_SECONDS, remaining)


def _start_probe_thread(
    send_probe: Any,
    probe_timeout: float,
) -> queue.Queue[tuple[Any, Exception | None]]:
    """Run one HTTP readiness probe on a daemon thread and return its result queue."""
    probe_queue: queue.Queue[tuple[Any, Exception | None]] = queue.Queue(maxsize=1)

    def _worker() -> None:
        try:
            probe_queue.put((send_probe(probe_timeout), None))
        except Exception as exc:
            probe_queue.put((None, exc))

    threading.Thread(target=_worker, daemon=True).start()
    return probe_queue


def extract_effective_context(payload: object) -> int | None:
    """Extract the active context size from a llama.cpp ``/props`` payload."""

    exact_keys = {"n_ctx", "ctx_size", "context_size", "context_tokens"}
    values: list[int] = []

    def _walk(node: object) -> None:
        if isinstance(node, dict):
            for raw_key, value in node.items():
                key = str(raw_key).strip().casefold()
                if key in exact_keys:
                    try:
                        numeric = int(value)
                    except (TypeError, ValueError):
                        numeric = 0
                    if numeric > 0:
                        values.append(numeric)
                if isinstance(value, (dict, list, tuple)):
                    _walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                _walk(value)

    _walk(payload)
    return max(values) if values else None


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


def _recent_modal_scheduling_hint(app_name: str, started_at: float) -> str | None:
    """Read scheduling messages that the live log stream may have missed."""
    for line in reversed(ModalBackend.recent_system_logs(app_name, started_at)):
        if hint := modal_gpu_scheduling_hint(line):
            return hint
    return None


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


def endpoint_url_error(server_url: str) -> str | None:
    """Return a useful error when an endpoint cannot be represented in DNS."""
    try:
        parsed = urlsplit(server_url)
        hostname = parsed.hostname or ""
    except ValueError as exc:
        return f"invalid hostname: {exc}"
    if parsed.scheme not in {"http", "https"}:
        return "URL must use http or https"
    if not hostname:
        return "URL has no hostname"
    raw_oversized = next((label for label in hostname.split(".") if len(label) > 63), None)
    if raw_oversized is not None:
        return f"DNS label is {len(raw_oversized)} characters; maximum is 63"
    try:
        ascii_hostname = hostname.encode("idna").decode("ascii")
    except UnicodeError as exc:
        return f"invalid hostname: {exc}"
    oversized = next((label for label in ascii_hostname.split(".") if len(label) > 63), None)
    if oversized is not None:
        return f"DNS label is {len(oversized)} characters; maximum is 63"
    if len(ascii_hostname) > 253:
        return f"hostname is {len(ascii_hostname)} characters; maximum is 253"
    return None


class _ModalLogTail:
    """A live tail of one Modal app's logs, drained without blocking the caller.

    A dedicated reader thread feeds a queue so that every line the subprocess
    emits is captured immediately, regardless of Python's TextIOWrapper
    read-ahead buffering. (``select()`` only checks the kernel pipe buffer, but
    ``readline()`` can consume up to 2 KB into an internal buffer in a single
    call -- a mismatch that leaves lines "stuck" until the next kernel read.)
    """

    RETRY_DELAY_SECONDS = 5.0

    def __init__(self, app_name: str) -> None:
        self.app_name = app_name
        self.seen_lines: set[str] = set()
        self.ordered_lines: list[str] = []
        self._proc: subprocess.Popen[str] | None = None
        self._queue: queue.Queue[str | None] | None = None
        self._retry_at = 0.0

    def record_line(self, line: str) -> bool:
        """Record a line in chronological arrival order; return True if newly seen."""
        self.ordered_lines.append(line)
        if line in self.seen_lines:
            return False
        self.seen_lines.add(line)
        return True

    def may_attach(self) -> bool:
        """Return whether the stream is detached and its retry delay has passed."""
        return self._proc is None and time.time() >= self._retry_at

    def attach(self) -> Generator[LogEvent, None, None]:
        """Start the live stream, reporting a failure as a warning log line."""
        try:
            self._proc = self._spawn()
        except Exception as exc:
            yield LogEvent(line=f"Warning: failed to start log tailing: {exc}")
            self._retry_at = time.time() + self.RETRY_DELAY_SECONDS

    def drain(self) -> Generator[LogEvent, None, None]:
        """Yield whatever the reader thread has queued, without blocking."""
        if self._queue is None:
            return
        while True:
            try:
                line = self._queue.get_nowait()
            except queue.Empty:
                break
            if line is None:
                # Reader thread finished -- subprocess exited.
                self._proc = None
                self._queue = None
                self._retry_at = time.time() + self.RETRY_DELAY_SECONDS
                break
            self.record_line(line)
            yield LogEvent(line=line, operation=OperationType.WARMUP)

    def stop(self, reason: str) -> None:
        """Terminate the stream, logging rather than raising on failure."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=2.0)
                except Exception:
                    pass
        except Exception:
            log_exception(f"Failed to terminate Modal log tail {reason}")
        finally:
            if proc.stdout is not None:
                try:
                    proc.stdout.close()
                except Exception:
                    pass

    def _spawn(self) -> subprocess.Popen[str]:
        follow = ModalBackend.logs_follow_args()
        # PYTHONUNBUFFERED forces the modal CLI (a Python program) to
        # flush each write immediately when stdout is a pipe.
        unbuf_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            ModalBackend._resolve_command(
                ["modal", "app", "logs", *follow, self.app_name]
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=unbuf_env,
        )
        q: queue.Queue[str | None] = queue.Queue()
        self._queue = q

        def _reader() -> None:
            try:
                assert proc.stdout is not None
                for raw_line in proc.stdout:
                    q.put(raw_line.rstrip("\n"))
            except Exception:
                log_exception("Modal log-tail reader stopped unexpectedly")
            finally:
                q.put(None)  # sentinel: reader finished

        threading.Thread(target=_reader, daemon=True).start()
        return proc


def _verify_runtime_context(
    requests_module: Any,
    *,
    backend: BackendType,
    server_url: str,
    headers: dict[str, str],
    serving_requirements: ServingRequirements,
) -> Generator[BaseEvent, None, tuple[LlamacppRuntimeEvidence, int] | None]:
    """Read ``/props`` for llama.cpp or accept vLLM's requested context."""
    from .runtime_evidence import combine_runtime_evidence, runtime_props_evidence

    runtime_evidence = combine_runtime_evidence()
    if backend != BackendType.LLAMACPP:
        return runtime_evidence, serving_requirements.context_tokens

    props_url = endpoint_root_url(server_url) + "/props"
    try:
        props_response = requests_module.get(
            props_url,
            headers=headers,
            timeout=15,
        )
        props_response.raise_for_status()
        runtime_evidence = combine_runtime_evidence(
            runtime_evidence,
            runtime_props_evidence(props_response.json()),
        )
        return runtime_evidence, runtime_evidence.effective_context_tokens or 0
    except Exception as exc:
        last_err = f"runtime property verification failed: {exc}"
        yield from fail_operation(
            OperationType.WARMUP,
            f"Endpoint responded, but Launchpad could not verify its effective context: {exc}",
            recoverable=False,
            detail=last_err,
        )
        return None


def _verify_gpu_residency(
    *,
    backend: BackendType,
    serving_requirements: ServingRequirements,
    placement_assessment: PlacementAssessment,
    runtime_evidence: LlamacppRuntimeEvidence,
    runtime_log_text: str,
    app_name: str,
    provider: ComputeProvider = ComputeProvider.MODAL,
    pod_id: str | None = None,
    prime_backend: Any | None = None,
) -> Generator[BaseEvent, None, tuple[LlamacppRuntimeEvidence, bool, EvidenceLevel | None] | None]:
    """Check runtime offload logs/cache and reject contradicted GPU residency."""
    from .runtime_evidence import (
        LlamacppRuntimeEvidence,
        combine_runtime_evidence,
        remembered_attestation_evidence,
        runtime_log_evidence,
    )

    if backend == BackendType.LLAMACPP and runtime_log_text.strip():
        runtime_evidence = combine_runtime_evidence(
            runtime_evidence,
            runtime_log_evidence(runtime_log_text),
        )
    observed_residency = runtime_evidence.gpu_resident
    residency_evidence: EvidenceLevel | None = (
        runtime_evidence.offload_evidence if observed_residency is not None else None
    )
    if backend == BackendType.LLAMACPP and serving_requirements.gpu_only:
        if observed_residency is None:
            # The live tail can lose the offload report to a burst upstream, so
            # look again before concluding the runtime never said anything.
            recovered = LlamacppRuntimeEvidence()
            if provider == ComputeProvider.MODAL:
                recovered = _refetched_log_evidence(app_name)
            elif provider == ComputeProvider.PRIME and pod_id and prime_backend is not None:
                try:
                    lines = prime_backend.get_pod_logs(pod_id, tail=200)
                    recovered = runtime_log_evidence("\n".join(lines))
                except Exception:
                    log_exception(f"Failed to re-read Prime pod logs for {pod_id}")
            if recovered.gpu_resident is None:
                recovered = remembered_attestation_evidence(
                    load_runtime_attestation(placement_assessment.fingerprint)
                )
            if recovered.gpu_resident is not None:
                runtime_evidence = combine_runtime_evidence(runtime_evidence, recovered)
                observed_residency = runtime_evidence.gpu_resident
                residency_evidence = runtime_evidence.offload_evidence
                yield LogEvent(
                    line=f"GPU residency: {recovered.detail}.",
                    operation=OperationType.WARMUP,
                )
        if observed_residency is False:
            # Fail when the runtime explicitly placed layers off GPU.
            detail = (
                f"Requested {serving_requirements.context_tokens:,} context with "
                f"GPU-only placement; the runtime reported "
                f"{runtime_evidence.gpu_layers}/{runtime_evidence.total_layers} "
                "layers on GPU."
            )
            yield from fail_operation(
                OperationType.WARMUP,
                f"Runtime attestation failed: {detail}",
                recoverable=False,
                detail=detail,
            )
            return None
        if observed_residency is None:
            residency_evidence = EvidenceLevel.PREDICTED
            yield LogEvent(
                line=(
                    "GPU residency could not be verified: the runtime published no "
                    "offload report in the captured logs. The endpoint is serving and "
                    "its certificate records residency as predicted, not observed."
                ),
                operation=OperationType.WARMUP,
            )
        gpu_resident = (
            bool(observed_residency)
            if observed_residency is not None
            else bool(
                placement_assessment.gpu_resident
                and placement_assessment.tuning.gpu_layers.casefold() == "all"
            )
        )
    else:
        gpu_resident = bool(
            placement_assessment.gpu_resident
            and placement_assessment.tuning.gpu_layers.casefold() == "all"
        )
    return runtime_evidence, gpu_resident, residency_evidence


def _certify_serving_requirements(
    requests_module: Any,
    *,
    backend: BackendType,
    server_url: str,
    headers: dict[str, str],
    probe_model: str,
    serving_requirements: ServingRequirements,
    placement_assessment: PlacementAssessment | None,
    runtime_id: str | None,
    phase_timer: StartupPhaseTimer | None = None,
    price_per_hour_usd: float | None = None,
    runtime_log_text: str = "",
    app_name: str = "",
    provider: ComputeProvider = ComputeProvider.MODAL,
    pod_id: str | None = None,
    prime_backend: Any | None = None,
) -> Generator[BaseEvent, None, RuntimeAttestation | None]:
    """Verify full-context GPU residency and throughput for a ready endpoint.

    Yields certification progress and, on failure, the events that end the
    warmup. Returns the attestation, or ``None`` when certification failed and
    the caller should stop.
    """
    yield StateChangeEvent(
        current=DeploymentState.VERIFYING,
        operation=OperationType.WARMUP,
        detail="Verifying full-context GPU residency",
    )
    context_result = yield from _verify_runtime_context(
        requests_module,
        backend=backend,
        server_url=server_url,
        headers=headers,
        serving_requirements=serving_requirements,
    )
    if context_result is None:
        return None
    runtime_evidence, effective_context = context_result

    if placement_assessment is None:
        detail = "No placement assessment was supplied for runtime certification."
        yield from fail_operation(
            OperationType.WARMUP,
            f"Runtime attestation failed: {detail}",
            recoverable=False,
            detail=detail,
        )
        return None

    residency_result = yield from _verify_gpu_residency(
        backend=backend,
        serving_requirements=serving_requirements,
        placement_assessment=placement_assessment,
        runtime_evidence=runtime_evidence,
        runtime_log_text=runtime_log_text,
        app_name=app_name,
        provider=provider,
        pod_id=pod_id,
        prime_backend=prime_backend,
    )
    if residency_result is None:
        return None
    runtime_evidence, gpu_resident, residency_evidence = residency_result

    if (
        effective_context < serving_requirements.context_tokens
        or (serving_requirements.gpu_only and not gpu_resident)
    ):
        detail = (
            f"Requested {serving_requirements.context_tokens:,} context, "
            f"runtime reported {effective_context:,}; "
            f"GPU-only={gpu_resident}."
        )
        yield from fail_operation(
            OperationType.WARMUP,
            f"Runtime attestation failed: {detail}",
            recoverable=False,
            detail=detail,
        )
        return None

    yield StateChangeEvent(
        current=DeploymentState.CALIBRATING,
        operation=OperationType.WARMUP,
        detail="Measuring endpoint throughput",
    )
    performance = _calibrate_endpoint(
        requests_module,
        server_url=server_url,
        model=probe_model,
        headers=headers,
        parallel_slots=placement_assessment.tuning.parallel_slots,
        price_per_hour_usd=price_per_hour_usd,
        budget_seconds=_CALIBRATION_BUDGET_SECONDS,
        phase_timer=phase_timer,
    )
    if phase_timer is not None:
        # A test double or future reimplementation may not report the start;
        # the calibration phase still ended here.
        phase_timer.calibration_started()
        calibration_line = phase_timer.calibration_event()
        if calibration_line is not None:
            yield calibration_line
    accepted, calibration_detail = _calibration_is_acceptable(
        performance,
        serving_requirements.objective,
    )
    if not accepted:
        yield from fail_operation(
            OperationType.WARMUP,
            f"Endpoint performance certification failed: {calibration_detail}",
        )
        return None

    return (
        yield from _record_certified_attestation(
            backend=backend,
            serving_requirements=serving_requirements,
            placement_assessment=placement_assessment,
            runtime_evidence=runtime_evidence,
            effective_context=effective_context,
            gpu_resident=gpu_resident,
            residency_evidence=residency_evidence,
            performance=performance,
            runtime_id=runtime_id,
        )
    )


def _record_certified_attestation(
    *,
    backend: BackendType,
    serving_requirements: ServingRequirements,
    placement_assessment: PlacementAssessment,
    runtime_evidence: LlamacppRuntimeEvidence,
    effective_context: int,
    gpu_resident: bool,
    residency_evidence: EvidenceLevel | None,
    performance: tuple[PerformancePoint, ...],
    runtime_id: str | None,
) -> Generator[BaseEvent, None, RuntimeAttestation]:
    """Persist and announce a verified runtime attestation."""
    total_layers = max(1, placement_assessment.memory.total_layer_count or 0)
    observed_gpu_layers = runtime_evidence.gpu_layers
    observed_total_layers = runtime_evidence.total_layers
    if backend == BackendType.LLAMACPP and observed_gpu_layers is not None:
        gpu_layers = observed_gpu_layers
        total_layers = observed_total_layers or total_layers
    else:
        gpu_layers = total_layers
    attestation = attestation_now(
        fingerprint=placement_assessment.fingerprint,
        requested_context_tokens=serving_requirements.context_tokens,
        effective_context_tokens=effective_context,
        gpu_layers=gpu_layers,
        total_layers=total_layers,
        gpu_resident=gpu_resident,
        memory=placement_assessment.memory,
        performance=performance,
        runtime_id=runtime_id,
        residency_evidence=residency_evidence,
    )
    try:
        save_runtime_attestation(attestation)
    except Exception as exc:
        yield LogEvent(
            line=f"Warning: could not cache serving certificate: {exc}",
            operation=OperationType.WARMUP,
        )
    best_single = max(
        (point.output_tokens_per_second or 0.0 for point in performance if point.concurrency == 1),
        default=0.0,
    )
    best_aggregate = max(
        (point.aggregate_output_tokens_per_second or 0.0 for point in performance),
        default=0.0,
    )
    metric = ""
    if best_single > 0:
        metric = f" · {best_single:.1f} tok/s single, {best_aggregate:.1f} tok/s aggregate"
    yield LogEvent(
        line=(
            f"Configured {effective_context:,}-token context with "
            f"{gpu_layers}/{total_layers} layers observed on GPU"
            f"{metric}."
        ),
        operation=OperationType.WARMUP,
        is_milestone=True,
    )
    return attestation


@dataclass(frozen=True)
class WarmupRequest:
    """Typed parameters for one endpoint warmup and certification run."""

    backend: BackendType
    server_url: str
    timeout: int = 1800
    tail_logs: bool = True
    app_name: str | None = None
    served_model_name: str | None = None
    provider: ComputeProvider = ComputeProvider.MODAL
    api_key: str | None = None
    pod_id: str | None = None
    prime_backend: PrimeBackend | None = None
    serving_requirements: ServingRequirements | None = None
    placement_assessment: PlacementAssessment | None = None
    runtime_id: str | None = None
    vision: VisionCapabilities | None = None
    phase_timer: StartupPhaseTimer | None = None
    price_per_hour_usd: float | None = None


class _WarmupSession:
    """Stateful execution driver for one :class:`WarmupRequest`."""

    _MAX_BACKOFF_SECONDS = 30.0

    def __init__(self, request: WarmupRequest) -> None:
        self.req = request
        self.start = time.time()
        self.is_vllm = request.backend == BackendType.VLLM
        self.probe_url = status_probe_url(request.backend, request.server_url)
        self.target_app_name = request.app_name or legacy_app_name(request.backend)
        self.tail_modal_logs = request.tail_logs and request.provider == ComputeProvider.MODAL
        self.calibration = FitCalibrationRecorder(request.placement_assessment)
        self.log_tail = _ModalLogTail(self.target_app_name)
        self.llama_probe_model = (request.served_model_name or "").strip() or "default"
        self.payload = {
            "model": self.llama_probe_model,
            "prompt": "ping",
            "max_tokens": 1,
            "temperature": 0,
        }
        self.headers = {"Content-Type": "application/json"}
        if request.api_key:
            self.headers["Authorization"] = f"Bearer {request.api_key}"
        self.last_err: str | None = None
        self.last_scheduling_hint: str | None = None
        self.next_system_log_check = self.start + 20
        self.unrecoverable: str | None = None
        self.draft_context_error = False
        self.managed_dspark = False
        self._log_tail_stopped = False

    def _stop_log_tail(self, reason: str) -> None:
        if self._log_tail_stopped:
            return
        self._log_tail_stopped = True
        self.log_tail.stop(reason)

    def _timeout_message(self, detail: str | None) -> str:
        if self.last_scheduling_hint:
            return (
                f"Timed out after {self.req.timeout}s "
                f"({self.last_scheduling_hint}). Last error: {detail}"
            )
        return f"Timed out after {self.req.timeout}s. Last error: {detail}"

    def _report_scheduling(self, hint: str | None) -> EventStream:
        if hint and hint != self.last_scheduling_hint:
            self.last_scheduling_hint = hint
            yield StateChangeEvent(
                current=DeploymentState.QUEUED,
                operation=OperationType.WARMUP,
                detail=hint,
            )
            yield LogEvent(line=hint, operation=OperationType.WARMUP)

    def _drain_live_logs(self, events: Iterable[BaseEvent] | None = None) -> EventStream:
        stream = events if events is not None else self.log_tail.drain()
        for event in stream:
            if not isinstance(event, LogEvent):
                yield event
                continue
            line = event.line
            if self.tail_modal_logs:
                yield from self._report_scheduling(modal_gpu_scheduling_hint(line))
            yield event
            if self.req.backend == BackendType.LLAMACPP:
                self.managed_dspark |= (
                    "using DSpark draft file:" in line or "--spec-type draft-dspark" in line
                )
                self.draft_context_error |= "dflash requires ctx_other to be set" in line
                failed_load = _LLAMACPP_CONTEXT_LOAD_FAILURE_RE.search(line)
                if (
                    failed_load
                    and not is_target_gguf(failed_load["path"].rsplit("/", 1)[-1])
                ) or (
                    self.draft_context_error
                    and not self.managed_dspark
                    and "exiting due to model loading error" in line
                ):
                    self.unrecoverable = _LLAMACPP_AUXILIARY_MODEL_FAILURE
            if self.unrecoverable is None:
                self.unrecoverable = terminal_startup_failure(line)
            measured = self.calibration.observe(line)
            if measured is not None:
                yield LogEvent(
                    line=(
                        "Recorded llama.cpp's own measurement: "
                        f"{measured.graph_gb:.1f} GB of graph memory per GPU "
                        f"(measured on {measured.gpu_count}x {measured.gpu_type})."
                    ),
                    operation=OperationType.WARMUP,
                )

    def _reattach_modal_logs_if_needed(self) -> EventStream:
        if self.tail_modal_logs and self.log_tail.may_attach():
            # Historical-fetch fallback: the live stream may miss the final
            # output of a crashing container while Modal's persisted logs keep it.
            yield from self._drain_live_logs(
                fetch_historical_logs(self.target_app_name, self.log_tail.seen_lines)
            )
            self._log_tail_stopped = False
            yield from self.log_tail.attach()

    def _fail_if_unrecoverable(self) -> Generator[BaseEvent, None, bool]:
        if self.unrecoverable is None:
            return False
        self._stop_log_tail("on unrecoverable startup failure")
        yield from fail_operation(
            OperationType.WARMUP,
            self.unrecoverable,
            recoverable=False,
            detail=self.unrecoverable,
        )
        return True

    def _maybe_check_system_scheduling(self, now: float) -> EventStream:
        if (
            self.tail_modal_logs
            and self.last_scheduling_hint is None
            and now >= self.next_system_log_check
        ):
            yield from self._report_scheduling(
                _recent_modal_scheduling_hint(self.target_app_name, self.start)
            )
            self.next_system_log_check = now + 30

    def _send_probe(self, requests_module: Any, probe_timeout: float) -> Any:
        if self.is_vllm:
            return requests_module.get(
                self.probe_url, headers=self.headers, timeout=probe_timeout
            )
        return requests_module.post(
            self.probe_url,
            headers=self.headers,
            data=json.dumps(self.payload),
            timeout=probe_timeout,
        )

    def _execute_probe(
        self, requests_module: Any, probe_timeout: float
    ) -> Generator[BaseEvent, None, tuple[Any, Exception | None, bool]]:
        """Run one probe attempt, draining Modal logs while waiting."""
        if self.req.provider != ComputeProvider.MODAL:
            try:
                return self._send_probe(requests_module, probe_timeout), None, False
            except Exception as exc:
                return None, exc, False

        probe_queue = _start_probe_thread(
            lambda t: self._send_probe(requests_module, t), probe_timeout
        )
        while True:
            try:
                probe_resp, probe_exc = probe_queue.get(timeout=0.5)
                return probe_resp, probe_exc, False
            except queue.Empty:
                if is_shutting_down():
                    self._stop_log_tail("on shutdown")
                    return None, None, True
                yield from self._reattach_modal_logs_if_needed()
                yield from self._drain_live_logs()
                if (yield from self._fail_if_unrecoverable()):
                    return None, None, True
                now = time.time()
                if now - self.start > self.req.timeout:
                    self._stop_log_tail("on timeout")
                    yield from fail_operation(
                        OperationType.WARMUP,
                        self._timeout_message(
                            self.last_err or f"Read timed out while probing {self.probe_url}"
                        ),
                    )
                    return None, None, True
                yield from self._maybe_check_system_scheduling(now)

    def _handle_ready_endpoint(self, requests_module: Any) -> EventStream:
        """Certify residency/vision, emit the curl hint, and finish healthy."""
        self._stop_log_tail("after readiness")
        if self.req.phase_timer is not None:
            self.req.phase_timer.ready()
        attestation = None
        if self.req.serving_requirements is not None:
            ordered_lines = getattr(self.log_tail, "ordered_lines", None)
            runtime_log_lines = (
                ordered_lines
                if ordered_lines is not None
                else sorted(self.log_tail.seen_lines)
            )
            attestation = yield from _certify_serving_requirements(
                requests_module,
                backend=self.req.backend,
                server_url=self.req.server_url,
                headers=self.headers,
                probe_model=self.llama_probe_model,
                serving_requirements=self.req.serving_requirements,
                placement_assessment=self.req.placement_assessment,
                runtime_id=self.req.runtime_id,
                phase_timer=self.req.phase_timer,
                price_per_hour_usd=self.req.price_per_hour_usd,
                runtime_log_text="\n".join(runtime_log_lines),
                app_name=self.target_app_name or "",
                provider=self.req.provider,
                pod_id=self.req.pod_id,
                prime_backend=self.req.prime_backend,
            )
            if attestation is None:
                return

        if self.req.vision is not None and self.req.vision.enabled:
            yield LogEvent(
                line="Verifying image input with a bundled PNG.",
                operation=OperationType.WARMUP,
            )
            try:
                verify_image_request(
                    self.req.server_url,
                    self.req.served_model_name,
                    self.req.api_key,
                    self.req.vision,
                )
            except Exception as exc:
                yield from fail_operation(
                    OperationType.WARMUP,
                    f"Image verification failed: {exc}",
                    data={VISION_PROBE_FAILED: True},
                )
                return
            yield LogEvent(
                line="Image request verified (not a visual accuracy benchmark).",
                operation=OperationType.WARMUP,
            )
        yield LogEvent(line="Server is ready!")
        if self.req.phase_timer is not None:
            yield from self.req.phase_timer.finish_events()
        curl_cmd = ModalBackend.test_curl_command(
            self.req.backend,
            self.req.server_url,
            served_model_name=self.req.served_model_name,
            api_key=self.req.api_key,
        )
        if self.req.api_key:
            curl_cmd = curl_cmd.replace(self.req.api_key, "$LLM_LAUNCHPAD_API_KEY")
        yield LogEvent(line=f"Test command:\n{curl_cmd}")
        yield StateChangeEvent(
            current=DeploymentState.HEALTHY, operation=OperationType.WARMUP
        )
        yield OperationCompleteEvent(
            operation=OperationType.WARMUP,
            success=True,
            data={"url": self.req.server_url, "attestation": attestation},
        )

    def _handle_non_ready_response(self, status_code: int, body: str, readiness_reason: str | None) -> EventStream:
        if 200 <= status_code < 300 and readiness_reason:
            self.last_err = f"HTTP {status_code} (not ready): {readiness_reason}; body={body[:200]}"
        else:
            self.last_err = f"HTTP {status_code}: {body[:200]}"
        if self.req.provider == ComputeProvider.MODAL:
            yield from self._report_scheduling(modal_gpu_scheduling_hint(body))
        if (
            self.req.tail_logs
            and self.req.provider == ComputeProvider.PRIME
            and self.req.pod_id
            and self.req.prime_backend is not None
        ):
            try:
                lines = self.req.prime_backend.get_pod_logs(self.req.pod_id, tail=200)
                for line in lines:
                    if line not in self.log_tail.seen_lines:
                        if hasattr(self.log_tail, "record_line"):
                            self.log_tail.record_line(line)
                        else:
                            self.log_tail.seen_lines.add(line)
                        yield LogEvent(line=line, operation=OperationType.WARMUP)
            except Exception:
                log_exception("Failed to fetch Prime pod logs during warmup")

    def _sleep_between_probes(self, wait_seconds: float) -> EventStream:
        sleep_end = time.time() + wait_seconds
        while time.time() < sleep_end:
            if is_shutting_down():
                break
            chunk = min(0.5, sleep_end - time.time())
            if chunk > 0:
                shutdown_event().wait(timeout=chunk)
            yield from self._drain_live_logs()
            if self.unrecoverable is not None:
                break

    def run(self) -> EventStream:
        """Execute the readiness probe loop and certification steps."""
        yield StateChangeEvent(
            current=DeploymentState.WARMING_UP,
            operation=OperationType.WARMUP,
            detail=f"Probing {self.req.server_url}",
            timeout_seconds=self.req.timeout,
        )
        yield LogEvent(line=f"Probing readiness at: {self.probe_url}")
        url_error = endpoint_url_error(self.probe_url)
        if url_error:
            yield from fail_operation(
                OperationType.WARMUP,
                f"Invalid endpoint URL: {url_error}",
                exit_code=2,
                recoverable=False,
            )
            return

        try:
            if self.tail_modal_logs:
                yield from self.log_tail.attach()

            try:
                import requests
            except ImportError:
                yield from fail_operation(
                    OperationType.WARMUP,
                    "'requests' is required. Install with: pip install requests",
                    recoverable=False,
                )
                return

            backoff = 2.0
            while True:
                if is_shutting_down():
                    self._stop_log_tail("on shutdown")
                    return

                yield from self._reattach_modal_logs_if_needed()
                yield from self._drain_live_logs()
                if (yield from self._fail_if_unrecoverable()):
                    return

                elapsed = time.time() - self.start
                if elapsed > self.req.timeout:
                    self._stop_log_tail("on timeout")
                    yield from fail_operation(
                        OperationType.WARMUP,
                        self._timeout_message(self.last_err),
                    )
                    return

                probe_timeout = _probe_timeout_seconds(
                    self.req.provider, self.req.timeout, elapsed
                )
                probe_resp, probe_exc, aborted = yield from self._execute_probe(
                    requests, probe_timeout
                )
                if aborted:
                    return

                if probe_exc is not None:
                    self.last_err = str(probe_exc)
                else:
                    try:
                        body = probe_resp.text or ""
                        ready_ok, readiness_reason = probe_response_is_ready(
                            self.req.backend, probe_resp.status_code, body
                        )
                        if ready_ok:
                            yield from self._handle_ready_endpoint(requests)
                            return
                        yield from self._handle_non_ready_response(
                            probe_resp.status_code, body, readiness_reason
                        )
                    except Exception as exc:
                        self.last_err = str(exc)

                yield from self._maybe_check_system_scheduling(time.time())

                # Keep retry pauses short after a Modal cold-start timeout/connect
                # error so Modal sees continuous demand until a container binds.
                cold_modal_retry = (
                    self.req.provider == ComputeProvider.MODAL and probe_exc is not None
                )
                wait_seconds = (
                    _MODAL_COLD_RETRY_BACKOFF_SECONDS if cold_modal_retry else backoff
                )
                yield from self._sleep_between_probes(wait_seconds)
                if not cold_modal_retry:
                    backoff = min(self._MAX_BACKOFF_SECONDS, backoff * 1.5)
        finally:
            self._stop_log_tail("on exit")


class WarmupRunner:
    """Probe endpoint readiness while optionally tailing Modal logs."""

    def run_request(self, request: WarmupRequest) -> EventStream:
        """Execute a typed :class:`WarmupRequest`."""
        yield from _WarmupSession(request).run()

    def run(
        self,
        backend: BackendType,
        server_url: str,
        timeout: int = 1800,
        tail_logs: bool = True,
        app_name: str | None = None,
        served_model_name: str | None = None,
        provider: ComputeProvider = ComputeProvider.MODAL,
        api_key: str | None = None,
        pod_id: str | None = None,
        prime_backend: PrimeBackend | None = None,
        serving_requirements: ServingRequirements | None = None,
        placement_assessment: PlacementAssessment | None = None,
        runtime_id: str | None = None,
        vision: VisionCapabilities | None = None,
        phase_timer: StartupPhaseTimer | None = None,
        price_per_hour_usd: float | None = None,
    ) -> EventStream:
        """Probe endpoint readiness and optionally tail logs."""
        yield from self.run_request(
            WarmupRequest(
                backend=backend,
                server_url=server_url,
                timeout=timeout,
                tail_logs=tail_logs,
                app_name=app_name,
                served_model_name=served_model_name,
                provider=provider,
                api_key=api_key,
                pod_id=pod_id,
                prime_backend=prime_backend,
                serving_requirements=serving_requirements,
                placement_assessment=placement_assessment,
                runtime_id=runtime_id,
                vision=vision,
                phase_timer=phase_timer,
                price_per_hour_usd=price_per_hour_usd,
            )
        )


def _calibrate_endpoint(
    requests_module: Any,
    *,
    server_url: str,
    model: str,
    headers: dict[str, str],
    parallel_slots: int,
    price_per_hour_usd: float | None,
    budget_seconds: float,
    phase_timer: StartupPhaseTimer | None = None,
) -> tuple[PerformancePoint, ...]:
    """Measure a bounded performance curve without requiring aiperf.

    This is a short deployment calibration, not a sustained benchmark: one
    request per worker per scenario, distinct prompts to avoid cache reuse,
    and usage-counted tokens only. Points carry actual token counts and
    sample sizes so callers can tell a calibration curve from a benchmark.
    """

    from ..protocol.enums import CachePolicy, EvidenceLevel

    root = endpoint_root_url(server_url)
    endpoint = root + "/v1/completions"
    if phase_timer is not None:
        phase_timer.calibration_started()
    # One untimed warmup request primes multi-token prefill/decode kernels,
    # CUDA graphs, and cold mmap pages before the measurement clock starts.
    # Without it, a 50-85s cold first request on large MoE/Blackwell setups
    # consumes the entire calibration budget and misreads a 60 tok/s server
    # as ~5 tok/s.
    try:
        _streaming_calibration_request(
            requests_module,
            endpoint=endpoint,
            model=model,
            headers=headers,
            prompt_tokens=256,
            output_tokens=64,
            timeout=min(
                _CALIBRATION_WARMUP_TIMEOUT_SECONDS,
                max(60.0, budget_seconds),
            ),
            prompt_seed=-1,
        )
    except Exception:
        pass
    started = time.monotonic()
    scenarios = [(512, 1), (4096, 1)] + [
        (512, concurrency)
        for concurrency in (2, 4, 8)
        if concurrency <= max(1, parallel_slots)
    ]
    points: list[PerformancePoint] = []
    previous_aggregate = 0.0
    for scenario_index, (prompt_tokens, concurrency) in enumerate(scenarios):
        remaining = budget_seconds - (time.monotonic() - started)
        if remaining <= 1.0:
            break
        scenario_started = time.monotonic()
        results: list[dict[str, float]] = []
        errors = 0
        timeout = max(1.0, min(60.0, remaining))
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    _streaming_calibration_request,
                    requests_module,
                    endpoint=endpoint,
                    model=model,
                    headers=headers,
                    prompt_tokens=prompt_tokens,
                    output_tokens=128,
                    timeout=timeout,
                    prompt_seed=scenario_index * 16 + worker,
                )
                for worker in range(concurrency)
            ]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception:
                    errors += 1
        elapsed = max(0.001, time.monotonic() - scenario_started)
        successful = len(results)
        completion_tokens = sum(row["completion_tokens"] for row in results)
        actual_prompt = (
            int(sum(row["actual_prompt_tokens"] for row in results))
            if results
            else 0
        )
        actual_output = int(completion_tokens)
        output_rates = [row["output_tps"] for row in results]
        prompt_rates = [row["prompt_tps"] for row in results]
        ttfts = [row["ttft"] for row in results]
        aggregate = completion_tokens / elapsed if successful else 0.0
        error_rate = errors / max(1, concurrency)
        # One request per worker per scenario is a calibration curve, not a
        # benchmark: tail percentiles from 1-8 samples are not evidence.
        point = PerformancePoint(
            prompt_tokens=prompt_tokens,
            output_tokens=128,
            concurrency=concurrency,
            prompt_tokens_per_second=(
                sum(prompt_rates) / len(prompt_rates) if prompt_rates else None
            ),
            output_tokens_per_second=(
                sum(output_rates) / len(output_rates) if output_rates else None
            ),
            aggregate_output_tokens_per_second=aggregate or None,
            time_to_first_token_seconds=(
                sum(ttfts) / len(ttfts) if ttfts else None
            ),
            p95_latency_seconds=None,
            error_rate=error_rate,
            output_tokens_per_dollar=(
                aggregate * 3600.0 / price_per_hour_usd
                if price_per_hour_usd is not None and price_per_hour_usd > 0
                else None
            ),
            measured=True,
            actual_prompt_tokens=actual_prompt or None,
            actual_output_tokens=actual_output or None,
            sample_count=successful,
            completion_reason="calibration-curve",
            cache_policy=CachePolicy.UNCACHED,
            evidence=EvidenceLevel.OBSERVED,
        )
        points.append(point)
        if concurrency > 1:
            if error_rate > 0.05 or (
                previous_aggregate > 0 and aggregate <= previous_aggregate * 1.02
            ):
                break
            previous_aggregate = aggregate
        elif prompt_tokens == 512:
            previous_aggregate = aggregate
    return tuple(points)


def _streaming_calibration_request(
    requests_module: Any,
    *,
    endpoint: str,
    model: str,
    headers: dict[str, str],
    prompt_tokens: int,
    output_tokens: int,
    timeout: float,
    prompt_seed: int = 0,
) -> dict[str, float]:
    """Run one streaming completion and return timing primitives.

    Prompts are distinct per worker: repeating one identical prompt across
    concurrent requests lets prefix caching subsidize the measurement.
    Token counts come from usage when the runtime reports it; streaming
    chunks are transport events and never stand in for tokens. A response
    without a usable completion count is an error, not a one-token success.
    """

    prompt = _calibration_prompt(prompt_tokens, seed=prompt_seed)
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.monotonic()
    response = requests_module.post(
        endpoint,
        headers=headers,
        data=json.dumps(payload),
        timeout=timeout,
        stream=True,
    )
    status_code = int(getattr(response, "status_code", 0))
    if not 200 <= status_code < 300:
        raise RuntimeError(f"calibration returned HTTP {status_code}")

    first_token_at: float | None = None
    usage_completion_tokens = 0
    usage_prompt_tokens = 0
    iterator = getattr(response, "iter_lines", None)
    if callable(iterator):
        for raw_line in iterator(decode_unicode=True):
            line = str(raw_line or "").strip()
            if not line.startswith("data:"):
                continue
            raw_data = line[5:].strip()
            if not raw_data or raw_data == "[DONE]":
                continue
            try:
                chunk = json.loads(raw_data)
            except json.JSONDecodeError:
                continue
            usage = chunk.get("usage") if isinstance(chunk, dict) else None
            if isinstance(usage, dict):
                try:
                    usage_completion_tokens = max(
                        usage_completion_tokens,
                        int(usage.get("completion_tokens") or 0),
                    )
                    usage_prompt_tokens = max(
                        usage_prompt_tokens,
                        int(usage.get("prompt_tokens") or 0),
                    )
                except (TypeError, ValueError):
                    pass
            choices = chunk.get("choices") if isinstance(chunk, dict) else None
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0] if isinstance(choices[0], dict) else {}
            raw_delta = choice.get("delta")
            delta = raw_delta if isinstance(raw_delta, dict) else {}
            text = str(choice.get("text") or delta.get("content") or "")
            if text and first_token_at is None:
                first_token_at = time.monotonic()
    else:
        # Test doubles and non-streaming compatibility proxies may only expose
        # a final JSON body. Usage counts still decide; without them the
        # request is unmeasurable rather than a one-token success.
        body = response.json()
        usage = body.get("usage") if isinstance(body, dict) else None
        if isinstance(usage, dict):
            try:
                usage_completion_tokens = int(usage.get("completion_tokens") or 0)
                usage_prompt_tokens = int(usage.get("prompt_tokens") or 0)
            except (TypeError, ValueError):
                pass
        choices = body.get("choices") if isinstance(body, dict) else None
        if usage_completion_tokens > 0:
            first_token_at = time.monotonic()

    finished = time.monotonic()
    if usage_completion_tokens <= 0:
        raise RuntimeError(
            "calibration response carried no usable completion-token count"
        )
    first = first_token_at or finished
    completion_tokens = float(usage_completion_tokens)
    latency = max(0.001, finished - started)
    ttft = max(0.001, first - started)
    decode_seconds = max(0.001, finished - first)
    if first >= finished:
        decode_seconds = latency
    return {
        "completion_tokens": completion_tokens,
        "actual_prompt_tokens": float(usage_prompt_tokens or prompt_tokens),
        "latency": latency,
        "ttft": ttft,
        "prompt_tps": float(usage_prompt_tokens or prompt_tokens) / ttft,
        "output_tps": completion_tokens / decode_seconds,
    }


def _calibration_prompt(prompt_tokens: int, *, seed: int = 0) -> str:
    """Build a distinct uncached prompt of roughly the requested length.

    Identical prompts across workers share a cacheable prefix, so each worker
    gets its own filler sentence. Length is approximate -- the usage counts
    decide the recorded token numbers, never this filler arithmetic.
    """

    filler = (
        f"Calibration sentence {seed}: the deployment pipeline records "
        "throughput, latency, and error rate for every serving configuration. "
    )
    repetitions = max(1, (max(1, prompt_tokens) * 4) // max(1, len(filler)))
    return filler * repetitions


def _calibration_is_acceptable(
    performance: tuple[PerformancePoint, ...],
    objective: ServingObjective,
) -> tuple[bool, str]:
    """Apply the minimal stability and interactivity policy for Fast Deploy."""

    healthy = [point for point in performance if point.error_rate <= 0.05]
    if not healthy:
        return False, "the bounded calibration produced no stable requests"
    if any(
        (point.actual_output_tokens or 0) <= 0 or (point.sample_count or 0) <= 0
        for point in healthy
    ):
        return False, "the bounded calibration produced no counted completions"
    if objective in {ServingObjective.GENERAL_PURPOSE, ServingObjective.INTERACTIVE}:
        # Per-stream decode speed across all healthy scenarios: a server cannot
        # legitimately decode faster per stream at higher concurrency than at
        # c=1 unless earlier c=1 probes were still warming lazy mmap pages.
        single = max(
            (
                point.output_tokens_per_second or 0.0
                for point in healthy
            ),
            default=0.0,
        )
        if single < _GENERAL_PURPOSE_MIN_OUTPUT_TPS:
            return (
                False,
                f"single-request output was {single:.1f} tok/s; "
                f"Fast Deploy requires {_GENERAL_PURPOSE_MIN_OUTPUT_TPS:.1f} tok/s",
            )
    return True, "performance policy satisfied"


def _refetched_log_evidence(app_name: str) -> LlamacppRuntimeEvidence:
    """Re-read the app's retained logs and grade whatever offload report is there.

    The live tail is a stream, and a stream can lose a burst: llama.cpp emits
    hundreds of tensor warnings in a millisecond just before it reports the
    offload, and that whole window has been seen to vanish. Asking again costs
    one CLI call against logs the provider already stored.
    """

    from .runtime_evidence import LlamacppRuntimeEvidence, runtime_log_evidence

    if not app_name.strip():
        return LlamacppRuntimeEvidence()
    try:
        result = subprocess.run(
            ModalBackend._resolve_command(["modal", "app", "logs", app_name]),
            capture_output=True,
            text=True,
            timeout=20,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        text = result.stdout or ""
    except Exception:
        log_exception(f"Failed to re-read retained logs for {app_name}")
        return LlamacppRuntimeEvidence()
    return runtime_log_evidence(text)


def fetch_historical_logs(
    app_name: str,
    seen: set[str],
) -> EventStream:
    """Re-run ``modal app logs`` to capture persisted crash output."""
    try:
        unbuf_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        hist = subprocess.run(
            ModalBackend._resolve_command(["modal", "app", "logs", app_name]),
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
