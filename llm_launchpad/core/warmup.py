"""Endpoint warmup and Modal log tailing helpers."""

from __future__ import annotations

from .shutdown import is_shutting_down, shutdown_event

import json
import math
import os
import queue
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from collections.abc import Generator
from urllib.parse import urlsplit

from ..protocol.enums import (
    BackendType,
    ComputeProvider,
    DeploymentState,
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
from .operation_events import fail_operation
from .diagnostics import log_exception
from .llamacpp_planner import attestation_now, save_runtime_attestation
from .naming import legacy_app_name
from .prime_backend import PrimeBackend

EventStream = Generator[BaseEvent, None, None]

_MODAL_GPU_WAIT_RE = re.compile(
    r"waiting to be scheduled on a (?P<worker>[A-Za-z0-9_:\-]+) worker",
    flags=re.IGNORECASE,
)
_MODAL_RELAX_RE = re.compile(r"Relaxing requirements \((?P<requirements>[^)]+)\)", flags=re.IGNORECASE)

_CALIBRATION_BUDGET_SECONDS = 90.0
_GENERAL_PURPOSE_MIN_OUTPUT_TPS = 8.0


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
        self._proc: subprocess.Popen[str] | None = None
        self._queue: queue.Queue[str | None] | None = None
        self._retry_at = 0.0

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
            self.seen_lines.add(line)
            yield LogEvent(line=line, operation=OperationType.WARMUP)

    def stop(self, reason: str) -> None:
        """Terminate the stream, logging rather than raising on failure."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.terminate()
        except Exception:
            log_exception(f"Failed to terminate Modal log tail {reason}")

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
    effective_context = serving_requirements.context_tokens
    if backend == BackendType.LLAMACPP:
        props_url = endpoint_root_url(server_url) + "/props"
        try:
            props_response = requests_module.get(
                props_url,
                headers=headers,
                timeout=15,
            )
            props_response.raise_for_status()
            effective_context = extract_effective_context(
                props_response.json()
            ) or 0
        except Exception as exc:
            last_err = f"runtime property verification failed: {exc}"
            yield from fail_operation(
                OperationType.WARMUP,
                f'Endpoint responded, but Launchpad could not verify its effective context: {exc}',
                recoverable=False,
                detail=last_err,
            )
            return None
    if placement_assessment is None:
        detail = "No placement assessment was supplied for runtime certification."
        yield from fail_operation(
            OperationType.WARMUP,
            f'Runtime attestation failed: {detail}',
            recoverable=False,
            detail=detail,
        )
        return None
    gpu_resident = bool(
        placement_assessment.gpu_resident
        and placement_assessment.tuning.gpu_layers.casefold() == "all"
    )
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
            f'Runtime attestation failed: {detail}',
            recoverable=False,
            detail=detail,
        )
        return None

    performance: tuple[PerformancePoint, ...] = ()
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
        parallel_slots=(
            placement_assessment.tuning.parallel_slots
            if placement_assessment is not None
            else 1
        ),
        price_per_hour_usd=serving_requirements.max_hourly_cost_usd,
        budget_seconds=_CALIBRATION_BUDGET_SECONDS,
    )
    accepted, calibration_detail = _calibration_is_acceptable(
        performance,
        serving_requirements.objective,
    )
    if not accepted:
        yield from fail_operation(
            OperationType.WARMUP,
            f'Endpoint performance certification failed: {calibration_detail}',
            detail=calibration_detail,
        )
        return None

    total_layers = max(
        1,
        placement_assessment.memory.total_layer_count or 0,
    )
    attestation = attestation_now(
        fingerprint=placement_assessment.fingerprint,
        requested_context_tokens=serving_requirements.context_tokens,
        effective_context_tokens=effective_context,
        gpu_layers=total_layers,
        total_layers=total_layers,
        gpu_resident=True,
        memory=placement_assessment.memory,
        performance=performance,
        runtime_id=runtime_id,
    )
    try:
        save_runtime_attestation(attestation)
    except Exception as exc:
        yield LogEvent(
            line=f"Warning: could not cache serving certificate: {exc}",
            operation=OperationType.WARMUP,
        )
    best_single = max(
        (
            point.output_tokens_per_second or 0.0
            for point in performance
            if point.concurrency == 1
        ),
        default=0.0,
    )
    best_aggregate = max(
        (
            point.aggregate_output_tokens_per_second or 0.0
            for point in performance
        ),
        default=0.0,
    )
    metric = ""
    if best_single > 0:
        metric = (
            f" · {best_single:.1f} tok/s single, "
            f"{best_aggregate:.1f} tok/s aggregate"
        )
    yield LogEvent(
        line=(
            f"Full {effective_context:,}-token context verified on GPU"
            f"{metric}."
        ),
        operation=OperationType.WARMUP,
        is_milestone=True,
    )
    return attestation


class WarmupRunner:
    """Probe endpoint readiness while optionally tailing Modal logs."""

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
        url_error = endpoint_url_error(probe_url)
        if url_error:
            yield from fail_operation(
                OperationType.WARMUP,
                f'Invalid endpoint URL: {url_error}',
                exit_code=2,
                recoverable=False,
                detail="",
            )
            return

        target_app_name = app_name or legacy_app_name(backend)
        log_tail = _ModalLogTail(target_app_name)
        tail_modal_logs = tail_logs and provider == ComputeProvider.MODAL
        if tail_modal_logs:
            yield from log_tail.attach()

        try:
            import requests
        except ImportError:
            yield from fail_operation(
                OperationType.WARMUP,
                "'requests' is required. Install with: pip install requests",
                recoverable=False,
                detail="",
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
        last_err: str | None = None
        last_scheduling_hint: str | None = None
        while True:
            # Exit promptly if the app is shutting down.
            if is_shutting_down():
                log_tail.stop("on shutdown")
                return

            if tail_modal_logs and log_tail.may_attach():
                # Historical-fetch fallback: the live stream may miss the
                # final output of a crashing container, while Modal's persisted
                # logs still have it.
                yield from fetch_historical_logs(target_app_name, log_tail.seen_lines)
                yield from log_tail.attach()

            # Drain log lines delivered by the reader thread.
            yield from log_tail.drain()

            elapsed = time.time() - start
            if elapsed > timeout:
                log_tail.stop("on timeout")
                yield from fail_operation(
                    OperationType.WARMUP,
                    f'Timed out after {timeout}s. Last error: {last_err}',
                    detail="",
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
                    log_tail.stop("after readiness")
                    attestation = None
                    if serving_requirements is not None:
                        attestation = yield from _certify_serving_requirements(
                            requests,
                            backend=backend,
                            server_url=server_url,
                            headers=headers,
                            probe_model=llama_probe_model,
                            serving_requirements=serving_requirements,
                            placement_assessment=placement_assessment,
                            runtime_id=runtime_id,
                        )
                        if attestation is None:
                            return

                    yield LogEvent(line="Server is ready!")
                    curl_cmd = ModalBackend.test_curl_command(
                        backend,
                        server_url,
                        served_model_name=served_model_name,
                        api_key=api_key,
                    )
                    if api_key:
                        curl_cmd = curl_cmd.replace(api_key, "$LLM_LAUNCHPAD_API_KEY")
                    yield LogEvent(line=f"Test command:\n{curl_cmd}")
                    yield StateChangeEvent(
                        current=DeploymentState.HEALTHY, operation=OperationType.WARMUP
                    )
                    yield OperationCompleteEvent(
                        operation=OperationType.WARMUP,
                        success=True,
                        data={"url": server_url, "attestation": attestation},
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
                            if line not in log_tail.seen_lines:
                                log_tail.seen_lines.add(line)
                                yield LogEvent(line=line, operation=OperationType.WARMUP)
                    except Exception:
                        log_exception("Failed to fetch Prime pod logs during warmup")
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
                yield from log_tail.drain()
            backoff = min(max_backoff, backoff * 1.5)


def _calibrate_endpoint(
    requests_module: Any,
    *,
    server_url: str,
    model: str,
    headers: dict[str, str],
    parallel_slots: int,
    price_per_hour_usd: float | None,
    budget_seconds: float,
) -> tuple[PerformancePoint, ...]:
    """Measure a bounded performance curve without requiring aiperf."""

    root = endpoint_root_url(server_url)
    endpoint = root + "/v1/completions"
    started = time.monotonic()
    scenarios = [(512, 1), (4096, 1)] + [
        (512, concurrency)
        for concurrency in (2, 4, 8)
        if concurrency <= max(1, parallel_slots)
    ]
    points: list[PerformancePoint] = []
    previous_aggregate = 0.0
    for prompt_tokens, concurrency in scenarios:
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
                )
                for _ in range(concurrency)
            ]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception:
                    errors += 1
        elapsed = max(0.001, time.monotonic() - scenario_started)
        successful = len(results)
        completion_tokens = sum(row["completion_tokens"] for row in results)
        latencies = sorted(row["latency"] for row in results)
        output_rates = [row["output_tps"] for row in results]
        prompt_rates = [row["prompt_tps"] for row in results]
        ttfts = [row["ttft"] for row in results]
        aggregate = completion_tokens / elapsed if successful else 0.0
        error_rate = errors / max(1, concurrency)
        p95_index = max(0, math.ceil(len(latencies) * 0.95) - 1) if latencies else 0
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
            p95_latency_seconds=(latencies[p95_index] if latencies else None),
            error_rate=error_rate,
            output_tokens_per_dollar=(
                aggregate * 3600.0 / price_per_hour_usd
                if price_per_hour_usd is not None and price_per_hour_usd > 0
                else None
            ),
            measured=True,
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
) -> dict[str, float]:
    """Run one streaming completion and return timing primitives."""

    prompt = " calibration" * max(1, prompt_tokens)
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
    observed_chunks = 0
    usage_completion_tokens = 0
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
                except (TypeError, ValueError):
                    pass
            choices = chunk.get("choices") if isinstance(chunk, dict) else None
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0] if isinstance(choices[0], dict) else {}
            raw_delta = choice.get("delta")
            delta = raw_delta if isinstance(raw_delta, dict) else {}
            text = str(choice.get("text") or delta.get("content") or "")
            if text:
                if first_token_at is None:
                    first_token_at = time.monotonic()
                observed_chunks += 1
    else:
        # Test doubles and non-streaming compatibility proxies may only expose
        # a final JSON body. This still supplies conservative end-to-end rates.
        body = response.json()
        first_token_at = time.monotonic()
        usage = body.get("usage") if isinstance(body, dict) else None
        if isinstance(usage, dict):
            usage_completion_tokens = int(usage.get("completion_tokens") or 0)
        observed_chunks = usage_completion_tokens

    finished = time.monotonic()
    first = first_token_at or finished
    completion_tokens = float(max(1, usage_completion_tokens or observed_chunks))
    latency = max(0.001, finished - started)
    ttft = max(0.001, first - started)
    decode_seconds = max(0.001, finished - first)
    if first >= finished:
        decode_seconds = latency
    return {
        "completion_tokens": completion_tokens,
        "latency": latency,
        "ttft": ttft,
        "prompt_tps": float(prompt_tokens) / ttft,
        "output_tps": completion_tokens / decode_seconds,
    }


def _calibration_is_acceptable(
    performance: tuple[PerformancePoint, ...],
    objective: ServingObjective,
) -> tuple[bool, str]:
    """Apply the minimal stability and interactivity policy for Fast Deploy."""

    healthy = [point for point in performance if point.error_rate <= 0.05]
    if not healthy:
        return False, "the bounded calibration produced no stable requests"
    if objective in {ServingObjective.GENERAL_PURPOSE, ServingObjective.INTERACTIVE}:
        single = max(
            (
                point.output_tokens_per_second or 0.0
                for point in healthy
                if point.concurrency == 1
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
