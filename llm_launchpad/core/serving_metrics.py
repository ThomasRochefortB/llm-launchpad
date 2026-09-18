"""Live serving statistics read from a runtime's Prometheus endpoint.

vLLM and llama.cpp both publish counters and gauges at ``/metrics`` on the same
host the fleet already probes for ``/health``, so these numbers cost no extra
container wake-ups: one request answers both "is it up" and "what has it
served".

The runtimes only ever report traffic since their own process started, and a
container that scales to zero restarts that count. Turning those readings into
a lifetime total is this module's job.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any

from ..protocol.enums import BackendType
from ..protocol.models import EndpointInfo, ServingSnapshot, ServingStats
from .coerce import optional_float
from .config import SETTINGS_DIR
from .diagnostics import log_exception


USAGE_PATH = SETTINGS_DIR / "serving_usage.json"

# Two readings further apart than this describe different traffic, not a rate.
# The fleet refreshes every 20 seconds; a gap this wide means the TUI was
# closed or the container slept in between, and dividing across it would
# invent a throughput nobody observed.
MAX_RATE_WINDOW_SECONDS = 300.0

# A runtime that answers /metrics with "not implemented" keeps answering that
# way until it is redeployed, so the probe stops asking for a while rather
# than spending a request per refresh to learn the same thing.
UNSUPPORTED_RETRY_SECONDS = 900.0

METRICS_PATH = "/metrics"


def usage_key(row: EndpointInfo) -> str:
    """Identify an endpoint's usage record across refreshes and restarts.

    Keyed on the endpoint name, like the connection store, so redeploying the
    same endpoint keeps its running total rather than starting a second one.
    The provider prefix keeps two clouds from sharing a record when they
    happen to agree on a name.
    """
    provider = row.provider.value
    name = (row.name or "").strip()
    if name:
        return f"{provider}:{name}"
    return f"{provider}:id:{(row.app_id or '').strip()}"


def parse_prometheus_text(text: str) -> dict[str, float]:
    """Sum each metric's samples into one value per metric name.

    Labels are collapsed deliberately. The series that matter here are either
    unlabelled or split only by model name or finish reason, and in both cases
    the sum is the number worth showing.
    """
    totals: dict[str, float] = {}
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "{" in line:
            name = line.partition("{")[0].strip()
            # The label set always ends at the last brace on the line: what
            # follows is a number, so a brace inside a label value cannot be
            # mistaken for the terminator.
            tail = line.rsplit("}", 1)[-1]
        else:
            name, _, tail = line.partition(" ")
        fields = tail.split()
        if not name or not fields:
            continue
        try:
            value = float(fields[0])
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        totals[name] = totals.get(name, 0.0) + value
    return totals


# Metric names per backend. A tuple holds the alternatives a field has been
# published under, newest first, so a rename between runtime releases does not
# silently blank the column.
_VLLM_FIELDS: dict[str, tuple[str, ...]] = {
    "prompt_tokens": ("vllm:prompt_tokens_total",),
    "generation_tokens": ("vllm:generation_tokens_total",),
    "requests_running": ("vllm:num_requests_running",),
    "requests_waiting": ("vllm:num_requests_waiting",),
    "requests_finished": ("vllm:request_success_total",),
    "kv_cache_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
}

_LLAMACPP_FIELDS: dict[str, tuple[str, ...]] = {
    "prompt_tokens": ("llamacpp:prompt_tokens_total",),
    "generation_tokens": ("llamacpp:tokens_predicted_total",),
    "requests_running": ("llamacpp:requests_processing",),
    "requests_waiting": ("llamacpp:requests_deferred",),
    "reported_tokens_per_second": ("llamacpp:predicted_tokens_seconds",),
}

_FIELDS_BY_BACKEND: dict[BackendType, dict[str, tuple[str, ...]]] = {
    BackendType.VLLM: _VLLM_FIELDS,
    BackendType.LLAMACPP: _LLAMACPP_FIELDS,
}

# vLLM exports TTFT as a histogram; llama.cpp exports no equivalent.
_TTFT_HISTOGRAM: dict[BackendType, str] = {
    BackendType.VLLM: "vllm:time_to_first_token_seconds",
}

# vLLM serves its metrics through prometheus_client, whose process collector
# publishes this. When it is present a change in it dates the container
# exactly, which beats inferring a restart from a counter that went backwards.
_PROCESS_START_METRIC = "process_start_time_seconds"


def _first_present(samples: dict[str, float], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in samples:
            return samples[name]
    return None


def _histogram_mean(samples: dict[str, float], base: str) -> float | None:
    """Average a Prometheus histogram from its ``_sum`` and ``_count``."""
    total = samples.get(f"{base}_sum")
    count = samples.get(f"{base}_count")
    if total is None or not count or count <= 0:
        return None
    return total / count


def stats_from_metrics(
    text: str,
    backend: BackendType | None,
    *,
    captured_at: float | None = None,
) -> ServingStats:
    """Read one runtime's exposition text into a backend-neutral reading."""
    samples = parse_prometheus_text(text)
    fields = _FIELDS_BY_BACKEND.get(backend, {}) if backend is not None else {}
    values = {name: _first_present(samples, candidates) for name, candidates in fields.items()}
    histogram = _TTFT_HISTOGRAM.get(backend) if backend is not None else None
    return ServingStats(
        captured_at=time.monotonic() if captured_at is None else captured_at,
        avg_ttft_seconds=_histogram_mean(samples, histogram) if histogram else None,
        runtime_started_at=samples.get(_PROCESS_START_METRIC),
        **values,
    )


def _same_runtime(previous: ServingStats, current: ServingStats) -> bool | None:
    """Whether both readings came from the same runtime process.

    ``None`` means the runtime does not date its process, so the caller has to
    fall back to reading a restart off the counters themselves.
    """
    if previous.runtime_started_at is None or current.runtime_started_at is None:
        return None
    return previous.runtime_started_at == current.runtime_started_at


def _advance_total(
    banked: float,
    previous_raw: float | None,
    current_raw: float | None,
    *,
    restarted: bool,
) -> float:
    """Add the traffic observed since the last reading to a lifetime total.

    A restarted runtime counts from zero again, so its whole current value is
    traffic that has not been banked yet. Whatever the old process served
    after the final reading of it is lost -- an undercount, which is the safe
    direction for a number presented as a total.
    """
    if current_raw is None:
        return banked
    if restarted or previous_raw is None or current_raw < previous_raw:
        return banked + max(current_raw, 0.0)
    return banked + (current_raw - previous_raw)


def derive_tokens_per_second(
    previous: ServingStats,
    current: ServingStats,
    *,
    restarted: bool,
) -> float | None:
    """Generation throughput between two readings, or None if unmeasurable."""
    if restarted:
        return None
    elapsed = current.captured_at - previous.captured_at
    if elapsed <= 0 or elapsed > MAX_RATE_WINDOW_SECONDS:
        return None
    if previous.generation_tokens is None or current.generation_tokens is None:
        return None
    delta = current.generation_tokens - previous.generation_tokens
    if delta < 0:
        return None
    return delta / elapsed


class ServingMetricsTracker:
    """Turn successive ``/metrics`` readings into rates and lifetime totals.

    Throughput needs two readings close together in time, so the previous one
    is held in memory and discarded with the process. Totals are meant to
    outlive the TUI, so they are banked to disk as they accumulate.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._previous: dict[str, ServingStats] = {}
        self._unsupported: dict[str, float] = {}
        self._entries: dict[str, dict[str, Any]] | None = None

    def reset(self) -> None:
        """Drop every cached reading and re-read the store on next use.

        The tracker is a process-wide singleton, so the test suite needs a way
        to stop one case's readings from banking into the next one's totals.
        """
        with self._lock:
            self._previous.clear()
            self._unsupported.clear()
            self._entries = None

    # -- persistence -----------------------------------------------------

    def _usage_path(self) -> Path:
        """Resolve the store path at call time.

        Binding ``USAGE_PATH`` as an argument default would capture it at
        import and never see a test redirect the module attribute, which is
        how the suite stays out of the user's real ~/.llm_launchpad.
        """
        return USAGE_PATH if self._path is None else self._path

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._entries is not None:
            return self._entries
        entries: dict[str, dict[str, Any]] = {}
        try:
            payload = json.loads(self._usage_path().read_text(encoding="utf-8"))
        except FileNotFoundError:
            payload = None
        except Exception:
            log_exception(f"Could not read serving usage from {self._usage_path()}")
            payload = None
        stored = payload.get("entries") if isinstance(payload, dict) else None
        if isinstance(stored, dict):
            entries = {
                str(key): value
                for key, value in stored.items()
                if str(key).strip() and isinstance(value, dict)
            }
        self._entries = entries
        return entries

    def _save(self, entries: dict[str, dict[str, Any]]) -> None:
        path = self._usage_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"entries": entries}, indent=2), encoding="utf-8")
        except Exception:
            log_exception(f"Could not write serving usage to {path}")
            return
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    # -- probing state ---------------------------------------------------

    def metrics_supported(self, key: str) -> bool:
        """Whether ``/metrics`` is worth asking this endpoint for right now."""
        with self._lock:
            marked_at = self._unsupported.get(key)
        if marked_at is None:
            return True
        return (time.monotonic() - marked_at) >= UNSUPPORTED_RETRY_SECONDS

    def mark_unsupported(self, key: str) -> None:
        """Record that the endpoint serves no metrics, so stop asking a while."""
        with self._lock:
            self._unsupported[key] = time.monotonic()

    def mark_supported(self, key: str) -> None:
        """Clear an earlier "no metrics" verdict after a successful reading."""
        with self._lock:
            self._unsupported.pop(key, None)

    # -- accumulation ----------------------------------------------------

    def snapshot(self, key: str) -> ServingSnapshot | None:
        """Return the banked totals for an endpoint without a new reading."""
        with self._lock:
            entry = self._load().get(key)
        if not entry:
            return None
        return ServingSnapshot(
            total_prompt_tokens=optional_float(entry.get("total_prompt_tokens")) or 0.0,
            total_generation_tokens=optional_float(entry.get("total_generation_tokens")) or 0.0,
            observed_at=optional_float(entry.get("updated_at_epoch")),
        )

    def record(self, key: str, stats: ServingStats) -> ServingSnapshot:
        """Bank a reading and return the totals and rate it produces."""
        with self._lock:
            entries = self._load()
            entry = dict(entries.get(key) or {})
            previous = self._previous.get(key)
            if previous is None:
                previous = _stats_from_entry(entry)

            same_runtime = _same_runtime(previous, stats) if previous else None
            restarted = same_runtime is False

            banked_prompt = optional_float(entry.get("total_prompt_tokens")) or 0.0
            banked_generation = optional_float(entry.get("total_generation_tokens")) or 0.0
            total_prompt = _advance_total(
                banked_prompt,
                previous.prompt_tokens if previous else None,
                stats.prompt_tokens,
                restarted=restarted,
            )
            total_generation = _advance_total(
                banked_generation,
                previous.generation_tokens if previous else None,
                stats.generation_tokens,
                restarted=restarted,
            )

            rate = (
                derive_tokens_per_second(previous, stats, restarted=restarted)
                if previous is not None and previous.captured_at > 0
                else None
            )

            changed = (
                total_prompt != banked_prompt
                or total_generation != banked_generation
                or entry.get("last_prompt_tokens") != stats.prompt_tokens
                or entry.get("last_generation_tokens") != stats.generation_tokens
                or entry.get("runtime_started_at") != stats.runtime_started_at
                or entries.get(key) is None
            )
            entry.update(
                {
                    "total_prompt_tokens": total_prompt,
                    "total_generation_tokens": total_generation,
                    "last_prompt_tokens": stats.prompt_tokens,
                    "last_generation_tokens": stats.generation_tokens,
                    "runtime_started_at": stats.runtime_started_at,
                    "updated_at_epoch": time.time(),
                }
            )
            entries[key] = entry
            self._previous[key] = stats
            if changed:
                self._save(entries)

        return ServingSnapshot(
            stats=stats,
            total_prompt_tokens=total_prompt,
            total_generation_tokens=total_generation,
            tokens_per_second=rate,
            observed_at=time.time(),
        )


def _stats_from_entry(entry: dict[str, Any]) -> ServingStats | None:
    """Rebuild the last persisted reading so totals survive a TUI restart.

    ``captured_at`` stays zero: the monotonic clock it was taken against is
    gone with the old process, so this reading can bank tokens but must never
    be used as one end of a throughput window.
    """
    if not entry:
        return None
    last_prompt = optional_float(entry.get("last_prompt_tokens"))
    last_generation = optional_float(entry.get("last_generation_tokens"))
    if last_prompt is None and last_generation is None:
        return None
    return ServingStats(
        captured_at=0.0,
        prompt_tokens=last_prompt,
        generation_tokens=last_generation,
        runtime_started_at=optional_float(entry.get("runtime_started_at")),
    )


_DEFAULT_TRACKER = ServingMetricsTracker()


def default_tracker() -> ServingMetricsTracker:
    """The process-wide tracker the fleet probe accumulates into."""
    return _DEFAULT_TRACKER
