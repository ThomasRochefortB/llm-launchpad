"""Shared fleet annotation: outage wording, and what each endpoint is serving.

The fleet screens each render their own layout, but they must not describe the
same outage in two different ways: an unreachable provider is reported with
what it last returned and how old that is, never as a shorter list. Live
traffic is shared for the same reason -- the home panel and Manage read the
same endpoints, and they are given one way to ask.

Background fleet refreshes are passive for scale-to-zero providers. A request
to a Modal ``/metrics`` or ``/health`` endpoint is served by the GPU container
itself, so polling it from a dashboard would keep that container warm and
defeat the idle timeout the cost story depends on. Passive refreshes therefore
read provider metadata plus locally banked observations only; contacting the
runtime requires an explicit, user-requested probe (status check, warmup,
benchmark, inference).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from rich.markup import escape

from ..core.serving_metrics import (
    METRICS_PATH,
    default_tracker,
    stats_from_metrics,
    usage_key,
)
from ..protocol.enums import BackendType, ComputeProvider
from ..protocol.models import EndpointInfo, FleetDiscovery, ServingSnapshot
from .connection import resolve_openai_base_url
from .format import clip, format_age

_ERROR_WIDTH = 72

# A reading has to come back inside the refresh cadence to be worth having.
_METRICS_TIMEOUT_SECONDS = 2.5

# Only an endpoint that is actually serving can answer for its own traffic.
# Probing one that is still starting would record a "this runtime has no
# metrics" verdict against a runtime that has not yet had the chance to
# publish any, and that verdict outlives the startup it was taken during.
_SERVING_STATES = frozenset({"active", "deployed", "running"})

# Statuses that mean a running endpoint will not serve metrics however often
# it is asked. Everything else -- 5xx, a dropped connection -- is a passing
# condition and leaves the endpoint in the queue for the next refresh.
_NO_METRICS_STATUSES = frozenset({401, 403, 404, 501})

_METRICS_BACKENDS = frozenset({BackendType.VLLM, BackendType.LLAMACPP})

# Providers whose inference endpoint is served by a container that scales to
# zero. Any background HTTP request to such an endpoint can wake it or extend
# its idle timeout, so passive refreshes must not contact the runtime at all.
SCALE_TO_ZERO_PROVIDERS = frozenset({ComputeProvider.MODAL})

# Shown wherever a live gauge would have appeared for a passively monitored
# endpoint, so a banked total is never mistaken for a current measurement.
PASSIVE_METRICS_NOTE = "Live metrics paused to allow scale-to-zero."


def _state_bucket(state: str) -> str:
    """Deployment state bucket: running vs starting vs stopped vs failed."""
    normalized = (state or "").strip().lower()
    if normalized in {"running", "deployed"}:
        return "running"
    if normalized in {"deploying", "starting", "initializing", "building", "ephemeral"}:
        return "starting"
    if normalized in {"queued", "pending"}:
        return "queued"
    if normalized in {"failed", "error", "crashed"}:
        return "failed"
    if normalized in {"stopped", "stopping"}:
        return "stopped"
    return "other"


def _health_bucket(row: EndpointInfo) -> str:
    """Observed runtime health: healthy / starting / failed / not checked.

    A deployed scale-to-zero app is not evidence its container answered, so
    without an explicit observation it is "not checked", never "healthy".
    Other providers keep live probing, so their provider state still implies
    health until a probe says otherwise.
    """
    status = (row.runtime_status or "").strip().lower()
    if status in {"healthy", "in_progress", "error", "unchecked"}:
        return status
    if row.provider in SCALE_TO_ZERO_PROVIDERS and _state_bucket(row.state) == "running":
        try:
            from ...core.runtime_health import get_health as _get_health

            stored = _get_health(row)
        except Exception:
            stored = None
        if stored is not None:
            return stored.status
        return "unchecked"
    return _health_bucket_from_state(row.state)


def _health_bucket_from_state(state: str) -> str:
    bucket = _state_bucket(state)
    if bucket == "running":
        return "healthy"
    if bucket in {"starting", "queued"}:
        return "in_progress"
    if bucket == "failed":
        return "error"
    return "in_progress"


def style_health_bucket(bucket: str) -> str:
    """One shared health label for Home and Manage.

    Deployment state ("Running") and observed health ("Healthy") are separate
    clauses everywhere, so a Modal row that exists but was never probed reads
    "Deployment: Running · Health: Not checked" on both screens.
    """
    normalized = (bucket or "").strip().lower()
    if normalized == "healthy":
        return "[$success]Healthy[/$success]"
    if normalized == "in_progress":
        return "[$warning]Starting[/$warning]"
    if normalized == "error":
        return "[$error]Failed[/$error]"
    if normalized == "unchecked":
        return "[dim]Not checked[/dim]"
    return f"[dim]{escape(normalized or 'unknown')}[/dim]"


def deployment_and_health_line(row: EndpointInfo) -> str:
    """Compact `Deployment: X · Health: Y` summary shared by fleet screens."""
    bucket = _state_bucket(row.state)
    state_label = {
        "running": "Running",
        "starting": "Starting",
        "queued": "Queued",
        "failed": "Failed",
        "stopped": "Stopped",
    }.get(bucket, (row.state or "unknown").strip().capitalize() or "Unknown")
    return (
        f"[dim]Deployment:[/dim] {escape(state_label)} · "
        f"[dim]Health:[/dim] {style_health_bucket(_health_bucket(row))}"
    )


def fleet_summary_line(rows: list[EndpointInfo]) -> str:
    """One-line fleet state for the compact home header.

    Counts deployments (running/starting/failed), never health observations:
    "0 healthy" on an unchecked fleet reads as a failure count.
    """
    from collections import Counter

    counts = Counter(_state_bucket(row.state) for row in rows)
    running = counts.get("running", 0)
    starting = counts.get("starting", 0) + counts.get("queued", 0)
    failed = counts.get("failed", 0)
    total = len(rows)
    noun = "endpoint" if total == 1 else "endpoints"
    parts = [f"[bold]{total} {noun}[/bold]"]
    if running:
        parts.append(f"[$success]{running} running[/$success]")
    if starting:
        parts.append(f"[$warning]{starting} starting[/$warning]")
    if failed:
        parts.append(f"[$error]{failed} failed[/$error]")
    return " · ".join(parts)


def provider_outage_lines(
    discovery: FleetDiscovery | None,
    *,
    now: float | None = None,
) -> list[str]:
    """Return one styled line per provider that could not be reached."""
    if discovery is None:
        return []
    moment = time.time() if now is None else now
    lines: list[str] = []
    for listing in discovery.unavailable:
        name = listing.provider.display_name
        reason = escape(clip(listing.error or "unknown error", _ERROR_WIDTH))
        age = listing.age_seconds(moment)
        if listing.rows and age is not None:
            count = len(listing.rows)
            noun = "endpoint" if count == 1 else "endpoints"
            detail = f"showing {count} {noun} from {format_age(age)}"
        else:
            detail = "its deployments are not listed"
        lines.append(f"[$warning]{escape(name)} unavailable:[/$warning] {reason} [dim]({detail})[/dim]")
    return lines


def retained_providers(discovery: FleetDiscovery | None) -> set[str]:
    """Provider values whose rows came from a failed pass and may be stale."""
    if discovery is None:
        return set()
    return {
        listing.provider.value for listing in discovery.unavailable if listing.is_retained
    }


def is_passively_monitored(row: EndpointInfo) -> bool:
    """Whether background refreshes must not contact this endpoint's runtime.

    Modal serves fleet HTTP requests from the GPU container itself, so even a
    health check can wake it. Prime and Vast bill continuously, so their
    existing live probing does not defeat any idle timeout.
    """
    return row.provider in SCALE_TO_ZERO_PROVIDERS


def can_report_traffic(row: EndpointInfo, *, explicit: bool = False) -> bool:
    """Whether this endpoint may be asked for its traffic right now.

    ``explicit`` marks a user-requested probe (status check, warmup,
    benchmark) that is allowed to wake a scaled-to-zero container.
    Background refreshes pass ``explicit=False`` and never contact Modal
    runtimes, even as a fallback.
    """
    if not explicit and is_passively_monitored(row):
        return False
    return (
        row.backend in _METRICS_BACKENDS
        and (row.state or "").strip().lower() in _SERVING_STATES
    )


def is_passive_traffic_row(row: EndpointInfo) -> bool:
    """Whether this row holds banked totals because live reads are paused.

    True when the endpoint is in a serving state on a scale-to-zero provider:
    it could answer for its traffic, but background refreshes deliberately do
    not ask so the container may go idle.
    """
    return (
        is_passively_monitored(row)
        and row.backend in _METRICS_BACKENDS
        and (row.state or "").strip().lower() in _SERVING_STATES
    )


def fetch_serving_snapshot(
    row: EndpointInfo,
    username: str = "",
    *,
    explicit: bool = False,
) -> ServingSnapshot | None:
    """Read one endpoint's own counters, or None when it publishes none.

    A 2xx from /metrics is the same liveness evidence /health gives, so a
    caller that needs both spends one request rather than two. A runtime that
    will not serve metrics -- llama.cpp started without ``--metrics`` answers
    501 -- is remembered, so the next refresh does not ask again.

    With ``explicit=False`` (the default) a Modal endpoint is never contacted:
    the call returns None before any HTTP request, including the health
    fallback, so blocking /metrics cannot accidentally trigger /health.
    """
    if not can_report_traffic(row, explicit=explicit):
        return None
    base_url, _was_derived = resolve_openai_base_url(row, username=username)
    if not base_url:
        return None
    try:
        import requests  # type: ignore
    except ImportError:
        return None

    tracker = default_tracker()
    key = usage_key(row)
    if not tracker.metrics_supported(key):
        return None

    base_root = base_url.rstrip("/")
    host_root = (base_root[:-3] if base_root.endswith("/v1") else base_root).rstrip("/")
    headers = (
        {"Authorization": f"Bearer {row.endpoint_api_key}"} if row.endpoint_api_key else None
    )
    try:
        response = requests.get(
            host_root + METRICS_PATH,
            headers=headers,
            timeout=_METRICS_TIMEOUT_SECONDS,
        )
    except Exception:
        # A transport failure says nothing about whether metrics are served.
        return None

    status = getattr(response, "status_code", 0)
    if not 200 <= status < 300:
        if status in _NO_METRICS_STATUSES:
            tracker.mark_unsupported(key)
        return None

    stats = stats_from_metrics(getattr(response, "text", "") or "", row.backend)
    if not stats.has_readings:
        # A 200 carrying something other than this runtime's metrics: usually
        # a proxy's own page standing in for an endpoint that is not up yet.
        tracker.mark_unsupported(key)
        return None
    tracker.mark_supported(key)
    return tracker.record(key, stats)


def attach_cached_serving_stats(rows: list[EndpointInfo]) -> None:
    """Attach banked totals without touching the network.

    Passive refreshes call this instead of probing: a stopped endpoint still
    reports what it served while it was up, and a scaled-to-zero Modal
    endpoint keeps its last totals without being woken to re-read them.
    Skipping the probe never marks metrics unsupported.
    """
    tracker = default_tracker()
    for row in rows:
        row.serving = tracker.snapshot(usage_key(row))


def annotate_serving_stats(
    rows: list[EndpointInfo],
    username: str = "",
    *,
    explicit: bool = False,
) -> None:
    """Attach traffic to every row, probing only where allowed.

    Every row gets its banked totals. With ``explicit=False`` Modal rows stop
    there -- no /metrics request, and therefore no /health fallback either.
    Prime and Vast rows are probed as before. Pass ``explicit=True`` only for
    a user-requested check that may wake a container.
    """
    attach_cached_serving_stats(rows)

    candidates = [row for row in rows if can_report_traffic(row, explicit=explicit)]
    if not candidates:
        return

    with ThreadPoolExecutor(max_workers=min(4, len(candidates))) as executor:
        futures = {
            executor.submit(fetch_serving_snapshot, row, username, explicit=explicit): row
            for row in candidates
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                snapshot = future.result()
            except Exception:
                snapshot = None
            if snapshot is not None:
                row.serving = snapshot
