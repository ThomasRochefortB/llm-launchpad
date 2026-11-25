"""Endpoint-first management screen and focused action forms."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from rich.markup import escape
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.message import Message
from textual.timer import Timer
from textual.widgets import Button, DataTable, Input, OptionList, Static, TabbedContent, TabPane
from textual.widgets.option_list import Option

import time

from ...core.benchmark import BENCHMARK_UNAVAILABLE_REASON, parse_concurrency_values
from ...core.deploy_log_summary import SUMMARY_SPINNER_FRAMES
from ...core.endpoint_runtime import attach_endpoint_runtime
from ...core.vision_probe import image_test_command
from ...protocol.enums import ComputeProvider, VisionVerification
from ...protocol.models import EndpointInfo, ServingSnapshot, VisionCapabilities
from ...core.deployment_states import is_terminal_deployment_state
from ..connection import endpoint_connection_payload
from ..format import (
    clip,
    format_age,
    format_cost,
    format_duration,
    format_money,
    format_token_count,
    format_token_rate,
)
from ..fleet_status import (
    PASSIVE_METRICS_NOTE,
    annotate_serving_stats,
    attach_cached_serving_stats,
    fetch_serving_snapshot_detailed,
    fleet_summary_line,
    has_explicit_live_metrics,
    is_passive_traffic_row,
    is_passively_monitored,
    provider_outage_lines,
    retained_providers,
)
from ..navigation import move_focus_across_widgets
from ..responsive import ViewportProfile, WidthMode
from ..widgets.adaptive_table import AdaptiveColumn, AdaptiveDataTable
from ..widgets.input_form import FormField
from ..workers import EndpointsFailed, EndpointsLoaded, ServingStatsReady
from ..widgets.fitted_footer import FittedFooter
from .copy_enabled import CopyEnabledScreen
from ..compat import highlighted_option
from ..visual import labelled_rows_markup, screen_title
from .operations import DeploymentJobsPanel


_STOPPABLE_STATES = frozenset(
    {
        "active",
        "building",
        "deployed",
        "deploying",
        "initializing",
        "pending",
        "queued",
        "running",
        "starting",
    }
)
_READY_STATES = frozenset({"active", "deployed", "running"})

_STATE_LABELS = {
    "active": "Running",
    "running": "Running",
    "deployed": "Ready",
    "ephemeral": "Temporary",
    "building": "Building",
    "deploying": "Deploying",
    "initializing": "Starting",
    "pending": "Pending",
    "queued": "Queued",
    "starting": "Starting",
    "stopped": "Stopped",
    "stopping": "Stopping",
    "terminated": "Stopped",
    "archived": "Archived",
    "failed": "Failed",
}


def _normalized_state(state: str) -> str:
    return (state or "").strip().lower()


def _state_label(state: str) -> str:
    """Convert provider lifecycle names into concise user-facing labels."""
    normalized = _normalized_state(state)
    return _STATE_LABELS.get(normalized, normalized.replace("_", " ").title() or "Unknown")


def _is_stoppable_state(state: str) -> bool:
    """Return whether a deployment can reasonably accept a stop request."""
    return _normalized_state(state) in _STOPPABLE_STATES


def _endpoint_key(row: EndpointInfo) -> str:
    """Build a provider-aware identity for table selection and action routing."""
    provider = row.provider.value
    if row.app_id:
        return f"{provider}:id:{row.app_id}"
    backend = row.backend.value if row.backend is not None else "unknown"
    return f"{provider}:name:{backend}:{row.name}:{row.instance_name or ''}"


def _is_passive_without_live(row: EndpointInfo) -> bool:
    """Whether live gauges should stay hidden because no explicit fetch ran."""
    return is_passive_traffic_row(row) and not has_explicit_live_metrics(row)


def _endpoint_throughput(row: EndpointInfo) -> str:
    """Generation throughput measured between the last two fleet refreshes.

    Passively monitored Modal rows without an explicit live fetch always show
    "-": their banked totals have no current rate, and showing a stale one
    would mistake history for a live measurement.
    """
    if _is_passive_without_live(row):
        return "-"
    serving = row.serving
    if serving is None or serving.tokens_per_second is None:
        return "-"
    return format_token_rate(serving.tokens_per_second)


def _endpoint_tokens_served(row: EndpointInfo) -> str:
    """Tokens this endpoint has served, counting across container restarts."""
    serving = row.serving
    if serving is None or serving.total_tokens <= 0:
        return "-"
    return format_token_count(serving.total_tokens)


def _serving_live_parts(serving: ServingSnapshot) -> list[str]:
    """Describe what the runtime is doing right now, skipping what it omits."""
    stats = serving.stats
    parts: list[str] = []
    if serving.tokens_per_second is not None:
        parts.append(format_token_rate(serving.tokens_per_second))
    elif stats.reported_tokens_per_second:
        # llama.cpp's own gauge holds the last request's speed rather than a
        # windowed rate, so it is labelled as such instead of passed off as
        # the measured throughput the column shows.
        parts.append(f"{format_token_rate(stats.reported_tokens_per_second)} last request")
    if stats.requests_running is not None:
        parts.append(f"{stats.requests_running:,.0f} running")
    if stats.requests_waiting:
        parts.append(f"{stats.requests_waiting:,.0f} queued")
    if stats.kv_cache_usage is not None:
        parts.append(f"KV {stats.kv_cache_usage * 100:,.0f}%")
    if stats.avg_ttft_seconds is not None:
        parts.append(f"TTFT {stats.avg_ttft_seconds:,.2f}s")
    return parts


def _serving_observed_age_label(row: EndpointInfo, *, now: float | None = None) -> str | None:
    """Age of the banked traffic total, or None when never collected."""
    serving = row.serving
    if serving is None or serving.total_tokens <= 0:
        return None
    observed_at = serving.observed_at
    if observed_at is None:
        return "cached"
    moment = time.time() if now is None else now
    age = max(0.0, moment - observed_at)
    if age < 60:
        return "just now"
    return format_age(age)


def _endpoint_run_time(row: EndpointInfo, *, now: float | None = None) -> str:
    """Current-run uptime for the table; "-" when the run start is unknown.

    Deployment age is never a substitute: a Modal app can exist while its GPU
    is asleep, so an unverified start reads as unknown rather than uptime.
    """
    started_at = row.serving.stats.runtime_started_at if row.serving is not None else None
    if started_at is None or started_at <= 0:
        started_at = row.run_started_at_epoch
    if started_at is None or started_at <= 0:
        return "-"
    moment = time.time() if now is None else now
    if moment < started_at:
        return "-"
    return format_duration(moment - started_at)


def _endpoint_cost(row: EndpointInfo) -> str:
    """Accumulated compute cost for the table; "-" when unknown."""
    if row.cumulative_cost_usd is None:
        return "-"
    rendered = format_cost(row.cumulative_cost_usd)
    return f"~{rendered}" if row.cost_estimated else rendered


def _runtime_detail_lines(row: EndpointInfo, *, now: float | None = None) -> str:
    """Render runclock and cumulative cost shown under the selected endpoint."""
    moment = time.time() if now is None else now
    lines: list[str] = []
    live_start = row.serving.stats.runtime_started_at if row.serving is not None else None
    started_at = live_start if live_start else row.run_started_at_epoch
    if started_at is not None and started_at > 0 and moment >= started_at:
        lines.append(f"\n[dim]Run time:[/dim] {format_duration(moment - started_at)} (current run)")
    elif is_terminal_deployment_state(row.state):
        lines.append("\n[dim]Run time:[/dim] - (not running)")
    else:
        # Say only that it is unknown; why (refreshes never wake a Modal
        # container) is the kind of paragraph that made this pane a wall.
        lines.append("\n[dim]Run time:[/dim] unknown")
    if row.cumulative_cost_usd is not None:
        cost_text = f"~{format_cost(row.cumulative_cost_usd)}" if row.cost_estimated else format_cost(row.cumulative_cost_usd)
        qualifier = "est." if row.cost_estimated else "tracked"
        detail = f"\n[dim]Compute cost:[/dim] {cost_text} ({qualifier} endpoint total)"
        if row.hourly_cost_usd is not None:
            detail += f" [dim]at {format_money(row.hourly_cost_usd)}/h[/dim]"
        if row.cost_tracked_since_epoch is not None:
            age = max(0.0, moment - row.cost_tracked_since_epoch)
            detail += f" [dim]since {format_age(age)}[/dim]"
        lines.append(detail)
    elif row.hourly_cost_usd is not None:
        lines.append(
            f"\n[dim]Compute cost:[/dim] tracking from {format_money(row.hourly_cost_usd)}/h"
        )
    else:
        lines.append("\n[dim]Compute cost:[/dim] unknown (hourly rate unavailable)")
    return "".join(lines)


def _serving_detail_lines(row: EndpointInfo, *, now: float | None = None) -> str:
    """Render traffic plus runclock/cost shown under the selected endpoint.

    Traffic stays passive for Modal: banked totals carry their observation age
    so a cached number is never mistaken for a fresh measurement. Run time and
    cost follow on every row so narrow layouts see them even when the table
    cannot fit the columns.
    """
    moment = time.time() if now is None else now
    lines: list[str] = []
    serving = row.serving
    if serving is None:
        lines.append("\n[dim]Traffic:[/dim] not collected yet")
    else:
        if serving.total_tokens > 0:
            age_label = _serving_observed_age_label(row, now=moment)
            age_suffix = f" [dim](observed {age_label})[/dim]" if age_label else ""
            lines.append(
                f"\n[dim]Traffic:[/dim] {format_token_count(serving.total_tokens)} tokens served"
                f" [dim]({format_token_count(serving.total_prompt_tokens)} in /"
                f" {format_token_count(serving.total_generation_tokens)} out)[/dim]"
                f"{age_suffix}"
            )
        # A passively monitored Modal row without an explicit live fetch shows
        # history only: live gauges would imply a probe that background
        # refreshes deliberately skip.
        if _is_passive_without_live(row):
            if serving.total_tokens > 0:
                lines.append(f"\n[dim]{PASSIVE_METRICS_NOTE}[/dim]")
            lines.append(_runtime_detail_lines(row, now=moment))
            if row.live_metrics_error:
                lines.append(f"\n[$warning]Last live fetch failed: {escape(row.live_metrics_error)}[/$warning]")
            elif row.live_metrics_checked_at is not None:
                lines.append(
                    f"\n[dim]Last live fetch: {format_age(max(0.0, moment - row.live_metrics_checked_at))}[/dim]"
                )
            return "".join(lines)
        live = _serving_live_parts(serving)
        if live:
            lines.append(f"\n[dim]Now:[/dim] {' · '.join(live)}")
        elif serving.total_tokens <= 0 and not _serving_live_parts(serving):
            # No traffic and no live gauges: traffic contributes nothing, but
            # the caller still gets runclock/cost below.
            pass
    lines.append(_runtime_detail_lines(row, now=moment))
    if row.live_metrics_error:
        lines.append(f"\n[$warning]Last live fetch failed: {escape(row.live_metrics_error)}[/$warning]")
    elif row.live_metrics_checked_at is not None:
        lines.append(
            f"\n[dim]Last live fetch: {format_age(max(0.0, moment - row.live_metrics_checked_at))}[/dim]"
        )
    return "".join(lines)


def _available_actions(row: EndpointInfo) -> frozenset[str]:
    """Return management actions supported by the endpoint's current state."""
    if row.backend is None:
        return frozenset()

    actions = {"logs"}
    state = _normalized_state(row.state)
    if state in _READY_STATES or bool((row.web_url or "").strip()):
        actions.update(("status", "benchmark", "connection"))
        if row.backend.value in {"vllm", "llamacpp"}:
            actions.add("live-metrics")
    if _is_stoppable_state(state) or row.provider == ComputeProvider.VAST:
        actions.add("stop")
    return frozenset(actions)


def _live_metrics_label(row: EndpointInfo) -> str:
    """Label the explicit metrics fetch with its Modal wake-up cost."""
    if row.provider == ComputeProvider.MODAL:
        return "  Fetch live metrics — may start GPU"
    return "  Fetch live metrics"


def _endpoint_model(row: EndpointInfo) -> str:
    return (row.served_model_name or row.model_name or "-").strip() or "-"


def _endpoint_name(row: EndpointInfo) -> str:
    return (row.instance_name or row.name or "unnamed").strip()


def _endpoint_backend(row: EndpointInfo) -> str:
    return row.backend.value if row.backend is not None else "unknown"


def _endpoint_host(row: EndpointInfo) -> str:
    """Return the compact provider/runtime label shown beside an endpoint."""
    provider = row.provider.value.title()
    backend = {
        "llamacpp": "llama.cpp",
        "vllm": "vLLM",
    }.get(_endpoint_backend(row), _endpoint_backend(row))
    return f"{provider}/{backend}"


def _endpoint_compact_label(row: EndpointInfo) -> str:
    return f"{_endpoint_name(row)} [{_endpoint_host(row)}]"


def _endpoint_summary(row: EndpointInfo) -> str:
    """Markup: the host, then the shared deployment and health line.

    Already markup, so callers must not escape it: escaping printed the
    health line's own ``[dim]`` tags as text.
    """
    from ..fleet_status import deployment_and_health_line

    return f"[dim]{escape(_endpoint_host(row))} ·[/dim] {deployment_and_health_line(row)}"


def _vision_summary(vision: VisionCapabilities | None) -> str:
    """Describe image input without implying a deployment was ever tested."""
    if vision is None:
        return "unknown"
    if not vision.enabled:
        return "disabled (text only)"
    return f"enabled · {_VISION_VERIFICATION_LABELS[vision.verification]}"


_VISION_VERIFICATION_LABELS = {
    VisionVerification.UNTESTED: "not verified on this deployment",
    VisionVerification.PASSED: "image request verified",
    VisionVerification.FAILED: "image request failed",
}


def _endpoint_provider_display(row: EndpointInfo) -> str:
    return {
        ComputeProvider.MODAL: "Modal",
        ComputeProvider.PRIME: "Prime",
        ComputeProvider.VAST: "Vast",
    }.get(row.provider, row.provider.value.title())


def _endpoint_backend_display(row: EndpointInfo) -> str:
    return {
        "llamacpp": "llama.cpp",
        "vllm": "vLLM",
    }.get(_endpoint_backend(row), _endpoint_backend(row))


def _styled_state_cell(row: EndpointInfo) -> Content:
    """Render a color-coded status glyph + label for the Endpoints table."""
    normalized = _normalized_state(row.state)
    label = escape(_state_label(row.state))
    if normalized in {"active", "running"}:
        return Content.from_markup(f"[$success]● {label}[/]")
    if normalized == "deployed":
        return Content.from_markup(f"[$primary]○ {label}[/]")
    if normalized in {"starting", "initializing", "building", "deploying", "pending", "queued", "ephemeral"}:
        return Content.from_markup(f"[$warning]◐ {label}[/]")
    if normalized in {"failed", "error", "crashed"}:
        return Content.from_markup(f"[$error]✕ {label}[/]")
    if normalized in {"stopped", "stopping", "terminated", "archived", "destroyed", "deleted"}:
        return Content.from_markup(f"[dim]◌ {label}[/dim]")
    return Content.from_markup(f"[dim]· {label}[/dim]")


def _styled_endpoint_cell(row: EndpointInfo) -> Content:
    return Content.from_markup(f"[bold]{escape(_endpoint_name(row))}[/bold]")


def _app_column_redundant(row: EndpointInfo) -> bool:
    """Hide the app/pod column when it simply restates the endpoint name."""
    raw = (row.name or row.app_id or "-").strip()
    endpoint = _endpoint_name(row)
    if raw == endpoint or raw == "-":
        return True
    folded_raw = raw.casefold()
    folded_endpoint = endpoint.casefold()
    return folded_raw.endswith(f"-{folded_endpoint}") or folded_raw.endswith(f"_{folded_endpoint}")


_WIDE = WidthMode.WIDE
_STANDARD = WidthMode.STANDARD
_COMPACT = WidthMode.COMPACT
_MINIMAL = WidthMode.MINIMAL

_ENDPOINT_COLUMNS = (
    AdaptiveColumn.visible(
        "endpoint",
        "endpoint",
        _styled_endpoint_cell,
        _WIDE,
        _STANDARD,
    ),
    AdaptiveColumn.visible(
        "endpoint-host",
        "endpoint / host",
        lambda row: Content.from_markup(f"[bold]{escape(_endpoint_compact_label(row))}[/bold]"),
        _COMPACT,
        _MINIMAL,
    ),
    AdaptiveColumn.visible(
        "provider",
        "cloud",
        _endpoint_provider_display,
        _WIDE,
        _STANDARD,
    ),
    AdaptiveColumn.visible(
        "backend",
        "runtime",
        _endpoint_backend_display,
        _WIDE,
        _STANDARD,
    ),
    AdaptiveColumn.visible(
        "state",
        "state",
        _styled_state_cell,
        _WIDE,
        _STANDARD,
        _COMPACT,
        _MINIMAL,
    ),
    # What the endpoint serves: the column that tells two rows apart, and the
    # one that uses the width the four short columns leave empty.
    AdaptiveColumn.visible(
        "model",
        "model",
        lambda row: escape(clip(_endpoint_model(row), 22)),
        _WIDE,
        _STANDARD,
    ),
    # Traffic only fits where there is room to spare; every width gets the
    # same numbers in the selection detail below the table.
    AdaptiveColumn.visible(
        "throughput",
        "tok/s",
        _endpoint_throughput,
        _WIDE,
        hide_when_empty=True,
        justify="right",
    ),
    AdaptiveColumn.visible(
        "served",
        "served",
        _endpoint_tokens_served,
        _WIDE,
        hide_when_empty=True,
        justify="right",
    ),
    AdaptiveColumn.visible(
        "run",
        "run",
        _endpoint_run_time,
        _WIDE,
        min_terminal_width=108,
        hide_when_empty=True,
        justify="right",
    ),
    AdaptiveColumn.visible(
        "cost",
        "cost",
        _endpoint_cost,
        _WIDE,
        hide_when_empty=True,
        justify="right",
    ),
    AdaptiveColumn.visible(
        "app",
        "app / pod",
        lambda row: escape(row.name or row.app_id or "-"),
        _WIDE,
        min_terminal_width=132,
        redundant=_app_column_redundant,
    ),
)


class BatchServingStatsReady(Message):
    """Background passive serving-stats pass finished on cloned rows."""

    def __init__(self, target_rows: list[EndpointInfo], updated: dict[str, EndpointInfo]) -> None:
        super().__init__()
        self.target_rows = target_rows
        self.updated = updated


class LiveMetricsReady(Message):
    """Explicit single-endpoint live metrics fetch finished on a cloned row."""

    def __init__(self, key: str, updated: EndpointInfo) -> None:
        super().__init__()
        self.key = key
        self.updated = updated


def _copy_runtime_metrics(target: EndpointInfo, source: EndpointInfo) -> None:
    """Apply background-collected serving/runtime telemetry onto a UI row."""
    target.serving = source.serving
    target.live_metrics_checked_at = source.live_metrics_checked_at
    target.live_metrics_error = source.live_metrics_error
    target.run_started_at_epoch = source.run_started_at_epoch
    target.hourly_cost_usd = source.hourly_cost_usd
    target.cumulative_cost_usd = source.cumulative_cost_usd
    target.cost_estimated = source.cost_estimated
    target.cost_tracked_since_epoch = source.cost_tracked_since_epoch
    if source.runtime_status is not None:
        target.runtime_status = source.runtime_status
        target.runtime_status_detail = source.runtime_status_detail
    if source.runtime_checked_at is not None:
        target.runtime_checked_at = source.runtime_checked_at


class ManageScreen(CopyEnabledScreen):
    """Show the endpoint fleet once, then route actions for the selected row."""

    # Matches the home screen's fleet cadence. A throughput column is only
    # worth the name if it moves on its own: the rate is a delta between two
    # readings, so a screen that reads once shows nothing at all until the
    # user thinks to press a key.
    _TRAFFIC_REFRESH_INTERVAL_SECONDS = 20.0
    # Repaint the runclock between network passes without touching any runtime.
    _CLOCK_REFRESH_INTERVAL_SECONDS = 30.0

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("r", "refresh_endpoints", "Refresh", show=True),
        Binding("enter", "open_actions", "Actions", show=True, priority=True),
        Binding("f", "fetch_live_selected", "Metrics", show=True),
        Binding("a", "toggle_stopped", "Stopped", show=True),
        Binding("d", "deploy", "Deploy", show=True),
        Binding("c", "connection_selected", "Connect", show=False),
        Binding("l", "logs_selected", "Logs", show=False),
        Binding("u", "copy_base_url", "Copy URL", show=False),
        Binding("s", "status_selected", "Status", show=False),
        # Keep the shortcuts for experienced users, but let the action menu
        # carry the discoverability burden in the footer and help text.
        Binding("b", "benchmark_selected", "Benchmark", show=False),
        Binding("x", "stop_selected", "Stop", show=False),
    ]

    def __init__(self, *, jobs: bool = False) -> None:
        super().__init__()
        self._initial_tab = "manage-jobs" if jobs else "manage-endpoints"

    def compose(self) -> ComposeResult:
        table = AdaptiveDataTable(id="manage-endpoint-table", fit_to_rows=True, min_fit_rows=5)
        table.border_title = "Endpoints"
        detail = Static("", id="manage-selection-detail")
        detail.border_title = "Selected Endpoint"
        with Vertical(id="manage-layout"):
            yield Static(screen_title("Manage"), id="manage-title")
            with TabbedContent(initial=self._initial_tab, id="manage-tabs"):
                with TabPane("Endpoints", id="manage-endpoints"):
                    yield Static("[dim]Loading endpoints…[/dim]", id="manage-status")
                    yield table
                    yield detail
                with TabPane("Jobs", id="manage-jobs"):
                    yield DeploymentJobsPanel()
        yield FittedFooter()

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        if event.tabbed_content.id != "manage-tabs":
            return
        self.refresh_bindings()
        self.query_one(DeploymentJobsPanel).set_active(event.pane.id == "manage-jobs")
        if event.pane.id == "manage-jobs":
            self.query_one("#deployment-jobs", OptionList).focus()
        else:
            self.query_one("#manage-endpoint-table", AdaptiveDataTable).focus()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if super().check_action(action, parameters) is False:
            return False
        if action == "open_actions":
            return self.focused is not None and self.focused.id == "manage-endpoint-table"
        if action in {
            "refresh_endpoints", "toggle_stopped", "logs_selected", "connection_selected",
            "copy_base_url", "status_selected", "benchmark_selected", "stop_selected",
            "fetch_live_selected", "deploy",
        }:
            return self.query_one("#manage-tabs", TabbedContent).active == "manage-endpoints"
        return super().check_action(action, parameters)

    def on_mount(self) -> None:
        self._rows: list[EndpointInfo] = []
        self._all_rows: list[EndpointInfo] = []
        # A fleet is mostly history: Modal keeps stopped apps listed long after
        # they served anything, and a stopped row offers only its logs. Manage
        # opens on what is running and keeps the rest one key away.
        self._show_stopped = False
        self._outage_lines: list[str] = []
        self._rows_by_key: dict[str, EndpointInfo] = {}
        self._selected_key: str | None = None
        self._retained_providers: set[str] = set()
        self._was_suspended = False
        self._skip_next_resume_refresh = False
        self._fleet_refresh_inflight = False
        self._refresh_spinner_index = 0
        self._refresh_spinner_timer: Timer | None = None
        self._traffic_refresh_timer: Timer | None = None
        self._clock_refresh_timer: Timer | None = None
        table = self.query_one("#manage-endpoint-table", AdaptiveDataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.configure(
            _ENDPOINT_COLUMNS,
            row_key=_endpoint_key,
            profile=self.viewport_profile,
        )
        if self._initial_tab == "manage-endpoints":
            table.focus()
        self._refresh_endpoints()
        self._traffic_refresh_timer = self.set_interval(
            self._TRAFFIC_REFRESH_INTERVAL_SECONDS,
            self._refresh_serving_stats,
            name="manage-traffic-refresh",
        )
        self._clock_refresh_timer = self.set_interval(
            self._CLOCK_REFRESH_INTERVAL_SECONDS,
            self._render_rows,
            name="manage-clock-refresh",
        )

    def viewport_profile_changed(
        self,
        profile: ViewportProfile,
        previous: ViewportProfile | None,
    ) -> None:
        _ = previous
        try:
            table = self.query_one("#manage-endpoint-table", AdaptiveDataTable)
        except Exception:
            return
        table.set_viewport_profile(profile)
        if getattr(self, "_rows", None) is not None:
            self._update_selection_detail()

    def on_screen_suspend(self, _: events.ScreenSuspend) -> None:
        self._was_suspended = True
        self.query_one(DeploymentJobsPanel).set_active(False)
        # Nothing on a screen nobody is looking at is worth a request, least
        # of all one that can wake a container that had scaled to zero.
        if self._traffic_refresh_timer is not None:
            self._traffic_refresh_timer.pause()
        if self._clock_refresh_timer is not None:
            self._clock_refresh_timer.pause()
        if self._refresh_spinner_timer is not None:
            self._refresh_spinner_timer.pause()

    def on_screen_resume(self, _: events.ScreenResume) -> None:
        """Refresh the fleet after returning from an endpoint operation."""
        if self._traffic_refresh_timer is not None:
            self._traffic_refresh_timer.resume()
        if self._clock_refresh_timer is not None:
            self._clock_refresh_timer.resume()
        if self._refresh_spinner_timer is not None and self._fleet_refresh_inflight:
            self._refresh_spinner_timer.resume()
        if self._was_suspended:
            self._was_suspended = False
            show_jobs = self.query_one("#manage-tabs", TabbedContent).active == "manage-jobs"
            self.query_one(DeploymentJobsPanel).set_active(show_jobs)
            if self._skip_next_resume_refresh:
                self._skip_next_resume_refresh = False
            else:
                self.call_after_refresh(self._refresh_if_current)

    def _refresh_spinner_frame(self) -> str:
        return SUMMARY_SPINNER_FRAMES[self._refresh_spinner_index % len(SUMMARY_SPINNER_FRAMES)]

    def _set_fleet_refreshing(self, active: bool) -> None:
        self._fleet_refresh_inflight = active
        if active:
            if self._refresh_spinner_timer is None:
                self._refresh_spinner_timer = self.set_interval(
                    0.15,
                    self._tick_refresh_spinner,
                    name="manage-refresh-spinner",
                )
            else:
                self._refresh_spinner_timer.resume()
        elif self._refresh_spinner_timer is not None:
            self._refresh_spinner_timer.stop()
            self._refresh_spinner_timer = None

    def _tick_refresh_spinner(self) -> None:
        if not self._fleet_refresh_inflight:
            self._set_fleet_refreshing(False)
            return
        if not self.is_current:
            return
        self._refresh_spinner_index = (
            self._refresh_spinner_index + 1
        ) % len(SUMMARY_SPINNER_FRAMES)
        self._update_refreshing_status()

    def _update_refreshing_status(self) -> None:
        frame = self._refresh_spinner_frame()
        if self._rows:
            noun = "endpoint" if len(self._rows) == 1 else "endpoints"
            text = (
                f"[bold $primary]{frame} Refreshing managed endpoints…[/] "
                f"[dim]· {len(self._rows)} cached {noun}[/dim]"
            )
        else:
            text = f"[bold $primary]{frame} Refreshing managed endpoints…[/]"
        try:
            self.query_one("#manage-status", Static).update(text)
        except Exception:
            return

    def on_endpoints_loaded(self, message: EndpointsLoaded) -> None:
        previous_by_key = {_endpoint_key(row): row for row in self._all_rows}
        self._all_rows = sorted(
            (row for row in message.rows if row.backend is not None),
            key=lambda row: (_endpoint_name(row).casefold(), _endpoint_key(row)),
        )
        for row in self._all_rows:
            prior = previous_by_key.get(_endpoint_key(row))
            if prior is None:
                continue
            if row.live_metrics_checked_at is None and prior.live_metrics_checked_at is not None:
                row.live_metrics_checked_at = prior.live_metrics_checked_at
                row.live_metrics_error = prior.live_metrics_error
                if row.serving is None or has_explicit_live_metrics(prior):
                    row.serving = prior.serving
                if prior.run_started_at_epoch is not None and row.run_started_at_epoch is None:
                    row.run_started_at_epoch = prior.run_started_at_epoch
        self._retained_providers = retained_providers(message.discovery)
        self._outage_lines = provider_outage_lines(message.discovery)
        if message.is_stale:
            # The app posts the cached fleet immediately, then the live
            # discovery. Claiming "refreshed" for the cache is what left
            # token totals visibly frozen after pressing r: the stale rows
            # render, but the status stays "Refreshing" until the live pass
            # lands.
            self._set_fleet_refreshing(True)
            self._render_rows()
            self._refresh_serving_stats()
            return
        self._set_fleet_refreshing(False)
        self._render_rows()
        self._refresh_serving_stats()

    def _demo_app(self) -> Any:
        try:
            app = self.app
        except Exception:
            return None
        return app if getattr(app, "is_demo_mode", False) is True else None

    def _refresh_serving_stats(self) -> None:
        """Read live traffic for the rows on screen.

        The fleet listing is shared across screens and carries no traffic: the
        provider knows an endpoint exists, only the runtime knows what it has
        served. Manage asks for its own rows rather than inheriting the home
        screen's last pass, whose refresh timer is paused while this screen is
        up and whose numbers would sit frozen on a column labelled "tok/s".

        The read stays passive for Modal: banked totals are re-attached
        without contacting the runtime, so leaving Manage open never wakes a
        scaled-to-zero container. Prime and Vast rows are probed live.
        """
        rows = self._all_rows
        if not rows:
            self._render_rows()
            return
        demo_app = self._demo_app()
        if demo_app is not None:
            demo_app.annotate_serving_stats(rows)
            self._render_rows()
            return
        # Modal-only fleets need no network at all: attach the banked totals
        # synchronously instead of spending a worker to learn the same thing.
        if all(is_passively_monitored(row) for row in rows):
            attach_cached_serving_stats(rows)
            attach_endpoint_runtime(rows, explicit=False)
            self._render_rows()
            return
        username = self._modal_username()
        self.run_worker(
            lambda: self._run_serving_stats(rows, username),
            name="manage-serving-stats-worker",
            thread=True,
            exclusive=True,
        )

    def _run_serving_stats(self, rows: list[EndpointInfo], username: str) -> None:
        copies = [replace(row) for row in rows]
        annotate_serving_stats(copies, username)
        attach_endpoint_runtime(copies, explicit=False)
        self.post_message(
            BatchServingStatsReady(rows, {_endpoint_key(row): row for row in copies})
        )

    def _run_live_single(self, key: str, row_copy: EndpointInfo, username: str) -> None:
        """Fetch one endpoint's live counters, even when that may wake it."""
        demo_app = self._demo_app()
        if demo_app is not None:
            updated = demo_app.fetch_live_metrics(row_copy)
            self.post_message(LiveMetricsReady(key, updated))
            return
        result = fetch_serving_snapshot_detailed(row_copy, username, explicit=True)
        moment = time.time()
        row_copy.live_metrics_checked_at = moment
        if result.snapshot is not None:
            row_copy.serving = result.snapshot
            row_copy.live_metrics_error = None
            if is_passively_monitored(row_copy):
                try:
                    from ...core.runtime_health import record_explicit_health

                    record_explicit_health(row_copy, "healthy", None, now=moment)
                except Exception:
                    row_copy.runtime_status = "healthy"
                    row_copy.runtime_checked_at = moment
            else:
                row_copy.runtime_status = "healthy"
                row_copy.runtime_status_detail = None
        else:
            # Keep the banked totals: a failed explicit read says the probe
            # missed, not that the endpoint served nothing.
            row_copy.live_metrics_error = result.error or "endpoint did not return metrics"
        attach_endpoint_runtime([row_copy], now=moment, explicit=True)
        self.post_message(LiveMetricsReady(key, row_copy))

    def action_fetch_live_selected(self) -> None:
        """Fetch live metrics for the selected endpoint (may start its GPU)."""
        row = self._row_for_action("live-metrics")
        if row is None:
            return
        key = _endpoint_key(row)
        row_copy = replace(row)
        self.query_one("#manage-status", Static).update(
            f"[dim]Fetching live metrics for {escape(_endpoint_name(row))}…[/dim]"
        )
        username = self._modal_username()
        self.run_worker(
            lambda: self._run_live_single(key, row_copy, username),
            name=f"manage-live-metrics-{key}",
            group="manage-live-metrics",
            thread=True,
            exclusive=True,
        )

    def on_batch_serving_stats_ready(self, message: BatchServingStatsReady) -> None:
        """Apply background serving stats on the UI thread, unless the fleet moved on."""
        if message.target_rows is not self._all_rows:
            return
        for row in self._all_rows:
            updated = message.updated.get(_endpoint_key(row))
            if updated is not None:
                _copy_runtime_metrics(row, updated)
        self._render_rows()

    def on_live_metrics_ready(self, message: LiveMetricsReady) -> None:
        """Apply an explicit single-endpoint live metrics fetch on the UI thread."""
        for row in self._all_rows:
            if _endpoint_key(row) == message.key:
                _copy_runtime_metrics(row, message.updated)
                break
        self._render_rows()

    def on_serving_stats_ready(self, message: ServingStatsReady) -> None:
        """Repaint with the traffic that was read, unless the fleet moved on."""
        if message.rows is not self._all_rows:
            return
        self._render_rows()

    def action_toggle_stopped(self) -> None:
        """Show or hide endpoints that are no longer running."""
        self._show_stopped = not self._show_stopped
        self._render_rows()

    def _render_rows(self) -> None:
        """Paint the table from the cached fleet under the current filter."""
        selected_key = self._selected_key
        hidden = 0 if self._show_stopped else sum(
            1 for row in self._all_rows if is_terminal_deployment_state(row.state)
        )
        self._rows = [
            row
            for row in self._all_rows
            if self._show_stopped or not is_terminal_deployment_state(row.state)
        ]
        self._rows_by_key = {_endpoint_key(row): row for row in self._rows}
        table = self.query_one("#manage-endpoint-table", AdaptiveDataTable)
        table.set_rows(self._rows)
        # An empty table is a header row saying nothing; the status line
        # carries the empty state on its own.
        table.set_class(not self._rows, "hidden")
        self.query_one("#manage-selection-detail", Static).set_class(
            not self._rows, "hidden"
        )

        if not self._rows:
            self._selected_key = None
            # An unreachable provider is not an empty fleet, and saying so would
            # invite a second deployment of something that is already running.
            if hidden:
                summary = (
                    f"Nothing running [dim]· {hidden} stopped hidden · "
                    "a to show · d to deploy[/dim]"
                )
            elif not self._outage_lines:
                summary = "No endpoints running [dim]· d to deploy[/dim]"
            else:
                summary = "[$warning]No endpoints could be listed.[/$warning]  Press r to retry."
            self.query_one("#manage-status", Static).update(
                "\n".join([summary, *self._outage_lines])
            )
            self._update_selection_detail()
            return

        self._selected_key = (
            selected_key
            if selected_key in self._rows_by_key
            else _endpoint_key(self._rows[0])
        )
        noun = "endpoint" if len(self._rows) == 1 else "endpoints"
        if getattr(self, "_fleet_refresh_inflight", False):
            frame = self._refresh_spinner_frame()
            summary = (
                f"[bold $primary]{frame} Refreshing managed endpoints…[/] "
                f"[dim]· {len(self._rows)} cached {noun}[/dim]"
            )
        else:
            overview = fleet_summary_line(self._rows)
            summary = (
                overview
                if not self._outage_lines
                else f"[$warning]Fleet partly refreshed[/$warning] [dim]·[/dim] {overview}"
            )
            live_tps = sum(
                row.serving.tokens_per_second
                for row in self._rows
                if row.serving is not None
                and row.serving.tokens_per_second is not None
                and row.serving.tokens_per_second > 0
                and not _is_passive_without_live(row)
            )
            if live_tps > 0:
                summary += f" [dim]·[/dim] [$primary]{format_token_rate(live_tps)}[/]"
            tracked_costs = [
                row.cumulative_cost_usd
                for row in self._rows
                if row.cumulative_cost_usd is not None
            ]
            if tracked_costs:
                total_cost = sum(tracked_costs)
                any_est = any(row.cost_estimated for row in self._rows if row.cumulative_cost_usd is not None)
                prefix = "~" if any_est else ""
                summary += f" [dim]· {prefix}{format_cost(total_cost)} total[/dim]"
        if hidden:
            summary += f" [dim]· {hidden} stopped hidden (a)[/dim]"
        elif self._show_stopped:
            summary += " [dim]· including stopped (a)[/dim]"
        self.query_one("#manage-status", Static).update(
            "\n".join([summary, *self._outage_lines])
        )
        self._move_cursor_to_selected()
        self._update_selection_detail()

    def action_deploy(self) -> None:
        self.app.action_push_deploy()

    def on_endpoints_failed(self, message: EndpointsFailed) -> None:
        self._set_fleet_refreshing(False)
        self.query_one("#manage-status", Static).update(
            f"[$warning]Endpoint refresh failed:[/$warning] {escape(message.error)}"
        )

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if getattr(event, "row_key", None) is None:
            return
        self._selected_key = str(event.row_key.value)
        self._update_selection_detail()

    def on_data_table_row_selected(self, _: DataTable.RowSelected) -> None:
        self.action_open_actions()

    def action_refresh_endpoints(self) -> None:
        self._refresh_endpoints(force=True)

    def action_status_selected(self) -> None:
        row = self._row_for_action("status")
        if row is not None:
            self.app.push_screen(StatusOptionsScreen(row))

    def action_open_actions(self) -> None:
        row = self._selected_endpoint()
        if row is None:
            self.notify("Choose an endpoint first.", severity="warning", timeout=4)
            return
        self.app.push_screen(EndpointActionsScreen(row))

    def action_connection_selected(self) -> None:
        row = self._row_for_action("connection")
        if row is not None:
            self.app.push_screen(ConnectionInfoScreen(row))

    def action_logs_selected(self) -> None:
        row = self._row_for_action("logs")
        if row is not None:
            self.app.begin_logs(row, follow=True)

    def action_benchmark_selected(self) -> None:
        self.notify(BENCHMARK_UNAVAILABLE_REASON, severity="warning", timeout=8)

    def action_stop_selected(self) -> None:
        row = self._row_for_action("stop")
        if row is not None:
            self.app.push_screen(StopConfirmScreen(row))

    def action_copy_base_url(self) -> None:
        row = self._selected_endpoint()
        if row is None:
            self.notify("Choose an endpoint first.", severity="warning", timeout=4)
            return
        payload = endpoint_connection_payload(row, username=self._modal_username())
        base_url = payload.get("base_url")
        if not base_url:
            self.notify(
                "No endpoint URL is stored for this deployment.",
                severity="warning",
                timeout=5,
            )
            return
        self.app.copy_to_clipboard(base_url)
        self.notify("Copy requested: base URL", timeout=2)

    def _modal_username(self) -> str:
        return str(getattr(self.app, "_username", "") or "")

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def _refresh_endpoints(self, force: bool = False) -> None:
        self._set_fleet_refreshing(True)
        self._update_refreshing_status()
        refresh = getattr(self.app, "begin_endpoint_refresh", None)
        if callable(refresh):
            refresh(self, force=force)
            return

        try:
            rows = self.app.list_instances()
        except Exception as exc:
            self.post_message(EndpointsFailed(error=str(exc)))
            return
        self.post_message(EndpointsLoaded(rows=list(rows)))

    def _refresh_if_current(self) -> None:
        """Avoid an intermediate refresh while an action opens its monitor."""
        if self.app.screen is self:
            self._refresh_endpoints(force=True)

    def _selected_endpoint(self) -> EndpointInfo | None:
        if self._selected_key in self._rows_by_key:
            return self._rows_by_key[self._selected_key]
        table = self.query_one("#manage-endpoint-table", AdaptiveDataTable)
        if self._rows and table.row_count:
            return self._rows[min(table.cursor_row, len(self._rows) - 1)]
        return None

    def _row_for_action(self, action: str) -> EndpointInfo | None:
        row = self._selected_endpoint()
        if row is None:
            self.notify("Choose an endpoint first.", severity="warning", timeout=4)
            return None
        if action not in _available_actions(row):
            self.notify(
                f"{action.title()} is unavailable while this endpoint is "
                f"{_normalized_state(row.state) or 'unknown'}.",
                severity="warning",
                timeout=5,
            )
            return None
        return row

    def _move_cursor_to_selected(self) -> None:
        if self._selected_key is None:
            return
        for index, row in enumerate(self._rows):
            if _endpoint_key(row) == self._selected_key:
                table = self.query_one("#manage-endpoint-table", AdaptiveDataTable)
                table.move_cursor(row=index, column=0, animate=False)
                return

    def _update_selection_detail(self) -> None:
        from ..fleet_status import deployment_and_health_line

        detail = self.query_one("#manage-selection-detail", Static)
        row = self._selected_endpoint()
        if row is None:
            detail.border_title = "Selected Endpoint"
            detail.update("[dim]No endpoint selected.[/dim]")
            return

        now = time.time()
        profile = self.viewport_profile
        compact_height = profile.short or profile.constrained
        name = _endpoint_name(row)
        host = _endpoint_host(row)
        ident_bits = [host]
        if row.name and row.name != name:
            ident_bits.append(row.name)
        if row.app_id:
            ident_bits.append(row.app_id)
        detail.border_title = f"{name} · {host}"

        health = deployment_and_health_line(row)
        if row.runtime_checked_at is not None and now >= row.runtime_checked_at:
            age = max(0.0, now - row.runtime_checked_at)
            checked_label = "just now" if age < 60 else format_age(age)
            health += f" [dim](checked {checked_label})[/dim]"

        # Traffic & live telemetry summary
        serving = row.serving
        traffic_parts: list[str] = []
        if serving is not None and serving.total_tokens > 0:
            traffic_text = (
                f"{format_token_count(serving.total_tokens)} tokens served "
                f"[dim]({format_token_count(serving.total_prompt_tokens)} in / "
                f"{format_token_count(serving.total_generation_tokens)} out)[/dim]"
            )
            age_label = _serving_observed_age_label(row, now=now)
            if age_label and _is_passive_without_live(row):
                traffic_text += f" [dim](observed {age_label})[/dim]"
            traffic_parts.append(traffic_text)
        if serving is not None and not _is_passive_without_live(row):
            live = _serving_live_parts(serving)
            if live:
                traffic_parts.append(f"[$primary]{' · '.join(live)}[/]")
        elif _is_passive_without_live(row):
            traffic_parts.append(f"[dim]{PASSIVE_METRICS_NOTE} (f to fetch)[/dim]")
        elif serving is None:
            if _normalized_state(row.state) in _READY_STATES:
                traffic_parts.append("[dim]not collected yet (f to fetch)[/dim]")
            else:
                traffic_parts.append("[dim]waiting for runtime[/dim]")

        # Runtime & compute cost summary
        runtime_bits: list[str] = []
        run_label = _endpoint_run_time(row, now=now)
        if run_label != "-":
            runtime_bits.append(f"Run {run_label}")
        elif is_terminal_deployment_state(row.state):
            runtime_bits.append("Not running")
        cost_label = _endpoint_cost(row)
        if cost_label != "-":
            rate_bit = f" at {format_money(row.hourly_cost_usd)}/h" if row.hourly_cost_usd is not None else ""
            runtime_bits.append(f"Cost {cost_label}[dim]{rate_bit}[/dim]")
        elif row.hourly_cost_usd is not None:
            runtime_bits.append(f"Rate {format_money(row.hourly_cost_usd)}/h")

        # Connection summary
        payload = endpoint_connection_payload(row, username=self._modal_username())
        conn_bits: list[str] = []
        base_url = (payload.get("base_url") or "").strip()
        if base_url:
            conn_bits.append(escape(clip(base_url, 56)))
        model_id = (payload.get("model_id") or "").strip()
        if model_id:
            conn_bits.append(f"[dim]model[/dim] {escape(clip(model_id, 32))}")
        if (payload.get("api_key") or "").strip():
            conn_bits.append("[dim]API key stored[/dim]")
        if row.vision is not None and row.vision.enabled:
            conn_bits.append(f"[dim]vision[/dim] {_VISION_VERIFICATION_LABELS[row.vision.verification]}")

        notices: list[str] = []
        if row.provider.value in self._retained_providers:
            notices.append(f"{row.provider.display_name} state may be out of date")
        if row.live_metrics_error:
            notices.append(clip(row.live_metrics_error, 88))
        warning_line = f"[$warning]⚠ {escape(' · '.join(notices))}[/]" if notices else ""

        if compact_height:
            lines = [health]
            second_parts = []
            if serving is not None and serving.total_tokens > 0:
                second_parts.append(f"{format_token_count(serving.total_tokens)} tokens served")
            if serving is not None and not _is_passive_without_live(row):
                live = _serving_live_parts(serving)
                if live:
                    second_parts.append(f"[$primary]{' · '.join(live[:2])}[/]")
            elif _is_passive_without_live(row):
                second_parts.append("[dim]passive · f fetch[/dim]")
            if runtime_bits:
                second_parts.append(" · ".join(runtime_bits))
            if second_parts:
                lines.append(" [dim]·[/dim] ".join(second_parts))
            if warning_line:
                lines.append(warning_line)
            detail.update("\n".join(lines))
            return

        rows_markup: list[tuple[str, str]] = [
            ("Status", f"{health}  [dim]· {escape(' · '.join(ident_bits[1:]))}[/dim]" if len(ident_bits) > 1 else health),
            ("Metrics", "  [dim]·[/dim]  ".join(traffic_parts) if traffic_parts else "[dim]—[/dim]"),
        ]
        if runtime_bits:
            rows_markup.append(("Runtime", "  [dim]·[/dim]  ".join(runtime_bits)))
        if conn_bits:
            rows_markup.append(("Endpoint", "  [dim]·[/dim]  ".join(conn_bits)))

        body_lines = [labelled_rows_markup(rows_markup)]
        if warning_line:
            body_lines.append(warning_line)

        available = _available_actions(row)
        shortcut_hints: list[str] = ["[bold $primary]⏎[/] actions"]
        if "connection" in available:
            shortcut_hints.append("[bold $primary]c[/] connect")
            shortcut_hints.append("[bold $primary]u[/] copy URL")
        if "live-metrics" in available:
            shortcut_hints.append("[bold $primary]f[/] live metrics")
        if "status" in available:
            shortcut_hints.append("[bold $primary]s[/] status")
        if "logs" in available:
            shortcut_hints.append("[bold $primary]l[/] logs")
        if "stop" in available:
            shortcut_hints.append("[bold $error]x[/] stop")
        body_lines.append("[dim]" + "  ·  ".join(shortcut_hints) + "[/dim]")
        detail.update("\n".join(body_lines))



def _compact_action_summary(row: EndpointInfo, *, now: float | None = None) -> str:
    """One concise status/metrics line for EndpointActionsScreen."""
    from ..fleet_status import deployment_and_health_line

    moment = time.time() if now is None else now
    status_line = deployment_and_health_line(row)
    if row.runtime_checked_at is not None and moment >= row.runtime_checked_at:
        age = max(0.0, moment - row.runtime_checked_at)
        checked = "just now" if age < 60 else format_age(age)
        status_line += f" [dim]({checked})[/dim]"

    bits: list[str] = []
    serving = row.serving
    if serving is not None and not _is_passive_without_live(row):
        live = _serving_live_parts(serving)
        if live:
            bits.append(f"[$primary]{' · '.join(live[:3])}[/]")
    if serving is not None and serving.total_tokens > 0:
        tok = f"{format_token_count(serving.total_tokens)} tokens"
        if _is_passive_without_live(row):
            age_label = _serving_observed_age_label(row, now=moment)
            if age_label:
                tok += f" [dim]({age_label})[/dim]"
        bits.append(tok)
    run_label = _endpoint_run_time(row, now=moment)
    if run_label != "-":
        bits.append(run_label)
    cost_label = _endpoint_cost(row)
    if cost_label != "-":
        rate = f" [dim]({format_money(row.hourly_cost_usd)}/h)[/dim]" if row.hourly_cost_usd is not None else ""
        bits.append(f"{cost_label}{rate}")
    elif row.hourly_cost_usd is not None:
        bits.append(f"[dim]{format_money(row.hourly_cost_usd)}/h[/dim]")
    if _is_passive_without_live(row):
        bits.append("[dim]live metrics paused[/dim]")

    lines = [status_line]
    if bits:
        lines.append(" [dim]·[/dim] ".join(bits))
    if row.live_metrics_error:
        lines.append(f"[$warning]⚠ {escape(clip(row.live_metrics_error, 60))}[/]")
    return "\n".join(lines)


class EndpointActionsScreen(CopyEnabledScreen):
    """Focused action menu for one selected endpoint."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("c", "quick_connection", "Connect", show=False),
        Binding("s", "quick_status", "Status", show=False),
        Binding("l", "quick_logs", "Logs", show=False),
        Binding("f", "quick_live_metrics", "Metrics", show=False),
        Binding("x", "quick_stop", "Stop", show=False),
    ]

    _ACTION_KEYS: dict[str, str] = {
        "connection": "c",
        "status": "s",
        "logs": "l",
        "live-metrics": "f",
        "stop": "x",
    }

    _ACTION_LABELS = (
        ("connection", "  Connection info"),
        ("status", "  Check status"),
        ("logs", "  View logs"),
        ("live-metrics", "  Fetch live metrics"),
        ("stop", "  Stop endpoint"),
    )

    def __init__(self, endpoint: EndpointInfo) -> None:
        super().__init__()
        self.endpoint = endpoint

    def _action_labels(self) -> tuple[tuple[str, str], ...]:
        """Label explicit probes with their cost on Modal.

        Background fleet refreshes never wake a scaled-to-zero container;
        these actions do, so the Modal labels say so up front.
        """
        labels: list[tuple[str, str]] = []
        for action, label in self._ACTION_LABELS:
            if action == "status" and self.endpoint.provider == ComputeProvider.MODAL:
                label = "  Check status — may start GPU"
            if action == "live-metrics":
                label = _live_metrics_label(self.endpoint)
            labels.append((action, label))
        return tuple(labels)

    def _format_action_option(self, action: str, label: str) -> Content:
        stripped = label.strip()
        key = self._ACTION_KEYS.get(action, " ")
        if action == "stop":
            if self.endpoint.provider == ComputeProvider.VAST:
                pad = " " * max(2, 20 - len(stripped))
                return Content.from_markup(
                    f"  [bold $error]{key}[/]  [bold $error]{escape(stripped)}[/]{pad}[$warning]deletes disk[/]"
                )
            return Content.from_markup(
                f"  [bold $error]{key}[/]  [bold $error]{escape(stripped)}[/]"
            )
        if " — " in stripped:
            head, note = stripped.split(" — ", 1)
            pad = " " * max(2, 20 - len(head))
            return Content.from_markup(
                f"  [bold $primary]{key}[/]  [bold]{escape(head)}[/bold]{pad}[$warning]{escape(note)}[/]"
            )
        return Content.from_markup(
            f"  [bold $primary]{key}[/]  [bold]{escape(stripped)}[/bold]"
        )

    def action_quick_connection(self) -> None:
        self._submit("connection")

    def action_quick_status(self) -> None:
        self._submit("status")

    def action_quick_logs(self) -> None:
        self._submit("logs")

    def action_quick_live_metrics(self) -> None:
        self._submit("live-metrics")

    def action_quick_stop(self) -> None:
        self._submit("stop")

    def compose(self) -> ComposeResult:
        actions = OptionList(
            *(
                Option(self._format_action_option(action, label), id=action)
                for action, label in self._action_labels()
                if action in _available_actions(self.endpoint)
            ),
            id="manage-actions",
        )
        actions.border_title = "Actions"
        with VerticalScroll(classes="screen-scroll focused-panel-scroll"):
            with Vertical(id="manage-action-panel", classes="focused-panel"):
                yield Static(
                    screen_title(_endpoint_name(self.endpoint), _endpoint_host(self.endpoint)),
                    classes="focused-panel-title",
                )
                yield Static(
                    _compact_action_summary(self.endpoint),
                    id="manage-action-context",
                )
                yield actions
        yield FittedFooter()

    def on_mount(self) -> None:
        actions = self.query_one("#manage-actions", OptionList)
        if actions.option_count:
            actions.highlighted = 0
            actions.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "manage-actions":
            self._submit(str(event.option.id))

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def action_submit_selected(self) -> None:
        actions = self.query_one("#manage-actions", OptionList)
        highlighted = highlighted_option(actions)
        if highlighted is not None:
            self._submit(str(highlighted.id))

    def _submit(self, action: str) -> None:
        if action not in _available_actions(self.endpoint):
            return
        endpoint = self.endpoint
        self.app.pop_screen()
        if action == "connection":
            self.app.push_screen(ConnectionInfoScreen(endpoint))
        elif action == "status":
            # The common path probes immediately with defaults; the URL-override
            # form stays available via the hidden "s" shortcut on Manage.
            self.app.begin_status(endpoint)
        elif action == "live-metrics":
            # Route through Manage so the fetched totals repaint the table and
            # detail rather than disappearing with this menu.
            manage = self.app.screen
            if isinstance(manage, ManageScreen):
                # The menu holds the selected row; Manage re-resolves its own
                # selection, so prefer the endpoint the user just chose and
                # avoid clobbering the live fetch with an immediate resume reload.
                manage._skip_next_resume_refresh = True
                manage._selected_key = _endpoint_key(endpoint)
                manage.action_fetch_live_selected()
            else:
                self.app.begin_status(endpoint)
        elif action == "logs":
            self.app.begin_logs(endpoint, follow=True)
        elif action == "benchmark":
            self.notify(BENCHMARK_UNAVAILABLE_REASON, severity="warning", timeout=8)
        elif action == "stop":
            self.app.push_screen(StopConfirmScreen(endpoint))


class ConnectionInfoScreen(CopyEnabledScreen):
    """Show OpenAI-compatible connection details for one endpoint."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("u", "copy_base_url", "Copy URL", show=True),
        Binding("m", "copy_model_id", "Copy model", show=False),
        Binding("k", "copy_api_key", "Copy key", show=False),
        Binding("i", "copy_image_request", "Copy image", show=False),
        Binding("e", "copy_curl_example", "Copy curl", show=True),
        Binding("j", "copy_json_config", "Copy JSON", show=True),
    ]

    def __init__(self, endpoint: EndpointInfo) -> None:
        super().__init__()
        self.endpoint = endpoint
        self._payload: dict[str, str | None] = {}

    def compose(self) -> ComposeResult:
        fields = Static("", id="connection-info-fields")
        fields.border_title = "OpenAI Endpoint"
        with VerticalScroll(classes="screen-scroll focused-panel-scroll"):
            with Vertical(classes="focused-panel"):
                yield Static(screen_title("Connection info"), classes="focused-panel-title")
                yield Static(
                    f"[bold]{escape(_endpoint_name(self.endpoint))}[/bold]  "
                    f"{_endpoint_summary(self.endpoint)}",
                    id="connection-info-context",
                )
                yield fields
                with Horizontal(id="connection-info-actions"):
                    yield Button("Copy URL", id="connection-copy-url", variant="primary")
                    yield Button("Copy model", id="connection-copy-model")
                    yield Button("Copy API key", id="connection-copy-key")
                    yield Button("Copy image", id="connection-copy-image")
                    yield Button("Copy curl", id="connection-copy-curl")
                    yield Button("Copy JSON", id="connection-copy-json")
        yield FittedFooter()

    def on_mount(self) -> None:
        self._payload = endpoint_connection_payload(
            self.endpoint,
            username=self._modal_username(),
        )
        self.query_one("#connection-info-fields", Static).update(
            self._fields_markup(self._payload, self.endpoint.vision)
        )
        has_key = bool((self._payload.get("api_key") or "").strip())
        self.query_one("#connection-copy-key", Button).display = has_key
        vision = self.endpoint.vision
        self.query_one("#connection-copy-image", Button).display = bool(
            vision and vision.enabled and self._payload.get("base_url")
        )
        self.query_one("#connection-copy-url", Button).focus()

    def _modal_username(self) -> str:
        return str(getattr(self.app, "_username", "") or "")

    @staticmethod
    def _fields_markup(payload: dict[str, str | None], vision: VisionCapabilities | None = None) -> str:
        base_url = payload.get("base_url") or "(unavailable while the app is starting)"
        model_id = payload.get("model_id") or "(unknown)"
        display_name = payload.get("display_name") or ""
        rows = [("Base URL", base_url), ("Model ID", model_id)]
        # Only a display name that differs from the served one says anything.
        if display_name and display_name != model_id:
            rows.append(("Display", display_name))
        rows.append(("API key", (payload.get("api_key") or "").strip() or "none"))
        if vision is not None:
            rows.append(("Images", _vision_summary(vision)))
        return labelled_rows_markup([(label, escape(value)) for label, value in rows])

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "connection-copy-url":
            self.action_copy_base_url()
        elif event.button.id == "connection-copy-model":
            self._copy_field("model_id", empty_message="No model ID to copy")
        elif event.button.id == "connection-copy-key":
            self.action_copy_api_key()
        elif event.button.id == "connection-copy-image":
            self._copy_image_request()
        elif event.button.id == "connection-copy-curl":
            self.action_copy_curl_example()
        elif event.button.id == "connection-copy-json":
            self.action_copy_json_config()

    def _copy_image_request(self) -> None:
        base_url = (self._payload.get("base_url") or "").strip()
        model_id = (self._payload.get("model_id") or "").strip()
        if not base_url or not model_id:
            self.notify("No endpoint URL or model ID to build an image request", timeout=2)
            return
        self.app.copy_to_clipboard(image_test_command(base_url, model_id))
        self.notify("Copy requested: image request", timeout=2)

    def _copy_field(self, field: str, *, empty_message: str) -> None:
        value = (self._payload.get(field) or "").strip()
        if not value:
            self.notify(empty_message, timeout=2)
            return
        self.app.copy_to_clipboard(value)
        self.notify(f"Copy requested: {field.replace('_', ' ')}", timeout=2)

    def action_copy_base_url(self) -> None:
        self._copy_field("base_url", empty_message="No base URL to copy")

    def action_copy_model_id(self) -> None:
        self._copy_field("model_id", empty_message="No model ID to copy")

    def action_copy_api_key(self) -> None:
        self._copy_field("api_key", empty_message="No API key to copy")

    def action_copy_image_request(self) -> None:
        self._copy_image_request()

    def action_copy_curl_example(self) -> None:
        from ..connection import connection_curl_example

        example = connection_curl_example(self._payload)
        if example is None:
            self.notify("No endpoint URL or model ID for a curl example", timeout=2)
            return
        self.app.copy_to_clipboard(example)
        self.notify("Copy requested: curl example", timeout=2)

    def action_copy_json_config(self) -> None:
        from ..connection import connection_json_example

        example = connection_json_example(self._payload)
        if example is None:
            self.notify("No endpoint URL or model ID for a client config", timeout=2)
            return
        self.app.copy_to_clipboard(example)
        self.notify("Copy requested: client JSON", timeout=2)

    def action_pop_screen(self) -> None:
        self.app.pop_screen()


class StatusOptionsScreen(CopyEnabledScreen):
    """Optional status probe overrides for one preselected endpoint."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("enter", "do_submit", "Check", show=True),
    ]

    def __init__(self, endpoint: EndpointInfo) -> None:
        super().__init__()
        self.endpoint = endpoint

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="screen-scroll focused-panel-scroll"):
            with Vertical(classes="focused-panel"):
                yield Static(screen_title("Status check"), classes="focused-panel-title")
                yield Static(
                    f"[bold]{escape(_endpoint_name(self.endpoint))}[/bold]  "
                    f"{_endpoint_summary(self.endpoint)}",
                    classes="focused-panel-context",
                )
                with Vertical(classes="focused-form-card"):
                    yield FormField(
                        "Server URL override (optional)",
                        "status-url",
                        hint="Leave blank to use the endpoint URL.",
                    )
                    yield FormField("Timeout (seconds)", "status-timeout", default="60")
                    yield Static("", id="status-feedback")
                    yield Button("Check status", id="status-submit", variant="primary")
        yield FittedFooter()

    def on_mount(self) -> None:
        self.query_one("#status-url", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "status-submit":
            self._submit()

    def action_do_submit(self) -> None:
        self._submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._submit()

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def _submit(self) -> None:
        timeout_text = self.query_one("#status-timeout", Input).value.strip()
        try:
            timeout = int(timeout_text or "60")
        except ValueError:
            self.query_one("#status-feedback", Static).update(
                "[$error]Timeout must be an integer.[/$error]"
            )
            return
        if timeout <= 0:
            self.query_one("#status-feedback", Static).update(
                "[$error]Timeout must be greater than zero.[/$error]"
            )
            return
        url_override = self.query_one("#status-url", Input).value.strip() or None
        self.app.pop_screen()
        self.app.begin_status(
            self.endpoint,
            url_override=url_override,
            timeout=timeout,
        )


class BenchmarkOptionsScreen(CopyEnabledScreen):
    """AIPerf options for one preselected endpoint."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("enter", "do_submit", "Benchmark", show=True),
        Binding("ctrl+b", "do_submit", "Benchmark", show=False),
    ]

    def __init__(self, endpoint: EndpointInfo) -> None:
        super().__init__()
        self.endpoint = endpoint

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="screen-scroll focused-panel-scroll"):
            with Vertical(classes="focused-panel"):
                yield Static(screen_title("Benchmark"), classes="focused-panel-title")
                yield Static(
                    f"[bold]{escape(_endpoint_name(self.endpoint))}[/bold]  "
                    f"{_endpoint_summary(self.endpoint)}",
                    classes="focused-panel-context",
                )
                with Vertical(classes="focused-form-card"):
                    yield FormField(
                        "Concurrency sweep",
                        "benchmark-concurrency",
                        default="1,2,4,8,16",
                        hint="Comma or space separated values.",
                    )
                    yield FormField(
                        "Request count (optional)",
                        "benchmark-request-count",
                        hint="Blank uses max(24, concurrency * 4).",
                    )
                    yield FormField("Input tokens", "benchmark-input-tokens", default="550")
                    yield FormField("Output tokens", "benchmark-output-tokens", default="256")
                    yield FormField("Tokenizer", "benchmark-tokenizer", default="gpt2")
                    yield FormField(
                        "Output directory (optional)",
                        "benchmark-output-dir",
                        hint="Blank stores under ~/.llm_launchpad/benchmarks.",
                    )
                    yield Static("", id="benchmark-feedback")
                    yield Button("Benchmark", id="benchmark-submit", variant="primary")
        yield FittedFooter()

    def on_mount(self) -> None:
        self.query_one("#benchmark-concurrency", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "benchmark-submit":
            self._submit()

    def action_do_submit(self) -> None:
        self._submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._submit()

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def _submit(self) -> None:
        concurrency = self.query_one("#benchmark-concurrency", Input).value
        try:
            parse_concurrency_values(concurrency)
        except ValueError as exc:
            self.query_one("#benchmark-feedback", Static).update(
                f"[$error]{escape(str(exc))}[/$error]"
            )
            return
        request_count_text = self.query_one("#benchmark-request-count", Input).value.strip()
        request_count: int | None = None
        if request_count_text:
            try:
                request_count = int(request_count_text)
            except ValueError:
                self.query_one("#benchmark-feedback", Static).update(
                    "[$error]Request count must be an integer.[/$error]"
                )
                return
        try:
            input_tokens = int(
                self.query_one("#benchmark-input-tokens", Input).value.strip() or "550"
            )
            output_tokens = int(
                self.query_one("#benchmark-output-tokens", Input).value.strip() or "256"
            )
        except ValueError:
            self.query_one("#benchmark-feedback", Static).update(
                "[$error]Token lengths must be integers.[/$error]"
            )
            return
        if request_count is not None and request_count <= 0:
            self.query_one("#benchmark-feedback", Static).update(
                "[$error]Request count must be greater than zero.[/$error]"
            )
            return
        if input_tokens <= 0 or output_tokens <= 0:
            self.query_one("#benchmark-feedback", Static).update(
                "[$error]Token lengths must be greater than zero.[/$error]"
            )
            return
        tokenizer = self.query_one("#benchmark-tokenizer", Input).value
        output_dir = self.query_one("#benchmark-output-dir", Input).value.strip() or None
        self.app.pop_screen()
        self.app.begin_benchmark(
            self.endpoint,
            concurrency=concurrency,
            request_count=request_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokenizer=tokenizer,
            output_dir=output_dir,
        )


class StopConfirmScreen(CopyEnabledScreen):
    """Require an explicit confirmation before stopping one endpoint."""

    BINDINGS = [
        Binding("left,up", "previous_action", show=False, priority=True),
        Binding("right,down", "next_action", show=False, priority=True),
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("x", "confirm_stop", "Confirm stop", show=True),
    ]
    ACTION_IDS = ("stop-cancel", "stop-confirm")

    def __init__(self, endpoint: EndpointInfo) -> None:
        super().__init__()
        self.endpoint = endpoint

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="screen-scroll dialog-scroll"):
            with Vertical(id="stop-confirm-dialog", classes="dialog-panel"):
                yield Static(
                    f"[bold $primary]Stop {escape(_endpoint_name(self.endpoint))}?[/]",
                    classes="dialog-title",
                )
                yield Static(
                    f"[dim]{escape(self.endpoint.provider.value)}/{escape(_endpoint_backend(self.endpoint))} · "
                    f"{escape(self.endpoint.app_id or self.endpoint.name)}[/dim]"
                )
                yield Static(
                    (
                        "[$warning]This destroys the Vast.ai rental and permanently deletes its disk and model cache.[/$warning]"
                        if self.endpoint.provider == ComputeProvider.VAST
                        else "[$warning]This will terminate the selected deployment.[/$warning]"
                    ),
                    id="stop-warning",
                )
                with Horizontal(id="stop-confirm-actions", classes="dialog-actions"):
                    yield Button("Cancel", id="stop-cancel")
                    yield Button(
                        "Destroy rental and disk" if self.endpoint.provider == ComputeProvider.VAST else "Stop endpoint",
                        id="stop-confirm", variant="error",
                    )
        yield FittedFooter()

    def on_mount(self) -> None:
        self.query_one("#stop-cancel", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "stop-cancel":
            self.action_cancel()
        elif event.button.id == "stop-confirm":
            self.action_confirm_stop()

    def action_cancel(self) -> None:
        self.app.pop_screen()

    def action_previous_action(self) -> None:
        move_focus_across_widgets(self, self.ACTION_IDS, -1)

    def action_next_action(self) -> None:
        move_focus_across_widgets(self, self.ACTION_IDS, 1)

    def action_confirm_stop(self) -> None:
        self.app.pop_screen()
        self.app.begin_stop(self.endpoint)
