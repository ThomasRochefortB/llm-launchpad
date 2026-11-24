"""Main menu screen: deploy, manage endpoints, settings.

Inspired by the Codex TUI main screen with a prominent banner,
auth status line, and keyboard-navigable option list.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass, replace
import time

from rich.cells import cell_len
from rich.markup import escape

from ..format import (
    clip,
    format_token_count,
    format_token_rate,
)
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, VerticalScroll
from textual.message import Message
from textual.timer import Timer
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from ...core.backend import ModalBackend
from ...core.last_launch import LastLaunch, load_last_launch
from ...core.serving_metrics import default_tracker, usage_key
from ...core.hf_auth import HuggingFaceAuthStatus, get_huggingface_auth_status
from ...core.modal_auth import ModalAuthStatus, get_modal_auth_status
from ...core.prime_auth import PrimeAuthStatus, get_prime_auth_status
from ...core.provider_readiness import ProviderReadiness, ProviderReadinessStage
from ...core.provider_billing import (
    PROVIDER_BILLING_ORDER,
    PROVIDER_SETUP_COMMANDS,
    BillingStatus,
    ProviderBilling,
    load_modal_billing,
    load_prime_billing,
    load_vast_billing,
)
from ...core.vast_auth import resolve_vast_credentials
from ...core.quick_deploy import (
    QuickDeployCatalogInfo,
    QuickDeployProfile,
    activate_quick_deploy_catalog,
    record_quick_deploy_catalog_failure,
)
from ...core.artificial_analysis import (
    ArtificialAnalysisAuthStatus,
    get_artificial_analysis_auth_status,
)
from ...core.quick_deploy_refresh import (
    attach_quick_deploy_mtp_recommendations,
    build_live_quick_deploy_catalog,
    is_fresh_cached_quick_deploy_catalog,
    load_cached_quick_deploy_catalog,
)
from ...protocol.enums import BackendType, ComputeProvider
from ...protocol.models import EndpointInfo, FleetDiscovery, ServingSnapshot, StorageSnapshot
from ..billing_panel import render_provider_billing
from ..connection import endpoint_model_summary, resolve_openai_base_url
from ..fleet_status import (
    SCALE_TO_ZERO_PROVIDERS,
    fetch_serving_snapshot,
    is_passive_traffic_row,
    provider_outage_lines,
)
from ..markers import (
    ARTIFICIAL_ANALYSIS_MARKER,
    HUGGINGFACE_MARKER,
    MODAL_MARKER,
    PRIME_MARKER,
    VAST_MARKER,
)
from ..workers import EndpointsFailed, EndpointsLoaded, StorageFailed, StorageLoaded
from ..responsive import ViewportProfile
from ..widgets.fitted_footer import FittedFooter
from .copy_enabled import CopyEnabledScreen
from ..compat import highlighted_option, replace_options
from ..visual import screen_title

PANEL_SEPARATOR = "[dim]----------------------------------------[/dim]"


class DeploymentsLoaded(Message):
    """Main-menu deployment status fetch completed."""

    def __init__(self, rows: list[EndpointInfo]) -> None:
        super().__init__()
        self.rows = rows


class DeploymentsLoadFailed(Message):
    """Main-menu deployment status fetch failed."""

    def __init__(self, error: str) -> None:
        super().__init__()
        self.error = error


class ProviderBillingLoaded(Message):
    """One provider's billing snapshot arrived; the panel redraws every row."""

    def __init__(self, row: ProviderBilling) -> None:
        super().__init__()
        self.row = row


class ProviderBillingFinished(Message):
    """Every provider's billing read has settled; another pass may start."""


class HuggingFaceAuthLoaded(Message):
    """Main-menu Hugging Face auth check completed."""

    def __init__(self, status: HuggingFaceAuthStatus) -> None:
        super().__init__()
        self.status = status


class ArtificialAnalysisAuthLoaded(Message):
    """Main-menu Artificial Analysis auth check completed."""

    def __init__(self, status: ArtificialAnalysisAuthStatus) -> None:
        super().__init__()
        self.status = status


class ModalAuthLoaded(Message):
    """Main-menu Modal auth check completed."""

    def __init__(self, status: ModalAuthStatus) -> None:
        super().__init__()
        self.status = status


class PrimeAuthLoaded(Message):
    """Main-menu Prime auth check completed."""

    def __init__(self, status: PrimeAuthStatus) -> None:
        super().__init__()
        self.status = status


class QuickDeployCatalogLoaded(Message):
    """A refreshed Deploy catalog was loaded."""

    def __init__(
        self,
        info: QuickDeployCatalogInfo,
        profiles: tuple[QuickDeployProfile, ...],
        *,
        from_cache: bool = False,
        refresh_complete: bool = True,
    ) -> None:
        super().__init__()
        self.info = info
        self.profiles = profiles
        self.from_cache = from_cache
        self.refresh_complete = refresh_complete


class QuickDeployCatalogLoadFailed(Message):
    """The live Deploy catalog refresh failed."""

    def __init__(self, error: str) -> None:
        super().__init__()
        self.error = error


def _render_hf_auth_status(status: HuggingFaceAuthStatus | None = None) -> str:
    if status is None:
        return f"[dim]{HUGGINGFACE_MARKER} Checking Hugging Face auth...[/dim]"
    if status.authenticated:
        return f"[$success]{HUGGINGFACE_MARKER} Hugging Face authenticated[/$success]"
    if status.error:
        color = "$error" if "invalid" in status.error.lower() else "$warning"
        detail = escape(clip(status.error, 72))
        return f"[{color}]{HUGGINGFACE_MARKER} Hugging Face auth check failed: {detail}[/{color}]"
    return f"[$warning]{HUGGINGFACE_MARKER} Hugging Face not authenticated (run: hf auth login)[/$warning]"


def _render_modal_auth_status(status: ModalAuthStatus | None = None) -> str:
    if status is None:
        return f"[dim]{MODAL_MARKER} Checking Modal auth...[/dim]"
    if status.authenticated:
        return f"[$success]{MODAL_MARKER} Modal authenticated[/$success]"
    if status.error:
        detail = escape(clip(status.error, 72))
        return f"[$warning]{MODAL_MARKER} Modal auth check failed: {detail}[/$warning]"
    return (
        f"[$warning]{MODAL_MARKER} Modal not authenticated "
        f"(run: {PROVIDER_SETUP_COMMANDS[ComputeProvider.MODAL]})[/$warning]"
    )


def _render_prime_auth_status(status: PrimeAuthStatus | None = None) -> str:
    if status is None:
        return f"[dim]{PRIME_MARKER} Checking Prime Intellect auth...[/dim]"
    if status.authenticated:
        return f"[$success]{PRIME_MARKER} Prime Intellect authenticated[/$success]"
    if status.error:
        detail = escape(clip(status.error, 72))
        return f"[$warning]{PRIME_MARKER} Prime Intellect auth check failed: {detail}[/$warning]"
    return (
        f"[$warning]{PRIME_MARKER} Prime Intellect not authenticated "
        f"(run: {PROVIDER_SETUP_COMMANDS[ComputeProvider.PRIME]})[/$warning]"
    )


def _render_vast_auth_status() -> str:
    """Report the local Vast key the way doctor does: no network, no rental."""
    try:
        credentials = resolve_vast_credentials()
    except ValueError as exc:
        detail = escape(clip(str(exc), 72))
        return f"[$warning]{VAST_MARKER} Vast.ai key unreadable: {detail}[/$warning]"
    if credentials.api_key:
        return f"[$success]{VAST_MARKER} Vast.ai key configured ({escape(credentials.source)})[/$success]"
    return (
        f"[$warning]{VAST_MARKER} Vast.ai not configured "
        f"(run: {PROVIDER_SETUP_COMMANDS[ComputeProvider.VAST]})[/$warning]"
    )


def _render_artificial_analysis_auth_status(
    status: ArtificialAnalysisAuthStatus | None = None,
) -> str:
    if status is None:
        return f"[dim]{ARTIFICIAL_ANALYSIS_MARKER} Checking Artificial Analysis auth...[/dim]"
    if status.authenticated:
        tier = f" ({escape(status.tier)} tier)" if status.tier else ""
        return f"[$success]{ARTIFICIAL_ANALYSIS_MARKER} Artificial Analysis authenticated{tier}[/$success]"
    if status.error:
        color = "$error" if "invalid" in status.error.casefold() else "$warning"
        detail = escape(clip(status.error, 72))
        return f"[{color}]{ARTIFICIAL_ANALYSIS_MARKER} Artificial Analysis auth check failed: {detail}[/{color}]"
    return (
        f"[$warning]{ARTIFICIAL_ANALYSIS_MARKER} Artificial Analysis not authenticated "
        "(run: llm-launchpad aai-auth login)[/$warning]"
    )


def _render_auth_status_block(
    username: str = "",
    modal_status: ModalAuthStatus | None = None,
    hf_status: HuggingFaceAuthStatus | None = None,
    prime_status: PrimeAuthStatus | None = None,
    aai_status: ArtificialAnalysisAuthStatus | None = None,
    spinner: str = "",
    readiness: tuple[ProviderReadiness, ...] | None = None,
) -> str:
    """Render one line per credential; ``spinner`` animates checks in flight."""
    lines = [line for _, _, line in _compute_auth_entries(readiness, modal_status, prime_status)]
    lines.append(_render_hf_auth_status(hf_status))
    lines.append(_render_artificial_analysis_auth_status(aai_status))
    if spinner:
        lines = [_with_spinner(line, spinner) for line in lines]
    return "\n".join(lines)


def _compute_auth_entries(
    readiness: tuple[ProviderReadiness, ...] | None,
    modal_status: ModalAuthStatus | None,
    prime_status: PrimeAuthStatus | None,
) -> list[tuple[str, str, str]]:
    if readiness is None:
        return [
            (MODAL_MARKER, "Modal", _render_modal_auth_status(modal_status)),
            (PRIME_MARKER, "Prime Intellect", _render_prime_auth_status(prime_status)),
            (VAST_MARKER, "Vast.ai", _render_vast_auth_status()),
        ]
    states = {row.provider: row for row in readiness}
    entries = []
    for provider, marker in (
        (ComputeProvider.MODAL, MODAL_MARKER),
        (ComputeProvider.PRIME, PRIME_MARKER),
        (ComputeProvider.VAST, VAST_MARKER),
    ):
        row = states.get(provider)
        name = provider.display_name
        if row is None or row.stage == ProviderReadinessStage.CREDENTIALS_PRESENT:
            line = f"[dim]{marker} Checking {name} auth...[/dim]"
        elif row.verified:
            line = f"[$success]{marker} {name} authenticated[/$success]"
        else:
            hint = f" ({escape(row.hint)})" if row.hint else ""
            line = f"[$warning]{marker} {name} {row.stage.display_name.lower()}{hint}[/$warning]"
        entries.append((marker, name, line))
    return entries


_CHECKING_SUFFIX = "...[/dim]"

def _with_spinner(line: str, spinner: str) -> str:
    """Turn a "Checking ... auth..." line's trailing dots into a spinner."""
    if not line.endswith(_CHECKING_SUFFIX):
        return line
    return f"{line.removesuffix(_CHECKING_SUFFIX)}[/dim] [$primary]{spinner}[/]"


def _state_bucket(state: str) -> str:
    normalized = (state or "").strip().lower()
    if normalized in {"running", "deployed"}:
        return "healthy"
    if normalized in {"deploying", "starting", "initializing", "building", "ephemeral"}:
        return "deploying"
    if normalized in {"queued", "pending"}:
        return "queued"
    if normalized in {"failed", "error", "crashed"}:
        return "error"
    if normalized in {"stopped", "stopping"}:
        return "stopped"
    return "other"


def _should_show_in_panel(state: str) -> bool:
    bucket = _state_bucket(state)
    return bucket in {"healthy", "deploying", "queued", "error"}


def _runtime_bucket_from_modal_state(state: str) -> str:
    bucket = _state_bucket(state)
    if bucket == "healthy":
        return "healthy"
    if bucket in {"deploying", "queued"}:
        return "in_progress"
    if bucket == "error":
        return "error"
    return "in_progress"


def _runtime_bucket(row: EndpointInfo) -> str:
    status = (row.runtime_status or "").strip().lower()
    if status in {"healthy", "in_progress", "error", "unchecked"}:
        return status
    # A deployed Modal app is not evidence its container is warm: background
    # refreshes never probe it, so without an explicit observation it is
    # "not checked", never "healthy". Other providers keep live probing, so
    # their provider state still implies health until a probe says otherwise.
    if row.provider in SCALE_TO_ZERO_PROVIDERS and _state_bucket(row.state) == "healthy":
        return "unchecked"
    return _runtime_bucket_from_modal_state(row.state)


@dataclass(frozen=True)
class RuntimeProbeResult:
    """What one fleet probe learned about an endpoint."""

    status: str
    detail: str | None = None
    serving: ServingSnapshot | None = None


def _probe_row_runtime_status(
    row: EndpointInfo,
    username: str,
    *,
    explicit: bool = False,
) -> RuntimeProbeResult:
    modal_runtime = _runtime_bucket_from_modal_state(row.state)
    if modal_runtime != "healthy":
        return RuntimeProbeResult(modal_runtime)
    if row.backend not in {BackendType.VLLM, BackendType.LLAMACPP}:
        return RuntimeProbeResult("in_progress", "unknown backend")
    # Background refreshes never contact a scale-to-zero runtime: the request
    # itself would wake the container or extend its idle timeout. The caller
    # re-attaches the last explicit observation instead, so this path returns
    # before any HTTP request -- including the /health fallback, which a
    # blocked /metrics must never trigger on its own.
    if row.provider in SCALE_TO_ZERO_PROVIDERS and not explicit:
        return RuntimeProbeResult("unchecked", "health not checked")

    base_url, was_derived = resolve_openai_base_url(row, username=username)
    if not base_url:
        return RuntimeProbeResult("in_progress", "missing URL")

    try:
        import requests
    except ImportError:
        return RuntimeProbeResult(modal_runtime, "requests unavailable")

    base_root = base_url.rstrip("/")
    host_root = base_root[:-3] if base_root.endswith("/v1") else base_root
    host_root = host_root.rstrip("/")
    headers = (
        {"Authorization": f"Bearer {row.endpoint_api_key}"}
        if row.endpoint_api_key
        else None
    )

    # /metrics answers health and traffic together, so an endpoint that serves
    # it never pays for a separate health request.
    serving = fetch_serving_snapshot(row, username, explicit=explicit)
    if serving is not None:
        return RuntimeProbeResult("healthy", None, serving)

    # A passive Modal probe ends here: without /metrics there is no liveness
    # evidence, and falling through to /health would still wake the container.
    if row.provider in SCALE_TO_ZERO_PROVIDERS and not explicit:
        return RuntimeProbeResult("unchecked", "health not checked")

    probe_url = host_root + "/health"
    try:
        response = requests.get(probe_url, headers=headers, timeout=2.5)
        if 200 <= response.status_code < 300:
            return RuntimeProbeResult("healthy")

        if not was_derived and response.status_code in {401, 403, 404}:
            return RuntimeProbeResult("error", f"HTTP {response.status_code}")
        return RuntimeProbeResult("in_progress", f"HTTP {response.status_code}")
    except Exception as exc:
        return RuntimeProbeResult("in_progress", str(exc))


def _annotate_runtime_statuses(
    rows: list[EndpointInfo],
    username: str,
    *,
    explicit: bool = False,
) -> None:
    tracker = default_tracker()
    try:
        from ...core.runtime_health import get_health
    except Exception:  # pragma: no cover - import-time safety
        get_health = None  # type: ignore[assignment]

    candidates: list[EndpointInfo] = []
    for row in rows:
        if _runtime_bucket_from_modal_state(row.state) != "healthy":
            row.runtime_status = _runtime_bucket_from_modal_state(row.state)
            row.runtime_status_detail = None
            row.runtime_checked_at = None
        elif row.provider in SCALE_TO_ZERO_PROVIDERS and not explicit:
            # Passive Modal row: provider state plus the last explicit
            # observation, never a new request. Banked traffic stays on the
            # row; live gauges stay off it.
            stored = get_health(row) if get_health is not None else None
            if stored is not None:
                row.runtime_status = stored.status
                row.runtime_status_detail = stored.detail
                row.runtime_checked_at = stored.checked_at_epoch
            else:
                row.runtime_status = "unchecked"
                row.runtime_status_detail = "health not checked"
                row.runtime_checked_at = None
        elif row.backend in {BackendType.VLLM, BackendType.LLAMACPP}:
            candidates.append(row)
        else:
            row.runtime_status = "in_progress"
            row.runtime_status_detail = "unknown backend"
            row.runtime_checked_at = None
        # A stopped endpoint still served what it served, so the banked total
        # stays on the row. Only the live gauges go away with the container.
        row.serving = tracker.snapshot(usage_key(row))

    if not candidates:
        return

    max_workers = min(4, len(candidates))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_probe_row_runtime_status, row, username, explicit=explicit): row
            for row in candidates
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = RuntimeProbeResult("in_progress", str(exc))
            row.runtime_status = result.status
            row.runtime_status_detail = result.detail
            if result.serving is not None:
                row.serving = result.serving
            # An explicit Modal verdict outlives this refresh: later passive
            # passes re-attach it with its age instead of reprobing.
            if row.provider in SCALE_TO_ZERO_PROVIDERS and explicit:
                try:
                    from ...core.runtime_health import record_explicit_health as _record

                    _record(row, result.status, result.detail)
                except Exception:
                    pass
                # record_explicit_health stamps the row itself.
            elif row.provider not in SCALE_TO_ZERO_PROVIDERS:
                row.runtime_checked_at = None


def _refresh_cached_passive_rows(rows: list[EndpointInfo]) -> None:
    """Re-attach passive Modal observations to cached annotated rows.

    The fleet fingerprint does not include health (it describes provider
    state), so a status check that lands inside the runtime cache TTL would
    otherwise stay invisible until the cache expires. This touches only
    scale-to-zero rows and never the network; Prime/Vast verdicts keep their
    cached values.
    """
    tracker = default_tracker()
    try:
        from ...core.runtime_health import get_health
    except Exception:  # pragma: no cover - import-time safety
        get_health = None  # type: ignore[assignment]
    for row in rows:
        if row.provider not in SCALE_TO_ZERO_PROVIDERS:
            continue
        if _runtime_bucket_from_modal_state(row.state) != "healthy":
            continue
        stored = get_health(row) if get_health is not None else None
        if stored is not None:
            row.runtime_status = stored.status
            row.runtime_status_detail = stored.detail
            row.runtime_checked_at = stored.checked_at_epoch
        row.serving = tracker.snapshot(usage_key(row))


def _backend_display_name(backend: BackendType | None) -> str:
    if backend == BackendType.VLLM:
        return "vLLM"
    if backend == BackendType.LLAMACPP:
        return "llama.cpp"
    return "unknown"


def _friendly_count_line(rows: list[EndpointInfo]) -> list[str]:
    from ..fleet_status import fleet_summary_line

    backend_counts = Counter(
        row.backend.value if row.backend is not None else "unknown"
        for row in rows
    )

    backend_parts = []
    if backend_counts.get("vllm", 0):
        backend_parts.append(f"{backend_counts['vllm']} vLLM")
    if backend_counts.get("llamacpp", 0):
        backend_parts.append(f"{backend_counts['llamacpp']} llama.cpp")
    if not backend_parts:
        backend_parts.append(f"{len(rows)} launchpad")
    return [f"{fleet_summary_line(rows)}  [dim]· {' · '.join(backend_parts)}[/dim]", ""]


def _wrap_url_for_panel(value: str, width: int = 44) -> list[str]:
    """Pre-wrap long URLs so they remain readable in the narrow status panel."""
    text = (value or "").strip()
    if not text:
        return []
    if len(text) <= width:
        return [text]

    # Keep separators attached to the preceding chunk so we don't end up
    # with visual artifacts like a line containing only "-".
    tokens: list[str] = []
    current = ""
    for char in text:
        current += char
        if char in {"/", "-"}:
            tokens.append(current)
            current = ""
    if current:
        tokens.append(current)

    lines: list[str] = []
    line = ""
    for token in tokens:
        if not line:
            line = token
            continue
        if len(line) + len(token) <= width:
            line += token
            continue
        lines.append(line)
        line = token
    if line:
        lines.append(line)
    return lines or [text]


# Each provider names its unit of compute differently, and a row that calls a
# Vast rental a "Prime Intellect pod" sends the reader looking for it in the
# wrong console.
_PROVIDER_RESOURCE_LABELS = {
    ComputeProvider.MODAL: "Modal app",
    ComputeProvider.PRIME: "Prime Intellect pod",
    ComputeProvider.VAST: "Vast.ai rental",
}


def _serving_panel_line(row: EndpointInfo) -> str | None:
    """One-line traffic summary for the fleet panel, or None with nothing to say.

    Passively monitored Modal rows show banked totals only: live gauges stay
    blank (``-`` in Manage) so a historical total is never mistaken for a
    current measurement.
    """
    serving = row.serving
    if serving is None:
        return None
    passive = is_passive_traffic_row(row)
    parts: list[str] = []
    if not passive and serving.tokens_per_second is not None:
        parts.append(format_token_rate(serving.tokens_per_second))
    if not passive and serving.stats.requests_running:
        parts.append(f"{serving.stats.requests_running:,.0f} running")
    if serving.total_tokens > 0:
        parts.append(f"{format_token_count(serving.total_tokens)} served")
    if not parts:
        return None
    return f"[dim]Traffic:[/dim] {' · '.join(parts)}"


def _passive_metrics_line(row: EndpointInfo) -> str | None:
    """Explain why a serving Modal row shows totals but no live gauges."""
    if not is_passive_traffic_row(row):
        return None
    if row.serving is None or row.serving.total_tokens <= 0:
        return None
    from ..fleet_status import PASSIVE_METRICS_NOTE

    return f"[dim]{PASSIVE_METRICS_NOTE}[/dim]"


def _render_deployment_status(
    rows: list[EndpointInfo],
    username: str = "",
    discovery: FleetDiscovery | None = None,
) -> str:
    # A provider that did not answer is reported before any count, because
    # "no apps" and "we could not ask" lead to opposite decisions.
    outage_lines = provider_outage_lines(discovery)
    if not rows:
        if outage_lines:
            return "\n".join(outage_lines)
        if discovery is not None and not discovery.listings:
            return "[dim]Fleet status unknown · No provider listing available[/dim]"
        return "[dim]No active launchpad apps.[/dim]"

    header_lines = _friendly_count_line(rows)

    display_rows = sorted(
        rows,
        key=lambda row: (
            {"healthy": 0, "deploying": 1, "queued": 2, "error": 3, "stopped": 4, "other": 5}.get(
                _state_bucket(row.state),
                6,
            ),
            row.name.casefold(),
        ),
    )

    app_lines = []
    for index, row in enumerate(display_rows):
        from ..fleet_status import deployment_and_health_line

        instance = (row.instance_name or "").strip() or "default"
        backend_name = _backend_display_name(row.backend)
        app_lines.append(
            f"[bold]{escape(instance)}[/bold]  "
            f"[dim]{escape(backend_name)} · {row.provider.value}[/dim]  "
            f"{deployment_and_health_line(row)}"
        )

        resource_label = _PROVIDER_RESOURCE_LABELS.get(
            row.provider, f"{row.provider.display_name} deployment"
        )
        modal_app_line = f"  [dim]{resource_label}:[/dim] {escape(row.name or '')}"
        if (row.app_id or "").strip():
            modal_app_line += f" [dim]({escape(row.app_id)})[/dim]"
        app_lines.append(modal_app_line)

        model_id, display_name = endpoint_model_summary(row)
        model_line = f"  [dim]Model ID:[/dim] {escape(model_id or '')}"
        if display_name and display_name != model_id:
            model_line += f"  [dim]Display name:[/dim] {escape(display_name)}"
        app_lines.append(model_line)

        traffic_line = _serving_panel_line(row)
        if traffic_line:
            passive_line = _passive_metrics_line(row)
            if passive_line:
                traffic_line = f"{traffic_line} · [dim]Live metrics paused[/dim]"
            app_lines.append(f"  {traffic_line}")

        base_url, _was_derived = resolve_openai_base_url(row, username=username)
        show_connection = _state_bucket(row.state) == "healthy" or bool((row.web_url or "").strip())
        if base_url:
            base_root = base_url.rstrip("/")
            if base_root.endswith("/v1"):
                host_url = base_root[: -len("/v1")] or base_root
            else:
                host_url = base_root

            if show_connection:
                key_status = "stored locally" if row.endpoint_api_key else "none"
                app_lines.append(
                    f"  [dim]Base URL:[/dim] {escape(host_url)}  "
                    f"[dim]API key:[/dim] {key_status}"
                )
            else:
                app_lines.append("  [dim]OpenAI URL available once serving.[/dim]")
        else:
            if _state_bucket(row.state) in {"deploying", "queued"}:
                app_lines.append("  [dim]OpenAI URL unavailable while starting.[/dim]")
            else:
                app_lines.append("  [dim]OpenAI URL unavailable.[/dim]")

        if index != len(display_rows) - 1:
            app_lines.append("")

    preface = [*outage_lines, ""] if outage_lines else []
    return "\n".join(preface + header_lines + app_lines)


_ActionLabels = tuple[tuple[str, str], ...]

# Fullest first. `_fit_action_labels` takes the first one that fits the width
# the option list actually receives, so a description is shortened rather than
# wrapped onto a second line that breaks the two-column grid.
_ACTION_LABEL_TIERS: tuple[_ActionLabels, ...] = (
    (
        ("deploy", "  Deploy model       Pick a model, get a live placement"),
        ("custom-deploy", "  Advanced deploy    llama.cpp / vLLM expert form"),
        ("manage", "  Manage             Endpoints and jobs"),
        ("storage", "  Storage            Cached models, pre-download, delete"),
        ("settings", "  Settings           Appearance and deploy defaults"),
    ),
    (
        ("deploy", "  Deploy model       Pick a model and deploy"),
        ("custom-deploy", "  Advanced deploy    llama.cpp / vLLM form"),
        ("manage", "  Manage             Endpoints and jobs"),
        ("storage", "  Storage            Cached models"),
        ("settings", "  Settings           Appearance, defaults"),
    ),
    (
        ("deploy", "  Deploy model"),
        ("custom-deploy", "  Advanced deploy"),
        ("manage", "  Manage"),
        ("storage", "  Storage"),
        ("settings", "  Settings"),
    ),
)


def _relaunch_labels(last: LastLaunch) -> tuple[tuple[str, str], ...]:
    """One relaunch row per label tier, fullest first (sized to avoid wrapping)."""
    prefix = "  Relaunch           "
    shape = f"{last.gpu_count}x {last.gpu_type}" if last.gpu_type else ""
    if shape:
        max_name = max(10, 56 - len(prefix) - len(shape) - 3)
        tier0 = f"{prefix}{clip(last.display_name, max_name)} · {shape}"
    else:
        tier0 = f"{prefix}{clip(last.display_name, 35)}"
    return (
        ("relaunch", tier0),
        ("relaunch", f"{prefix}{clip(last.display_name, 26)}"),
        ("relaunch", "  Relaunch last"),
    )


def _action_label_tiers(last: LastLaunch | None) -> tuple[_ActionLabels, ...]:
    """The static tiers, headed by a relaunch row once something was deployed."""
    if last is None:
        return _ACTION_LABEL_TIERS
    rows = _relaunch_labels(last)
    return tuple((row, *tier) for row, tier in zip(rows, _ACTION_LABEL_TIERS, strict=True))


class MainMenuScreen(CopyEnabledScreen):
    """Top-level menu: deploy a model, custom deploy, manage, storage, settings."""

    BINDINGS = [
        Binding("d", "select_deploy", "Deploy", show=True),
        Binding("r", "select_relaunch", "Relaunch", show=True),
        # The menu itself lists every destination; the footer keeps the
        # frequent ones and help carries the rest.
        Binding("c", "select_custom_deploy", "Advanced", show=False),
        Binding("m", "select_manage", "Manage", show=True),
        Binding("t", "select_storage", "Storage", show=False),
        Binding("s", "select_settings", "Settings", show=False),
        Binding("i", "toggle_details", "Details", show=True),
    ]
    _ENDPOINT_REFRESH_INTERVAL_SECONDS = 20.0
    _BILLING_REFRESH_INTERVAL_SECONDS = 300.0
    _SECONDARY_REFRESH_DELAY_SECONDS = 0.5
    _RUNTIME_STATUS_CACHE_TTL_SECONDS = 20.0

    def __init__(self, username: str = "", version: str = "") -> None:
        super().__init__()
        self.username = username
        self.version = version
        self._readiness: tuple[ProviderReadiness, ...] | None = None
        self._modal_auth_status: ModalAuthStatus | None = None
        self._prime_auth_status: PrimeAuthStatus | None = None
        self._hf_auth_status: HuggingFaceAuthStatus | None = None
        self._aai_auth_status: ArtificialAnalysisAuthStatus | None = None
        self._hf_auth_refresh_inflight = False
        self._aai_auth_refresh_inflight = False
        self._modal_auth_refresh_inflight = False
        self._prime_auth_refresh_inflight = False
        self._quick_deploy_catalog_refresh_inflight = False
        self._last_catalog: QuickDeployCatalogInfo | None = None
        self._status_refresh_inflight = False
        self._billing_refresh_inflight = False
        self._provider_billing: dict[ComputeProvider, ProviderBilling] = {
            provider: ProviderBilling.loading(provider)
            for provider in PROVIDER_BILLING_ORDER
        }
        self._storage_snapshot: StorageSnapshot | None = None
        self._was_suspended = False
        self._secondary_refresh_started = False
        self._last_billing_refresh_at = 0.0
        self._endpoint_refresh_timer: Timer | None = None
        self._secondary_refresh_timer: Timer | None = None
        self._runtime_rows: list[EndpointInfo] = []
        self._runtime_rows_fingerprint: tuple[tuple[object, ...], ...] = ()
        self._runtime_rows_cached_at = 0.0
        self._fleet_discovery: FleetDiscovery | None = None
        self._visible_fleet: list[EndpointInfo] = []
        self._fleet_detail_error: str | None = None
        self._action_labels: _ActionLabels | None = None
        self._action_label_ceiling = 0
        self._last_launch = load_last_launch()

    def compose(self) -> ComposeResult:
        snapshot = getattr(self.app, "provider_readiness_snapshot", None)
        self._readiness = snapshot() if callable(snapshot) else None
        with Center(id="main-menu-center"):
            with VerticalScroll(id="main-menu-primary"):
                yield Static("[bold $primary]LLM Launchpad[/]", id="compact-menu-header")
                yield OptionList(
                    *(Option(label, id=option_id)
                      for option_id, label in _action_label_tiers(self._last_launch)[0]),
                    id="action-list",
                )
                yield Static("[dim]Checking fleet…[/dim]", id="fleet-summary-line")
                yield Static("", id="auth-status-block", classes="hidden")
        yield FittedFooter()

    def on_mount(self) -> None:
        """Focus the option list so arrow-key navigation works immediately."""
        action_list = self.query_one("#action-list", OptionList)
        if action_list.option_count > 0:
            action_list.action_first()
        action_list.focus()
        if self._readiness is None:
            self._refresh_modal_auth_status()
            self._refresh_prime_auth_status()
        self._refresh_hf_auth_status()
        self._refresh_aai_auth_status()
        self._render_connection_widgets()
        self._refresh_panels()
        # Start the catalog build immediately: it is the longest pole and
        # used to wait behind the deferred secondary pass. Billing/storage
        # stay deferred so first paint stays fast.
        self._refresh_quick_deploy_catalog()
        self._secondary_refresh_timer = self.set_timer(
            self._SECONDARY_REFRESH_DELAY_SECONDS,
            self._refresh_secondary_panels,
            name="main-menu-secondary-refresh-delay",
        )
        self._endpoint_refresh_timer = self.set_interval(
            self._ENDPOINT_REFRESH_INTERVAL_SECONDS,
            self._refresh_panels,
            name="main-menu-endpoint-refresh",
        )
        self.set_interval(
            self._BILLING_REFRESH_INTERVAL_SECONDS,
            self._refresh_billing_panels,
            name="main-menu-billing-refresh",
        )

    def on_screen_suspend(self, _: events.ScreenSuspend) -> None:
        self._was_suspended = True
        self._pause_refresh_timers()

    def on_screen_resume(self, _: events.ScreenResume) -> None:
        """Refresh the fleet when returning from a nested flow."""
        if not self._was_suspended:
            return
        self._was_suspended = False
        self._resume_refresh_timers()
        self._refresh_panels()
        self._sync_storage_snapshot()
        last = load_last_launch()
        if last != self._last_launch:
            # A deploy started from a nested flow is now the one to offer.
            self._last_launch = last
            self._action_labels = None
            self._fit_action_labels()
            self.refresh_bindings()
        if not self._secondary_refresh_started:
            if self._secondary_refresh_timer is not None:
                self._secondary_refresh_timer.stop()
                self._secondary_refresh_timer = None
            self._refresh_secondary_panels()

    def _pause_refresh_timers(self) -> None:
        for timer in (
            self._secondary_refresh_timer,
            self._endpoint_refresh_timer,
        ):
            if timer is not None:
                timer.pause()

    def _resume_refresh_timers(self) -> None:
        for timer in (
            self._secondary_refresh_timer,
            self._endpoint_refresh_timer,
        ):
            if timer is not None:
                timer.resume()

    def set_modal_username(self, username: str) -> None:
        """Apply the asynchronously resolved profile name to connection URLs."""
        if username == self.username:
            return
        self.username = username
        if self._runtime_rows:
            self._show_deployments(self._runtime_rows)

    def viewport_profile_changed(
        self,
        profile: ViewportProfile,
        previous: ViewportProfile | None,
    ) -> None:
        self._refresh_action_labels(profile)

    def _refresh_action_labels(self, profile: ViewportProfile) -> None:
        """Use concise action records when descriptive columns no longer fit."""
        # A compact terminal deliberately drops the description column for
        # concise names; measurement may shorten labels further but must never
        # talk that decision back up into a padded two-column grid.
        tiers = _action_label_tiers(self._last_launch)
        self._action_label_ceiling = len(tiers) - 1 if profile.compact else 0
        try:
            action_list = self.query_one("#action-list", OptionList)
            has_width = action_list.content_size.width > 0
        except Exception:
            has_width = False
        if has_width:
            self._fit_action_labels()
        else:
            self._apply_action_labels(tiers[self._action_label_ceiling])
        # Re-fit after layout in case width changed on the same resize pass.
        self.call_after_refresh(self._fit_action_labels)

    def _fit_action_labels(self) -> None:
        """Shorten the labels if the list is narrower than the terminal implied."""
        try:
            action_list = self.query_one("#action-list", OptionList)
        except Exception:
            return
        available = action_list.content_size.width
        if available <= 0:
            return
        tiers = _action_label_tiers(self._last_launch)
        for tier in tiers[self._action_label_ceiling:]:
            if max(cell_len(label) for _option_id, label in tier) <= available:
                self._apply_action_labels(tier)
                return
        self._apply_action_labels(tiers[-1])

    def _apply_action_labels(self, labels: _ActionLabels) -> None:
        """Install a label set, keeping whichever action was highlighted."""
        try:
            action_list = self.query_one("#action-list", OptionList)
        except Exception:
            return
        if self._action_labels == labels:
            return
        highlighted = highlighted_option(action_list)
        selected_id = str(highlighted.id) if highlighted is not None else "deploy"
        self._action_labels = labels
        replace_options(action_list, 
            [Option(label, id=option_id) for option_id, label in labels]
        )
        for index, (option_id, _) in enumerate(labels):
            if option_id == selected_id:
                action_list.highlighted = index
                break

    def action_toggle_details(self) -> None:
        """Open fleet, billing and connection detail as a full-width view."""
        self.app.push_screen(HomeDetailsScreen(self))

    def _refresh_quick_deploy_catalog(self) -> None:
        if self._quick_deploy_catalog_refresh_inflight or (
            self._last_catalog is not None and self._can_reuse_catalog(self._last_catalog)
        ):
            return
        self._quick_deploy_catalog_refresh_inflight = True
        self.run_worker(
            self._run_load_quick_deploy_catalog,
            name="main-menu-quick-deploy-catalog-worker",
            thread=True,
        )

    def ensure_quick_deploy_catalog_refresh(self) -> None:
        """Start the live model-catalog refresh before opening the picker."""
        self._refresh_quick_deploy_catalog()

    @staticmethod
    def _can_reuse_catalog(info: QuickDeployCatalogInfo) -> bool:
        return info.ready and not info.error and not info.is_fallback and is_fresh_cached_quick_deploy_catalog(info)

    def _run_load_quick_deploy_catalog(self) -> None:
        # Publish the disk snapshot while the slower live refresh is in progress.
        try:
            cached = load_cached_quick_deploy_catalog()
        except Exception:
            cached = None
        if cached is not None:
            info, profiles = cached
            if profiles or info.exclusions:
                fresh = self._can_reuse_catalog(info)
                self.post_message(QuickDeployCatalogLoaded(info, profiles, from_cache=True, refresh_complete=fresh))
                if fresh:
                    return
        self._run_refresh_quick_deploy_catalog()

    def _run_refresh_quick_deploy_catalog(self) -> None:
        try:
            info, profiles = build_live_quick_deploy_catalog()
        except Exception as exc:
            self.post_message(QuickDeployCatalogLoadFailed(error=str(exc)))
            return
        self.post_message(QuickDeployCatalogLoaded(info=info, profiles=profiles, refresh_complete=False))
        # MTP probes are the slowest per-model fetch and only feed the
        # draft-model toggle; attach them as a trailing update so the
        # picker stays usable while they resolve. They are written back to
        # the snapshot so the next launch opens with MTP already resolved
        # instead of reprobing every repository.
        try:
            upgraded = attach_quick_deploy_mtp_recommendations(profiles, info=info)
        except Exception:
            upgraded = profiles
        self.post_message(QuickDeployCatalogLoaded(info=info, profiles=upgraded))

    def on_quick_deploy_catalog_loaded(
        self,
        message: QuickDeployCatalogLoaded,
    ) -> None:
        self._quick_deploy_catalog_refresh_inflight = not message.refresh_complete
        self._last_catalog = message.info
        activate_quick_deploy_catalog(
            message.info,
            message.profiles,
        )
        notifier = getattr(self.app, "quick_deploy_catalog_updated", None)
        if callable(notifier):
            notifier()

    def on_quick_deploy_catalog_load_failed(
        self,
        message: QuickDeployCatalogLoadFailed,
    ) -> None:
        self._quick_deploy_catalog_refresh_inflight = False
        if record_quick_deploy_catalog_failure(message.error):
            notifier = getattr(self.app, "quick_deploy_catalog_updated", None)
            if callable(notifier):
                notifier()

    def _render_connection_widgets(self) -> None:
        if not self.is_mounted:
            return
        if self._readiness is None:
            issues = [
                provider.display_name
                for provider, status in (
                    (ComputeProvider.MODAL, self._modal_auth_status),
                    (ComputeProvider.PRIME, self._prime_auth_status),
                )
                if status is not None and status.error
            ]
        else:
            issues = [
                row.provider.display_name for row in self._readiness
                if row.stage in (ProviderReadinessStage.AUTH_FAILED, ProviderReadinessStage.UNREACHABLE)
            ]
        block = self.query_one("#auth-status-block", Static)
        block.set_class(not issues, "hidden")
        block.update(f"[$warning]Check {escape(', '.join(issues))}[/] · i Details" if issues else "")
        details = self.app.screen
        if isinstance(details, HomeDetailsScreen) and details._menu is self and details.is_mounted:
            details.update_connections()

    def on_provider_readiness(self, readiness: tuple[ProviderReadiness, ...]) -> None:
        self._readiness = readiness
        self._render_connection_widgets()

    def _refresh_modal_auth_status(self) -> None:
        if self._modal_auth_refresh_inflight:
            return
        self._modal_auth_refresh_inflight = True
        self._render_connection_widgets()
        self.run_worker(
            self._run_load_modal_auth_status,
            name="main-menu-modal-auth-worker",
            thread=True,
        )

    def _run_load_modal_auth_status(self) -> None:
        poster = getattr(self, "post_message", None)
        if poster is None:
            return
        try:
            status = get_modal_auth_status()
        except Exception as exc:
            status = ModalAuthStatus(authenticated=False, error=str(exc))
        poster(ModalAuthLoaded(status=status))

    def on_modal_auth_loaded(self, message: ModalAuthLoaded) -> None:
        self._modal_auth_refresh_inflight = False
        self._modal_auth_status = message.status
        self._render_connection_widgets()
        self._apply_auth_to_billing(
            ComputeProvider.MODAL, message.status.authenticated
        )

    def _refresh_prime_auth_status(self) -> None:
        if self._prime_auth_refresh_inflight:
            return
        self._prime_auth_refresh_inflight = True
        self.run_worker(
            self._run_load_prime_auth_status,
            name="main-menu-prime-auth-worker",
            thread=True,
        )

    def _run_load_prime_auth_status(self) -> None:
        self.post_message(PrimeAuthLoaded(status=get_prime_auth_status()))

    def on_prime_auth_loaded(self, message: PrimeAuthLoaded) -> None:
        self._prime_auth_refresh_inflight = False
        self._prime_auth_status = message.status
        self._render_connection_widgets()
        self._apply_auth_to_billing(ComputeProvider.PRIME, message.status.authenticated)

    def _refresh_hf_auth_status(self) -> None:
        if self._hf_auth_refresh_inflight:
            return
        self._hf_auth_refresh_inflight = True
        self._render_connection_widgets()
        self.run_worker(
            self._run_load_hf_auth_status,
            name="main-menu-hf-auth-worker",
            thread=True,
        )

    def _run_load_hf_auth_status(self) -> None:
        poster = getattr(self, "post_message", None)
        if poster is None:
            return
        try:
            status = get_huggingface_auth_status()
        except Exception as exc:
            status = HuggingFaceAuthStatus(authenticated=False, error=str(exc))
        poster(HuggingFaceAuthLoaded(status=status))

    def on_hugging_face_auth_loaded(self, message: HuggingFaceAuthLoaded) -> None:
        self._hf_auth_refresh_inflight = False
        self._hf_auth_status = message.status
        self._render_connection_widgets()

    def _refresh_aai_auth_status(self) -> None:
        if self._aai_auth_refresh_inflight:
            return
        self._aai_auth_refresh_inflight = True
        self.run_worker(
            self._run_load_aai_auth_status,
            name="main-menu-aai-auth-worker",
            thread=True,
        )

    def _run_load_aai_auth_status(self) -> None:
        poster = getattr(self, "post_message", None)
        if poster is None:
            return
        try:
            status = get_artificial_analysis_auth_status()
        except Exception as exc:
            status = ArtificialAnalysisAuthStatus(
                authenticated=False,
                error=str(exc),
            )
        poster(ArtificialAnalysisAuthLoaded(status=status))

    def on_artificial_analysis_auth_loaded(
        self,
        message: ArtificialAnalysisAuthLoaded,
    ) -> None:
        self._aai_auth_refresh_inflight = False
        self._aai_auth_status = message.status
        self._render_connection_widgets()

    def _refresh_panels(self) -> None:
        if not self._is_active_screen():
            return
        self._refresh_deployment_status()

    def _refresh_secondary_panels(self) -> None:
        self._secondary_refresh_timer = None
        if not self._is_active_screen():
            return
        self._secondary_refresh_started = True
        # The catalog refresh already started on mount; this is only a
        # backstop for screens mounted before that change or refreshes
        # skipped while suspended.
        self._refresh_quick_deploy_catalog()
        self._refresh_storage_estimate()

    def _refresh_billing_panels(self) -> None:
        if not self.is_mounted:
            return
        active = self.app.screen
        if not isinstance(active, HomeDetailsScreen) or active._menu is not self:
            return
        self._sync_storage_snapshot()
        self._last_billing_refresh_at = time.monotonic()
        self._refresh_provider_billing()

    def _is_active_screen(self) -> bool:
        """Return whether this screen is the visible top of the app stack."""
        try:
            return self.app.screen is self
        except Exception:
            return False

    def _sync_storage_snapshot(self) -> None:
        if not self.is_mounted:
            return
        cached_storage_snapshot = getattr(self.app, "cached_storage_snapshot", None)
        if callable(cached_storage_snapshot):
            snapshot = cached_storage_snapshot()
            if snapshot is not self._storage_snapshot:
                self._storage_snapshot = snapshot
                self._update_billing_panel()

    def _refresh_storage_estimate(self) -> None:
        refresh_storage = getattr(self.app, "begin_storage_refresh", None)
        if callable(refresh_storage):
            refresh_storage(self, force=False)

    def _refresh_deployment_status(self) -> None:
        if self._status_refresh_inflight:
            return
        self._status_refresh_inflight = True
        if not self._runtime_rows_cached_at:
            self._update_fleet_summary()
        refresh = getattr(self.app, "begin_endpoint_refresh", None)
        if callable(refresh):
            refresh(self, force=False)
            return
        self.run_worker(self._run_load_deployments, name="main-menu-status-worker", thread=True)

    def _run_load_deployments(self) -> None:
        poster = getattr(self, "post_message", None)
        if poster is None:
            return
        try:
            list_instances = getattr(self.app, "list_instances", None)
            rows = list_instances() if callable(list_instances) else ModalBackend.list_apps()
        except Exception as exc:
            poster(DeploymentsLoadFailed(error=str(exc)))
            return
        last_discovery = getattr(self.app, "last_fleet_discovery", None)
        if callable(last_discovery):
            self._fleet_discovery = last_discovery()
        if rows is None:
            poster(DeploymentsLoadFailed(error="Could not read Modal app list."))
            return
        _annotate_runtime_statuses(rows, self.username)
        poster(DeploymentsLoaded(rows=[row for row in rows if row.backend is not None]))

    def on_endpoints_loaded(self, message: EndpointsLoaded) -> None:
        """Add runtime health details without mutating the shared endpoint cache."""
        self._fleet_discovery = message.discovery
        rows = [replace(row) for row in message.rows if row.backend is not None]
        fingerprint = self._runtime_fingerprint(rows)
        cached_runtime_is_fresh = (
            fingerprint == self._runtime_rows_fingerprint
            and time.monotonic() - self._runtime_rows_cached_at
            <= self._RUNTIME_STATUS_CACHE_TTL_SECONDS
        )
        if cached_runtime_is_fresh:
            if not message.is_stale:
                self._status_refresh_inflight = False
            # Passive rows are cheap to refresh: re-attach banked totals and
            # any explicit health recorded since the cache was written, so a
            # just-completed status check shows its age instead of hiding
            # behind the TTL. Prime/Vast rows keep their cached verdicts --
            # reprobing them here would defeat the cache.
            cached = [replace(row) for row in self._runtime_rows]
            try:
                _refresh_cached_passive_rows(cached)
            except Exception:
                pass
            self._show_deployments(cached)
            return
        if message.is_stale:
            self._show_deployments(rows)
            return
        self.run_worker(
            lambda: self._run_annotate_deployments(rows),
            name="main-menu-runtime-status-worker",
            group="main-menu-runtime-status",
            thread=True,
            exclusive=True,
        )

    def _run_annotate_deployments(self, rows: list[EndpointInfo]) -> None:
        _annotate_runtime_statuses(rows, self.username)
        self.post_message(DeploymentsLoaded(rows=rows))

    def on_endpoints_failed(self, message: EndpointsFailed) -> None:
        self.post_message(DeploymentsLoadFailed(error=message.error))

    def _update_billing_panel(self) -> None:
        details = self.app.screen
        if isinstance(details, HomeDetailsScreen) and details._menu is self and details.is_mounted:
            details.update_billing()

    def _apply_auth_to_billing(
        self, provider: ComputeProvider, authenticated: bool
    ) -> None:
        """Fold a fresh auth verdict into that provider's billing row.

        Auth resolves separately from billing and usually first. A provider
        that cannot be read is named as unconfigured straight away, rather than
        waiting on a request that can only come back refused and then reporting
        the refusal as though the provider were broken.
        """
        if authenticated:
            self._refresh_billing_panels()
            return
        if self._provider_billing[provider].status is BillingStatus.READY:
            # A reading already in hand outlives a later auth wobble.
            return
        self._provider_billing[provider] = ProviderBilling.unconfigured(provider)
        self._update_billing_panel()

    def _refresh_provider_billing(self) -> None:
        """Read every provider's billing in one pass off the UI thread."""
        if self._billing_refresh_inflight:
            return
        self._billing_refresh_inflight = True
        # Auth is resolved on the UI thread where the cached status lives; the
        # loaders take it as a fact so an unauthenticated provider names its
        # setup command instead of reaching the network to be refused.
        modal_authenticated = (
            None if self._modal_auth_status is None else self._modal_auth_status.authenticated
        )
        prime_authenticated = (
            None if self._prime_auth_status is None else self._prime_auth_status.authenticated
        )
        if self._readiness is not None:
            auth = {
                row.provider: row.verified
                for row in self._readiness
                if row.stage not in (ProviderReadinessStage.CREDENTIALS_PRESENT, ProviderReadinessStage.UNREACHABLE)
            }
            modal_authenticated = auth.get(ComputeProvider.MODAL)
            prime_authenticated = auth.get(ComputeProvider.PRIME)
        self.run_worker(
            lambda: self._run_load_provider_billing(
                modal_authenticated=modal_authenticated,
                prime_authenticated=prime_authenticated,
            ),
            name="main-menu-billing-worker",
            thread=True,
        )

    def _run_load_provider_billing(
        self,
        *,
        modal_authenticated: bool | None,
        prime_authenticated: bool | None,
    ) -> None:
        poster = getattr(self, "post_message", None)
        if poster is None:
            return
        loaders = {
            ComputeProvider.MODAL: lambda: load_modal_billing(
                authenticated=modal_authenticated
            ),
            ComputeProvider.PRIME: lambda: load_prime_billing(
                authenticated=prime_authenticated
            ),
            ComputeProvider.VAST: load_vast_billing,
        }
        # One provider's slow API must not hold up the two that already
        # answered, so each row is posted as it lands.
        with ThreadPoolExecutor(max_workers=len(loaders)) as pool:
            futures = {
                pool.submit(loader): provider for provider, loader in loaders.items()
            }
            for future in as_completed(futures):
                provider = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # pragma: no cover - loaders are total
                    row = ProviderBilling.failed(provider, str(exc))
                poster(ProviderBillingLoaded(row=row))
        poster(ProviderBillingFinished())

    def on_deployments_loaded(self, message: DeploymentsLoaded) -> None:
        self._status_refresh_inflight = False
        self._runtime_rows = [replace(row) for row in message.rows]
        self._runtime_rows_fingerprint = self._runtime_fingerprint(message.rows)
        self._runtime_rows_cached_at = time.monotonic()
        self._show_deployments(message.rows)

    def _show_deployments(self, rows: list[EndpointInfo]) -> None:
        self._visible_fleet = [row for row in rows if _should_show_in_panel(row.state)]
        self._fleet_detail_error = None
        self._update_fleet_summary()
        self._update_fleet_details()

    def _update_fleet_details(self) -> None:
        details = self.app.screen
        if isinstance(details, HomeDetailsScreen) and details._menu is self and details.is_mounted:
            details.update_fleet()

    def _update_fleet_summary(self) -> None:
        """Keep the compact always-visible fleet line in step with the panel."""
        from ..fleet_status import fleet_summary_line

        try:
            summary = self.query_one("#fleet-summary-line", Static)
        except Exception:
            return
        current = self._visible_fleet
        unavailable = self._fleet_discovery.unavailable if self._fleet_discovery else ()
        if unavailable:
            providers = {listing.provider for listing in unavailable}
            names = ", ".join(listing.provider.display_name for listing in unavailable)
            confirmed = [row for row in current if row.provider not in providers]
            state = fleet_summary_line(confirmed) if confirmed else "Fleet status unknown"
            summary.update(f"{state} · [$warning]{names} unavailable[/]")
            return
        if not current and self._status_refresh_inflight:
            summary.update("[dim]Checking fleet...[/dim]")
            return
        if not current and self._fleet_discovery is not None and not self._fleet_discovery.listings:
            summary.update("[dim]Fleet status unknown · No provider listing available[/dim]")
            return
        if not current:
            summary.update("[dim]No active endpoints · Deploy to start[/dim]")
            return
        summary.update(fleet_summary_line(current))

    @staticmethod
    def _runtime_fingerprint(rows: list[EndpointInfo]) -> tuple[tuple[object, ...], ...]:
        return tuple(
            sorted(
                (
                    row.provider.value,
                    row.backend.value if row.backend is not None else "",
                    row.name or "",
                    row.app_id or "",
                    row.state or "",
                    row.web_url or "",
                    row.endpoint_api_key or "",
                )
                for row in rows
            )
        )

    def on_deployments_load_failed(self, message: DeploymentsLoadFailed) -> None:
        self._status_refresh_inflight = False
        self._fleet_detail_error = message.error
        self._update_fleet_details()
        self.query_one("#fleet-summary-line", Static).update("[$warning]Fleet status unknown[/]")

    def on_provider_billing_loaded(self, message: ProviderBillingLoaded) -> None:
        self._provider_billing[message.row.provider] = message.row
        self._update_billing_panel()

    def on_provider_billing_finished(self, _: ProviderBillingFinished) -> None:
        self._billing_refresh_inflight = False

    def on_storage_loaded(self, message: StorageLoaded) -> None:
        self._storage_snapshot = message.snapshot
        self._update_billing_panel()

    def on_storage_failed(self, _: StorageFailed) -> None:
        return

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        option_id = event.option.id
        if option_id == "deploy":
            self.app.action_push_deploy()
        elif option_id == "relaunch":
            self.app.action_push_relaunch()
        elif option_id == "custom-deploy":
            self.app.action_push_custom_deploy()
        elif option_id == "manage":
            self.app.action_push_manage()
        elif option_id == "storage":
            self.app.action_push_storage()
        elif option_id == "settings":
            self.app.action_push_settings()

    def action_select_deploy(self) -> None:
        self.app.action_push_deploy()

    def action_select_relaunch(self) -> None:
        self.app.action_push_relaunch()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "select_relaunch" and self._last_launch is None:
            return False
        return super().check_action(action, parameters)

    def action_select_custom_deploy(self) -> None:
        self.app.action_push_custom_deploy()

    def action_select_manage(self) -> None:
        self.app.action_push_manage()

    def action_select_storage(self) -> None:
        self.app.action_push_storage()

    def action_select_settings(self) -> None:
        self.app.action_push_settings()


class HomeDetailsScreen(CopyEnabledScreen):
    """Fleet, billing, and connection status from Home."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("m", "open_manage", "Manage", show=True),
    ]

    def __init__(self, menu: MainMenuScreen) -> None:
        super().__init__()
        self._menu = menu

    def compose(self) -> ComposeResult:
        from textual.containers import VerticalScroll

        from ..widgets.fitted_footer import FittedFooter

        with VerticalScroll(classes="screen-scroll"):
            version = f"v{self._menu.version}" if self._menu.version else ""
            yield Static(screen_title("Details", version))
            yield Static("[bold]Fleet[/bold]", classes="settings-section")
            yield Static("[dim]Loading fleet...[/dim]", id="home-details-fleet")
            yield Static("[bold]Billing[/bold]", classes="settings-section")
            yield Static("[dim]Loading billing...[/dim]", id="home-details-billing")
            yield Static("[bold]Connections[/bold]", classes="settings-section")
            yield Static("[dim]Loading connections...[/dim]", id="home-details-connections")
        yield FittedFooter()

    def on_mount(self) -> None:
        self.update_fleet()
        self.update_billing()
        self.update_connections()
        menu = self._menu
        if not menu._last_billing_refresh_at or (
            time.monotonic() - menu._last_billing_refresh_at >= menu._BILLING_REFRESH_INTERVAL_SECONDS
        ):
            menu._refresh_billing_panels()

    def on_screen_resume(self, _: events.ScreenResume) -> None:
        if self.is_mounted:
            self.update_fleet()
            self.update_billing()
            self.update_connections()

    def update_fleet(self) -> None:
        menu = self._menu
        if menu._fleet_detail_error:
            status = f"[$warning]Fleet status unavailable.[/]\n{escape(menu._fleet_detail_error)}"
        elif menu._status_refresh_inflight and not menu._visible_fleet:
            status = "[dim]Checking fleet…[/]"
        else:
            status = _render_deployment_status(
                menu._visible_fleet, username=menu.username, discovery=menu._fleet_discovery,
            )
        self.query_one("#home-details-fleet", Static).update(status)

    def update_connections(self) -> None:
        menu = self._menu
        self.query_one("#home-details-connections", Static).update(
            _render_auth_status_block(
                username=menu.username,
                modal_status=menu._modal_auth_status,
                hf_status=menu._hf_auth_status,
                prime_status=menu._prime_auth_status,
                aai_status=menu._aai_auth_status,
                readiness=menu._readiness,
            )
        )

    def update_billing(self) -> None:
        """Refresh billing as provider results arrive."""
        menu = self._menu
        self.query_one("#home-details-billing", Static).update(
            render_provider_billing(
                [menu._provider_billing[provider] for provider in PROVIDER_BILLING_ORDER],
                storage_snapshot=menu._storage_snapshot,
            )
        )

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def action_open_manage(self) -> None:
        self.app.pop_screen()
        self.app.action_push_manage()
