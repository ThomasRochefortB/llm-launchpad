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
from textual.containers import Center, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.timer import Timer
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from ...core.backend import ModalBackend
from ...core.compute_availability import display_gpu_type
from ...core.last_launch import LastLaunch, load_last_launch
from ...core.serving_metrics import default_tracker, usage_key
from ...core.hf_auth import HuggingFaceAuthStatus, get_huggingface_auth_status
from ...core.modal_auth import ModalAuthStatus, get_modal_auth_status
from ...core.prime_auth import PrimeAuthStatus, get_prime_auth_status
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
from ...core.deploy_log_summary import SUMMARY_SPINNER_FRAMES
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
from ..visual import screen_title

# A letter-spaced wordmark rather than figlet art: one row instead of seven,
# so the menu sits near the top of the column instead of under a block of
# ASCII, and it reads the same at every width that shows it.
BANNER = "[bold $primary]L L M[/]   [bold]L A U N C H P A D[/]"

_STATUS_PLACEHOLDER = "Refreshing deployment status…"
_SPINNER_INTERVAL_SECONDS = 0.1

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
    ) -> None:
        super().__init__()
        self.info = info
        self.profiles = profiles


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
) -> str:
    """Render one line per credential; ``spinner`` animates checks in flight."""
    lines: list[str] = [_render_modal_auth_status(modal_status)]
    lines.append(_render_prime_auth_status(prime_status))
    lines.append(_render_vast_auth_status())
    lines.append(_render_hf_auth_status(hf_status))
    lines.append(_render_artificial_analysis_auth_status(aai_status))
    if spinner:
        lines = [_with_spinner(line, spinner) for line in lines]
    return "\n".join(lines)


_CHECKING_SUFFIX = "...[/dim]"

# Compute providers get a line of their own on Home when they need attention;
# the optional services only change colour there, since the Details view
# carries their full text.
_COMPUTE_CREDENTIALS = frozenset({"Modal", "Prime Intellect", "Vast.ai"})


def _render_auth_status_compact(
    modal_status: ModalAuthStatus | None = None,
    hf_status: HuggingFaceAuthStatus | None = None,
    prime_status: PrimeAuthStatus | None = None,
    aai_status: ArtificialAnalysisAuthStatus | None = None,
    spinner: str = "",
) -> str:
    """One line naming every credential, coloured by state, plus fixes.

    Five full sentences -- mostly "Checking ... auth..." or "authenticated"
    -- took five rows of the home screen to say "all fine". The marker line
    says the same at a glance; a compute provider that needs attention still
    gets its own line with the command that fixes it.
    """
    entries = (
        (MODAL_MARKER, "Modal", _render_modal_auth_status(modal_status)),
        (PRIME_MARKER, "Prime Intellect", _render_prime_auth_status(prime_status)),
        (VAST_MARKER, "Vast.ai", _render_vast_auth_status()),
        (HUGGINGFACE_MARKER, "Hugging Face", _render_hf_auth_status(hf_status)),
        (
            ARTIFICIAL_ANALYSIS_MARKER,
            "Artificial Analysis",
            _render_artificial_analysis_auth_status(aai_status),
        ),
    )
    chips: list[str] = []
    problems: list[str] = []
    for marker, name, line in entries:
        label = f"{marker} {name}"
        if line.startswith("[$success]"):
            chips.append(f"[$success]{label}[/]")
        elif line.endswith(_CHECKING_SUFFIX):
            frame = f" [$primary]{spinner}[/]" if spinner else ""
            chips.append(f"[dim]{label}[/]{frame}")
        elif line.startswith(("[$warning]", "[$error]")):
            style = "$error" if line.startswith("[$error]") else "$warning"
            chips.append(f"[{style}]{label}[/]")
            if name in _COMPUTE_CREDENTIALS:
                problems.append(line)
        else:
            chips.append(f"[dim]{label}[/]")
    # Compute providers on one line, optional services on the next: each fits
    # 40 columns, where a single line wrapped mid-name ("Artificial /
    # Analysis") at 80.
    compute = "  ".join(chips[:3])
    services = "  ".join(chips[3:])
    return "\n".join([compute, services, *problems])


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
        import requests  # type: ignore
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
    return [fleet_summary_line(rows), f"[dim]{' | '.join(backend_parts)}[/dim]", ""]


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
            f"[dim]{escape(backend_name)} · {row.provider.value}[/dim]\n"
            f"  {deployment_and_health_line(row)}"
        )
        traffic_line = _serving_panel_line(row)
        if traffic_line:
            app_lines.append(traffic_line)
        passive_line = _passive_metrics_line(row)
        if passive_line:
            app_lines.append(passive_line)
        resource_label = _PROVIDER_RESOURCE_LABELS.get(
            row.provider, f"{row.provider.display_name} deployment"
        )
        modal_app_line = f"[dim]{resource_label}:[/dim] {escape(row.name or '')}"
        if (row.app_id or "").strip():
            modal_app_line += f" [dim]({escape(row.app_id)})[/dim]"
        app_lines.append(modal_app_line)

        model_id, display_name = endpoint_model_summary(row)
        app_lines.append(f"[dim]Display name:[/dim] {escape(display_name or '')}")
        app_lines.append(f"[dim]Model ID:[/dim] {escape(model_id or '')}")

        base_url, _was_derived = resolve_openai_base_url(row, username=username)
        show_connection = _state_bucket(row.state) == "healthy" or bool((row.web_url or "").strip())
        if base_url:
            base_root = base_url.rstrip("/")
            if base_root.endswith("/v1"):
                host_url = base_root[: -len("/v1")] or base_root
            else:
                host_url = base_root

            if show_connection:
                app_lines.append(PANEL_SEPARATOR)
                wrapped_url_lines = _wrap_url_for_panel(host_url)
                if wrapped_url_lines:
                    app_lines.append(f"  [dim]Base URL:[/dim] {escape(wrapped_url_lines[0])}")
                    for line in wrapped_url_lines[1:]:
                        app_lines.append(f"    {escape(line)}")
                key_status = "stored locally" if row.endpoint_api_key else ""
                app_lines.append(f"  [dim]API key[/dim] {key_status}")
            else:
                app_lines.append("[dim]OpenAI URL will be available once the app is serving traffic.[/dim]")
        else:
            if _state_bucket(row.state) in {"deploying", "queued"}:
                app_lines.append("[dim]OpenAI URL unavailable while the app is still starting.[/dim]")
            else:
                app_lines.append("[dim]OpenAI URL unavailable (provider has no web URL yet).[/dim]")

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
    """One relaunch row per label tier, fullest first."""
    shape = f"{display_gpu_type(last.gpu_type)} x{last.gpu_count}" if last.gpu_type else ""
    name = clip(last.display_name, 30)
    return (
        ("relaunch", f"  Relaunch           {name}{f' · {shape}' if shape else ''}"),
        ("relaunch", f"  Relaunch           {name}"),
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
        Binding("c", "select_custom_deploy", "Advanced", show=True),
        Binding("m", "select_manage", "Manage", show=True),
        Binding("t", "select_storage", "Storage", show=True),
        Binding("s", "select_settings", "Settings", show=True),
        Binding("i", "toggle_details", "Details", show=True),
        Binding("escape", "close_details", "Close details", show=False),
    ]
    _ENDPOINT_REFRESH_INTERVAL_SECONDS = 20.0
    _BILLING_REFRESH_INTERVAL_SECONDS = 300.0
    _SECONDARY_REFRESH_DELAY_SECONDS = 0.5
    _RUNTIME_STATUS_CACHE_TTL_SECONDS = 20.0

    def __init__(self, username: str = "", version: str = "") -> None:
        super().__init__()
        self.username = username
        self.version = version
        self._modal_auth_status: ModalAuthStatus | None = None
        self._prime_auth_status: PrimeAuthStatus | None = None
        self._hf_auth_status: HuggingFaceAuthStatus | None = None
        self._aai_auth_status: ArtificialAnalysisAuthStatus | None = None
        self._hf_auth_refresh_inflight = False
        self._aai_auth_refresh_inflight = False
        self._modal_auth_refresh_inflight = False
        self._prime_auth_refresh_inflight = False
        self._quick_deploy_catalog_refresh_inflight = False
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
        self._billing_refresh_timer: Timer | None = None
        self._secondary_refresh_timer: Timer | None = None
        self._runtime_rows: list[EndpointInfo] = []
        self._runtime_rows_fingerprint: tuple[tuple[object, ...], ...] = ()
        self._runtime_rows_cached_at = 0.0
        self._fleet_discovery: FleetDiscovery | None = None
        self._action_labels: _ActionLabels | None = None
        self._action_label_ceiling = 0
        self._last_launch = load_last_launch()
        self._spinner_index = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="main-menu-root"):
            with Center(id="main-menu-center"):
                with Horizontal(id="main-menu-layout"):
                    with VerticalScroll(id="main-menu-primary"):
                        yield Static(BANNER, id="banner-text")
                        version_text = f"v{self.version}  " if self.version else ""
                        yield Static(
                            "[bold]LLM Launchpad[/bold]\n"
                            f"[dim]{version_text}Deploy and manage inference endpoints[/dim]",
                            id="compact-menu-header",
                        )
                        # Wraps rather than stopping mid-word: the narrow left
                        # column used to cut this to "... Vast.ai LLM".
                        yield Static(
                            f"[dim]{version_text.strip()}[/dim]",
                            classes="centered main-menu-version",
                        )
                        yield Static("", classes="decorative-spacer")
                        yield OptionList(
                            *(
                                Option(label, id=option_id)
                                for option_id, label in _action_label_tiers(self._last_launch)[0]
                            ),
                            id="action-list",
                        )
                        yield Static(
                            "[dim]↑/↓ select · enter open · i details[/dim]",
                            id="compact-menu-help",
                        )
                        yield Static("", id="fleet-summary-line")
                    with Vertical(id="main-menu-side-column"):
                        status_panel = Vertical(id="deployment-status-panel")
                        status_panel.border_title = "Deployment Status"
                        with status_panel:
                            yield Static(self._status_placeholder(), id="deployment-status-body")
                        billing_panel = Vertical(id="billing-report-panel")
                        billing_panel.border_title = "Provider Billing"
                        with billing_panel:
                            # First paint already names every provider, so the
                            # panel does not change shape as readings land.
                            yield Static(
                                render_provider_billing(
                                    [
                                        self._provider_billing[provider]
                                        for provider in PROVIDER_BILLING_ORDER
                                    ]
                                ),
                                id="billing-report-body",
                            )
            yield Static(
                _render_auth_status_compact(),
                id="auth-status-block",
            )
        yield FittedFooter()

    def on_mount(self) -> None:
        """Focus the option list so arrow-key navigation works immediately."""
        action_list = self.query_one("#action-list", OptionList)
        if action_list.option_count > 0:
            action_list.action_first()
        action_list.focus()
        self._refresh_modal_auth_status()
        self._refresh_prime_auth_status()
        self._refresh_hf_auth_status()
        self._refresh_aai_auth_status()
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
        self._billing_refresh_timer = self.set_interval(
            self._BILLING_REFRESH_INTERVAL_SECONDS,
            self._refresh_billing_panels,
            name="main-menu-billing-refresh",
        )
        self.set_interval(
            _SPINNER_INTERVAL_SECONDS,
            self._tick_loading,
            name="main-menu-loading-spinner",
        )

    def _loading_frame(self) -> str:
        return SUMMARY_SPINNER_FRAMES[self._spinner_index]

    def _status_placeholder(self) -> str:
        return f"[$primary]{self._loading_frame()}[/] [dim]{_STATUS_PLACEHOLDER}[/dim]"

    def _tick_loading(self) -> None:
        """Turn the spinner on every loading placeholder that is still showing.

        Static "checking..." text reads the same whether a request is in flight
        or has hung; motion says work is happening. Nothing is re-rendered once
        every placeholder has been replaced by a result.
        """
        try:
            body = self.query_one("#deployment-status-body", Static)
        except Exception:
            return
        status_loading = _STATUS_PLACEHOLDER in str(body.content)
        billing_loading = any(
            row.status is BillingStatus.LOADING for row in self._provider_billing.values()
        )
        auth_loading = None in (
            self._modal_auth_status,
            self._prime_auth_status,
            self._hf_auth_status,
            self._aai_auth_status,
        )
        if not (status_loading or billing_loading or auth_loading):
            return
        self._spinner_index = (self._spinner_index + 1) % len(SUMMARY_SPINNER_FRAMES)
        if status_loading:
            body.update(self._status_placeholder())
        if billing_loading:
            self._update_billing_panel()
        if auth_loading:
            self._render_connection_widgets()

    def on_screen_suspend(self, _: events.ScreenSuspend) -> None:
        self._was_suspended = True
        self._pause_refresh_timers()

    def on_screen_resume(self, _: events.ScreenResume) -> None:
        """Refresh fleet and billing when returning from a nested flow."""
        if not self._was_suspended:
            return
        self._was_suspended = False
        self._resume_refresh_timers()
        self._refresh_panels()
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
        elif (
            time.monotonic() - self._last_billing_refresh_at
            >= self._BILLING_REFRESH_INTERVAL_SECONDS
        ):
            self._refresh_billing_panels()

    def _pause_refresh_timers(self) -> None:
        for timer in (
            self._secondary_refresh_timer,
            self._endpoint_refresh_timer,
            self._billing_refresh_timer,
        ):
            if timer is not None:
                timer.pause()

    def _resume_refresh_timers(self) -> None:
        for timer in (
            self._secondary_refresh_timer,
            self._endpoint_refresh_timer,
            self._billing_refresh_timer,
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
        """Dismiss the narrow detail drawer once both panels fit again."""
        _ = previous
        self._refresh_action_labels(profile)
        self._refresh_menu_help(profile)
        if not profile.narrow and not profile.short:
            self.remove_class("show-secondary-panel")

    def _menu_help_text(self, profile: ViewportProfile | None = None) -> str:
        """Return the menu help line; the details hint only fits narrow layouts."""
        try:
            active_profile = profile or self.viewport_profile
        except Exception:
            active_profile = None
        if active_profile is not None and (active_profile.narrow or active_profile.short):
            return "[dim]↑/↓ select · enter open · i details[/dim]"
        return "[dim]↑/↓ select · enter open[/dim]"

    def _refresh_menu_help(self, profile: ViewportProfile | None = None) -> None:
        try:
            self.query_one("#compact-menu-help", Static).update(self._menu_help_text(profile))
        except Exception:
            return

    def _refresh_action_labels(self, profile: ViewportProfile) -> None:
        """Use concise action records when descriptive columns no longer fit."""
        # A compact terminal deliberately drops the description column for
        # concise names; measurement may shorten labels further but must never
        # talk that decision back up into a padded two-column grid.
        tiers = _action_label_tiers(self._last_launch)
        self._action_label_ceiling = len(tiers) - 1 if profile.compact else 0
        self._apply_action_labels(tiers[self._action_label_ceiling])
        # The terminal is not the width these labels have to fit. Whenever the
        # side column is showing, the list gets what is left of a layout capped
        # at 126 columns -- which is how "Cached models, pre-download, delete"
        # came to wrap, orphaning "delete" on its own line, on a 120-column
        # terminal that the breakpoints called wide.
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
        highlighted = action_list.highlighted_option
        selected_id = str(highlighted.id) if highlighted is not None else "deploy"
        self._action_labels = labels
        action_list.set_options(
            [Option(label, id=option_id) for option_id, label in labels]
        )
        for index, (option_id, _) in enumerate(labels):
            if option_id == selected_id:
                action_list.highlighted = index
                break

    def action_toggle_details(self) -> None:
        """Open fleet, billing and connection detail as a full-width view."""
        self.app.push_screen(HomeDetailsScreen(self))  # type: ignore[attr-defined]

    def action_close_details(self) -> None:
        """Close the narrow details drawer without changing screens."""
        self.remove_class("show-secondary-panel")

    def _refresh_quick_deploy_catalog(self) -> None:
        if self._quick_deploy_catalog_refresh_inflight:
            return
        self._quick_deploy_catalog_refresh_inflight = True
        if self._activate_warm_quick_deploy_catalog():
            # A fresh disk snapshot is already live; still refresh in the
            # background so pricing/benchmarks stay current, but the picker
            # never sits in the "Building…" empty state meanwhile.
            self.run_worker(
                self._run_refresh_quick_deploy_catalog,
                name="main-menu-quick-deploy-catalog-worker",
                thread=True,
            )
            return
        self.run_worker(
            self._run_refresh_quick_deploy_catalog,
            name="main-menu-quick-deploy-catalog-worker",
            thread=True,
        )

    def _activate_warm_quick_deploy_catalog(self) -> bool:
        """Activate a fresh disk snapshot so Deploy opens instantly.

        Returns True when a fresh snapshot was activated (a background
        refresh is still worthwhile). Stale snapshots are also activated
        so the picker has content, but return False so the caller treats
        the refresh as the load-bearing path.
        """

        try:
            cached = load_cached_quick_deploy_catalog()
        except Exception:
            return False
        if cached is None:
            return False
        info, profiles = cached
        try:
            activate_quick_deploy_catalog(info, profiles)
        except ValueError:
            return False
        try:
            notifier = getattr(self.app, "quick_deploy_catalog_updated", None)
        except Exception:
            notifier = None
        if callable(notifier):
            try:
                notifier()
            except Exception:
                pass
        return is_fresh_cached_quick_deploy_catalog(info)

    def ensure_quick_deploy_catalog_refresh(self) -> None:
        """Start the live model-catalog refresh before opening the picker."""
        self._refresh_quick_deploy_catalog()

    def _run_refresh_quick_deploy_catalog(self) -> None:
        try:
            info, profiles = build_live_quick_deploy_catalog()
        except Exception as exc:
            self.post_message(QuickDeployCatalogLoadFailed(error=str(exc)))
            return
        self.post_message(QuickDeployCatalogLoaded(info=info, profiles=profiles))
        # MTP probes are the slowest per-model fetch and only feed the
        # draft-model toggle; attach them as a trailing update so the
        # picker stays usable while they resolve. They are written back to
        # the snapshot so the next launch opens with MTP already resolved
        # instead of reprobing every repository.
        try:
            upgraded = attach_quick_deploy_mtp_recommendations(profiles, info=info)
        except Exception:
            return
        if upgraded != tuple(profiles):
            self.post_message(QuickDeployCatalogLoaded(info=info, profiles=upgraded))

    def on_quick_deploy_catalog_loaded(
        self,
        message: QuickDeployCatalogLoaded,
    ) -> None:
        self._quick_deploy_catalog_refresh_inflight = False
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
        """Refresh both the full auth block and the compact summary line."""
        try:
            self.query_one("#auth-status-block", Static).update(
                _render_auth_status_compact(
                    modal_status=self._modal_auth_status,
                    hf_status=self._hf_auth_status,
                    prime_status=self._prime_auth_status,
                    aai_status=self._aai_auth_status,
                    spinner=self._loading_frame(),
                )
            )
        except Exception:
            pass
        # The compact line is fleet-first once rows arrive; connection state
        # until then. _update_fleet_summary owns it afterwards.
        # The marker line above already says which providers are connected;
        # the compact line waits for fleet rows rather than repeating it.

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
        # Storage first: the cached snapshot is a local read, and it is part of
        # the Modal row the billing pass is about to paint.
        self._refresh_storage_estimate()
        self._refresh_billing_panels()

    def _refresh_billing_panels(self) -> None:
        if not self._is_active_screen():
            return
        self._last_billing_refresh_at = time.monotonic()
        self._refresh_provider_billing()

    def _is_active_screen(self) -> bool:
        """Return whether this screen is the visible top of the app stack."""
        try:
            return self.app.screen is self
        except Exception:
            return False

    def _refresh_storage_estimate(self) -> None:
        cached_storage_snapshot = getattr(self.app, "cached_storage_snapshot", None)
        if callable(cached_storage_snapshot):
            self._storage_snapshot = cached_storage_snapshot()
        refresh_storage = getattr(self.app, "begin_storage_refresh", None)
        if callable(refresh_storage):
            refresh_storage(self, force=False)

    def _refresh_deployment_status(self) -> None:
        if self._status_refresh_inflight:
            return
        self._status_refresh_inflight = True
        self.query_one("#deployment-status-body", Static).update(self._status_placeholder())
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
            list_instances = getattr(self.app, "list_instances", None)  # type: ignore[attr-defined]
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
            thread=True,
            exclusive=True,
        )

    def _run_annotate_deployments(self, rows: list[EndpointInfo]) -> None:
        _annotate_runtime_statuses(rows, self.username)
        self.post_message(DeploymentsLoaded(rows=rows))

    def on_endpoints_failed(self, message: EndpointsFailed) -> None:
        self.post_message(DeploymentsLoadFailed(error=message.error))

    def _update_billing_panel(self) -> None:
        self.query_one("#billing-report-body", Static).update(
            render_provider_billing(
                [self._provider_billing[provider] for provider in PROVIDER_BILLING_ORDER],
                storage_snapshot=self._storage_snapshot,
                spinner=self._loading_frame(),
            )
        )

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
            if self._secondary_refresh_started:
                self._refresh_provider_billing()
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
        visible_rows = [row for row in rows if _should_show_in_panel(row.state)]
        self.query_one("#deployment-status-body", Static).update(
            _render_deployment_status(
                visible_rows,
                username=self.username,
                discovery=self._fleet_discovery,
            )
        )
        self._update_fleet_summary(visible_rows)

    def _update_fleet_summary(self, rows: list[EndpointInfo] | None = None) -> None:
        """Keep the compact always-visible fleet line in step with the panel."""
        from ..fleet_status import fleet_summary_line

        try:
            summary = self.query_one("#fleet-summary-line", Static)
        except Exception:
            return
        current = rows if rows is not None else [
            row for row in self._runtime_rows if _should_show_in_panel(row.state)
        ]
        if not current and self._status_refresh_inflight:
            summary.update("[dim]Checking fleet...[/dim]")
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
        self.query_one("#deployment-status-body", Static).update(
            "[$warning]Status unavailable.[/$warning]\n"
            f"[dim]{clip(message.error, 80)}[/dim]"
        )
        self._update_fleet_summary([])

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
            self.app.action_push_deploy()  # type: ignore[attr-defined]
        elif option_id == "relaunch":
            self.app.action_push_relaunch()  # type: ignore[attr-defined]
        elif option_id == "custom-deploy":
            self.app.action_push_custom_deploy()  # type: ignore[attr-defined]
        elif option_id == "manage":
            self.app.action_push_manage()  # type: ignore[attr-defined]
        elif option_id == "storage":
            self.app.action_push_storage()  # type: ignore[attr-defined]
        elif option_id == "settings":
            self.app.action_push_settings()  # type: ignore[attr-defined]

    def action_select_deploy(self) -> None:
        self.app.action_push_deploy()  # type: ignore[attr-defined]

    def action_select_relaunch(self) -> None:
        self.app.action_push_relaunch()  # type: ignore[attr-defined]

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "select_relaunch" and self._last_launch is None:
            return False
        return super().check_action(action, parameters)

    def action_select_custom_deploy(self) -> None:
        self.app.action_push_custom_deploy()  # type: ignore[attr-defined]

    def action_select_manage(self) -> None:
        self.app.action_push_manage()  # type: ignore[attr-defined]

    def action_select_storage(self) -> None:
        self.app.action_push_storage()  # type: ignore[attr-defined]

    def action_select_settings(self) -> None:
        self.app.action_push_settings()  # type: ignore[attr-defined]


class HomeDetailsScreen(CopyEnabledScreen):
    """Full-width Fleet / Billing / Connections view for compact terminals.

    Replaces the narrow overlay drawer, which covered the menu it was meant
    to accompany and left the fleet panel too short to read. Escape returns
    to the menu with its selection intact.
    """

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
            yield Static(screen_title("Details"))
            yield Static("[bold]Fleet[/bold]", classes="settings-section")
            yield Static("[dim]Loading fleet...[/dim]", id="home-details-fleet")
            yield Static("[bold]Billing[/bold]", classes="settings-section")
            yield Static("[dim]Loading billing...[/dim]", id="home-details-billing")
            yield Static("[bold]Connections[/bold]", classes="settings-section")
            yield Static("[dim]Loading connections...[/dim]", id="home-details-connections")
        yield FittedFooter()

    def on_mount(self) -> None:
        menu = self._menu
        try:
            fleet = menu.query_one("#deployment-status-body", Static).content
            self.query_one("#home-details-fleet", Static).update(fleet)
        except Exception:
            pass
        try:
            billing = menu.query_one("#billing-report-body", Static).content
            self.query_one("#home-details-billing", Static).update(billing)
        except Exception:
            pass
        try:
            # Home shows the compact marker line; this view is where each
            # credential's full sentence belongs.
            self.query_one("#home-details-connections", Static).update(
                _render_auth_status_block(
                    username=menu.username,
                    modal_status=menu._modal_auth_status,
                    hf_status=menu._hf_auth_status,
                    prime_status=menu._prime_auth_status,
                    aai_status=menu._aai_auth_status,
                )
            )
        except Exception:
            pass

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def action_open_manage(self) -> None:
        self.app.pop_screen()
        self.app.action_push_manage()  # type: ignore[attr-defined]
