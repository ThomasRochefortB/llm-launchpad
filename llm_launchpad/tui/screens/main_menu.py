"""Main menu screen: deploy, manage endpoints, settings.

Inspired by the Codex TUI main screen with a prominent banner,
auth status line, and keyboard-navigable option list.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import replace
import time
from typing import Any, Literal

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.timer import Timer
from textual.widgets import Footer, OptionList, Static
from textual.widgets.option_list import Option

from ...core.backend import ModalBackend
from ...core.hf_auth import HuggingFaceAuthStatus, get_huggingface_auth_status
from ...core.modal_auth import ModalAuthStatus, get_modal_auth_status
from ...core.prime_auth import PrimeAuthStatus, get_prime_auth_status
from ...core.prime_backend import PrimeBackend
from ...core.quick_deploy import (
    QuickDeployCatalogInfo,
    QuickDeployProfile,
    activate_quick_deploy_catalog,
    record_quick_deploy_catalog_failure,
)
from ...core.quick_deploy_refresh import (
    ArtificialAnalysisAuthStatus,
    attach_quick_deploy_mtp_recommendations,
    build_live_quick_deploy_catalog,
    get_artificial_analysis_auth_status,
    is_fresh_cached_quick_deploy_catalog,
    load_cached_quick_deploy_catalog,
)
from ...core.storage_costs import (
    MODAL_VOLUME_FREE_TIER_GIB_MONTH,
    estimate_monthly_storage_cost,
)
from ...protocol.enums import BackendType, ComputeProvider
from ...protocol.models import EndpointInfo, StorageSnapshot
from ..connection import endpoint_model_summary, resolve_openai_base_url
from ..workers import EndpointsFailed, EndpointsLoaded, StorageFailed, StorageLoaded
from ..responsive import ViewportProfile
from .copy_enabled import CopyEnabledScreen

BANNER = r"""[bold #7bf168]
_     _     __  __
| |   | |   |  \/  |
| |   | |   | |\/| |
| |___| |___| |  | |
|_____|_____|_|  |_|
    LAUNCHPAD
[/]"""

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


class BillingReportLoaded(Message):
    """Main-menu billing report fetch completed."""

    def __init__(self, payload: Any, storage_snapshot: StorageSnapshot | None = None) -> None:
        super().__init__()
        self.payload = payload
        self.storage_snapshot = storage_snapshot


class BillingReportLoadFailed(Message):
    """Main-menu billing report fetch failed."""

    def __init__(self, error: str) -> None:
        super().__init__()
        self.error = error


class PrimeBillingReportLoaded(Message):
    """Main-menu Prime billing fetch completed."""

    def __init__(self, payload: Any) -> None:
        super().__init__()
        self.payload = payload


class PrimeBillingReportLoadFailed(Message):
    """Main-menu Prime billing fetch failed."""

    def __init__(self, error: str) -> None:
        super().__init__()
        self.error = error


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


def _clip(value: str, width: int) -> str:
    text = (value or "").strip()
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return f"{text[:width - 3]}..."


def _escape_markup(value: str) -> str:
    return (value or "").replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def _render_hf_auth_status(status: HuggingFaceAuthStatus | None = None) -> str:
    if status is None:
        return "[dim]🤗 Checking Hugging Face auth...[/dim]"
    if status.authenticated:
        return "[green]🤗 Hugging Face authenticated[/green]"
    if status.error:
        color = "red" if "invalid" in status.error.lower() else "yellow"
        detail = _escape_markup(_clip(status.error, 72))
        return f"[{color}]🤗 Hugging Face auth check failed: {detail}[/{color}]"
    return "[yellow]🤗 Hugging Face not authenticated (run: hf auth login)[/yellow]"


def _render_modal_auth_status(status: ModalAuthStatus | None = None) -> str:
    if status is None:
        return "[dim]▰ Checking Modal auth...[/dim]"
    if status.authenticated:
        return "[green]▰ Modal authenticated[/green]"
    if status.error:
        detail = _escape_markup(_clip(status.error, 72))
        return f"[yellow]▰ Modal auth check failed: {detail}[/yellow]"
    return "[yellow]▰ Modal not authenticated (run: modal setup)[/yellow]"


def _render_prime_auth_status(status: PrimeAuthStatus | None = None) -> str:
    if status is None:
        return "[dim]◆ Checking Prime Intellect auth...[/dim]"
    if status.authenticated:
        return "[green]◆ Prime Intellect authenticated[/green]"
    if status.error:
        detail = _escape_markup(_clip(status.error, 72))
        return f"[yellow]◆ Prime Intellect auth check failed: {detail}[/yellow]"
    return "[yellow]◆ Prime Intellect not authenticated (run: prime login)[/yellow]"


def _render_artificial_analysis_auth_status(
    status: ArtificialAnalysisAuthStatus | None = None,
) -> str:
    if status is None:
        return "[dim]◈ Checking Artificial Analysis auth...[/dim]"
    if status.authenticated:
        tier = f" ({_escape_markup(status.tier)} tier)" if status.tier else ""
        return f"[green]◈ Artificial Analysis authenticated{tier}[/green]"
    if status.error:
        color = "red" if "invalid" in status.error.casefold() else "yellow"
        detail = _escape_markup(_clip(status.error, 72))
        return f"[{color}]◈ Artificial Analysis auth check failed: {detail}[/{color}]"
    return (
        "[yellow]◈ Artificial Analysis not authenticated "
        "(run: llm-launchpad aai-auth login)[/yellow]"
    )


def _render_auth_status_block(
    username: str = "",
    modal_status: ModalAuthStatus | None = None,
    hf_status: HuggingFaceAuthStatus | None = None,
    prime_status: PrimeAuthStatus | None = None,
    aai_status: ArtificialAnalysisAuthStatus | None = None,
) -> str:
    lines: list[str] = [_render_modal_auth_status(modal_status)]
    lines.append(_render_prime_auth_status(prime_status))
    lines.append(_render_hf_auth_status(hf_status))
    lines.append(_render_artificial_analysis_auth_status(aai_status))
    return "\n".join(lines)


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
    if status in {"healthy", "in_progress", "error"}:
        return status
    return _runtime_bucket_from_modal_state(row.state)


def _style_runtime_bucket(bucket: str) -> str:
    normalized = (bucket or "").strip().lower()
    if normalized == "healthy":
        return "[green]healthy[/green]"
    if normalized == "in_progress":
        return "[yellow]in progress[/yellow]"
    if normalized == "error":
        return "[red]error[/red]"
    return f"[dim]{_escape_markup(normalized or 'unknown')}[/dim]"


def _probe_row_runtime_status(row: EndpointInfo, username: str) -> tuple[str, str | None]:
    modal_runtime = _runtime_bucket_from_modal_state(row.state)
    if modal_runtime != "healthy":
        return modal_runtime, None
    if row.backend not in {BackendType.VLLM, BackendType.LLAMACPP}:
        return "in_progress", "unknown backend"

    base_url, was_derived = resolve_openai_base_url(row, username=username)
    if not base_url:
        return "in_progress", "missing URL"

    try:
        import requests  # type: ignore
    except ImportError:
        return modal_runtime, "requests unavailable"

    base_root = base_url.rstrip("/")
    host_root = base_root[:-3] if base_root.endswith("/v1") else base_root
    probe_url = host_root.rstrip("/") + "/health"
    try:
        headers = (
            {"Authorization": f"Bearer {row.endpoint_api_key}"}
            if row.endpoint_api_key
            else None
        )
        response = requests.get(probe_url, headers=headers, timeout=2.5)
        if 200 <= response.status_code < 300:
            return "healthy", None

        if not was_derived and response.status_code in {401, 403, 404}:
            return "error", f"HTTP {response.status_code}"
        return "in_progress", f"HTTP {response.status_code}"
    except Exception as exc:
        return "in_progress", str(exc)


def _annotate_runtime_statuses(rows: list[EndpointInfo], username: str) -> None:
    candidates = [
        row
        for row in rows
        if _runtime_bucket_from_modal_state(row.state) == "healthy"
        and row.backend in {BackendType.VLLM, BackendType.LLAMACPP}
    ]
    for row in rows:
        if _runtime_bucket_from_modal_state(row.state) != "healthy":
            row.runtime_status = _runtime_bucket_from_modal_state(row.state)
            row.runtime_status_detail = None

    if not candidates:
        return

    max_workers = min(4, len(candidates))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_probe_row_runtime_status, row, username): row
            for row in candidates
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                status, detail = future.result()
            except Exception as exc:
                status, detail = "in_progress", str(exc)
            row.runtime_status = status
            row.runtime_status_detail = detail


def _backend_display_name(backend: BackendType | None) -> str:
    if backend == BackendType.VLLM:
        return "vLLM"
    if backend == BackendType.LLAMACPP:
        return "llama.cpp"
    return "unknown"


def _friendly_count_line(rows: list[EndpointInfo]) -> list[str]:
    runtime_counts = Counter(_runtime_bucket(row) for row in rows)
    backend_counts = Counter(
        row.backend.value if row.backend is not None else "unknown"
        for row in rows
    )
    healthy = runtime_counts.get("healthy", 0)
    in_progress = runtime_counts.get("in_progress", 0)
    errors = runtime_counts.get("error", 0)

    noun = "launchpad app" if len(rows) == 1 else "launchpad apps"
    chips = [f"[green]{healthy} healthy[/green]"]
    if in_progress:
        chips.append(f"[yellow]{in_progress} in progress[/yellow]")
    if errors:
        chips.append(f"[red]{errors} error[/red]")
    summary = f"[bold]{len(rows)} active {noun}[/bold]"
    if chips:
        summary += "  " + "  ".join(chips)

    backend_parts = []
    if backend_counts.get("vllm", 0):
        backend_parts.append(f"{backend_counts['vllm']} vLLM")
    if backend_counts.get("llamacpp", 0):
        backend_parts.append(f"{backend_counts['llamacpp']} llama.cpp")
    if not backend_parts:
        backend_parts.append(f"{len(rows)} launchpad")
    return [summary, f"[dim]{' | '.join(backend_parts)}[/dim]", ""]


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


def _render_deployment_status(rows: list[EndpointInfo], username: str = "") -> str:
    if not rows:
        return (
            "[bold]Fleet Pulse[/bold]\n"
            "[dim]No active launchpad apps.[/dim]"
        )

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
        instance = (row.instance_name or "").strip() or "default"
        backend_name = _backend_display_name(row.backend)
        app_lines.append(
            f"[bold]{_escape_markup(instance)}[/bold]  "
            f"[dim]{_escape_markup(backend_name)}[/dim]  {_style_runtime_bucket(_runtime_bucket(row))} "
            f"[dim]({row.provider.value}: {_escape_markup((row.state or 'unknown').strip().lower())})[/dim]"
        )
        resource_label = (
            "Modal app" if row.provider == ComputeProvider.MODAL else "Prime Intellect pod"
        )
        modal_app_line = f"[dim]{resource_label}:[/dim] {_escape_markup(row.name or '')}"
        if (row.app_id or "").strip():
            modal_app_line += f" [dim]({_escape_markup(row.app_id)})[/dim]"
        app_lines.append(modal_app_line)

        model_id, display_name = endpoint_model_summary(row)
        app_lines.append(f"[dim]Display name:[/dim] {_escape_markup(display_name or '')}")
        app_lines.append(f"[dim]Model ID:[/dim] {_escape_markup(model_id or '')}")

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
                    app_lines.append(f"  [dim]Base URL:[/dim] {_escape_markup(wrapped_url_lines[0])}")
                    for line in wrapped_url_lines[1:]:
                        app_lines.append(f"    {_escape_markup(line)}")
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

    return "\n".join(header_lines + app_lines)


def _format_money(value: float) -> str:
    return f"${value:,.2f}"


def _format_gib(value: float) -> str:
    if value >= 100:
        return f"{value:,.0f} GiB"
    if value >= 10:
        return f"{value:,.1f} GiB"
    return f"{value:,.2f} GiB"


def _format_free_tier(value_gib: float) -> str:
    if value_gib > 0 and value_gib % 1024 == 0:
        return f"{value_gib / 1024:,.0f} TiB"
    return _format_gib(value_gib)


def _storage_estimate_lines(snapshot: StorageSnapshot | None) -> list[str]:
    if snapshot is None:
        return []
    estimate = estimate_monthly_storage_cost(snapshot)
    return [
        f"[dim]Launchpad storage est.[/dim] {_format_money(estimate.estimated_monthly_cost_usd)}/mo",
        (
            f"[dim]{_format_gib(estimate.total_gib_month)} cached; "
            f"{_format_gib(estimate.billable_gib_month)} billable after "
            f"{_format_free_tier(MODAL_VOLUME_FREE_TIER_GIB_MONTH)} free[/dim]"
        ),
    ]


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace("$", "").replace(",", "")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _find_first_float(payload: Any, dotted_keys: list[str]) -> float | None:
    for dotted_key in dotted_keys:
        current = payload
        found = True
        for key in dotted_key.split("."):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                found = False
                break
        if not found:
            continue
        parsed = _coerce_float(current)
        if parsed is not None:
            return parsed
    return None


def _normalize_billing_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        if isinstance(payload.get("report"), dict):
            return payload["report"]
        if isinstance(payload.get("data"), dict):
            return payload["data"]
        return payload
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload
    return payload


def _render_billing_report(
    payload: Any,
    storage_snapshot: StorageSnapshot | None = None,
) -> str:
    normalized = _normalize_billing_payload(payload)
    if isinstance(normalized, list):
        rows = [row for row in normalized if isinstance(row, dict)]
        if not rows:
            lines = [
                "[bold]Workspace Spend[/bold]",
                "[dim]Current month spend[/dim]",
                "[dim]total[/dim] [bold]$0.00[/bold]",
            ]
            lines.extend(_storage_estimate_lines(storage_snapshot))
            return "\n".join(lines)

        total = 0.0
        has_total = False
        for row in rows:
            cost = _coerce_float(row.get("Cost"))
            if cost is None:
                cost = _coerce_float(row.get("cost"))
            if cost is not None:
                total += cost
                has_total = True

        lines = ["[bold]Workspace Spend[/bold]", "[dim]Current month spend[/dim]"]
        if has_total:
            lines.append(f"[dim]total[/dim] [bold]{_format_money(total)}[/bold]")
        else:
            lines.append("[dim]Total unavailable in report payload.[/dim]")
        lines.extend(_storage_estimate_lines(storage_snapshot))
        return "\n".join(lines)

    if not isinstance(normalized, dict):
        lines = [
            "[bold]Workspace Spend[/bold]",
            "[dim]Billing data unavailable.[/dim]",
            "[dim]Check `modal billing report --json`.[/dim]",
        ]
        lines.extend(_storage_estimate_lines(storage_snapshot))
        return "\n".join(lines)

    total = _find_first_float(
        normalized,
        [
            "summary.total_usd",
            "summary.cost_usd",
            "summary.spend_usd",
            "totals.total_usd",
            "totals.cost_usd",
            "total_usd",
            "cost_usd",
            "spend_usd",
        ],
    )
    gpu_cost = _find_first_float(
        normalized,
        [
            "summary.gpu_cost_usd",
            "totals.gpu_cost_usd",
            "gpu_cost_usd",
        ],
    )
    lines = ["[bold]Workspace Spend[/bold]", "[dim]Current month spend[/dim]"]
    if total is not None:
        lines.append(f"[dim]total[/dim] [bold]{_format_money(total)}[/bold]")
    else:
        lines.append("[dim]Total unavailable in report payload.[/dim]")
    if gpu_cost is not None:
        lines.append(f"[dim]gpu[/dim] {_format_money(gpu_cost)}")
    lines.extend(_storage_estimate_lines(storage_snapshot))
    return "\n".join(lines)


def _render_billing_load_error(error: str) -> str:
    return (
        "[bold]Workspace Spend[/bold]\n"
        "[yellow]Billing unavailable.[/yellow]\n"
        f"[dim]{_clip(_escape_markup(error), 80)}[/dim]"
    )


def _render_prime_billing_report(payload: Any) -> str:
    """Render one Prime wallet billing snapshot for the shared panel."""

    lines = ["[bold]Prime Intellect Wallet[/bold]"]
    if not isinstance(payload, dict):
        lines.append("[dim]Wallet data unavailable.[/dim]")
        lines.append("[dim]Check `prime wallet`.[/dim]")
        return "\n".join(lines)

    balance = _coerce_float(payload.get("balance_usd"))
    if balance is None:
        lines.append("[dim]Balance unavailable in wallet payload.[/dim]")
    else:
        lines.append(f"[dim]balance[/dim] [bold]{_format_money(balance)}[/bold]")

    totals: dict[str, float] = {}
    rows = payload.get("recent_billings")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            amount = _coerce_float(row.get("amount_usd"))
            if amount is None:
                continue
            resource = str(row.get("resource_type") or "other").strip().casefold() or "other"
            totals[resource] = totals.get(resource, 0.0) + amount

    if totals:
        summary = " · ".join(
            f"{resource} {_format_money(amount)}" for resource, amount in sorted(totals.items())
        )
        lines.append(f"[dim]recent charges[/dim] {summary}")
    else:
        lines.append("[dim]No recent billing rows.[/dim]")
    return "\n".join(lines)


def _render_prime_billing_load_error(error: str) -> str:
    return (
        "[bold]Prime Intellect Wallet[/bold]\n"
        "[yellow]Wallet unavailable.[/yellow]\n"
        f"[dim]{_clip(_escape_markup(error), 80)}[/dim]"
    )


def _render_provider_billing_body(
    *,
    modal_payload: Any | None,
    modal_error: str | None,
    prime_state: Literal["loading", "loaded", "failed", "unavailable"],
    prime_payload: Any | None,
    prime_error: str | None,
    storage_snapshot: StorageSnapshot | None = None,
) -> str:
    """Compose the Modal and Prime billing sections of the shared panel."""

    if modal_error is not None:
        modal_section = _render_billing_load_error(modal_error)
    elif modal_payload is not None:
        modal_section = _render_billing_report(
            modal_payload,
            storage_snapshot=storage_snapshot,
        )
    else:
        modal_section = (
            "[bold]Workspace Spend[/bold]\n"
            "[dim]Refreshing billing report...[/dim]"
        )

    if prime_state == "unavailable":
        prime_section = (
            "[bold]Prime Intellect Wallet[/bold]\n"
            "[dim]Not authenticated (run: prime login)[/dim]"
        )
    elif prime_state == "failed":
        prime_section = _render_prime_billing_load_error(
            prime_error or "Could not read Prime billing wallet."
        )
    elif prime_state == "loaded":
        prime_section = _render_prime_billing_report(prime_payload)
    else:
        prime_section = (
            "[bold]Prime Intellect Wallet[/bold]\n"
            "[dim]Refreshing wallet...[/dim]"
        )

    return f"{modal_section}\n\n{prime_section}"


class MainMenuScreen(CopyEnabledScreen):
    """Top-level menu: deploy a model, custom deploy, manage, storage, settings."""

    BINDINGS = [
        Binding("d", "select_deploy", "Deploy", show=True),
        Binding("c", "select_custom_deploy", "Advanced", show=True),
        Binding("m", "select_manage", "Manage", show=True),
        Binding("t", "select_storage", "Storage", show=True),
        Binding("s", "select_settings", "Settings", show=True),
        Binding("i", "toggle_details", "Details", show=True),
        Binding("escape", "close_details", show=False),
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
        self._billing_payload: Any | None = None
        self._billing_error: str | None = None
        self._prime_billing_refresh_inflight = False
        self._prime_billing_state: Literal["loading", "loaded", "failed", "unavailable"] = "loading"
        self._prime_billing_payload: Any | None = None
        self._prime_billing_error: str | None = None
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
                        yield Static(
                            f"[bold]{version_text}[/bold][dim]Modal + Prime LLM backends[/dim]",
                            classes="centered main-menu-version",
                        )
                        yield Static("", classes="decorative-spacer")
                        yield OptionList(
                            Option("  Deploy model       Pick a model, get a live placement", id="deploy"),
                            Option("  Advanced deploy    llama.cpp / vLLM expert form", id="custom-deploy"),
                            Option("  Manage             Status, logs, benchmark, stop", id="manage"),
                            Option("  Storage            Cached models, pre-download, delete", id="storage"),
                            Option("  Settings           Appearance and deploy defaults", id="settings"),
                            id="action-list",
                        )
                        yield Static(
                            "[dim]↑/↓ select · enter open · i details[/dim]",
                            id="compact-menu-help",
                        )
                    with Vertical(id="main-menu-side-column"):
                        with Vertical(id="deployment-status-panel"):
                            yield Static("[bold #7bf168]Deployment Status[/]", id="deployment-status-title")
                            yield Static("[dim]Refreshing deployment status...[/dim]", id="deployment-status-body")
                        with Vertical(id="billing-report-panel"):
                            yield Static("[bold #7bf168]Provider Billing[/]", id="billing-report-title")
                            yield Static("[dim]Refreshing billing report...[/dim]", id="billing-report-body")
            yield Static(
                _render_auth_status_block(username=self.username),
                id="auth-status-block",
            )
        yield Footer()

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
        try:
            action_list = self.query_one("#action-list", OptionList)
        except Exception:
            return
        highlighted = action_list.highlighted_option
        selected_id = str(highlighted.id) if highlighted is not None else "deploy"
        if profile.compact:
            labels = (
                ("deploy", "  Deploy model"),
                ("custom-deploy", "  Advanced deploy"),
                ("manage", "  Manage endpoints"),
                ("storage", "  Storage"),
                ("settings", "  Settings"),
            )
        else:
            labels = (
                ("deploy", "  Deploy model       Pick a model, get a live placement"),
                ("custom-deploy", "  Advanced deploy    llama.cpp / vLLM expert form"),
                ("manage", "  Manage             Status, logs, benchmark, stop"),
                ("storage", "  Storage            Cached models, pre-download, delete"),
                ("settings", "  Settings           Appearance and deploy defaults"),
            )
        action_list.set_options(
            [Option(label, id=option_id) for option_id, label in labels]
        )
        for index, (option_id, _) in enumerate(labels):
            if option_id == selected_id:
                action_list.highlighted = index
                break

    def action_toggle_details(self) -> None:
        """Expose secondary fleet and billing panels as a narrow drawer."""
        if not self.viewport_profile.narrow and not self.viewport_profile.short:
            self.notify(
                "Fleet and billing panels are already visible on this screen size.",
                title="Details",
                timeout=3,
            )
            return
        self.toggle_class("show-secondary-panel")

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
        # picker stays usable while they resolve.
        try:
            upgraded = attach_quick_deploy_mtp_recommendations(profiles)
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

    def _refresh_modal_auth_status(self) -> None:
        if self._modal_auth_refresh_inflight:
            return
        self._modal_auth_refresh_inflight = True
        self.query_one("#auth-status-block", Static).update(
            _render_auth_status_block(
                username=self.username,
                modal_status=self._modal_auth_status,
                hf_status=self._hf_auth_status,
                prime_status=self._prime_auth_status,
                aai_status=self._aai_auth_status,
            )
        )
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
        self.query_one("#auth-status-block", Static).update(
            _render_auth_status_block(
                username=self.username,
                modal_status=self._modal_auth_status,
                hf_status=self._hf_auth_status,
                prime_status=self._prime_auth_status,
                aai_status=self._aai_auth_status,
            )
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
        self.query_one("#auth-status-block", Static).update(
            _render_auth_status_block(
                username=self.username,
                modal_status=self._modal_auth_status,
                hf_status=self._hf_auth_status,
                prime_status=self._prime_auth_status,
                aai_status=self._aai_auth_status,
            )
        )
        if message.status.authenticated:
            if self._secondary_refresh_started:
                self._refresh_prime_billing_report()
        elif self._prime_billing_state != "loaded":
            self._prime_billing_state = "unavailable"
            self._prime_billing_error = None
            self._prime_billing_refresh_inflight = False
            self._update_billing_panel()

    def _refresh_hf_auth_status(self) -> None:
        if self._hf_auth_refresh_inflight:
            return
        self._hf_auth_refresh_inflight = True
        self.query_one("#auth-status-block", Static).update(
            _render_auth_status_block(
                username=self.username,
                modal_status=self._modal_auth_status,
                hf_status=self._hf_auth_status,
                prime_status=self._prime_auth_status,
                aai_status=self._aai_auth_status,
            )
        )
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
        self.query_one("#auth-status-block", Static).update(
            _render_auth_status_block(
                username=self.username,
                modal_status=self._modal_auth_status,
                hf_status=self._hf_auth_status,
                prime_status=self._prime_auth_status,
                aai_status=self._aai_auth_status,
            )
        )

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
        self.query_one("#auth-status-block", Static).update(
            _render_auth_status_block(
                username=self.username,
                modal_status=self._modal_auth_status,
                hf_status=self._hf_auth_status,
                prime_status=self._prime_auth_status,
                aai_status=self._aai_auth_status,
            )
        )

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
        self._refresh_billing_panels()
        self._refresh_storage_estimate()

    def _refresh_billing_panels(self) -> None:
        if not self._is_active_screen():
            return
        self._last_billing_refresh_at = time.monotonic()
        self._refresh_billing_report()
        self._refresh_prime_billing_report()

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
        self.query_one("#deployment-status-body", Static).update("[dim]Refreshing deployment status...[/dim]")
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
        if rows is None:
            poster(DeploymentsLoadFailed(error="Could not read Modal app list."))
            return
        _annotate_runtime_statuses(rows, self.username)
        poster(DeploymentsLoaded(rows=[row for row in rows if row.backend is not None]))

    def on_endpoints_loaded(self, message: EndpointsLoaded) -> None:
        """Add runtime health details without mutating the shared endpoint cache."""
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
            self._show_deployments([replace(row) for row in self._runtime_rows])
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

    def _refresh_billing_report(self) -> None:
        if self._billing_refresh_inflight:
            return
        self._billing_refresh_inflight = True
        self.run_worker(
            self._run_load_billing_report,
            name="main-menu-billing-worker",
            thread=True,
        )

    def _update_billing_panel(self) -> None:
        self.query_one("#billing-report-body", Static).update(
            _render_provider_billing_body(
                modal_payload=self._billing_payload,
                modal_error=self._billing_error,
                prime_state=self._prime_billing_state,
                prime_payload=self._prime_billing_payload,
                prime_error=self._prime_billing_error,
                storage_snapshot=self._storage_snapshot,
            )
        )

    def _run_load_billing_report(self) -> None:
        poster = getattr(self, "post_message", None)
        if poster is None:
            return
        try:
            payload, error = ModalBackend.billing_report_json()
        except Exception as exc:
            poster(BillingReportLoadFailed(error=str(exc)))
            return
        if payload is None:
            poster(BillingReportLoadFailed(error=error or "Could not read billing report."))
            return
        storage_snapshot = None
        cached_storage_snapshot = getattr(self.app, "cached_storage_snapshot", None)
        if callable(cached_storage_snapshot):
            storage_snapshot = cached_storage_snapshot()
        poster(BillingReportLoaded(payload=payload, storage_snapshot=storage_snapshot))

    def _refresh_prime_billing_report(self) -> None:
        if self._prime_billing_refresh_inflight:
            return
        status = self._prime_auth_status
        if status is None:
            # Auth resolution runs concurrently; fall back to the local check.
            try:
                status = get_prime_auth_status()
            except Exception:
                status = None
        if status is not None and not status.authenticated:
            self._prime_billing_state = "unavailable"
            self._prime_billing_error = None
            self._update_billing_panel()
            return
        self._prime_billing_refresh_inflight = True
        self.run_worker(
            self._run_load_prime_billing_report,
            name="main-menu-prime-billing-worker",
            thread=True,
        )

    def _run_load_prime_billing_report(self) -> None:
        poster = getattr(self, "post_message", None)
        if poster is None:
            return
        try:
            payload, error = PrimeBackend().billing_wallet()
        except Exception as exc:
            poster(PrimeBillingReportLoadFailed(error=str(exc)))
            return
        if payload is None:
            poster(PrimeBillingReportLoadFailed(error=error or "Could not read Prime billing wallet."))
            return
        poster(PrimeBillingReportLoaded(payload=payload))

    def on_deployments_loaded(self, message: DeploymentsLoaded) -> None:
        self._status_refresh_inflight = False
        self._runtime_rows = [replace(row) for row in message.rows]
        self._runtime_rows_fingerprint = self._runtime_fingerprint(message.rows)
        self._runtime_rows_cached_at = time.monotonic()
        self._show_deployments(message.rows)

    def _show_deployments(self, rows: list[EndpointInfo]) -> None:
        visible_rows = [row for row in rows if _should_show_in_panel(row.state)]
        self.query_one("#deployment-status-body", Static).update(
            _render_deployment_status(visible_rows, username=self.username)
        )

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
            "[bold]Fleet Pulse[/bold]\n"
            "[yellow]Status unavailable.[/yellow]\n"
            f"[dim]{_clip(message.error, 80)}[/dim]"
        )

    def on_billing_report_loaded(self, message: BillingReportLoaded) -> None:
        self._billing_refresh_inflight = False
        self._billing_payload = message.payload
        self._billing_error = None
        if message.storage_snapshot is not None:
            self._storage_snapshot = message.storage_snapshot
        self._update_billing_panel()

    def on_billing_report_load_failed(self, message: BillingReportLoadFailed) -> None:
        self._billing_refresh_inflight = False
        self._billing_error = message.error
        self._update_billing_panel()

    def on_prime_billing_report_loaded(self, message: PrimeBillingReportLoaded) -> None:
        self._prime_billing_refresh_inflight = False
        self._prime_billing_state = "loaded"
        self._prime_billing_payload = message.payload
        self._prime_billing_error = None
        self._update_billing_panel()

    def on_prime_billing_report_load_failed(self, message: PrimeBillingReportLoadFailed) -> None:
        self._prime_billing_refresh_inflight = False
        self._prime_billing_state = "failed"
        self._prime_billing_error = message.error
        self._update_billing_panel()

    def on_storage_loaded(self, message: StorageLoaded) -> None:
        self._storage_snapshot = message.snapshot
        if self._billing_payload is None:
            return
        self._update_billing_panel()

    def on_storage_failed(self, _: StorageFailed) -> None:
        return

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        option_id = event.option.id
        if option_id == "deploy":
            self.app.action_push_deploy()  # type: ignore[attr-defined]
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

    def action_select_custom_deploy(self) -> None:
        self.app.action_push_custom_deploy()  # type: ignore[attr-defined]

    def action_select_manage(self) -> None:
        self.app.action_push_manage()  # type: ignore[attr-defined]

    def action_select_storage(self) -> None:
        self.app.action_push_storage()  # type: ignore[attr-defined]

    def action_select_settings(self) -> None:
        self.app.action_push_settings()  # type: ignore[attr-defined]
