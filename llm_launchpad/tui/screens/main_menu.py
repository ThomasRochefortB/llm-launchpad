"""Main menu screen: deploy, manage endpoints, settings.

Inspired by the Codex TUI main screen with a prominent banner,
auth status line, and keyboard-navigable option list.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from typing import Any, Literal

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, Vertical
from textual.message import Message
from textual.widgets import Footer, OptionList, Static
from textual.widgets.option_list import Option

from ...core.backend import ModalBackend
from ...core.hf_auth import HuggingFaceAuthStatus, get_huggingface_auth_status
from ...core.modal_auth import ModalAuthStatus, get_modal_auth_status
from ...core.inference_options import PrimeInferenceAdapter
from ...core.prime_auth import PrimeAuthStatus, get_prime_auth_status
from ...core.prime_backend import PrimeBackend
from ...core.naming import default_llamacpp_served_model_name, default_served_model_name
from ...core.quick_deploy import (
    QuickDeployCatalogInfo,
    QuickDeployProfile,
    activate_quick_deploy_catalog,
    format_context_length,
    format_hourly_cost,
    get_quick_deploy_catalog_info,
    list_quick_deploy_profiles,
    quick_deploy_profile_for_plan,
    quick_deploy_model_label_parts,
    resolve_quick_deploy_plans,
)
from ...core.quick_deploy_refresh import (
    ArtificialAnalysisAuthStatus,
    build_live_quick_deploy_catalog,
    get_artificial_analysis_auth_status,
)
from ...core.storage_costs import (
    MODAL_VOLUME_FREE_TIER_GIB_MONTH,
    estimate_monthly_storage_cost,
)
from ...protocol.enums import BackendType, ComputeProvider
from ...protocol.models import EndpointInfo, InferencePlan, StorageSnapshot
from ..workers import StorageFailed, StorageLoaded
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


class _LauncherOptionList(OptionList):
    """OptionList with cross-panel arrow navigation at list boundaries."""

    def __init__(
        self,
        *content: Option | None,
        handoff_up_target: str | None = None,
        handoff_down_target: str | None = None,
        **kwargs: object,
    ) -> None:
        self._handoff_up_target = handoff_up_target
        self._handoff_down_target = handoff_down_target
        super().__init__(*content, **kwargs)

    def _has_enabled_neighbor(self, direction: Literal[-1, 1]) -> bool:
        highlighted = self.highlighted
        if highlighted is None:
            return False
        if direction == -1:
            indexes = range(highlighted - 1, -1, -1)
        else:
            indexes = range(highlighted + 1, self.option_count)
        return any(not self.get_option_at_index(index).disabled for index in indexes)

    def _handoff_focus(self, target_id: str | None, position: Literal["first", "last"]) -> bool:
        if not target_id:
            return False
        try:
            target = self.screen.query_one(f"#{target_id}", OptionList)
        except Exception:
            return False
        if target.option_count <= 0:
            return False
        if position == "first":
            target.action_first()
        else:
            target.action_last()
        target.focus()
        return True

    def action_cursor_up(self) -> None:
        if self.highlighted is not None and not self._has_enabled_neighbor(-1):
            if self._handoff_focus(self._handoff_up_target, "last"):
                return
        super().action_cursor_up()

    def action_cursor_down(self) -> None:
        if self.highlighted is not None and not self._has_enabled_neighbor(1):
            if self._handoff_focus(self._handoff_down_target, "first"):
                return
        super().action_cursor_down()


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


class PrimeInferencePlansLoaded(Message):
    """Live Prime inference options were resolved for popular models."""

    def __init__(self, plans: tuple[InferencePlan, ...]) -> None:
        super().__init__()
        self.plans = plans


class PrimeInferencePlansLoadFailed(Message):
    """Live Prime inference option lookup failed."""

    def __init__(self, error: str) -> None:
        super().__init__()
        self.error = error


class QuickDeployCatalogLoaded(Message):
    """A refreshed Recommended Models catalog was loaded."""

    def __init__(
        self,
        info: QuickDeployCatalogInfo,
        profiles: tuple[QuickDeployProfile, ...],
    ) -> None:
        super().__init__()
        self.info = info
        self.profiles = profiles


class QuickDeployCatalogLoadFailed(Message):
    """The live Recommended Models refresh failed; keep the bundled catalog."""

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
        "(set: ARTIFICIAL_ANALYSIS_API_KEY)[/yellow]"
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


def _resolve_openai_base_url(row: EndpointInfo, username: str = "") -> tuple[str | None, bool]:
    raw_url = (row.web_url or "").strip()
    if raw_url:
        base_root = raw_url.rstrip("/")
        return (base_root if base_root.endswith("/v1") else f"{base_root}/v1"), False
    if row.provider != ComputeProvider.MODAL or not username.strip() or not row.name.strip():
        return None, False
    derived = ModalBackend.default_server_url(username.strip(), app_name=row.name.strip()).rstrip("/")
    return (derived if derived.endswith("/v1") else f"{derived}/v1"), True


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

    base_url, was_derived = _resolve_openai_base_url(row, username=username)
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


def _endpoint_model_summary(row: EndpointInfo) -> tuple[str | None, str | None]:
    explicit_display_name = (row.display_name or "").strip() or None

    if row.backend == BackendType.VLLM:
        model_id = (row.served_model_name or "").strip()
        if not model_id and (row.model_name or "").strip():
            model_id = default_served_model_name(row.model_name)
        display_name = explicit_display_name or (row.model_name or row.served_model_name or "").strip() or None
        return model_id or None, display_name

    if row.backend == BackendType.LLAMACPP:
        model_id = (row.served_model_name or "").strip()
        if not model_id and (row.repo_id or "").strip():
            model_id = default_llamacpp_served_model_name(row.repo_id, row.quant)
        if explicit_display_name:
            display_name = explicit_display_name
        else:
            repo = (row.repo_id or "").strip()
            quant = (row.quant or "").strip()
            if repo:
                display_name = f"{repo} ({quant})" if quant else repo
            else:
                display_name = (row.served_model_name or "").strip() or None
        return model_id or None, display_name or None

    return (row.served_model_name or "").strip() or None, explicit_display_name


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

        model_id, display_name = _endpoint_model_summary(row)
        app_lines.append(f"[dim]Display name:[/dim] {_escape_markup(display_name or '')}")
        app_lines.append(f"[dim]Model ID:[/dim] {_escape_markup(model_id or '')}")

        base_url, _was_derived = _resolve_openai_base_url(row, username=username)
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


def _quick_deploy_options(
    profiles: list[QuickDeployProfile],
    plans: tuple[InferencePlan, ...] | None = None,
) -> list[Option | None]:
    if plans is not None:
        return _resolved_quick_deploy_options(profiles, plans)

    grouped = _has_tiered_quick_deploy_groups(profiles)
    if not grouped:
        return [
            Option(
                _render_quick_deploy_option(profile),
                id=profile.id,
            )
            for profile in profiles
        ]

    options: list[Option | None] = []
    groups = _quick_deploy_profile_groups(profiles)
    for group_index, group in enumerate(groups):
        if group_index > 0:
            options.append(None)
        options.extend(
            Option(
                _render_quick_deploy_tier_row(
                    profile,
                    show_header=index == 0,
                ),
                id=profile.id,
            )
            for index, profile in enumerate(group)
        )
    return options


def _resolved_quick_deploy_options(
    profiles: list[QuickDeployProfile],
    plans: tuple[InferencePlan, ...],
) -> list[Option | None]:
    """Render one selectable row per provider quote, grouped by model."""

    rows: list[tuple[QuickDeployProfile, InferencePlan]] = []
    for plan in plans:
        try:
            profile = quick_deploy_profile_for_plan(plan, profiles)
        except KeyError:
            continue
        rows.append((profile, plan))
    if not rows:
        return []

    group_by_key: dict[str, list[tuple[QuickDeployProfile, InferencePlan]]] = {}
    group_order: list[str] = []
    for profile, plan in rows:
        key = _quick_deploy_group_key(profile)
        if key not in group_by_key:
            group_by_key[key] = []
            group_order.append(key)
        group_by_key[key].append((profile, plan))

    groups = [group_by_key[key] for key in group_order]
    for group in groups:
        group.sort(key=_resolved_quick_deploy_sort_key)
    groups = [
        group
        for _index, group in sorted(
            enumerate(groups),
            key=lambda item: _quick_deploy_group_sort_key(
                item[0],
                [profile for profile, _plan in item[1]],
            ),
        )
    ]

    show_rows = any(len(group) > 1 for group in groups)
    options: list[Option | None] = []
    for group_index, group in enumerate(groups):
        if group_index > 0 and show_rows:
            options.append(None)
        for row_index, (profile, plan) in enumerate(group):
            prompt = (
                _render_quick_deploy_tier_row(
                    profile,
                    show_header=row_index == 0,
                    plan=plan,
                )
                if show_rows
                else _render_quick_deploy_option(profile, plan)
            )
            options.append(Option(prompt, id=plan.quote.id))
    return options


def _resolved_quick_deploy_sort_key(
    row: tuple[QuickDeployProfile, InferencePlan],
) -> tuple[int, int, float, str, str]:
    profile, plan = row
    monthly = plan.estimated_monthly_cost_usd
    return (
        _quick_deploy_quant_order(profile.quant),
        1 if monthly is None else 0,
        monthly if monthly is not None else float("inf"),
        plan.quote.provider.value,
        plan.quote.id,
    )


def _has_tiered_quick_deploy_groups(profiles: list[QuickDeployProfile]) -> bool:
    counts = Counter(_quick_deploy_group_key(profile) for profile in profiles)
    return any(count > 1 for count in counts.values())


def _quick_deploy_profile_groups(profiles: list[QuickDeployProfile]) -> list[list[QuickDeployProfile]]:
    groups: list[list[QuickDeployProfile]] = []
    group_by_key: dict[str, list[QuickDeployProfile]] = {}
    for profile in profiles:
        key = _quick_deploy_group_key(profile)
        if key not in group_by_key:
            group_by_key[key] = []
            groups.append(group_by_key[key])
        group_by_key[key].append(profile)
    for group in groups:
        group.sort(key=_quick_deploy_profile_sort_key)
    groups = [
        group
        for _index, group in sorted(
            enumerate(groups),
            key=lambda index_group: _quick_deploy_group_sort_key(index_group[0], index_group[1]),
        )
    ]
    return groups


def _quick_deploy_group_key(profile: QuickDeployProfile) -> str:
    label, _quant_suffix = quick_deploy_model_label_parts(profile)
    return label.casefold()


def _quick_deploy_group_sort_key(
    index: int,
    group: list[QuickDeployProfile],
) -> tuple[int, int, float, int]:
    size_order = min(
        (_quick_deploy_size_order(profile.model_size_label) for profile in group),
        default=3,
    )
    score = max(
        (profile.aa_coding_score for profile in group if profile.aa_coding_score is not None),
        default=None,
    )
    if score is None:
        return (size_order, 1, 0.0, index)
    return (size_order, 0, -score, index)


def _quick_deploy_size_order(label: str | None) -> int:
    normalized = (label or "").strip().casefold()
    if normalized.startswith("compact"):
        return 0
    if normalized.startswith("medium"):
        return 1
    if normalized.startswith("large"):
        return 2
    return 3


def _quick_deploy_profile_sort_key(profile: QuickDeployProfile) -> tuple[int, int, float, str]:
    return (
        _quick_deploy_quant_order(profile.quant),
        _quick_deploy_tier_order(profile.resource_tier),
        profile.approx_cost_per_hour_usd,
        profile.id,
    )


def _quick_deploy_quant_order(quant: str) -> int:
    compact = _compact_quant_label(quant).casefold()
    if compact.startswith("q2"):
        return 0
    if compact.startswith("q3"):
        return 1
    if compact.startswith("q4"):
        return 2
    return 3


def _quick_deploy_tier_order(resource_tier: str | None) -> int:
    tier = (resource_tier or "").strip().casefold()
    if tier == "cheap":
        return 0
    if tier == "rtx-pro":
        return 1
    if tier == "b200":
        return 2
    return 3


def _render_quick_deploy_option(
    profile: QuickDeployProfile,
    plan: InferencePlan | None = None,
) -> str:
    label, quant_suffix = quick_deploy_model_label_parts(profile)
    model = _escape_markup(_clip(label, 18))
    quant = _compact_quant_label(quant_suffix or profile.quant)
    quant_markup = f" [dim]{_escape_markup(quant)}[/dim]" if quant else ""
    tier_markup = _quick_deploy_plan_tier_markup(profile, plan)
    gpu_shape = _escape_markup(_compact_plan_gpu_shape(profile, plan))
    cost = _compact_plan_cost(profile, plan)
    provider = f" · {plan.quote.provider.display_name}" if plan else ""
    return (
        f"  [bold]{model}[/bold]{quant_markup} {tier_markup}  "
        f"[dim]{gpu_shape} · "
        f"{_escape_markup(format_context_length(profile.max_context_tokens))} · "
        f"{_escape_markup(cost)}{_escape_markup(provider)}[/dim]"
    )


def _render_quick_deploy_tier_row(
    profile: QuickDeployProfile,
    *,
    show_header: bool = False,
    plan: InferencePlan | None = None,
) -> str:
    tier_markup = _quick_deploy_plan_tier_markup(profile, plan)
    gpu_shape = _escape_markup(_compact_plan_gpu_shape(profile, plan))
    quant_markup = _quick_deploy_quant_markup(profile)
    cost = _compact_plan_cost(profile, plan)
    provider = f" · {plan.quote.provider.display_name}" if plan else ""
    row = (
        f"    {quant_markup} {tier_markup}  "
        f"[dim]{gpu_shape} · {_escape_markup(cost)}"
        f"{_escape_markup(provider)}[/dim]"
    )
    if not show_header:
        return row

    label, _quant_suffix = quick_deploy_model_label_parts(profile)
    model = _escape_markup(_clip(label, 18))
    header = f"  [bold]{model}[/bold] [dim]{_escape_markup(_quick_deploy_header_metrics(profile))}[/dim]"
    return f"{header}\n{row}"


def _quick_deploy_header_metrics(profile: QuickDeployProfile) -> str:
    metrics = [format_context_length(profile.max_context_tokens)]
    if profile.aa_coding_score is not None:
        metrics.append(f"AA {_format_coding_index(profile.aa_coding_score)}")
    if profile.model_size_label:
        metrics.append(profile.model_size_label)
    return " · ".join(metrics)


def _format_coding_index(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _compact_quant_label(value: str) -> str:
    cleaned = value.strip().strip("()")
    if not cleaned:
        return ""
    return (
        cleaned.replace("UD-", "")
        .replace("_K_XL", "XL")
        .replace("_K_M", "M")
        .replace("_K_S", "S")
        .replace("_", "")
        .replace("-", "")
    )


def _compact_gpu_shape(profile: QuickDeployProfile) -> str:
    gpu = profile.gpu_type.strip()
    compact = {
        "A100-80GB": "A100",
        "A100-40GB": "A100-40",
        "RTX-PRO-6000": "RTX6000",
    }.get(gpu, gpu.replace("-80GB", ""))
    return f"{compact}x{profile.gpu_count}"


def _compact_plan_gpu_shape(
    profile: QuickDeployProfile,
    plan: InferencePlan | None,
) -> str:
    if plan is None:
        return _compact_gpu_shape(profile)
    gpu = plan.quote.gpu_type.strip()
    compact = {
        "A100-80GB": "A100",
        "A100-40GB": "A100-40",
        "RTX-PRO-6000": "RTX6000",
    }.get(gpu, gpu.replace("-80GB", ""))
    return f"{compact}x{plan.quote.gpu_count}"


def _compact_hourly_cost(value: float) -> str:
    return format_hourly_cost(value).replace("/hr", "/h")


def _compact_plan_cost(
    profile: QuickDeployProfile,
    plan: InferencePlan | None,
) -> str:
    value = (
        plan.quote.price_per_hour_usd
        if plan is not None and plan.quote.price_per_hour_usd is not None
        else profile.approx_cost_per_hour_usd
    )
    hourly = _compact_hourly_cost(value)
    if plan is None or plan.estimated_monthly_cost_usd is None:
        return hourly
    return f"{hourly} · ~${plan.estimated_monthly_cost_usd:,.0f}/mo"


def _quick_deploy_plan_tier_markup(
    profile: QuickDeployProfile,
    plan: InferencePlan | None,
) -> str:
    if plan is not None and plan.quote.configuration_id is None:
        return "[#7bf168]live[/]" if not plan.quote.is_estimate else "[dim]estimate[/dim]"
    return _quick_deploy_tier_markup(profile)


def _quick_deploy_tier_markup(profile: QuickDeployProfile) -> str:
    label = (profile.resource_tier_label or "").strip()
    if label and label != _default_resource_tier_label(profile.resource_tier):
        return _quick_deploy_tier_label_markup(label)

    tier = (profile.resource_tier or "").strip().casefold()
    if tier == "cheap":
        return "[dim]$[/dim]"
    if tier == "rtx-pro":
        return "[#7bf168]$$[/]"
    if tier == "b200":
        return "[yellow]$$$[/yellow]"
    label = (profile.resource_tier_label or profile.profile_label or "").strip()
    return f"[dim]{_escape_markup(label.casefold())}[/dim]" if label else ""


def _default_resource_tier_label(resource_tier: str | None) -> str:
    tier = (resource_tier or "").strip().casefold()
    if tier == "cheap":
        return "$"
    if tier == "rtx-pro":
        return "$$"
    if tier == "b200":
        return "$$$"
    return ""


def _quick_deploy_tier_label_markup(label: str) -> str:
    pieces: list[str] = []
    for piece in label.split("/"):
        token = piece.strip()
        if not token:
            continue
        if pieces:
            pieces.append("[dim]/[/dim]")
        pieces.append(_quick_deploy_tier_token_markup(token))
    return "".join(pieces) if pieces else f"[dim]{_escape_markup(label)}[/dim]"


def _quick_deploy_tier_token_markup(token: str) -> str:
    if token == "$":
        return "[dim]$[/dim]"
    if token == "$$":
        return "[#7bf168]$$[/]"
    if token == "$$$":
        return "[yellow]$$$[/yellow]"
    return f"[dim]{_escape_markup(token)}[/dim]"


def _quick_deploy_quant_markup(profile: QuickDeployProfile) -> str:
    quant = _compact_quant_label(profile.quant)
    if not quant:
        return ""
    if quant.casefold().startswith("q2"):
        return f"[bold #7bf168]{_escape_markup(quant)}[/]"
    return f"[dim]{_escape_markup(quant)}[/dim]"


def _quick_deploy_subtitle(info: QuickDeployCatalogInfo) -> str:
    if info.is_fallback:
        return "Choose an inference option · curated coding models"
    generated = (info.generated_at or "").strip()
    if generated:
        generated = generated.split("T", 1)[0]
    if generated:
        catalog_origin = "live" if info.is_live else "bundled"
        if info.source_label.casefold().startswith("artificial analysis"):
            benchmark_state = (
                "cached benchmarks"
                if "(cached" in info.source_label.casefold()
                else "fresh benchmarks"
            )
            source = (
                f"AA size leaders ({benchmark_state}), "
                f"{catalog_origin} {generated}"
            )
        else:
            source = f"{info.source_label}, {catalog_origin} {generated}"
        return f"Choose an inference option · {source}"
    return f"Choose an inference option · {info.source_label}"


class MainMenuScreen(CopyEnabledScreen):
    """Top-level menu: deploy, manage, settings."""

    BINDINGS = [
        Binding("d", "select_deploy", "Deploy", show=True),
        Binding("m", "select_manage", "Manage", show=True),
        Binding("t", "select_storage", "Storage", show=True),
        Binding("s", "select_settings", "Settings", show=True),
        Binding("tab", "focus_next_launcher", show=False),
        Binding("shift+tab", "focus_previous_launcher", show=False),
    ]

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
        self._prime_inference_refresh_inflight = False
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
        self._quick_deploy_profiles = list_quick_deploy_profiles()
        self._modal_quick_deploy_plans = resolve_quick_deploy_plans(
            self._quick_deploy_profiles
        )
        self._quick_deploy_plans = self._modal_quick_deploy_plans
        self._quick_deploy_catalog_info = get_quick_deploy_catalog_info()

    def compose(self) -> ComposeResult:
        with Center():
            with Horizontal(id="main-menu-layout"):
                with Vertical(id="menu-container"):
                    yield Static(BANNER, id="banner-text")
                    version_text = f"v{self.version}  " if self.version else ""
                    yield Static(
                        f"[bold]{version_text}[/bold][dim]Modal + Prime LLM backends[/dim]",
                        classes="centered",
                    )
                    yield Static("")  # spacer
                    yield _LauncherOptionList(
                        Option("  Deploy            Launch a new LLM backend", id="deploy"),
                        Option("  Manage            List, status, logs, stop", id="manage"),
                        Option("  Storage           Cached models and pre-download", id="storage"),
                        Option("  Settings          Scaledown defaults", id="settings"),
                        id="action-list",
                        handoff_up_target="quick-deploy-list",
                        handoff_down_target="quick-deploy-list",
                    )
                with Vertical(id="main-menu-side-column"):
                    with Vertical(id="deployment-status-panel"):
                        yield Static("[bold #7bf168]Deployment Status[/]", id="deployment-status-title")
                        yield Static("[dim]Refreshing deployment status...[/dim]", id="deployment-status-body")
                    with Vertical(id="billing-report-panel"):
                        yield Static("[bold #7bf168]Provider Billing[/]", id="billing-report-title")
                        yield Static("[dim]Refreshing billing report...[/dim]", id="billing-report-body")
                    with Vertical(id="quick-deploy-panel"):
                        yield Static(
                            "[bold #7bf168]Recommended Models[/]",
                            id="landing-quick-deploy-title",
                        )
                        yield Static(
                            _quick_deploy_subtitle(self._quick_deploy_catalog_info),
                            id="landing-quick-deploy-subtitle",
                        )
                        yield _LauncherOptionList(
                            *_quick_deploy_options(
                                self._quick_deploy_profiles,
                                self._quick_deploy_plans,
                            ),
                            id="quick-deploy-list",
                            handoff_up_target="action-list",
                            handoff_down_target="action-list",
                        )
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
        quick_list = self.query_one("#quick-deploy-list", OptionList)
        if quick_list.option_count > 0:
            quick_list.action_first()
        self._refresh_quick_deploy_catalog()
        self._refresh_modal_auth_status()
        self._refresh_prime_auth_status()
        self._refresh_hf_auth_status()
        self._refresh_aai_auth_status()
        self._refresh_panels()
        self._refresh_storage_estimate()
        self.set_interval(20.0, self._refresh_panels)

    def _refresh_quick_deploy_catalog(self) -> None:
        if self._quick_deploy_catalog_refresh_inflight:
            return
        self._quick_deploy_catalog_refresh_inflight = True
        self.query_one("#landing-quick-deploy-subtitle", Static).update(
            f"{_quick_deploy_subtitle(self._quick_deploy_catalog_info)} · refreshing"
        )
        self.run_worker(
            self._run_refresh_quick_deploy_catalog,
            name="main-menu-quick-deploy-catalog-worker",
            thread=True,
        )

    def _run_refresh_quick_deploy_catalog(self) -> None:
        try:
            info, profiles = build_live_quick_deploy_catalog()
        except Exception as exc:
            self.post_message(QuickDeployCatalogLoadFailed(error=str(exc)))
            return
        self.post_message(QuickDeployCatalogLoaded(info=info, profiles=profiles))

    def on_quick_deploy_catalog_loaded(
        self,
        message: QuickDeployCatalogLoaded,
    ) -> None:
        self._quick_deploy_catalog_refresh_inflight = False
        info, profiles = activate_quick_deploy_catalog(
            message.info,
            message.profiles,
        )
        self._quick_deploy_catalog_info = info
        self._quick_deploy_profiles = profiles
        self._modal_quick_deploy_plans = resolve_quick_deploy_plans(
            self._quick_deploy_profiles
        )
        self._quick_deploy_plans = self._modal_quick_deploy_plans
        self._replace_quick_deploy_options()
        self.query_one("#landing-quick-deploy-subtitle", Static).update(
            _quick_deploy_subtitle(self._quick_deploy_catalog_info)
        )
        if self._prime_auth_status is not None and self._prime_auth_status.authenticated:
            self._refresh_prime_inference_plans()

    def on_quick_deploy_catalog_load_failed(
        self,
        _: QuickDeployCatalogLoadFailed,
    ) -> None:
        self._quick_deploy_catalog_refresh_inflight = False
        self.query_one("#landing-quick-deploy-subtitle", Static).update(
            _quick_deploy_subtitle(self._quick_deploy_catalog_info)
        )
        if self._prime_auth_status is not None and self._prime_auth_status.authenticated:
            self._refresh_prime_inference_plans()

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
            if not self._quick_deploy_catalog_refresh_inflight:
                self._refresh_prime_inference_plans()
            self._refresh_prime_billing_report()
        elif self._prime_billing_state != "loaded":
            self._prime_billing_state = "unavailable"
            self._prime_billing_error = None
            self._prime_billing_refresh_inflight = False
            self._update_billing_panel()

    def _refresh_prime_inference_plans(self) -> None:
        if self._prime_inference_refresh_inflight:
            return
        self._prime_inference_refresh_inflight = True
        self.query_one("#landing-quick-deploy-subtitle", Static).update(
            f"{_quick_deploy_subtitle(self._quick_deploy_catalog_info)} · checking Prime"
        )
        self.run_worker(
            self._run_load_prime_inference_plans,
            name="main-menu-prime-inference-worker",
            thread=True,
        )

    def _run_load_prime_inference_plans(self) -> None:
        try:
            plans = resolve_quick_deploy_plans(
                self._quick_deploy_profiles,
                adapters=(PrimeInferenceAdapter(),),
            )
        except Exception as exc:
            self.post_message(PrimeInferencePlansLoadFailed(error=str(exc)))
            return
        # The full marketplace remains available in manual deploy. Popular
        # Models shows the lowest-cost compatible Prime option per recipe.
        recommended = tuple(
            plan for plan in plans if plan.recommendation_reason is not None
        )
        self.post_message(PrimeInferencePlansLoaded(plans=recommended))

    def on_prime_inference_plans_loaded(
        self,
        message: PrimeInferencePlansLoaded,
    ) -> None:
        self._prime_inference_refresh_inflight = False
        self._quick_deploy_plans = self._modal_quick_deploy_plans + message.plans
        self._replace_quick_deploy_options()
        suffix = (
            f" · {len(message.plans)} live Prime option"
            f"{'s' if len(message.plans) != 1 else ''}"
            if message.plans
            else " · no compatible Prime offers live"
        )
        self.query_one("#landing-quick-deploy-subtitle", Static).update(
            f"{_quick_deploy_subtitle(self._quick_deploy_catalog_info)}{suffix}"
        )

    def on_prime_inference_plans_load_failed(
        self,
        _: PrimeInferencePlansLoadFailed,
    ) -> None:
        self._prime_inference_refresh_inflight = False
        self.query_one("#landing-quick-deploy-subtitle", Static).update(
            f"{_quick_deploy_subtitle(self._quick_deploy_catalog_info)} · Prime pricing unavailable"
        )

    def _replace_quick_deploy_options(self) -> None:
        option_list = self.query_one("#quick-deploy-list", OptionList)
        selected_id = (
            option_list.highlighted_option.id
            if option_list.highlighted_option is not None
            else None
        )
        options = _quick_deploy_options(
            self._quick_deploy_profiles,
            self._quick_deploy_plans,
        )
        option_list.set_options(options)
        if selected_id is not None:
            for index in range(option_list.option_count):
                if option_list.get_option_at_index(index).id == selected_id:
                    option_list.highlighted = index
                    break
        if option_list.highlighted is None and option_list.option_count > 0:
            option_list.action_first()

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
        self._refresh_deployment_status()
        self._refresh_billing_report()
        self._refresh_prime_billing_report()

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
        self.run_worker(
            self._run_load_deployments,
            name="main-menu-status-worker",
            thread=True,
        )

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
        visible_rows = [row for row in message.rows if _should_show_in_panel(row.state)]
        self.query_one("#deployment-status-body", Static).update(
            _render_deployment_status(visible_rows, username=self.username)
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
        option_list_id = event.option_list.id
        option_id = event.option.id
        if option_list_id == "quick-deploy-list":
            plan = self._quick_deploy_plan_for_id(str(option_id))
            if plan is not None:
                self.app.push_quick_deploy(plan)  # type: ignore[attr-defined]
            return
        if option_id == "deploy":
            self.app.action_push_deploy()  # type: ignore[attr-defined]
        elif option_id == "manage":
            self.app.action_push_manage()  # type: ignore[attr-defined]
        elif option_id == "storage":
            self.app.action_push_storage()  # type: ignore[attr-defined]
        elif option_id == "settings":
            self.app.action_push_settings()  # type: ignore[attr-defined]

    def action_select_deploy(self) -> None:
        self.app.action_push_deploy()  # type: ignore[attr-defined]

    def action_select_manage(self) -> None:
        self.app.action_push_manage()  # type: ignore[attr-defined]

    def action_select_storage(self) -> None:
        self.app.action_push_storage()  # type: ignore[attr-defined]

    def action_select_settings(self) -> None:
        self.app.action_push_settings()  # type: ignore[attr-defined]

    def _quick_deploy_plan_for_id(self, plan_id: str) -> InferencePlan | None:
        for plan in self._quick_deploy_plans:
            if plan.quote.id == plan_id:
                return plan
        return None

    def _focus_launcher(self, target_id: str) -> None:
        target = self.query_one(f"#{target_id}", OptionList)
        target.focus()
        if target.highlighted is None and target.option_count > 0:
            target.highlighted = 0

    def action_focus_next_launcher(self) -> None:
        focused_id = getattr(self.focused, "id", "")
        self._focus_launcher("quick-deploy-list" if focused_id == "action-list" else "action-list")

    def action_focus_previous_launcher(self) -> None:
        focused_id = getattr(self.focused, "id", "")
        self._focus_launcher("action-list" if focused_id == "quick-deploy-list" else "quick-deploy-list")
