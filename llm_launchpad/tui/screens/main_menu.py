"""Main menu screen: deploy, manage endpoints, settings.

Inspired by the Codex TUI main screen with a prominent banner,
auth status line, and keyboard-navigable option list.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import json
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, Vertical
from textual.message import Message
from textual.widgets import Footer, OptionList, Static
from textual.widgets.option_list import Option

from ...core.backend import ModalBackend
from ...core.hf_auth import HuggingFaceAuthStatus, get_huggingface_auth_status
from ...core.naming import default_llamacpp_served_model_name, default_served_model_name
from ...protocol.enums import BackendType
from ...protocol.models import EndpointInfo
from .copy_enabled import CopyEnabledScreen

BANNER = r"""[bold cyan]
_     _     __  __
| |   | |   |  \/  |
| |   | |   | |\/| |
| |___| |___| |  | |
|_____|_____|_|  |_|
    LAUNCHPAD
[/bold cyan]"""

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

    def __init__(self, payload: Any) -> None:
        super().__init__()
        self.payload = payload


class BillingReportLoadFailed(Message):
    """Main-menu billing report fetch failed."""

    def __init__(self, error: str) -> None:
        super().__init__()
        self.error = error


class HuggingFaceAuthLoaded(Message):
    """Main-menu Hugging Face auth check completed."""

    def __init__(self, status: HuggingFaceAuthStatus) -> None:
        super().__init__()
        self.status = status


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
        if status.username:
            return f"[green]🤗 Hugging Face authenticated as {_escape_markup(status.username)}[/green]"
        return "[green]🤗 Hugging Face authenticated[/green]"
    if status.error:
        color = "red" if "invalid" in status.error.lower() else "yellow"
        detail = _escape_markup(_clip(status.error, 72))
        return f"[{color}]🤗 Hugging Face auth check failed: {detail}[/{color}]"
    return "[yellow]🤗 Hugging Face not authenticated (run: hf auth login)[/yellow]"


def _render_auth_status_block(
    username: str = "",
    hf_status: HuggingFaceAuthStatus | None = None,
) -> str:
    lines: list[str] = []
    if username:
        lines.append(f"[green]▰ Modal authenticated as: {_escape_markup(username)}[/green]")
    lines.append(_render_hf_auth_status(hf_status))
    return "\n".join(lines)


def _state_bucket(state: str) -> str:
    normalized = (state or "").strip().lower()
    if normalized in {"running", "deployed"}:
        return "healthy"
    if normalized in {"deploying", "starting", "initializing", "building"}:
        return "deploying"
    if normalized in {"queued", "pending"}:
        return "queued"
    if normalized in {"failed", "error", "crashed"}:
        return "error"
    if normalized in {"stopped", "stopping"}:
        return "stopped"
    return "other"


def _style_state(state: str) -> str:
    normalized = (state or "").strip().lower() or "unknown"
    bucket = _state_bucket(normalized)
    if bucket == "healthy":
        return f"[green]{normalized}[/green]"
    if bucket == "deploying":
        return f"[cyan]{normalized}[/cyan]"
    if bucket == "queued":
        return f"[yellow]{normalized}[/yellow]"
    if bucket == "error":
        return f"[red]{normalized}[/red]"
    if bucket == "stopped":
        return f"[magenta]{normalized}[/magenta]"
    return f"[dim]{normalized}[/dim]"


def _should_show_in_panel(state: str) -> bool:
    bucket = _state_bucket(state)
    return bucket in {"healthy", "deploying", "queued", "error"}


def _resolve_openai_base_url(row: EndpointInfo, username: str = "") -> tuple[str | None, bool]:
    raw_url = (row.web_url or "").strip()
    if raw_url:
        base_root = raw_url.rstrip("/")
        return (base_root if base_root.endswith("/v1") else f"{base_root}/v1"), False
    if not username.strip() or not row.name.strip():
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


def _llamacpp_probe_ready(status_code: int, body: str) -> bool:
    if not (200 <= status_code < 300):
        return False
    text = (body or "").strip()
    if not text:
        return False
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    return isinstance(payload.get("choices"), list)


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
    try:
        if row.backend == BackendType.VLLM:
            probe_url = host_root.rstrip("/") + "/health"
            response = requests.get(probe_url, timeout=2.5)
            if 200 <= response.status_code < 300:
                return "healthy", None
        else:
            probe_url = base_root.rstrip("/") + "/completions"
            response = requests.post(
                probe_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(
                    {
                        "model": (row.served_model_name or "").strip() or "default",
                        "prompt": "ping",
                        "max_tokens": 1,
                        "temperature": 0,
                    }
                ),
                timeout=2.5,
            )
            if _llamacpp_probe_ready(response.status_code, response.text or ""):
                return "healthy", None

        # If we had to derive the URL, avoid false "error" when function slug differs.
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

    noun = "deployment" if len(rows) == 1 else "deployments"
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
            "[dim]No active launchpad deployments.[/dim]"
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
            f"[dim](modal: {_escape_markup((row.state or 'unknown').strip().lower())})[/dim]"
        )

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
                app_lines.append("  [dim]API key[/dim] ")
            else:
                app_lines.append("[dim]OpenAI URL will be available once deployment is running.[/dim]")
        else:
            app_lines.append("[dim]OpenAI URL unavailable (Modal app list has no web URL yet).[/dim]")

        if index != len(display_rows) - 1:
            app_lines.append("")

    return "\n".join(header_lines + app_lines)


def _format_money(value: float) -> str:
    return f"${value:,.2f}"


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


def _find_first_text(payload: Any, dotted_keys: list[str]) -> str | None:
    for dotted_key in dotted_keys:
        current = payload
        found = True
        for key in dotted_key.split("."):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                found = False
                break
        if not found or current is None:
            continue
        text = str(current).strip()
        if text:
            return text
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


def _render_billing_report(payload: Any) -> str:
    normalized = _normalize_billing_payload(payload)
    if isinstance(normalized, list):
        rows = [row for row in normalized if isinstance(row, dict)]
        if not rows:
            return (
                "[bold]Workspace Spend[/bold]\n"
                "[dim]Current month spend[/dim]\n"
                "[dim]total[/dim] [bold]$0.00[/bold]\n"
                "[dim]No billed usage in the selected monthly period.[/dim]"
            )

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
        return "\n".join(lines)

    if not isinstance(normalized, dict):
        return (
            "[bold]Workspace Spend[/bold]\n"
            "[dim]Billing data unavailable.[/dim]\n"
            "[dim]Check `modal billing report --json`.[/dim]"
        )

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
    return "\n".join(lines)


def _render_billing_load_error(error: str) -> str:
    return (
        "[bold]Workspace Spend[/bold]\n"
        "[yellow]Billing unavailable.[/yellow]\n"
        f"[dim]{_clip(_escape_markup(error), 80)}[/dim]"
    )


class MainMenuScreen(CopyEnabledScreen):
    """Top-level menu: deploy, manage, settings."""

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("d", "select_deploy", "Deploy", show=True),
        Binding("m", "select_manage", "Manage", show=True),
        Binding("t", "select_storage", "Storage", show=True),
        Binding("s", "select_settings", "Settings", show=True),
    ]

    def __init__(self, username: str = "", version: str = "") -> None:
        super().__init__()
        self.username = username
        self.version = version
        self._hf_auth_refresh_inflight = False
        self._status_refresh_inflight = False
        self._billing_refresh_inflight = False

    def compose(self) -> ComposeResult:
        with Center():
            with Horizontal(id="main-menu-layout"):
                with Vertical(id="menu-container"):
                    yield Static(BANNER, id="banner-text")
                    version_text = f"v{self.version}  " if self.version else ""
                    yield Static(
                        f"[bold]{version_text}[/bold][dim]Modal LLM backends[/dim]",
                        classes="centered",
                    )
                    yield Static("")  # spacer
                    yield OptionList(
                        Option("  Deploy            Launch a new LLM backend", id="deploy"),
                        Option("  Manage            List, status, logs, stop", id="manage"),
                        Option("  Storage           Cached models and pre-download", id="storage"),
                        Option("  Settings          Scaledown defaults", id="settings"),
                        id="action-list",
                    )
                with Vertical(id="deployment-status-panel"):
                    yield Static("[bold cyan]Deployment Status[/bold cyan]", id="deployment-status-title")
                    yield Static("[dim]Refreshing deployment status...[/dim]", id="deployment-status-body")
                    yield Static(PANEL_SEPARATOR, id="deployment-billing-separator")
                    yield Static("[bold cyan]Modal Billing Report[/bold cyan]", id="billing-report-title")
                    yield Static("[dim]Refreshing billing report...[/dim]", id="billing-report-body")
        yield Static(
            _render_auth_status_block(username=self.username),
            id="auth-status-block",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Focus the option list so arrow-key navigation works immediately."""
        self.query_one("#action-list", OptionList).focus()
        self._refresh_hf_auth_status()
        self._refresh_panels()
        self.set_interval(20.0, self._refresh_panels)

    def _refresh_hf_auth_status(self) -> None:
        if self._hf_auth_refresh_inflight:
            return
        self._hf_auth_refresh_inflight = True
        self.query_one("#auth-status-block", Static).update(
            _render_auth_status_block(username=self.username)
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
        self.query_one("#auth-status-block", Static).update(
            _render_auth_status_block(username=self.username, hf_status=message.status)
        )

    def _refresh_panels(self) -> None:
        self._refresh_deployment_status()
        self._refresh_billing_report()

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
        self.query_one("#billing-report-body", Static).update("[dim]Refreshing billing report...[/dim]")
        self.run_worker(
            self._run_load_billing_report,
            name="main-menu-billing-worker",
            thread=True,
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
        poster(BillingReportLoaded(payload=payload))

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
        self.query_one("#billing-report-body", Static).update(_render_billing_report(message.payload))

    def on_billing_report_load_failed(self, message: BillingReportLoadFailed) -> None:
        self._billing_refresh_inflight = False
        self.query_one("#billing-report-body", Static).update(_render_billing_load_error(message.error))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        option_id = event.option.id
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

    async def action_quit(self) -> None:
        await self.app.action_quit()  # type: ignore[attr-defined]
