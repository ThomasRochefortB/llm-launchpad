"""Endpoint-first management screen and focused action forms."""

from __future__ import annotations

from rich.markup import escape
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, DataTable, Footer, Input, OptionList, Static
from textual.widgets.option_list import Option

from ...protocol.models import EndpointInfo
from ..connection import endpoint_connection_payload, resolve_openai_base_url
from ..navigation import move_focus_across_widgets
from ..responsive import ViewportProfile, WidthMode
from ..widgets.adaptive_table import AdaptiveColumn, AdaptiveDataTable
from ..widgets.input_form import FormField
from ..workers import EndpointsFailed, EndpointsLoaded
from .copy_enabled import CopyEnabledScreen


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


def _available_actions(row: EndpointInfo) -> frozenset[str]:
    """Return management actions supported by the endpoint's current state."""
    if row.backend is None:
        return frozenset()

    actions = {"logs"}
    state = _normalized_state(row.state)
    if state in _READY_STATES or bool((row.web_url or "").strip()):
        actions.update(("status", "benchmark", "connection"))
    if _is_stoppable_state(state):
        actions.add("stop")
    return frozenset(actions)


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
    return f"{_endpoint_host(row)} · {_state_label(row.state)}"


_WIDE = WidthMode.WIDE
_STANDARD = WidthMode.STANDARD
_COMPACT = WidthMode.COMPACT
_MINIMAL = WidthMode.MINIMAL

_ENDPOINT_COLUMNS = (
    AdaptiveColumn.visible(
        "endpoint",
        "endpoint",
        _endpoint_name,
        _WIDE,
        _STANDARD,
    ),
    AdaptiveColumn.visible(
        "endpoint-host",
        "endpoint / host",
        _endpoint_compact_label,
        _COMPACT,
        _MINIMAL,
    ),
    AdaptiveColumn.visible(
        "provider",
        "provider",
        lambda row: row.provider.value,
        _WIDE,
        _STANDARD,
    ),
    AdaptiveColumn.visible(
        "backend",
        "backend",
        _endpoint_backend,
        _WIDE,
        _STANDARD,
    ),
    AdaptiveColumn.visible(
        "state",
        "state",
        lambda row: _state_label(row.state),
        _WIDE,
        _STANDARD,
        _COMPACT,
        _MINIMAL,
    ),
    AdaptiveColumn.visible(
        "app",
        "app / pod",
        lambda row: row.name or row.app_id or "-",
        _WIDE,
    ),
)


class ManageScreen(CopyEnabledScreen):
    """Show the endpoint fleet once, then route actions for the selected row."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("r", "refresh_endpoints", "Refresh", show=True),
        Binding("enter", "open_actions", "Actions", show=True),
        # Keep the shortcuts for experienced users, but let the action menu
        # carry the discoverability burden in the footer and help text.
        Binding("s", "status_selected", "Status", show=False),
        Binding("l", "logs_selected", "Logs", show=False),
        Binding("b", "benchmark_selected", "Benchmark", show=False),
        Binding("x", "stop_selected", "Stop", show=False),
        Binding("u", "copy_base_url", "Copy URL", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="manage-layout"):
            yield Static(
                "[bold #7bf168]Manage Endpoints[/]  [dim]Choose an endpoint, then press Enter[/dim]",
                id="manage-title",
            )
            yield Static("[dim]Loading managed endpoints...[/dim]", id="manage-status")
            yield AdaptiveDataTable(id="manage-endpoint-table")
            yield Static("[dim]No endpoint selected.[/dim]", id="manage-selection-detail")
            yield Static(
                "[dim]↑/↓ choose · Enter actions · r refresh · Esc back[/dim]",
                id="manage-help",
            )
        yield Footer()

    def on_mount(self) -> None:
        self._rows: list[EndpointInfo] = []
        self._rows_by_key: dict[str, EndpointInfo] = {}
        self._selected_key: str | None = None
        self._was_suspended = False
        table = self.query_one("#manage-endpoint-table", AdaptiveDataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.configure(
            _ENDPOINT_COLUMNS,
            row_key=_endpoint_key,
            profile=self.viewport_profile,
        )
        table.focus()
        self._refresh_endpoints()

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

    def on_screen_suspend(self, _: events.ScreenSuspend) -> None:
        self._was_suspended = True

    def on_screen_resume(self, _: events.ScreenResume) -> None:
        """Refresh the fleet after returning from an endpoint operation."""
        if self._was_suspended:
            self._was_suspended = False
            self.call_after_refresh(self._refresh_if_current)

    def on_endpoints_loaded(self, message: EndpointsLoaded) -> None:
        selected_key = self._selected_key
        self._rows = sorted(
            (row for row in message.rows if row.backend is not None),
            key=lambda row: (_endpoint_name(row).casefold(), _endpoint_key(row)),
        )
        self._rows_by_key = {_endpoint_key(row): row for row in self._rows}
        table = self.query_one("#manage-endpoint-table", AdaptiveDataTable)
        table.set_rows(self._rows)

        if not self._rows:
            self._selected_key = None
            self.query_one("#manage-status", Static).update(
                "[yellow]No managed endpoints found.[/yellow]  Press r to refresh."
            )
            self._update_selection_detail()
            return

        self._selected_key = (
            selected_key
            if selected_key in self._rows_by_key
            else _endpoint_key(self._rows[0])
        )
        noun = "endpoint" if len(self._rows) == 1 else "endpoints"
        self.query_one("#manage-status", Static).update(
            f"[green]Fleet refreshed.[/green] {len(self._rows)} managed {noun}."
        )
        self._move_cursor_to_selected()
        self._update_selection_detail()

    def on_endpoints_failed(self, message: EndpointsFailed) -> None:
        self.query_one("#manage-status", Static).update(
            f"[yellow]Endpoint refresh failed:[/yellow] {escape(message.error)}"
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

    def action_logs_selected(self) -> None:
        row = self._row_for_action("logs")
        if row is not None:
            self.app.begin_logs(row, follow=True)  # type: ignore[attr-defined]

    def action_benchmark_selected(self) -> None:
        row = self._row_for_action("benchmark")
        if row is not None:
            self.app.push_screen(BenchmarkOptionsScreen(row))

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
        self.notify("Copied base URL", timeout=2)

    def _modal_username(self) -> str:
        return str(getattr(self.app, "_username", "") or "")

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def _refresh_endpoints(self, force: bool = False) -> None:
        self.query_one("#manage-status", Static).update(
            "[dim]Refreshing managed endpoints...[/dim]"
        )
        refresh = getattr(self.app, "begin_endpoint_refresh", None)
        if callable(refresh):
            refresh(self, force=force)
            return

        try:
            rows = self.app.list_instances()  # type: ignore[attr-defined]
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
        detail = self.query_one("#manage-selection-detail", Static)
        row = self._selected_endpoint()
        if row is None:
            detail.update("[dim]No endpoint selected.[/dim]")
            return
        actions = _available_actions(row)
        action_labels = [
            label
            for action, label in (
                ("status", "status"),
                ("logs", "logs"),
                ("benchmark", "benchmark"),
                ("stop", "stop"),
                ("connection", "connection info"),
            )
            if action in actions
        ]
        base_url, _derived = resolve_openai_base_url(row, username=self._modal_username())
        url_line = f"\n[dim]Base URL:[/dim] {escape(base_url)}" if base_url else ""
        detail.update(
            f"[bold]{escape(_endpoint_name(row))}[/bold]  "
            f"[dim]{escape(_endpoint_host(row))} · {escape(_state_label(row.state))}[/dim]"
            f"{url_line}\n"
            f"Actions: {', '.join(action_labels) or 'none'}"
        )


class EndpointActionsScreen(CopyEnabledScreen):
    """Let users choose an action without memorizing shortcut keys."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
    ]

    _ACTION_LABELS = (
        ("connection", "  Connection info"),
        ("status", "  Check status"),
        ("logs", "  View logs"),
        ("benchmark", "  Run benchmark"),
        ("stop", "  Stop endpoint"),
    )

    def __init__(self, endpoint: EndpointInfo) -> None:
        super().__init__()
        self.endpoint = endpoint

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="screen-scroll"):
            yield Static("[bold #7bf168]Endpoint Actions[/]")
            yield Static(
                f"[bold]{escape(_endpoint_name(self.endpoint))}[/bold]  "
                f"[dim]{escape(_endpoint_host(self.endpoint))} · "
                f"{escape(_state_label(self.endpoint.state))}[/dim]",
                id="manage-action-context",
            )
            yield OptionList(
                *(
                    Option(label, id=action)
                    for action, label in self._ACTION_LABELS
                    if action in _available_actions(self.endpoint)
                ),
                id="manage-actions",
            )
            yield Static(
                "[dim]↑/↓ choose · Enter open · Esc back[/dim]",
                id="manage-action-help",
            )
        yield Footer()

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
        highlighted = actions.highlighted_option
        if highlighted is not None:
            self._submit(str(highlighted.id))

    def _submit(self, action: str) -> None:
        if action not in _available_actions(self.endpoint):
            return
        self.app.pop_screen()
        if action == "connection":
            self.app.push_screen(ConnectionInfoScreen(self.endpoint))
        elif action == "status":
            # The common path probes immediately with defaults; the URL-override
            # form stays available via the hidden "s" shortcut on Manage.
            self.app.begin_status(self.endpoint)  # type: ignore[attr-defined]
        elif action == "logs":
            self.app.begin_logs(self.endpoint, follow=True)  # type: ignore[attr-defined]
        elif action == "benchmark":
            self.app.push_screen(BenchmarkOptionsScreen(self.endpoint))
        elif action == "stop":
            self.app.push_screen(StopConfirmScreen(self.endpoint))


class ConnectionInfoScreen(CopyEnabledScreen):
    """Show OpenAI-compatible connection details for one endpoint."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("u", "copy_base_url", "Copy URL", show=False),
        Binding("k", "copy_api_key", "Copy key", show=False),
    ]

    def __init__(self, endpoint: EndpointInfo) -> None:
        super().__init__()
        self.endpoint = endpoint
        self._payload: dict[str, str | None] = {}

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="screen-scroll"):
            yield Static("[bold #7bf168]Connection Info[/]")
            yield Static(
                f"[bold]{escape(_endpoint_name(self.endpoint))}[/bold]  "
                f"[dim]{escape(_endpoint_summary(self.endpoint))}[/dim]"
            )
            yield Static("", id="connection-info-fields")
            with Horizontal(id="connection-info-actions"):
                yield Button("Copy base URL", id="connection-copy-url")
                yield Button("Copy model ID", id="connection-copy-model")
                yield Button("Copy API key", id="connection-copy-key")
        yield Footer()

    def on_mount(self) -> None:
        self._payload = endpoint_connection_payload(
            self.endpoint,
            username=self._modal_username(),
        )
        self.query_one("#connection-info-fields", Static).update(
            self._fields_markup(self._payload)
        )
        has_key = bool((self._payload.get("api_key") or "").strip())
        self.query_one("#connection-copy-key", Button).display = has_key
        self.query_one("#connection-copy-url", Button).focus()

    def _modal_username(self) -> str:
        return str(getattr(self.app, "_username", "") or "")

    @staticmethod
    def _fields_markup(payload: dict[str, str | None]) -> str:
        base_url = payload.get("base_url") or "(unavailable while the app is starting)"
        model_id = payload.get("model_id") or "(unknown)"
        display_name = payload.get("display_name") or "(unknown)"
        api_key = (payload.get("api_key") or "").strip()
        key_line = (
            f"[dim]API key[/dim]   {escape(api_key)}"
            if api_key
            else "[dim]API key[/dim]   none (no auth by default)"
        )
        return (
            f"[dim]Base URL[/dim]   {escape(base_url)}\n"
            f"[dim]Model ID[/dim]   {escape(model_id)}\n"
            f"[dim]Display[/dim]    {escape(display_name)}\n"
            f"{key_line}"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "connection-copy-url":
            self.action_copy_base_url()
        elif event.button.id == "connection-copy-model":
            self._copy_field("model_id", empty_message="No model ID to copy")
        elif event.button.id == "connection-copy-key":
            self.action_copy_api_key()

    def _copy_field(self, field: str, *, empty_message: str) -> None:
        value = (self._payload.get(field) or "").strip()
        if not value:
            self.notify(empty_message, timeout=2)
            return
        self.app.copy_to_clipboard(value)
        self.notify(f"Copied {field.replace('_', ' ')}", timeout=2)

    def action_copy_base_url(self) -> None:
        self._copy_field("base_url", empty_message="No base URL to copy")

    def action_copy_api_key(self) -> None:
        self._copy_field("api_key", empty_message="No API key to copy")

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
        with VerticalScroll(classes="screen-scroll"):
            yield Static("[bold #7bf168]Status Check[/]")
            yield Static(
                f"[bold]{escape(_endpoint_name(self.endpoint))}[/bold]  "
                f"[dim]{escape(_endpoint_summary(self.endpoint))}[/dim]"
            )
            yield FormField(
                "Server URL override (optional)",
                "status-url",
                hint="Leave blank to use the endpoint URL.",
            )
            yield FormField("Timeout (seconds)", "status-timeout", default="60")
            yield Static("", id="status-feedback")
            yield Button("Check status", id="status-submit", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#status-url", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "status-submit":
            self._submit()

    def action_do_submit(self) -> None:
        self._submit()

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def _submit(self) -> None:
        timeout_text = self.query_one("#status-timeout", Input).value.strip()
        try:
            timeout = int(timeout_text or "60")
        except ValueError:
            self.query_one("#status-feedback", Static).update(
                "[red]Timeout must be an integer.[/red]"
            )
            return
        if timeout <= 0:
            self.query_one("#status-feedback", Static).update(
                "[red]Timeout must be greater than zero.[/red]"
            )
            return
        url_override = self.query_one("#status-url", Input).value.strip() or None
        self.app.pop_screen()
        self.app.begin_status(  # type: ignore[attr-defined]
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
        with VerticalScroll(classes="screen-scroll"):
            yield Static("[bold #7bf168]Benchmark[/]")
            yield Static(
                f"[bold]{escape(_endpoint_name(self.endpoint))}[/bold]  "
                f"[dim]{escape(_endpoint_summary(self.endpoint))}[/dim]"
            )
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
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#benchmark-concurrency", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "benchmark-submit":
            self._submit()

    def action_do_submit(self) -> None:
        self._submit()

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def _submit(self) -> None:
        request_count_text = self.query_one("#benchmark-request-count", Input).value.strip()
        request_count: int | None = None
        if request_count_text:
            try:
                request_count = int(request_count_text)
            except ValueError:
                self.query_one("#benchmark-feedback", Static).update(
                    "[red]Request count must be an integer.[/red]"
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
                "[red]Token lengths must be integers.[/red]"
            )
            return
        self.app.pop_screen()
        self.app.begin_benchmark(  # type: ignore[attr-defined]
            self.endpoint,
            concurrency=self.query_one("#benchmark-concurrency", Input).value,
            request_count=request_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokenizer=self.query_one("#benchmark-tokenizer", Input).value,
            output_dir=self.query_one("#benchmark-output-dir", Input).value.strip() or None,
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
        with VerticalScroll(classes="screen-scroll"):
            yield Static("[bold #7bf168]Stop Endpoint[/]")
            yield Static("")
            yield Static(
                f"Stop [bold]{escape(_endpoint_name(self.endpoint))}[/bold]?\n"
                f"[dim]{escape(self.endpoint.provider.value)}/{escape(_endpoint_backend(self.endpoint))} · "
                f"{escape(self.endpoint.app_id or self.endpoint.name)}[/dim]"
            )
            yield Static(
                "[yellow]This will terminate the selected deployment.[/yellow]",
                id="stop-warning",
            )
            with Horizontal(id="stop-confirm-actions"):
                yield Button("Cancel", id="stop-cancel")
                yield Button("Stop endpoint", id="stop-confirm", variant="error")
        yield Footer()

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
        self.app.begin_stop(self.endpoint)  # type: ignore[attr-defined]
