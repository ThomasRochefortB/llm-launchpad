"""Manage endpoints screen: list, status, logs, stop."""

from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Button, Footer, Input, OptionList, Static, Switch
from textual.widgets.option_list import Option

from ...protocol.models import EndpointInfo
from ..widgets.input_form import FormField, ToggleField
from .copy_enabled import CopyEnabledScreen


def _is_stoppable_state(state: str) -> bool:
    """Return True when the app state should appear in the manage UI."""
    return state.strip().lower() in {
        "building",
        "deployed",
        "deploying",
        "ephemeral",
        "initializing",
        "pending",
        "queued",
        "running",
        "starting",
    }


def _build_backend_app_options(
    instances: list[EndpointInfo],
) -> tuple[list[Option], dict[str, EndpointInfo]]:
    """Build options and map option IDs to the originating Modal row."""
    options: list[Option] = []
    option_to_target: dict[str, EndpointInfo] = {}

    for index, row in enumerate(instances):
        if row.backend is None:
            continue
        option_id = f"app-id:{row.app_id}" if row.app_id else f"app-name:{row.name}:{index}"
        label = f"  [{row.backend.value}] {row.name}  ({row.state}"
        if row.app_id:
            label += f", {row.app_id}"
        label += ")"
        options.append(Option(label, id=option_id))
        option_to_target[option_id] = row

    return options, option_to_target


def _set_option_list_options(option_list: OptionList, options: list[Option]) -> None:
    """Replace all options using the Textual API available across supported versions."""
    setter = getattr(option_list, "set_options", None)
    if callable(setter):
        setter(options)
        return
    option_list.clear_options()
    option_list.add_options(options)


class ManageScreen(CopyEnabledScreen):
    """Manage endpoints: pick action, then pick backend and params."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="menu-container"):
            yield Static("[bold #7bf168]Manage Endpoints[/]")
            yield Static("")
            yield OptionList(
                Option("  List apps                 Show active launchpad Modal apps", id="list"),
                Option("  Status check              Probe endpoint health", id="status"),
                Option("  Tail logs                 Stream Modal app logs", id="logs"),
                Option("  Benchmark                 Measure endpoint throughput", id="benchmark"),
                Option("  Stop app                  Stop an active Modal app", id="stop"),
                id="manage-action-list",
            )
        yield Footer()

    def on_mount(self) -> None:
        action_list = self.query_one("#manage-action-list", OptionList)
        if action_list.option_count > 0:
            action_list.highlighted = 0
        action_list.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        opt = event.option.id
        if opt == "list":
            self.app.begin_list()  # type: ignore[attr-defined]
        elif opt == "status":
            self.app.push_screen(StatusParamsScreen())
        elif opt == "logs":
            self.app.push_screen(LogsParamsScreen())
        elif opt == "benchmark":
            self.app.push_screen(BenchmarkParamsScreen())
        elif opt == "stop":
            self.app.push_screen(StopParamsScreen())

    def action_pop_screen(self) -> None:
        self.app.pop_screen()


class StatusParamsScreen(CopyEnabledScreen):
    """Params for status check: backend, optional URL override, timeout."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("enter", "do_submit", "Submit", show=True),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="menu-container"):
            yield Static("[bold #7bf168]Status Check[/]")
            yield Static("")
            yield Static("[bold]Choose active app[/bold]")
            yield OptionList(id="status-instance-list")
            yield FormField(
                "Server URL override (optional)",
                "status-url",
                hint="Leave blank for default",
            )
            yield FormField("Timeout (seconds)", "status-timeout", default="60")
            yield Static("")
        yield Footer()

    def on_mount(self) -> None:
        self._selected_row: EndpointInfo | None = None
        self._row_by_option_id: dict[str, EndpointInfo] = {}
        self._load_instances()
        self.query_one("#status-instance-list", OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "status-instance-list":
            selected = str(event.option.id)
            row = self._row_by_option_id.get(selected)
            if row is None:
                return
            self._selected_row = row
            self._submit()

    def action_do_submit(self) -> None:
        self._submit()

    def _submit(self) -> None:
        if self._selected_row is None or self._selected_row.backend is None:
            return
        url = self.query_one("#status-url", Input).value.strip() or self._selected_row.web_url or None
        timeout_str = self.query_one("#status-timeout", Input).value.strip()
        try:
            timeout = int(timeout_str) if timeout_str else 60
        except ValueError:
            timeout = 60
        self.app.begin_status(  # type: ignore[attr-defined]
            self._selected_row.backend,
            url,
            timeout,
            app_name=self._selected_row.name,
            served_model_name=self._selected_row.served_model_name,
        )

    def _load_instances(self) -> None:
        instance_list = self.query_one("#status-instance-list", OptionList)
        instances = self.app.list_instances()  # type: ignore[attr-defined]
        checkable_instances = [row for row in instances if _is_stoppable_state(row.state)]
        if not checkable_instances:
            self._row_by_option_id = {}
            _set_option_list_options(instance_list, [Option("  No active apps found")])
            if instance_list.option_count > 0:
                instance_list.highlighted = 0
            return
        options, self._row_by_option_id = _build_backend_app_options(checkable_instances)
        _set_option_list_options(instance_list, options)
        if options:
            instance_list.highlighted = 0

    def action_pop_screen(self) -> None:
        self.app.pop_screen()


class LogsParamsScreen(CopyEnabledScreen):
    """Params for log tailing: backend, follow toggle."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="menu-container"):
            yield Static("[bold #7bf168]Tail Logs[/]")
            yield Static("")
            yield Static("[bold]Choose active app[/bold]")
            yield OptionList(id="logs-instance-list")
            yield ToggleField("Follow log stream", "logs-follow", default=True)
        yield Footer()

    def on_mount(self) -> None:
        self._selected_row: EndpointInfo | None = None
        self._target_by_option_id: dict[str, EndpointInfo] = {}
        self._load_instances()
        self.query_one("#logs-instance-list", OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "logs-instance-list":
            selected = str(event.option.id)
            self._selected_row = self._target_by_option_id.get(selected)
            if self._selected_row is None:
                return
            self._submit()

    def _submit(self) -> None:
        if self._selected_row is None or self._selected_row.backend is None:
            return
        follow = self.query_one("#logs-follow", Switch).value
        self.app.begin_logs(  # type: ignore[attr-defined]
            self._selected_row.backend,
            follow,
            app_name=self._selected_row.name,
            app_id=self._selected_row.app_id or None,
        )

    def _load_instances(self) -> None:
        instance_list = self.query_one("#logs-instance-list", OptionList)
        instances = self.app.list_instances()  # type: ignore[attr-defined]
        loggable_instances = [row for row in instances if _is_stoppable_state(row.state)]
        if not loggable_instances:
            self._target_by_option_id = {}
            _set_option_list_options(instance_list, [Option("  No active apps found")])
            if instance_list.option_count > 0:
                instance_list.highlighted = 0
            return
        options, self._target_by_option_id = _build_backend_app_options(loggable_instances)
        _set_option_list_options(instance_list, options)
        if options:
            instance_list.highlighted = 0

    def action_pop_screen(self) -> None:
        self.app.pop_screen()


class BenchmarkParamsScreen(CopyEnabledScreen):
    """Params for running an AIPerf benchmark against an active endpoint."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("enter", "do_submit", "Benchmark", show=True),
        Binding("ctrl+b", "do_submit", "Benchmark", show=True),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="menu-container"):
            yield Static("[bold #7bf168]Benchmark[/]")
            yield Static("")
            yield Static("[bold]Choose active app[/bold]")
            yield OptionList(id="benchmark-instance-list")
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
        self._selected_row: EndpointInfo | None = None
        self._row_by_option_id: dict[str, EndpointInfo] = {}
        self._load_instances()
        self.query_one("#benchmark-instance-list", OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "benchmark-instance-list":
            return
        selected = str(event.option.id)
        self._selected_row = self._row_by_option_id.get(selected)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "benchmark-submit":
            self._submit()

    def action_do_submit(self) -> None:
        self._submit()

    def _submit(self) -> None:
        row = self._selected_row or self._highlighted_row()
        if row is None or row.backend is None:
            self.query_one("#benchmark-feedback", Static).update("[yellow]Choose an app first.[/yellow]")
            return
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
            input_tokens = int(self.query_one("#benchmark-input-tokens", Input).value.strip() or "550")
            output_tokens = int(self.query_one("#benchmark-output-tokens", Input).value.strip() or "256")
        except ValueError:
            self.query_one("#benchmark-feedback", Static).update(
                "[red]Token lengths must be integers.[/red]"
            )
            return
        self.app.begin_benchmark(  # type: ignore[attr-defined]
            row,
            concurrency=self.query_one("#benchmark-concurrency", Input).value,
            request_count=request_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokenizer=self.query_one("#benchmark-tokenizer", Input).value,
            output_dir=self.query_one("#benchmark-output-dir", Input).value.strip() or None,
        )

    def _highlighted_row(self) -> EndpointInfo | None:
        instance_list = self.query_one("#benchmark-instance-list", OptionList)
        highlighted = instance_list.highlighted_option
        if highlighted is None:
            return None
        return self._row_by_option_id.get(str(highlighted.id))

    def _load_instances(self) -> None:
        instance_list = self.query_one("#benchmark-instance-list", OptionList)
        instances = self.app.list_instances()  # type: ignore[attr-defined]
        benchmarkable_instances = [row for row in instances if _is_stoppable_state(row.state)]
        if not benchmarkable_instances:
            self._row_by_option_id = {}
            _set_option_list_options(instance_list, [Option("  No active apps found")])
            if instance_list.option_count > 0:
                instance_list.highlighted = 0
            return
        options, self._row_by_option_id = _build_backend_app_options(benchmarkable_instances)
        _set_option_list_options(instance_list, options)
        if options:
            instance_list.highlighted = 0

    def action_pop_screen(self) -> None:
        self.app.pop_screen()


class StopParamsScreen(CopyEnabledScreen):
    """Confirmation for stopping a backend."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("enter", "do_submit", "Stop", show=True),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="menu-container"):
            yield Static("[bold #7bf168]Stop App[/]")
            yield Static("")
            yield Static("[bold]Choose active app[/bold]")
            yield OptionList(id="stop-instance-list")
            yield Static("")
            yield Static("[yellow]Warning:[/yellow] This will stop the selected Modal app.")
            yield Static("[dim]Select an app to confirm and stop.[/dim]")
        yield Footer()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "stop-instance-list":
            selected = str(event.option.id)
            row = self._target_by_option_id.get(selected)
            if row is None or row.backend is None:
                return
            self.app.begin_stop(  # type: ignore[attr-defined]
                row.backend,
                app_name=row.name,
                app_id=row.app_id or None,
            )

    def on_mount(self) -> None:
        self._target_by_option_id: dict[str, EndpointInfo] = {}
        self._was_suspended = False
        self._load_instances()
        self.query_one("#stop-instance-list", OptionList).focus()

    def on_screen_suspend(self, _: events.ScreenSuspend) -> None:
        self._was_suspended = True

    def on_screen_resume(self, _: events.ScreenResume) -> None:
        """Refresh instance list when returning from the stop monitor screen."""
        if self._was_suspended:
            self._was_suspended = False
            self._load_instances()

    def action_do_submit(self) -> None:
        instance_list = self.query_one("#stop-instance-list", OptionList)
        highlighted = instance_list.highlighted_option
        if highlighted is None:
            return
        selected = str(highlighted.id)
        row = self._target_by_option_id.get(selected)
        if row is None or row.backend is None:
            return
        self.app.begin_stop(  # type: ignore[attr-defined]
            row.backend,
            app_name=row.name,
            app_id=row.app_id or None,
        )

    def _load_instances(self) -> None:
        instance_list = self.query_one("#stop-instance-list", OptionList)
        instances = self.app.list_instances()  # type: ignore[attr-defined]
        stoppable_instances = [row for row in instances if _is_stoppable_state(row.state)]
        if not stoppable_instances:
            self._target_by_option_id = {}
            _set_option_list_options(instance_list, [Option("  No active apps found")])
            if instance_list.option_count > 0:
                instance_list.highlighted = 0
            return
        options, self._target_by_option_id = _build_backend_app_options(stoppable_instances)
        _set_option_list_options(instance_list, options)
        if options:
            instance_list.highlighted = 0

    def action_pop_screen(self) -> None:
        self.app.pop_screen()
