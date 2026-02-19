"""Manage endpoints screen: list, status, logs, stop."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Input, OptionList, Static, Switch
from textual.widgets.option_list import Option

from ...protocol.enums import BackendType
from ...protocol.models import EndpointInfo
from ..widgets.input_form import FormField, ToggleField
from .copy_enabled import CopyEnabledScreen


def _is_stoppable_state(state: str) -> bool:
    """Return True when the app state should appear in Stop UI."""
    return state.strip().lower() in {"deployed", "running"}


def _build_backend_app_options(
    instances: list[EndpointInfo],
) -> tuple[list[Option], dict[str, tuple[BackendType, str]]]:
    """Build options and map option IDs to backend/app pairs."""
    options: list[Option] = []
    option_to_target: dict[str, tuple[BackendType, str]] = {}

    for index, row in enumerate(instances):
        if row.backend is None:
            continue
        option_id = f"app-id:{row.app_id}" if row.app_id else f"app-name:{row.name}:{index}"
        label = f"  [{row.backend.value}] {row.name}  ({row.state})"
        options.append(Option(label, id=option_id))
        option_to_target[option_id] = (row.backend, row.name)

    return options, option_to_target


class ManageScreen(CopyEnabledScreen):
    """Manage endpoints: pick action, then pick backend and params."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="menu-container"):
            yield Static("[bold cyan]Manage Endpoints[/bold cyan]")
            yield Static("")
            yield OptionList(
                Option("  List deployments          Show deployed launchpad apps", id="list"),
                Option("  Status check              Probe endpoint health", id="status"),
                Option("  Tail logs                 Stream Modal app logs", id="logs"),
                Option("  Stop deployment           Stop a running app", id="stop"),
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
            yield Static("[bold cyan]Status Check[/bold cyan]")
            yield Static("")
            yield Static("[bold]Choose running instance[/bold]")
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
        self._backend: BackendType | None = None
        self._instance_app_name: str | None = None
        self._target_by_option_id: dict[str, tuple[BackendType, str]] = {}
        self._load_instances()
        self.query_one("#status-instance-list", OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "status-instance-list":
            selected = str(event.option.id)
            target = self._target_by_option_id.get(selected)
            if target is None:
                return
            self._backend, self._instance_app_name = target
            self._submit()

    def action_do_submit(self) -> None:
        self._submit()

    def _submit(self) -> None:
        if self._backend is None or self._instance_app_name is None:
            return
        url = self.query_one("#status-url", Input).value.strip() or None
        timeout_str = self.query_one("#status-timeout", Input).value.strip()
        try:
            timeout = int(timeout_str) if timeout_str else 60
        except ValueError:
            timeout = 60
        self.app.begin_status(  # type: ignore[attr-defined]
            self._backend,
            url,
            timeout,
            app_name=self._instance_app_name,
        )

    def _load_instances(self) -> None:
        instance_list = self.query_one("#status-instance-list", OptionList)
        instances: list[EndpointInfo] = []
        for backend in (BackendType.LLAMACPP, BackendType.VLLM):
            instances.extend(self.app.list_instances(backend))  # type: ignore[attr-defined]
        checkable_instances = [row for row in instances if _is_stoppable_state(row.state)]
        if not checkable_instances:
            self._target_by_option_id = {}
            instance_list.set_options([Option("  No running deployments found")])
            if instance_list.option_count > 0:
                instance_list.highlighted = 0
            return
        options, self._target_by_option_id = _build_backend_app_options(checkable_instances)
        instance_list.set_options(options)
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
            yield Static("[bold cyan]Tail Logs[/bold cyan]")
            yield Static("")
            yield Static("[bold]Choose running instance[/bold]")
            yield OptionList(id="logs-instance-list")
            yield ToggleField("Follow log stream", "logs-follow", default=True)
        yield Footer()

    def on_mount(self) -> None:
        self._backend: BackendType | None = None
        self._instance_app_name: str | None = None
        self._target_by_option_id: dict[str, tuple[BackendType, str]] = {}
        self._load_instances()
        self.query_one("#logs-instance-list", OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "logs-instance-list":
            selected = str(event.option.id)
            target = self._target_by_option_id.get(selected)
            if target is None:
                return
            self._backend, self._instance_app_name = target
            self._submit()

    def _submit(self) -> None:
        if self._backend is None or self._instance_app_name is None:
            return
        follow = self.query_one("#logs-follow", Switch).value
        self.app.begin_logs(self._backend, follow, app_name=self._instance_app_name)  # type: ignore[attr-defined]

    def _load_instances(self) -> None:
        instance_list = self.query_one("#logs-instance-list", OptionList)
        instances: list[EndpointInfo] = []
        for backend in (BackendType.LLAMACPP, BackendType.VLLM):
            instances.extend(self.app.list_instances(backend))  # type: ignore[attr-defined]
        loggable_instances = [row for row in instances if _is_stoppable_state(row.state)]
        if not loggable_instances:
            self._target_by_option_id = {}
            instance_list.set_options([Option("  No running deployments found")])
            if instance_list.option_count > 0:
                instance_list.highlighted = 0
            return
        options, self._target_by_option_id = _build_backend_app_options(loggable_instances)
        instance_list.set_options(options)
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
            yield Static("[bold cyan]Stop Deployment[/bold cyan]")
            yield Static("")
            yield Static("[bold]Choose running instance[/bold]")
            yield OptionList(id="stop-instance-list")
            yield Static("")
            yield Static("[yellow]Warning:[/yellow] This will stop the running deployment.")
            yield Static("[dim]Select an instance to confirm and stop.[/dim]")
        yield Footer()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "stop-instance-list":
            selected = str(event.option.id)
            target = self._target_by_option_id.get(selected)
            if target is None:
                return
            backend, app_name = target
            self.app.begin_stop(backend, app_name=app_name)  # type: ignore[attr-defined]

    def on_mount(self) -> None:
        self._target_by_option_id: dict[str, tuple[BackendType, str]] = {}
        self._load_instances()
        self.query_one("#stop-instance-list", OptionList).focus()

    def action_do_submit(self) -> None:
        instance_list = self.query_one("#stop-instance-list", OptionList)
        highlighted = instance_list.highlighted_option
        if highlighted is None:
            return
        selected = str(highlighted.id)
        target = self._target_by_option_id.get(selected)
        if target is None:
            return
        backend, app_name = target
        self.app.begin_stop(backend, app_name=app_name)  # type: ignore[attr-defined]

    def _load_instances(self) -> None:
        instance_list = self.query_one("#stop-instance-list", OptionList)
        instances: list[EndpointInfo] = []
        for backend in (BackendType.LLAMACPP, BackendType.VLLM):
            instances.extend(self.app.list_instances(backend))  # type: ignore[attr-defined]
        stoppable_instances = [row for row in instances if _is_stoppable_state(row.state)]
        if not stoppable_instances:
            self._target_by_option_id = {}
            instance_list.set_options([Option("  No running deployments found")])
            if instance_list.option_count > 0:
                instance_list.highlighted = 0
            return
        options, self._target_by_option_id = _build_backend_app_options(stoppable_instances)
        instance_list.set_options(options)
        if options:
            instance_list.highlighted = 0

    def action_pop_screen(self) -> None:
        self.app.pop_screen()
