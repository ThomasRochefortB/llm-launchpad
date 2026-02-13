"""Manage endpoints screen: list, status, logs, stop."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, Input, OptionList, Static, Switch
from textual.widgets.option_list import Option

from ...protocol.enums import BackendType
from ...core.naming import legacy_app_name
from ..widgets.input_form import FormField, ToggleField


class ManageScreen(Screen):
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


class BackendPickMixin:
    """Shared backend selection for manage sub-screens."""

    pass


class StatusParamsScreen(Screen):
    """Params for status check: backend, optional URL override, timeout."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("enter", "do_submit", "Submit", show=True),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="menu-container"):
            yield Static("[bold cyan]Status Check[/bold cyan]")
            yield Static("")
            yield Static("[bold]Choose backend[/bold]")
            yield OptionList(
                Option("  llama.cpp (llamacpp-server)", id="be-llamacpp"),
                Option("  vLLM (vllm-server)", id="be-vllm"),
                id="status-backend-list",
            )
            yield Static("[bold]Choose instance[/bold]")
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

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "status-backend-list":
            if event.option.id == "be-llamacpp":
                self._backend = BackendType.LLAMACPP
            elif event.option.id == "be-vllm":
                self._backend = BackendType.VLLM
            self._load_instances()
            return
        if event.option_list.id == "status-instance-list":
            self._instance_app_name = event.option.id
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
        if self._backend is None:
            return
        instance_list = self.query_one("#status-instance-list", OptionList)
        options = []
        instances = self.app.list_instances(self._backend)  # type: ignore[attr-defined]
        if instances:
            for row in instances:
                label = f"  {row.name}  ({row.state})"
                options.append(Option(label, id=row.name))
        else:
            fallback = legacy_app_name(self._backend)
            options.append(Option(f"  {fallback}  (legacy default)", id=fallback))
        instance_list.set_options(options)

    def action_pop_screen(self) -> None:
        self.app.pop_screen()


class LogsParamsScreen(Screen):
    """Params for log tailing: backend, follow toggle."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="menu-container"):
            yield Static("[bold cyan]Tail Logs[/bold cyan]")
            yield Static("")
            yield Static("[bold]Choose backend[/bold]")
            yield OptionList(
                Option("  llama.cpp (llamacpp-server)", id="log-be-llamacpp"),
                Option("  vLLM (vllm-server)", id="log-be-vllm"),
                id="logs-backend-list",
            )
            yield Static("[bold]Choose instance[/bold]")
            yield OptionList(id="logs-instance-list")
            yield ToggleField("Follow log stream", "logs-follow", default=True)
        yield Footer()

    def on_mount(self) -> None:
        self._backend: BackendType | None = None
        self._instance_app_name: str | None = None

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "logs-backend-list":
            if event.option.id == "log-be-llamacpp":
                self._backend = BackendType.LLAMACPP
            elif event.option.id == "log-be-vllm":
                self._backend = BackendType.VLLM
            self._load_instances()
            return
        if event.option_list.id == "logs-instance-list":
            self._instance_app_name = event.option.id
            self._submit()

    def _submit(self) -> None:
        if self._backend is None or self._instance_app_name is None:
            return
        follow = self.query_one("#logs-follow", Switch).value
        self.app.begin_logs(self._backend, follow, app_name=self._instance_app_name)  # type: ignore[attr-defined]

    def _load_instances(self) -> None:
        if self._backend is None:
            return
        instance_list = self.query_one("#logs-instance-list", OptionList)
        options = []
        instances = self.app.list_instances(self._backend)  # type: ignore[attr-defined]
        if instances:
            for row in instances:
                label = f"  {row.name}  ({row.state})"
                options.append(Option(label, id=row.name))
        else:
            fallback = legacy_app_name(self._backend)
            options.append(Option(f"  {fallback}  (legacy default)", id=fallback))
        instance_list.set_options(options)

    def action_pop_screen(self) -> None:
        self.app.pop_screen()


class StopParamsScreen(Screen):
    """Confirmation for stopping a backend."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="menu-container"):
            yield Static("[bold cyan]Stop Deployment[/bold cyan]")
            yield Static("")
            yield Static("[bold]Choose backend to stop[/bold]")
            yield OptionList(
                Option("  llama.cpp (llamacpp-server)", id="stop-be-llamacpp"),
                Option("  vLLM (vllm-server)", id="stop-be-vllm"),
                id="stop-backend-list",
            )
            yield Static("[bold]Choose instance[/bold]")
            yield OptionList(id="stop-instance-list")
            yield Static("")
            yield Static("[yellow]Warning:[/yellow] This will stop the running deployment.")
            yield Static("[dim]Select a backend above to confirm and stop.[/dim]")
        yield Footer()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "stop-backend-list":
            if event.option.id == "stop-be-llamacpp":
                self._backend = BackendType.LLAMACPP
            elif event.option.id == "stop-be-vllm":
                self._backend = BackendType.VLLM
            self._load_instances()
            return
        if event.option_list.id == "stop-instance-list":
            if self._backend is None:
                return
            self.app.begin_stop(self._backend, app_name=event.option.id)  # type: ignore[attr-defined]

    def on_mount(self) -> None:
        self._backend: BackendType | None = None

    def _load_instances(self) -> None:
        if self._backend is None:
            return
        instance_list = self.query_one("#stop-instance-list", OptionList)
        options = []
        instances = self.app.list_instances(self._backend)  # type: ignore[attr-defined]
        if instances:
            for row in instances:
                label = f"  {row.name}  ({row.state})"
                options.append(Option(label, id=row.name))
        else:
            fallback = legacy_app_name(self._backend)
            options.append(Option(f"  {fallback}  (legacy default)", id=fallback))
        instance_list.set_options(options)

    def action_pop_screen(self) -> None:
        self.app.pop_screen()
