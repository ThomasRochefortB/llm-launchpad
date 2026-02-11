"""Monitor screen: real-time log streaming and operation status.

Shows the status header + scrolling log output for any running
operation (deploy, warmup, logs, status, stop).
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, Static

from ...protocol.enums import DeploymentState, OperationType
from ..widgets.log_viewer import LogViewer
from ..widgets.status_header import StatusHeader
from ..workers import LogMessage, OperationDone, OperationError, StateChanged


class MonitorScreen(Screen):
    """Full-screen operation monitor with streaming logs."""

    BINDINGS = [
        Binding("escape", "go_back", "Back", show=True),
        Binding("q", "go_back", "Back"),
        Binding("c", "clear_log", "Clear log", show=True),
    ]

    def __init__(self, title: str = "Operation") -> None:
        super().__init__()
        self._title = title
        self._done = False

    def compose(self) -> ComposeResult:
        yield StatusHeader(id="monitor-status-header")
        with Vertical(id="monitor-layout"):
            yield Static(
                f"[bold cyan]{self._title}[/bold cyan]  [dim]press esc to return[/dim]",
                id="monitor-title",
            )
            yield LogViewer(id="monitor-log-viewer")
        yield Footer()

    @property
    def log_viewer(self) -> LogViewer:
        return self.query_one("#monitor-log-viewer", LogViewer)

    @property
    def status_header(self) -> StatusHeader:
        return self.query_one("#monitor-status-header", StatusHeader)

    # -- Message handlers --

    def on_log_message(self, message: LogMessage) -> None:
        style = "red" if message.stream == "stderr" else ""
        self.log_viewer.write_line(message.line, style=style)

    def on_state_changed(self, message: StateChanged) -> None:
        self.status_header.update_from_event(
            state=message.state,
            operation=message.operation,
            detail=message.detail,
        )

    def on_operation_done(self, message: OperationDone) -> None:
        self._done = True
        if message.success:
            self.log_viewer.write_line(
                f"\n[green bold]  Operation complete ({message.operation.value})[/green bold]",
            )
        else:
            self.log_viewer.write_line(
                f"\n[red bold]  Operation failed (exit code {message.exit_code})[/red bold]",
            )
            if message.detail:
                self.log_viewer.write_line(f"[red]{message.detail}[/red]")
        self.log_viewer.write_line("\n[dim]Press esc or q to return.[/dim]")

    def on_operation_error(self, message: OperationError) -> None:
        self.log_viewer.write_line(f"[red bold]Error:[/red bold] {message.message}")

    # -- Actions --

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_clear_log(self) -> None:
        self.log_viewer.clear()
