"""Monitor screen: real-time log streaming and operation status.

Shows the status header + scrolling log output for any running
operation (deploy, warmup, logs, status, stop).
"""

from __future__ import annotations

import subprocess
import sys

from textual.actions import SkipAction
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, Static

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
                (
                    f"[bold cyan]{self._title}[/bold cyan]  "
                    "[dim]drag to select, dbl-click for line  "
                    "ctrl+c to copy  esc to return[/dim]"
                ),
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
        prefix = "stderr | " if message.stream == "stderr" else ""
        self.log_viewer.write_line(f"{prefix}{message.line}")

    def on_state_changed(self, message: StateChanged) -> None:
        self.status_header.update_from_event(
            state=message.state,
            operation=message.operation,
            detail=message.detail,
        )

    def on_operation_done(self, message: OperationDone) -> None:
        self._done = True
        self.log_viewer.write_line("")
        if message.success:
            self.log_viewer.write_line(
                f"Operation complete ({message.operation.value}).",
            )
        else:
            self.log_viewer.write_line(
                f"Operation failed (exit code {message.exit_code}).",
            )
            if message.detail:
                self.log_viewer.write_line(f"Detail: {message.detail}")
        self.log_viewer.write_line("Press esc or q to return.")

    def on_operation_error(self, message: OperationError) -> None:
        self.log_viewer.write_line(f"Error: {message.message}")

    # -- Actions --

    def action_copy_text(self) -> None:
        """Copy selected log text to clipboard.

        Uses OSC 52 (built-in) **and** ``pbcopy`` on macOS as a fallback so
        the copy works even in terminals that don't support OSC 52.
        """
        text = self.get_selected_text()
        if text is None:
            raise SkipAction()

        # OSC 52 (works in iTerm2, Kitty, WezTerm, Ghostty, etc.)
        self.app.copy_to_clipboard(text)

        # macOS: also pipe into pbcopy for terminals that ignore OSC 52
        if sys.platform == "darwin":
            try:
                subprocess.run(
                    ["pbcopy"],
                    input=text.encode(),
                    check=True,
                    timeout=2,
                )
            except Exception:
                pass

        lines = text.count("\n") + 1
        self.notify(f"Copied {lines} line{'s' if lines != 1 else ''}", timeout=2)

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_clear_log(self) -> None:
        self.log_viewer.clear()
