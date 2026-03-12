"""Monitor screen: real-time log streaming and operation status.

Shows the status header + scrolling log output for any running
operation (deploy, warmup, logs, status, stop).
"""

from __future__ import annotations

import re

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Static

from ...protocol.enums import BackendType, OperationType
from ..deploy_log_summary import DeployLogSummarizer
from .copy_enabled import CopyEnabledScreen
from ..widgets.log_viewer import LogViewer
from ..widgets.status_header import StatusHeader
from ..workers import LogMessage, OperationDone, OperationError, StateChanged

_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text or "")


class MonitorScreen(CopyEnabledScreen):
    """Full-screen operation monitor with streaming logs."""

    BINDINGS = [
        Binding("escape", "go_back", "Back", show=True),
        Binding("q", "go_back", "Back"),
        Binding(
            "y",
            "copy_text",
            "Copy",
            key_display="y",
            show=True,
        ),
        Binding(
            "ctrl+shift+c,super+c,meta+c,cmd+c,command+c",
            "copy_text_to_clipboard",
            "Copy",
            show=False,
        ),
        Binding("ctrl+l", "clear_log", "Clear log", show=True),
    ]

    def __init__(
        self,
        title: str = "Operation",
        deploy_backend: BackendType | None = None,
        summarize_backend_logs: bool = False,
        show_debug_logs: bool = True,
    ) -> None:
        super().__init__()
        self._title = title
        self._done = False
        self._deploy_backend = deploy_backend
        self._summarize_backend_logs = summarize_backend_logs
        self._show_debug_logs = show_debug_logs
        self._current_operation: OperationType | None = None
        self._deploy_summarizer = (
            DeployLogSummarizer(deploy_backend)
            if deploy_backend is not None and summarize_backend_logs and not show_debug_logs
            else None
        )

    def compose(self) -> ComposeResult:
        mouse_enabled = getattr(self.app, "mouse_enabled", True)
        copy_help = (
            "terminal selection mode  use your terminal copy shortcut  ctrl+c exits"
            if not mouse_enabled
            else "drag to select, dbl-click for line  ctrl+shift+c copy  y fallback  ctrl+c exits"
        )
        yield StatusHeader(id="monitor-status-header")
        with Vertical(id="monitor-layout"):
            yield Static(
                (
                    f"[bold cyan]{self._title}[/bold cyan]  "
                    f"[dim]{copy_help}  ctrl+l clear  esc to return[/dim]"
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

    @property
    def _summary_mode_enabled(self) -> bool:
        return self._deploy_summarizer is not None

    # -- Message handlers --

    def on_mount(self) -> None:
        if self._summary_mode_enabled:
            self.log_viewer.write_line(
                "Log view: summary (normalized milestones; raw backend logs hidden)"
            )
            self.log_viewer.write_line("")

    def on_log_message(self, message: LogMessage) -> None:
        cleaned = _strip_ansi(message.line)
        prefix = "stderr | " if message.stream == "stderr" else ""
        if self._summary_mode_enabled:
            assert self._deploy_summarizer is not None
            for line in self._deploy_summarizer.transform(cleaned, self._current_operation):
                self.log_viewer.write_line(f"{prefix}{line}" if prefix else line)
            return
        self.log_viewer.write_line(f"{prefix}{cleaned}")

    def on_state_changed(self, message: StateChanged) -> None:
        if message.operation is not None:
            self._current_operation = message.operation
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
            if self._summary_mode_enabled:
                self.log_viewer.write_line(
                    'Tip: re-run with "Show debug logs" enabled to see full backend logs.'
                )
        self.log_viewer.write_line("Press esc or q to return.")

    def on_operation_error(self, message: OperationError) -> None:
        self.log_viewer.write_line(f"Error: {message.message}")

    def _selected_text_for_copy(self) -> str | None:
        """Return selected log text before falling back to screen selections."""
        log_widget = self.log_viewer.log_widget
        getter = getattr(log_widget, "get_selected_text", None)
        text = None
        if callable(getter):
            text = getter()
        if not text:
            text = self.get_selected_text()
        return text

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_clear_log(self) -> None:
        self.log_viewer.clear()
