"""Monitor screen: real-time log streaming and operation status.

Shows the status header + scrolling log output for any running
operation (deploy, warmup, logs, status, stop).
"""

from __future__ import annotations

import re

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Static

from ...protocol.enums import BackendType, DeploymentState, OperationType
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

    AUTO_FOCUS = "#log-output"

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
        Binding("pageup", "page_up_log", "Page up", show=True, priority=True),
        Binding("pagedown", "page_down_log", "Page down", show=True, priority=True),
        Binding("end", "resume_follow", "Follow", show=True, priority=True),
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
        self._last_summary_state_detail = ""
        self._following = True
        self._unseen_lines = 0
        self._line_count = 0
        self._deploy_summarizer = (
            DeployLogSummarizer(deploy_backend)
            if deploy_backend is not None and summarize_backend_logs and not show_debug_logs
            else None
        )

    def compose(self) -> ComposeResult:
        with Vertical(id="monitor-layout"):
            yield StatusHeader(id="monitor-status-header")
            with Horizontal(id="monitor-toolbar"):
                yield Static(self._title_markup(), id="monitor-title")
                yield Static(self._view_status_markup(), id="monitor-view-status")
            yield LogViewer(id="monitor-log-viewer")
        yield Footer()

    def _title_markup(self) -> str:
        """Render a compact operation title and input-mode badge."""
        mouse_enabled = getattr(self.app, "mouse_enabled", True)
        mode = "MOUSE" if mouse_enabled else "TERMINAL SELECT"
        return (
            f"[bold #7bf168]{escape(self._title)}[/]  "
            f"[dim]·[/dim]  [#93a596]{mode}[/]"
        )

    def _view_status_markup(self) -> str:
        """Render live follow and line-count state for the log viewport."""
        line_label = "line" if self._line_count == 1 else "lines"
        if self._following:
            state = "[bold #7bf168]FOLLOWING[/]"
        else:
            new_label = "line" if self._unseen_lines == 1 else "lines"
            state = (
                f"[bold yellow]PAUSED[/]  [yellow]· {self._unseen_lines} new {new_label}[/]"
            )
        return f"{state}  [dim]· {self._line_count} {line_label}[/dim]"

    def refresh_copy_help(self) -> None:
        """Refresh compact chrome after toggling app/terminal mouse mode."""
        self.query_one("#monitor-title", Static).update(self._title_markup())
        self.query_one("#monitor-view-status", Static).update(
            self._view_status_markup()
        )

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
            self.status_header.update_from_event(
                state=DeploymentState.QUEUED,
                backend=self._deploy_backend,
                operation=OperationType.DEPLOY,
                detail="Preparing deployment",
            )
            self.log_viewer.write_line(
                "Log view: summary (normalized milestones; raw backend logs hidden)"
            )
            self.log_viewer.write_line("")
            self.log_viewer.write_line("Preparing deployment...")
            self._last_summary_state_detail = "Preparing deployment"

    def on_log_message(self, message: LogMessage) -> None:
        prefix = "stderr | " if message.stream == "stderr" else ""
        if self._summary_mode_enabled:
            assert self._deploy_summarizer is not None
            cleaned = _strip_ansi(message.line)
            if message.is_milestone:
                self.log_viewer.write_line(f"{prefix}{cleaned}" if prefix else cleaned)
                return
            for line in self._deploy_summarizer.transform(cleaned, self._current_operation):
                self.log_viewer.write_line(f"{prefix}{line}" if prefix else line)
            return
        line = (
            message.line
            if self._show_debug_logs and self._summarize_backend_logs
            else _strip_ansi(message.line)
        )
        self.log_viewer.write_line(f"{prefix}{line}")

    def on_state_changed(self, message: StateChanged) -> None:
        if message.operation is not None:
            self._current_operation = message.operation
        self.status_header.update_from_event(
            state=message.state,
            operation=message.operation,
            detail=message.detail,
        )
        detail = _strip_ansi(message.detail).strip()
        if (
            self._summary_mode_enabled
            and detail
            and detail != self._last_summary_state_detail
        ):
            self.log_viewer.write_line(detail)
            self._last_summary_state_detail = detail

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

    def on_log_viewer_status_changed(self, message: LogViewer.StatusChanged) -> None:
        """Keep compact monitor chrome synchronized with the log viewport."""
        self._following = message.following
        self._unseen_lines = message.unseen_lines
        self._line_count = message.line_count
        self.query_one("#monitor-view-status", Static).update(
            self._view_status_markup()
        )

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

    def action_page_up_log(self) -> None:
        self.log_viewer.page_up()

    def action_page_down_log(self) -> None:
        self.log_viewer.page_down()

    def action_resume_follow(self) -> None:
        self.log_viewer.resume_following()

    def action_clear_log(self) -> None:
        self.log_viewer.clear()
        self._following = True
        self._unseen_lines = 0
        self._line_count = 0
        self.query_one("#monitor-view-status", Static).update(
            self._view_status_markup()
        )
