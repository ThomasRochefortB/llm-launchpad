"""Monitor screen: real-time log streaming and operation status.

Shows the status header + scrolling log output for any running
operation (deploy, warmup, logs, status, stop).
"""

from __future__ import annotations

import re

from textual import events
from textual.actions import SkipAction
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
            "c,ctrl+c,ctrl+shift+c,meta+c,super+c",
            "copy_text",
            "Copy",
            key_display="cmd+c",
            show=True,
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
        yield StatusHeader(id="monitor-status-header")
        with Vertical(id="monitor-layout"):
            yield Static(
                (
                    f"[bold cyan]{self._title}[/bold cyan]  "
                    "[dim]drag to select, dbl-click for line  "
                    "cmd+c/ctrl+c/c to copy  ctrl+l clear  esc to return[/dim]"
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

    def on_key(self, event: events.Key) -> None:
        """Handle additional copy aliases across terminal implementations."""
        key_forms = {event.key, event.name, *event.aliases}
        if key_forms.intersection(
            {
                "c",
                "ctrl+c",
                "ctrl+shift+c",
                "meta+c",
                "super+c",
                "cmd+c",
                "command+c",
            }
        ):
            self.action_copy_text()
            event.stop()
            event.prevent_default()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        """Fallback: copy selected text when drag selection ends in log view.

        Some terminals reserve Cmd+C and never forward it to Textual. In that
        case, copying on mouse release keeps log copying usable with Cmd-based
        workflows.
        """
        if event.button in (1, "left"):
            self.call_after_refresh(
                lambda: self._copy_selected_text(notify=False, raise_on_empty=False)
            )

    # -- Actions --

    def _copy_selected_text(
        self, *, notify: bool = True, raise_on_empty: bool = True
    ) -> bool:
        text = None
        log_widget = self.log_viewer.log_widget
        getter = getattr(log_widget, "get_selected_text", None)
        if callable(getter):
            text = getter()
        if not text:
            text = self.get_selected_text()
        if not text:
            if notify:
                self.notify("No text selected to copy", timeout=2)
            if raise_on_empty:
                raise SkipAction()
            return False

        # OSC 52 (works in iTerm2, Kitty, WezTerm, Ghostty, etc.)
        self.app.copy_to_clipboard(text)

        if notify:
            lines = text.count("\n") + 1
            self.notify(f"Copied {lines} line{'s' if lines != 1 else ''}", timeout=2)
        return True

    def action_copy_text(self) -> None:
        """Copy selected log text to clipboard."""
        self._copy_selected_text(notify=True, raise_on_empty=True)

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_clear_log(self) -> None:
        self.log_viewer.clear()
