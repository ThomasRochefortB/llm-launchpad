"""Log viewer widget: RichLog wrapper with auto-scroll."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import RichLog, Static
from textual.containers import Vertical


class LogViewer(Vertical):
    """Scrollable log output panel with auto-scroll.

    Wraps a ``RichLog`` and exposes a simple ``write_line`` API.
    """

    DEFAULT_CSS = """
    LogViewer {
        height: 1fr;
        border: solid $primary-background;
        padding: 0 1;
    }
    LogViewer RichLog {
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        yield RichLog(highlight=True, markup=True, wrap=True, id="log-output")

    @property
    def log_widget(self) -> RichLog:
        return self.query_one("#log-output", RichLog)

    def write_line(self, line: str, style: str = "") -> None:
        """Append a line and auto-scroll to bottom."""
        if style:
            self.log_widget.write(f"[{style}]{line}[/{style}]")
        else:
            self.log_widget.write(line)

    def clear(self) -> None:
        self.log_widget.clear()
