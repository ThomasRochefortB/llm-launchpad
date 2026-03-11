"""Log viewer widget: selectable plain-text Log wrapper."""

from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.geometry import Offset
from textual.selection import Selection
from textual.widgets import Log


class SelectableLog(Log):
    """Log with line-level double-click selection.

    * **Double-click** selects the entire clicked line.
    * **Triple-click** selects all log content.
    * Single click / drag selection is unchanged.
    """

    async def _on_click(self, event: events.Click) -> None:
        if event.widget is self:
            if (
                self.allow_select
                and self.screen.allow_select
                and self.app.ALLOW_SELECT
            ):
                if event.chain == 2:
                    # Select only the clicked line (not all text).
                    # prevent_default() stops Textual from also calling
                    # the parent Log._on_click which would text_select_all().
                    line_index = int(self.scroll_y) + event.y
                    if 0 <= line_index < self.line_count:
                        line_text = self.lines[line_index]
                        start = Offset(0, line_index)
                        end = Offset(len(line_text), line_index)
                        self.screen.selections = {
                            self: Selection(start, end)
                        }
                    event.prevent_default()
                    return
                elif event.chain == 3:
                    self.text_select_all()
                    event.prevent_default()
                    return
        await self.broker_event("click", event)


class LogViewer(Vertical):
    """Scrollable log output panel with auto-scroll.

    Wraps a selectable ``SelectableLog`` and exposes a simple ``write_line``
    API.
    """

    DEFAULT_CSS = """
    LogViewer {
        height: 1fr;
        border: solid #17321e;
        background: #0a0f0b;
        padding: 0 1;
    }
    LogViewer Log {
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        yield SelectableLog(highlight=True, auto_scroll=True, id="log-output")

    @property
    def log_widget(self) -> SelectableLog:
        return self.query_one("#log-output", SelectableLog)

    def write_line(self, line: str, style: str = "") -> None:
        """Append a line and auto-scroll to bottom.

        ``style`` is accepted for compatibility with existing call sites.
        """
        _ = style
        self.log_widget.write_line(line)

    def clear(self) -> None:
        self.log_widget.clear()
