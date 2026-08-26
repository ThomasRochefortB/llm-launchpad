"""Log viewer widget: selectable plain-text Log wrapper."""

from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.geometry import Offset
from textual.message import Message
from textual.selection import Selection
from textual.widgets import Log


class SelectableLog(Log):
    """Log with line-level double-click selection.

    * **Double-click** selects the entire clicked line.
    * **Triple-click** selects all log content.
    * Single click / drag selection is unchanged.
    """

    class ViewportChanged(Message):
        """Posted when the visible log range changes."""

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        """Refresh the viewport and expose follow-state changes to the wrapper."""
        super().watch_scroll_y(old_value, new_value)
        if round(old_value) != round(new_value):
            self.post_message(self.ViewportChanged())

    def _select_line(self, line_index: int) -> bool:
        """Select a single line."""
        if not (0 <= line_index < self.line_count):
            return False
        line_text = self.lines[line_index]
        start = Offset(0, line_index)
        end = Offset(len(line_text), line_index)
        self.screen.selections = {  # ty: ignore[invalid-assignment]
            self: Selection(start, end)
        }
        return True

    def _select_all(self) -> None:
        """Select all log content."""
        self.text_select_all()

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
                    if self._select_line(line_index):
                        event.prevent_default()
                        return
                elif event.chain == 3:
                    self._select_all()
                    event.prevent_default()
                    return
        await self.broker_event("click", event)


class LogViewer(Vertical):
    """Scrollable log output panel with auto-scroll.

    Wraps a selectable ``SelectableLog`` and exposes a simple ``write_line``
    API.
    """

    _following = True
    _unseen_lines = 0

    class StatusChanged(Message):
        """Current follow mode, unread count, and total line count."""

        def __init__(
            self,
            *,
            following: bool,
            unseen_lines: int,
            line_count: int,
        ) -> None:
            super().__init__()
            self.following = following
            self.unseen_lines = unseen_lines
            self.line_count = line_count

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
        """Append a line and retain the user's current follow mode.

        ``style`` is accepted for compatibility with existing call sites.
        """
        _ = style
        log = self.log_widget
        was_following = log.is_vertical_scroll_end
        previous_line_count = log.line_count
        log.write_line(line)
        added_lines = max(0, log.line_count - previous_line_count)
        if was_following:
            self._following = True
            self._unseen_lines = 0
        else:
            self._following = False
            self._unseen_lines += added_lines
        self._post_status()

    def on_selectable_log_viewport_changed(
        self,
        event: SelectableLog.ViewportChanged,
    ) -> None:
        """Update follow state when the user scrolls through the log."""
        event.stop()
        self._following = self.log_widget.is_vertical_scroll_end
        if self._following:
            self._unseen_lines = 0
        self._post_status()

    @property
    def following(self) -> bool:
        """Whether new output is currently kept in view."""
        return self._following

    @property
    def unseen_lines(self) -> int:
        """Number of lines appended since the user paused following."""
        return self._unseen_lines

    def resume_following(self) -> None:
        """Jump to the latest output and resume following."""
        self._following = True
        self._unseen_lines = 0
        self.log_widget.scroll_end(
            animate=False,
            immediate=True,
            x_axis=False,
        )
        self._post_status()

    def page_up(self) -> None:
        """Scroll the log up by one viewport."""
        self.log_widget.scroll_page_up(animate=False)

    def page_down(self) -> None:
        """Scroll the log down by one viewport."""
        self.log_widget.scroll_page_down(animate=False)

    def _post_status(self) -> None:
        self.post_message(
            self.StatusChanged(
                following=self._following,
                unseen_lines=self._unseen_lines,
                line_count=self.log_widget.line_count,
            )
        )

    def clear(self) -> None:
        self.log_widget.clear()
        self._following = True
        self._unseen_lines = 0
