"""Log viewer widget: selectable plain-text Log wrapper."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

from rich.cells import cell_len
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.geometry import Offset, Size
from textual.message import Message
from textual.selection import Selection
from textual.strip import Strip
from textual.widgets import Log

_SUMMARY_DONE = "#7bf168"
_SUMMARY_INFO = "#93a596"
_SUMMARY_ERROR = "#ff6b6b"
_SUMMARY_PROGRESS = "#e6c07b"
MAX_RETAINED_LOG_LINES = 10_000
_RETENTION_PRUNE_CHUNK = 1_000
_RetainedItem = TypeVar("_RetainedItem")
_SPINNER_PREFIXES = (
    "⠋ ",
    "⠙ ",
    "⠹ ",
    "⠸ ",
    "⠼ ",
    "⠴ ",
    "⠦ ",
    "⠧ ",
    "⠇ ",
    "⠏ ",
)


def prune_retained_items(
    items: list[_RetainedItem],
    limit: int = MAX_RETAINED_LOG_LINES,
) -> int:
    """Bound history while amortizing the cost of front-pruning a list."""
    if len(items) <= limit:
        return 0
    if limit <= 0:
        pruned_count = len(items)
        items.clear()
        return pruned_count
    pruned_count = max(len(items) - limit, min(_RETENTION_PRUNE_CHUNK, limit))
    del items[:pruned_count]
    return pruned_count


def _style_summary_prefix(line_text: Text, line: str) -> None:
    """Color compact summary markers without storing markup in the log text."""
    if line.startswith("✓ "):
        line_text.stylize(_SUMMARY_DONE, 0, 1)
    elif line.startswith("✗ "):
        line_text.stylize(_SUMMARY_ERROR, 0, len(line))
    elif line.startswith(_SPINNER_PREFIXES):
        line_text.stylize(_SUMMARY_PROGRESS, 0, 1)
    elif line.startswith("· "):
        line_text.stylize(_SUMMARY_INFO, 0, 1)


class SelectableLog(Log):
    """Width-aware log with line-level double-click selection.

    * **Double-click** selects the entire clicked line.
    * **Triple-click** selects all log content.
    * Single click / drag selection is unchanged.

    Textual's plain :class:`Log` uses its longest line as the virtual width and
    doesn't offer a wrapping mode. This subclass retains those logical lines
    while rendering a second, word-wrapped view that is rebuilt whenever the
    terminal width changes. Keeping the logical lines intact preserves search,
    line counts, and copy behavior.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        max_lines = kwargs.pop("max_lines", None)
        self._retention_limit = int(max_lines) if max_lines is not None else None
        super().__init__(*args, **kwargs)
        self._wrapped_lines: list[str] = []
        self._visual_starts: list[int] = []
        self._wrap_width = 1

    class ViewportChanged(Message):
        """Posted when the visible log range changes."""

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        """Refresh the viewport and expose follow-state changes to the wrapper."""
        super().watch_scroll_y(old_value, new_value)
        if round(old_value) != round(new_value):
            self.post_message(self.ViewportChanged())

    @property
    def wrap_width(self) -> int:
        """Current cell width used to reflow each logical log line."""
        return self._wrap_width

    @property
    def wrapped_lines(self) -> tuple[str, ...]:
        """Rendered visual rows, exposed for selection and regression tests."""
        return tuple(self._wrapped_lines)

    def visual_row_for_line(self, line_index: int) -> int:
        """Return the first rendered row for a logical source line."""
        if 0 <= line_index < len(self._visual_starts):
            return self._visual_starts[line_index]
        return line_index

    def _available_wrap_width(self) -> int:
        # Reserve the configured vertical scrollbar column. This avoids a
        # horizontal scrollbar appearing when wrapping creates enough rows to
        # make the vertical scrollbar visible.
        scrollbar_width = int(self.styles.scrollbar_size_vertical)
        return max(1, self.size.width - scrollbar_width)

    def _wrap_source_line(self, source_line: str) -> list[str]:
        """Wrap one source line by terminal cells, ignoring ANSI control codes."""
        processed = self._process_line(source_line)
        if not processed:
            return [""]
        rows = Text.from_ansi(processed).wrap(
            self.app.console,
            self._wrap_width,
            overflow="fold",
            no_wrap=False,
        )
        return [row.plain for row in rows] or [""]

    def _sync_virtual_size(self) -> None:
        """Match the canvas to wrapped rows without rescanning log history."""
        self._width = self._wrap_width if self._wrapped_lines else 0
        self.virtual_size = Size(self._width, len(self._wrapped_lines))

    def _reflow(self, *, follow: bool | None = None) -> None:
        """Rebuild visual rows for the widget's current content width."""
        if follow is None:
            follow = self.is_vertical_scroll_end
        self._wrap_width = self._available_wrap_width()
        wrapped_lines: list[str] = []
        visual_starts: list[int] = []
        for source_line in self._lines:
            visual_starts.append(len(wrapped_lines))
            wrapped_lines.extend(self._wrap_source_line(source_line))

        self._wrapped_lines = wrapped_lines
        self._visual_starts = visual_starts
        self._render_line_cache.clear()
        self._sync_virtual_size()
        self.scroll_to(x=0, animate=False, immediate=True)
        if follow:
            self.scroll_end(animate=False, immediate=True, x_axis=False)
        else:
            self.refresh()

    def write_lines(
        self,
        lines: Iterable[str],
        scroll_end: bool | None = None,
    ) -> SelectableLog:
        """Append logical lines and reflow them to the current viewport."""
        source_values = list(lines)
        was_following = self.is_vertical_scroll_end
        previous_line_count = len(self._lines)
        previous_wrap_width = self._wrap_width
        super().write_lines(source_values, scroll_end=False)
        pruned_count = (
            prune_retained_items(self._lines, self._retention_limit)
            if self._retention_limit is not None
            else 0
        )
        should_follow = self.auto_scroll if scroll_end is None else scroll_end
        if (
            previous_wrap_width != self._available_wrap_width()
            or len(self._visual_starts) != previous_line_count
        ):
            self._reflow(follow=should_follow and was_following)
            return self

        pruned_previous_count = min(pruned_count, previous_line_count)
        if pruned_previous_count:
            if pruned_previous_count >= len(self._visual_starts):
                pruned_visual_count = len(self._wrapped_lines)
            else:
                pruned_visual_count = self._visual_starts[pruned_previous_count]
            del self._wrapped_lines[:pruned_visual_count]
            del self._visual_starts[:pruned_previous_count]
            self._visual_starts = [
                start - pruned_visual_count for start in self._visual_starts
            ]
            # Cached visual row indexes shift when retained source lines are pruned.
            self._render_line_cache.clear()

        retained_previous_count = max(0, previous_line_count - pruned_count)
        first_new_visual_row = len(self._wrapped_lines)
        for source_line in self._lines[retained_previous_count:]:
            self._visual_starts.append(len(self._wrapped_lines))
            self._wrapped_lines.extend(self._wrap_source_line(source_line))
        self._sync_virtual_size()
        new_visual_count = len(self._wrapped_lines) - first_new_visual_row
        if pruned_previous_count:
            self.refresh()
        elif new_visual_count:
            self.refresh_lines(first_new_visual_row, new_visual_count)
        self.scroll_to(x=0, animate=False, immediate=True)
        if should_follow and was_following:
            self.scroll_end(animate=False, immediate=True, x_axis=False)
        else:
            self.refresh()
        return self

    def _update_size(self, updates: int, lines: list[str]) -> None:
        """Skip the base log's asynchronous longest-line width calculation."""
        _ = (updates, lines)

    def clear(self) -> SelectableLog:
        """Clear logical and rendered rows."""
        super().clear()
        self._wrapped_lines = []
        self._visual_starts = []
        self._reflow(follow=True)
        return self

    def on_resize(self, event: events.Resize) -> None:
        """Reflow existing output whenever the terminal changes width."""
        _ = event
        following = bool(getattr(self.parent, "_following", self.is_vertical_scroll_end))
        self._reflow(follow=following)

    def render_line(self, y: int) -> Strip:
        """Render a row from the reflowed visual-line buffer."""
        scroll_x, scroll_y = self.scroll_offset
        visual_y = scroll_y + y
        width = self.size.width
        rich_style = self.rich_style
        if visual_y >= len(self._wrapped_lines):
            return Strip.blank(width, rich_style)

        line = self._wrapped_lines[visual_y]
        line_text = Text(line, no_wrap=True)
        line_text.stylize(rich_style)
        if self.highlight:
            line_text = self.highlighter(line_text)
        _style_summary_prefix(line_text, line)
        if self.text_selection is not None:
            span = self.text_selection.get_span(visual_y - self._clear_y)
            if span is not None:
                start, end = span
                if end == -1:
                    end = len(line_text)
                line_text.stylize(
                    self.screen.get_component_rich_style("screen--selection"),
                    start,
                    end,
                )

        strip = Strip(line_text.render(self.app.console), cell_len(line))
        strip = strip.crop_extend(scroll_x, scroll_x + width, rich_style)
        return strip.apply_offsets(scroll_x, visual_y)

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        """Extract selected text from the displayed, wrapped representation."""
        return selection.extract("\n".join(self._wrapped_lines)), "\n"

    def _select_line(self, line_index: int) -> bool:
        """Select a single line."""
        if not (0 <= line_index < len(self._wrapped_lines)):
            return False
        line_text = self._wrapped_lines[line_index]
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

    class SearchChanged(Message):
        """Current search query, selected match, and total match count."""

        def __init__(self, *, query: str, current: int, total: int) -> None:
            super().__init__()
            self.query = query
            self.current = current
            self.total = total

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._plain_lines: list[str] = []
        self._search_query = ""
        self._search_matches: list[int] = []
        self._search_index = -1
        self._status_update_scheduled = False

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
        yield SelectableLog(
            highlight=True,
            auto_scroll=True,
            max_lines=MAX_RETAINED_LOG_LINES,
            id="log-output",
        )

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
        log.write_line(line)
        new_plain_lines = line.splitlines()
        self._plain_lines.extend(new_plain_lines)
        prune_retained_items(self._plain_lines)
        added_lines = len(new_plain_lines)
        if was_following:
            self._following = True
            self._unseen_lines = 0
        else:
            self._following = False
            self._unseen_lines += added_lines
        if self._search_query:
            self._refresh_search(keep_current=True)
        self._post_status()

    def replace_lines(self, lines: list[str]) -> None:
        """Replace displayed content while retaining the active search query."""
        self._following = True
        self._unseen_lines = 0
        self.set_lines(lines, keep_follow=True)

    def set_lines(self, lines: list[str], *, keep_follow: bool = True) -> None:
        """Replace log contents, optionally preserving the current follow mode."""
        was_following = self._following
        self._plain_lines = list(lines[-MAX_RETAINED_LOG_LINES:])
        self.log_widget.clear()
        if self._plain_lines:
            self.log_widget.write_lines(self._plain_lines, scroll_end=False)
        if keep_follow and was_following:
            self._following = True
            self._unseen_lines = 0
            self.log_widget.scroll_end(animate=False, immediate=True, x_axis=False)
        elif not was_following:
            self._following = False
        if self._search_query:
            self._refresh_search(keep_current=True)
        self._post_status()

    def search(self, query: str) -> None:
        """Find case-insensitive matches and jump to the first result."""
        self._search_query = query.strip()
        self._refresh_search(keep_current=False)

    def next_match(self, direction: int = 1) -> None:
        """Move to the next or previous log-search result."""
        if not self._search_matches:
            self._post_search_status()
            return
        self._search_index = (self._search_index + direction) % len(self._search_matches)
        self._scroll_to_search_match()
        self._post_search_status()

    def _refresh_search(self, *, keep_current: bool) -> None:
        current_line = None
        if keep_current and 0 <= self._search_index < len(self._search_matches):
            current_line = self._search_matches[self._search_index]
        if self._search_query:
            needle = self._search_query.casefold()
            self._search_matches = [
                index
                for index, line in enumerate(self._plain_lines)
                if needle in line.casefold()
            ]
        else:
            self._search_matches = []
        if not self._search_matches:
            self._search_index = -1
        elif current_line in self._search_matches:
            self._search_index = self._search_matches.index(current_line)
        else:
            self._search_index = 0
            self._scroll_to_search_match()
        self._post_search_status()

    def _scroll_to_search_match(self) -> None:
        if not (0 <= self._search_index < len(self._search_matches)):
            return
        line = self.log_widget.visual_row_for_line(
            self._search_matches[self._search_index]
        )
        viewport_rows = max(1, self.log_widget.scrollable_content_region.height)
        self.log_widget.scroll_to(
            y=max(0, line - viewport_rows // 2),
            animate=False,
            immediate=True,
        )

    def _post_search_status(self) -> None:
        total = len(self._search_matches)
        current = self._search_index + 1 if total else 0
        self.post_message(
            self.SearchChanged(
                query=self._search_query,
                current=current,
                total=total,
            )
        )

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
        if self._status_update_scheduled:
            return
        self._status_update_scheduled = True
        self.call_later(self._emit_status)

    def _emit_status(self) -> None:
        self._status_update_scheduled = False
        self.post_message(
            self.StatusChanged(
                following=self._following,
                unseen_lines=self._unseen_lines,
                line_count=len(self._plain_lines),
            )
        )

    def clear(self) -> None:
        self.log_widget.clear()
        self._plain_lines = []
        self._search_matches = []
        self._search_index = -1
        self._following = True
        self._unseen_lines = 0
        self._post_search_status()
        self._post_status()
