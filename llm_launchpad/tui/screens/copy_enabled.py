"""Shared screen behavior for consistent copy-to-clipboard support."""

from __future__ import annotations

from typing import Any

from rich.errors import MarkupError
from rich.markup import render as render_markup
from textual import events
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import DataTable, Input, OptionList, Select, Static


class CopyEnabledScreen(Screen):
    """Screen base class that provides consistent copy behavior."""

    _COPY_KEY_ALIASES = {
        "ctrl+c",
        "ctrl+shift+c",
        "meta+c",
        "super+c",
        "cmd+c",
        "command+c",
    }

    BINDINGS = [
        Binding("meta+c,ctrl+shift+c", "copy_text", "Copy", show=False),
    ]

    def action_copy_text(self) -> None:
        """Copy selected text, or focused widget text fallback."""
        text = self.get_selected_text() or self._focused_text_fallback()
        normalized = self._normalize_text(text)
        if normalized:
            self.app.copy_to_clipboard(normalized)
            return
        self.notify("Nothing to copy", timeout=2)

    def on_key(self, event: events.Key) -> None:
        """Handle additional copy key aliases seen across terminal implementations."""
        key_forms = {event.key, event.name, *event.aliases}
        if key_forms.intersection(self._COPY_KEY_ALIASES):
            self.action_copy_text()
            event.stop()
            event.prevent_default()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        """Copy selected text on drag-end as a fallback when Cmd+C is intercepted."""
        if event.button in (1, "left"):
            self.call_after_refresh(self._copy_selected_text_if_any)

    def _focused_text_fallback(self) -> str | None:
        focused = self.focused
        if isinstance(focused, OptionList):
            return self._copy_from_option_list(focused)
        if isinstance(focused, DataTable):
            return self._copy_from_data_table(focused)
        if isinstance(focused, Input):
            return focused.value
        if isinstance(focused, Select):
            selected = focused.selection
            return str(selected) if selected is not None else None
        if isinstance(focused, Static):
            return self._to_plain_text(focused.render())
        return None

    def _copy_from_option_list(self, widget: OptionList) -> str | None:
        highlighted = widget.highlighted
        if highlighted is None:
            return None
        try:
            option = widget.get_option_at_index(highlighted)
        except Exception:
            return None
        return self._to_plain_text(option.prompt)

    def _copy_from_data_table(self, widget: DataTable) -> str | None:
        if widget.row_count <= 0:
            return None
        try:
            row_values = widget.get_row_at(widget.cursor_row)
        except Exception:
            return None
        return "\t".join(self._to_plain_text(value) for value in row_values)

    def _normalize_text(self, text: str | None) -> str | None:
        if text is None:
            return None
        normalized = text.rstrip("\n")
        return normalized if normalized else None

    def _copy_selected_text_if_any(self) -> bool:
        text = self._normalize_text(self.get_selected_text())
        if not text:
            return False
        self.app.copy_to_clipboard(text)
        return True

    def _to_plain_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return self._strip_markup(value)
        return str(value)

    def _strip_markup(self, text: str) -> str:
        try:
            return render_markup(text).plain
        except MarkupError:
            return text
