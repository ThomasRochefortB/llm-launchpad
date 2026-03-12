"""Shared screen behavior for native-first copy-to-clipboard support."""

from __future__ import annotations

from typing import Any

from rich.errors import MarkupError
from rich.markup import render as render_markup
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import DataTable, Input, OptionList, Select, Static


class CopyEnabledScreen(Screen):
    """Screen base class that provides consistent copy behavior."""

    BINDINGS = [
        Binding("y", "copy_text", "Copy", key_display="y", show=False),
        Binding(
            "ctrl+shift+c,super+c,meta+c,cmd+c,command+c",
            "copy_text_to_clipboard",
            "Copy",
            show=False,
        ),
    ]

    def action_copy_text(self) -> None:
        """Copy selected or focused text to the clipboard."""
        normalized = self._copyable_text()
        if normalized:
            self.app.copy_to_clipboard(normalized)
            return
        self.notify("Nothing to copy", timeout=2)

    def action_copy_text_to_clipboard(self) -> None:
        """Copy selected text directly to the clipboard when the terminal supports it."""
        normalized = self._copyable_text()
        if normalized:
            self.app.copy_to_clipboard(normalized)
            return
        self.notify("Nothing to copy", timeout=2)

    def _copyable_text(self) -> str | None:
        return self._normalize_text(
            self._selected_text_for_copy() or self._focused_text_fallback()
        )

    def _selected_text_for_copy(self) -> str | None:
        return self.get_selected_text()

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
