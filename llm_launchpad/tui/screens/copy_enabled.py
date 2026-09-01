"""Shared screen behavior for native-first copy-to-clipboard support."""

from __future__ import annotations

from typing import Any

from rich.errors import MarkupError
from rich.markup import render as render_markup
from textual import events
from textual.binding import Binding
from textual.css.query import NoMatches
from textual.geometry import Size
from textual.selection import Selection
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import DataTable, Input, OptionList, Select, Static

from ..responsive import (
    MIN_TERMINAL_HEIGHT,
    MIN_TERMINAL_WIDTH,
    RESPONSIVE_CLASS_NAMES,
    ViewportProfile,
)
from ..visual import DEFAULT_TUI_DENSITY, DEFAULT_TUI_THEME, TUI_THEME_OPTIONS


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

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._selection_sync_scheduled = False
        self._last_synced_selection: str | None = None
        self._viewport_profile: ViewportProfile | None = None
        self._focus_before_size_gate: Widget | None = None

    async def on_resize(self, event: events.Resize) -> None:
        """Apply responsive classes and gate terminals below the supported floor."""
        await self._apply_viewport_size(event.size)

    async def _apply_viewport_size(self, size: Size) -> None:
        profile = ViewportProfile.from_size(size)
        previous = self._viewport_profile

        if profile != previous:
            active_classes = set(profile.class_names)
            for class_name in RESPONSIVE_CLASS_NAMES:
                self.set_class(class_name in active_classes, class_name)
            self._viewport_profile = profile
            self._sync_visual_classes()
            self.refresh(repaint=True, layout=True)
            self.viewport_profile_changed(profile, previous)

        try:
            overlay = self.query_one("#minimum-size-overlay", Static)
        except NoMatches:
            overlay = Static("", id="minimum-size-overlay")
            await self.mount(overlay)

        overlay.update(
            "[bold #7bf168]Terminal too small[/]\n"
            f"Current: {size.width}×{size.height}  ·  "
            f"Minimum: {MIN_TERMINAL_WIDTH}×{MIN_TERMINAL_HEIGHT}\n"
            "[dim]Resize the terminal to continue.[/dim]"
        )
        overlay.display = profile.too_small

        if profile.too_small:
            if self.focused is not None:
                self._focus_before_size_gate = self.focused
            self.set_focus(None)
        elif self._focus_before_size_gate is not None:
            previous_focus = self._focus_before_size_gate
            self._focus_before_size_gate = None
            if previous_focus.is_mounted and previous_focus.can_focus:
                previous_focus.focus()

    @property
    def viewport_profile(self) -> ViewportProfile:
        """Return the latest profile, classifying current size if necessary."""
        if self._viewport_profile is not None:
            return self._viewport_profile
        return ViewportProfile.from_size(self.size)

    def viewport_profile_changed(
        self,
        profile: ViewportProfile,
        previous: ViewportProfile | None,
    ) -> None:
        """Allow screens to adapt content after responsive classes change."""
        _ = (profile, previous)

    def refresh_visual_preferences(self) -> None:
        """Reapply app-level theme and density classes immediately."""
        self._sync_visual_classes()
        self.refresh(repaint=True, layout=True)

    def _sync_visual_classes(self) -> None:
        app_theme = str(getattr(self.app, "theme", DEFAULT_TUI_THEME))
        for _, theme_name in TUI_THEME_OPTIONS:
            self.set_class(app_theme == theme_name, f"theme-{theme_name}")
        density = str(getattr(self.app, "tui_density", DEFAULT_TUI_DENSITY))
        self.set_class(density == "compact", "density-compact")

    def on_key(self, event: events.Key) -> None:
        """Block clipped controls while the minimum-size overlay is active."""
        if not self.has_class("viewport-too-small"):
            return
        if event.key in {"escape", "q", "ctrl+c"}:
            return
        event.prevent_default()
        event.stop()

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

    async def _watch_selections(
        self,
        old_selections: dict[Static, Selection],
        selections: dict[Static, Selection],
    ) -> None:
        await super()._watch_selections(old_selections, selections)
        if not selections:
            self._last_synced_selection = None
            return
        if self._selection_sync_scheduled:
            return
        self._selection_sync_scheduled = True
        self.call_after_refresh(self._sync_selection_to_clipboard)

    def _copyable_text(self) -> str | None:
        return self._normalize_text(
            self._selected_text_for_copy() or self._focused_text_fallback()
        )

    def _sync_selection_to_clipboard(self) -> None:
        self._selection_sync_scheduled = False
        selection = self._normalize_text(self._selected_text_for_copy())
        if not selection:
            self._last_synced_selection = None
            return
        if selection == self._last_synced_selection:
            return
        self.app.copy_to_clipboard(selection)
        self._last_synced_selection = selection

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
