"""Shared keyboard-navigation helpers for TUI screens."""

from __future__ import annotations

from collections.abc import Callable

from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import OptionList


def first_enabled_option_index(option_list: OptionList) -> int | None:
    """Return the first enabled option index, if any."""
    for index in range(option_list.option_count):
        if not option_list.get_option_at_index(index).disabled:
            return index
    return None


def last_enabled_option_index(option_list: OptionList) -> int | None:
    """Return the last enabled option index, if any."""
    for index in range(option_list.option_count - 1, -1, -1):
        if not option_list.get_option_at_index(index).disabled:
            return index
    return None


def next_enabled_option_index(option_list: OptionList, direction: int) -> int | None:
    """Return the next enabled option index when moving ``direction``."""
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    highlighted = option_list.highlighted
    if highlighted is None:
        return first_enabled_option_index(option_list) if direction == 1 else last_enabled_option_index(option_list)
    if direction == 1:
        indexes = range(highlighted + 1, option_list.option_count)
    else:
        indexes = range(highlighted - 1, -1, -1)
    for index in indexes:
        if not option_list.get_option_at_index(index).disabled:
            return index
    return None


def has_hidden_ancestor(widget: Widget) -> bool:
    """Return True when the widget has a hidden ancestor."""
    current: Widget | None = widget
    while current is not None:
        if current.has_class("hidden"):
            return True
        parent = current.parent
        current = parent if isinstance(parent, Widget) else None
    return False


def is_focusable_for_navigation(
    widget: Widget,
    *,
    check_hidden_ancestor: bool = False,
    check_size: bool = False,
) -> bool:
    """Return True when a widget should be included in keyboard navigation."""
    if not widget.can_focus:
        return False
    if getattr(widget, "disabled", False):
        return False
    if check_hidden_ancestor and has_hidden_ancestor(widget):
        return False
    if check_size:
        return widget.size.height > 0 and widget.size.width > 0
    return True


def move_focus_across_option_lists(
    screen: Screen,
    option_list_ids: tuple[str, ...],
    direction: int,
    *,
    is_focusable: Callable[[OptionList], bool] | None = None,
) -> bool:
    """Move focus/highlight across one or more option lists."""
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    focused = screen.focused
    if not isinstance(focused, OptionList):
        return False
    option_lists = tuple(
        option_list
        for option_list_id in option_list_ids
        for option_list in (screen.query_one(f"#{option_list_id}", OptionList),)
        if is_focusable is None or is_focusable(option_list)
    )
    if not option_lists:
        return False
    try:
        focused_index = option_lists.index(focused)
    except ValueError:
        return False
    next_index = next_enabled_option_index(focused, direction)
    if next_index is not None:
        focused.highlighted = next_index
        return True
    neighbor_index = focused_index + direction
    if neighbor_index < 0 or neighbor_index >= len(option_lists):
        return False
    neighbor = option_lists[neighbor_index]
    neighbor.focus()
    neighbor_highlighted = first_enabled_option_index(neighbor) if direction == 1 else last_enabled_option_index(neighbor)
    if neighbor_highlighted is not None:
        neighbor.highlighted = neighbor_highlighted
    return True


def move_focus_across_widgets(
    screen: Screen,
    widget_ids: tuple[str, ...],
    direction: int,
    *,
    is_focusable: Callable[[Widget], bool] | None = None,
    fallback_to_edge_if_focus_missing: bool = False,
) -> bool:
    """Move focus across ordered widgets, optionally with edge fallback."""
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    focused = screen.focused
    if not isinstance(focused, Widget):
        return False
    widgets = [
        widget
        for widget_id in widget_ids
        for widget in (screen.query_one(f"#{widget_id}", Widget),)
        if is_focusable is None or is_focusable(widget)
    ]
    if not widgets:
        return False
    try:
        focused_index = widgets.index(focused)
    except ValueError:
        if not fallback_to_edge_if_focus_missing:
            return False
        fallback = widgets[0] if direction == 1 else widgets[-1]
        fallback.focus()
        if isinstance(fallback, OptionList) and fallback.highlighted is None:
            fallback.highlighted = first_enabled_option_index(fallback) if direction == 1 else last_enabled_option_index(
                fallback
            )
        return True
    neighbor_index = focused_index + direction
    if neighbor_index < 0 or neighbor_index >= len(widgets):
        return False
    neighbor = widgets[neighbor_index]
    neighbor.focus()
    if isinstance(neighbor, OptionList) and neighbor.highlighted is None:
        neighbor.highlighted = first_enabled_option_index(neighbor) if direction == 1 else last_enabled_option_index(
            neighbor
        )
    return True
