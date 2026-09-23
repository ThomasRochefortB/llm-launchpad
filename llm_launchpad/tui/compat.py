"""Typed stand-ins for widget APIs that newer Textual releases removed.

Textual 5 dropped ``OptionList.set_options``, ``OptionList.highlighted_option``
and ``Static.content``. ``install()`` puts them back at runtime for code and
tests written against them, but a type checker cannot see a monkeypatch, so
application code calls the functions below instead.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option


def replace_options(option_list: OptionList, options: Iterable[Any]) -> None:
    """Replace every option in ``option_list``."""
    option_list.clear_options()
    option_list.add_options(list(options))


def highlighted_option(option_list: OptionList) -> Option | None:
    """Return the highlighted option, or ``None`` when nothing is highlighted."""
    index = option_list.highlighted
    if index is None:
        return None
    try:
        return option_list.get_option_at_index(index)
    except Exception:
        return None


def static_content(static: Static) -> Any:
    """Return what a ``Static`` was last given to display."""
    return getattr(static, "_content", "")


def install() -> None:
    """Restore the removed APIs as attributes, for callers that use them.

    ``setattr`` rather than assignment: these attributes do not exist on the
    classes as Textual declares them, which is the point.
    """
    if not hasattr(OptionList, "set_options"):
        setattr(OptionList, "set_options", replace_options)  # noqa: B010
    if not hasattr(OptionList, "highlighted_option"):
        setattr(OptionList, "highlighted_option", property(highlighted_option))  # noqa: B010
    if not hasattr(Static, "content"):
        setattr(Static, "content", property(static_content))  # noqa: B010
