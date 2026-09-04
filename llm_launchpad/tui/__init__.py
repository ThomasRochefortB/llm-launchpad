"""Textual TUI frontend for llm-launchpad.

Consumes protocol events from the Core layer and renders
screens and widgets. No direct subprocess execution.
"""

from __future__ import annotations


def _install_option_list_compatibility() -> None:
    """Restore small widget APIs removed in newer Textual releases."""
    try:
        from textual.widgets import OptionList, Static
    except Exception:
        return

    if not hasattr(OptionList, "set_options"):

        def set_options(self: object, options: list[object]) -> None:
            clear_options = self.clear_options
            add_options = self.add_options
            clear_options()
            add_options(options)

        OptionList.set_options = set_options

    if not hasattr(OptionList, "highlighted_option"):

        @property
        def highlighted_option(self: object) -> object | None:
            highlighted = getattr(self, "highlighted", None)
            if highlighted is None:
                return None
            try:
                return self.get_option_at_index(highlighted)
            except Exception:
                return None

        OptionList.highlighted_option = highlighted_option

    if not hasattr(Static, "content"):

        @property
        def content(self: object) -> object:
            return getattr(self, "_content", "")

        Static.content = content


_install_option_list_compatibility()
