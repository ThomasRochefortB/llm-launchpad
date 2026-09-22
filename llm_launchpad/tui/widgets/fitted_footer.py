"""Footer that fits its key hints to the terminal instead of truncating them."""

from __future__ import annotations

from collections import defaultdict

from rich.cells import cell_len
from textual.app import ComposeResult
from textual.binding import Binding
from textual.widgets import Footer
from textual.widgets._footer import FooterKey


# Textual renders each hint as `<pad>key<pad><pad>description<pad>`, which costs
# three cells beyond the two strings themselves.
_HINT_PADDING_CELLS = 3
# "+3 more" style overflow marker.
_OVERFLOW_TEMPLATE = "+{count} more (?)"


def _hint_width(key_display: str, description: str) -> int:
    return cell_len(key_display) + cell_len(description) + _HINT_PADDING_CELLS


class FittedFooter(Footer):
    """Drop the hints that do not fit rather than clipping the last one.

    Textual lays the footer out as a single grid row sized to its content. Once
    the hints outgrow the terminal the final column is simply cut, which is how
    ``i Details`` became ``i D`` on the main menu at 80 columns and
    ``^l Clear log`` became ``^l Cl`` on the monitor at 100. A half-rendered key
    is worse than an absent one: it names a shortcut the reader cannot act on
    and hides that anything was omitted.

    Hints are kept in presentation-priority order and whatever is left over is
    summarised by a trailing marker pointing at the help overlay -- which
    lists all of them. Help itself is never dropped: the marker that replaces
    the hints must never replace the way to see them.
    """

    # Lower values are rendered first. ``_DEFAULT_FOOTER_PRIORITY`` is the
    # fallback for bindings that carry no explicit priority: Back/palette
    # management are navigation, everything else is a screen verb. Global
    # overlays (operations) sort last; help itself stays
    # first in the overflow marker's guarantee (see _fit), never dropped.
    FOOTER_PRIORITY: dict[str, int] = {
        "show_help": 0,
        "pop_screen": 10,
        "go_back": 10,
        "push_operations": 96,
        "choose_selected": 20,
        "deploy": 20,
        "do_deploy": 20,
        "copy_base_url": 20,
        "copy_text": 20,
        "copy_all": 20,
        "save": 20,
        "refresh_storage": 30,
        "refresh_endpoints": 30,
        "refresh_availability": 30,
        "refresh_billing": 30,
        "open_actions": 30,
        "status_selected": 30,
        "logs_selected": 30,
        "focus_model_search": 31,
        "focus_model_filter": 31,
        "focus_gpu_filter": 32,
        "toggle_model_view": 33,
        "focus_next_control": 40,
        "focus_previous_control": 40,
        "navigate_option_list_up": 40,
        "navigate_option_list_down": 40,
    }
    _DEFAULT_FOOTER_PRIORITY = 50

    DEFAULT_CSS = """
    FittedFooter {
        FooterKey.-overflow {
            text-style: italic;
            .footer-key--description {
                color: $footer-description-foreground 70%;
            }
        }
    }
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._fitted_width = 0

    async def on_resize(self) -> None:
        """Re-fit the hints when the terminal width changes.

        Textual only recomposes the footer when the bindings change, so without
        this a resize keeps whichever selection suited the previous width.
        """
        width = self.size.width
        if width == self._fitted_width:
            return
        self._fitted_width = width
        if self._bindings_ready and self.is_mounted:
            await self.recompose()

    def compose(self) -> ComposeResult:
        if not self._bindings_ready:
            return

        action_to_bindings: defaultdict[str, list[tuple[Binding, bool, str]]]
        action_to_bindings = defaultdict(list)
        for _, binding, enabled, tooltip in self.screen.active_bindings.values():
            if binding.show:
                action_to_bindings[binding.action].append((binding, enabled, tooltip))

        palette_binding = self._command_palette_binding()
        budget = self.size.width or self.app.size.width
        if palette_binding is not None:
            # The palette hint docks right and is laid out outside the grid, so
            # its width is unavailable to everything else.
            binding, _enabled, _tooltip = palette_binding
            budget -= _hint_width(self.app.get_key_display(binding), binding.description) + 1

        shown, hidden = self._fit(
            self._prioritize(list(action_to_bindings.values())), budget
        )

        self.styles.grid_size_columns = len(shown) + (1 if hidden else 0)
        for binding, enabled, tooltip in shown:
            yield FooterKey(
                binding.key,
                self.app.get_key_display(binding),
                binding.description,
                binding.action,
                disabled=not enabled,
                tooltip=tooltip,
            ).data_bind(Footer.compact)
        if hidden:
            yield FooterKey(
                "question_mark",
                "",
                _OVERFLOW_TEMPLATE.format(count=hidden),
                "show_help",
                classes="-overflow",
                tooltip="Press ? to see every shortcut for this screen",
            ).data_bind(Footer.compact)
        if palette_binding is not None:
            binding, enabled, tooltip = palette_binding
            yield FooterKey(
                binding.key,
                self.app.get_key_display(binding),
                binding.description,
                binding.action,
                classes="-command-palette",
                disabled=not enabled,
                tooltip=tooltip,
            )

    def _command_palette_binding(self) -> tuple[Binding, bool, str] | None:
        if not (self.show_command_palette and self.app.ENABLE_COMMAND_PALETTE):
            return None
        try:
            _node, binding, enabled, tooltip = self.screen.active_bindings[
                self.app.COMMAND_PALETTE_BINDING
            ]
        except KeyError:
            return None
        return binding, enabled, binding.tooltip or binding.description

    def _prioritize(
        self, groups: list[list[tuple[Binding, bool, str]]]
    ) -> list[list[tuple[Binding, bool, str]]]:
        """Order footer hints so primary actions survive narrow terminals."""
        return sorted(groups, key=lambda group: self._priority_for(group[0][0]))

    def _priority_for(self, binding: Binding) -> int:
        """Return the presentation priority for one footer binding."""
        explicit = getattr(binding, "priority", 0)
        # Textual's priority is a dispatch order; screens use it for hidden
        # keys (enter-to-choose shadows submit). Only non-zero footer-visible
        # priorities participate here so that mechanism is never overloaded.
        try:
            action = str(binding.action or "")
        except Exception:
            action = ""
        base = self.FOOTER_PRIORITY.get(action, self._DEFAULT_FOOTER_PRIORITY)
        if explicit:
            return base - 1
        return base

    def _fit(
        self,
        groups: list[list[tuple[Binding, bool, str]]],
        budget: int,
    ) -> tuple[list[tuple[Binding, bool, str]], int]:
        """Return the hints that fit in ``budget`` cells, and how many do not.

        The overflow marker only earns its place if it stands for more hints
        than the one it would displace, so the last hint is kept whenever
        dropping it would buy nothing.
        """
        if budget <= 0:
            return ([group[0] for group in groups], 0)

        widths = [
            _hint_width(self.app.get_key_display(group[0][0]), group[0][0].description)
            for group in groups
        ]
        used = 0
        fitted = 0
        for width in widths:
            if used + width > budget:
                break
            used += width
            fitted += 1

        if fitted == len(groups):
            return ([group[0] for group in groups], 0)

        # Help is the way back to the dropped hints, so it is kept even when
        # that costs another hint's width. The loop below never drops index 0
        # because _prioritize sorts show_help there.
        marker_width = _hint_width("", _OVERFLOW_TEMPLATE.format(count=len(groups)))
        while fitted > 1 and used + marker_width > budget:
            fitted -= 1
            used -= widths[fitted]

        return ([group[0] for group in groups[:fitted]], len(groups) - fitted)
