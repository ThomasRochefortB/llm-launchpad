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

    Hints are kept in binding order, which every screen already declares
    most-important-first, and whatever is left over is summarised by a trailing
    marker pointing at the help overlay -- which lists all of them.
    """

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

        shown, hidden = self._fit(list(action_to_bindings.values()), budget)

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

        marker_width = _hint_width("", _OVERFLOW_TEMPLATE.format(count=len(groups)))
        while fitted > 0 and used + marker_width > budget:
            fitted -= 1
            used -= widths[fitted]

        return ([group[0] for group in groups[:fitted]], len(groups) - fitted)
