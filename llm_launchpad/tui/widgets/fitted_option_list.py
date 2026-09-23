"""An option list that stops growing once its options fit."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Self

from textual import events
from textual.widgets import OptionList
from textual.widgets._option_list import OptionListContent

# Border rows around the options.
_CHROME_ROWS = 2
# The stylesheet's ceiling for the list on an ordinary terminal.
_DEFAULT_MAX_ROWS = 24


class FittedOptionList(OptionList):
    """An ``OptionList`` that takes ``1fr`` but never more rows than it has.

    With ``height: 1fr`` alone, three models sat in a twenty-row box and the
    detail panel describing the highlighted one was pushed to the bottom of
    the screen. CSS cannot say "fill, but no taller than the content"
    (``max-height: 1fr`` resolves against the whole container), so the cap is
    set here whenever the options change or the list is resized.

    Each option is assumed to occupy one row, which holds for lists styled
    ``text-wrap: nowrap``.
    """

    _applied_cap: int | None = None

    def clear_options(self) -> Self:
        super().clear_options()
        self._fit_to_options()
        return self

    def add_options(self, new_options: Iterable[OptionListContent]) -> Self:
        super().add_options(new_options)
        self._fit_to_options()
        return self

    def on_mount(self) -> None:
        self._fit_to_options()

    def on_resize(self, _event: events.Resize) -> None:
        self._fit_to_options()

    def _fit_to_options(self) -> None:
        if not self.is_mounted:
            return
        # A narrow, short viewport pins the list to a few rows in the
        # stylesheet; an inline cap would override that, so leave it be.
        if self.screen.has_class("viewport-minimal") and self.screen.has_class(
            "viewport-short"
        ):
            cap = None
        else:
            cap = min(max(self.option_count, 1) + _CHROME_ROWS, _DEFAULT_MAX_ROWS)
        # styles.max_height reads back as a Scalar, so remember what was set;
        # re-applying an unchanged cap on every resize would relayout for
        # nothing.
        if cap != self._applied_cap:
            self._applied_cap = cap
            self.styles.max_height = cap
