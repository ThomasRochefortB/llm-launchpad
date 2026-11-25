"""Data table that changes columns without losing the highlighted row."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Callable, Iterable

from rich.cells import cell_len
from rich.console import JustifyMethod
from rich.text import Text
from textual.content import Content
from textual.coordinate import Coordinate
from textual.widgets import DataTable

from ..responsive import ViewportProfile, WidthMode


CellFactory = Callable[[Any], object]
RowKeyFactory = Callable[[Any], str]


def _justified(column: AdaptiveColumn, value: object) -> object:
    """Align a string cell as its column asks; other renderables pass through."""
    if column.justify == "left" or not isinstance(value, str):
        return value
    try:
        return Text.from_markup(value, justify=column.justify)
    except Exception:
        return Text(value, justify=column.justify)


@dataclass(frozen=True)
class AdaptiveColumn:
    """A table column and the viewport modes in which it is useful."""

    key: str
    label: str
    value: CellFactory
    modes: frozenset[WidthMode]
    width: int | None = None
    # Hide the column when the terminal is narrower than this width even within
    # an allowed WidthMode (prevents horizontal clipping around 100 columns).
    min_terminal_width: int | None = None
    # Hide the column when every row renders one of these values: a column of
    # dashes is noise, and it pushes the useful columns apart.
    empty_values: frozenset[str] = frozenset()
    # Also hide it when every row says what another column already says.
    redundant: Callable[[Any], bool] | None = None
    # Numbers read down a column only when their digits line up.
    justify: JustifyMethod = "left"

    @classmethod
    def visible(
        cls,
        key: str,
        label: str,
        value: CellFactory,
        *modes: WidthMode,
        width: int | None = None,
        min_terminal_width: int | None = None,
        hide_when_empty: bool = False,
        redundant: Callable[[Any], bool] | None = None,
        justify: JustifyMethod = "left",
    ) -> AdaptiveColumn:
        return cls(
            key=key,
            label=label,
            value=value,
            modes=frozenset(modes),
            width=width,
            min_terminal_width=min_terminal_width,
            empty_values=frozenset({"", "-"}) if hide_when_empty else frozenset(),
            redundant=redundant,
            justify=justify,
        )

    def is_empty_for(self, rows: tuple[Any, ...]) -> bool:
        """Whether this column says nothing for any of ``rows``."""
        if not rows or (not self.empty_values and self.redundant is None):
            return False
        return all(
            str(self.value(row)).strip() in self.empty_values
            or (self.redundant is not None and self.redundant(row))
            for row in rows
        )


# The fewest rows a fitted table shrinks to: the header and a few rows, so an
# empty or one-row inventory still reads as a table.
_FIT_MIN_ROWS = 9


class AdaptiveDataTable(DataTable[Any]):
    """Rebuild visible columns while retaining row data and cursor identity."""

    def __init__(
        self,
        *args: Any,
        fit_to_rows: bool = False,
        min_fit_rows: int = _FIT_MIN_ROWS,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # With `height: 1fr` alone a two-row inventory sat in a box twenty rows
        # tall. Fitting caps the height at the rows plus header and border;
        # the stylesheet's min-height still keeps a usable floor.
        self._fit_to_rows = fit_to_rows
        self._min_fit_rows = max(4, min_fit_rows)
        self._adaptive_columns: tuple[AdaptiveColumn, ...] = ()
        self._adaptive_rows: tuple[Any, ...] = ()
        self._row_key: RowKeyFactory = lambda row: str(row)
        self._profile: ViewportProfile | None = None

    @property
    def visible_column_keys(self) -> tuple[str, ...]:
        """Column keys currently rendered, primarily for diagnostics/tests."""
        return tuple(column.key for column in self._visible_columns())

    def _visible_columns(self) -> list[AdaptiveColumn]:
        if self._profile is None:
            return []
        return [
            column
            for column in self._adaptive_columns
            if self._profile.width_mode in column.modes
            and (
                column.min_terminal_width is None
                or self._profile.width >= column.min_terminal_width
            )
            and not column.is_empty_for(self._adaptive_rows)
        ]

    def configure(
        self,
        columns: Iterable[AdaptiveColumn],
        *,
        row_key: RowKeyFactory,
        profile: ViewportProfile,
    ) -> None:
        """Install a schema and render any retained rows."""
        self._adaptive_columns = tuple(columns)
        self._row_key = row_key
        self._profile = profile
        self._rebuild()

    def set_rows(self, rows: Iterable[Any]) -> None:
        """Replace source rows and repaint using the active presentation."""
        self._adaptive_rows = tuple(rows)
        self._rebuild()

    def set_viewport_profile(self, profile: ViewportProfile) -> None:
        """Rebuild when the horizontal mode or column visibility changes."""
        previous_keys = self.visible_column_keys
        previous_mode = self._profile.width_mode if self._profile is not None else None
        self._profile = profile
        if previous_mode == profile.width_mode and self.visible_column_keys == previous_keys:
            return
        self._rebuild()

    def _highlighted_row_key(self) -> str | None:
        if self.row_count == 0 or not self.columns:
            return None
        row = min(self.cursor_row, self.row_count - 1)
        try:
            return str(self.coordinate_to_cell_key(Coordinate(row, 0)).row_key.value)
        except Exception:
            return None

    def _rebuild(self) -> None:
        if not self.is_mounted or self._profile is None:
            return
        highlighted_key = self._highlighted_row_key()
        columns = self._visible_columns()
        rendered_rows: list[tuple[str, tuple[object, ...]]] = [
            (
                self._row_key(row),
                tuple(_justified(column, column.value(row)) for column in columns),
            )
            for row in self._adaptive_rows
        ]
        self.clear(columns=True)
        for col_index, column in enumerate(columns):
            label: str | Text = column.label
            if column.justify != "left":
                label = Text(column.label, justify=column.justify)
            col_width = column.width
            # DataTable's auto-width only measures str/Text cells, not Textual
            # Content objects; size columns containing Content cells explicitly
            # so theme-styled badges and bold labels are never clipped.
            if col_width is None and any(
                isinstance(cells[col_index], Content) for _, cells in rendered_rows
            ):
                col_width = max(
                    cell_len(column.label),
                    *(
                        cell.cell_length if isinstance(cell, Content) else cell_len(str(cell))
                        for _, cells in rendered_rows
                        for cell in (cells[col_index],)
                    ),
                )
            self.add_column(label, key=column.key, width=col_width)

        restored_row = 0
        for index, (key, cells) in enumerate(rendered_rows):
            self.add_row(*cells, key=key)
            if key == highlighted_key:
                restored_row = index
        if self.row_count:
            self.move_cursor(row=restored_row, column=0, animate=False)
        if self._fit_to_rows:
            # Rows, the header, two border rows and a horizontal scrollbar
            # row. An inline cap outranks the stylesheet's min-height, so the
            # floor is restated here rather than inherited.
            self.styles.max_height = max(self.row_count + 4, self._min_fit_rows)
