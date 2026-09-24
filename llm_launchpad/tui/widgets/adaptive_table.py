"""Data table that changes columns without losing the highlighted row."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Callable, Iterable

from textual.coordinate import Coordinate
from textual.widgets import DataTable

from ..responsive import ViewportProfile, WidthMode


CellFactory = Callable[[Any], object]
RowKeyFactory = Callable[[Any], str]


@dataclass(frozen=True)
class AdaptiveColumn:
    """A table column and the viewport modes in which it is useful."""

    key: str
    label: str
    value: CellFactory
    modes: frozenset[WidthMode]
    width: int | None = None
    # Hide the column when every row renders one of these values: a column of
    # dashes is noise, and it pushes the useful columns apart.
    empty_values: frozenset[str] = frozenset()
    # Also hide it when every row says what another column already says.
    redundant: Callable[[Any], bool] | None = None

    @classmethod
    def visible(
        cls,
        key: str,
        label: str,
        value: CellFactory,
        *modes: WidthMode,
        width: int | None = None,
        hide_when_empty: bool = False,
        redundant: Callable[[Any], bool] | None = None,
    ) -> AdaptiveColumn:
        return cls(
            key=key,
            label=label,
            value=value,
            modes=frozenset(modes),
            width=width,
            empty_values=frozenset({"", "-"}) if hide_when_empty else frozenset(),
            redundant=redundant,
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

    def __init__(self, *args: Any, fit_to_rows: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # With `height: 1fr` alone a two-row inventory sat in a box twenty rows
        # tall. Fitting caps the height at the rows plus header and border;
        # the stylesheet's min-height still keeps a usable floor.
        self._fit_to_rows = fit_to_rows
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
        """Change presentation only when the horizontal mode changes."""
        if self._profile is not None and self._profile.width_mode == profile.width_mode:
            self._profile = profile
            return
        self._profile = profile
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
        self.clear(columns=True)
        for column in columns:
            self.add_column(column.label, key=column.key, width=column.width)

        restored_row = 0
        for index, row in enumerate(self._adaptive_rows):
            key = self._row_key(row)
            self.add_row(
                *(column.value(row) for column in columns),
                key=key,
            )
            if key == highlighted_key:
                restored_row = index
        if self.row_count:
            self.move_cursor(row=restored_row, column=0, animate=False)
        if self._fit_to_rows:
            # Rows, the header, two border rows and a horizontal scrollbar
            # row. An inline cap outranks the stylesheet's min-height, so the
            # floor is restated here rather than inherited.
            self.styles.max_height = max(self.row_count + 4, _FIT_MIN_ROWS)
