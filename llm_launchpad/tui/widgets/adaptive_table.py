"""Data table that changes columns without losing the highlighted row."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

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

    @classmethod
    def visible(
        cls,
        key: str,
        label: str,
        value: CellFactory,
        *modes: WidthMode,
        width: int | None = None,
    ) -> AdaptiveColumn:
        return cls(
            key=key,
            label=label,
            value=value,
            modes=frozenset(modes),
            width=width,
        )


class AdaptiveDataTable(DataTable[Any]):
    """Rebuild visible columns while retaining row data and cursor identity."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._adaptive_columns: tuple[AdaptiveColumn, ...] = ()
        self._adaptive_rows: tuple[Any, ...] = ()
        self._row_key: RowKeyFactory = lambda row: str(row)
        self._profile: ViewportProfile | None = None

    @property
    def visible_column_keys(self) -> tuple[str, ...]:
        """Column keys currently rendered, primarily for diagnostics/tests."""
        if self._profile is None:
            return ()
        return tuple(
            column.key
            for column in self._adaptive_columns
            if self._profile.width_mode in column.modes
        )

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
        columns = [
            column
            for column in self._adaptive_columns
            if self._profile.width_mode in column.modes
        ]
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
