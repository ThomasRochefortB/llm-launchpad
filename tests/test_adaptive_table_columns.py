"""AdaptiveDataTable leaves out columns that say nothing for any row."""

from __future__ import annotations

import unittest

from textual.app import App, ComposeResult
from textual.geometry import Size

from llm_launchpad.tui.responsive import ViewportProfile, WidthMode
from llm_launchpad.tui.widgets.adaptive_table import AdaptiveColumn, AdaptiveDataTable

_COLUMNS = (
    AdaptiveColumn.visible("name", "name", lambda row: row["name"], WidthMode.WIDE),
    AdaptiveColumn.visible(
        "rate", "tok/s", lambda row: row["rate"], WidthMode.WIDE, hide_when_empty=True
    ),
    # Opted out: a dash here is information ("no state"), not an empty column.
    AdaptiveColumn.visible("state", "state", lambda row: row["state"], WidthMode.WIDE),
)


class _TableApp(App):
    def compose(self) -> ComposeResult:
        yield AdaptiveDataTable()


class EmptyColumnTests(unittest.IsolatedAsyncioTestCase):
    async def _keys(self, rows: list[dict[str, str]]) -> tuple[str, ...]:
        app = _TableApp()
        async with app.run_test(size=(100, 20)) as pilot:
            table = app.query_one(AdaptiveDataTable)
            table.configure(
                _COLUMNS,
                row_key=lambda row: row["name"],
                profile=ViewportProfile.from_size(Size(140, 40)),
            )
            table.set_rows(rows)
            await pilot.pause()
            return table.visible_column_keys

    async def test_a_column_of_dashes_is_left_out(self) -> None:
        keys = await self._keys([
            {"name": "a", "rate": "-", "state": "-"},
            {"name": "b", "rate": "", "state": "-"},
        ])
        self.assertEqual(keys, ("name", "state"))

    async def test_one_value_brings_the_column_back(self) -> None:
        keys = await self._keys([
            {"name": "a", "rate": "-", "state": "-"},
            {"name": "b", "rate": "12 tok/s", "state": "-"},
        ])
        self.assertEqual(keys, ("name", "rate", "state"))

    async def test_a_column_repeating_another_is_left_out(self) -> None:
        app_column = AdaptiveColumn.visible(
            "app", "app", lambda row: row["app"], WidthMode.WIDE,
            redundant=lambda row: row["app"] == row["name"],
        )
        rows = [{"name": "a", "app": "a"}, {"name": "b", "app": "b"}]
        self.assertTrue(app_column.is_empty_for(tuple(rows)))
        rows[1]["app"] = "pod-7"
        self.assertFalse(app_column.is_empty_for(tuple(rows)))

    async def test_an_empty_table_keeps_its_header(self) -> None:
        self.assertEqual(await self._keys([]), ("name", "rate", "state"))


if __name__ == "__main__":
    unittest.main()
