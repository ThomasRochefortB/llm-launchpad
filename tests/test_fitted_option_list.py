"""FittedOptionList fills its space but never grows past its options."""

from __future__ import annotations

import unittest

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from llm_launchpad.tui.widgets.fitted_option_list import FittedOptionList


class _ListApp(App):
    CSS = """
    Vertical { height: 1fr; }
    FittedOptionList { height: 1fr; border: round $primary; }
    Static { height: 3; }
    """

    def __init__(self, count: int) -> None:
        super().__init__()
        self._count = count

    def compose(self) -> ComposeResult:
        with Vertical():
            yield FittedOptionList(*(f"option {index}" for index in range(self._count)))
            yield Static("detail", id="detail")


class FittedOptionListTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_short_list_stops_at_its_options(self) -> None:
        app = _ListApp(3)
        async with app.run_test(size=(40, 30)) as pilot:
            await pilot.pause()
            option_list = app.query_one(FittedOptionList)
            self.assertEqual(option_list.size.height, 3)
            detail = app.query_one("#detail")
            self.assertEqual(detail.region.y, option_list.region.bottom)

    async def test_a_long_list_still_fills_only_the_space_it_has(self) -> None:
        app = _ListApp(60)
        async with app.run_test(size=(40, 20)) as pilot:
            await pilot.pause()
            option_list = app.query_one(FittedOptionList)
            # 20 rows less the 3-row detail below it.
            self.assertEqual(option_list.region.height, 17)

    async def test_the_cap_follows_the_options(self) -> None:
        app = _ListApp(2)
        async with app.run_test(size=(40, 30)) as pilot:
            await pilot.pause()
            option_list = app.query_one(FittedOptionList)
            option_list.set_options([f"new {index}" for index in range(5)])
            await pilot.pause()
            self.assertEqual(option_list.size.height, 5)
            option_list.clear_options()
            await pilot.pause()
            self.assertEqual(option_list.size.height, 1)


if __name__ == "__main__":
    unittest.main()
