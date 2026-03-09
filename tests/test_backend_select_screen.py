from __future__ import annotations

import unittest

from textual.app import App
from textual.widgets import OptionList

from llm_launchpad.tui.screens.deploy import BackendSelectScreen


class _TestApp(App[None]):
    pass


class BackendSelectScreenTests(unittest.IsolatedAsyncioTestCase):
    async def test_backend_menu_is_focused_and_arrow_navigable(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(BackendSelectScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, BackendSelectScreen)
            backend_list = screen.query_one("#backend-list", OptionList)

            self.assertTrue(backend_list.has_focus)
            highlighted = backend_list.highlighted_option
            self.assertIsNotNone(highlighted)
            assert highlighted is not None
            self.assertEqual(highlighted.id, "llamacpp")

            await pilot.press("down")
            await pilot.pause()

            highlighted = backend_list.highlighted_option
            self.assertIsNotNone(highlighted)
            assert highlighted is not None
            self.assertEqual(highlighted.id, "vllm")


if __name__ == "__main__":
    unittest.main()
