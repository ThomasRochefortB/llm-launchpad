from __future__ import annotations

import unittest

from textual.app import App
from textual.widgets import Input, Static

from llm_launchpad.tui.screens.settings import SettingsScreen


class _TestApp(App[None]):
    pass


class SettingsScreenTests(unittest.IsolatedAsyncioTestCase):
    async def test_save_scaledown_persists_settings(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(SettingsScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, SettingsScreen)

            screen.query_one("#scaledown-window", Input).value = "900"
            saved: dict[str, object] = {}

            def fake_save(settings: object) -> None:
                saved["settings"] = settings

            screen._store.save = fake_save  # type: ignore[method-assign]
            screen._save()

            saved_settings = saved.get("settings")
            self.assertIsNotNone(saved_settings)
            assert saved_settings is not None
            self.assertEqual(saved_settings.scaledown_window, 900)
            feedback = str(screen.query_one("#save-feedback", Static).content)
            self.assertIn("Settings saved", feedback)

    async def test_invalid_scaledown_shows_error(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(SettingsScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SettingsScreen)

            screen.query_one("#scaledown-window", Input).value = "nope"
            screen._save()

            feedback = str(screen.query_one("#save-feedback", Static).content)
            self.assertIn("Scaledown must be an integer", feedback)


if __name__ == "__main__":
    unittest.main()
