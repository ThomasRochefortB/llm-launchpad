from __future__ import annotations

import unittest

from textual.app import App
from textual.widgets import Input, Static

from llm_launchpad.core.config import ConfigSaveResult
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

            def fake_save_result(settings: object) -> ConfigSaveResult:
                saved["settings"] = settings
                return ConfigSaveResult(success=True, path=screen._store.path)

            screen._store.save_result = fake_save_result  # type: ignore[method-assign]
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


class SettingsArrowNavigationTests(unittest.IsolatedAsyncioTestCase):
    """Every control here has to be reachable without a tab key.

    This screen had no arrow navigation at all: focus started in an input and
    the arrows moved nothing, so everything below was tab-only. On a phone
    keyboard tab is a modifier-bar item -- and this is the screen where mouse
    support gets switched on when taps are being ignored, so it is the last
    place that should require one.
    """

    async def test_arrows_reach_the_mouse_switch_and_the_save_button(self) -> None:
        app = _TestApp()
        async with app.run_test(size=(52, 30)) as pilot:
            app.push_screen(SettingsScreen())
            await pilot.pause()
            await pilot.pause()

            reached = []
            for _ in range(6):
                await pilot.press("down")
                await pilot.pause()
                reached.append(getattr(app.focused, "id", None))

            self.assertIn("tui-mouse", reached)
            self.assertIn("save-btn", reached)

    async def test_up_walks_back(self) -> None:
        app = _TestApp()
        async with app.run_test(size=(52, 30)) as pilot:
            app.push_screen(SettingsScreen())
            await pilot.pause()
            await pilot.pause()
            start = getattr(app.focused, "id", None)

            await pilot.press("down")
            await pilot.pause()
            moved = getattr(app.focused, "id", None)
            await pilot.press("up")
            await pilot.pause()

            self.assertNotEqual(moved, start)
            self.assertEqual(getattr(app.focused, "id", None), start)


class MouseSettingReflectsRealStateTests(unittest.IsolatedAsyncioTestCase):
    """An unset preference is not the same as an enabled one."""

    async def test_the_switch_shows_the_resolved_default(self) -> None:
        from unittest.mock import patch

        from textual.widgets import Switch

        # Over SSH the default resolves to off so the terminal keeps its own
        # selection. Showing a flat True told an SSH user clicks were enabled
        # while the app was ignoring every one of them.
        with patch(
            "llm_launchpad.tui.screens.settings.default_tui_mouse_enabled",
            return_value=False,
        ):
            app = _TestApp()
            async with app.run_test(size=(52, 30)) as pilot:
                app.push_screen(SettingsScreen())
                await pilot.pause()

                self.assertFalse(app.screen.query_one("#tui-mouse", Switch).value)
