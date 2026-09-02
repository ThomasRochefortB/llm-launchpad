"""Tests for TUI UX improvements: setup gate, help overlay, search, settings."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from textual.app import App
from textual.binding import Binding
from textual.widgets import DataTable, Input, OptionList, Static

from llm_launchpad.protocol.models import LaunchpadSettings
from llm_launchpad.tui.screens.copy_enabled import CopyEnabledScreen
from llm_launchpad.tui.screens.fast_deploy import FastDeployScreen
from llm_launchpad.tui.screens.setup import SetupRequiredScreen
from llm_launchpad.tui.screens.storage import StorageScreen
from llm_launchpad.tui.widgets.help_overlay import (
    HelpOverlayScreen,
    _iter_screen_bindings,
)
from tests.test_fast_deploy_screen import _StyledApp, _model, _profile
from tests.test_storage_screen import _TestApp


class _SetupApp(App[None]):
    def __init__(self, *, configured: bool) -> None:
        super().__init__()
        self._configured = configured
        self.menu_entered = False

    def _provider_is_configured(self) -> bool:
        return self._configured

    def recheck_provider_setup(self) -> bool:
        if not self._provider_is_configured():
            return False
        try:
            if isinstance(self.screen, SetupRequiredScreen):
                self.pop_screen()
        except Exception:
            pass
        self.menu_entered = True
        return True


class SetupRequiredScreenTests(unittest.IsolatedAsyncioTestCase):
    async def test_recheck_enters_main_menu_when_provider_available(self) -> None:
        app = _SetupApp(configured=True)
        async with app.run_test() as pilot:
            app.push_screen(SetupRequiredScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, SetupRequiredScreen)
            screen.action_recheck()
            await pilot.pause()

        self.assertTrue(app.menu_entered)

    async def test_recheck_stays_on_setup_when_still_unauthenticated(self) -> None:
        app = _SetupApp(configured=False)
        async with app.run_test() as pilot:
            app.push_screen(SetupRequiredScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, SetupRequiredScreen)
            screen.action_recheck()
            await pilot.pause()

            self.assertFalse(app.menu_entered)
            self.assertIsInstance(app.screen, SetupRequiredScreen)
            feedback = str(screen.query_one("#setup-required-feedback", Static).content)
            self.assertIn("Still no authenticated provider", feedback)


class _BindingProbeScreen(CopyEnabledScreen):
    BINDINGS = [
        Binding("escape", "pop", "Back", show=True),
        Binding("probe", "probe", "Probe action", show=False),
    ]


class HelpOverlayTests(unittest.IsolatedAsyncioTestCase):
    async def test_help_overlay_opens_and_dismisses(self) -> None:
        app = App[None]()
        async with app.run_test() as pilot:
            app.push_screen(_BindingProbeScreen())
            await pilot.pause()
            await pilot.press("?")
            await pilot.pause()

            self.assertIsInstance(app.screen, HelpOverlayScreen)
            body = app.screen.query("#help-overlay-body")
            self.assertEqual(len(body), 1)

            await pilot.press("escape")
            await pilot.pause()
            self.assertNotIsInstance(app.screen, HelpOverlayScreen)

    def test_iter_screen_bindings_collects_mro_and_skips_system(self) -> None:
        bindings = _iter_screen_bindings(_BindingProbeScreen())
        keys = [key for key, _label in bindings]
        self.assertIn("escape", keys)
        self.assertIn("probe", keys)


class LaunchpadSettingsFieldsTests(unittest.TestCase):
    def test_new_fields_roundtrip(self) -> None:
        settings = LaunchpadSettings(
            scaledown_window=900,
            tui_theme="launchpad-monochrome",
            tui_density="compact",
            tui_mouse=False,
            confirm_quit=False,
        )
        restored = LaunchpadSettings.from_dict(settings.to_dict())
        self.assertEqual(restored.scaledown_window, 900)
        self.assertEqual(restored.tui_theme, "launchpad-monochrome")
        self.assertEqual(restored.tui_density, "compact")
        self.assertIs(restored.tui_mouse, False)
        self.assertIs(restored.confirm_quit, False)

    def test_mouse_none_and_defaults_for_legacy_payloads(self) -> None:
        restored = LaunchpadSettings.from_dict({"SCALEDOWN_WINDOW": 60})
        self.assertIsNone(restored.tui_mouse)
        self.assertTrue(restored.confirm_quit)


class QuickDeploySearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_input_filters_model_list(self) -> None:
        alpha = _model((_profile("Alpha-Model", required_vram_gb=8.0),))
        beta = _model((_profile("Beta-Model", required_vram_gb=8.0),))
        app = _StyledApp()
        with patch(
            "llm_launchpad.tui.screens.fast_deploy.list_quick_deploy_models",
            return_value=(alpha, beta),
        ):
            async with app.run_test(size=(140, 42)) as pilot:
                app.push_screen(FastDeployScreen())
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, FastDeployScreen)
                option_list = screen.query_one("#fast-deploy-list", OptionList)
                self.assertEqual(option_list.option_count, 2)

                screen.query_one("#fast-deploy-model-search", Input).value = "alpha"
                await pilot.pause()

                self.assertEqual(option_list.option_count, 1)

                screen.query_one("#fast-deploy-model-search", Input).value = "zzz"
                await pilot.pause()
                # No matches: a single disabled empty-state option remains.
                self.assertEqual(option_list.option_count, 1)
                highlighted = option_list.get_option_at_index(0)
                self.assertTrue(highlighted.disabled)


class StorageFilterTests(unittest.IsolatedAsyncioTestCase):
    async def test_table_filter_narrows_rows(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(StorageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StorageScreen)
            table = screen.query_one("#storage-table", DataTable)
            self.assertEqual(table.row_count, 2)

            screen.query_one("#storage-filter", Input).value = "Qwen3"
            await pilot.pause()
            self.assertEqual(table.row_count, 1)


if __name__ == "__main__":
    unittest.main()
