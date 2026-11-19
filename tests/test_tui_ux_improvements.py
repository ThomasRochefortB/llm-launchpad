"""Tests for TUI UX improvements: setup gate, help overlay, search, settings."""

from __future__ import annotations

import asyncio
import threading
import unittest
from unittest.mock import patch

from textual.app import App
from textual.binding import Binding
from textual.widgets import DataTable, Input, OptionList

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


class SetupRequiredScreenTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_prime_key_opens_home_while_modal_is_still_checking(self) -> None:
        from llm_launchpad.core.provider_readiness import ProviderReadiness, ProviderReadinessStage
        from llm_launchpad.protocol.enums import ComputeProvider
        from llm_launchpad.tui.app import TuiApp
        from llm_launchpad.tui.screens.main_menu import MainMenuScreen

        release = threading.Event()

        def slow_modal(**_kwargs: bool) -> ProviderReadiness:
            release.wait(5)
            return ProviderReadiness(ComputeProvider.MODAL, ProviderReadinessStage.UNREACHABLE)

        prime = ComputeProvider.PRIME
        with (
            patch("llm_launchpad.tui.app.check_modal_readiness", side_effect=slow_modal),
            patch("llm_launchpad.tui.app.check_prime_readiness", side_effect=[
                ProviderReadiness(prime, ProviderReadinessStage.CREDENTIALS_PRESENT),
                ProviderReadiness(prime, ProviderReadinessStage.READY),
            ]),
        ):
            app = TuiApp()
            async with app.run_test() as pilot:
                try:
                    async with asyncio.timeout(3):
                        while not isinstance(app.screen, MainMenuScreen):
                            await pilot.pause(0.02)
                    self.assertIn(ComputeProvider.MODAL, app._readiness_pending)
                    self.assertIn(prime, {row.provider for row in app.provider_readiness_snapshot()})
                finally:
                    release.set()

    async def test_recheck_enters_home_after_auth_without_repeating_a_pending_probe(self) -> None:
        from llm_launchpad.core.modal_auth import ModalAuthStatus
        from llm_launchpad.tui.app import TuiApp
        from llm_launchpad.tui.screens.main_menu import MainMenuScreen

        app = TuiApp()
        release = threading.Event()

        def slow_auth() -> ModalAuthStatus:
            release.wait(5)
            return ModalAuthStatus(authenticated=True, profile="test-profile")

        with (
            patch("llm_launchpad.core.provider_readiness.get_modal_auth_status",
                  return_value=ModalAuthStatus(False)) as auth,
            patch("llm_launchpad.core.provider_readiness.ModalBackend.is_cli_available", return_value=True),
            patch.object(MainMenuScreen, "_refresh_modal_auth_status") as home_probe,
        ):
            async with app.run_test() as pilot:
                async with asyncio.timeout(5):
                    while len(app.provider_readiness_snapshot()) < 3:
                        await pilot.pause(0.02)
                self.assertIsInstance(app.screen, SetupRequiredScreen)
                self.assertEqual(auth.call_count, 1)
                auth.side_effect = slow_auth
                try:
                    await pilot.press("r", "r", "?")
                    self.assertIsInstance(app.screen, HelpOverlayScreen)
                    self.assertEqual(auth.call_count, 2)
                    release.set()
                    async with asyncio.timeout(5):
                        while not any(row.verified for row in app.provider_readiness_snapshot()):
                            await pilot.pause(0.02)
                    self.assertIsInstance(app.screen, HelpOverlayScreen)
                    await pilot.press("escape")
                    await pilot.pause()
                    self.assertIsInstance(app.screen, MainMenuScreen)
                    self.assertEqual(app.screen.username, "test-profile")
                    self.assertEqual(len(app.screen_stack), 2)
                    home_probe.assert_not_called()
                    self.assertEqual(auth.call_count, 2)
                finally:
                    release.set()


class _BindingProbeScreen(CopyEnabledScreen):
    BINDINGS = [
        Binding("escape", "pop", "Back", show=True),
        Binding("probe", "probe", "Probe action", show=False),
        Binding("up", "navigate_option_list_up", show=False),
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

    def test_undescribed_bindings_do_not_leak_their_action_names(self) -> None:
        """A binding with no description is plumbing, not a shortcut.

        Falling back to the action name printed "up navigate_option_list_up"
        in the help of every screen that hops focus with the arrow keys.
        """
        bindings = _iter_screen_bindings(_BindingProbeScreen())
        self.assertNotIn("up", [key for key, _label in bindings])
        for _key, label in bindings:
            self.assertTrue(label)
            self.assertNotIn("_", label)

    async def test_help_lists_only_described_bindings_on_a_real_screen(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(StorageScreen())
            await pilot.pause()
            labels = [label for _key, label in _iter_screen_bindings(app.screen)]
            self.assertIn("Pre-download", labels)
            self.assertNotIn("navigate_option_list_up", labels)
            self.assertNotIn("navigate_option_list_down", labels)


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
        from llm_launchpad.core.compute_availability import aggregate_compute_availability
        from llm_launchpad.core.modal_gpu import ModalGpuSpec

        self.enterContext(patch(
            "llm_launchpad.tui.screens.fast_deploy.load_compute_availability",
            return_value=aggregate_compute_availability(modal_catalog=[ModalGpuSpec("L4", 0.8)]),
        ))
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


class SetupAutoContinueTests(unittest.IsolatedAsyncioTestCase):
    """Credentials found in the background take the reader past Setup."""

    async def _run(self, *, stage: str, configured: bool) -> type:
        from llm_launchpad.core.provider_readiness import (
            ProviderReadiness,
            ProviderReadinessStage,
        )
        from llm_launchpad.protocol.enums import ComputeProvider
        from llm_launchpad.tui.app import ProviderReadinessUpdated, TuiApp
        from llm_launchpad.tui.screens.main_menu import MainMenuScreen

        class _App(TuiApp):
            def on_mount(self) -> None:
                self.push_screen(SetupRequiredScreen())

            def _provider_is_configured(self) -> bool:
                return configured

        app = _App()
        with patch.object(MainMenuScreen, "on_mount", lambda self: None):
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                app.post_message(ProviderReadinessUpdated((
                    ProviderReadiness(
                        provider=ComputeProvider.VAST,
                        stage=ProviderReadinessStage(stage),
                    ),
                )))
                await pilot.pause()
                await pilot.pause()
                return type(app.screen)

    async def test_a_verified_provider_enters_the_main_menu(self) -> None:
        from llm_launchpad.tui.screens.main_menu import MainMenuScreen

        screen_type = await self._run(stage="ready", configured=True)
        self.assertIs(screen_type, MainMenuScreen)

    async def test_a_failed_check_leaves_setup_in_place(self) -> None:
        screen_type = await self._run(stage="auth_failed", configured=False)
        self.assertIs(screen_type, SetupRequiredScreen)
