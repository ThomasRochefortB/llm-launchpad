from __future__ import annotations

import unittest
from collections.abc import Callable
from contextlib import ExitStack
from unittest.mock import patch

from textual.app import App
from textual.containers import VerticalScroll
from textual.geometry import Size
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Footer, Input, OptionList, Static

from llm_launchpad.core.compute_availability import aggregate_compute_availability
from llm_launchpad.protocol.enums import BackendType
from llm_launchpad.protocol.models import EndpointInfo
from llm_launchpad.tui.app import TuiApp
from llm_launchpad.tui.responsive import HeightMode, ViewportProfile, WidthMode
from llm_launchpad.tui.screens.deploy import (
    BackendSelectScreen,
    LlamaCppDeployScreen,
    VllmDeployScreen,
)
from llm_launchpad.tui.screens.fast_deploy import FastDeployScreen
from llm_launchpad.tui.screens.main_menu import MainMenuScreen
from llm_launchpad.tui.screens.manage import (
    BenchmarkOptionsScreen,
    EndpointActionsScreen,
    ManageScreen,
    StatusOptionsScreen,
    StopConfirmScreen,
)
from llm_launchpad.tui.screens.monitor import MonitorScreen
from llm_launchpad.tui.screens.quick_deploy import QuickDeployScreen
from llm_launchpad.tui.screens.settings import SettingsScreen
from llm_launchpad.tui.screens.storage import StorageScreen


class _ScreenApp(App[None]):
    CSS_PATH = TuiApp.CSS_PATH

    def __init__(self, screen: Screen) -> None:
        super().__init__()
        self._initial_screen = screen

    def on_mount(self) -> None:
        self.push_screen(self._initial_screen)

    def list_instances(self) -> list[object]:
        return []


def _viewport_test_endpoint() -> EndpointInfo:
    return EndpointInfo(
        name="vllm-test",
        app_id="ap-test",
        state="running",
        backend=BackendType.VLLM,
    )


def _has_ancestor(widget: Widget, ancestor_type: type[Widget]) -> bool:
    parent = widget.parent
    while isinstance(parent, Widget):
        if isinstance(parent, ancestor_type):
            return True
        parent = parent.parent
    return False


class ResponsiveLayoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_breakpoints_and_size_gate_follow_runtime_resize(self) -> None:
        screen = SettingsScreen()
        app = _ScreenApp(screen)

        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause()
            field = screen.query_one("#scaledown-window", Input)
            field.value = "777"
            field.focus()
            await pilot.pause()

            self.assertFalse(screen.has_class("viewport-narrow"))
            await pilot.resize_terminal(50, 18)
            await pilot.pause()
            self.assertTrue(screen.has_class("viewport-narrow"))
            self.assertTrue(screen.has_class("viewport-compact"))
            self.assertTrue(screen.has_class("viewport-minimal"))
            self.assertTrue(screen.has_class("viewport-short"))
            self.assertEqual(field.value, "777")

            await pilot.resize_terminal(39, 12)
            await pilot.pause()
            overlay = screen.query_one("#minimum-size-overlay", Static)
            self.assertTrue(screen.has_class("viewport-too-small"))
            self.assertTrue(overlay.display)
            self.assertIn("39×12", str(overlay.content))
            self.assertIsNone(screen.focused)

            await pilot.press("x")
            self.assertEqual(field.value, "777")
            await pilot.resize_terminal(140, 45)
            await pilot.pause()
            self.assertFalse(screen.has_class("viewport-too-small"))
            self.assertFalse(overlay.display)
            self.assertIs(screen.focused, field)
            self.assertEqual(field.value, "777")

    async def test_main_menu_hides_secondary_content_when_space_is_constrained(self) -> None:
        screen = MainMenuScreen(username="alice", version="1.0")
        app = _ScreenApp(screen)

        with patch.object(MainMenuScreen, "on_mount", lambda self: None):
            async with app.run_test(size=(140, 45)) as pilot:
                await pilot.pause()
                side_column = screen.query_one("#main-menu-side-column")
                banner = screen.query_one("#banner-text")
                self.assertTrue(side_column.display)
                self.assertTrue(banner.display)

                await pilot.resize_terminal(100, 30)
                await pilot.pause()
                self.assertFalse(side_column.display)
                self.assertTrue(screen.query_one("#auth-status-block").display)
                self.assertTrue(screen.query_one("#action-list").display)

                screen.action_toggle_details()
                await pilot.pause()
                self.assertTrue(side_column.display)
                screen.action_close_details()
                await pilot.pause()
                self.assertFalse(side_column.display)

                await pilot.resize_terminal(50, 35)
                await pilot.pause()
                self.assertFalse(banner.display)

                await pilot.resize_terminal(120, 16)
                await pilot.pause()
                self.assertFalse(side_column.display)

    async def test_navigation_main_menu_fits_supported_viewports(self) -> None:
        await self._assert_screen_fits_supported_viewports(
            MainMenuScreen(username="alice", version="1.0")
        )

    async def test_navigation_backend_select_fits_supported_viewports(self) -> None:
        await self._assert_screen_fits_supported_viewports(BackendSelectScreen())

    async def test_navigation_quick_deploy_fits_supported_viewports(self) -> None:
        from tests.catalog_fixtures import STATIC_LIKE_PROFILES

        await self._assert_screen_fits_supported_viewports(
            QuickDeployScreen(STATIC_LIKE_PROFILES[0])
        )

    async def test_navigation_fast_deploy_fits_supported_viewports(self) -> None:
        await self._assert_screen_fits_supported_viewports(FastDeployScreen())

    async def test_deployment_llamacpp_fits_supported_viewports(self) -> None:
        await self._assert_screen_fits_supported_viewports(LlamaCppDeployScreen())

    async def test_deployment_vllm_fits_supported_viewports(self) -> None:
        await self._assert_screen_fits_supported_viewports(VllmDeployScreen())

    async def test_deployment_settings_fits_supported_viewports(self) -> None:
        await self._assert_screen_fits_supported_viewports(SettingsScreen())

    async def test_management_manage_fits_supported_viewports(self) -> None:
        await self._assert_screen_fits_supported_viewports(ManageScreen())

    async def test_management_endpoint_actions_fits_supported_viewports(self) -> None:
        await self._assert_screen_fits_supported_viewports(
            EndpointActionsScreen(_viewport_test_endpoint())
        )

    async def test_management_status_options_fits_supported_viewports(self) -> None:
        await self._assert_screen_fits_supported_viewports(
            StatusOptionsScreen(_viewport_test_endpoint())
        )

    async def test_management_benchmark_options_fits_supported_viewports(self) -> None:
        await self._assert_screen_fits_supported_viewports(
            BenchmarkOptionsScreen(_viewport_test_endpoint())
        )

    async def test_management_stop_confirm_fits_supported_viewports(self) -> None:
        await self._assert_screen_fits_supported_viewports(
            StopConfirmScreen(_viewport_test_endpoint())
        )

    async def test_operational_storage_fits_supported_viewports(self) -> None:
        await self._assert_screen_fits_supported_viewports(StorageScreen())

    async def test_operational_monitor_fits_supported_viewports(self) -> None:
        await self._assert_screen_fits_supported_viewports(MonitorScreen("Logs"))

    async def _assert_screen_fits_supported_viewports(
        self,
        screen: Screen,
    ) -> None:
        await self._assert_screen_families_fit_supported_viewports(
            (lambda screen=screen: screen,),
        )

    async def _assert_screen_families_fit_supported_viewports(
        self,
        factories: tuple[Callable[[], Screen], ...],
    ) -> None:
        sizes = (
            (140, 45),
            (100, 30),
            (80, 24),
            (60, 20),
            (50, 40),
            (50, 35),
            (40, 12),
            (120, 16),
            (200, 15),
            (220, 60),
        )

        for factory in factories:
            screen = factory()
            app = _ScreenApp(screen)
            with ExitStack() as patches:
                if isinstance(screen, MainMenuScreen):
                    for method_name in (
                        "_refresh_modal_auth_status",
                        "_refresh_prime_auth_status",
                        "_refresh_hf_auth_status",
                        "_refresh_aai_auth_status",
                        "_refresh_panels",
                        "_refresh_quick_deploy_catalog",
                        "_refresh_storage_estimate",
                    ):
                        patches.enter_context(
                            patch.object(MainMenuScreen, method_name, lambda self: None)
                        )
                if isinstance(screen, (LlamaCppDeployScreen, VllmDeployScreen)):
                    patches.enter_context(
                        patch.object(type(screen), "_refresh_gpu_types", lambda self: None)
                    )
                    patches.enter_context(
                        patch.object(
                            type(screen),
                            "_refresh_cached_models_from_storage",
                            lambda self: None,
                        )
                    )
                if isinstance(screen, VllmDeployScreen):
                    patches.enter_context(
                        patch.object(
                            VllmDeployScreen,
                            "_refresh_vllm_memory_status",
                            lambda self, *args, **kwargs: None,
                        )
                    )
                if isinstance(screen, FastDeployScreen):
                    patches.enter_context(
                        patch(
                            "llm_launchpad.tui.screens.fast_deploy.load_compute_availability",
                            return_value=aggregate_compute_availability(),
                        )
                    )
                if isinstance(screen, StorageScreen):
                    patches.enter_context(
                        patch.object(StorageScreen, "_refresh_storage_snapshot", lambda self: None)
                    )

                first_width, first_height = sizes[0]
                async with app.run_test(size=(first_width, first_height)) as pilot:
                    for index, (width, height) in enumerate(sizes):
                        with self.subTest(screen=type(screen).__name__, size=(width, height)):
                            if index:
                                await pilot.resize_terminal(width, height)
                            self.assertEqual(screen.region.width, width)
                            self.assertEqual(screen.region.height, height)
                            self.assertFalse(screen.has_class("viewport-too-small"))

                            for widget in screen.query("*"):
                                if not widget.display or widget.region.width == 0:
                                    continue
                                if _has_ancestor(widget, Footer):
                                    continue
                                self.assertLessEqual(
                                    widget.region.right,
                                    width,
                                    f"{type(widget).__name__}#{widget.id} exceeds viewport width",
                                )
                                if _has_ancestor(widget, VerticalScroll):
                                    continue
                                self.assertLessEqual(
                                    widget.region.bottom,
                                    height,
                                    f"{type(widget).__name__}#{widget.id} exceeds viewport height",
                                )

    def test_viewport_profile_classifies_width_and_height_independently(self) -> None:
        narrow_tall = ViewportProfile.from_size(Size(50, 40))
        self.assertEqual(narrow_tall.width_mode, WidthMode.MINIMAL)
        self.assertEqual(narrow_tall.height_mode, HeightMode.TALL)
        self.assertTrue(narrow_tall.compact)
        self.assertFalse(narrow_tall.short)

        wide_shallow = ViewportProfile.from_size(Size(200, 15))
        self.assertEqual(wide_shallow.width_mode, WidthMode.WIDE)
        self.assertEqual(wide_shallow.height_mode, HeightMode.SHALLOW)
        self.assertTrue(wide_shallow.ultra_wide)
        self.assertTrue(wide_shallow.short)

    async def test_main_menu_keeps_selected_action_when_labels_compact(self) -> None:
        screen = MainMenuScreen(username="alice", version="1.0")
        app = _ScreenApp(screen)
        with patch.object(MainMenuScreen, "on_mount", lambda self: None):
            async with app.run_test(size=(140, 35)) as pilot:
                await pilot.pause()
                action_list = screen.query_one("#action-list", OptionList)
                action_list.highlighted = 2
                selected_id = str(action_list.highlighted_option.id)

                await pilot.resize_terminal(50, 35)
                await pilot.pause()

                self.assertEqual(str(action_list.highlighted_option.id), selected_id)
                self.assertIn("Manage endpoints", str(action_list.highlighted_option.prompt))

    async def test_phone_portrait_main_menu_uses_contiguous_top_stack(self) -> None:
        screen = MainMenuScreen(username="alice", version="1.0")
        app = _ScreenApp(screen)
        with patch.object(MainMenuScreen, "on_mount", lambda self: None):
            async with app.run_test(size=(55, 60)) as pilot:
                await pilot.pause()

                header = screen.query_one("#compact-menu-header", Static)
                help_text = screen.query_one("#compact-menu-help", Static)
                auth = screen.query_one("#auth-status-block", Static)
                footer = screen.query_one(Footer)

                self.assertTrue(header.display)
                self.assertTrue(help_text.display)
                self.assertFalse(footer.display)
                self.assertLessEqual(auth.region.y - help_text.region.bottom, 2)
                self.assertLess(auth.region.y, screen.region.height // 2)


if __name__ == "__main__":
    unittest.main()
