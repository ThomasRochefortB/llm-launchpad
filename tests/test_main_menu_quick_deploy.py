from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from textual.app import App
from textual.widgets import OptionList

from llm_launchpad.tui.screens.main_menu import MainMenuScreen
from llm_launchpad.tui.screens.quick_deploy import QuickDeployScreen


class _TestApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.quick_deploy_calls: list[str] = []

    def push_quick_deploy(self, profile_id: str) -> None:
        self.quick_deploy_calls.append(profile_id)
        self.push_screen(QuickDeployScreen(profile_id=profile_id))

    def action_push_deploy(self) -> None:
        pass

    def action_push_manage(self) -> None:
        pass

    def action_push_storage(self) -> None:
        pass

    def action_push_settings(self) -> None:
        pass


class MainMenuQuickDeployTests(unittest.IsolatedAsyncioTestCase):
    async def test_main_menu_renders_quick_deploy_panel(self) -> None:
        app = _TestApp()
        with patch.object(MainMenuScreen, "_refresh_modal_auth_status", lambda self: None), patch.object(
            MainMenuScreen, "_refresh_hf_auth_status", lambda self: None
        ), patch.object(MainMenuScreen, "_refresh_panels", lambda self: None):
            async with app.run_test() as pilot:
                app.push_screen(MainMenuScreen(username="alice", version="1.0.0"))
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, MainMenuScreen)
                quick_list = screen.query_one("#quick-deploy-list", OptionList)
                self.assertEqual(quick_list.option_count, 6)
                self.assertIn("Qwen3.5 397B A17B", str(quick_list.get_option_at_index(0).prompt))
                self.assertIn("Cheap but good", str(quick_list.get_option_at_index(0).prompt))
                self.assertIn("Fast but $$$", str(quick_list.get_option_at_index(1).prompt))
                self.assertIn("GLM-5", str(quick_list.get_option_at_index(2).prompt))
                self.assertIn("RTX-PRO-6000 x4", str(quick_list.get_option_at_index(2).prompt))
                self.assertIn("B200 x2", str(quick_list.get_option_at_index(3).prompt))
                self.assertIn("Kimi K2.5", str(quick_list.get_option_at_index(4).prompt))
                self.assertIn("RTX-PRO-6000 x5", str(quick_list.get_option_at_index(4).prompt))
                self.assertIn("B200 x3", str(quick_list.get_option_at_index(5).prompt))

    async def test_tab_moves_focus_between_action_menu_and_quick_deploy(self) -> None:
        app = _TestApp()
        with patch.object(MainMenuScreen, "_refresh_modal_auth_status", lambda self: None), patch.object(
            MainMenuScreen, "_refresh_hf_auth_status", lambda self: None
        ), patch.object(MainMenuScreen, "_refresh_panels", lambda self: None):
            async with app.run_test() as pilot:
                app.push_screen(MainMenuScreen(username="alice", version="1.0.0"))
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, MainMenuScreen)
                action_list = screen.query_one("#action-list", OptionList)
                quick_list = screen.query_one("#quick-deploy-list", OptionList)

                self.assertTrue(action_list.has_focus)
                await pilot.press("tab")
                await pilot.pause()
                self.assertTrue(quick_list.has_focus)

                await pilot.press("shift+tab")
                await pilot.pause()
                self.assertTrue(action_list.has_focus)

    async def test_selecting_quick_deploy_entry_opens_detail_screen(self) -> None:
        app = _TestApp()
        with patch.object(MainMenuScreen, "_refresh_modal_auth_status", lambda self: None), patch.object(
            MainMenuScreen, "_refresh_hf_auth_status", lambda self: None
        ), patch.object(MainMenuScreen, "_refresh_panels", lambda self: None):
            async with app.run_test() as pilot:
                app.push_screen(MainMenuScreen(username="alice", version="1.0.0"))
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, MainMenuScreen)
                quick_list = screen.query_one("#quick-deploy-list", OptionList)
                option = quick_list.get_option_at_index(4)
                screen.on_option_list_option_selected(
                    SimpleNamespace(option=option, option_list=quick_list)
                )
                await pilot.pause()

                self.assertEqual(app.quick_deploy_calls, ["kimi25-rtxpro"])
                self.assertIsInstance(app.screen, QuickDeployScreen)


if __name__ == "__main__":
    unittest.main()
