from __future__ import annotations

import unittest

from textual.app import App
from textual.widgets import Button, Input, Static, Switch

from llm_launchpad.tui.screens.quick_deploy import QuickDeployScreen


class _TestApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.deployed_config = None

    def begin_deploy(self, config) -> None:  # type: ignore[no-untyped-def]
        self.deployed_config = config


class QuickDeployScreenTests(unittest.IsolatedAsyncioTestCase):
    async def test_screen_renders_selected_profile_summary(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(QuickDeployScreen(profile_id="kimi25-rtxpro"))
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, QuickDeployScreen)
            summary = str(screen.query_one("#quick-deploy-profile-body", Static).content)
            title = str(screen.query_one("#quick-deploy-title", Static).content)
            self.assertIn("Kimi K2.5", title)
            self.assertIn("Kimi K2.5", summary)
            self.assertIn("RTX-PRO-6000 x5", summary)
            self.assertIn("Cheap but good", summary)
            self.assertIn("UD-Q2_K_XL", summary)

    async def test_deploy_uses_blank_override_defaults(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(QuickDeployScreen(profile_id="qwen35-397b-b200"))
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, QuickDeployScreen)
            screen._deploy()

        self.assertIsNotNone(app.deployed_config)
        self.assertEqual(app.deployed_config.instance_name, "qwen35-397b-b200")
        self.assertEqual(app.deployed_config.app_name, "llamacpp-qwen35-397b-b200")
        self.assertEqual(app.deployed_config.gpu_type, "B200")
        self.assertEqual(app.deployed_config.gpu_count, 2)
        self.assertTrue(app.deployed_config.preload)

    async def test_advanced_options_are_hidden_by_default(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(QuickDeployScreen(profile_id="qwen35-397b-b200"))
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, QuickDeployScreen)
            instance_input = screen.query_one("#quick-instance-name", Input)
            self.assertTrue(instance_input.parent is not None)
            assert instance_input.parent is not None
            self.assertTrue(instance_input.parent.has_class("hidden"))
            self.assertTrue(screen.query_one("#quick-deploy-btn", Button).has_focus)

    async def test_deploy_maps_override_fields_into_config(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(QuickDeployScreen(profile_id="qwen35-397b-rtxpro"))
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, QuickDeployScreen)
            for widget in screen.query(".quick-advanced"):
                widget.remove_class("hidden")
            screen.query_one("#quick-instance-name", Input).value = "Team Alpha"
            screen.query_one("#quick-app-name", Input).value = "llamacpp-team-alpha"
            screen.query_one("#quick-warmup", Switch).value = False
            screen.query_one("#quick-debug-logs", Switch).value = True

            screen._deploy()

        self.assertIsNotNone(app.deployed_config)
        self.assertEqual(app.deployed_config.instance_name, "team-alpha")
        self.assertEqual(app.deployed_config.app_name, "llamacpp-team-alpha")
        self.assertFalse(app.deployed_config.do_warmup)
        self.assertTrue(app.deployed_config.show_debug_logs)


if __name__ == "__main__":
    unittest.main()
