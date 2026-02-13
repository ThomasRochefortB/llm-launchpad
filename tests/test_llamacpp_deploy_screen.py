from __future__ import annotations

import unittest
from types import SimpleNamespace

from textual.app import App
from textual.widgets import Input, OptionList

from llm_launchpad.tui.screens.deploy import LlamaCppDeployScreen


class _TestApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.deployed_config = None

    def begin_deploy(self, config) -> None:  # type: ignore[no-untyped-def]
        self.deployed_config = config


class LlamaCppDeployScreenTests(unittest.IsolatedAsyncioTestCase):
    async def test_preset_selection_maps_into_deploy_config(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(LlamaCppDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, LlamaCppDeployScreen)

            preset_list = screen.query_one("#preset-list", OptionList)
            option = preset_list.get_option_at_index(0)
            screen.on_option_list_option_selected(SimpleNamespace(option_list=preset_list, option=option))
            screen._do_deploy()

        self.assertIsNotNone(app.deployed_config)
        self.assertEqual(app.deployed_config.preset, "qwen3-coder-480b")

    async def test_custom_mode_and_advanced_fields_map_into_config(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(LlamaCppDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, LlamaCppDeployScreen)
            preset_list = screen.query_one("#preset-list", OptionList)
            custom = SimpleNamespace(id="preset-custom")
            screen.on_option_list_option_selected(SimpleNamespace(option_list=preset_list, option=custom))

            screen.query_one("#repo-id", Input).value = "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
            screen.query_one("#quant", Input).value = "Q4_K_M"
            screen.query_one("#revision", Input).value = "main"
            screen.query_one("#advanced-fields").remove_class("hidden")
            screen.query_one("#server-args", Input).value = "--ctx-size 65536"
            screen.query_one("#host-input", Input).value = "0.0.0.0"
            screen.query_one("#port-input", Input).value = "8088"
            screen.query_one("#n-gpu-layers", Input).value = "99"

            screen._do_deploy()

        self.assertIsNotNone(app.deployed_config)
        self.assertEqual(app.deployed_config.repo_id, "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF")
        self.assertEqual(app.deployed_config.quant, "Q4_K_M")
        self.assertEqual(app.deployed_config.revision, "main")
        self.assertEqual(app.deployed_config.server_args, "--ctx-size 65536")
        self.assertEqual(app.deployed_config.host, "0.0.0.0")
        self.assertEqual(app.deployed_config.port, 8088)
        self.assertEqual(app.deployed_config.n_gpu_layers, 99)

    async def test_instance_and_app_override_behavior(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(LlamaCppDeployScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, LlamaCppDeployScreen)

            screen.query_one("#instance-name-llama", Input).value = "Team Prod"
            screen._do_deploy()
            self.assertEqual(app.deployed_config.instance_name, "team-prod")
            self.assertEqual(app.deployed_config.app_name, "llamacpp-team-prod")

            screen.query_one("#app-name-llama", Input).value = "llamacpp-special"
            screen._do_deploy()

        self.assertEqual(app.deployed_config.app_name, "llamacpp-special")
        self.assertEqual(app.deployed_config.instance_name, "team-prod")


if __name__ == "__main__":
    unittest.main()
