from __future__ import annotations

import unittest
from types import SimpleNamespace

from textual.app import App
from textual.widgets import Input, OptionList

from llm_launchpad.core.hf_models import ModelCandidate
from llm_launchpad.tui.screens.deploy import VllmDeployScreen
from llm_launchpad.tui.workers import VllmModelsLoaded


class _TestApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.fetch_calls: list[str] = []
        self.deployed_config = None
        self.notifications: list[tuple[str, str]] = []

    def begin_fetch_vllm_models(self, mode: str, receiver: object) -> None:
        self.fetch_calls.append(mode)

    def begin_deploy(self, config) -> None:  # type: ignore[no-untyped-def]
        self.deployed_config = config

    def notify(self, message: object, *, severity: str = "information", **kwargs: object) -> None:
        self.notifications.append((str(message), severity))


class VllmDeployScreenTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_selection_prefills_model_input(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(VllmDeployScreen())
            await pilot.pause()
            self.assertEqual(app.fetch_calls, ["downloads"])

            screen = app.screen
            assert isinstance(screen, VllmDeployScreen)

            screen.on_vllm_models_loaded(
                VllmModelsLoaded(
                    mode="downloads",
                    models=[
                        ModelCandidate(repo_id="Qwen/Qwen2.5-7B-Instruct", downloads=1000, likes=10),
                        ModelCandidate(repo_id="meta-llama/Llama-3.1-8B-Instruct", downloads=900, likes=8),
                    ],
                )
            )
            model_list = screen.query_one("#vllm-model-list", OptionList)
            option = model_list.get_option_at_index(1)
            screen.on_option_list_option_selected(
                SimpleNamespace(
                    option_list=model_list,
                    option=option,
                )
            )
            model_name = screen.query_one("#model-name", Input).value
            self.assertEqual(model_name, "meta-llama/Llama-3.1-8B-Instruct")

    async def test_model_highlight_prefills_model_input(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(VllmDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, VllmDeployScreen)
            screen.on_vllm_models_loaded(
                VllmModelsLoaded(
                    mode="downloads",
                    models=[
                        ModelCandidate(repo_id="Qwen/Qwen2.5-7B-Instruct", downloads=1000, likes=10),
                        ModelCandidate(repo_id="meta-llama/Llama-3.1-8B-Instruct", downloads=900, likes=8),
                    ],
                )
            )
            model_list = screen.query_one("#vllm-model-list", OptionList)
            option = model_list.get_option_at_index(1)
            screen.on_option_list_option_highlighted(
                SimpleNamespace(
                    option_list=model_list,
                    option=option,
                )
            )
            model_name = screen.query_one("#model-name", Input).value
            self.assertEqual(model_name, "meta-llama/Llama-3.1-8B-Instruct")

    async def test_advanced_fields_are_mapped_into_deploy_config(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(VllmDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, VllmDeployScreen)
            screen.query_one("#model-name", Input).value = "Qwen/Qwen3-8B"
            screen.query_one("#reasoning-parser", Input).value = "qwen3"
            screen.query_one("#chat-template-kwargs", Input).value = '{"enable_thinking": false}'
            # Expand advanced fields
            for w in screen.query(".vllm-advanced"):
                w.remove_class("hidden")

            screen._do_deploy()
            self.assertIsNotNone(app.deployed_config)
            self.assertEqual(app.deployed_config.reasoning_parser, "qwen3")
            self.assertEqual(
                app.deployed_config.default_chat_template_kwargs,
                '{"enable_thinking": false}',
            )

    async def test_invalid_chat_template_kwargs_json_blocks_deploy(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(VllmDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, VllmDeployScreen)
            screen.query_one("#chat-template-kwargs", Input).value = '{"thinking": true'
            for w in screen.query(".vllm-advanced"):
                w.remove_class("hidden")

            screen._do_deploy()
            self.assertIsNone(app.deployed_config)
            self.assertTrue(app.notifications)
            self.assertEqual(app.notifications[-1][1], "error")


if __name__ == "__main__":
    unittest.main()

