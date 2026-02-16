from __future__ import annotations

import unittest
from types import SimpleNamespace

from textual.app import App
from textual.widgets import Input, OptionList

from llm_launchpad.core.hf_models import ModelCandidate
from llm_launchpad.protocol.enums import BackendType
from llm_launchpad.protocol.models import StorageSnapshot, StoredModelInfo
from llm_launchpad.tui.screens.deploy import LlamaCppDeployScreen
from llm_launchpad.tui.workers import LlamaCppModelsLoaded, StorageLoaded


class _TestApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.fetch_calls: list[str] = []
        self.quant_fetch_calls: list[tuple[str, str | None]] = []
        self.storage_refresh_calls: list[bool] = []
        self.deployed_config = None

    def begin_fetch_llamacpp_models(self, mode: str, receiver: object) -> None:
        self.fetch_calls.append(mode)

    def begin_storage_refresh(self, receiver: object, force: bool = False) -> None:
        self.storage_refresh_calls.append(force)

    def begin_fetch_llamacpp_quants(self, repo_id: str, revision: str | None, receiver: object) -> None:
        self.quant_fetch_calls.append((repo_id, revision))

    def begin_deploy(self, config) -> None:  # type: ignore[no-untyped-def]
        self.deployed_config = config


class LlamaCppDeployScreenTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_rank_mode_highlight_matches_cached(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(LlamaCppDeployScreen())
            await pilot.pause()
            self.assertEqual(app.fetch_calls, [])
            self.assertEqual(app.storage_refresh_calls, [False])

            screen = app.screen
            assert isinstance(screen, LlamaCppDeployScreen)
            rank_mode_list = screen.query_one("#llama-rank-mode", OptionList)
            highlighted = rank_mode_list.highlighted_option
            self.assertIsNotNone(highlighted)
            assert highlighted is not None
            self.assertEqual(highlighted.id, "rank-cached")

    async def test_ranked_model_selection_prefills_repo_id(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(LlamaCppDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, LlamaCppDeployScreen)
            rank_mode_list = screen.query_one("#llama-rank-mode", OptionList)
            downloads_option = rank_mode_list.get_option_at_index(1)
            screen.on_option_list_option_selected(SimpleNamespace(option_list=rank_mode_list, option=downloads_option))
            self.assertEqual(app.fetch_calls, ["downloads"])

            screen.on_llama_cpp_models_loaded(
                LlamaCppModelsLoaded(
                    mode="downloads",
                    models=[
                        ModelCandidate(
                            repo_id="unsloth/Qwen3-Coder-Next-GGUF",
                            downloads=1000,
                            likes=10,
                            quantizations=("Q4_K_M", "Q5_K_M"),
                        ),
                        ModelCandidate(
                            repo_id="Qwen/Qwen3-Coder-Next-GGUF",
                            downloads=900,
                            likes=8,
                            quantizations=("Q5_K_M",),
                        ),
                    ],
                )
            )

            model_list = screen.query_one("#llama-model-list", OptionList)
            option = model_list.get_option_at_index(1)
            screen.on_option_list_option_selected(SimpleNamespace(option_list=model_list, option=option))

            repo_id = screen.query_one("#repo-id", Input).value
            self.assertEqual(repo_id, "Qwen/Qwen3-Coder-Next-GGUF")
            self.assertEqual(screen.query_one("#quant", Input).value, "Q5_K_M")
            self.assertEqual(app.quant_fetch_calls, [])

    async def test_cached_mode_prefills_repo_and_quant_from_storage(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(LlamaCppDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, LlamaCppDeployScreen)

            rank_mode_list = screen.query_one("#llama-rank-mode", OptionList)
            self.assertEqual(rank_mode_list.get_option_at_index(0).id, "rank-cached")

            screen.on_storage_loaded(
                StorageLoaded(
                    snapshot=StorageSnapshot(
                        llamacpp_models=[
                            StoredModelInfo(
                                backend=BackendType.LLAMACPP,
                                model_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
                                quant="Q4_K_M",
                                size_bytes=4096,
                            )
                        ],
                        vllm_models=[],
                    )
                )
            )

            cached_option = rank_mode_list.get_option_at_index(0)
            screen.on_option_list_option_selected(SimpleNamespace(option_list=rank_mode_list, option=cached_option))

            model_list = screen.query_one("#llama-model-list", OptionList)
            selected_option = model_list.get_option_at_index(0)
            screen.on_option_list_option_selected(SimpleNamespace(option_list=model_list, option=selected_option))

            self.assertEqual(screen.query_one("#repo-id", Input).value, "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF")
            self.assertEqual(screen.query_one("#quant", Input).value, "Q4_K_M")
            self.assertEqual(app.fetch_calls, [])
            self.assertEqual(app.quant_fetch_calls, [])

    async def test_repo_input_triggers_quant_lookup_when_not_in_ranked_cache(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(LlamaCppDeployScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, LlamaCppDeployScreen)
            screen.query_one("#repo-id", Input).value = "new-author/new-model-GGUF"
            await pilot.pause()
            self.assertEqual(app.quant_fetch_calls, [("new-author/new-model-GGUF", None)])

    async def test_repo_quant_and_advanced_fields_map_into_config(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(LlamaCppDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, LlamaCppDeployScreen)

            screen.query_one("#repo-id", Input).value = "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
            screen.query_one("#quant", Input).value = "Q4_K_M"
            screen.query_one("#revision", Input).value = "main"
            for widget in screen.query(".llama-advanced"):
                widget.remove_class("hidden")
            screen.query_one("#server-args", Input).value = "--ctx-size 65536"
            screen.query_one("#host-input", Input).value = "0.0.0.0"
            screen.query_one("#port-input", Input).value = "8088"
            screen.query_one("#n-gpu-layers", Input).value = "99"
            screen.query_one("#gpu-count-llama", Input).value = "3"

            screen._do_deploy()

        self.assertIsNotNone(app.deployed_config)
        self.assertEqual(app.deployed_config.repo_id, "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF")
        self.assertEqual(app.deployed_config.quant, "Q4_K_M")
        self.assertEqual(app.deployed_config.revision, "main")
        self.assertEqual(app.deployed_config.server_args, "--ctx-size 65536")
        self.assertEqual(app.deployed_config.host, "0.0.0.0")
        self.assertEqual(app.deployed_config.port, 8088)
        self.assertEqual(app.deployed_config.n_gpu_layers, 99)
        self.assertEqual(app.deployed_config.gpu_count, 3)
        self.assertEqual(app.deployed_config.gpu_type, "A100-80GB")

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
