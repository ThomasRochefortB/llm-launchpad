from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from textual.app import App
from textual.widgets import Input, OptionList, Static, Switch

from llm_launchpad.core.hf_models import ModelCandidate, VllmMemoryBreakdown
from llm_launchpad.protocol.enums import BackendType
from llm_launchpad.protocol.models import StorageSnapshot, StoredModelInfo
from llm_launchpad.tui.screens.deploy import VllmDeployScreen
from llm_launchpad.tui.workers import StorageLoaded, VllmModelsLoaded


class _TestApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.fetch_calls: list[str] = []
        self.storage_refresh_calls: list[bool] = []
        self.predownload_calls: list[tuple[BackendType, str, str | None, str | None]] = []
        self.deployed_config = None
        self.notifications: list[tuple[str, str]] = []

    def begin_fetch_vllm_models(self, mode: str, receiver: object) -> None:
        self.fetch_calls.append(mode)

    def begin_storage_refresh(self, receiver: object, force: bool = False) -> None:
        self.storage_refresh_calls.append(force)

    def begin_deploy(self, config) -> None:  # type: ignore[no-untyped-def]
        self.deployed_config = config

    def begin_storage_predownload(
        self,
        backend: BackendType,
        model_id: str,
        quant: str | None = None,
        revision: str | None = None,
    ) -> None:
        self.predownload_calls.append((backend, model_id, quant, revision))

    def notify(self, message: object, *, severity: str = "information", **kwargs: object) -> None:
        self.notifications.append((str(message), severity))


class VllmDeployScreenTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_rank_mode_highlight_matches_cached(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(VllmDeployScreen())
            await pilot.pause()
            self.assertEqual(app.fetch_calls, [])
            self.assertEqual(app.storage_refresh_calls, [False])

            screen = app.screen
            assert isinstance(screen, VllmDeployScreen)
            rank_mode_list = screen.query_one("#vllm-rank-mode", OptionList)
            highlighted = rank_mode_list.highlighted_option
            self.assertIsNotNone(highlighted)
            assert highlighted is not None
            self.assertEqual(highlighted.id, "rank-cached")

    async def test_served_model_alias_defaults_to_model_suffix(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(VllmDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, VllmDeployScreen)
            self.assertEqual(screen.query_one("#model-name", Input).value, "")
            alias = screen.query_one("#served-model-name", Input).value
            self.assertEqual(alias, "llm")

            screen.query_one("#model-name", Input).value = "Qwen/Qwen3-0.6B"
            await pilot.pause()
            alias = screen.query_one("#served-model-name", Input).value
            self.assertEqual(alias, "Qwen3-0.6B")

    async def test_served_model_alias_respects_manual_override(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(VllmDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, VllmDeployScreen)
            screen.query_one("#served-model-name", Input).value = "my-alias"
            await pilot.pause()
            screen.query_one("#model-name", Input).value = "meta-llama/Llama-3.1-8B-Instruct"
            await pilot.pause()
            alias = screen.query_one("#served-model-name", Input).value
            self.assertEqual(alias, "my-alias")

    async def test_deploy_uses_default_alias_when_blank(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(VllmDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, VllmDeployScreen)
            screen.query_one("#model-name", Input).value = "Qwen/Qwen3-0.6B"
            screen.query_one("#served-model-name", Input).value = ""

            screen._do_deploy()
            self.assertIsNotNone(app.deployed_config)
            self.assertEqual(app.deployed_config.served_model_name, "Qwen3-0.6B")
            self.assertEqual(app.deployed_config.gpu_type, "A100-80GB")
            self.assertEqual(app.deployed_config.gpu_count, 1)

    async def test_enforce_eager_defaults_to_false(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(VllmDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, VllmDeployScreen)
            self.assertFalse(screen.query_one("#fast-boot", Switch).value)
            self.assertFalse(screen.query_one("#trust-remote-code", Switch).value)
            screen._do_deploy()
            self.assertIsNotNone(app.deployed_config)
            self.assertFalse(app.deployed_config.fast_boot)
            self.assertFalse(app.deployed_config.trust_remote_code)

    async def test_smoke_only_defaults_to_deploy(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(VllmDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, VllmDeployScreen)
            screen._do_deploy()
            self.assertIsNotNone(app.deployed_config)
            self.assertTrue(app.deployed_config.do_deploy)
            self.assertFalse(app.deployed_config.run_smoke)

    async def test_smoke_only_toggle_disables_deploy_and_warmup(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(VllmDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, VllmDeployScreen)
            for w in screen.query(".vllm-advanced"):
                w.remove_class("hidden")
            screen.query_one("#smoke-only-vllm", Switch).value = True
            screen.query_one("#warmup-vllm", Switch).value = True
            screen._do_deploy()
            self.assertIsNotNone(app.deployed_config)
            self.assertFalse(app.deployed_config.do_deploy)
            self.assertTrue(app.deployed_config.run_smoke)
            self.assertFalse(app.deployed_config.do_warmup)

    async def test_model_selection_prefills_model_input(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(VllmDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, VllmDeployScreen)
            rank_mode_list = screen.query_one("#vllm-rank-mode", OptionList)
            downloads_option = rank_mode_list.get_option_at_index(1)
            screen.on_option_list_option_selected(SimpleNamespace(option_list=rank_mode_list, option=downloads_option))
            self.assertEqual(app.fetch_calls, ["downloads"])

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

    async def test_cached_mode_prefills_model_name_from_storage(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(VllmDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, VllmDeployScreen)

            rank_mode_list = screen.query_one("#vllm-rank-mode", OptionList)
            self.assertEqual(rank_mode_list.get_option_at_index(0).id, "rank-cached")

            screen.on_storage_loaded(
                StorageLoaded(
                    snapshot=StorageSnapshot(
                        llamacpp_models=[],
                        vllm_models=[
                            StoredModelInfo(
                                backend=BackendType.VLLM,
                                model_id="Qwen/Qwen3-4B-Thinking-2507-FP8",
                                size_bytes=8192,
                            )
                        ],
                    )
                )
            )

            cached_option = rank_mode_list.get_option_at_index(0)
            screen.on_option_list_option_selected(SimpleNamespace(option_list=rank_mode_list, option=cached_option))

            model_list = screen.query_one("#vllm-model-list", OptionList)
            selected_option = model_list.get_option_at_index(0)
            screen.on_option_list_option_selected(SimpleNamespace(option_list=model_list, option=selected_option))

            model_name = screen.query_one("#model-name", Input).value
            self.assertEqual(model_name, "Qwen/Qwen3-4B-Thinking-2507-FP8")
            self.assertEqual(app.fetch_calls, [])

    async def test_model_highlight_prefills_model_input(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(VllmDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, VllmDeployScreen)
            rank_mode_list = screen.query_one("#vllm-rank-mode", OptionList)
            downloads_option = rank_mode_list.get_option_at_index(1)
            screen.on_option_list_option_selected(SimpleNamespace(option_list=rank_mode_list, option=downloads_option))
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
            screen.query_one("#fast-boot", Switch).value = True
            screen.query_one("#trust-remote-code", Switch).value = True
            # Expand advanced fields
            for w in screen.query(".vllm-advanced"):
                w.remove_class("hidden")

            screen._do_deploy()
            self.assertIsNotNone(app.deployed_config)
            self.assertTrue(app.deployed_config.fast_boot)
            self.assertTrue(app.deployed_config.trust_remote_code)
            self.assertEqual(app.deployed_config.reasoning_parser, "qwen3")
            self.assertEqual(
                app.deployed_config.default_chat_template_kwargs,
                '{"enable_thinking": false}',
            )

    async def test_model_revision_is_hidden_with_advanced_fields_and_maps_to_config(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(VllmDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, VllmDeployScreen)
            revision_input = screen.query_one("#model-revision", Input)
            parent = revision_input.parent
            self.assertIsNotNone(parent)
            assert parent is not None
            self.assertTrue(parent.has_class("vllm-advanced"))
            self.assertTrue(parent.has_class("hidden"))

            for w in screen.query(".vllm-advanced"):
                w.remove_class("hidden")

            revision_input.value = "main"
            screen._do_deploy()

            self.assertIsNotNone(app.deployed_config)
            self.assertEqual(app.deployed_config.model_revision, "main")

    async def test_deployment_gpu_shape_is_independent_from_tensor_parallel(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(VllmDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, VllmDeployScreen)
            screen.query_one("#gpu-count-vllm", Input).value = "2"
            screen.query_one("#n-gpu", Input).value = "4"
            screen._do_deploy()

            self.assertIsNotNone(app.deployed_config)
            self.assertEqual(app.deployed_config.gpu_count, 2)
            self.assertEqual(app.deployed_config.n_gpu, 4)

    async def test_predownload_uses_highlighted_model_from_rank_list(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(VllmDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, VllmDeployScreen)
            screen.on_storage_loaded(
                StorageLoaded(
                    snapshot=StorageSnapshot(
                        llamacpp_models=[],
                        vllm_models=[
                            StoredModelInfo(
                                backend=BackendType.VLLM,
                                model_id="Qwen/Qwen3-4B-Thinking-2507-FP8",
                                size_bytes=8192,
                            )
                        ],
                    )
                )
            )
            model_list = screen.query_one("#vllm-model-list", OptionList)
            model_list.highlighted = 0
            await pilot.pause()
            screen.query_one("#model-revision", Input).value = "main"

            screen.action_predownload_highlighted()

            self.assertEqual(
                app.predownload_calls,
                [(BackendType.VLLM, "Qwen/Qwen3-4B-Thinking-2507-FP8", None, "main")],
            )

    async def test_predownload_requires_highlighted_model(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(VllmDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, VllmDeployScreen)
            screen.action_predownload_highlighted()

            self.assertEqual(app.predownload_calls, [])
            self.assertEqual(app.notifications[-1], ("Highlight a model in Model ranking first.", "warning"))

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

    async def test_vllm_memory_status_updates_from_estimate(self) -> None:
        app = _TestApp()
        estimate = VllmMemoryBreakdown(
            total_gb=120.0,
            weights_gb=90.0,
            kv_cache_gb=20.0,
            overhead_gb=10.0,
            context_tokens=8192,
        )
        with patch("llm_launchpad.tui.screens.deploy.fetch_vllm_memory_breakdown", return_value=estimate):
            async with app.run_test() as pilot:
                app.push_screen(VllmDeployScreen())
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, VllmDeployScreen)
                screen.query_one("#model-name", Input).value = "Qwen/Qwen3-8B"
                screen.query_one("#n-gpu", Input).value = "2"
                await pilot.pause()

                text = str(screen.query_one("#vllm-vram-status", Static).content)
                self.assertIn("Estimated VRAM", text)
                self.assertIn("~60.0 GB/GPU @ TP=2", text)

    async def test_vllm_memory_status_handles_unavailable_estimate(self) -> None:
        app = _TestApp()
        with patch("llm_launchpad.tui.screens.deploy.fetch_vllm_memory_breakdown", return_value=None):
            async with app.run_test() as pilot:
                app.push_screen(VllmDeployScreen())
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, VllmDeployScreen)
                screen.query_one("#model-name", Input).value = "custom/model-no-metadata"
                await pilot.pause()

                text = str(screen.query_one("#vllm-vram-status", Static).content)
                self.assertIn("N/A", text)


if __name__ == "__main__":
    unittest.main()
