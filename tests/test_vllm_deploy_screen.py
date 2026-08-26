from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from textual.app import App
from textual.widgets import Input, OptionList, Select, Static, Switch

from llm_launchpad.core.hf_models import ModelCandidate, VllmMemoryBreakdown
from llm_launchpad.core.modal_gpu import ModalGpuSpec
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import ComputeOffer, StorageSnapshot, StoredModelInfo
from llm_launchpad.tui.screens.deploy import (
    GpuTypesLoaded,
    PrimeOffersLoaded,
    VllmDeployScreen,
    VllmMemoryLoaded,
)
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
    async def test_down_from_rank_mode_last_option_moves_focus_to_model_list(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(VllmDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, VllmDeployScreen)
            screen.on_vllm_models_loaded(
                VllmModelsLoaded(
                    mode="cached",
                    models=[ModelCandidate(repo_id="Qwen/Qwen3-0.6B", downloads=10, likes=1)],
                )
            )

            rank_mode_list = screen.query_one("#vllm-rank-mode", OptionList)
            model_list = screen.query_one("#vllm-model-list", OptionList)
            rank_mode_list.focus()
            rank_mode_list.highlighted = 2
            await pilot.pause()

            await pilot.press("down")
            await pilot.pause()

            self.assertTrue(model_list.has_focus)
            highlighted = model_list.highlighted_option
            self.assertIsNotNone(highlighted)
            assert highlighted is not None
            self.assertEqual(highlighted.id, "model-0")

    async def test_up_from_model_list_first_option_moves_focus_to_rank_mode(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(VllmDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, VllmDeployScreen)
            screen.on_vllm_models_loaded(
                VllmModelsLoaded(
                    mode="cached",
                    models=[ModelCandidate(repo_id="Qwen/Qwen3-0.6B", downloads=10, likes=1)],
                )
            )

            rank_mode_list = screen.query_one("#vllm-rank-mode", OptionList)
            model_list = screen.query_one("#vllm-model-list", OptionList)
            model_list.focus()
            model_list.highlighted = 0
            await pilot.pause()

            await pilot.press("up")
            await pilot.pause()

            self.assertTrue(rank_mode_list.has_focus)
            highlighted = rank_mode_list.highlighted_option
            self.assertIsNotNone(highlighted)
            assert highlighted is not None
            self.assertEqual(highlighted.id, "rank-trending")

    async def test_down_from_model_list_last_option_moves_focus_to_model_name(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(VllmDeployScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, VllmDeployScreen)
            screen.on_vllm_models_loaded(
                VllmModelsLoaded(
                    mode="cached",
                    models=[
                        ModelCandidate(repo_id="Qwen/Qwen3-0.6B", downloads=10, likes=1),
                        ModelCandidate(repo_id="Qwen/Qwen3-4B", downloads=8, likes=1),
                    ],
                )
            )
            model_list = screen.query_one("#vllm-model-list", OptionList)
            model_name = screen.query_one("#model-name", Input)

            model_list.focus()
            model_list.highlighted = 1
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()

            self.assertTrue(model_name.has_focus)

    async def test_enter_on_model_list_commits_and_exits_to_model_name(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(VllmDeployScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, VllmDeployScreen)
            screen.on_vllm_models_loaded(
                VllmModelsLoaded(
                    mode="cached",
                    models=[
                        ModelCandidate(repo_id="Qwen/Qwen3-0.6B", downloads=10, likes=1),
                        ModelCandidate(repo_id="Qwen/Qwen3-4B", downloads=8, likes=1),
                    ],
                )
            )
            model_list = screen.query_one("#vllm-model-list", OptionList)
            model_name = screen.query_one("#model-name", Input)

            model_list.focus()
            model_list.highlighted = 1
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            self.assertTrue(model_name.has_focus)
            self.assertEqual(model_name.value, "Qwen/Qwen3-4B")

            await pilot.press("down")
            await pilot.pause()

            self.assertFalse(model_list.has_focus)
            self.assertEqual(model_name.value, "Qwen/Qwen3-4B")

    async def test_enter_on_rank_mode_moves_focus_to_model_list(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(VllmDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, VllmDeployScreen)
            rank_mode_list = screen.query_one("#vllm-rank-mode", OptionList)
            model_list = screen.query_one("#vllm-model-list", OptionList)

            rank_mode_list.focus()
            rank_mode_list.highlighted = 1
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            self.assertEqual(app.fetch_calls, ["downloads"])
            self.assertTrue(model_list.has_focus)

    async def test_down_from_model_name_moves_focus_to_gpu_type_select(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(VllmDeployScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, VllmDeployScreen)

            model_name = screen.query_one("#model-name", Input)
            gpu_type = screen.query_one("#gpu-type-vllm", Select)
            model_name.focus()
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()

            self.assertTrue(gpu_type.has_focus)

    async def test_down_from_gpu_type_moves_focus_to_gpu_count_input(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(VllmDeployScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, VllmDeployScreen)

            gpu_type = screen.query_one("#gpu-type-vllm", Select)
            gpu_count = screen.query_one("#gpu-count-vllm", Input)
            gpu_type.focus()
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()

            self.assertTrue(gpu_count.has_focus)

    async def test_gpu_type_dropdown_shows_hourly_price_labels(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(VllmDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, VllmDeployScreen)
            screen.on_gpu_types_loaded(
                GpuTypesLoaded(
                    gpu_types=[
                        ModalGpuSpec("A100-80GB", price_per_hour_usd=2.4984),
                        ModalGpuSpec("H100", price_per_hour_usd=3.9492),
                    ]
                )
            )

            gpu_type = screen.query_one("#gpu-type-vllm", Select)
            self.assertEqual(gpu_type.value, "A100-80GB")
            self.assertEqual(
                gpu_type._options[1:],
                [
                    ("A100-80GB ($2.50/hr)", "A100-80GB"),
                    ("H100 ($3.95/hr)", "H100"),
                ],
            )

    async def test_rank_mode_menu_is_focused_for_arrow_navigation(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(VllmDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, VllmDeployScreen)
            rank_mode_list = screen.query_one("#vllm-rank-mode", OptionList)

            self.assertTrue(rank_mode_list.has_focus)
            highlighted = rank_mode_list.highlighted_option
            self.assertIsNotNone(highlighted)
            assert highlighted is not None
            self.assertEqual(highlighted.id, "rank-cached")

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
            screen.query_one("#tool-call-parser", Input).value = "qwen3_xml"
            screen.query_one("#chat-template-kwargs", Input).value = '{"enable_thinking": false}'
            screen.query_one("#fast-boot", Switch).value = True
            screen.query_one("#trust-remote-code", Switch).value = True
            screen.query_one("#show-debug-logs-vllm", Switch).value = True
            # Expand advanced fields
            for w in screen.query(".vllm-advanced"):
                w.remove_class("hidden")

            screen._do_deploy()
            self.assertIsNotNone(app.deployed_config)
            self.assertTrue(app.deployed_config.fast_boot)
            self.assertTrue(app.deployed_config.trust_remote_code)
            self.assertTrue(app.deployed_config.show_debug_logs)
            self.assertEqual(app.deployed_config.reasoning_parser, "qwen3")
            self.assertEqual(app.deployed_config.tool_call_parser, "qwen3_xml")
            self.assertEqual(
                app.deployed_config.default_chat_template_kwargs,
                '{"enable_thinking": false}',
            )

    async def test_show_debug_logs_toggle_defaults_false_and_maps_to_config(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(VllmDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, VllmDeployScreen)
            debug_toggle = screen.query_one("#show-debug-logs-vllm", Switch)
            self.assertFalse(debug_toggle.value)

            for widget in screen.query(".vllm-advanced"):
                widget.remove_class("hidden")
            debug_toggle.value = True
            screen._do_deploy()

            self.assertIsNotNone(app.deployed_config)
            self.assertTrue(app.deployed_config.show_debug_logs)

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

    async def test_prime_offers_exclude_cpu_and_follow_model_memory(self) -> None:
        app = _TestApp()
        estimate = VllmMemoryBreakdown(
            total_gb=70.0,
            weights_gb=60.0,
            kv_cache_gb=5.0,
            overhead_gb=5.0,
            context_tokens=8192,
        )
        with patch(
            "llm_launchpad.tui.screens.deploy.fetch_vllm_memory_breakdown",
            return_value=estimate,
        ):
            async with app.run_test() as pilot:
                app.push_screen(VllmDeployScreen())
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, VllmDeployScreen)
                screen._provider = ComputeProvider.PRIME
                screen.query_one("#model-name", Input).value = "Qwen/Qwen3-8B"
                offers = [
                    ComputeOffer(
                        id="cpu",
                        cloud_id="cpu",
                        provider_name="provider",
                        gpu_type="CPU_NODE",
                        gpu_count=1,
                        gpu_memory_gb=512,
                        price_per_hour=0.05,
                        security="secure_cloud",
                        stock_status="Available",
                        images=("ubuntu_22_cuda_12",),
                    ),
                    ComputeOffer(
                        id="l4",
                        cloud_id="l4",
                        provider_name="provider",
                        gpu_type="L4_24GB",
                        gpu_count=3,
                        gpu_memory_gb=24,
                        price_per_hour=1.0,
                        security="secure_cloud",
                        stock_status="Available",
                        images=("ubuntu_22_cuda_12",),
                    ),
                    ComputeOffer(
                        id="h100",
                        cloud_id="h100",
                        provider_name="provider",
                        gpu_type="H100_80GB",
                        gpu_count=1,
                        gpu_memory_gb=80,
                        price_per_hour=2.0,
                        security="secure_cloud",
                        stock_status="Available",
                        images=("ubuntu_22_cuda_12",),
                    ),
                ]
                screen.on_prime_offers_loaded(PrimeOffersLoaded(offers))
                screen.on_vllm_memory_loaded(
                    VllmMemoryLoaded(
                        repo_id="Qwen/Qwen3-8B",
                        revision=None,
                        estimate=estimate,
                    )
                )
                screen.query_one("#prime-insecure-http", Switch).value = True

                selector = screen.query_one("#prime-offer-vllm", Select)
                labels = [str(label) for label, _value in selector._options[1:]]
                status = str(screen.query_one("#prime-offer-status", Static).content)
                self.assertEqual(len(labels), 1)
                self.assertIn("H100_80GB", labels[0])
                self.assertNotIn("CPU_NODE", " ".join(labels))
                self.assertIn("~73.5 GB requirement", status)
                self.assertNotIn("HTTP-only", status)
                self.assertEqual(screen.query_one("#n-gpu", Input).value, "1")

                screen._do_deploy()

        self.assertEqual(app.deployed_config.required_vram_gb, 70.0)
        self.assertEqual(app.deployed_config.n_gpu, 1)


if __name__ == "__main__":
    unittest.main()
