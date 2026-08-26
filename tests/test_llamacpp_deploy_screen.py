from __future__ import annotations

import unittest
from types import SimpleNamespace

from textual.app import App
from textual.widgets import Button, Input, OptionList, Select, Static, Switch

from llm_launchpad.core.hf_models import ModelCandidate
from llm_launchpad.core.modal_gpu import ModalGpuSpec
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import (
    ComputeOffer,
    PrimeProviderOptions,
    StorageSnapshot,
    StoredModelInfo,
)
from llm_launchpad.tui.screens.deploy import (
    GpuTypesLoaded,
    LlamaCppDeployScreen,
    PrimeOffersLoaded,
)
from llm_launchpad.tui.workers import LlamaCppModelsLoaded, LlamaCppQuantsLoaded, StorageLoaded


class _TestApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.fetch_calls: list[str] = []
        self.quant_fetch_calls: list[tuple[str, str | None]] = []
        self.storage_refresh_calls: list[bool] = []
        self.predownload_calls: list[tuple[BackendType, str, str | None, str | None]] = []
        self.notifications: list[tuple[str, str]] = []
        self.deployed_config = None

    def begin_fetch_llamacpp_models(self, mode: str, receiver: object) -> None:
        self.fetch_calls.append(mode)

    def begin_storage_refresh(self, receiver: object, force: bool = False) -> None:
        self.storage_refresh_calls.append(force)

    def begin_fetch_llamacpp_quants(self, repo_id: str, revision: str | None, receiver: object) -> None:
        self.quant_fetch_calls.append((repo_id, revision))

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


class LlamaCppDeployScreenTests(unittest.IsolatedAsyncioTestCase):
    async def test_down_from_rank_mode_last_option_moves_focus_to_model_list(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(LlamaCppDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, LlamaCppDeployScreen)
            screen.on_llama_cpp_models_loaded(
                LlamaCppModelsLoaded(
                    mode="cached",
                    models=[
                        ModelCandidate(
                            repo_id="Qwen/Qwen3-Coder-Next-GGUF",
                            downloads=1000,
                            likes=10,
                            quantizations=("Q4_K_M",),
                        )
                    ],
                )
            )

            rank_mode_list = screen.query_one("#llama-rank-mode", OptionList)
            model_list = screen.query_one("#llama-model-list", OptionList)
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
            app.push_screen(LlamaCppDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, LlamaCppDeployScreen)
            screen.on_llama_cpp_models_loaded(
                LlamaCppModelsLoaded(
                    mode="cached",
                    models=[
                        ModelCandidate(
                            repo_id="Qwen/Qwen3-Coder-Next-GGUF",
                            downloads=1000,
                            likes=10,
                            quantizations=("Q4_K_M",),
                        )
                    ],
                )
            )

            rank_mode_list = screen.query_one("#llama-rank-mode", OptionList)
            model_list = screen.query_one("#llama-model-list", OptionList)
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

    async def test_down_from_quant_list_last_option_moves_focus_to_provider_select(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(LlamaCppDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, LlamaCppDeployScreen)
            screen.query_one("#repo-id", Input).value = "Qwen/Qwen3-Coder-Next-GGUF"
            await pilot.pause()
            screen.on_llama_cpp_quants_loaded(
                LlamaCppQuantsLoaded(
                    repo_id="Qwen/Qwen3-Coder-Next-GGUF",
                    revision=None,
                    quantizations=["Q4_K_M", "Q8_0"],
                    vram_gb_by_quant={},
                )
            )
            quant_list = screen.query_one("#llama-quant-list", OptionList)
            provider = screen.query_one("#provider-llama", Select)

            quant_list.focus()
            quant_list.highlighted = 1
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()

            self.assertTrue(provider.has_focus)

    async def test_enter_on_model_list_commits_and_exits_to_repo_id(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(LlamaCppDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, LlamaCppDeployScreen)
            screen.on_llama_cpp_models_loaded(
                LlamaCppModelsLoaded(
                    mode="cached",
                    models=[
                        ModelCandidate(
                            repo_id="Qwen/Qwen3-Coder-Next-GGUF",
                            downloads=1000,
                            likes=10,
                            quantizations=("Q4_K_M",),
                        ),
                        ModelCandidate(
                            repo_id="unsloth/Qwen3-Coder-Next-GGUF",
                            downloads=900,
                            likes=8,
                            quantizations=("Q5_K_M",),
                        ),
                    ],
                )
            )
            model_list = screen.query_one("#llama-model-list", OptionList)
            repo_id = screen.query_one("#repo-id", Input)

            model_list.focus()
            model_list.highlighted = 1
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            self.assertTrue(repo_id.has_focus)
            self.assertEqual(repo_id.value, "unsloth/Qwen3-Coder-Next-GGUF")

            await pilot.press("down")
            await pilot.pause()

            self.assertFalse(model_list.has_focus)
            self.assertEqual(repo_id.value, "unsloth/Qwen3-Coder-Next-GGUF")

    async def test_enter_on_rank_mode_moves_focus_to_model_list(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(LlamaCppDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, LlamaCppDeployScreen)
            rank_mode_list = screen.query_one("#llama-rank-mode", OptionList)
            model_list = screen.query_one("#llama-model-list", OptionList)

            rank_mode_list.focus()
            rank_mode_list.highlighted = 1
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            self.assertEqual(app.fetch_calls, ["downloads"])
            self.assertTrue(model_list.has_focus)

    async def test_enter_on_quant_list_commits_and_exits_to_provider(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(LlamaCppDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, LlamaCppDeployScreen)
            screen.query_one("#repo-id", Input).value = "Qwen/Qwen3-Coder-Next-GGUF"
            await pilot.pause()
            screen.on_llama_cpp_quants_loaded(
                LlamaCppQuantsLoaded(
                    repo_id="Qwen/Qwen3-Coder-Next-GGUF",
                    revision=None,
                    quantizations=["Q4_K_M", "Q8_0"],
                    vram_gb_by_quant={},
                )
            )
            quant_list = screen.query_one("#llama-quant-list", OptionList)
            quant_input = screen.query_one("#quant", Input)
            provider = screen.query_one("#provider-llama", Select)

            quant_list.focus()
            quant_list.highlighted = 1
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            self.assertEqual(quant_input.value, "Q8_0")
            self.assertTrue(provider.has_focus)

    async def test_gpu_type_dropdown_shows_hourly_price_labels(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(LlamaCppDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, LlamaCppDeployScreen)
            screen.on_gpu_types_loaded(
                GpuTypesLoaded(
                    gpu_types=[
                        ModalGpuSpec("A100-80GB", price_per_hour_usd=2.4984),
                        ModalGpuSpec("H100", price_per_hour_usd=3.9492),
                    ]
                )
            )

            gpu_type = screen.query_one("#gpu-type-llama", Select)
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
            app.push_screen(LlamaCppDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, LlamaCppDeployScreen)
            rank_mode_list = screen.query_one("#llama-rank-mode", OptionList)

            self.assertTrue(rank_mode_list.has_focus)
            highlighted = rank_mode_list.highlighted_option
            self.assertIsNotNone(highlighted)
            assert highlighted is not None
            self.assertEqual(highlighted.id, "rank-cached")

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
            self.assertEqual(app.quant_fetch_calls, [("Qwen/Qwen3-Coder-Next-GGUF", None)])

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
            self.assertEqual(app.quant_fetch_calls, [("Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", None)])

    async def test_cached_mode_keeps_quant_list_filtered_after_metadata_lookup(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(LlamaCppDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, LlamaCppDeployScreen)
            screen.on_storage_loaded(
                StorageLoaded(
                    snapshot=StorageSnapshot(
                        llamacpp_models=[
                            StoredModelInfo(
                                backend=BackendType.LLAMACPP,
                                model_id="unsloth/GLM-5-GGUF",
                                quant="Q4_K_M",
                                size_bytes=4096,
                            )
                        ],
                        vllm_models=[],
                    )
                )
            )

            model_list = screen.query_one("#llama-model-list", OptionList)
            selected_option = model_list.get_option_at_index(0)
            screen.on_option_list_option_selected(SimpleNamespace(option_list=model_list, option=selected_option))

            screen.on_llama_cpp_quants_loaded(
                LlamaCppQuantsLoaded(
                    repo_id="unsloth/GLM-5-GGUF",
                    revision=None,
                    quantizations=["Q3_K_S", "Q4_K_M", "Q5_K_M"],
                    vram_gb_by_quant={"Q3_K_S": 326.3, "Q4_K_M": 360.0, "Q5_K_M": 410.0},
                )
            )

            quant_list = screen.query_one("#llama-quant-list", OptionList)
            self.assertEqual(quant_list.option_count, 1)
            self.assertIn("Q4_K_M", quant_list.get_option_at_index(0).prompt)

            screen._lookup_quantizations_for_current_repo()
            self.assertEqual(quant_list.option_count, 1)
            self.assertIn("Q4_K_M", quant_list.get_option_at_index(0).prompt)

    async def test_quants_loaded_shows_vram_in_option_list_and_status(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(LlamaCppDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, LlamaCppDeployScreen)
            screen.query_one("#repo-id", Input).value = "Qwen/Qwen3-Coder-Next-GGUF"
            await pilot.pause()

            screen.on_llama_cpp_quants_loaded(
                LlamaCppQuantsLoaded(
                    repo_id="Qwen/Qwen3-Coder-Next-GGUF",
                    revision=None,
                    quantizations=["Q4_K_M", "Q8_0"],
                    vram_gb_by_quant={"Q4_K_M": 4.66},
                )
            )

            quant_list = screen.query_one("#llama-quant-list", OptionList)
            first = quant_list.get_option_at_index(0)
            second = quant_list.get_option_at_index(1)
            self.assertIn("Q4_K_M (~4.7 GB)", first.prompt)
            self.assertEqual(second.prompt.strip(), "Q8_0")

            quant_status = str(screen.query_one("#llama-quant-status", Static).content)
            self.assertIn("Quantizations:", quant_status)

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
            screen.query_one("#warmup", Switch).value = False
            screen.query_one("#llama-image-no-cache", Switch).value = True
            screen.query_one("#show-debug-logs-llama", Switch).value = True
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
        self.assertFalse(app.deployed_config.preload)
        self.assertTrue(app.deployed_config.do_deploy)
        self.assertFalse(app.deployed_config.do_warmup)
        self.assertTrue(app.deployed_config.llamacpp_image_no_cache)
        self.assertTrue(app.deployed_config.show_debug_logs)
        self.assertEqual(app.deployed_config.gpu_count, 3)
        self.assertEqual(app.deployed_config.gpu_type, "A100-80GB")

    async def test_prime_offer_maps_into_llamacpp_deployment(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(LlamaCppDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, LlamaCppDeployScreen)
            offer = ComputeOffer(
                id="abc123",
                cloud_id="cloud-h100",
                provider_name="provider",
                gpu_type="H100_80GB",
                gpu_count=1,
                price_per_hour=1.75,
                security="secure_cloud",
                stock_status="Available",
                images=("ubuntu_22_cuda_12",),
            )
            screen._provider = ComputeProvider.PRIME
            screen._sync_prime_visibility()
            screen.on_prime_offers_loaded(PrimeOffersLoaded([offer]))
            screen._selected_prime_offer_id = offer.id
            screen.query_one("#repo-id", Input).value = "org/Model-GGUF"
            screen.query_one("#quant", Input).value = "Q4_K_M"
            screen._do_deploy()

        self.assertIsNotNone(app.deployed_config)
        config = app.deployed_config
        self.assertEqual(config.provider, ComputeProvider.PRIME)
        self.assertEqual(config.backend, BackendType.LLAMACPP)
        self.assertEqual(config.gpu_type, "H100_80GB")
        self.assertEqual(config.gpu_count, 1)
        self.assertEqual(config.app_name, "llp-prime-llamacpp-org-model-gguf")
        self.assertEqual(
            config.provider_options,
            PrimeProviderOptions(
                offer_id="abc123",
            ),
        )

    async def test_prime_disk_submits_through_real_form_buttons(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(LlamaCppDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, LlamaCppDeployScreen)
            offer = ComputeOffer(
                id="disk-offer",
                cloud_id="cloud-rtx4000ada",
                provider_name="provider",
                gpu_type="RTX4000Ada_20GB",
                gpu_count=1,
                price_per_hour=0.30,
                security="secure_cloud",
                stock_status="Available",
                images=("ubuntu_22_cuda_12",),
            )
            screen.query_one("#repo-id", Input).value = (
                "bartowski/Qwen2.5-0.5B-Instruct-GGUF"
            )
            screen.query_one("#quant", Input).value = "Q4_K_M"
            screen.on_prime_offers_loaded(PrimeOffersLoaded([offer]))
            screen.query_one("#provider-llama", Select).value = "prime"
            await pilot.pause()
            screen.query_one("#toggle-advanced-llama", Button).press()
            await pilot.pause()
            screen.query_one("#prime-disk-id-llama", Input).value = "disk-1"
            screen.query_one("#prime-insecure-http-llama", Switch).value = True
            await pilot.pause()
            screen.query_one("#deploy-btn", Button).press()
            await pilot.pause()

        self.assertIsNotNone(app.deployed_config)
        config = app.deployed_config
        self.assertEqual(config.provider, ComputeProvider.PRIME)
        self.assertEqual(
            config.provider_options,
            PrimeProviderOptions(
                offer_id="disk-offer",
                disk_id="disk-1",
                allow_insecure_http=True,
            ),
        )

    async def test_prime_offers_exclude_cpu_and_follow_quant_vram(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(LlamaCppDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, LlamaCppDeployScreen)
            screen._provider = ComputeProvider.PRIME
            screen.query_one("#repo-id", Input).value = "org/Model-GGUF"
            screen.query_one("#quant", Input).value = "Q4_K_M"
            screen._repo_to_quant_vram = {
                "org/model-gguf": {"Q4_K_M": 70.0}
            }
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
            screen.query_one("#prime-insecure-http-llama", Switch).value = True

            selector = screen.query_one("#prime-offer-llama", Select)
            labels = [str(label) for label, _value in selector._options[1:]]
            status = str(
                screen.query_one("#prime-offer-status-llama", Static).content
            )
            self.assertEqual(len(labels), 1)
            self.assertIn("H100_80GB", labels[0])
            self.assertNotIn("CPU_NODE", " ".join(labels))
            self.assertIn("~73.5 GB requirement", status)
            self.assertNotIn("HTTP-only", status)

            screen._do_deploy()

        self.assertEqual(app.deployed_config.required_vram_gb, 70.0)

    async def test_warmup_toggle_is_hidden_by_default_and_enabled(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(LlamaCppDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, LlamaCppDeployScreen)
            warmup_toggle = screen.query_one("#warmup", Switch)

            self.assertIsNotNone(warmup_toggle.parent)
            assert warmup_toggle.parent is not None
            self.assertTrue(warmup_toggle.parent.has_class("hidden"))
            self.assertTrue(warmup_toggle.value)

    async def test_show_debug_logs_toggle_defaults_false_and_maps_to_config(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(LlamaCppDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, LlamaCppDeployScreen)
            debug_toggle = screen.query_one("#show-debug-logs-llama", Switch)
            self.assertFalse(debug_toggle.value)

            for widget in screen.query(".llama-advanced"):
                widget.remove_class("hidden")
            debug_toggle.value = True
            screen._do_deploy()

        self.assertIsNotNone(app.deployed_config)
        self.assertTrue(app.deployed_config.show_debug_logs)

    async def test_predownload_uses_highlighted_model_from_rank_list(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(LlamaCppDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, LlamaCppDeployScreen)
            screen.on_llama_cpp_models_loaded(
                LlamaCppModelsLoaded(
                    mode="cached",
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
                            quantizations=("Q6_K",),
                        ),
                    ],
                )
            )
            model_list = screen.query_one("#llama-model-list", OptionList)
            model_list.highlighted = 1
            await pilot.pause()
            screen.query_one("#quant", Input).value = "Q6_K"
            screen.query_one("#revision", Input).value = "main"

            screen.action_predownload_highlighted()

            self.assertEqual(
                app.predownload_calls,
                [(BackendType.LLAMACPP, "Qwen/Qwen3-Coder-Next-GGUF", "Q6_K", "main")],
            )

    async def test_predownload_requires_highlighted_model(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(LlamaCppDeployScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, LlamaCppDeployScreen)
            screen.action_predownload_highlighted()

            self.assertEqual(app.predownload_calls, [])
            self.assertEqual(app.notifications[-1], ("Highlight a model in Model ranking first.", "warning"))

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
