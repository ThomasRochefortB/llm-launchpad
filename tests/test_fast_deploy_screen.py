from __future__ import annotations

from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

from textual.app import App
from textual.widgets import OptionList, Static

from llm_launchpad.core import quick_deploy
from llm_launchpad.core.compute_availability import aggregate_compute_availability
from llm_launchpad.core.modal_gpu import ModalGpuSpec
from llm_launchpad.core.quick_deploy import (
    QuickDeployCatalogInfo,
    QuickDeployModel,
    QuickDeployProfile,
    list_quick_deploy_recipes,
)
from llm_launchpad.core.prime_backend import preferred_prime_offer_image
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import ComputeOffer, InferencePlan
from llm_launchpad.tui.app import TuiApp
from llm_launchpad.tui.screens.fast_deploy import (
    FastDeployAvailabilityFailed,
    FastDeployAvailabilityLoaded,
    FastDeployScreen,
    _gpu_filter_options,
    infra_rows_for_model,
    representative_profiles_for_model,
)
from llm_launchpad.tui.screens.main_menu import MainMenuScreen


def _profile(
    profile_id: str,
    *,
    required_vram_gb: float,
    quant: str = "Q4_K_M",
    gpu_type: str = "H100",
    display_name: str | None = None,
    repo_id: str | None = None,
) -> QuickDeployProfile:
    return QuickDeployProfile(
        id=profile_id,
        display_name=display_name or profile_id.replace("-", " ").title(),
        repo_id=repo_id or f"acme/{profile_id}-GGUF",
        quant=quant,
        gpu_type=gpu_type,
        gpu_count=1,
        profile_label="Test",
        approx_cost_per_hour_usd=4.0,
        max_context_tokens=32768,
        instance_slug_hint=profile_id,
        summary=f"Summary for {profile_id}.",
        server_args=(),
        required_vram_gb=required_vram_gb,
        backend=BackendType.LLAMACPP,
    )


def _model(profiles: tuple[QuickDeployProfile, ...]) -> QuickDeployModel:
    return QuickDeployModel(
        id=profiles[0].id,
        display_name=profiles[0].display_name,
        recipes=list_quick_deploy_recipes(profiles),
        profiles=profiles,
        max_context_tokens=max(row.max_context_tokens for row in profiles),
        quality_score=12.5,
    )


def _shared_recipe_model() -> QuickDeployModel:
    profiles = tuple(
        _profile(
            profile_id,
            required_vram_gb=100.0,
            display_name="Mega Model",
            repo_id="acme/Mega-GGUF",
            gpu_type=gpu_type,
            quant=quant,
        )
        for profile_id, gpu_type, quant in (
            ("mega-q4-l4", "L4", "Q4_K_M"),
            ("mega-q4-rtx", "RTX-PRO-6000", "Q4_K_M"),
            ("mega-q4-b200", "B200", "Q4_K_M"),
            ("mega-q2-l4", "L4", "Q2_K_M"),
        )
    )
    return QuickDeployModel(
        id="mega-model",
        display_name="Mega Model",
        recipes=list_quick_deploy_recipes(profiles),
        profiles=profiles,
        max_context_tokens=32768,
        quality_score=12.5,
    )


class _TestApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.fast_deploy_calls = 0
        self.quick_deploy_calls: list[tuple[InferencePlan, tuple[InferencePlan, ...]]] = []

    def action_push_deploy(self) -> None:
        self.fast_deploy_calls += 1
        self.push_screen(FastDeployScreen())

    def action_push_custom_deploy(self) -> None:
        pass

    def action_push_manage(self) -> None:
        pass

    def action_push_settings(self) -> None:
        pass

    def action_push_storage(self) -> None:
        pass

    def push_quick_deploy(
        self,
        profile: str | QuickDeployProfile | InferencePlan,
        *,
        alternative_plans: tuple[InferencePlan, ...] | None = None,
    ) -> None:
        if not isinstance(profile, InferencePlan):
            return
        self.quick_deploy_calls.append((profile, alternative_plans or ()))


class _StyledApp(_TestApp):
    CSS_PATH = TuiApp.CSS_PATH


class FastDeployScreenTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._availability_patch = patch(
            "llm_launchpad.tui.screens.fast_deploy.load_compute_availability",
            return_value=aggregate_compute_availability(),
        )
        self._availability_patch.start()

    def tearDown(self) -> None:
        self._availability_patch.stop()

    async def test_model_list_renders_catalog_models(self) -> None:
        model = _model((_profile("test-model", required_vram_gb=100.0),))
        app = _TestApp()
        with patch(
            "llm_launchpad.tui.screens.fast_deploy.list_quick_deploy_models",
            return_value=(model,),
        ):
            async with app.run_test() as pilot:
                app.push_screen(FastDeployScreen())
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, FastDeployScreen)
                option_list = screen.query_one("#fast-deploy-list", OptionList)
                self.assertEqual(option_list.option_count, 1)
                prompt = str(option_list.get_option_at_index(0).prompt)
                self.assertIn("Test Model", prompt)
                self.assertIn("AAI 12.5", prompt)
                self.assertIn("from ~$4.00/hr", prompt)

    async def test_infra_rows_sorted_cheapest_first_and_routes_selection(self) -> None:
        model = _model((_profile("fits", required_vram_gb=100.0),))
        snapshot = aggregate_compute_availability(
            modal_catalog=[
                ModalGpuSpec("H100", price_per_hour_usd=4.0),
                ModalGpuSpec("L40S", price_per_hour_usd=2.0),
            ]
        )
        app = _TestApp()
        with patch(
            "llm_launchpad.tui.screens.fast_deploy.list_quick_deploy_models",
            return_value=(model,),
        ):
            async with app.run_test() as pilot:
                app.push_screen(FastDeployScreen())
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, FastDeployScreen)
                screen._selected_model = model
                screen.on_fast_deploy_availability_loaded(
                    FastDeployAvailabilityLoaded(snapshot)
                )
                await pilot.pause()

                option_list = screen.query_one("#fast-deploy-list", OptionList)
                self.assertEqual(option_list.option_count, 2)
                first = str(option_list.get_option_at_index(0).prompt)
                second = str(option_list.get_option_at_index(1).prompt)
                self.assertIn("L40S", first)
                self.assertIn("~$6.00/hr", first)
                self.assertIn("H100", second)
                self.assertIn("~$8.00/hr", second)

                await pilot.press("enter")
                await pilot.pause()

        self.assertEqual(len(app.quick_deploy_calls), 1)
        selected, alternatives = app.quick_deploy_calls[0]
        self.assertEqual(selected.quote.gpu_count, 3)
        self.assertEqual(selected.quote.gpu_type, "L40S")
        self.assertEqual(len(alternatives), 1)
        self.assertEqual(alternatives[0].quote.id, selected.quote.id)

    async def test_availability_failure_renders_catalog_estimates(self) -> None:
        model = _model((_profile("too-large", required_vram_gb=200.0),))
        app = _TestApp()
        with patch(
            "llm_launchpad.tui.screens.fast_deploy.list_quick_deploy_models",
            return_value=(model,),
        ):
            async with app.run_test() as pilot:
                app.push_screen(FastDeployScreen())
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, FastDeployScreen)
                screen._selected_model = model
                screen.on_fast_deploy_availability_failed(
                    FastDeployAvailabilityFailed("provider down")
                )
                await pilot.pause()

                option_list = screen.query_one("#fast-deploy-list", OptionList)
                self.assertEqual(option_list.option_count, 1)
                prompt = str(option_list.get_option_at_index(0).prompt)
                self.assertIn("H100x1", prompt)
                detail = str(screen.query_one("#fast-deploy-detail", Static).content)
                self.assertIn("catalog estimate", detail)
                status = str(screen.query_one("#fast-deploy-status", Static).renderable)
                self.assertIn("availability unavailable", status)
                self.assertIn("provider down", status)

                await pilot.press("enter")
                await pilot.pause()

        self.assertEqual(len(app.quick_deploy_calls), 1)
        selected, alternatives = app.quick_deploy_calls[0]
        self.assertEqual(alternatives, (selected,))

    async def test_prime_only_no_fit_does_not_route_catalog_fallback_to_modal(self) -> None:
        model = _model((_profile("too-large", required_vram_gb=200.0),))
        snapshot = SimpleNamespace(
            configurations=(),
            errors=(),
            providers=(ComputeProvider.PRIME,),
        )
        app = _TestApp()
        with patch(
            "llm_launchpad.tui.screens.fast_deploy.list_quick_deploy_models",
            return_value=(model,),
        ):
            async with app.run_test() as pilot:
                app.push_screen(FastDeployScreen())
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, FastDeployScreen)
                screen._selected_model = model
                screen.on_fast_deploy_availability_loaded(
                    FastDeployAvailabilityLoaded(snapshot)
                )
                await pilot.pause()

                await pilot.press("enter")
                await pilot.pause()

        self.assertEqual(app.quick_deploy_calls, [])

    async def test_no_fit_snapshot_falls_back_to_catalog_estimates(self) -> None:
        model = _model((_profile("too-large", required_vram_gb=200.0),))
        snapshot = aggregate_compute_availability(
            modal_catalog=[ModalGpuSpec("T4", price_per_hour_usd=0.5)]
        )
        app = _TestApp()
        with patch(
            "llm_launchpad.tui.screens.fast_deploy.list_quick_deploy_models",
            return_value=(model,),
        ):
            async with app.run_test() as pilot:
                app.push_screen(FastDeployScreen())
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, FastDeployScreen)
                screen._selected_model = model
                screen.on_fast_deploy_availability_loaded(
                    FastDeployAvailabilityLoaded(snapshot)
                )
                await pilot.pause()

                option_list = screen.query_one("#fast-deploy-list", OptionList)
                self.assertEqual(option_list.option_count, 1)
                status = str(screen.query_one("#fast-deploy-status", Static).renderable)
                self.assertIn("No connected placement fits", status)

    async def test_shared_recipe_profiles_do_not_duplicate_option_ids(self) -> None:
        model = _shared_recipe_model()
        snapshot = aggregate_compute_availability(
            modal_catalog=[
                ModalGpuSpec("H100", price_per_hour_usd=4.0),
                ModalGpuSpec("L40S", price_per_hour_usd=2.0),
            ]
        )
        rows = infra_rows_for_model(model, snapshot)
        quote_ids = [row.plan.quote.id for row in rows]
        self.assertEqual(len(representative_profiles_for_model(model)), 2)
        self.assertEqual(len(quote_ids), len(set(quote_ids)))
        self.assertEqual(len(quote_ids), 4)
        self.assertLess(len(quote_ids), len(model.profiles) * 2)

        app = _TestApp()
        with patch(
            "llm_launchpad.tui.screens.fast_deploy.list_quick_deploy_models",
            return_value=(model,),
        ):
            async with app.run_test() as pilot:
                app.push_screen(FastDeployScreen())
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, FastDeployScreen)
                detail = str(screen.query_one("#fast-deploy-detail", Static).renderable)
                self.assertIn("2 quant options", detail)
                self.assertNotIn("ctx context", detail)
                self.assertNotIn("4 quant", detail)

                screen._selected_model = model
                screen.on_fast_deploy_availability_loaded(
                    FastDeployAvailabilityLoaded(snapshot)
                )
                await pilot.pause()

                option_list = screen.query_one("#fast-deploy-list", OptionList)
                self.assertEqual(option_list.option_count, len(rows))
                rendered_ids = [
                    option_list.get_option_at_index(index).id
                    for index in range(option_list.option_count)
                ]
                self.assertEqual(len(rendered_ids), len(set(rendered_ids)))
                first = str(option_list.get_option_at_index(0).prompt)
                self.assertIn("Modal", first)

    async def test_same_gpu_provider_regions_collapse_to_one_row(self) -> None:
        model = _model((_profile("fits", required_vram_gb=20.0),))
        snapshot = aggregate_compute_availability(
            prime_offers=(
                ComputeOffer(
                    id="offer-us",
                    cloud_id="cloud-1",
                    provider_name="provider-1",
                    gpu_type="H100_80GB",
                    gpu_count=1,
                    gpu_memory_gb=80.0,
                    country="US",
                    security="secure_cloud",
                    price_per_hour=3.29,
                    stock_status="Available",
                    images=(preferred_prime_offer_image(BackendType.LLAMACPP),),
                ),
                ComputeOffer(
                    id="offer-in",
                    cloud_id="cloud-1",
                    provider_name="provider-1",
                    gpu_type="H100_80GB",
                    gpu_count=1,
                    gpu_memory_gb=80.0,
                    country="IN",
                    security="secure_cloud",
                    price_per_hour=3.49,
                    stock_status="Available",
                    images=(preferred_prime_offer_image(BackendType.LLAMACPP),),
                ),
            )
        )
        rows = infra_rows_for_model(model, snapshot)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].plan.quote.region, "US")
        self.assertEqual(len(rows[0].alternative_plans), 2)
        self.assertEqual(
            {plan.quote.region for plan in rows[0].alternative_plans},
            {"US", "IN"},
        )

    async def test_escape_during_load_returns_to_models_and_ignores_stale_result(
        self,
    ) -> None:
        model = _model((_profile("fits", required_vram_gb=100.0),))
        snapshot = aggregate_compute_availability(
            modal_catalog=[ModalGpuSpec("H100", price_per_hour_usd=4.0)]
        )
        started = threading.Event()
        release = threading.Event()

        def fake_load() -> object:
            started.set()
            release.wait(timeout=5)
            return snapshot

        app = _TestApp()
        try:
            with patch(
                "llm_launchpad.tui.screens.fast_deploy.list_quick_deploy_models",
                return_value=(model,),
            ), patch(
                "llm_launchpad.tui.screens.fast_deploy.load_compute_availability",
                side_effect=fake_load,
            ):
                async with app.run_test() as pilot:
                    app.push_screen(FastDeployScreen())
                    await pilot.pause()

                    screen = app.screen
                    assert isinstance(screen, FastDeployScreen)
                    await pilot.press("enter")
                    self.assertTrue(started.wait(2))
                    self.assertEqual(screen._phase, "loading")
                    request_id = screen._availability_request_id

                    await pilot.press("escape")
                    await pilot.pause()

                    self.assertIsInstance(app.screen, FastDeployScreen)
                    self.assertEqual(screen._phase, "models")
                    status = str(screen.query_one("#fast-deploy-status", Static).renderable)
                    self.assertIn("1 model", status)

                    screen.on_fast_deploy_availability_loaded(
                        FastDeployAvailabilityLoaded(snapshot, request_id=request_id)
                    )
                    await pilot.pause()
                    self.assertEqual(screen._phase, "models")
                    self.assertEqual(
                        screen.query_one("#fast-deploy-list", OptionList).option_count,
                        1,
                    )
        finally:
            release.set()

    async def test_live_catalog_activation_refreshes_model_list(self) -> None:
        first = _model((_profile("first-model", required_vram_gb=20.0),))
        second = _model((_profile("second-model", required_vram_gb=20.0),))
        catalog_models: list[tuple[QuickDeployModel, ...]] = [(first,)]
        catalog_info: list[QuickDeployCatalogInfo] = [
            QuickDeployCatalogInfo(source_label="Bundled catalog")
        ]

        app = _TestApp()
        with patch(
            "llm_launchpad.tui.screens.fast_deploy.list_quick_deploy_models",
            side_effect=lambda: catalog_models[0],
        ), patch(
            "llm_launchpad.tui.screens.fast_deploy.get_quick_deploy_catalog_info",
            side_effect=lambda: catalog_info[0],
        ):
            async with app.run_test() as pilot:
                app.push_screen(FastDeployScreen())
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, FastDeployScreen)
                option_list = screen.query_one("#fast-deploy-list", OptionList)
                self.assertEqual(option_list.option_count, 1)
                self.assertIn(
                    "Bundled catalog",
                    str(screen.query_one("#fast-deploy-status", Static).renderable),
                )

                catalog_models[0] = (first, second)
                catalog_info[0] = QuickDeployCatalogInfo(
                    source_label="Live AAI rankings",
                    is_live=True,
                )
                screen._apply_model_catalog()
                await pilot.pause()

                option_list = screen.query_one("#fast-deploy-list", OptionList)
                self.assertEqual(option_list.option_count, 2)
                self.assertIn(
                    "Live AAI rankings",
                    str(screen.query_one("#fast-deploy-status", Static).renderable),
                )

    async def test_layout_css_gives_the_model_list_flexible_height(self) -> None:
        model = _model((_profile("fits", required_vram_gb=100.0),))
        app = _StyledApp()
        with patch(
            "llm_launchpad.tui.screens.fast_deploy.list_quick_deploy_models",
            return_value=(model,),
        ):
            async with app.run_test(size=(140, 42)) as pilot:
                app.push_screen(FastDeployScreen())
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, FastDeployScreen)
                option_list = screen.query_one("#fast-deploy-list", OptionList)
                detail = screen.query_one("#fast-deploy-detail", Static)
                self.assertGreaterEqual(option_list.size.height, 10)
                self.assertGreaterEqual(detail.size.height, 3)

    async def test_gpu_filter_limits_model_list_to_compatible_shapes(self) -> None:
        small = _model((_profile("small-fit", required_vram_gb=20.0, gpu_type="T4"),))
        huge = _model(
            (
                _profile(
                    "huge-fit",
                    required_vram_gb=200.0,
                    gpu_type="B200",
                    display_name="Huge Model",
                ),
            )
        )
        snapshot = aggregate_compute_availability(
            modal_catalog=[ModalGpuSpec("T4", price_per_hour_usd=0.5)]
        )
        self.assertEqual(len(snapshot.configurations), 1)
        gpu_type = snapshot.configurations[0].gpu_type
        option_values = [value for _label, value in _gpu_filter_options(snapshot)]
        self.assertEqual(option_values[0], "any")
        self.assertIn(gpu_type, option_values)

        app = _TestApp()
        with patch(
            "llm_launchpad.tui.screens.fast_deploy.list_quick_deploy_models",
            return_value=(small, huge),
        ):
            async with app.run_test() as pilot:
                app.push_screen(FastDeployScreen())
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, FastDeployScreen)
                screen.on_fast_deploy_availability_loaded(
                    FastDeployAvailabilityLoaded(snapshot, purpose="filter")
                )
                await pilot.pause()
                screen._gpu_filter = gpu_type
                screen._render_model_list()
                await pilot.pause()

                option_list = screen.query_one("#fast-deploy-list", OptionList)
                ids = [
                    option_list.get_option_at_index(index).id
                    for index in range(option_list.option_count)
                ]
                self.assertEqual(ids, ["small-fit"])
                status = str(screen.query_one("#fast-deploy-status", Static).renderable)
                self.assertIn(gpu_type, status)

    async def test_model_list_uses_live_cheapest_price_when_snapshot_loaded(self) -> None:
        model = _model((_profile("fits", required_vram_gb=100.0),))
        snapshot = aggregate_compute_availability(
            modal_catalog=[ModalGpuSpec("L40S", price_per_hour_usd=2.0)]
        )
        app = _TestApp()
        with patch(
            "llm_launchpad.tui.screens.fast_deploy.list_quick_deploy_models",
            return_value=(model,),
        ):
            async with app.run_test() as pilot:
                app.push_screen(FastDeployScreen())
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, FastDeployScreen)
                option_list = screen.query_one("#fast-deploy-list", OptionList)
                self.assertIn(
                    "from ~$4.00/hr",
                    str(option_list.get_option_at_index(0).prompt),
                )

                screen.on_fast_deploy_availability_loaded(
                    FastDeployAvailabilityLoaded(snapshot, purpose="filter")
                )
                await pilot.pause()

                option_list = screen.query_one("#fast-deploy-list", OptionList)
                self.assertIn(
                    "from ~$6.00/hr",
                    str(option_list.get_option_at_index(0).prompt),
                )

    async def test_infra_rows_use_canonical_gpu_labels(self) -> None:
        model = _model((_profile("fits", required_vram_gb=20.0),))
        snapshot = aggregate_compute_availability(
            modal_catalog=[ModalGpuSpec("H100!", price_per_hour_usd=4.0)],
            prime_offers=(
                ComputeOffer(
                    id="offer-us",
                    cloud_id="cloud-1",
                    provider_name="provider-1",
                    gpu_type="A6000_48GB",
                    gpu_count=1,
                    gpu_memory_gb=48.0,
                    country="US",
                    security="secure_cloud",
                    price_per_hour=0.54,
                    stock_status="Available",
                    images=(preferred_prime_offer_image(BackendType.LLAMACPP),),
                ),
            ),
        )
        labels = {
            row.configuration.gpu_type for row in infra_rows_for_model(model, snapshot)
        }
        self.assertEqual(labels, {"A6000 48GB", "H100 80GB"})

        app = _TestApp()
        with patch(
            "llm_launchpad.tui.screens.fast_deploy.list_quick_deploy_models",
            return_value=(model,),
        ):
            async with app.run_test() as pilot:
                app.push_screen(FastDeployScreen())
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, FastDeployScreen)
                screen._selected_model = model
                screen.on_fast_deploy_availability_loaded(
                    FastDeployAvailabilityLoaded(snapshot)
                )
                await pilot.pause()

                option_list = screen.query_one("#fast-deploy-list", OptionList)
                prompts = [
                    str(option_list.get_option_at_index(index).prompt)
                    for index in range(option_list.option_count)
                ]
                joined = "\n".join(prompts)
                self.assertIn("A6000 48GB", joined)
                self.assertIn("H100 80GB", joined)
                self.assertNotIn("H100!", joined)
                self.assertNotIn("A6000_48GB", joined)
                subtitle = str(screen.query_one("#fast-deploy-subtitle", Static).renderable)
                self.assertIn("Pick infrastructure", subtitle)

    async def test_gpu_filter_without_matching_infra_does_not_show_other_gpus(
        self,
    ) -> None:
        model = _model((_profile("fits", required_vram_gb=20.0, gpu_type="T4"),))
        snapshot = aggregate_compute_availability(
            modal_catalog=[ModalGpuSpec("T4", price_per_hour_usd=0.5)]
        )
        app = _TestApp()
        with patch(
            "llm_launchpad.tui.screens.fast_deploy.list_quick_deploy_models",
            return_value=(model,),
        ):
            async with app.run_test() as pilot:
                app.push_screen(FastDeployScreen())
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, FastDeployScreen)
                screen._selected_model = model
                screen._gpu_filter = "B200 180GB"
                screen.on_fast_deploy_availability_loaded(
                    FastDeployAvailabilityLoaded(snapshot)
                )
                await pilot.pause()

                option_list = screen.query_one("#fast-deploy-list", OptionList)
                prompt = str(option_list.get_option_at_index(0).prompt)
                self.assertIn("No placements on B200 180GB", prompt)
                self.assertNotIn("T4", prompt)
                status = str(screen.query_one("#fast-deploy-status", Static).renderable)
                self.assertIn("No live placements on B200 180GB", status)

    async def test_compact_viewport_keeps_model_list_usable(self) -> None:
        model = _model((_profile("fits", required_vram_gb=100.0),))
        app = _StyledApp()
        with patch(
            "llm_launchpad.tui.screens.fast_deploy.list_quick_deploy_models",
            return_value=(model,),
        ):
            async with app.run_test(size=(80, 24)) as pilot:
                app.push_screen(FastDeployScreen())
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, FastDeployScreen)
                option_list = screen.query_one("#fast-deploy-list", OptionList)
                self.assertGreaterEqual(option_list.size.height, 6)


class FastDeployMainMenuTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._catalog_patch = patch(
            "llm_launchpad.core.quick_deploy._read_bundled_catalog_text",
            return_value=None,
        )
        self._catalog_patch.start()
        self._prime_refresh_patch = patch.object(
            MainMenuScreen,
            "_refresh_prime_auth_status",
            lambda self: None,
        )
        self._prime_refresh_patch.start()
        self._catalog_refresh_patch = patch.object(
            MainMenuScreen,
            "_refresh_quick_deploy_catalog",
            lambda self: None,
        )
        self._catalog_refresh_patch.start()
        self._aai_refresh_patch = patch.object(
            MainMenuScreen,
            "_refresh_aai_auth_status",
            lambda self: None,
        )
        self._aai_refresh_patch.start()
        self._availability_patch = patch(
            "llm_launchpad.tui.screens.fast_deploy.load_compute_availability",
            return_value=aggregate_compute_availability(),
        )
        self._availability_patch.start()
        quick_deploy._reset_quick_deploy_catalog_cache()

    def tearDown(self) -> None:
        self._catalog_patch.stop()
        self._prime_refresh_patch.stop()
        self._catalog_refresh_patch.stop()
        self._aai_refresh_patch.stop()
        self._availability_patch.stop()
        quick_deploy._reset_quick_deploy_catalog_cache()

    async def test_main_menu_entry_routes_to_fast_deploy(self) -> None:
        app = _TestApp()
        with patch.object(MainMenuScreen, "_refresh_modal_auth_status", lambda self: None), patch.object(
            MainMenuScreen, "_refresh_hf_auth_status", lambda self: None
        ), patch.object(MainMenuScreen, "_refresh_panels", lambda self: None), patch.object(
            MainMenuScreen, "_refresh_storage_estimate", lambda self: None
        ):
            async with app.run_test() as pilot:
                app.push_screen(MainMenuScreen(username="alice", version="1.0.0"))
                await pilot.pause()

                screen = app.screen
                assert isinstance(screen, MainMenuScreen)
                action_list = screen.query_one("#action-list", OptionList)
                fast_option = next(
                    action_list.get_option_at_index(index)
                    for index in range(action_list.option_count)
                    if action_list.get_option_at_index(index).id == "deploy"
                )
                screen.on_option_list_option_selected(
                    SimpleNamespace(option=fast_option, option_list=action_list)
                )
                await pilot.pause()

                self.assertEqual(app.fast_deploy_calls, 1)
                self.assertIsInstance(app.screen, FastDeployScreen)


if __name__ == "__main__":
    unittest.main()
