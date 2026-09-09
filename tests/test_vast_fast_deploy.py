from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from textual.widgets import OptionList, Static

from llm_launchpad.core.compute_availability import aggregate_compute_availability, load_compute_availability
from llm_launchpad.core.modal_gpu import ModalGpuSpec
from llm_launchpad.core.quick_deploy import QuickDeployModel
from llm_launchpad.core.vast_auth import VastCredentials
from llm_launchpad.core.vast_backend import VastApiError, parse_vast_offer, vast_offer_search_payload
from llm_launchpad.core.vast_comparison import vast_offers_for_model
from llm_launchpad.core.serving_tiers import serving_tiers
from llm_launchpad.protocol.enums import ComputeProvider, ServingObjective
from llm_launchpad.protocol.models import MemoryEstimate, RuntimeTuning, ServingRequirements, VastOffer, VastOfferQuery
from llm_launchpad.tui.screens.fast_deploy import (
    FastDeployAvailabilityLoaded, FastDeployScreen, _gpu_filter_options,
    _model_cost_label, _model_fits_gpu_type, infra_rows_for_model,
)
from tests.test_fast_deploy_screen import _model, _profile, _StyledApp
from tests.test_vast_backend import offer_payload


def comparison_model(*, weights: float = 8, total: float = 14) -> QuickDeployModel:
    profile = replace(
        _profile("small-model", required_vram_gb=total),
        memory_estimate=MemoryEstimate(
            weights_gb=weights, kv_cache_gb=total - weights - 3,
            compute_gb=1, speculative_gb=0, reserve_gb=2, total_gb=total,
        ),
        serving_requirements=ServingRequirements(context_tokens=32768),
        runtime_tuning=RuntimeTuning(),
    )
    return _model((profile,))


def vast_offer(**changes: object) -> VastOffer:
    offer = parse_vast_offer(offer_payload(**changes), VastOfferQuery(gpu_count=None))
    assert offer is not None
    return offer


class VastModelComparisonTests(unittest.TestCase):
    def test_cheaper_supported_vast_offer_wins_economy_serving_tier(self) -> None:
        snapshot = replace(
            aggregate_compute_availability(modal_catalog=[ModalGpuSpec("L4", price_per_hour_usd=0.8)]),
            vast_offers=(vast_offer(), vast_offer(id=1002, num_gpus=2, dph_total=0.2)),
        )
        rows = infra_rows_for_model(comparison_model(), snapshot)
        tiers = serving_tiers([row.plan for row in rows], ServingObjective.GENERAL_PURPOSE)
        economy = next(tier for tier in tiers if tier.key == "economy")
        self.assertEqual(economy.plan.quote.provider, ComputeProvider.VAST)
        self.assertEqual(economy.price_per_hour_usd, 0.42)
        self.assertTrue(all(row.plan.quote.gpu_count == 1 for row in rows))

    def test_search_includes_single_and_multi_gpu_topologies(self) -> None:
        query = VastOfferQuery(gpu_count=None, limit=500)
        payload = vast_offer_search_payload(query)
        self.assertEqual(payload["num_gpus"], {"gte": 1, "lte": 8})
        self.assertIsNotNone(parse_vast_offer(offer_payload(num_gpus=4), query))
        self.assertIsNone(parse_vast_offer(offer_payload(num_gpus=16), query))
        self.assertIsNone(parse_vast_offer(offer_payload(num_gpus=None), query))

    def test_cheaper_vast_fit_changes_model_price_and_gpu_filter(self) -> None:
        model = comparison_model()
        snapshot = replace(
            aggregate_compute_availability(modal_catalog=[ModalGpuSpec("L4", price_per_hour_usd=0.8)]),
            vast_offers=(vast_offer(),), vast_configured=True,
        )
        self.assertEqual(_model_cost_label(model, snapshot), "~$0.42/hr")
        options = _gpu_filter_options(snapshot)
        self.assertIn("Vast.ai", options[1][0])
        self.assertTrue(_model_fits_gpu_type(model, snapshot, "RTX 4090 24GB"))
        self.assertFalse(_model_fits_gpu_type(comparison_model(total=50), snapshot, "RTX 4090 24GB"))
        self.assertEqual(_model_cost_label(model, snapshot, "L4 24GB"), "~$0.80/hr")

    def test_unknown_and_more_expensive_vast_prices_do_not_lower_model_price(self) -> None:
        model = comparison_model()
        snapshot = aggregate_compute_availability(modal_catalog=[ModalGpuSpec("L4", price_per_hour_usd=0.3)])
        for offer in (vast_offer(), vast_offer(dph_base=None, dph_total=None)):
            self.assertEqual(_model_cost_label(model, replace(snapshot, vast_offers=(offer,))), "~$0.30/hr")

    def test_vast_only_model_price_does_not_fall_back_to_unconnected_modal(self) -> None:
        snapshot = replace(aggregate_compute_availability(), providers=(), vast_offers=(vast_offer(dph_base=5, dph_total=5.02),))
        self.assertEqual(_model_cost_label(comparison_model(), snapshot), "~$5.02/hr")

    def test_full_context_reserve_uses_actual_memory_not_cli_display_gb(self) -> None:
        # 24576 MiB is 24 GiB, not the CLI's display value of 24.576.
        self.assertEqual(vast_offer().gpu_memory_gib, 24)
        self.assertEqual(vast_offers_for_model(comparison_model(total=24.2), (vast_offer(),)), ())

    def test_multi_gpu_offer_keeps_whole_machine_price(self) -> None:
        rows = vast_offers_for_model(comparison_model(total=40), (
            vast_offer(), vast_offer(id=1002, num_gpus=2, dph_base=0.7, dph_total=0.72),
        ))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].offer.gpu_count, 2)
        self.assertEqual(rows[0].costs.total_per_hour_usd, 0.72)
        self.assertTrue(rows[0].assessment.fits)

    def test_incomplete_context_or_memory_metadata_cannot_claim_fit(self) -> None:
        model = comparison_model()
        for changes in (
            {"memory_estimate": None}, {"runtime_tuning": None},
            {"serving_requirements": ServingRequirements(context_tokens=8192)},
        ):
            with self.subTest(changes=changes):
                candidate = replace(model, profiles=(replace(model.profiles[0], **changes),))
                self.assertEqual(vast_offers_for_model(candidate, (vast_offer(),)), ())

    def test_disk_capacity_excludes_hosts_and_resizes_price_for_large_models(self) -> None:
        model = comparison_model(weights=110, total=130)
        rows = vast_offers_for_model(model, (
            vast_offer(num_gpus=8, disk_space=120),
            vast_offer(id=1002, num_gpus=8, disk_space=500),
        ))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.offer.id, "1002")
        self.assertEqual(row.disk_gb, 131)
        self.assertAlmostEqual(row.costs.total_per_hour_usd, 0.4262)
        self.assertEqual(row.costs.download_per_gb_usd, 0.002)

    def test_unknown_disk_price_stays_unknown_when_allocation_grows(self) -> None:
        rows = vast_offers_for_model(comparison_model(weights=110, total=130), (
            vast_offer(num_gpus=8, dph_base=None, disk_space=500),
        ))
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0].costs.total_per_hour_usd)

    def test_duplicate_topologies_use_cheapest_offer_and_preserve_different_quants(self) -> None:
        model = comparison_model()
        model = replace(model, profiles=(model.profiles[0], replace(model.profiles[0], quant="Q8_0")))
        rows = vast_offers_for_model(model, (
            vast_offer(id=1002, dph_base=None, dph_total=None), vast_offer(),
            vast_offer(id=1003, dph_base=0.2, dph_total=0.22),
        ))
        self.assertEqual(len(rows), 2)
        self.assertEqual({row.offer.id for row in rows}, {"1003"})
        self.assertEqual(len({row.id for row in rows}), 2)


class VastAvailabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        for patcher in (
            patch("llm_launchpad.core.compute_availability.resolve_modal_cli_path", return_value="modal"),
            patch("llm_launchpad.core.compute_availability.get_prime_auth_status", return_value=SimpleNamespace(authenticated=False)),
            patch("llm_launchpad.core.compute_availability.fetch_modal_gpu_catalog", return_value=[ModalGpuSpec("L4", price_per_hour_usd=0.8)]),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_loads_vast_alongside_modal_without_enabling_vast_deploy_routes(self) -> None:
        offer = vast_offer()
        with patch("llm_launchpad.core.compute_availability.resolve_vast_credentials", return_value=VastCredentials("key", "stored")), patch(
            "llm_launchpad.core.compute_availability.VastBackend.list_offers", return_value=[offer]
        ) as search:
            snapshot = load_compute_availability()
        self.assertEqual(snapshot.vast_offers, (offer,))
        self.assertTrue(snapshot.vast_configured)
        self.assertEqual(snapshot.providers, (ComputeProvider.MODAL,))
        self.assertIsNone(search.call_args.args[0].gpu_count)

    def test_vast_failure_keeps_modal_results_and_reports_partial_availability(self) -> None:
        with patch("llm_launchpad.core.compute_availability.resolve_vast_credentials", return_value=VastCredentials("key", "stored")), patch(
            "llm_launchpad.core.compute_availability.VastBackend.list_offers", side_effect=VastApiError("Rate limit")
        ):
            snapshot = load_compute_availability()
        self.assertEqual(len(snapshot.configurations), 1)
        self.assertEqual(snapshot.vast_offers, ())
        self.assertIn("Vast availability unavailable: Rate limit", snapshot.errors)

    def test_missing_key_does_not_contact_vast(self) -> None:
        with patch("llm_launchpad.core.compute_availability.resolve_vast_credentials", return_value=VastCredentials()), patch(
            "llm_launchpad.core.compute_availability.VastBackend"
        ) as backend:
            snapshot = load_compute_availability()
        backend.assert_not_called()
        self.assertFalse(snapshot.vast_configured)


class VastFastDeployScreenTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_gpu_is_deployable_and_multi_gpu_stays_a_comparison(self) -> None:
        model = comparison_model()
        snapshot = replace(
            aggregate_compute_availability(modal_catalog=[ModalGpuSpec("L4", price_per_hour_usd=0.8)]),
            vast_offers=(vast_offer(), vast_offer(id=1002, num_gpus=2)), vast_configured=True, providers=(ComputeProvider.MODAL,),
        )
        with patch("llm_launchpad.tui.screens.fast_deploy.list_quick_deploy_models", return_value=(model,)), patch(
            "llm_launchpad.tui.screens.fast_deploy.load_compute_availability", return_value=snapshot
        ):
            app = _StyledApp()
            async with app.run_test(size=(100, 35)) as pilot:
                screen = FastDeployScreen()
                await app.push_screen(screen)
                await pilot.pause()
                screen._open_model(model.id)
                await pilot.pause()
                options = screen.query_one(OptionList)
                self.assertTrue(any("Vast preview" in str(options.get_option_at_index(i).prompt) for i in range(options.option_count)))
                comparison_id = next(iter(screen._vast_rows))
                options.highlighted = options.get_option_index(comparison_id)
                options.focus()
                await pilot.press("enter")
                await pilot.pause()
                self.assertEqual(app.quick_deploy_calls, [])
                self.assertIn("Comparison only", str(screen.query_one("#fast-deploy-detail", Static).content))
                screen._choose(next(key for key, row in screen._infra_rows.items() if row.plan.quote.provider == ComputeProvider.VAST))
                self.assertEqual(len(app.quick_deploy_calls), 1)
                self.assertEqual(app.quick_deploy_calls[0][0].quote.provider, ComputeProvider.VAST)
                self.assertTrue(all(plan.quote.gpu_count == 1 for plan in app.quick_deploy_calls[0][1]))

    async def test_vast_only_fit_survives_filter_and_refresh_discards_old_prices(self) -> None:
        model = comparison_model()
        snapshot = replace(aggregate_compute_availability(), providers=(), vast_offers=(vast_offer(),), vast_configured=True)
        with patch("llm_launchpad.tui.screens.fast_deploy.list_quick_deploy_models", return_value=(model,)), patch(
            "llm_launchpad.tui.screens.fast_deploy.load_compute_availability", return_value=snapshot
        ) as load:
            app = _StyledApp()
            async with app.run_test(size=(80, 30)) as pilot:
                screen = FastDeployScreen()
                await app.push_screen(screen)
                await pilot.pause()
                screen._gpu_filter = "RTX 4090 24GB"
                screen._open_model(model.id)
                await pilot.pause()
                self.assertEqual(screen._phase, "infra")
                self.assertFalse(screen._vast_rows)
                self.assertEqual(len(screen._infra_rows), 1)
                old_request = screen._availability_request_id
                load.return_value = replace(snapshot, vast_offers=(), errors=("Vast unavailable",))
                screen.action_refresh_availability()
                await pilot.pause()
                self.assertFalse(screen._vast_rows)
                self.assertFalse(screen._snapshot.vast_offers)
                screen.on_fast_deploy_availability_loaded(FastDeployAvailabilityLoaded(snapshot, request_id=old_request, purpose="filter"))
                self.assertFalse(screen._snapshot.vast_offers)
                self.assertEqual(app.quick_deploy_calls, [])
