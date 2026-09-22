from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from dataclasses import replace

from llm_launchpad.core.compute_availability import (
    aggregate_compute_availability,
    canonical_gpu_identity,
    display_gpu_type,
    load_compute_availability,
    plans_for_compute_profile,
    recipe_for_placement,
)
from llm_launchpad.core.llamacpp_planner import compile_server_args
from llm_launchpad.core.quick_deploy import quick_deploy_recipe
from llm_launchpad.core.modal_gpu import ModalGpuSpec
from llm_launchpad.core.prime_backend import PrimeBackend, preferred_prime_offer_image
from llm_launchpad.core.quick_deploy import QuickDeployProfile
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import ComputeOffer, PrimeProviderOptions


def _profile(
    *,
    profile_id: str = "test-model",
    backend: BackendType = BackendType.LLAMACPP,
    required_vram_gb: float | None = 150.0,
) -> QuickDeployProfile:
    return QuickDeployProfile(
        id=profile_id,
        display_name="Test Model",
        repo_id="acme/Test-Model-GGUF",
        quant="Q4_K_M",
        gpu_type="H100",
        gpu_count=2,
        profile_label="Test",
        approx_cost_per_hour_usd=8.0,
        max_context_tokens=32768,
        instance_slug_hint="test-model",
        summary="A test model.",
        server_args=(),
        required_vram_gb=required_vram_gb,
        backend=backend,
        model_name="acme/Test-Model" if backend == BackendType.VLLM else None,
    )


def _prime_offer(
    *,
    offer_id: str = "abc123",
    gpu_type: str = "H100_80GB",
    gpu_count: int = 4,
    price: float = 7.0,
    security: str = "secure_cloud",
    stock_status: str = "Available",
    is_spot: bool = False,
    gpu_memory_gb: float = 80.0,
    images: tuple[str, ...] | None = None,
) -> ComputeOffer:
    return ComputeOffer(
        id=offer_id,
        cloud_id="cloud-1",
        provider_name="provider-1",
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        gpu_memory_gb=gpu_memory_gb,
        country="CA",
        security=security,
        price_per_hour=price,
        stock_status=stock_status,
        is_spot=is_spot,
        images=images
        or (
            preferred_prime_offer_image(BackendType.LLAMACPP),
            preferred_prime_offer_image(BackendType.VLLM),
        ),
    )


class ComputeAvailabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        from llm_launchpad.core.vast_auth import VastCredentials

        patcher = patch(
            "llm_launchpad.core.compute_availability.resolve_vast_credentials",
            return_value=VastCredentials(),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_load_records_authenticated_provider_when_fetch_fails(self) -> None:
        with patch(
            "llm_launchpad.core.compute_availability.resolve_modal_cli_path",
            return_value=None,
        ), patch(
            "llm_launchpad.core.compute_availability.get_prime_auth_status",
            return_value=SimpleNamespace(authenticated=True),
        ), patch.object(
            PrimeBackend,
            "list_offers",
            side_effect=RuntimeError("provider down"),
        ):
            snapshot = load_compute_availability()

        self.assertEqual(snapshot.providers, (ComputeProvider.PRIME,))
        self.assertIn("provider down", snapshot.errors[0])

    def test_load_uses_public_modal_catalog_without_token_preflight(self) -> None:
        with patch(
            "llm_launchpad.core.compute_availability.resolve_modal_cli_path",
            return_value="/usr/bin/modal",
        ), patch(
            "llm_launchpad.core.compute_availability.get_prime_auth_status",
            return_value=SimpleNamespace(authenticated=False),
        ), patch(
            "llm_launchpad.core.compute_availability.fetch_modal_gpu_catalog",
            return_value=[ModalGpuSpec("H100", price_per_hour_usd=3.95)],
        ):
            snapshot = load_compute_availability()

        self.assertEqual(snapshot.providers, (ComputeProvider.MODAL,))
        self.assertEqual(snapshot.configurations[0].gpu_type, "H100 80GB")

    def test_a_provider_that_never_answers_cannot_hold_the_whole_screen(self) -> None:
        """One stalled catalog used to leave Fast Deploy on "loading" forever.

        The fetches were awaited with no timeout, so a provider that never
        returned had no deadline and no way back -- a live run sat in the
        loading phase for 300s and was killed rather than recovering. The
        other providers' rows are still real, so they are still shown.
        """
        import threading

        release = threading.Event()
        self.addCleanup(release.set)

        def never_answers() -> list[ComputeOffer]:
            release.wait(30)
            raise AssertionError("cancelled fetch should not decide the snapshot")

        with patch(
            "llm_launchpad.core.compute_availability.COMPUTE_AVAILABILITY_TIMEOUT_SECONDS",
            0.2,
        ), patch(
            "llm_launchpad.core.compute_availability.resolve_modal_cli_path",
            return_value="/usr/bin/modal",
        ), patch(
            "llm_launchpad.core.compute_availability.get_prime_auth_status",
            return_value=SimpleNamespace(authenticated=True),
        ), patch(
            "llm_launchpad.core.compute_availability.fetch_modal_gpu_catalog",
            return_value=[ModalGpuSpec("H100", price_per_hour_usd=3.95)],
        ), patch.object(PrimeBackend, "list_offers", side_effect=never_answers):
            started = time.monotonic()
            snapshot = load_compute_availability()
            elapsed = time.monotonic() - started

        # Returns on the budget, not on the stalled provider.
        self.assertLess(elapsed, 15)
        self.assertTrue(
            any("did not answer" in error for error in snapshot.errors), snapshot.errors
        )
        # The provider that did answer is still usable.
        self.assertEqual(snapshot.configurations[0].gpu_type, "H100 80GB")

    def test_display_gpu_type_strips_provider_suffixes(self) -> None:
        self.assertEqual(display_gpu_type("H100!"), "H100 80GB")
        self.assertEqual(display_gpu_type("B200+"), "B200 180GB")
        self.assertEqual(display_gpu_type("A6000_48GB"), "A6000 48GB")
        self.assertEqual(display_gpu_type("RTX-PRO-6000"), "RTX PRO 6000 96GB")

    def test_aggregates_equivalent_gpu_types_across_providers(self) -> None:
        snapshot = aggregate_compute_availability(
            modal_catalog=[ModalGpuSpec("H100", price_per_hour_usd=3.95)],
            prime_offers=[_prime_offer()],
        )

        self.assertEqual(len(snapshot.configurations), 1)
        configuration = snapshot.configurations[0]
        self.assertEqual(configuration.id, "h100-80gb")
        self.assertEqual(configuration.gpu_type, "H100 80GB")
        self.assertEqual(configuration.gpu_count_min, 1)
        self.assertEqual(configuration.gpu_count_max, 8)
        self.assertEqual(configuration.live_placement_count, 1)
        self.assertTrue(configuration.has_on_demand_capacity)
        self.assertEqual(configuration.source_count, 2)
        self.assertEqual(configuration.regions, ("CA",))
        self.assertEqual(configuration.minimum_price_per_hour_usd, 3.95)

    def test_multi_gpu_prime_total_memory_groups_by_per_gpu_type(self) -> None:
        snapshot = aggregate_compute_availability(
            modal_catalog=[ModalGpuSpec("A100-80GB", price_per_hour_usd=2.50)],
            prime_offers=[
                _prime_offer(
                    offer_id="eight",
                    gpu_type="A100_80GB",
                    gpu_count=8,
                    gpu_memory_gb=640.0,
                    price=22.4,
                )
            ],
        )

        self.assertEqual([row.gpu_type for row in snapshot.configurations], ["A100 80GB"])
        configuration = snapshot.configurations[0]
        self.assertEqual(configuration.id, "a100-80gb")
        self.assertEqual(configuration.gpu_memory_gb, 80.0)
        self.assertEqual(configuration.gpu_count_min, 1)
        self.assertEqual(configuration.gpu_count_max, 8)

    def test_excludes_unusable_marketplace_rows(self) -> None:
        snapshot = aggregate_compute_availability(
            prime_offers=[
                _prime_offer(offer_id="insecure", security="community_cloud"),
                _prime_offer(offer_id="gone", stock_status="Out of stock"),
            ]
        )

        self.assertEqual(snapshot.configurations, ())

    def test_labels_spot_capacity_and_keeps_it_out_of_plans(self) -> None:
        snapshot = aggregate_compute_availability(
            prime_offers=[
                _prime_offer(offer_id="ondemand", price=7.0),
                _prime_offer(offer_id="spot", is_spot=True, price=2.0),
            ]
        )

        self.assertEqual(len(snapshot.configurations), 1)
        configuration = snapshot.configurations[0]
        self.assertEqual(configuration.live_placement_count, 1)
        self.assertEqual(configuration.spot_placement_count, 1)
        self.assertFalse(configuration.has_on_demand_capacity)
        self.assertEqual(configuration.minimum_price_per_hour_usd, 7.0)
        self.assertEqual(configuration.gpu_count_min, 4)
        self.assertEqual(configuration.gpu_count_max, 4)

        plans = plans_for_compute_profile(configuration, _profile())

        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].quote.provider_reference, "ondemand")
        self.assertEqual(plans[0].quote.price_per_hour_usd, 7.0)

    def test_spot_only_configurations_are_visible_but_not_deployable(self) -> None:
        snapshot = aggregate_compute_availability(
            prime_offers=[_prime_offer(is_spot=True)]
        )

        self.assertEqual(len(snapshot.configurations), 1)
        configuration = snapshot.configurations[0]
        self.assertEqual(configuration.spot_placement_count, 1)
        self.assertEqual(configuration.live_placement_count, 0)
        self.assertFalse(configuration.has_on_demand_capacity)
        self.assertEqual(configuration.minimum_price_per_hour_usd, 7.0)
        self.assertEqual(configuration.gpu_count_max, 4)

        plans = plans_for_compute_profile(configuration, _profile())

        self.assertEqual(plans, ())

    def test_spot_teaser_price_does_not_outrank_deployable_gpu_types(self) -> None:
        snapshot = aggregate_compute_availability(
            modal_catalog=[ModalGpuSpec("L40S", price_per_hour_usd=2.0)],
            prime_offers=[
                _prime_offer(offer_id="h100-live", price=10.0),
                _prime_offer(offer_id="h100-spot", is_spot=True, price=1.0),
            ],
        )

        self.assertEqual(
            [row.gpu_type for row in snapshot.configurations],
            ["L40S 48GB", "H100 80GB"],
        )
        self.assertEqual(snapshot.configurations[1].minimum_price_per_hour_usd, 10.0)

    def test_spot_scale_does_not_inflate_deployable_gpu_counts(self) -> None:
        snapshot = aggregate_compute_availability(
            prime_offers=[
                _prime_offer(offer_id="ondemand", gpu_count=4, price=7.0),
                _prime_offer(offer_id="spot", gpu_count=8, is_spot=True, price=2.0),
            ]
        )

        configuration = snapshot.configurations[0]
        self.assertEqual(configuration.gpu_count_min, 4)
        self.assertEqual(configuration.gpu_count_max, 4)
        self.assertEqual(configuration.total_vram_max_gb, 320.0)
        self.assertEqual(configuration.minimum_price_per_hour_usd, 7.0)
        self.assertEqual(configuration.spot_placement_count, 1)

    def test_configurations_are_sorted_by_lowest_price_with_unknowns_last(self) -> None:
        snapshot = aggregate_compute_availability(
            modal_catalog=[
                ModalGpuSpec("H100", price_per_hour_usd=4.0),
                ModalGpuSpec("L40S", price_per_hour_usd=2.0),
                ModalGpuSpec("B200", price_per_hour_usd=None),
            ]
        )

        self.assertEqual(
            [row.gpu_type for row in snapshot.configurations],
            ["L40S 48GB", "H100 80GB", "B200 180GB"],
        )

    def test_builds_ranked_plans_and_sizes_scalable_placements(self) -> None:
        configuration = aggregate_compute_availability(
            modal_catalog=[ModalGpuSpec("H100", price_per_hour_usd=4.0)],
            prime_offers=[_prime_offer(price=20.0)],
        ).configurations[0]

        plans = plans_for_compute_profile(configuration, _profile())

        self.assertEqual(len(plans), 2)
        self.assertEqual(plans[0].quote.provider, ComputeProvider.PRIME)
        self.assertEqual(plans[0].quote.gpu_count, 4)
        self.assertIsInstance(plans[0].quote.provider_options, PrimeProviderOptions)
        modal_plan = next(
            plan for plan in plans if plan.quote.provider == ComputeProvider.MODAL
        )
        self.assertEqual(modal_plan.quote.gpu_count, 2)
        self.assertEqual(modal_plan.quote.price_per_hour_usd, 8.0)

    def test_rejects_gpu_type_when_maximum_vram_is_too_small(self) -> None:
        configuration = aggregate_compute_availability(
            modal_catalog=[ModalGpuSpec("T4", price_per_hour_usd=0.6)],
        ).configurations[0]

        plans = plans_for_compute_profile(
            configuration,
            _profile(required_vram_gb=140.0),
        )

        self.assertEqual(plans, ())

    def test_an_excluded_placement_reports_why(self) -> None:
        # A shorter list with no explanation reads as missing hardware rather
        # than as hardware that would not have worked.
        configuration = aggregate_compute_availability(
            modal_catalog=[ModalGpuSpec("T4", price_per_hour_usd=0.6)],
        ).configurations[0]
        rejected: list[str] = []

        plans = plans_for_compute_profile(
            configuration,
            _profile(required_vram_gb=140.0),
            rejected=rejected,
        )

        self.assertEqual(plans, ())
        self.assertTrue(rejected)
        self.assertTrue(all(reason.strip() for reason in rejected))

    def test_nothing_is_reported_when_every_placement_qualifies(self) -> None:
        configuration = aggregate_compute_availability(
            modal_catalog=[ModalGpuSpec("H100", price_per_hour_usd=4.0)],
        ).configurations[0]
        rejected: list[str] = []

        plans = plans_for_compute_profile(configuration, _profile(), rejected=rejected)

        self.assertTrue(plans)
        self.assertEqual(rejected, [])

    def test_infers_missing_vram_from_the_profile_gpu_shape(self) -> None:
        configuration = aggregate_compute_availability(
            modal_catalog=[ModalGpuSpec("T4", price_per_hour_usd=0.6)],
        ).configurations[0]

        plans = plans_for_compute_profile(
            configuration,
            _profile(required_vram_gb=None),
        )

        self.assertEqual(plans, ())

    def test_canonical_identity_keeps_memory_variants_separate(self) -> None:
        self.assertEqual(
            canonical_gpu_identity("A100-40GB", 40.0),
            ("a100-40gb", "A100 40GB"),
        )
        self.assertEqual(
            canonical_gpu_identity("A100_80GB", 80.0),
            ("a100-80gb", "A100 80GB"),
        )

    def test_reported_capacity_is_rounded_to_whole_gigabytes(self) -> None:
        """Providers report VRAM from the device byte count, not the spec sheet.

        Printing it raw filled the GPU filter with "RTX A2000 5.99414GB" and,
        because the identity is built from the label, listed one card several
        times when two hosts reported slightly different capacities.
        """
        for gpu, memory_gb, expected in (
            ("RTX A2000", 5.99414, "RTX A2000 6GB"),
            ("RTX 4060 TI", 15.9961, "RTX 4060 TI 16GB"),
            ("RTX 5060 TI", 15.9287, "RTX 5060 TI 16GB"),
            ("H100 SXM", 79.6475, "H100 SXM 80GB"),
            ("RTX PRO 6000 MAX Q", 95.5928, "RTX PRO 6000 MAX Q 96GB"),
        ):
            with self.subTest(gpu=gpu):
                self.assertEqual(canonical_gpu_identity(gpu, memory_gb)[1], expected)

    def test_hosts_reporting_the_same_card_share_one_identity(self) -> None:
        first = canonical_gpu_identity("RTX 5060 TI", 15.9287)
        second = canonical_gpu_identity("RTX 5060 TI", 15.9961)
        self.assertEqual(first, second)
        # Genuinely different variants of one card still stay apart.
        self.assertNotEqual(first, canonical_gpu_identity("RTX 5060 TI", 7.95996))

    def test_a_name_that_already_carries_its_capacity_is_not_doubled(self) -> None:
        # The suffix used to be stripped with int() truncation, so a 5.99414
        # reading never matched the "-6GB" the name already ended with.
        self.assertEqual(
            canonical_gpu_identity("RTX-A2000-6GB", 5.99414)[1],
            "RTX A2000 6GB",
        )


if __name__ == "__main__":
    unittest.main()


class PlacementRuntimeMarginTests(unittest.TestCase):
    """The runtime margin is a share of the device, so it is set per placement."""

    def _profile_with_plan(self) -> QuickDeployProfile:
        from llm_launchpad.protocol.models import (
            MemoryEstimate,
            RuntimeTuning,
            ServingRequirements,
        )
        from llm_launchpad.protocol.enums import ServingObjective

        tuning = RuntimeTuning(
            parallel_slots=4, batch_size=2048, ubatch_size=64,
            cache_type_k="f16", cache_type_v="f16", flash_attention=False,
            gpu_layers="all", fit_target_mib=2048,
        )
        requirements = ServingRequirements(
            context_tokens=32768, objective=ServingObjective.GENERAL_PURPOSE,
            full_context_per_request=True, gpu_only=True,
        )
        memory = MemoryEstimate(
            weights_gb=20.0, kv_cache_gb=8.0, compute_gb=2.0,
            attention_scratch_gb=1.0, speculative_gb=0.0, reserve_gb=0.0,
            total_gb=31.0, per_device_required_gb=(31.0,), confidence=0.82,
            source="gguf-metadata", total_layer_count=32,
        )
        base = _profile(required_vram_gb=31.0)
        return replace(
            base,
            serving_requirements=requirements,
            runtime_tuning=tuning,
            memory_estimate=memory,
            server_args=tuple(compile_server_args(requirements, tuning)),
        )

    def test_a_large_device_gets_the_margin_its_memory_model_promises(self) -> None:
        # Live: the catalog's 2 GiB floor packed a 180 GB B200 to within 2 GiB
        # and llama.cpp then had nothing left for its compute graphs. Five
        # GLM-5.3-Flash deploys died in graph_reserve; the one that survived
        # differed from them in this argument alone (9216 against 2048).
        recipe = quick_deploy_recipe(self._profile_with_plan())

        retuned = recipe_for_placement(recipe, 180.0)

        assert retuned.runtime_tuning is not None
        self.assertEqual(retuned.runtime_tuning.fit_target_mib, 9216)
        args = list(retuned.server_args or ())
        self.assertEqual(args[args.index("--fit-target") + 1], "9216")

    def test_a_small_device_keeps_the_two_gibibyte_floor(self) -> None:
        recipe = quick_deploy_recipe(self._profile_with_plan())

        retuned = recipe_for_placement(recipe, 24.0)

        # 5% of 24 GB is under the floor, so nothing changes and the recipe is
        # returned unchanged rather than rebuilt.
        self.assertIs(retuned, recipe)

    def test_an_unknown_device_size_changes_nothing(self) -> None:
        recipe = quick_deploy_recipe(self._profile_with_plan())

        self.assertIs(recipe_for_placement(recipe, None), recipe)
