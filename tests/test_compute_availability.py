from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm_launchpad.core.compute_availability import (
    aggregate_compute_availability,
    canonical_gpu_identity,
    display_gpu_type,
    load_compute_availability,
    plans_for_compute_profile,
)
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


if __name__ == "__main__":
    unittest.main()
