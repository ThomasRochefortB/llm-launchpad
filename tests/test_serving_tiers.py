"""Step two should ask a question people can answer, not name hardware."""

from __future__ import annotations

import unittest

from llm_launchpad.core.serving_tiers import (
    BALANCED,
    ECONOMY,
    FASTEST,
    serving_tiers,
)
from llm_launchpad.protocol.enums import (
    BackendType,
    BillingModel,
    CertificationState,
    ComputeProvider,
    ServingObjective,
)
from llm_launchpad.protocol.models import (
    InferencePlan,
    InferenceRecipe,
    MemoryEstimate,
    PerformancePoint,
    PlacementAssessment,
    ProviderQuote,
    RuntimeTuning,
)


def _memory(total: float = 60.0) -> MemoryEstimate:
    return MemoryEstimate(
        weights_gb=10.0,
        kv_cache_gb=45.0,
        compute_gb=1.5,
        speculative_gb=0.0,
        reserve_gb=3.5,
        total_gb=total,
        per_device_required_gb=(total,),
        confidence=0.82,
        source="gguf-metadata",
    )


def _plan(
    quote_id: str,
    *,
    price: float,
    single_tps: float,
    aggregate_tps: float,
    fits: bool = True,
    gpu_resident: bool = True,
    measured: bool = False,
) -> InferencePlan:
    recipe = InferenceRecipe(
        id="recipe",
        model_key="model",
        display_name="Model",
        backend=BackendType.LLAMACPP,
        model_id="org/model",
    )
    quote = ProviderQuote(
        id=quote_id,
        recipe_id="recipe",
        provider=ComputeProvider.MODAL,
        provider_reference=quote_id,
        gpu_type="L4",
        gpu_count=1,
        price_per_hour_usd=price,
        billing_model=BillingModel.SCALE_TO_ZERO,
        gpu_memory_gb=24.0,
    )
    performance = (
        PerformancePoint(
            prompt_tokens=512,
            output_tokens=128,
            concurrency=1,
            output_tokens_per_second=single_tps,
            aggregate_output_tokens_per_second=single_tps,
            output_tokens_per_dollar=single_tps * 3600 / max(0.01, price) / 1_000,
            measured=measured,
        ),
        PerformancePoint(
            prompt_tokens=512,
            output_tokens=128,
            concurrency=4,
            output_tokens_per_second=single_tps,
            aggregate_output_tokens_per_second=aggregate_tps,
            output_tokens_per_dollar=aggregate_tps * 3600 / max(0.01, price) / 1_000,
            measured=measured,
        ),
    )
    assessment = PlacementAssessment(
        fingerprint=quote_id,
        memory=_memory(),
        tuning=RuntimeTuning(),
        performance=performance,
        certification=CertificationState.ESTIMATED,
        fits=fits,
        gpu_resident=gpu_resident,
    )
    return InferencePlan(recipe=recipe, quote=quote, assessment=assessment)


class ServingTierTests(unittest.TestCase):
    def _frontier(self) -> list[InferencePlan]:
        return [
            _plan("cheap", price=1.95, single_tps=55.0, aggregate_tps=60.0),
            _plan("mid", price=2.50, single_tps=95.0, aggregate_tps=110.0),
            _plan("fast", price=6.25, single_tps=180.0, aggregate_tps=190.0),
        ]

    def test_the_frontier_collapses_to_cheapest_best_value_and_fastest(self) -> None:
        tiers = serving_tiers(self._frontier())

        self.assertEqual([tier.key for tier in tiers], [ECONOMY, BALANCED, FASTEST])
        self.assertEqual(tiers[0].plan.quote.id, "cheap")
        self.assertEqual(tiers[2].plan.quote.id, "fast")

    def test_exactly_one_tier_is_recommended(self) -> None:
        tiers = serving_tiers(self._frontier())

        recommended = [tier for tier in tiers if tier.is_recommended]
        self.assertEqual(len(recommended), 1)
        self.assertEqual(recommended[0].key, BALANCED)

    def test_a_placement_that_cannot_hold_full_context_is_never_offered(self) -> None:
        plans = self._frontier()
        plans.append(_plan("tiny", price=0.40, single_tps=20.0, aggregate_tps=22.0, fits=False))

        tiers = serving_tiers(plans)

        self.assertNotIn("tiny", {tier.plan.quote.id for tier in tiers})

    def test_a_cpu_offloaded_placement_is_never_offered(self) -> None:
        plans = self._frontier()
        plans.append(
            _plan("spill", price=0.50, single_tps=25.0, aggregate_tps=26.0, gpu_resident=False)
        )

        tiers = serving_tiers(plans)

        self.assertNotIn("spill", {tier.plan.quote.id for tier in tiers})

    def test_variety_is_not_manufactured_when_one_placement_wins_twice(self) -> None:
        # Cheapest and fastest being the same placement means two honest
        # options, not three padded ones.
        single = [_plan("only", price=2.0, single_tps=90.0, aggregate_tps=95.0)]

        tiers = serving_tiers(single)

        self.assertEqual(len(tiers), 1)
        self.assertTrue(tiers[0].is_recommended)

    def test_the_objective_changes_which_placement_counts_as_fastest(self) -> None:
        # One machine wins on single-stream speed, the other on batch: which
        # is "fastest" is a property of the objective, not of the hardware.
        plans = [
            _plan("cheap", price=1.0, single_tps=30.0, aggregate_tps=35.0),
            _plan("value", price=2.0, single_tps=100.0, aggregate_tps=300.0),
            _plan("single-fast", price=8.0, single_tps=250.0, aggregate_tps=260.0),
            _plan("batch-fast", price=8.0, single_tps=90.0, aggregate_tps=500.0),
        ]

        def fastest_plan(objective: ServingObjective) -> str:
            tiers = serving_tiers(plans, objective)
            return next(tier.plan.quote.id for tier in tiers if tier.key == FASTEST)

        self.assertEqual(fastest_plan(ServingObjective.INTERACTIVE), "single-fast")
        self.assertEqual(fastest_plan(ServingObjective.THROUGHPUT), "batch-fast")

    def test_the_tradeoff_is_stated_relative_to_the_recommendation(self) -> None:
        tiers = serving_tiers(self._frontier())
        by_key = {tier.key: tier for tier in tiers}

        self.assertIsNone(by_key[BALANCED].tradeoff)
        self.assertIn("slower", by_key[ECONOMY].tradeoff or "")
        self.assertIn("cheaper", by_key[ECONOMY].tradeoff or "")
        self.assertIn("faster", by_key[FASTEST].tradeoff or "")
        self.assertIn("the price", by_key[FASTEST].tradeoff or "")

    def test_evidence_reports_whether_numbers_were_measured(self) -> None:
        estimated = serving_tiers(self._frontier())
        self.assertFalse(any(tier.measured for tier in estimated))

        measured_plans = [
            _plan("cheap", price=1.95, single_tps=55.0, aggregate_tps=60.0, measured=True)
        ]
        self.assertTrue(serving_tiers(measured_plans)[0].measured)

    def test_an_empty_frontier_offers_nothing(self) -> None:
        self.assertEqual(serving_tiers([]), ())
        self.assertEqual(
            serving_tiers([_plan("no", price=1.0, single_tps=1.0, aggregate_tps=1.0, fits=False)]),
            (),
        )


if __name__ == "__main__":
    unittest.main()
