"""Step two should ask a question people can answer, not name hardware."""

from __future__ import annotations

import unittest

from llm_launchpad.core.llamacpp_planner import predict_performance
from llm_launchpad.core.serving_tiers import (
    BALANCED,
    ECONOMY,
    FASTEST,
    SAVER,
    reduced_quality_note,
    reduced_quality_plan_note,
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
    quant: str | None = None,
) -> InferencePlan:
    recipe = InferenceRecipe(
        id="recipe",
        model_key="model",
        display_name="Model",
        quant=quant,
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
        # Both axes are always named, so nothing is silently favourable.
        for key in (ECONOMY, FASTEST):
            self.assertEqual((by_key[key].tradeoff or "").count(","), 1)

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


    def test_a_bad_deal_is_shown_with_its_absent_benefit_named(self) -> None:
        # A flank that is much slower and no cheaper is still offered: hiding
        # it would remove the reader's ability to check the recommendation.
        # It has to read as the bad deal it is instead.
        plans = [
            _plan("barely-cheaper", price=14.99, single_tps=10.0, aggregate_tps=12.0),
            _plan("value", price=15.16, single_tps=28.0, aggregate_tps=34.0),
            _plan("fast", price=31.92, single_tps=50.0, aggregate_tps=60.0),
        ]

        tiers = serving_tiers(plans)
        by_id = {tier.plan.quote.id: tier for tier in tiers}

        self.assertIn("barely-cheaper", by_id)
        self.assertIn("slower", by_id["barely-cheaper"].tradeoff or "")
        self.assertIn("no cheaper", by_id["barely-cheaper"].tradeoff or "")

    def test_a_flank_that_is_barely_faster_is_still_offered(self) -> None:
        plans = [
            _plan("cheap", price=1.0, single_tps=40.0, aggregate_tps=45.0),
            _plan("value", price=2.0, single_tps=100.0, aggregate_tps=300.0),
            _plan("barely-faster", price=9.0, single_tps=102.0, aggregate_tps=104.0),
        ]

        tiers = serving_tiers(plans)
        by_id = {tier.plan.quote.id: tier for tier in tiers}

        self.assertIn("barely-faster", by_id)
        self.assertIn("the price", by_id["barely-faster"].tradeoff or "")

    def test_a_genuinely_cheaper_flank_is_still_offered(self) -> None:
        tiers = serving_tiers(self._frontier())

        self.assertIn(ECONOMY, {tier.key for tier in tiers})


if __name__ == "__main__":
    unittest.main()


class EstimatedTopologyTierTests(unittest.TestCase):
    """Estimated multi-GPU rows must not win on speed they cannot deliver."""

    def _estimated_plan(self, quote_id: str, *, gpu_count: int, price: float, gpu_type: str) -> InferencePlan:
        tuning = RuntimeTuning(parallel_slots=4)
        performance = predict_performance(
            weights_gb=40.0,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            tuning=tuning,
            price_per_hour_usd=price,
        )
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
            provider=ComputeProvider.VAST if gpu_count > 1 else ComputeProvider.MODAL,
            provider_reference=quote_id,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            price_per_hour_usd=price,
            billing_model=BillingModel.PROVISIONED,
            gpu_memory_gb=24.0,
        )
        assessment = PlacementAssessment(
            fingerprint=quote_id,
            memory=_memory(),
            tuning=tuning,
            performance=performance,
            certification=CertificationState.ESTIMATED,
            fits=True,
            gpu_resident=True,
        )
        return InferencePlan(recipe=recipe, quote=quote, assessment=assessment)

    def test_cheap_multi_gpu_estimate_does_not_win_the_fastest_tier(self) -> None:
        cheap_four = self._estimated_plan("vast-4x", gpu_count=4, price=0.90, gpu_type="RTX 3060")
        single_fast = self._estimated_plan("modal-1x", gpu_count=1, price=4.00, gpu_type="H100")

        tiers = {tier.key: tier for tier in serving_tiers([cheap_four, single_fast])}

        # The cheap 4x row is the cheapest and should say so -- and nothing
        # more. Every speed-ranked role belongs to the card that is actually
        # faster. Under the old 1.62x-per-GPU estimate the 4x row took the
        # value tier on throughput-per-dollar it could not deliver.
        self.assertEqual(tiers[ECONOMY].plan.quote.id, "vast-4x")
        speed_roles = [key for key in (BALANCED, FASTEST) if key in tiers]
        self.assertTrue(speed_roles)
        for key in speed_roles:
            self.assertEqual(tiers[key].plan.quote.id, "modal-1x")



class QuantQualityFloorTests(unittest.TestCase):
    """Price and speed may not buy themselves a smaller quantization."""

    def _pair(self, *, small_price: float) -> list[InferencePlan]:
        """A 4-bit frontier plus a 2-bit build that is cheaper and faster.

        Fewer bytes per weight is both cheaper to hold and faster to read, so
        the small build wins every axis the tier picker ranks on. That is
        precisely the sweep this floor exists to stop.
        """

        return [
            _plan("q4-cheap", price=3.03, single_tps=60.0, aggregate_tps=70.0, quant="UD-Q4_K_XL"),
            _plan("q4-fast", price=6.25, single_tps=115.0, aggregate_tps=130.0, quant="UD-Q4_K_XL"),
            _plan("q2-cheap", price=small_price, single_tps=120.0, aggregate_tps=140.0, quant="UD-Q2_K_XL"),
        ]

    def test_every_serving_tier_stays_at_the_best_available_width(self) -> None:
        tiers = serving_tiers(self._pair(small_price=1.00))

        serving = [tier for tier in tiers if tier.key != SAVER]
        self.assertTrue(serving)
        for tier in serving:
            self.assertEqual(tier.plan.recipe.quant, "UD-Q4_K_XL")

    def test_a_materially_cheaper_small_build_is_offered_and_labelled(self) -> None:
        # 1.00 against a 3.03 floor is a 67% discount: a real decision.
        tiers = {tier.key: tier for tier in serving_tiers(self._pair(small_price=1.00))}

        self.assertIn(SAVER, tiers)
        saver = tiers[SAVER]
        self.assertEqual(saver.plan.recipe.quant, "UD-Q2_K_XL")
        self.assertIn("2-bit", saver.tradeoff or "")
        self.assertFalse(saver.is_recommended)

    def test_the_saver_goes_last_rather_than_leading_on_price(self) -> None:
        tiers = serving_tiers(self._pair(small_price=1.00))

        # It is an opt-out from the quality floor, not the way into the list,
        # so it does not take the top row just for being cheapest.
        self.assertEqual(tiers[-1].key, SAVER)

    def test_a_small_build_on_the_same_hardware_is_not_offered(self) -> None:
        # The compact models put both widths on one RTX-PRO-6000. Serving 2-bit
        # weights to save nothing is pure loss, so the option should not exist.
        tiers = serving_tiers(self._pair(small_price=3.03))

        self.assertNotIn(SAVER, {tier.key for tier in tiers})

    def test_a_small_discount_does_not_buy_a_smaller_quantization(self) -> None:
        # 17% off, the real Qwen3-Next-80B gap. Under the bar, so no offer.
        tiers = serving_tiers(self._pair(small_price=2.50))

        self.assertNotIn(SAVER, {tier.key for tier in tiers})

    def test_a_model_that_only_fits_small_is_served_and_disclosed(self) -> None:
        # Kimi K3 is 2-bit or nothing. Offering it is right; offering it
        # silently is the failure this note exists to prevent.
        plans = [
            _plan("q2-cheap", price=34.65, single_tps=3.8, aggregate_tps=10.4, quant="UD-Q2_K_XL"),
            _plan("q2-fast", price=37.50, single_tps=5.6, aggregate_tps=15.3, quant="UD-Q2_K_XL"),
        ]

        tiers = serving_tiers(plans)
        note = reduced_quality_note(tiers)

        self.assertTrue(tiers)
        self.assertIsNotNone(note)
        self.assertIn("2-bit", note or "")
        self.assertIn("4-bit baseline", note or "")

    def test_no_note_when_the_floor_is_met(self) -> None:
        self.assertIsNone(reduced_quality_note(serving_tiers(self._pair(small_price=1.00))))

    def test_widths_spelled_differently_are_one_quality_level(self) -> None:
        # Q4_K_M and UD-Q4_K_XL are the same floor with two spellings; reading
        # them as rival quality levels would strand half the frontier.
        plans = [
            _plan("xl", price=6.25, single_tps=115.0, aggregate_tps=130.0, quant="UD-Q4_K_XL"),
            _plan("km", price=3.03, single_tps=60.0, aggregate_tps=70.0, quant="Q4_K_M"),
        ]

        tiers = serving_tiers(plans)

        self.assertNotIn(SAVER, {tier.key for tier in tiers})
        self.assertEqual({tier.plan.quote.id for tier in tiers}, {"xl", "km"})

    def test_an_unlabelled_frontier_is_unchanged(self) -> None:
        # Curated recipes carry no quant. They must not be read as degraded.
        plans = [
            _plan("a", price=1.95, single_tps=55.0, aggregate_tps=60.0),
            _plan("b", price=4.00, single_tps=110.0, aggregate_tps=130.0),
        ]

        tiers = serving_tiers(plans)

        self.assertNotIn(SAVER, {tier.key for tier in tiers})
        self.assertIsNone(reduced_quality_note(tiers))


class MergedRoleTests(unittest.TestCase):
    """A role whose placement another row already claimed is named, not lost."""

    def test_a_placement_winning_value_and_speed_says_it_is_also_fastest(self) -> None:
        # The common shape: one B200 is both the best tokens-per-dollar and
        # the fastest thing available. Balanced claims the row first, and the
        # Fastest label used to vanish without trace -- sending the reader to
        # the full list after a faster placement that does not exist.
        plans = [
            _plan("cheap", price=4.95, single_tps=27.0, aggregate_tps=30.0),
            _plan("big", price=6.25, single_tps=41.0, aggregate_tps=300.0),
        ]

        tiers = {tier.key: tier for tier in serving_tiers(plans)}

        self.assertNotIn(FASTEST, tiers)
        self.assertEqual(tiers[BALANCED].also, (FASTEST,))
        self.assertIn("also the fastest", tiers[BALANCED].tradeoff or "")

    def test_the_merged_role_composes_with_the_ratio_it_trades_on(self) -> None:
        # Cheapest and fastest in one placement, with the value tier elsewhere:
        # the row is not the baseline, so it carries both clauses.
        plans = [
            _plan("quick", price=1.0, single_tps=100.0, aggregate_tps=100.0),
            _plan("value", price=2.0, single_tps=50.0, aggregate_tps=300.0),
        ]

        tiers = {tier.key: tier for tier in serving_tiers(plans)}

        self.assertEqual(tiers[ECONOMY].also, (FASTEST,))
        tradeoff = tiers[ECONOMY].tradeoff or ""
        self.assertIn("also the fastest", tradeoff)
        self.assertIn("cheaper", tradeoff)

    def test_being_the_best_value_is_left_to_the_recommendation_marker(self) -> None:
        # Cheapest and best value in one placement. "also the best value" and
        # "recommended" are the same claim; the row should not make it twice.
        plans = [
            _plan("one", price=3.03, single_tps=72.0, aggregate_tps=400.0),
            _plan("fast", price=6.25, single_tps=136.0, aggregate_tps=150.0),
        ]

        tiers = {tier.key: tier for tier in serving_tiers(plans)}

        self.assertNotIn(BALANCED, tiers)
        self.assertEqual(tiers[ECONOMY].also, (BALANCED,))
        self.assertTrue(tiers[ECONOMY].is_recommended)
        self.assertIsNone(tiers[ECONOMY].tradeoff)

    def test_nothing_is_claimed_when_each_role_wins_its_own_placement(self) -> None:
        tiers = serving_tiers(
            [
                _plan("cheap", price=1.0, single_tps=30.0, aggregate_tps=32.0),
                _plan("value", price=2.0, single_tps=60.0, aggregate_tps=400.0),
                _plan("fast", price=8.0, single_tps=200.0, aggregate_tps=210.0),
            ]
        )

        self.assertEqual(len(tiers), 3)
        for tier in tiers:
            with self.subTest(tier=tier.key):
                self.assertEqual(tier.also, ())

    def test_one_placement_winning_everything_stays_one_row(self) -> None:
        tiers = serving_tiers([_plan("only", price=2.0, single_tps=90.0, aggregate_tps=95.0)])

        self.assertEqual(len(tiers), 1)
        self.assertEqual(tiers[0].also, (BALANCED, FASTEST))
        # Balanced stays unsaid, so the clause names the one role that is not
        # already carried by the recommendation marker.
        self.assertEqual(tiers[0].tradeoff, "also the fastest")

    def test_a_superlative_is_not_claimed_over_a_faster_saver_row(self) -> None:
        # Observed on screen for GLM 5.3 Flash: the recommended row read
        # "also the fastest ~12 tok/s" directly above a Saver row reading
        # "~30 tok/s · 2.5x faster". The roles are decided across the primary
        # bit width alone, but they are *read* against the whole list, so an
        # unqualified superlative is disproved by the next line down.
        plans = [
            _plan("q8", price=18.75, single_tps=12.0, aggregate_tps=33.0, quant="Q8_0"),
            _plan("q2", price=6.25, single_tps=30.0, aggregate_tps=80.0, quant="UD-Q2_K_XL"),
        ]

        tiers = {tier.key: tier for tier in serving_tiers(plans)}

        self.assertIn(SAVER, tiers)
        claim = tiers[ECONOMY].tradeoff or ""
        self.assertIn("also the fastest at 8-bit", claim)
        # The saver row is the faster one, and says so.
        self.assertIn("faster", tiers[SAVER].tradeoff or "")

    def test_an_unbeaten_superlative_is_still_claimed_plainly(self) -> None:
        # The saver is cheaper but slower, so "also the fastest" holds across
        # the rendered list and must not be hedged into uselessness.
        plans = [
            _plan("q8", price=8.00, single_tps=90.0, aggregate_tps=300.0, quant="Q8_0"),
            _plan("q2", price=1.00, single_tps=24.0, aggregate_tps=40.0, quant="UD-Q2_K_XL"),
        ]

        tiers = {tier.key: tier for tier in serving_tiers(plans)}

        self.assertIn(SAVER, tiers)
        self.assertIn("also the fastest", tiers[ECONOMY].tradeoff or "")
        self.assertNotIn("8-bit", tiers[ECONOMY].tradeoff or "")

    def test_a_saver_never_claims_a_frontier_role(self) -> None:
        plans = [
            _plan("q4", price=4.00, single_tps=40.0, aggregate_tps=300.0, quant="UD-Q4_K_XL"),
            _plan("q2", price=1.00, single_tps=90.0, aggregate_tps=95.0, quant="UD-Q2_K_XL"),
        ]

        tiers = {tier.key: tier for tier in serving_tiers(plans)}

        self.assertEqual(tiers[SAVER].also, ())


class SinglePlacementDisclosureTests(unittest.TestCase):
    """One live placement has no tiers, and is where the floor most often bites."""

    def test_a_lone_reduced_placement_is_still_explained(self) -> None:
        # Kimi K3 on this account resolves to exactly one placement, so the
        # screen lists it directly instead of collapsing it into tiers. The
        # note used to be read off tiers alone, which left the one model that
        # needed the sentence without it.
        plans = [_plan("only", price=37.5, single_tps=5.6, aggregate_tps=15.3, quant="UD-Q2_K_XL")]

        note = reduced_quality_plan_note(plans)

        self.assertIsNotNone(note)
        self.assertIn("2-bit", note or "")

    def test_a_lone_placement_that_meets_the_floor_says_nothing(self) -> None:
        plans = [_plan("only", price=3.03, single_tps=72.0, aggregate_tps=80.0, quant="UD-Q4_K_XL")]

        self.assertIsNone(reduced_quality_plan_note(plans))

    def test_a_reachable_wider_build_is_not_a_floor(self) -> None:
        # Both widths are placeable, so nothing was forced on the reader.
        plans = [
            _plan("q4", price=6.25, single_tps=41.0, aggregate_tps=50.0, quant="UD-Q4_K_XL"),
            _plan("q2", price=2.50, single_tps=90.0, aggregate_tps=95.0, quant="UD-Q2_K_XL"),
        ]

        self.assertIsNone(reduced_quality_plan_note(plans))

    def test_an_ineligible_placement_cannot_lift_the_floor(self) -> None:
        # A 4-bit plan that cannot hold the context is not an option, so the
        # reader is still on 2-bit whether or not the row exists.
        plans = [
            _plan("q4", price=6.25, single_tps=41.0, aggregate_tps=50.0, quant="UD-Q4_K_XL", fits=False),
            _plan("q2", price=2.50, single_tps=90.0, aggregate_tps=95.0, quant="UD-Q2_K_XL"),
        ]

        self.assertIsNotNone(reduced_quality_plan_note(plans))

    def test_nothing_placeable_yields_no_opinion(self) -> None:
        self.assertIsNone(reduced_quality_plan_note([]))
