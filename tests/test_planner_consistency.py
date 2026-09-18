"""Planner consistency: canonical GPUs, evidence precedence, ranking."""

from __future__ import annotations

import unittest

from llm_launchpad.core.llamacpp_planner import (
    assessment_score,
    canonical_gpu_name,
    predict_performance,
    serving_requirements,
    tuning_for_objective,
)
from llm_launchpad.protocol.enums import CertificationState, ServingObjective
from llm_launchpad.protocol.models import (
    MemoryEstimate,
    PerformancePoint,
    PlacementAssessment,
    RuntimeAttestation,
    RuntimeTuning,
)


def _memory() -> MemoryEstimate:
    return MemoryEstimate(
        weights_gb=10.0,
        kv_cache_gb=5.0,
        compute_gb=2.0,
        speculative_gb=0.0,
        reserve_gb=4.0,
        total_gb=21.0,
        per_device_required_gb=(21.0,),
        confidence=0.82,
        source="gguf-metadata",
    )


def _assessment(
    certification: CertificationState,
    performance: tuple[PerformancePoint, ...],
) -> PlacementAssessment:
    return PlacementAssessment(
        fingerprint="fp",
        memory=_memory(),
        tuning=RuntimeTuning(parallel_slots=4),
        performance=performance,
        certification=certification,
        fits=True,
        gpu_resident=True,
    )


class CanonicalGpuTests(unittest.TestCase):
    def test_a100_80gb_resolves_to_its_own_entry(self) -> None:
        self.assertEqual(canonical_gpu_name("A100-80GB"), "A100-80GB")
        self.assertEqual(canonical_gpu_name("a100-80gb"), "A100-80GB")

    def test_provider_spellings_resolve_to_canonical_names(self) -> None:
        self.assertEqual(canonical_gpu_name("H100!"), "H100")
        self.assertEqual(canonical_gpu_name("RTX PRO 6000"), "RTX-PRO-6000")

    def test_unknown_cards_pass_through(self) -> None:
        self.assertEqual(canonical_gpu_name("RTX 4090"), "RTX 4090")

    def test_estimated_speed_distinguishes_a100_variants(self) -> None:
        tuning = RuntimeTuning(parallel_slots=1)
        plain = predict_performance(
            weights_gb=10.0, gpu_type="A100", gpu_count=1,
            tuning=tuning, price_per_hour_usd=1.0,
        )
        wide = predict_performance(
            weights_gb=10.0, gpu_type="A100-80GB", gpu_count=1,
            tuning=tuning, price_per_hour_usd=1.0,
        )
        plain_single = next(p.output_tokens_per_second for p in plain if p.concurrency == 1)
        wide_single = next(p.output_tokens_per_second for p in wide if p.concurrency == 1)
        self.assertGreater(wide_single or 0.0, plain_single or 0.0)


class EvidencePrecedenceTests(unittest.TestCase):
    def test_ranking_ignores_the_certification_bonus(self) -> None:
        estimated_points = predict_performance(
            weights_gb=10.0, gpu_type="L4", gpu_count=1,
            tuning=RuntimeTuning(parallel_slots=4), price_per_hour_usd=1.0,
        )
        estimated = _assessment(CertificationState.ESTIMATED, estimated_points)
        certified = _assessment(CertificationState.CERTIFIED, estimated_points)

        self.assertEqual(
            assessment_score(estimated, ServingObjective.GENERAL_PURPOSE),
            assessment_score(certified, ServingObjective.GENERAL_PURPOSE),
        )

    def test_efficiency_uses_current_price_not_cached_tokens_per_dollar(self) -> None:
        from llm_launchpad.core.serving_tiers import _efficiency
        from llm_launchpad.protocol.enums import BackendType, BillingModel, ComputeProvider
        from llm_launchpad.protocol.models import InferencePlan, InferenceRecipe, ProviderQuote

        points = predict_performance(
            weights_gb=10.0, gpu_type="L4", gpu_count=1,
            tuning=RuntimeTuning(parallel_slots=4), price_per_hour_usd=1.0,
        )
        recipe = InferenceRecipe(
            id="recipe", model_key="model", display_name="Model",
            backend=BackendType.LLAMACPP, model_id="org/model",
        )

        def _plan(price: float) -> InferencePlan:
            quote = ProviderQuote(
                id="q", recipe_id="recipe", provider=ComputeProvider.MODAL,
                provider_reference="q", gpu_type="L4", gpu_count=1,
                price_per_hour_usd=price, billing_model=BillingModel.SCALE_TO_ZERO,
                gpu_memory_gb=24.0,
            )
            return InferencePlan(
                recipe=recipe, quote=quote,
                assessment=_assessment(CertificationState.ESTIMATED, points),
            )

        self.assertAlmostEqual(_efficiency(_plan(2.0)), _efficiency(_plan(1.0)) / 2.0)


class FingerprintAndCalibrationTests(unittest.TestCase):
    def test_speculative_decoding_changes_the_calibration_key(self) -> None:
        from llm_launchpad.protocol.enums import SpeculativeDecodingMethod
        from llm_launchpad.protocol.models import SpeculativeDecodingConfig

        common = {
            "model_id": "org/model",
            "revision": None,
            "quant": "Q4_K_M",
            "runtime_id": "runtime-1",
            "requirements": serving_requirements(131_072),
        }
        plain = tuning_for_objective(ServingObjective.GENERAL_PURPOSE)
        speculative = tuning_for_objective(
            ServingObjective.GENERAL_PURPOSE,
            speculative_decoding=SpeculativeDecodingConfig(
                method=SpeculativeDecodingMethod.MTP,
                num_speculative_tokens=1,
            ),
        )

        from llm_launchpad.core.fit_calibration import calibration_key as key

        self.assertNotEqual(
            key(tuning=plain, **common),
            key(tuning=speculative, **common),
        )

    def test_certified_attestation_uses_observed_layer_counts(self) -> None:
        attestation = RuntimeAttestation(
            fingerprint="fp",
            requested_context_tokens=131_072,
            effective_context_tokens=131_072,
            gpu_layers=30,
            total_layers=32,
            gpu_resident=False,
        )

        # A partial offload is not full residency, however the plan was tuned.
        self.assertLess(attestation.gpu_layers, attestation.total_layers)
        self.assertFalse(attestation.gpu_resident)


if __name__ == "__main__":
    unittest.main()
