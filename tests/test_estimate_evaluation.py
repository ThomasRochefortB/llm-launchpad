"""Estimate-evaluation scoring is offline, hermetic, and hand-checkable."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime

from llm_launchpad.core.estimate_evaluation import build_scorecard
from llm_launchpad.core.evaluation_store import (
    evaluation_run_dir,
    format_scorecard,
    read_evaluation_records,
    score_run_dir,
    write_evaluation_record,
)
from llm_launchpad.protocol.enums import (
    CachePolicy,
    EvidenceLevel,
    OutcomeKind,
)
from llm_launchpad.protocol.models import (
    CompatibilityEvidence,
    EvaluationRecord,
    MemoryObservation,
    PerformancePoint,
    PredictionSnapshot,
    RuntimeIdentity,
    WorkloadScenario,
)

from pathlib import Path
from tempfile import TemporaryDirectory


def _record(
    record_id: str,
    *,
    outcome: OutcomeKind,
    predicted_fits: bool,
    predicted_single: float | None = 20.0,
    predicted_aggregate: float | None = 40.0,
    predicted_per_device: tuple[float, ...] = (70.0,),
    observed_peak: float | None = 68.0,
    observed_single: float | None = 20.0,
    observed_aggregate: float | None = 40.0,
    price: float | None = 2.0,
    recommended: bool = False,
) -> EvaluationRecord:
    performance: tuple[PerformancePoint, ...] = ()
    if observed_single is not None or observed_aggregate is not None:
        performance = (
            PerformancePoint(
                prompt_tokens=512,
                output_tokens=128,
                concurrency=1,
                output_tokens_per_second=observed_single,
                aggregate_output_tokens_per_second=observed_single,
                error_rate=0.0,
                measured=True,
                actual_prompt_tokens=512,
                actual_output_tokens=128,
                sample_count=4,
                completion_reason="benchmark",
                cache_policy=CachePolicy.UNCACHED,
                evidence=EvidenceLevel.OBSERVED,
            ),
            PerformancePoint(
                prompt_tokens=512,
                output_tokens=128,
                concurrency=2,
                output_tokens_per_second=observed_single,
                aggregate_output_tokens_per_second=observed_aggregate,
                error_rate=0.0,
                measured=True,
                actual_prompt_tokens=1024,
                actual_output_tokens=256,
                sample_count=8,
                completion_reason="benchmark",
                cache_policy=CachePolicy.UNCACHED,
                evidence=EvidenceLevel.OBSERVED,
            ),
        )
    observations: tuple[MemoryObservation, ...] = ()
    if observed_peak is not None:
        observations = (
            MemoryObservation(
                per_device_used_gb=(observed_peak,),
                per_device_total_gb=(80.0,),
                phase="decode",
            ),
        )
    return EvaluationRecord(
        record_id=record_id,
        scenario=WorkloadScenario(
            id="short-batch",
            display_name="Short batch",
            prompt_tokens=512,
            output_tokens=128,
            concurrency=2,
            request_count=32,
        ),
        runtime=RuntimeIdentity(model_id="org/model", gpu_type="A100-80GB", gpu_count=1),
        prediction=PredictionSnapshot(
            estimator_version="estimate-eval-v1",
            raw_per_device_required_gb=predicted_per_device,
            calibrated_per_device_required_gb=predicted_per_device,
            predicted_fits=predicted_fits,
            predicted_single_tps=predicted_single,
            predicted_aggregate_tps=predicted_aggregate,
            recommended=recommended,
        ),
        outcome=outcome,
        compatibility=CompatibilityEvidence(),
        memory_observations=observations,
        performance=performance,
        price_per_hour_usd=price,
        created_at=datetime.now(UTC).isoformat(),
    )


class EstimateEvaluationTests(unittest.TestCase):
    def test_exact_predictions_score_zero_error(self) -> None:
        scorecard = build_scorecard((_record("a", outcome=OutcomeKind.SUCCESS, predicted_fits=True),))

        self.assertEqual(scorecard.fit_accuracy, 1.0)
        self.assertEqual(scorecard.false_accepts, 0)
        self.assertEqual(scorecard.false_rejects, 0)
        self.assertEqual(scorecard.memory_p50_error_gb, 2.0)
        self.assertEqual(scorecard.throughput_median_relative_error, 0.0)
        self.assertEqual(scorecard.overprediction_rate, 0.0)

    def test_false_accepts_and_rejects_are_distinguished(self) -> None:
        scorecard = build_scorecard(
            (
                _record("accept", outcome=OutcomeKind.OUT_OF_MEMORY, predicted_fits=True),
                _record("reject", outcome=OutcomeKind.SUCCESS, predicted_fits=False),
            )
        )

        self.assertEqual(scorecard.fit_accuracy, 0.0)
        self.assertEqual(scorecard.false_accepts, 1)
        self.assertEqual(scorecard.false_rejects, 1)

    def test_unknown_outcomes_are_excluded_not_counted_as_success(self) -> None:
        scorecard = build_scorecard(
            (
                _record("good", outcome=OutcomeKind.SUCCESS, predicted_fits=True),
                _record("cancelled", outcome=OutcomeKind.CANCELLED, predicted_fits=True),
                _record("no-telemetry", outcome=OutcomeKind.TELEMETRY_MISSING, predicted_fits=True),
            )
        )

        self.assertEqual(scorecard.informative_records, 1)
        self.assertEqual(scorecard.records, 3)
        self.assertEqual(scorecard.fit_accuracy, 1.0)
        self.assertEqual(scorecard.exclusions_by_outcome.get("cancelled"), 1)
        self.assertEqual(scorecard.exclusions_by_outcome.get("telemetry_missing"), 1)

    def test_overprediction_bias_is_reported(self) -> None:
        scorecard = build_scorecard(
            (_record("hot", outcome=OutcomeKind.SUCCESS, predicted_fits=True,
                      predicted_single=40.0, predicted_aggregate=80.0),)
        )

        self.assertEqual(scorecard.overprediction_rate, 1.0)
        self.assertEqual(scorecard.throughput_median_relative_error, 1.0)

    def test_pairwise_ranking_and_regret_hand_calculation(self) -> None:
        scorecard = build_scorecard(
            (
                _record("cheap", outcome=OutcomeKind.SUCCESS, predicted_fits=True,
                        predicted_aggregate=30.0, observed_aggregate=30.0,
                        price=1.0, recommended=False),
                _record("pricey", outcome=OutcomeKind.SUCCESS, predicted_fits=True,
                        predicted_aggregate=60.0, observed_aggregate=50.0,
                        price=4.0, recommended=True),
            )
        )

        self.assertEqual(scorecard.pairwise_ranking_accuracy, 1.0)
        # Recommended $4/hr against a measured $1/hr alternative: $3/hr regret.
        self.assertEqual(scorecard.recommendation_regret_usd_per_hour, 3.0)

    def test_run_dir_round_trip_reproduces_the_scorecard(self) -> None:
        records = (
            _record("a", outcome=OutcomeKind.SUCCESS, predicted_fits=True),
            _record("b", outcome=OutcomeKind.OUT_OF_MEMORY, predicted_fits=True),
        )
        expected = build_scorecard(records)

        with TemporaryDirectory() as directory:
            run_dir = evaluation_run_dir("pilot-1", base=Path(directory))
            for record in records:
                write_evaluation_record(record, run_dir)
            loaded = read_evaluation_records(run_dir)
            self.assertEqual([row.record_id for row in loaded], ["a", "b"])
            _, rescored, report_path = score_run_dir(run_dir)
            exists_inside = report_path.exists()

        self.assertEqual(rescored, expected)
        self.assertTrue(exists_inside)
        lines = format_scorecard(rescored)
        self.assertTrue(any("fit accuracy" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
