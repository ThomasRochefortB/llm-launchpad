"""Offline scoring for estimate evaluation: pure functions, no live calls.

An evaluation run freezes what the planner believed *before* deployment and
pairs it with what the runtime observed. This module scores those pairs. It
never reads the live calibration or certificate caches: evaluation evidence
stays isolated until scoring is complete, so a predictor cannot be graded on
numbers it just learned.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..protocol.enums import OutcomeKind
from ..protocol.models import EvaluationRecord, PerformancePoint

ESTIMATOR_VERSION = "estimate-eval-v1"

# Outcomes that teach nothing about the estimate: the run never reached the
# point where the prediction could succeed or fail.
NON_INFORMATIVE_OUTCOMES = frozenset(
    {
        OutcomeKind.PROVISIONING_FAILED,
        OutcomeKind.CANCELLED,
        OutcomeKind.TELEMETRY_MISSING,
        OutcomeKind.UNKNOWN,
    }
)


@dataclass(frozen=True)
class ScoredRecord:
    """One evaluation record reduced to comparable errors."""

    record_id: str
    informative: bool
    fit_correct: bool | None
    per_device_errors_gb: tuple[float, ...]
    reserve_covered: bool | None
    single_relative_error: float | None
    aggregate_relative_error: float | None
    overpredicted: bool | None


@dataclass(frozen=True)
class EstimateScorecard:
    """Aggregate accuracy and decision quality across scored records."""

    records: int
    informative_records: int
    fit_accuracy: float | None
    false_accepts: int
    false_rejects: int
    memory_p50_error_gb: float | None
    memory_p90_error_gb: float | None
    reserve_coverage: float | None
    throughput_median_relative_error: float | None
    throughput_p90_relative_error: float | None
    overprediction_rate: float | None
    pairwise_ranking_accuracy: float | None
    recommendation_regret_usd_per_hour: float | None
    exclusions_by_outcome: dict[str, int]


def score_record(record: EvaluationRecord) -> ScoredRecord:
    """Score one frozen prediction against its independent observation."""

    informative = record.outcome not in NON_INFORMATIVE_OUTCOMES
    fit_correct: bool | None = None
    if informative and record.outcome in {
        OutcomeKind.SUCCESS,
        OutcomeKind.OUT_OF_MEMORY,
        OutcomeKind.RUNTIME_INCOMPATIBLE,
    }:
        # An incompatible runtime rejects for reasons outside memory, so only
        # the memory outcomes judge the fit prediction.
        if record.outcome == OutcomeKind.SUCCESS:
            fit_correct = record.prediction.predicted_fits
        elif record.outcome == OutcomeKind.OUT_OF_MEMORY:
            fit_correct = not record.prediction.predicted_fits

    per_device_errors: list[float] = []
    reserve_covered: bool | None = None
    observed_peak = _observed_peak_per_device_gb(record)
    predicted = record.prediction.calibrated_per_device_required_gb
    if observed_peak and predicted and informative:
        width = min(len(observed_peak), len(predicted))
        per_device_errors = [
            observed_peak[index] - predicted[index] for index in range(width)
        ]
        reserve_covered = all(error <= 0.0 for error in per_device_errors)

    single_error, single_over = _relative_error(
        record.prediction.predicted_single_tps,
        _observed_single_tps(record.performance),
    )
    aggregate_error, aggregate_over = _relative_error(
        record.prediction.predicted_aggregate_tps,
        _observed_aggregate_tps(record.performance),
    )
    overpredicted: bool | None = None
    if single_over is not None or aggregate_over is not None:
        overpredicted = bool(single_over or aggregate_over)

    return ScoredRecord(
        record_id=record.record_id,
        informative=informative,
        fit_correct=fit_correct,
        per_device_errors_gb=tuple(per_device_errors),
        reserve_covered=reserve_covered,
        single_relative_error=single_error,
        aggregate_relative_error=aggregate_error,
        overpredicted=overpredicted,
    )


def build_scorecard(records: tuple[EvaluationRecord, ...]) -> EstimateScorecard:
    """Aggregate scored records into accuracy and decision-quality metrics."""

    scored = [score_record(record) for record in records]
    informative = [row for row in scored if row.informative]
    fit_judged = [row for row in informative if row.fit_correct is not None]
    false_accepts = sum(
        1
        for row, record in zip(scored, records, strict=True)
        if row.fit_correct is False and record.prediction.predicted_fits
    )
    false_rejects = sum(
        1
        for row, record in zip(scored, records, strict=True)
        if row.fit_correct is False and not record.prediction.predicted_fits
    )
    memory_errors = sorted(
        abs(error) for row in informative for error in row.per_device_errors_gb
    )
    reserve_judged = [row for row in informative if row.reserve_covered is not None]
    throughput_errors = sorted(
        error
        for row in informative
        for error in (row.single_relative_error, row.aggregate_relative_error)
        if error is not None
    )
    over_judged = [row for row in informative if row.overpredicted is not None]
    exclusions: dict[str, int] = {}
    for record in records:
        if record.outcome in NON_INFORMATIVE_OUTCOMES:
            key = record.outcome.value
            exclusions[key] = exclusions.get(key, 0) + 1
    return EstimateScorecard(
        records=len(records),
        informative_records=len(informative),
        fit_accuracy=(
            sum(1 for row in fit_judged if row.fit_correct) / len(fit_judged)
            if fit_judged
            else None
        ),
        false_accepts=false_accepts,
        false_rejects=false_rejects,
        memory_p50_error_gb=_percentile(memory_errors, 0.50),
        memory_p90_error_gb=_percentile(memory_errors, 0.90),
        reserve_coverage=(
            sum(1 for row in reserve_judged if row.reserve_covered) / len(reserve_judged)
            if reserve_judged
            else None
        ),
        throughput_median_relative_error=_percentile(throughput_errors, 0.50),
        throughput_p90_relative_error=_percentile(throughput_errors, 0.90),
        overprediction_rate=(
            sum(1 for row in over_judged if row.overpredicted) / len(over_judged)
            if over_judged
            else None
        ),
        pairwise_ranking_accuracy=_pairwise_ranking_accuracy(records),
        recommendation_regret_usd_per_hour=_recommendation_regret(records),
        exclusions_by_outcome=exclusions,
    )


def _observed_peak_per_device_gb(
    record: EvaluationRecord,
) -> tuple[float, ...]:
    peaks: list[float] = []
    for observation in record.memory_observations:
        if observation.per_device_used_gb:
            peaks.append(max(observation.per_device_used_gb))
    if not peaks:
        return ()
    peak = max(peaks)
    width = max(
        (len(observation.per_device_used_gb) for observation in record.memory_observations if observation.per_device_used_gb),
        default=1,
    )
    return tuple([peak] * width)


def _observed_single_tps(
    performance: tuple[PerformancePoint, ...],
) -> float | None:
    values = [
        point.output_tokens_per_second or 0.0
        for point in performance
        if point.measured and point.concurrency == 1
        and (point.actual_output_tokens or 0) > 0
    ]
    return max(values) if values else None


def _observed_aggregate_tps(
    performance: tuple[PerformancePoint, ...],
) -> float | None:
    values = [
        point.aggregate_output_tokens_per_second or 0.0
        for point in performance
        if point.measured and (point.actual_output_tokens or 0) > 0
    ]
    return max(values) if values else None


def _relative_error(
    predicted: float | None,
    observed: float | None,
) -> tuple[float | None, bool | None]:
    if predicted is None or observed is None or observed <= 0:
        return None, None
    return abs(predicted - observed) / observed, predicted > observed


def _percentile(values: list[float], rank: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(rank * (len(ordered) - 1))))
    return ordered[index]


def _pairwise_ranking_accuracy(
    records: tuple[EvaluationRecord, ...],
) -> float | None:
    """Fraction of GPU pairs whose predicted order matches the observed one."""

    comparable: list[tuple[float, float]] = []
    for record in records:
        if record.outcome != OutcomeKind.SUCCESS:
            continue
        predicted = record.prediction.predicted_aggregate_tps
        observed = _observed_aggregate_tps(record.performance)
        if predicted is not None and observed is not None:
            comparable.append((predicted, observed))
    if len(comparable) < 2:
        return None
    correct = 0
    total = 0
    for left in range(len(comparable)):
        for right in range(left + 1, len(comparable)):
            predicted_order = (comparable[left][0] > comparable[right][0]) - (
                comparable[left][0] < comparable[right][0]
            )
            observed_order = (comparable[left][1] > comparable[right][1]) - (
                comparable[left][1] < comparable[right][1]
            )
            if predicted_order == 0 or observed_order == 0:
                continue
            total += 1
            if predicted_order == observed_order:
                correct += 1
    return correct / total if total else None


def _recommendation_regret(
    records: tuple[EvaluationRecord, ...],
) -> float | None:
    """Extra hourly cost of recommended placements versus the best measured.

    Only successful records with a price participate. Unknown outcomes never
    enter the comparison as successes.
    """

    priced = [
        record
        for record in records
        if record.outcome == OutcomeKind.SUCCESS
        and record.price_per_hour_usd is not None
    ]
    if len(priced) < 2:
        return None
    best_price = min(
        record.price_per_hour_usd for record in priced if record.price_per_hour_usd is not None
    )
    recommended = [record for record in priced if record.prediction.recommended]
    if not recommended or best_price is None:
        return None
    paid = min(
        record.price_per_hour_usd for record in recommended if record.price_per_hour_usd is not None
    )
    if paid is None:
        return None
    return max(0.0, paid - best_price)
