"""Versioned storage for estimate-evaluation records and reports.

Each run directory holds a manifest, one frozen prediction per candidate, the
observed evidence, and the scored report. Records are plain JSON so a pilot
can be inspected without the application installed.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..protocol.models import EvaluationRecord
from .config import SETTINGS_DIR
from .estimate_evaluation import ESTIMATOR_VERSION, EstimateScorecard, build_scorecard

EVALUATIONS_DIR = SETTINGS_DIR / "estimate_evaluations"
EVALUATION_SCHEMA_VERSION = 1


def evaluation_run_dir(run_id: str, *, base: Path | None = None) -> Path:
    """Return the directory for one evaluation run, creating nothing."""

    root = base if base is not None else EVALUATIONS_DIR
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in run_id).strip("-")
    return root / (safe or "evaluation")


def write_evaluation_record(record: EvaluationRecord, run_dir: Path) -> Path:
    """Persist one frozen prediction-plus-observation pair."""

    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"record-{record.record_id}.json"
    payload = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "estimator_version": ESTIMATOR_VERSION,
        "record": _jsonable(asdict(record)),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_evaluation_records(run_dir: Path) -> tuple[EvaluationRecord, ...]:
    """Load every record in a run directory, skipping corrupt files."""

    if not run_dir.exists():
        return ()
    records: list[EvaluationRecord] = []
    for path in sorted(run_dir.glob("record-*.json")):
        record = _record_from_dict(_read_json(path))
        if record is not None:
            records.append(record)
    return tuple(records)


def write_scorecard_report(
    run_dir: Path,
    records: tuple[EvaluationRecord, ...],
    scorecard: EstimateScorecard,
) -> Path:
    """Persist the scored report alongside the records it was computed from."""

    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "scorecard.json"
    payload = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "estimator_version": ESTIMATOR_VERSION,
        "scorecard": _jsonable(asdict(scorecard)),
        "record_ids": [record.record_id for record in records],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def score_run_dir(run_dir: Path) -> tuple[tuple[EvaluationRecord, ...], EstimateScorecard, Path]:
    """Load, score, and persist the report for one run directory."""

    records = read_evaluation_records(run_dir)
    scorecard = build_scorecard(records)
    path = write_scorecard_report(run_dir, records, scorecard)
    return records, scorecard, path


def format_scorecard(scorecard: EstimateScorecard) -> list[str]:
    """Render the scorecard as human-readable lines."""

    def _format(value: float | None, suffix: str = "") -> str:
        return f"{value:.3f}{suffix}" if value is not None else "n/a"

    return [
        f"records: {scorecard.informative_records}/{scorecard.records} informative",
        f"fit accuracy: {_format(scorecard.fit_accuracy)} "
        f"(false accepts={scorecard.false_accepts}, "
        f"false rejects={scorecard.false_rejects})",
        f"memory error p50/p90 GB: {_format(scorecard.memory_p50_error_gb)} / "
        f"{_format(scorecard.memory_p90_error_gb)}",
        f"reserve coverage: {_format(scorecard.reserve_coverage)}",
        f"throughput relative error median/p90: "
        f"{_format(scorecard.throughput_median_relative_error)} / "
        f"{_format(scorecard.throughput_p90_relative_error)}",
        f"overprediction rate: {_format(scorecard.overprediction_rate)}",
        f"pairwise ranking accuracy: {_format(scorecard.pairwise_ranking_accuracy)}",
        f"recommendation regret $/hr: {_format(scorecard.recommendation_regret_usd_per_hour)}",
        f"exclusions: {scorecard.exclusions_by_outcome or 'none'}",
    ]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _record_from_dict(raw: Any) -> EvaluationRecord | None:
    if not isinstance(raw, dict):
        return None
    payload = raw.get("record", raw)
    if not isinstance(payload, dict):
        return None
    try:
        from ..protocol.enums import (
            BackendType,
            CachePolicy,
            ComputeProvider,
            EvidenceLevel,
            MemoryObservationSource,
            OutcomeKind,
        )
        from ..protocol.models import (
            CompatibilityEvidence,
            MemoryObservation,
            PerformancePoint,
            PredictionSnapshot,
            RuntimeIdentity,
            WorkloadScenario,
        )

        scenario_raw = payload.get("scenario") or {}
        runtime_raw = payload.get("runtime") or {}
        prediction_raw = payload.get("prediction") or {}
        compatibility_raw = payload.get("compatibility") or {}
        scenario = WorkloadScenario(
            id=str(scenario_raw.get("id") or ""),
            display_name=str(scenario_raw.get("display_name") or ""),
            prompt_tokens=int(scenario_raw.get("prompt_tokens") or 0),
            output_tokens=int(scenario_raw.get("output_tokens") or 0),
            concurrency=max(1, int(scenario_raw.get("concurrency") or 1)),
            request_count=max(1, int(scenario_raw.get("request_count") or 1)),
            cache_policy=_enum_or(CachePolicy, scenario_raw.get("cache_policy"), CachePolicy.UNCACHED),
            min_samples_for_p95=int(scenario_raw.get("min_samples_for_p95") or 20),
            latency_target_seconds=_optional_float(scenario_raw.get("latency_target_seconds")),
            distinct_prompts=bool(scenario_raw.get("distinct_prompts", True)),
        )
        backend_raw = runtime_raw.get("backend")
        provider_raw = runtime_raw.get("provider")
        runtime = RuntimeIdentity(
            model_id=str(runtime_raw.get("model_id") or ""),
            revision=runtime_raw.get("revision"),
            quant=runtime_raw.get("quant"),
            runtime_id=runtime_raw.get("runtime_id"),
            backend=BackendType(backend_raw) if backend_raw else None,
            provider=ComputeProvider(provider_raw) if provider_raw else None,
            gpu_type=str(runtime_raw.get("gpu_type") or ""),
            gpu_count=max(1, int(runtime_raw.get("gpu_count") or 1)),
            server_args=tuple(runtime_raw.get("server_args") or ()),
            effective_flags=tuple(runtime_raw.get("effective_flags") or ()),
        )
        prediction = PredictionSnapshot(
            estimator_version=str(prediction_raw.get("estimator_version") or ""),
            raw_per_device_required_gb=tuple(
                float(value) for value in prediction_raw.get("raw_per_device_required_gb") or ()
            ),
            calibrated_per_device_required_gb=tuple(
                float(value) for value in prediction_raw.get("calibrated_per_device_required_gb") or ()
            ),
            predicted_fits=bool(prediction_raw.get("predicted_fits")),
            predicted_single_tps=_optional_float(prediction_raw.get("predicted_single_tps")),
            predicted_aggregate_tps=_optional_float(prediction_raw.get("predicted_aggregate_tps")),
            recommended=bool(prediction_raw.get("recommended")),
            recommendation_reason=prediction_raw.get("recommendation_reason"),
        )
        compatibility = CompatibilityEvidence(
            runtime_supported=_optional_evidence(compatibility_raw.get("runtime_supported")),
            memory_fit=_optional_evidence(compatibility_raw.get("memory_fit")),
            gpu_resident=_optional_evidence(compatibility_raw.get("gpu_resident")),
            context_executed_tokens=_optional_int(compatibility_raw.get("context_executed_tokens")),
            context_executed_evidence=_optional_evidence(compatibility_raw.get("context_executed_evidence")),
            concurrent_capacity=_optional_evidence(compatibility_raw.get("concurrent_capacity")),
            service_quality=_optional_evidence(compatibility_raw.get("service_quality")),
            detail=str(compatibility_raw.get("detail") or ""),
        )
        observations = tuple(
            MemoryObservation(
                per_device_used_gb=tuple(float(value) for value in (item.get("per_device_used_gb") or ())),
                per_device_total_gb=tuple(float(value) for value in (item.get("per_device_total_gb") or ())),
                source=_enum_or(MemoryObservationSource, item.get("source"), MemoryObservationSource.DEVICE_SAMPLE),
                phase=str(item.get("phase") or ""),
                margin_mib=_optional_int(item.get("margin_mib")),
                observed_at=item.get("observed_at"),
            )
            for item in (payload.get("memory_observations") or ())
            if isinstance(item, dict)
        )
        performance = tuple(
            PerformancePoint(
                prompt_tokens=int(item.get("prompt_tokens") or 0),
                output_tokens=int(item.get("output_tokens") or 0),
                concurrency=max(1, int(item.get("concurrency") or 1)),
                prompt_tokens_per_second=_optional_float(item.get("prompt_tokens_per_second")),
                output_tokens_per_second=_optional_float(item.get("output_tokens_per_second")),
                aggregate_output_tokens_per_second=_optional_float(item.get("aggregate_output_tokens_per_second")),
                time_to_first_token_seconds=_optional_float(item.get("time_to_first_token_seconds")),
                p95_latency_seconds=_optional_float(item.get("p95_latency_seconds")),
                error_rate=float(item.get("error_rate") or 0.0),
                output_tokens_per_dollar=_optional_float(item.get("output_tokens_per_dollar")),
                measured=bool(item.get("measured")),
                actual_prompt_tokens=_optional_int(item.get("actual_prompt_tokens")),
                actual_output_tokens=_optional_int(item.get("actual_output_tokens")),
                sample_count=_optional_int(item.get("sample_count")),
                completion_reason=item.get("completion_reason"),
                cache_policy=_enum_or(CachePolicy, item.get("cache_policy"), CachePolicy.UNSPECIFIED),
                evidence=_enum_or(EvidenceLevel, item.get("evidence"), EvidenceLevel.PREDICTED),
            )
            for item in (payload.get("performance") or ())
            if isinstance(item, dict)
        )
        return EvaluationRecord(
            record_id=str(payload.get("record_id") or path_key(payload)),
            scenario=scenario,
            runtime=runtime,
            prediction=prediction,
            outcome=_enum_or(OutcomeKind, payload.get("outcome"), OutcomeKind.UNKNOWN),
            compatibility=compatibility,
            memory_observations=observations,
            performance=performance,
            price_per_hour_usd=_optional_float(payload.get("price_per_hour_usd")),
            detail=str(payload.get("detail") or ""),
            created_at=payload.get("created_at"),
        )
    except (TypeError, ValueError):
        return None


def path_key(payload: dict[str, Any]) -> str:
    """Fallback record id when an old artifact has none."""

    return str(payload.get("record_id") or "record")


def _enum_or(enum: Any, raw: Any, default: Any) -> Any:
    try:
        return enum(raw) if raw is not None else default
    except ValueError:
        return default


def _optional_evidence(raw: Any) -> Any:
    if raw is None:
        return None
    try:
        from ..protocol.enums import EvidenceLevel

        return EvidenceLevel(raw)
    except ValueError:
        return None


def _optional_float(raw: Any) -> float | None:
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _optional_int(raw: Any) -> int | None:
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _jsonable(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
