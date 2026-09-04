"""Full-context memory planning and runtime argument generation for llama.cpp."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, UTC
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
from typing import Any
from collections.abc import Iterable

from ..protocol.enums import CertificationState, ServingObjective
from ..protocol.models import (
    MemoryEstimate,
    PerformancePoint,
    PlacementAssessment,
    RuntimeAttestation,
    RuntimeTuning,
    ServingRequirements,
    SpeculativeDecodingConfig,
)
from .config import SETTINGS_DIR
from .coerce import optional_float
from .hf_models import GgufQuantMetadata


PLANNER_SCHEMA_VERSION = 1
CERTIFICATE_CACHE_PATH = SETTINGS_DIR / "serving_certificates.json"

# Conservative relative decode capacity. The values only rank unmeasured
# candidates; measured performance always supersedes them.
_GPU_THROUGHPUT_INDEX = {
    "T4": 1.0,
    "L4": 1.25,
    "A10": 1.4,
    "A100": 2.8,
    "A100-40GB": 2.8,
    "A100-80GB": 3.0,
    "L40S": 3.3,
    "RTX-PRO-6000": 4.8,
    "H100": 5.5,
    "H100!": 5.5,
    "H200": 6.1,
    "B200": 9.0,
    "B200+": 9.0,
}

_CACHE_BYTES = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 1.0625,
    "q5_0": 0.6875,
    "q5_1": 0.75,
    "q4_0": 0.5625,
    "q4_1": 0.625,
}

_MANAGED_FLAGS_WITH_VALUES = {
    "-c",
    "--ctx-size",
    "-np",
    "--parallel",
    "-b",
    "--batch-size",
    "-ub",
    "--ubatch-size",
    "-ctk",
    "--cache-type-k",
    "-ctv",
    "--cache-type-v",
    "-fa",
    "--flash-attn",
    "-ngl",
    "--n-gpu-layers",
    "--gpu-layers",
    "--kv-unified-per-slot",
    "-fit",
    "--fit",
    "-fitt",
    "--fit-target",
}

_MANAGED_STANDALONE_FLAGS = {
    "-kvu",
    "--kv-unified",
    "-no-kvu",
    "--no-kv-unified",
}


def serving_requirements(
    context_tokens: int,
    *,
    objective: ServingObjective = ServingObjective.GENERAL_PURPOSE,
    max_hourly_cost_usd: float | None = None,
) -> ServingRequirements:
    """Build the invariant requirements used by Fast Deploy."""

    return ServingRequirements(
        context_tokens=max(1, int(context_tokens)),
        objective=objective,
        full_context_per_request=True,
        gpu_only=True,
        max_hourly_cost_usd=max_hourly_cost_usd,
    )


def tuning_for_objective(
    objective: ServingObjective,
    *,
    speculative_decoding: SpeculativeDecodingConfig | None = None,
) -> RuntimeTuning:
    """Return stable initial tuning for a workload objective.

    Calibration evidence can replace this in future catalog revisions. These
    defaults use llama.cpp's unified KV cache: every slot advertises the full
    model context while concurrent requests share one dynamically allocated
    pool instead of reserving a full worst-case context per slot.
    """

    parallel_slots = {
        ServingObjective.INTERACTIVE: 1,
        ServingObjective.GENERAL_PURPOSE: 4,
        ServingObjective.THROUGHPUT: 8,
        ServingObjective.BENCHMARK: 1,
    }[objective]
    return RuntimeTuning(
        parallel_slots=parallel_slots,
        batch_size=2048,
        ubatch_size=512,
        cache_type_k="f16",
        cache_type_v="f16",
        flash_attention=True,
        gpu_layers="all",
        speculative_decoding=speculative_decoding,
    )


# Ordered most to least precise. The runtime accepts more types than these;
# the planner only offers the ones whose memory cost it models exactly.
KV_CACHE_TYPES: tuple[str, ...] = ("f16", "q8_0", "q4_0")


# Quantized KV is planned only for architectures a bake-off has actually
# measured. This is an allowlist, not a denylist: an unknown architecture gets
# f16, which is always correct, rather than inheriting a default that has never
# been run on it. Architectures move in when scripts/bakeoff_kv_cache.py shows
# recall matching f16 and no throughput regression on that architecture.
#
# Deliberately absent are the MLA-style families (deepseek2/32/4, glm-dsa),
# which compress the KV cache in the attention design itself, so quantizing it
# again is both less valuable and less understood.
QUANTIZED_KV_ARCHITECTURES: frozenset[str] = frozenset()
QUANTIZED_KV_CACHE_TYPE = "q8_0"


def default_cache_type(architecture: str | None) -> str:
    """Return the KV precision Fast Deploy should plan with for an architecture.

    Separate from ``with_cache_type``, which is the mechanism: callers running a
    deliberate comparison set any precision they like, while planning only ever
    picks one that has been measured on the architecture in hand.
    """

    key = (architecture or "").strip().casefold()
    if key and key in QUANTIZED_KV_ARCHITECTURES:
        return QUANTIZED_KV_CACHE_TYPE
    return "f16"


def with_cache_type(tuning: RuntimeTuning, cache_type: str) -> RuntimeTuning:
    """Return tuning that stores the KV cache at one precision.

    The pinned runtime can only quantize the V cache while flash attention is
    enabled, so a quantized request without it is a planner error rather than
    something to silently downgrade. Callers compare precisions by deploying
    each one: quantization halves cache bandwidth but adds dequantization work,
    so whether it is faster is a property of the placement, not of the format.
    """

    kind = cache_type.strip().casefold()
    if kind not in KV_CACHE_TYPES:
        supported = ", ".join(KV_CACHE_TYPES)
        raise ValueError(f"Unsupported KV cache type {cache_type!r}; expected one of {supported}.")
    if kind != "f16" and not tuning.flash_attention:
        raise ValueError(
            "A quantized KV cache requires flash attention in the pinned runtime."
        )
    return replace(tuning, cache_type_k=kind, cache_type_v=kind)


def memory_for_cache_type(
    memory: MemoryEstimate,
    *,
    from_cache_type: str,
    to_cache_type: str,
) -> MemoryEstimate:
    """Retarget an existing estimate at a different KV precision.

    The KV term is linear in bytes per cache element, so a known estimate can be
    rescaled exactly instead of re-reading GGUF metadata. Weights, compute and
    speculative buffers are unaffected by cache precision. The reserve is left
    alone here because it is a property of the device, not the cache; callers
    pass the result to ``assess_memory_placement``, which recomputes it.
    """

    source_bytes = _cache_bytes(from_cache_type)
    target_bytes = _cache_bytes(to_cache_type)
    if source_bytes <= 0:
        raise ValueError(f"Unknown source cache type: {from_cache_type!r}")
    if memory.source != "gguf-metadata":
        raise ValueError(
            "Only a metadata-derived estimate can be rescaled; the fallback "
            "heuristic does not model cache precision."
        )
    kv_gb = memory.kv_cache_gb * (target_bytes / source_bytes)
    total = (
        memory.weights_gb
        + kv_gb
        + memory.compute_gb
        + memory.speculative_gb
        + memory.reserve_gb
    )
    count = max(1, len(memory.per_device_required_gb))
    return replace(
        memory,
        kv_cache_gb=round(kv_gb, 3),
        total_gb=round(total, 3),
        per_device_required_gb=tuple(round(total / count, 3) for _ in range(count)),
    )


def compile_server_args(
    requirements: ServingRequirements,
    tuning: RuntimeTuning,
    *,
    extra_args: Iterable[str] = (),
) -> tuple[str, ...]:
    """Compile canonical llama.cpp arguments, stripping managed duplicates."""

    extras = _without_managed_args(tuple(extra_args))
    arguments = [
        "--ctx-size",
        str(requirements.context_tokens),
        "--parallel",
        str(max(1, tuning.parallel_slots)),
        "--batch-size",
        str(max(1, tuning.batch_size)),
        "--ubatch-size",
        str(max(1, min(tuning.ubatch_size, tuning.batch_size))),
        "--cache-type-k",
        tuning.cache_type_k,
        "--cache-type-v",
        tuning.cache_type_v,
        "--flash-attn",
        "on" if tuning.flash_attention else "off",
        # An explicit parallel count otherwise selects llama.cpp's partitioned
        # KV mode in the pinned runtime, which divides --ctx-size by the number
        # of slots. Unified KV plus a per-slot cap preserves the full advertised
        # context for every individual request while retaining continuous
        # batching for ordinary shorter requests.
        "--kv-unified",
        "--kv-unified-per-slot",
        str(requirements.context_tokens),
        # The runtime fitter may tune unspecified values, but context and GPU
        # layers below are explicit hard constraints. A plan that cannot retain
        # the 2 GiB per-device margin therefore fails instead of silently
        # shrinking context or spilling layers to CPU. The topology-specific
        # compiler raises this margin to 5% on larger accelerators.
        "--fit",
        "on",
        "--fit-target",
        str(max(2048, tuning.fit_target_mib)),
    ]
    if requirements.gpu_only:
        arguments.extend(["--n-gpu-layers", tuning.gpu_layers or "all"])
    arguments.extend(extras)
    return tuple(arguments)


def compile_server_args_string(
    requirements: ServingRequirements,
    tuning: RuntimeTuning,
    *,
    extra_args: Iterable[str] = (),
) -> str:
    """Return shell-safe canonical llama.cpp server arguments."""

    return shlex.join(compile_server_args(requirements, tuning, extra_args=extra_args))


def estimate_memory(
    metadata: GgufQuantMetadata,
    *,
    weights_gb: float,
    requirements: ServingRequirements,
    tuning: RuntimeTuning,
    gpu_count: int = 1,
    gpu_memory_gb: float | None = None,
) -> MemoryEstimate:
    """Estimate full-context llama.cpp memory using GGUF architecture metadata."""

    weights = max(0.0, float(weights_gb))
    context = max(1, requirements.context_tokens)
    kv_gb, metadata_complete = _kv_cache_gb(
        metadata,
        context,
        tuning,
        weights_gb=weights,
    )

    # Compute graphs, CUDA workspaces, and allocations outside the model/KV
    # tensors are deliberately conservative until an exact runtime certificate
    # replaces this estimate.
    compute_gb = max(1.5, min(16.0, weights * 0.08 + tuning.batch_size / 4096.0))
    speculative_gb = (
        max(0.5, weights * 0.04)
        if tuning.speculative_decoding is not None
        else 0.0
    )
    count = max(1, int(gpu_count))
    reserve_per_device = (
        max(2.0, float(gpu_memory_gb) * 0.05)
        if gpu_memory_gb is not None and gpu_memory_gb > 0
        else 0.0
    )
    reserve_gb = reserve_per_device * count
    total = weights + kv_gb + compute_gb + speculative_gb + reserve_gb
    per_device = tuple(total / count for _ in range(count))
    return MemoryEstimate(
        weights_gb=round(weights, 3),
        kv_cache_gb=round(kv_gb, 3),
        compute_gb=round(compute_gb, 3),
        speculative_gb=round(speculative_gb, 3),
        reserve_gb=round(reserve_gb, 3),
        total_gb=round(total, 3),
        per_device_required_gb=tuple(round(value, 3) for value in per_device),
        confidence=0.82 if metadata_complete else 0.55,
        source="gguf-metadata" if metadata_complete else "conservative-fallback",
        total_layer_count=metadata.block_count,
    )


def assess_placement(
    metadata: GgufQuantMetadata,
    *,
    model_id: str,
    revision: str | None,
    quant: str | None,
    runtime_id: str | None,
    weights_gb: float,
    requirements: ServingRequirements,
    tuning: RuntimeTuning,
    gpu_type: str,
    gpu_count: int,
    gpu_memory_gb: float,
    price_per_hour_usd: float | None = None,
) -> PlacementAssessment:
    """Assess one GPU topology against full-context serving requirements."""

    memory = estimate_memory(
        metadata,
        weights_gb=weights_gb,
        requirements=requirements,
        tuning=tuning,
        gpu_count=gpu_count,
        gpu_memory_gb=gpu_memory_gb,
    )
    fits = all(value <= gpu_memory_gb for value in memory.per_device_required_gb)
    fingerprint = serving_fingerprint(
        model_id=model_id,
        revision=revision,
        quant=quant,
        runtime_id=runtime_id,
        requirements=requirements,
        tuning=tuning,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
    )
    cached = load_runtime_attestation(fingerprint)
    performance = (
        cached.performance
        if cached is not None
        else predict_performance(
            weights_gb=weights_gb,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            tuning=tuning,
            price_per_hour_usd=price_per_hour_usd,
        )
    )
    certified = (
        cached is not None
        and cached.gpu_resident
        and cached.effective_context_tokens >= requirements.context_tokens
    )
    if cached is not None and not certified:
        return PlacementAssessment(
            fingerprint=fingerprint,
            memory=memory,
            tuning=tuning,
            performance=performance,
            certification=CertificationState.REJECTED,
            fits=False,
            gpu_resident=False,
            rejection_reason="Cached runtime attestation does not satisfy full-context GPU residency.",
        )
    return PlacementAssessment(
        fingerprint=fingerprint,
        memory=cached.memory if certified and cached.memory is not None else memory,
        tuning=tuning,
        performance=performance,
        certification=(
            CertificationState.CERTIFIED if certified else CertificationState.ESTIMATED
        ),
        fits=fits,
        gpu_resident=fits and requirements.gpu_only,
        rejection_reason=(
            None
            if fits
            else (
                f"Full-context plan needs {max(memory.per_device_required_gb, default=memory.total_gb):.1f} "
                f"GB per GPU; {gpu_type} provides {gpu_memory_gb:.1f} GB."
            )
        ),
    )


def assess_memory_placement(
    base_memory: MemoryEstimate,
    *,
    model_id: str,
    revision: str | None,
    quant: str | None,
    runtime_id: str | None,
    requirements: ServingRequirements,
    tuning: RuntimeTuning,
    gpu_type: str,
    gpu_count: int,
    gpu_memory_gb: float,
    price_per_hour_usd: float | None = None,
) -> PlacementAssessment:
    """Assess a topology from a catalog's model/KV/compute breakdown."""

    count = max(1, int(gpu_count))
    reserve_per_device = max(2.0, float(gpu_memory_gb) * 0.05)
    base_without_reserve = max(0.0, base_memory.total_gb - base_memory.reserve_gb)
    reserve = reserve_per_device * count
    total = base_without_reserve + reserve
    memory = replace(
        base_memory,
        reserve_gb=round(reserve, 3),
        total_gb=round(total, 3),
        per_device_required_gb=tuple(round(total / count, 3) for _ in range(count)),
    )
    fits = all(value <= gpu_memory_gb for value in memory.per_device_required_gb)
    fingerprint = serving_fingerprint(
        model_id=model_id,
        revision=revision,
        quant=quant,
        runtime_id=runtime_id,
        requirements=requirements,
        tuning=tuning,
        gpu_type=gpu_type,
        gpu_count=count,
    )
    cached = load_runtime_attestation(fingerprint)
    certified = (
        cached is not None
        and cached.gpu_resident
        and cached.effective_context_tokens >= requirements.context_tokens
    )
    performance = (
        cached.performance
        if cached is not None and cached.performance
        else predict_performance(
            weights_gb=base_memory.weights_gb,
            gpu_type=gpu_type,
            gpu_count=count,
            tuning=tuning,
            price_per_hour_usd=price_per_hour_usd,
        )
    )
    return PlacementAssessment(
        fingerprint=fingerprint,
        memory=cached.memory if certified and cached.memory is not None else memory,
        tuning=tuning,
        performance=performance,
        certification=(
            CertificationState.CERTIFIED
            if certified
            else (
                CertificationState.REJECTED
                if cached is not None and not certified
                else CertificationState.ESTIMATED
            )
        ),
        fits=fits and (cached is None or certified),
        gpu_resident=fits and requirements.gpu_only and (cached is None or certified),
        rejection_reason=(
            None
            if fits and (cached is None or certified)
            else (
                "A previous runtime attestation rejected this configuration."
                if cached is not None and not certified
                else (
                    f"Full-context plan needs {total / count:.1f} GB per GPU; "
                    f"{gpu_type} provides {gpu_memory_gb:.1f} GB."
                )
            )
        ),
    )


def tuning_for_gpu_memory(
    tuning: RuntimeTuning,
    gpu_memory_gb: float | None,
) -> RuntimeTuning:
    """Apply the runtime margin promised by the placement memory model."""

    if gpu_memory_gb is None or gpu_memory_gb <= 0:
        return tuning
    target_mib = math.ceil(max(2.0, gpu_memory_gb * 0.05) * 1024.0)
    return replace(tuning, fit_target_mib=target_mib)


def predict_performance(
    *,
    weights_gb: float,
    gpu_type: str,
    gpu_count: int,
    tuning: RuntimeTuning,
    price_per_hour_usd: float | None,
) -> tuple[PerformancePoint, ...]:
    """Return conservative relative performance for an uncertified placement."""

    index = _gpu_index(gpu_type)
    weight_factor = (10.0 / max(1.0, weights_gb)) ** 0.82
    topology_factor = 1.0 + 0.62 * (max(1, gpu_count) - 1)
    single_tps = max(1.0, 24.0 * index * weight_factor * topology_factor)
    points: list[PerformancePoint] = []
    for concurrency in (1, 2, 4, 8):
        if concurrency > max(1, tuning.parallel_slots):
            continue
        saturation = 1.0 + 0.58 * (concurrency - 1)
        aggregate = single_tps * saturation
        per_dollar = (
            aggregate * 3600.0 / price_per_hour_usd
            if price_per_hour_usd is not None and price_per_hour_usd > 0
            else None
        )
        points.append(
            PerformancePoint(
                prompt_tokens=4096,
                output_tokens=128,
                concurrency=concurrency,
                prompt_tokens_per_second=round(single_tps * 5.0, 2),
                output_tokens_per_second=round(single_tps, 2),
                aggregate_output_tokens_per_second=round(aggregate, 2),
                time_to_first_token_seconds=round(4096.0 / max(1.0, single_tps * 5.0), 3),
                output_tokens_per_dollar=(round(per_dollar, 2) if per_dollar else None),
                measured=False,
            )
        )
    return tuple(points)


def serving_fingerprint(
    *,
    model_id: str,
    revision: str | None,
    quant: str | None,
    runtime_id: str | None,
    requirements: ServingRequirements,
    tuning: RuntimeTuning,
    gpu_type: str,
    gpu_count: int,
) -> str:
    """Return a stable identifier for all serving-affecting inputs."""

    payload = {
        "schema": PLANNER_SCHEMA_VERSION,
        "model_id": model_id.strip(),
        "revision": (revision or "").strip(),
        "quant": (quant or "").strip(),
        "runtime_id": (runtime_id or "").strip(),
        "requirements": {
            "context_tokens": requirements.context_tokens,
            "objective": requirements.objective.value,
            "full_context_per_request": requirements.full_context_per_request,
            "gpu_only": requirements.gpu_only,
        },
        "tuning": _jsonable(asdict(tuning)),
        "gpu_type": _normalized_gpu(gpu_type),
        "gpu_count": max(1, int(gpu_count)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def assessment_score(
    assessment: PlacementAssessment,
    objective: ServingObjective,
) -> float:
    """Score an eligible placement from its conservative performance curve."""

    if not assessment.fits or not assessment.gpu_resident:
        return float("-inf")
    points = [point for point in assessment.performance if point.error_rate <= 0.05]
    if not points:
        return 0.0
    single = max(
        (point.output_tokens_per_second or 0.0 for point in points if point.concurrency == 1),
        default=0.0,
    )
    aggregate = max(
        (point.aggregate_output_tokens_per_second or 0.0 for point in points),
        default=single,
    )
    efficiency = max(
        (point.output_tokens_per_dollar or 0.0 for point in points),
        default=0.0,
    )
    prompt = max((point.prompt_tokens_per_second or 0.0 for point in points), default=0.0)
    certification_bonus = 1.10 if assessment.certification == CertificationState.CERTIFIED else 1.0
    if objective == ServingObjective.INTERACTIVE:
        return certification_bonus * math.sqrt(max(0.001, single) * max(0.001, prompt))
    if objective == ServingObjective.THROUGHPUT:
        return certification_bonus * (efficiency or aggregate)
    if objective == ServingObjective.BENCHMARK:
        return certification_bonus * aggregate
    values = [max(0.001, single), max(0.001, aggregate), max(0.001, efficiency or aggregate)]
    return certification_bonus * math.prod(values) ** (1.0 / len(values))


def save_runtime_attestation(
    attestation: RuntimeAttestation,
    path: Path | None = None,
) -> None:
    """Persist a successful or rejected runtime attestation atomically."""

    path = path if path is not None else CERTIFICATE_CACHE_PATH
    payload = _read_certificate_payload(path)
    entries = payload.setdefault("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        payload["entries"] = entries
    entries[attestation.fingerprint] = _jsonable(asdict(attestation))
    payload["schema_version"] = PLANNER_SCHEMA_VERSION
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.chmod(temp_path, 0o600)
    temp_path.replace(path)


def load_runtime_attestation(
    fingerprint: str,
    path: Path | None = None,
) -> RuntimeAttestation | None:
    """Load one compatible cached attestation.

    The cache location is resolved on every call so tests and alternate
    settings roots can redirect it without rebinding a default argument.
    """

    payload = _read_certificate_payload(path if path is not None else CERTIFICATE_CACHE_PATH)
    entries = payload.get("entries")
    raw = entries.get(fingerprint) if isinstance(entries, dict) else None
    return runtime_attestation_from_dict(raw)


def runtime_attestation_to_dict(attestation: RuntimeAttestation | None) -> dict[str, Any] | None:
    """Serialize an attestation for connection and certificate stores."""

    return _jsonable(asdict(attestation)) if attestation is not None else None


def runtime_attestation_from_dict(raw: Any) -> RuntimeAttestation | None:
    """Deserialize an attestation from a versioned local store."""

    return _attestation_from_dict(raw)


def attestation_now(
    *,
    fingerprint: str,
    requested_context_tokens: int,
    effective_context_tokens: int,
    gpu_layers: int,
    total_layers: int,
    gpu_resident: bool,
    memory: MemoryEstimate | None,
    performance: tuple[PerformancePoint, ...],
    runtime_id: str | None,
) -> RuntimeAttestation:
    """Build a timestamped attestation from runtime verification evidence."""

    return RuntimeAttestation(
        fingerprint=fingerprint,
        requested_context_tokens=requested_context_tokens,
        effective_context_tokens=effective_context_tokens,
        gpu_layers=gpu_layers,
        total_layers=total_layers,
        gpu_resident=gpu_resident,
        memory=memory,
        performance=performance,
        runtime_id=runtime_id,
        verified_at=datetime.now(UTC).isoformat(),
    )


def _kv_cache_gb(
    metadata: GgufQuantMetadata,
    context_tokens: int,
    tuning: RuntimeTuning,
    *,
    weights_gb: float,
) -> tuple[float, bool]:
    layers = metadata.block_count
    embedding = metadata.embedding_length
    heads = metadata.attention_head_count
    kv_heads = metadata.attention_head_count_kv or heads
    key_length = metadata.attention_key_length
    value_length = metadata.attention_value_length
    if key_length is None and embedding and heads:
        key_length = max(1, embedding // heads)
    if value_length is None:
        value_length = key_length

    if layers and kv_heads and key_length and value_length:
        k_elements = layers * kv_heads * key_length * context_tokens
        v_elements = layers * kv_heads * value_length * context_tokens
        k_bytes = _cache_bytes(tuning.cache_type_k)
        v_bytes = _cache_bytes(tuning.cache_type_v)
        return ((k_elements * k_bytes + v_elements * v_bytes) / 1_000_000_000.0, True)

    # The fallback intentionally grows aggressively with context. It prevents
    # the old failure mode where only the weight file size influenced fit.
    context_units = context_tokens / 32768.0
    return (max(1.0, context_units * 0.10 * max(1.0, weights_gb)), False)


def _cache_bytes(cache_type: str) -> float:
    return _CACHE_BYTES.get(cache_type.strip().casefold(), 2.0)


def _without_managed_args(tokens: tuple[str, ...]) -> tuple[str, ...]:
    filtered: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        key = token.split("=", 1)[0]
        if key in _MANAGED_STANDALONE_FLAGS:
            index += 1
            continue
        if key in _MANAGED_FLAGS_WITH_VALUES:
            index += 1 if "=" in token else 2
            continue
        filtered.append(token)
        index += 1
    return tuple(filtered)


def _normalized_gpu(gpu_type: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "-", gpu_type.upper()).strip("-")


def _gpu_index(gpu_type: str) -> float:
    normalized = _normalized_gpu(gpu_type)
    for key, value in _GPU_THROUGHPUT_INDEX.items():
        if _normalized_gpu(key) in normalized or normalized in _normalized_gpu(key):
            return value
    return 1.0


def _jsonable(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _read_certificate_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"schema_version": PLANNER_SCHEMA_VERSION, "entries": {}}
    if not isinstance(payload, dict) or payload.get("schema_version") != PLANNER_SCHEMA_VERSION:
        return {"schema_version": PLANNER_SCHEMA_VERSION, "entries": {}}
    return payload


def _attestation_from_dict(raw: Any) -> RuntimeAttestation | None:
    if not isinstance(raw, dict):
        return None
    try:
        memory_raw = raw.get("memory")
        memory = (
            MemoryEstimate(
                weights_gb=float(memory_raw["weights_gb"]),
                kv_cache_gb=float(memory_raw["kv_cache_gb"]),
                compute_gb=float(memory_raw["compute_gb"]),
                speculative_gb=float(memory_raw["speculative_gb"]),
                reserve_gb=float(memory_raw["reserve_gb"]),
                total_gb=float(memory_raw["total_gb"]),
                per_device_required_gb=tuple(
                    float(value) for value in memory_raw.get("per_device_required_gb", ())
                ),
                confidence=float(memory_raw.get("confidence", 0.0)),
                source=str(memory_raw.get("source") or "runtime"),
                total_layer_count=(
                    int(memory_raw["total_layer_count"])
                    if memory_raw.get("total_layer_count") is not None
                    else None
                ),
            )
            if isinstance(memory_raw, dict)
            else None
        )
        performance = tuple(_performance_from_dict(item) for item in raw.get("performance", ()))
        return RuntimeAttestation(
            fingerprint=str(raw["fingerprint"]),
            requested_context_tokens=int(raw["requested_context_tokens"]),
            effective_context_tokens=int(raw["effective_context_tokens"]),
            gpu_layers=int(raw.get("gpu_layers", 0)),
            total_layers=int(raw.get("total_layers", 0)),
            gpu_resident=bool(raw.get("gpu_resident")),
            memory=memory,
            performance=performance,
            runtime_id=str(raw.get("runtime_id") or "") or None,
            verified_at=str(raw.get("verified_at") or "") or None,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _performance_from_dict(raw: Any) -> PerformancePoint:
    if not isinstance(raw, dict):
        raise TypeError("performance point must be an object")
    return PerformancePoint(
        prompt_tokens=int(raw.get("prompt_tokens", 0)),
        output_tokens=int(raw.get("output_tokens", 0)),
        concurrency=max(1, int(raw.get("concurrency", 1))),
        prompt_tokens_per_second=optional_float(raw.get("prompt_tokens_per_second")),
        output_tokens_per_second=optional_float(raw.get("output_tokens_per_second")),
        aggregate_output_tokens_per_second=optional_float(
            raw.get("aggregate_output_tokens_per_second")
        ),
        time_to_first_token_seconds=optional_float(raw.get("time_to_first_token_seconds")),
        p95_latency_seconds=optional_float(raw.get("p95_latency_seconds")),
        error_rate=float(raw.get("error_rate", 0.0)),
        output_tokens_per_dollar=optional_float(raw.get("output_tokens_per_dollar")),
        measured=bool(raw.get("measured")),
    )
