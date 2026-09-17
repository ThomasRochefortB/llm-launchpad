"""Build a live Deploy catalog for the TUI."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, UTC
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from statistics import median
import time
from typing import Any
from collections.abc import Sequence
from urllib.parse import urlparse

from .artificial_analysis import (
    AAModelCandidate,
    AA_ATTRIBUTION,
    AA_CACHE_PATH,
    ModelSizeBucket,
    _load_aa_rankings,
    _model_key,
    _size_bucket_for_parameters,
)
from .artificial_analysis_auth import resolve_artificial_analysis_api_key
from .coerce import clean_string
from .diagnostics import log_debug
from .gguf_metadata import GgufMtpStatus
from .hf_models import (
    GgufQuantMetadata,
    ModelCandidate,
    fetch_gguf_quant_metadata,
    fetch_model_max_context,
    list_llamacpp_candidates,
)
from .modal_gpu import ModalGpuSpec, fetch_modal_gpu_catalog
from .hf_budget import (
    DEFAULT_BUDGET,
    METADATA_REQUEST_COST,
    HubBudgetExhausted,
    HubLookupIncomplete,
    HubRequestBudget,
    UnlimitedBudget,
)
from .hf_repo_matches import RepoMatchStore
from .naming import slugify_instance_name
from .quant_quality import quant_bits
from .quick_deploy import QuickDeployCatalogInfo, QuickDeployProfile
from .llamacpp_planner import (
    attention_head_count,
    per_device_requirements,
    tuning_for_architecture,
    tuning_for_attention_scratch,
    ubatch_for_attention_scratch,
    compile_server_args,
    estimate_memory,
    serving_requirements,
    tuning_for_objective,
)
from .runtime_support import (
    DEFAULT_MTP_DRAFT_TOKENS,
    evaluate_llamacpp_architecture,
    evaluate_llamacpp_mtp,
)
from ..protocol.enums import ServingObjective, SpeculativeDecodingMethod
from ..protocol.models import (
    CatalogExclusion,
    MemoryEstimate,
    RuntimeTuning,
    ServingRequirements,
    SpeculativeDecodingConfig,
)

DEFAULT_MODEL_LIMIT = 3
DEFAULT_OVERALL_MODEL_LIMIT = 10
DEFAULT_CANDIDATE_LIMIT = 80
_AA_RESOLUTION_WORKERS = 24
_FALLBACK_TRENDING_LIMIT = 8
_HF_SEARCH_WORKERS = 4
_HF_SEARCH_LIMIT = 10
_HF_SEARCH_TIMEOUT_SECONDS = 10.0
DEFAULT_CONTEXT_TOKENS = 65_536
LOW_VRAM_QUANT = "UD-Q2_K_XL"

# Marks a candidate the build never got to. Shared so the screen can count
# these and say the shortlist is provisional rather than final.
UNCHECKED_BUDGET_REASON = (
    "Not checked: this catalog build reached its Hugging Face request budget. "
    "Refresh the catalog to continue from here."
)
# A Hub failure that is merely transient must not be recorded as "this model
# does not exist": every dropped model silently shrinks the user's catalog.
_HUB_FETCH_ATTEMPTS = 3
_HUB_RETRY_BASE_SECONDS = 0.5
# A refresh that loses a large share of the catalog is far more likely to be a
# run of transient upstream failures than a real change in the rankings, so the
# previous snapshot is kept and the next refresh recovers on its own.
_CATALOG_RETENTION_RATIO = 0.6


def _is_transient_hub_error(exc: BaseException) -> bool:
    """Return whether a Hub failure is worth retrying rather than treating as absent."""

    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status == 429 or 500 <= status < 600
    text = f"{type(exc).__name__}: {exc}".casefold()
    return any(
        marker in text
        for marker in (
            "429",
            "too many requests",
            "timeout",
            "timed out",
            "connection",
            "temporarily",
            "503",
            "502",
        )
    )


def _fetch_serving_metadata(repo_id: str) -> GgufQuantMetadata:
    """Fetch GGUF serving metadata, retrying transient Hub failures."""

    delay = _HUB_RETRY_BASE_SECONDS
    for attempt in range(1, _HUB_FETCH_ATTEMPTS + 1):
        try:
            return fetch_gguf_quant_metadata(repo_id, inspect_serving=True)
        except Exception as exc:
            if attempt >= _HUB_FETCH_ATTEMPTS or not _is_transient_hub_error(exc):
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"Unreachable metadata retry loop for {repo_id}")


def _quick_deploy_catalog_cache_path() -> Path:
    """Resolve the catalog snapshot path from the current settings dir."""

    from .config import SETTINGS_DIR as current_settings_dir

    return current_settings_dir / "quick_deploy_catalog.json"


QUICK_DEPLOY_CATALOG_CACHE_TTL = timedelta(hours=6)
QUICK_DEPLOY_CATALOG_CACHE_SCHEMA_VERSION = 10
_MODEL_SIZE_BUCKETS: tuple[ModelSizeBucket, ...] = ("compact", "medium", "large")
_MODEL_SIZE_LABELS: dict[ModelSizeBucket, str] = {
    "compact": "Compact ≤40B",
    "medium": "Medium 40–150B",
    "large": "Large >150B",
}

_PREFERRED_QUANT_ORDER = (
    "UD-Q4_K_XL",
    "Q4_K_M",
    "UD-Q3_K_XL",
    "UD-Q2_K_XL",
    "Q4_K_S",
    "Q5_K_M",
    "Q5_K_S",
    "Q6_K",
    "Q8_0",
)
_FALLBACK_GPU_PRICE_PER_HOUR = {
    "T4": 0.5904,
    "L4": 0.7992,
    "A10": 1.1016,
    "A100": 2.0988,
    "A100-40GB": 2.0988,
    "A100-80GB": 2.4984,
    "L40S": 1.9512,
    "RTX-PRO-6000": 3.0312,
    "H100": 3.9492,
    "H100!": 3.9492,
    "H200": 4.5396,
    "B200": 6.2496,
    "B200+": 6.2496,
}
_GPU_MEMORY_GB = {
    "T4": 16.0,
    "L4": 24.0,
    "A10": 24.0,
    "A100": 40.0,
    "A100-40GB": 40.0,
    "A100-80GB": 80.0,
    "L40S": 48.0,
    "RTX-PRO-6000": 96.0,
    "H100": 80.0,
    "H100!": 80.0,
    "H200": 141.0,
    "B200": 180.0,
    "B200+": 180.0,
}
_RESOURCE_TIER_LABELS = {
    "cheap": "$",
    "rtx-pro": "$$",
    "b200": "$$$",
}
_RESOURCE_TIER_DESCRIPTIONS = {
    "cheap": "Slow but cheap",
    "rtx-pro": "RTX PRO",
    "b200": "B200",
}


@dataclass(frozen=True)
class _GpuSelection:
    quant: str
    gpu_type: str
    gpu_count: int
    cost_per_hour_usd: float
    required_vram_gb: float


@dataclass(frozen=True)
class _ResolvedAAModel:
    candidate: AAModelCandidate
    repo_id: str
    size_bucket: ModelSizeBucket
    profiles: tuple[QuickDeployProfile, ...]


def build_live_quick_deploy_catalog(
    *,
    model_limit: int = DEFAULT_MODEL_LIMIT,
    overall_model_limit: int = DEFAULT_OVERALL_MODEL_LIMIT,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
) -> tuple[QuickDeployCatalogInfo, tuple[QuickDeployProfile, ...]]:
    """Rebuild the Deploy catalog from the top AAI open models and live pricing."""

    normalized_model_limit = max(1, int(model_limit))
    normalized_candidate_limit = max(
        normalized_model_limit,
        int(candidate_limit),
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        rankings_future = executor.submit(
            _load_aa_rankings,
            api_key=resolve_artificial_analysis_api_key(),
            cache_path=AA_CACHE_PATH,
        )
        gpu_future = executor.submit(_fetch_modal_gpu_catalog_or_fallback)
        rankings = rankings_future.result()
        modal_gpu_catalog, has_live_modal_pricing = gpu_future.result()

    exclusions: list[CatalogExclusion] = []
    if rankings is not None:
        profiles = _profiles_from_aa_rankings(
            rankings.candidates,
            modal_gpu_catalog,
            model_limit=normalized_model_limit,
            overall_model_limit=max(0, int(overall_model_limit)),
            candidate_limit=normalized_candidate_limit,
            exclusions=exclusions,
        )
        if profiles:
            tier_suffix = f", {rankings.tier} tier" if rankings.tier else ""
            info = QuickDeployCatalogInfo(
                source_label=(
                    "Artificial Analysis top open models "
                    f"({rankings.freshness}{tier_suffix}) + "
                    f"{'live' if has_live_modal_pricing else 'fallback'} Modal pricing"
                ),
                generated_at=_utc_now_iso(),
                attribution=(
                    f"{AA_ATTRIBUTION} Model metadata sourced from Hugging Face; "
                    "GPU pricing sourced from Modal."
                ),
                is_live=True,
                exclusions=tuple(sorted(exclusions, key=lambda row: row.rank or 10**9)),
            )
            retained = _retained_catalog_for(profiles)
            if retained is not None:
                return retained
            _write_quick_deploy_catalog_cache(info, profiles)
            return (info, profiles)

    fallback = _build_trending_fallback_catalog(
        model_limit=normalized_model_limit,
        modal_gpu_catalog=modal_gpu_catalog,
        has_live_modal_pricing=has_live_modal_pricing,
    )
    fallback = (replace(fallback[0], exclusions=tuple(exclusions)), fallback[1])
    _write_quick_deploy_catalog_cache(fallback[0], fallback[1])
    return fallback


def _retained_catalog_for(
    profiles: Sequence[QuickDeployProfile],
    *,
    cache_path: Path | None = None,
) -> tuple[QuickDeployCatalogInfo, tuple[QuickDeployProfile, ...]] | None:
    """Return the cached catalog when a rebuild lost most of its profiles.

    Every model whose Hub metadata cannot be read is dropped from the rebuild,
    so a run of transient failures produces a small but structurally valid
    catalog. Persisting that would replace a good snapshot with a worse one and
    hide models the user deployed yesterday, so the previous snapshot wins and
    the next refresh recovers on its own.
    """

    previous = _read_quick_deploy_catalog_cache(
        cache_path or _quick_deploy_catalog_cache_path()
    )
    if previous is None:
        return None
    previous_info, previous_profiles = previous
    if not previous_profiles:
        return None
    if len(profiles) >= len(previous_profiles) * _CATALOG_RETENTION_RATIO:
        return None
    return (
        replace(
            previous_info,
            error=(
                f"Kept the previous catalog: this refresh resolved only "
                f"{len(profiles)} of {len(previous_profiles)} models, which "
                "usually means Hugging Face was rate limiting or unreachable."
            ),
        ),
        previous_profiles,
    )


def load_cached_quick_deploy_catalog(
    *,
    cache_path: Path | None = None,
    now: datetime | None = None,
) -> tuple[QuickDeployCatalogInfo, tuple[QuickDeployProfile, ...]] | None:
    """Return the last successfully built catalog, whatever its age.

    The caller decides freshness (``is_fresh_cached_quick_deploy_catalog``);
    a stale snapshot still beats the "Building…" empty state, so loading and
    freshness are separate steps.
    """

    return _read_quick_deploy_catalog_cache(cache_path or _quick_deploy_catalog_cache_path())


def is_fresh_cached_quick_deploy_catalog(
    info: QuickDeployCatalogInfo,
    *,
    now: datetime | None = None,
) -> bool:
    """Return True when a cached catalog snapshot is fresh enough to trust.

    A build that stopped at its request budget is never fresh, however
    recently it ran. It is a partial answer, and the whole point of writing
    down what it resolved is that reopening the screen carries on from there
    instead of serving the short list for another six hours.
    """

    if any(
        exclusion.reason == UNCHECKED_BUDGET_REASON for exclusion in info.exclusions
    ):
        return False
    generated_at = clean_string(info.generated_at)
    if not generated_at:
        return False
    try:
        fetched_at = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=UTC)
    current_time = now or datetime.now(UTC)
    return current_time - fetched_at.astimezone(UTC) <= QUICK_DEPLOY_CATALOG_CACHE_TTL


def _quick_deploy_profile_to_dict(profile: QuickDeployProfile) -> dict[str, Any]:
    speculative = profile.speculative_decoding
    requirements = profile.serving_requirements
    tuning = profile.runtime_tuning
    memory = profile.memory_estimate
    return {
        "id": profile.id,
        "display_name": profile.display_name,
        "repo_id": profile.repo_id,
        "quant": profile.quant,
        "gpu_type": profile.gpu_type,
        "gpu_count": profile.gpu_count,
        "profile_label": profile.profile_label,
        "approx_cost_per_hour_usd": profile.approx_cost_per_hour_usd,
        "max_context_tokens": profile.max_context_tokens,
        "instance_slug_hint": profile.instance_slug_hint,
        "summary": profile.summary,
        "server_args": list(profile.server_args),
        "required_vram_gb": profile.required_vram_gb,
        "gpu_memory_gb": profile.gpu_memory_gb,
        "resource_tier": profile.resource_tier,
        "resource_tier_label": profile.resource_tier_label,
        "source_label": profile.source_label,
        "aa_model_id": profile.aa_model_id,
        "aa_model_name": profile.aa_model_name,
        "aa_model_slug": profile.aa_model_slug,
        "aa_coding_score": profile.aa_coding_score,
        "aa_intelligence_score": profile.aa_intelligence_score,
        "aa_rank": profile.aa_rank,
        "model_size_label": profile.model_size_label,
        "backend": profile.backend.value,
        "model_name": profile.model_name,
        "gguf_architecture": profile.gguf_architecture,
        "llamacpp_runtime_id": profile.llamacpp_runtime_id,
        "speculative_decoding": (
            {
                "method": speculative.method.value,
                "num_speculative_tokens": speculative.num_speculative_tokens,
                "nextn_predict_layers": speculative.nextn_predict_layers,
            }
            if speculative is not None
            else None
        ),
        "serving_requirements": (
            {
                "context_tokens": requirements.context_tokens,
                "objective": requirements.objective.value,
                "full_context_per_request": requirements.full_context_per_request,
                "gpu_only": requirements.gpu_only,
                "max_hourly_cost_usd": requirements.max_hourly_cost_usd,
            }
            if requirements is not None
            else None
        ),
        "runtime_tuning": (
            {
                "parallel_slots": tuning.parallel_slots,
                "batch_size": tuning.batch_size,
                "ubatch_size": tuning.ubatch_size,
                "cache_type_k": tuning.cache_type_k,
                "cache_type_v": tuning.cache_type_v,
                "flash_attention": tuning.flash_attention,
                "gpu_layers": tuning.gpu_layers,
                "fit_target_mib": tuning.fit_target_mib,
            }
            if tuning is not None
            else None
        ),
        "memory_estimate": (
            {
                "weights_gb": memory.weights_gb,
                "kv_cache_gb": memory.kv_cache_gb,
                "compute_gb": memory.compute_gb,
                "attention_scratch_gb": memory.attention_scratch_gb,
                "speculative_gb": memory.speculative_gb,
                "reserve_gb": memory.reserve_gb,
                "total_gb": memory.total_gb,
                "per_device_required_gb": list(memory.per_device_required_gb),
                "confidence": memory.confidence,
                "source": memory.source,
                "total_layer_count": memory.total_layer_count,
                "recurrent_gb": memory.recurrent_gb,
                "recurrent_state_copies": memory.recurrent_state_copies,
            }
            if memory is not None
            else None
        ),
    }


def _quick_deploy_profile_from_dict(payload: Any) -> QuickDeployProfile | None:
    if not isinstance(payload, dict):
        return None
    try:
        from ..protocol.enums import BackendType
    except Exception:
        return None
    try:
        backend = BackendType(str(payload.get("backend") or "llamacpp"))
    except ValueError:
        return None
    speculative_payload = payload.get("speculative_decoding")
    speculative = None
    if isinstance(speculative_payload, dict):
        try:
            speculative = SpeculativeDecodingConfig(
                method=SpeculativeDecodingMethod(
                    str(speculative_payload.get("method") or "mtp")
                ),
                num_speculative_tokens=int(
                    speculative_payload.get("num_speculative_tokens") or 0
                ),
                nextn_predict_layers=(
                    int(speculative_payload["nextn_predict_layers"])
                    if speculative_payload.get("nextn_predict_layers") is not None
                    else None
                ),
            )
        except (ValueError, TypeError):
            speculative = None
    requirements = _serving_requirements_from_dict(payload.get("serving_requirements"))
    tuning = _runtime_tuning_from_dict(
        payload.get("runtime_tuning"),
        speculative_decoding=speculative,
    )
    memory = _memory_estimate_from_dict(payload.get("memory_estimate"))
    try:
        return QuickDeployProfile(
            id=str(payload.get("id") or ""),
            display_name=str(payload.get("display_name") or ""),
            repo_id=str(payload.get("repo_id") or ""),
            quant=str(payload.get("quant") or ""),
            gpu_type=str(payload.get("gpu_type") or ""),
            gpu_count=int(payload.get("gpu_count") or 0),
            profile_label=str(payload.get("profile_label") or ""),
            approx_cost_per_hour_usd=float(payload.get("approx_cost_per_hour_usd") or 0.0),
            max_context_tokens=int(payload.get("max_context_tokens") or 0),
            instance_slug_hint=str(payload.get("instance_slug_hint") or ""),
            summary=str(payload.get("summary") or ""),
            server_args=tuple(str(arg) for arg in (payload.get("server_args") or ())),
            required_vram_gb=(
                float(payload["required_vram_gb"])
                if payload.get("required_vram_gb") is not None
                else None
            ),
            gpu_memory_gb=(
                float(payload["gpu_memory_gb"])
                if payload.get("gpu_memory_gb") is not None
                else None
            ),
            resource_tier=payload.get("resource_tier"),
            resource_tier_label=payload.get("resource_tier_label"),
            source_label=str(payload.get("source_label") or "Curated"),
            aa_model_id=payload.get("aa_model_id"),
            aa_model_name=payload.get("aa_model_name"),
            aa_model_slug=payload.get("aa_model_slug"),
            aa_coding_score=(
                float(payload["aa_coding_score"])
                if payload.get("aa_coding_score") is not None
                else None
            ),
            aa_intelligence_score=(
                float(payload["aa_intelligence_score"])
                if payload.get("aa_intelligence_score") is not None
                else None
            ),
            aa_rank=(
                int(payload["aa_rank"]) if payload.get("aa_rank") is not None else None
            ),
            model_size_label=payload.get("model_size_label"),
            backend=backend,
            model_name=payload.get("model_name"),
            gguf_architecture=payload.get("gguf_architecture"),
            llamacpp_runtime_id=payload.get("llamacpp_runtime_id"),
            speculative_decoding=speculative,
            serving_requirements=requirements,
            runtime_tuning=tuning,
            memory_estimate=memory,
        )
    except (ValueError, TypeError):
        return None


def _serving_requirements_from_dict(payload: Any) -> ServingRequirements | None:
    if not isinstance(payload, dict):
        return None
    try:
        return ServingRequirements(
            context_tokens=max(1, int(payload["context_tokens"])),
            objective=ServingObjective(
                str(payload.get("objective") or ServingObjective.GENERAL_PURPOSE.value)
            ),
            full_context_per_request=bool(payload.get("full_context_per_request", True)),
            gpu_only=bool(payload.get("gpu_only", True)),
            max_hourly_cost_usd=(
                float(payload["max_hourly_cost_usd"])
                if payload.get("max_hourly_cost_usd") is not None
                else None
            ),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _runtime_tuning_from_dict(
    payload: Any,
    *,
    speculative_decoding: SpeculativeDecodingConfig | None,
) -> RuntimeTuning | None:
    if not isinstance(payload, dict):
        return None
    try:
        return RuntimeTuning(
            parallel_slots=max(1, int(payload.get("parallel_slots", 1))),
            batch_size=max(1, int(payload.get("batch_size", 2048))),
            ubatch_size=max(1, int(payload.get("ubatch_size", 512))),
            cache_type_k=str(payload.get("cache_type_k") or "f16"),
            cache_type_v=str(payload.get("cache_type_v") or "f16"),
            flash_attention=bool(payload.get("flash_attention", True)),
            gpu_layers=str(payload.get("gpu_layers") or "all"),
            fit_target_mib=max(2048, int(payload.get("fit_target_mib", 2048))),
            speculative_decoding=speculative_decoding,
        )
    except (TypeError, ValueError):
        return None


def _memory_estimate_from_dict(payload: Any) -> MemoryEstimate | None:
    if not isinstance(payload, dict):
        return None
    try:
        return MemoryEstimate(
            weights_gb=float(payload["weights_gb"]),
            kv_cache_gb=float(payload["kv_cache_gb"]),
            compute_gb=float(payload["compute_gb"]),
            attention_scratch_gb=float(payload.get("attention_scratch_gb", 0.0)),
            speculative_gb=float(payload.get("speculative_gb", 0.0)),
            reserve_gb=float(payload.get("reserve_gb", 0.0)),
            total_gb=float(payload["total_gb"]),
            per_device_required_gb=tuple(
                float(value) for value in payload.get("per_device_required_gb", ())
            ),
            confidence=float(payload.get("confidence", 0.0)),
            source=str(payload.get("source") or "estimated"),
            recurrent_gb=float(payload.get("recurrent_gb", 0.0)),
            recurrent_state_copies=int(payload.get("recurrent_state_copies", 1)),
            total_layer_count=(
                int(payload["total_layer_count"])
                if payload.get("total_layer_count") is not None
                else None
            ),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _quick_deploy_catalog_info_to_dict(info: QuickDeployCatalogInfo) -> dict[str, Any]:
    return {
        "source_label": info.source_label,
        "generated_at": info.generated_at,
        "attribution": info.attribution,
        "is_fallback": info.is_fallback,
        "is_live": info.is_live,
        "ready": info.ready,
        "error": info.error,
        "exclusions": [asdict(row) for row in info.exclusions],
    }


def _quick_deploy_catalog_info_from_dict(payload: Any) -> QuickDeployCatalogInfo | None:
    if not isinstance(payload, dict):
        return None
    try:
        return QuickDeployCatalogInfo(
            source_label=str(payload.get("source_label") or "Cached catalog"),
            generated_at=payload.get("generated_at"),
            attribution=payload.get("attribution"),
            is_fallback=bool(payload.get("is_fallback", False)),
            is_live=bool(payload.get("is_live", True)),
            ready=bool(payload.get("ready", True)),
            error=payload.get("error"),
            exclusions=tuple(CatalogExclusion(**row) for row in payload.get("exclusions", ())),
        )
    except (ValueError, TypeError):
        return None


def _write_quick_deploy_catalog_cache(
    info: QuickDeployCatalogInfo,
    profiles: Sequence[QuickDeployProfile],
    *,
    cache_path: Path | None = None,
) -> None:
    envelope = {
        "schema_version": QUICK_DEPLOY_CATALOG_CACHE_SCHEMA_VERSION,
        "info": _quick_deploy_catalog_info_to_dict(info),
        "profiles": [_quick_deploy_profile_to_dict(profile) for profile in profiles],
    }
    path = cache_path or _quick_deploy_catalog_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        temporary_path.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(path)
    except Exception:
        return


def _read_quick_deploy_catalog_cache(
    path: Path,
) -> tuple[QuickDeployCatalogInfo, tuple[QuickDeployProfile, ...]] | None:
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(envelope, dict):
        return None
    if envelope.get("schema_version") != QUICK_DEPLOY_CATALOG_CACHE_SCHEMA_VERSION:
        return None
    info = _quick_deploy_catalog_info_from_dict(envelope.get("info"))
    raw_profiles = envelope.get("profiles")
    if info is None or not isinstance(raw_profiles, list):
        return None
    profiles = tuple(
        profile
        for raw in raw_profiles
        if (profile := _quick_deploy_profile_from_dict(raw)) is not None
        and profile.id
        and profile.repo_id
    )
    if not profiles:
        return None
    if not info.ready:
        info = replace(info, ready=True, error=None)
    return (info, profiles)


def _with_unique_ids(
    profiles: Sequence[QuickDeployProfile],
) -> tuple[QuickDeployProfile, ...]:
    """Drop profiles whose id is already taken.

    Ids are derived from repository, quantization and resource tier, so a
    collision means two catalog entries describe the same deployable artifact --
    a benchmark feed listing one model twice under different effort settings,
    for instance. Keeping both would make ``get_quick_deploy_profile`` return
    whichever happened to be first, so the duplicate is dropped rather than
    disambiguated with an order-dependent suffix that would be unstable again.
    """

    unique: list[QuickDeployProfile] = []
    seen: set[str] = set()
    for profile in profiles:
        if profile.id in seen:
            log_debug(
                f"Dropping duplicate catalog profile {profile.id} "
                f"({profile.display_name} / {profile.repo_id})"
            )
            continue
        seen.add(profile.id)
        unique.append(profile)
    return tuple(unique)


def _profiles_from_aa_rankings(
    candidates: Sequence[AAModelCandidate],
    modal_gpu_catalog: Sequence[ModalGpuSpec],
    *,
    model_limit: int,
    candidate_limit: int,
    exclusions: list[CatalogExclusion] | None = None,
    hub_request_budget: int = DEFAULT_BUDGET,
    overall_model_limit: int = 0,
) -> tuple[QuickDeployProfile, ...]:
    """Select the union of overall and per-size shortlists of deployable models."""

    # Deduplicate variants of the same model *before* the candidate window, so
    # a budget of N admits N distinct models rather than N benchmark rows. The
    # feed lists one row per reasoning-effort setting -- Claude Fable 5.1 alone
    # occupies five -- and the window used to spend its budget on those rows.
    # The unknown-size pool starved first, because effort variants cluster at
    # the top of the ranking where nearly every model is API-only: the pool ran
    # out at rank 83, and open-weight models below it with real GGUF
    # repositories were never inspected and never recorded as excluded, so
    # nothing on screen distinguished them from models that had been rejected.
    #
    # Deduplicating first also keeps the original reason this step exists:
    # _resolve_aa_model caches by model key, but the check-then-act on the
    # shared dict races, so two variants with the same key can both miss the
    # cache and issue duplicate HF lookups (flaky call counts under xdist).
    # Resolving one representative per key keeps behavior identical -- the
    # loser would be dropped by seen_repos anyway -- and the representative is
    # the highest-ranked variant, which is the one the shortlist wants.
    deduped_window = list(_aa_candidate_window(_deduped_candidates(candidates), candidate_limit))
    selected_by_bucket: dict[ModelSizeBucket, list[_ResolvedAAModel]] = {
        bucket: [] for bucket in _MODEL_SIZE_BUCKETS
    }
    selected_overall: list[_ResolvedAAModel] = []
    seen_repos: set[str] = set()
    repo_by_model_key: dict[str, str] = {}
    # Remembered match results make most candidates free, which is what lets
    # the window be swept at all; the budget is the backstop for the rest.
    match_store = RepoMatchStore.load()
    budget = HubRequestBudget(hub_request_budget)
    try:
        from huggingface_hub import HfApi

        shared_hf_api: Any | None = HfApi()
    except Exception:
        shared_hf_api = None

    # Keep a bounded lookahead, then consume results in benchmark order. A
    # faster response must never displace a higher-ranked eligible model.
    # Skip a full category only after the overall shortlist is also full.
    # Otherwise a fourth large model could outrank every smaller model.
    remaining = iter(deduped_window)
    pending: deque[tuple[AAModelCandidate, Future[_ResolvedAAModel | None], list[CatalogExclusion]]] = deque()

    def category_full(candidate: AAModelCandidate) -> bool:
        bucket = _size_bucket_for_parameters(candidate.parameter_count_b)
        return (
            len(selected_overall) >= overall_model_limit
            and bucket is not None
            and len(selected_by_bucket[bucket]) >= model_limit
        )

    executor = ThreadPoolExecutor(max_workers=_AA_RESOLUTION_WORKERS)

    def fill_pending() -> None:
        while len(pending) < _AA_RESOLUTION_WORKERS:
            if budget.exhausted:
                # Scheduling past this point buys nothing: every worker would
                # refuse its requests and report the same unchecked result.
                break
            candidate = next(remaining, None)
            if candidate is None:
                break
            if category_full(candidate):
                continue
            # Workers own their diagnostics. Cancelled/unused lookahead must
            # not mutate an already-published catalog after this call returns.
            reasons: list[CatalogExclusion] = []
            future = executor.submit(
                _resolve_aa_model,
                candidate,
                modal_gpu_catalog,
                repo_by_model_key=repo_by_model_key,
                hf_api=shared_hf_api,
                exclusions=reasons,
                match_store=match_store,
                budget=budget,
            )
            pending.append((candidate, future, reasons))

    try:
        fill_pending()
        while pending:
            candidate, future, reasons = pending.popleft()
            if category_full(candidate):
                future.cancel()
                fill_pending()
                continue
            resolved = future.result()
            if exclusions is not None:
                exclusions.extend(reasons)
            if resolved is not None and resolved.repo_id and resolved.repo_id not in seen_repos:
                seen_repos.add(resolved.repo_id)
                in_overall = len(selected_overall) < overall_model_limit
                if in_overall:
                    selected_overall.append(resolved)
                bucket_models = selected_by_bucket[resolved.size_bucket]
                if len(bucket_models) < model_limit:
                    bucket_models.append(resolved)
                elif not in_overall:
                    _record_exclusion(
                        exclusions, candidate, resolved.repo_id,
                        f"Outside the top {model_limit} {_MODEL_SIZE_LABELS[resolved.size_bucket]} recommendations.",
                    )
            if len(selected_overall) >= overall_model_limit and all(
                len(models) >= model_limit for models in selected_by_bucket.values()
            ):
                break
            fill_pending()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        # Whatever was learned this run is worth keeping even if the run was
        # cut short; that is what makes the next one reach further.
        match_store.save()

    profiles: list[QuickDeployProfile] = []
    for bucket in _MODEL_SIZE_BUCKETS:
        for resolved in selected_by_bucket[bucket]:
            profiles.extend(resolved.profiles)
    included_repos = {profile.repo_id for profile in profiles}
    for resolved in selected_overall:
        if resolved.repo_id not in included_repos:
            profiles.extend(resolved.profiles)
    return _with_unique_ids(profiles)


def _deduped_candidates(
    candidates: Sequence[AAModelCandidate],
) -> tuple[AAModelCandidate, ...]:
    """Keep the highest-ranked variant of each distinct model.

    The benchmark feed carries one row per reasoning-effort setting. They
    share a model key, a repository and a size, so for shortlisting purposes
    they are one candidate, and the best-scoring row represents it.
    """

    seen_keys: set[str] = set()
    deduped: list[AAModelCandidate] = []
    for candidate in candidates:
        model_key = _model_key(candidate.name) or _model_key(candidate.slug)
        if model_key:
            if model_key in seen_keys:
                continue
            seen_keys.add(model_key)
        deduped.append(candidate)
    return tuple(deduped)


def _aa_candidate_window(
    candidates: Sequence[AAModelCandidate],
    candidate_limit: int,
) -> tuple[AAModelCandidate, ...]:
    """Keep a ranked candidate budget for every known size bucket.

    Candidates whose size the feed does not state share one pool, because
    until their weights are inspected they could belong to any bucket. Pass
    a deduplicated ranking: the budget counts candidates, so variants of one
    model would otherwise consume several slots each.
    """

    candidates_by_bucket: dict[ModelSizeBucket, list[AAModelCandidate]] = {
        bucket: [] for bucket in _MODEL_SIZE_BUCKETS
    }
    # A candidate of unknown size competes in every bucket until its weights
    # are read, so it gets every bucket's budget rather than one bucket's
    # worth. Widening this once before was a mistake: the discovery loop only
    # stops early when every bucket holds a full shortlist, so a bucket that
    # cannot fill made it sweep the whole window and exhaust Hugging Face's
    # quota, and throttled lookups then dropped models silently. The sweep is
    # now bounded by a request budget and most of it is answered from
    # remembered match results, so depth costs requests only the first time.
    unknown_size_candidates: list[AAModelCandidate] = []
    selected_ids: set[int] = set()
    for candidate in candidates:
        bucket = _size_bucket_for_parameters(candidate.parameter_count_b)
        if bucket is None:
            if len(unknown_size_candidates) < candidate_limit * len(_MODEL_SIZE_BUCKETS):
                unknown_size_candidates.append(candidate)
                selected_ids.add(id(candidate))
            continue
        bucket_candidates = candidates_by_bucket[bucket]
        if len(bucket_candidates) < candidate_limit:
            bucket_candidates.append(candidate)
            selected_ids.add(id(candidate))

    return tuple(candidate for candidate in candidates if id(candidate) in selected_ids)


def _resolve_aa_model(
    candidate: AAModelCandidate,
    modal_gpu_catalog: Sequence[ModalGpuSpec],
    *,
    repo_by_model_key: dict[str, str] | None = None,
    hf_api: Any | None = None,
    exclusions: list[CatalogExclusion] | None = None,
    match_store: RepoMatchStore | None = None,
    budget: HubRequestBudget | None = None,
) -> _ResolvedAAModel | None:
    model_key = _model_key(candidate.name) or _model_key(candidate.slug)
    if repo_by_model_key is not None and model_key in repo_by_model_key:
        return _build_resolved_aa_model(
            candidate,
            modal_gpu_catalog,
            repo_by_model_key[model_key],
            exclusions=exclusions,
            budget=budget,
        )
    # A remembered answer costs nothing. Misses dominate -- the feed is mostly
    # API-only models -- and rediscovering them is what used to consume the
    # request quota before the open models further down were ever reached.
    remembered = match_store.get(model_key) if match_store is not None else None
    if remembered is not None:
        if remembered.repo_id is None:
            return None
        return _build_resolved_aa_model(
            candidate, modal_gpu_catalog, remembered.repo_id,
            exclusions=exclusions, budget=budget,
        )
    try:
        api = hf_api
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi()
        repo_id = _find_unsloth_gguf_match(candidate, api, budget)
    except HubLookupIncomplete:
        # Unknown, not absent: leave the store untouched so the next build
        # asks again rather than inheriting a conclusion nobody established.
        _record_exclusion(
            exclusions, candidate, "",
            "Not checked: every Hugging Face lookup for this model failed, most likely "
            "rate limiting. Refresh the catalog to retry.",
        )
        return None
    except HubBudgetExhausted:
        # Not checked, so not an answer. Recording a miss here would poison
        # the store with a conclusion no request was made to support.
        _record_exclusion(exclusions, candidate, "", UNCHECKED_BUDGET_REASON)
        return None
    except Exception as exc:
        # "The search failed" and "this model has no GGUF weights" are
        # different facts, and only the second is a reason to drop a model
        # without saying so. Hugging Face rate limiting removes whole
        # stretches of the ranking at once, and a silently shorter shortlist
        # gives the reader nothing to notice: a build that lost two thirds of
        # a category to throttling published as though it were complete.
        _record_exclusion(exclusions, candidate, "", _hf_failure_reason(exc, "search for weights"))
        return None
    if match_store is not None:
        match_store.record(model_key, repo_id)
    if repo_id is None:
        return None
    resolved = _build_resolved_aa_model(
        candidate, modal_gpu_catalog, repo_id, exclusions=exclusions, budget=budget
    )
    if resolved is not None and repo_by_model_key is not None and model_key:
        repo_by_model_key[model_key] = repo_id
    return resolved


def _build_resolved_aa_model(
    candidate: AAModelCandidate,
    modal_gpu_catalog: Sequence[ModalGpuSpec],
    repo_id: str,
    *,
    exclusions: list[CatalogExclusion] | None = None,
    budget: HubRequestBudget | None = None,
) -> _ResolvedAAModel | None:
    # Metadata was the unbudgeted half of a build. Reading it past the ceiling
    # gets the whole session throttled, and a throttled read returns an empty
    # weight-size table -- which then reads as "this model's size cannot be
    # determined" rather than as the rate limit it is.
    if budget is not None and not budget.spend(METADATA_REQUEST_COST):
        _record_exclusion(exclusions, candidate, repo_id, UNCHECKED_BUDGET_REASON)
        return None
    try:
        metadata = _fetch_serving_metadata(repo_id)
    except Exception as exc:
        _record_exclusion(exclusions, candidate, repo_id, _hf_failure_reason(exc, "read serving metadata"))
        return None
    size_bucket = _size_bucket_for_parameters(candidate.parameter_count_b)
    if size_bucket is None:
        size_bucket = _size_bucket_from_gguf_metadata(metadata)
    if size_bucket is None:
        _record_exclusion(exclusions, candidate, repo_id, "Model size could not be determined from the available metadata.")
        return None
    compatibility = evaluate_llamacpp_architecture(metadata.architecture)
    if not compatibility.is_supported:
        _record_exclusion(exclusions, candidate, repo_id, compatibility.message)
        return None
    model = ModelCandidate(repo_id=repo_id)
    reasons: list[str] = []
    profiles = _profiles_for_model(
        model,
        modal_gpu_catalog,
        metadata=metadata,
        aa_candidate=candidate,
        size_bucket=size_bucket,
        rejected=reasons,
        budget=budget,
    )
    if not profiles:
        reason = " ".join(dict.fromkeys(reasons)) or "No GGUF quantization has a usable weight-size estimate."
        _record_exclusion(exclusions, candidate, repo_id, reason)
        return None
    return _ResolvedAAModel(
        candidate=candidate,
        repo_id=profiles[0].repo_id,
        size_bucket=size_bucket,
        profiles=tuple(profiles),
    )


def _hf_failure_reason(exc: BaseException, action: str) -> str:
    """Name a Hub failure precisely enough to act on.

    Rate limiting is worth calling out by itself: it is transient, it is the
    reader's own quota, and it explains a whole run of missing models rather
    than one bad repository.
    """

    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 429:
        return (
            "Hugging Face rate limit reached; this model was not checked. "
            "Wait a few minutes and refresh the catalog."
        )
    return f"Could not {action} on Hugging Face. Refresh the catalog to retry."


def _record_exclusion(
    exclusions: list[CatalogExclusion] | None,
    candidate: AAModelCandidate,
    repo_id: str,
    reason: str,
) -> None:
    if exclusions is not None:
        exclusions.append(CatalogExclusion(
            model_id=candidate.aa_model_id or candidate.slug,
            display_name=candidate.name, repo_id=repo_id, reason=reason, rank=candidate.rank,
        ))


def _build_trending_fallback_catalog(
    *,
    model_limit: int,
    modal_gpu_catalog: Sequence[ModalGpuSpec],
    has_live_modal_pricing: bool,
) -> tuple[QuickDeployCatalogInfo, tuple[QuickDeployProfile, ...]]:
    models = list_llamacpp_candidates(mode="trending", limit=_FALLBACK_TRENDING_LIMIT)
    if not models:
        raise RuntimeError(
            "Neither Artificial Analysis rankings nor Hugging Face trending models "
            "were available"
        )

    profiles_by_repo: dict[str, list[QuickDeployProfile]] = {}
    with ThreadPoolExecutor(max_workers=min(8, len(models))) as executor:
        metadata_futures = {
            executor.submit(_safe_fetch_gguf_metadata, model.repo_id): model
            for model in models
        }
        for metadata_future in as_completed(metadata_futures):
            model = metadata_futures[metadata_future]
            metadata = metadata_future.result()
            if metadata is None:
                continue
            group = _profiles_for_model(
                model,
                modal_gpu_catalog,
                metadata=metadata,
                skip_context_lookup=True,
            )
            if group:
                profiles_by_repo[model.repo_id] = group
    _attach_profile_context_lengths(profiles_by_repo, max_workers=8)

    profiles: list[QuickDeployProfile] = []
    selected_models = 0
    for model in models:
        group = profiles_by_repo.get(model.repo_id)
        if not group:
            continue
        profiles.extend(group)
        selected_models += 1
        if selected_models >= model_limit:
            break
    if not profiles:
        raise RuntimeError("No deployable profiles could be built from fallback trending models")

    info = QuickDeployCatalogInfo(
        source_label=(
            "Hugging Face trending GGUF models + "
            f"{'live' if has_live_modal_pricing else 'fallback'} Modal pricing"
        ),
        generated_at=_utc_now_iso(),
        attribution=(
            "Model metadata sourced from Hugging Face; GPU pricing sourced from Modal."
        ),
        is_live=True,
    )
    return (info, tuple(profiles))


# Decimal GB of weights per billion parameters, by bit width. The unquantized
# widths are exact arithmetic. The quantized ones are medians measured across
# published Unsloth GGUF repositories, which is what this catalog matches: they
# absorb the block scales and the higher-precision embedding and output tensors
# GGUF keeps, so the effective figure always sits above the nominal bits/8.
_GB_PER_BILLION_PARAMETERS = {
    32: 4.0,
    16: 2.0,
    8: 1.10,
    6: 0.88,
    5: 0.73,
    4: 0.60,
    3: 0.44,
    2: 0.32,
    1: 0.27,
}


def _total_parameters_b(metadata: GgufQuantMetadata) -> float | None:
    """Estimate total parameters in billions from published GGUF weight sizes.

    Weights are what a repository actually publishes, and for a mixture of
    experts they cover every expert -- an 80B A3B model ships 80B of weights
    and needs all of them resident. Sizing on total parameters is therefore
    the same thing as sizing on what has to fit, and it is what the category
    labels claim to mean.

    Every width is converted to a parameter count and the median is taken,
    rather than trusting the widest or the largest file. Published size tables
    carry junk at both ends and in both directions, and no single row can be
    relied on: Unsloth's Qwen3.8-Flash-Next page lists a 2.79 GB Q4_K_M beside
    a 111 GB one, and its DeepSeek-V4-Flash page lists an 11 GB BF16 beside a
    155 GB Q4 -- a 16-bit copy cannot be smaller than a 4-bit copy, so that row
    is a projector or a single shard. Reading the first match put a 177B model
    in the compact category; reading the widest put a 250B model there. A
    median over eight or so widths survives a bad row at either extreme.
    """

    by_bits: dict[int, list[float]] = {}
    for quant, weights_gb in metadata.vram_gb_by_quant.items():
        if weights_gb <= 0:
            continue
        bits = quant_bits(quant)
        if bits is None or bits not in _GB_PER_BILLION_PARAMETERS:
            continue
        by_bits.setdefault(bits, []).append(weights_gb)
    if not by_bits:
        return None
    # One width contributes one estimate however many files it published, so a
    # width with a dozen variants cannot outvote the rest of the table.
    estimates = [
        median(sizes) / _GB_PER_BILLION_PARAMETERS[bits]
        for bits, sizes in by_bits.items()
    ]
    return median(estimates)


def _size_bucket_from_gguf_metadata(
    metadata: GgufQuantMetadata,
) -> ModelSizeBucket | None:
    """Bucket a model the benchmark feed gave no parameter count for.

    This used to compare gigabytes of quantized weights against thresholds
    named in billions of parameters, so "Compact <=40B" silently admitted
    anything up to roughly 71B. The count is derived first, then bucketed by
    the same rule the feed's own parameter counts go through.
    """

    return _size_bucket_for_parameters(_total_parameters_b(metadata))


def _find_unsloth_gguf_match(
    candidate: AAModelCandidate,
    hf_api: Any,
    budget: HubRequestBudget | None = None,
) -> str | None:
    budget = budget or UnlimitedBudget()
    direct_repo = _repo_id_from_huggingface_url(candidate.huggingface_url)
    if direct_repo and direct_repo.casefold().endswith("-gguf"):
        return direct_repo

    # A probe that errors is skipped, but skipping every probe is not the
    # same answer as checking them and finding nothing.
    attempted = 0
    failed = 0
    for repo_id in _canonical_unsloth_gguf_repo_ids(candidate):
        if not budget.spend():
            raise HubBudgetExhausted(repo_id)
        attempted += 1
        try:
            row = hf_api.model_info(repo_id=repo_id, timeout=_HF_SEARCH_TIMEOUT_SECONDS)
        except Exception:
            failed += 1
            continue
        resolved_repo_id = _repo_id_from_hf_row(row) or repo_id
        if (
            resolved_repo_id.casefold().startswith("unsloth/")
            and resolved_repo_id.casefold().endswith("-gguf")
            and _aa_hf_match_score(candidate, resolved_repo_id) >= 90.0
        ):
            return resolved_repo_id

    rows: list[Any] = []
    # The fan-out is claimed up front rather than per search: a matcher handed
    # half its searches would report a miss it had not established.
    search_terms = _ranked_hf_search_terms(candidate)
    if search_terms and not budget.spend(len(search_terms)):
        raise HubBudgetExhausted(candidate.name)
    search_executor = ThreadPoolExecutor(max_workers=_HF_SEARCH_WORKERS)
    try:
        search_futures = [
            search_executor.submit(
                _list_unsloth_gguf_search,
                hf_api,
                search,
            )
            for search in search_terms
        ]
        for search_future in as_completed(search_futures):
            attempted += 1
            try:
                rows.extend(search_future.result())
            except Exception:
                failed += 1
                continue
            if _has_strong_unsloth_gguf_match(candidate, rows):
                break
    finally:
        search_executor.shutdown(wait=False, cancel_futures=True)

    scored: list[tuple[float, str]] = []
    seen: set[str] = set()
    for row in rows:
        repo_id = _repo_id_from_hf_row(row)
        repo_key = repo_id.casefold()
        if (
            not repo_id
            or repo_key in seen
            or not repo_key.startswith("unsloth/")
            or not repo_key.endswith("-gguf")
        ):
            continue
        seen.add(repo_key)
        score = _aa_hf_match_score(candidate, repo_id)
        if score >= 90.0:
            scored.append((score, repo_id))
    if not scored:
        if attempted and failed == attempted:
            # Every probe and every search errored, so nothing was actually
            # checked. Reporting that as a miss is how a throttled run wrote
            # "no weights exist" into the match store and was believed for
            # days afterwards.
            raise HubLookupIncomplete(candidate.name)
        return None
    best = min(scored, key=lambda item: _aa_hf_match_rank_key(candidate, item[1]))
    return best[1]


def _aa_hf_match_rank_key(
    candidate: AAModelCandidate,
    repo_id: str,
) -> tuple[float, int, str]:
    """Deterministic order: score, then fewest extra tokens, then name."""

    repo_key = _model_key(_repo_model_name(repo_id))
    candidate_keys = {
        key
        for key in (
            _model_key(candidate.name),
            _model_key(candidate.slug),
            _model_key(
                _repo_model_name(
                    _repo_id_from_huggingface_url(candidate.huggingface_url) or ""
                )
            ),
        )
        if key
    }
    extra_tokens = min(
        (abs(len(repo_key) - len(key)) for key in candidate_keys),
        default=len(repo_key),
    )
    return (-_aa_hf_match_score(candidate, repo_id), extra_tokens, repo_id.casefold())


def _canonical_unsloth_gguf_repo_ids(
    candidate: AAModelCandidate,
) -> tuple[str, ...]:
    values = [
        _repo_model_name(
            _repo_id_from_huggingface_url(candidate.huggingface_url) or ""
        ),
        _strip_creator_prefix(candidate.name, candidate.creator_name),
        candidate.name,
    ]
    repo_ids: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", value).strip()
        cleaned = re.sub(r"(?i)-gguf$", "", cleaned)
        repo_name = re.sub(r"[-_\s/]+", "-", cleaned).strip("-")
        repo_id = f"unsloth/{repo_name}-GGUF" if repo_name else ""
        key = repo_id.casefold()
        if repo_id and key not in seen:
            seen.add(key)
            repo_ids.append(repo_id)
    return tuple(repo_ids)


def _ranked_hf_search_terms(candidate: AAModelCandidate) -> list[str]:
    values = [
        _repo_model_name(_repo_id_from_huggingface_url(candidate.huggingface_url) or ""),
        _strip_creator_prefix(candidate.name, candidate.creator_name),
        candidate.name,
        candidate.slug,
    ]
    terms: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", value).strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            terms.append(cleaned)
    return terms


def _list_unsloth_gguf_search(hf_api: Any, search: str) -> list[Any]:
    """Run one Unsloth-scoped Hub search with lightweight result rows.

    ``limit=10`` matches the scorer: it only needs the best exact-token
    hit, and the slimmer ``expand`` payload keeps each search to one small
    response instead of 25 full model cards.
    """

    kwargs: dict[str, Any] = {
        "author": "unsloth",
        "search": search,
        "limit": _HF_SEARCH_LIMIT,
        "expand": ["siblings"],
    }
    try:
        return list(hf_api.list_models(**kwargs))
    except TypeError:
        # Older huggingface_hub releases lack author/expand; retry unscoped.
        fallback = {"search": search, "limit": _HF_SEARCH_LIMIT}
        try:
            return list(hf_api.list_models(**fallback))
        except Exception:
            return []
    except Exception:
        return []


def _has_strong_unsloth_gguf_match(
    candidate: AAModelCandidate,
    rows: Sequence[Any],
) -> bool:
    """Return True once an exact-token Unsloth GGUF match is available."""

    for row in rows:
        repo_id = _repo_id_from_hf_row(row)
        repo_key = repo_id.casefold()
        if (
            not repo_id
            or not repo_key.startswith("unsloth/")
            or not repo_key.endswith("-gguf")
        ):
            continue
        if _aa_hf_match_score(candidate, repo_id) >= 100.0:
            return True
    return False


def _repo_id_from_huggingface_url(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.netloc.casefold() not in {"huggingface.co", "www.huggingface.co"}:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return None
    return f"{parts[0]}/{parts[1]}"


def _repo_id_from_hf_row(row: Any) -> str:
    if isinstance(row, dict):
        return clean_string(row.get("id") or row.get("modelId"))
    return clean_string(getattr(row, "id", None) or getattr(row, "modelId", None))


def _aa_hf_match_score(candidate: AAModelCandidate, repo_id: str) -> float:
    candidate_keys = {
        _model_key(candidate.name),
        _model_key(candidate.slug),
        _model_key(
            _repo_model_name(
                _repo_id_from_huggingface_url(candidate.huggingface_url) or ""
            )
        ),
    }
    repo_key = _model_key(_repo_model_name(repo_id))
    scores: list[float] = []
    for candidate_key in candidate_keys:
        if not candidate_key:
            continue
        if candidate_key == repo_key:
            scores.append(100.0)
        elif candidate_key in repo_key or repo_key in candidate_key:
            scores.append(95.0)
        else:
            scores.append(SequenceMatcher(None, candidate_key, repo_key).ratio() * 100.0)
    return max(scores or [0.0])


def _strip_creator_prefix(name: str, creator_name: str) -> str:
    text = name.strip()
    creator = creator_name.strip()
    if not text or not creator:
        return text
    creator_pattern = re.escape(creator).replace(r"\.", r"\.?")
    return re.sub(
        rf"(?i)^{creator_pattern}\s*(?:[:/\-|]|\s)\s*",
        "",
        text,
    ).strip()


def _repo_model_name(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1]


def _fetch_modal_gpu_catalog_or_fallback() -> tuple[tuple[ModalGpuSpec, ...], bool]:
    try:
        catalog = tuple(fetch_modal_gpu_catalog())
    except Exception:
        return ((), False)
    return (catalog, any(spec.price_per_hour_usd is not None for spec in catalog))


def _safe_fetch_gguf_metadata(repo_id: str) -> GgufQuantMetadata | None:
    """Fetch GGUF metadata without MTP inspection or context lookups."""

    try:
        return fetch_gguf_quant_metadata(repo_id, inspect_serving=True)
    except Exception:
        return None


def _attach_profile_context_lengths(
    profiles_by_repo: dict[str, list[QuickDeployProfile]],
    *,
    max_workers: int = 8,
) -> None:
    """Fill per-repo context lengths after profiles are selectable.

    Only upgrades rows still carrying the conservative default; AA rows
    already know their context window and are left untouched.
    """

    pending = [
        repo_id
        for repo_id, group in profiles_by_repo.items()
        if group and all(profile.max_context_tokens == DEFAULT_CONTEXT_TOKENS for profile in group)
    ]
    if not pending:
        return
    context_by_repo: dict[str, int | None] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(pending))) as executor:
        futures = {
            executor.submit(_safe_fetch_model_max_context, repo_id): repo_id
            for repo_id in pending
        }
        for future in as_completed(futures):
            context_by_repo[futures[future]] = future.result()
    for repo_id, context_tokens in context_by_repo.items():
        if context_tokens is None:
            continue
        upgraded_group: list[QuickDeployProfile] = []
        for profile in profiles_by_repo[repo_id]:
            requirements = (
                replace(profile.serving_requirements, context_tokens=context_tokens)
                if profile.serving_requirements is not None
                else serving_requirements(context_tokens)
            )
            tuning = profile.runtime_tuning or tuning_for_objective(
                requirements.objective,
                speculative_decoding=profile.speculative_decoding,
            )
            memory = profile.memory_estimate
            if memory is not None:
                old_context = max(1, profile.max_context_tokens)
                scale = context_tokens / old_context
                kv_cache_gb = memory.kv_cache_gb * scale
                # A graph built without flash attention reserves scores for the
                # whole context, so this term follows the window just as the
                # cache does -- and the physical batch it was chosen against
                # has to be re-chosen with it, or a longer window silently
                # scales a batch that no longer fits.
                scratch_gb = memory.attention_scratch_gb * scale
                if scratch_gb > 0:
                    per_token_gb = scratch_gb / max(1, tuning.ubatch_size)
                    ubatch = ubatch_for_attention_scratch(
                        per_token_gb, max_ubatch=tuning.ubatch_size
                    )
                    scratch_gb = per_token_gb * ubatch
                    tuning = replace(tuning, ubatch_size=ubatch)
                memory = replace(
                    memory,
                    kv_cache_gb=round(kv_cache_gb, 3),
                    attention_scratch_gb=round(scratch_gb, 3),
                    total_gb=round(
                        memory.weights_gb
                        + kv_cache_gb
                        + memory.compute_gb
                        + scratch_gb
                        + memory.speculative_gb
                        + memory.reserve_gb,
                        3,
                    ),
                )
            upgraded_group.append(
                replace(
                    profile,
                    max_context_tokens=context_tokens,
                    server_args=compile_server_args(requirements, tuning),
                    required_vram_gb=(memory.total_gb if memory is not None else profile.required_vram_gb),
                    serving_requirements=requirements,
                    runtime_tuning=tuning,
                    memory_estimate=memory,
                )
            )
        profiles_by_repo[repo_id] = upgraded_group


def _safe_fetch_model_max_context(repo_id: str) -> int | None:
    try:
        return fetch_model_max_context(repo_id)
    except Exception:
        return None


def attach_quick_deploy_mtp_recommendations(
    profiles: Sequence[QuickDeployProfile],
    *,
    max_workers: int = 8,
    info: QuickDeployCatalogInfo | None = None,
    cache_path: Path | None = None,
) -> tuple[QuickDeployProfile, ...]:
    """Attach MTP recommendations to catalog profiles without blocking.

    Intended as a lazy second pass after the catalog is already active:
    the 1 MiB GGUF range probes dominate cold-start latency but only feed
    the speculative-decoding toggle on the confirm screen. Deploy-time
    preflight revalidates MTP anyway, so a missing recommendation here is
    always safe to recompute later.

    Pass ``info`` to write the upgraded profiles back to the catalog snapshot.
    The snapshot the build wrote predates this pass, so without it every launch
    reprobes every repository and opens the confirm screen with no MTP toggle
    until the probes land.
    """

    pending = [profile for profile in profiles if profile.speculative_decoding is None]
    if not pending:
        return tuple(profiles)
    metadata_by_repo: dict[str, GgufQuantMetadata | None] = {}
    repos = list({profile.repo_id for profile in pending})
    with ThreadPoolExecutor(max_workers=min(max_workers, len(repos))) as executor:
        futures = {
            executor.submit(
                fetch_gguf_quant_metadata,
                repo_id,
                inspect_mtp=True,
                inspect_serving=True,
            ): repo_id
            for repo_id in repos
        }
        for future in as_completed(futures):
            try:
                metadata_by_repo[futures[future]] = future.result()
            except Exception:
                metadata_by_repo[futures[future]] = None
    upgraded: list[QuickDeployProfile] = []
    for profile in profiles:
        if profile.speculative_decoding is not None:
            upgraded.append(profile)
            continue
        metadata = metadata_by_repo.get(profile.repo_id)
        recommendation = _mtp_recommendation(metadata) if metadata is not None else None
        if recommendation is None or metadata is None:
            upgraded.append(profile)
            continue
        requirements = profile.serving_requirements or serving_requirements(
            profile.max_context_tokens
        )
        tuning = tuning_for_objective(
            requirements.objective,
            speculative_decoding=recommendation,
        )
        weights_gb = _required_vram_for_quant(metadata, profile.quant)
        memory = (
            estimate_memory(
                metadata,
                weights_gb=weights_gb,
                requirements=requirements,
                tuning=tuning,
            )
            if weights_gb is not None
            else profile.memory_estimate
        )
        upgraded.append(
            replace(
                profile,
                speculative_decoding=recommendation,
                server_args=compile_server_args(requirements, tuning),
                runtime_tuning=tuning,
                memory_estimate=memory,
                required_vram_gb=(memory.total_gb if memory is not None else profile.required_vram_gb),
            )
        )
    result = tuple(upgraded)
    if info is not None and result != tuple(profiles):
        _write_quick_deploy_catalog_cache(info, result, cache_path=cache_path)
    return result


def _verified_mtp_variant(
    repo_id: str,
    metadata: GgufQuantMetadata,
    budget: HubRequestBudget | None,
) -> tuple[str, GgufQuantMetadata] | None:
    """Find an embedded-head variant and verify every quant we would offer."""

    if (
        not repo_id.casefold().startswith("unsloth/")
        or not repo_id.casefold().endswith("-gguf")
        or repo_id.casefold().endswith("-mtp-gguf")
        or _mtp_recommendation(metadata) is not None
        or not evaluate_llamacpp_mtp(metadata.architecture, 1).is_supported
    ):
        return None
    variant_repo = f"{repo_id[:-5]}-MTP-GGUF"
    request_budget = budget if budget is not None else UnlimitedBudget()
    try:
        if not request_budget.spend(METADATA_REQUEST_COST):
            return None
        variant = fetch_gguf_quant_metadata(
            variant_repo, inspect_serving=True, inspect_mtp=True,
        )
        if (
            variant.architecture != metadata.architecture
            or _mtp_recommendation(variant) is None
        ):
            return None
        quants = _selected_quants(variant)
        if not quants:
            return None
        for quant in quants:
            if not request_budget.spend(METADATA_REQUEST_COST):
                return None
            evidence = fetch_gguf_quant_metadata(
                variant_repo, inspect_mtp=True, mtp_quant=quant,
            )
            if (
                evidence.architecture != variant.architecture
                or _mtp_recommendation(evidence) is None
            ):
                return None
    except Exception as exc:
        # Optional acceleration must not remove an otherwise deployable model.
        log_debug(f"Could not verify MTP variant {variant_repo}: {exc}")
        return None
    return variant_repo, variant


def _profiles_for_model(
    model: ModelCandidate,
    modal_gpu_catalog: Sequence[ModalGpuSpec],
    *,
    metadata: GgufQuantMetadata | None = None,
    aa_candidate: AAModelCandidate | None = None,
    size_bucket: ModelSizeBucket | None = None,
    skip_context_lookup: bool = False,
    rejected: list[str] | None = None,
    budget: HubRequestBudget | None = None,
    prefer_mtp: bool = True,
) -> list[QuickDeployProfile]:
    if metadata is None:
        try:
            metadata = _fetch_serving_metadata(model.repo_id)
        except Exception:
            return []
    variant = _verified_mtp_variant(model.repo_id, metadata, budget) if prefer_mtp else None
    if variant is not None:
        variant_repo, variant_metadata = variant
        accelerated = _profiles_for_model(
            ModelCandidate(repo_id=variant_repo),
            modal_gpu_catalog,
            metadata=variant_metadata,
            aa_candidate=aa_candidate,
            size_bucket=size_bucket,
            skip_context_lookup=skip_context_lookup,
            budget=budget,
            prefer_mtp=False,
        )
        if accelerated:
            return accelerated
    compatibility = evaluate_llamacpp_architecture(metadata.architecture)
    if not compatibility.is_supported:
        return []
    quants = _selected_quants(metadata)
    if not quants:
        return []
    max_context_tokens = metadata.context_length
    if max_context_tokens is None and not skip_context_lookup:
        try:
            max_context_tokens = fetch_model_max_context(model.repo_id)
        except Exception:
            max_context_tokens = None
    if max_context_tokens is None and aa_candidate is not None:
        # Retain catalog coverage for repositories whose GGUF metadata is not
        # yet exposed by the Hub. The lower-confidence planner estimate remains
        # subject to runtime attestation before the endpoint is published.
        max_context_tokens = aa_candidate.max_context_tokens
    context_tokens = max_context_tokens or DEFAULT_CONTEXT_TOKENS
    display_name = aa_candidate.name if aa_candidate else _display_name(model.repo_id)
    slug_hint = slugify_instance_name(display_name)
    profiles: list[QuickDeployProfile] = []
    for quant in quants:
        profiles.extend(
            _profiles_for_quant(
                repo_id=model.repo_id,
                display_name=display_name,
                slug_hint=slug_hint,
                context_tokens=context_tokens,
                quant=quant,
                metadata=metadata,
                modal_gpu_catalog=modal_gpu_catalog,
                aa_candidate=aa_candidate,
                size_bucket=size_bucket,
                llamacpp_runtime_id=compatibility.runtime_id,
                speculative_decoding=_mtp_recommendation(metadata),
                rejected=rejected,
            )
        )
    return profiles


def _selected_quants(metadata: GgufQuantMetadata) -> list[str]:
    available = {
        _quant_key(quant): quant
        for quant in metadata.quantizations
        if _required_vram_for_quant(metadata, quant) is not None
    }
    if not available:
        return []
    selected: list[str] = []
    # Quality leads. Both quantizations are still built -- the low-VRAM one is
    # what makes an economy option possible at all, and is the only build that
    # fits for the largest models -- but the preferred width comes first, so
    # every consumer that reads ``profiles[0]`` gets the better weights.
    for preferred in _PREFERRED_QUANT_ORDER:
        quant = available.get(_quant_key(preferred))
        if quant:
            selected.append(quant)
            break
    low_vram = available.get(_quant_key(LOW_VRAM_QUANT))
    if low_vram and low_vram not in selected:
        selected.append(low_vram)
    if not selected:
        selected.append(next(iter(available.values())))
    return selected


def _stable_profile_id(repo_id: str, quant_slug: str, resource_tier: str) -> str:
    """Build a catalog id from identity alone, never from a planner decision.

    The id used to embed the GPU the planner had selected, so improving
    placement renamed every profile: fixing full-context sizing moved a 27B
    model off an L4 and turned ``...-cheap-l4`` into ``...-cheap-rtx-pro-6000``
    overnight. Anything holding an id -- a saved deployment, a script, a
    benchmark run, a half-finished flow in the UI -- broke.

    The repository is used rather than the display name because benchmark feeds
    rename models ("Qwen3.8 27B" gaining an "(xhigh)" suffix) without the
    underlying artifact changing at all. ``resource_tier`` stays because it is
    an input to selection -- which GPU pool to search -- not an output of it.
    """

    name = repo_id.strip().rsplit("/", 1)[-1]
    for suffix in ("-GGUF", "-gguf"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return f"{slugify_instance_name(name)}-{quant_slug}-{resource_tier}"


def _profiles_for_quant(
    *,
    repo_id: str,
    display_name: str,
    slug_hint: str,
    context_tokens: int,
    quant: str,
    metadata: GgufQuantMetadata,
    modal_gpu_catalog: Sequence[ModalGpuSpec],
    aa_candidate: AAModelCandidate | None = None,
    size_bucket: ModelSizeBucket | None = None,
    llamacpp_runtime_id: str,
    speculative_decoding: SpeculativeDecodingConfig | None = None,
    rejected: list[str] | None = None,
) -> list[QuickDeployProfile]:
    weights_gb = _required_vram_for_quant(metadata, quant)
    if weights_gb is None:
        return []
    requirements = serving_requirements(
        context_tokens,
        objective=ServingObjective.GENERAL_PURPOSE,
    )
    runtime_tuning = tuning_for_objective(
        requirements.objective,
        speculative_decoding=speculative_decoding,
    )
    runtime_tuning = tuning_for_architecture(runtime_tuning, metadata.architecture)
    runtime_tuning = tuning_for_attention_scratch(
        runtime_tuning,
        context_tokens=requirements.context_tokens,
        attention_head_count=attention_head_count(metadata),
    )
    memory_estimate = estimate_memory(
        metadata,
        weights_gb=weights_gb,
        requirements=requirements,
        tuning=runtime_tuning,
    )
    if metadata.architecture in {"kimi-k3", "kimi-linear", "glm5next"} and memory_estimate.source == "conservative-fallback":
        if rejected is not None:
            rejected.append("Hybrid attention memory layout is incomplete; GPU fit cannot be verified. Refresh model metadata to retry.")
        return []
    required_vram_gb = memory_estimate.total_gb
    per_device_overhead_gb = (
        memory_estimate.compute_gb + memory_estimate.attention_scratch_gb
    )
    selections = (
        (
            "cheap",
            _select_gpu_shape(
                quant,
                required_vram_gb,
                modal_gpu_catalog,
                per_device_overhead_gb=per_device_overhead_gb,
                layer_count=memory_estimate.total_layer_count,
            ),
        ),
        (
            "rtx-pro",
            _select_gpu_shape(
                quant,
                required_vram_gb,
                modal_gpu_catalog,
                gpu_type="RTX-PRO-6000",
                per_device_overhead_gb=per_device_overhead_gb,
                layer_count=memory_estimate.total_layer_count,
            ),
        ),
        (
            "b200",
            _select_gpu_shape(
                quant,
                required_vram_gb,
                modal_gpu_catalog,
                gpu_type="B200",
                per_device_overhead_gb=per_device_overhead_gb,
                layer_count=memory_estimate.total_layer_count,
            ),
        ),
    )
    profiles: list[QuickDeployProfile] = []
    profile_index_by_shape: dict[tuple[str, int], int] = {}
    for resource_tier, selection in selections:
        if selection is None:
            continue
        shape = (selection.gpu_type, selection.gpu_count)
        existing_index = profile_index_by_shape.get(shape)
        if existing_index is not None:
            existing = profiles[existing_index]
            profiles[existing_index] = replace(
                existing,
                profile_label=_join_unique(
                    existing.profile_label,
                    _RESOURCE_TIER_DESCRIPTIONS[resource_tier],
                    separator=" / ",
                ),
                resource_tier_label=_join_unique(
                    existing.resource_tier_label or "",
                    _RESOURCE_TIER_LABELS[resource_tier],
                ),
            )
            continue
        profile_index_by_shape[shape] = len(profiles)
        quant_slug = _quant_slug(quant)
        profiles.append(
            QuickDeployProfile(
                id=_stable_profile_id(repo_id, quant_slug, resource_tier),
                display_name=display_name,
                repo_id=repo_id,
                quant=selection.quant,
                gpu_type=selection.gpu_type,
                gpu_count=selection.gpu_count,
                profile_label=_RESOURCE_TIER_DESCRIPTIONS[resource_tier],
                resource_tier=resource_tier,
                resource_tier_label=_RESOURCE_TIER_LABELS[resource_tier],
                approx_cost_per_hour_usd=round(selection.cost_per_hour_usd, 2),
                max_context_tokens=context_tokens,
                instance_slug_hint=f"{slug_hint}-{quant_slug}-{resource_tier}",
                summary=(
                    f"Artificial Analysis-ranked {_MODEL_SIZE_LABELS[size_bucket]} "
                    "open-weight model matched to verified Hugging Face GGUF weights."
                    if aa_candidate is not None and size_bucket is not None
                    else "Live Hugging Face trending GGUF model matched to current "
                    "Modal GPU pricing."
                ),
                server_args=compile_server_args(requirements, runtime_tuning),
                required_vram_gb=round(selection.required_vram_gb, 1),
                gpu_memory_gb=_GPU_MEMORY_GB.get(selection.gpu_type),
                source_label=(
                    "Artificial Analysis"
                    if aa_candidate is not None
                    else "Hugging Face trending"
                ),
                aa_model_id=(aa_candidate.aa_model_id or None) if aa_candidate else None,
                aa_model_name=aa_candidate.name if aa_candidate else None,
                aa_model_slug=aa_candidate.slug or None if aa_candidate else None,
                aa_coding_score=aa_candidate.coding_score if aa_candidate else None,
                aa_intelligence_score=(
                    aa_candidate.intelligence_score if aa_candidate else None
                ),
                aa_rank=aa_candidate.rank if aa_candidate else None,
                model_size_label=(
                    _MODEL_SIZE_LABELS[size_bucket] if size_bucket is not None else None
                ),
                gguf_architecture=metadata.architecture,
                llamacpp_runtime_id=llamacpp_runtime_id,
                speculative_decoding=speculative_decoding,
                serving_requirements=requirements,
                runtime_tuning=runtime_tuning,
                memory_estimate=memory_estimate,
            )
        )
    if not profiles and rejected is not None:
        reason = (
            f"{quant}: estimated {required_vram_gb:,.0f} GB at {context_tokens:,} tokens "
            "does not fit any priced catalog topology (maximum 8 GPUs)."
        )
        if memory_estimate.attention_scratch_gb > 0:
            # Naming the term matters here: it is the one requirement extra
            # GPUs cannot share, so the shortfall does not look like one more
            # card would close it.
            reason += (
                f" {memory_estimate.attention_scratch_gb:,.0f} GB of that is attention"
                " scratch every GPU needs in full, because this architecture's"
                " runtime requires flash attention off."
            )
        rejected.append(reason)
    return profiles


def _mtp_recommendation(
    metadata: GgufQuantMetadata,
) -> SpeculativeDecodingConfig | None:
    capability = metadata.mtp
    if capability is None or capability.status != GgufMtpStatus.SUPPORTED:
        return None
    layers = capability.nextn_predict_layers
    decision = evaluate_llamacpp_mtp(metadata.architecture, layers)
    if not decision.is_supported or layers is None:
        return None
    return SpeculativeDecodingConfig(
        method=SpeculativeDecodingMethod.MTP,
        num_speculative_tokens=DEFAULT_MTP_DRAFT_TOKENS,
        nextn_predict_layers=layers,
    )


def _select_gpu_shape(
    quant: str,
    required_vram_gb: float,
    modal_gpu_catalog: Sequence[ModalGpuSpec],
    *,
    gpu_type: str | None = None,
    per_device_overhead_gb: float = 0.0,
    layer_count: int | None = None,
) -> _GpuSelection | None:
    prices = _price_by_gpu(modal_gpu_catalog)
    available = _available_gpu_types(modal_gpu_catalog)
    # Graph memory is replicated on every device, so only the remainder of the
    # requirement gets smaller as GPUs are added. Dividing the whole figure
    # would keep offering a placement that llama.cpp then refuses at startup.
    shardable_gb = max(0.0, required_vram_gb - per_device_overhead_gb)
    candidates: list[tuple[float, int, float, str]] = []
    for candidate_gpu in available:
        if gpu_type is not None and candidate_gpu != gpu_type:
            continue
        memory_gb = _GPU_MEMORY_GB.get(candidate_gpu)
        if memory_gb is None:
            continue
        for gpu_count in range(1, 9):
            reserve_per_gpu = max(2.0, memory_gb * 0.05)
            # The busiest device is the one that has to fit, and layers are
            # indivisible, so an uneven split is sized on its larger half.
            busiest = max(
                per_device_requirements(
                    shardable_gb=shardable_gb,
                    per_device_gb=per_device_overhead_gb + reserve_per_gpu,
                    gpu_count=gpu_count,
                    layer_count=layer_count,
                )
            )
            if busiest > memory_gb:
                continue
            cost = prices.get(
                candidate_gpu,
                _FALLBACK_GPU_PRICE_PER_HOUR["RTX-PRO-6000"],
            ) * gpu_count
            candidates.append((cost, gpu_count, -memory_gb, candidate_gpu))
            break
    if not candidates:
        return None
    cost, gpu_count, _negative_memory, selected_gpu = min(candidates)
    return _GpuSelection(
        quant=quant,
        gpu_type=selected_gpu,
        gpu_count=gpu_count,
        cost_per_hour_usd=cost,
        required_vram_gb=required_vram_gb,
    )


def _available_gpu_types(modal_gpu_catalog: Sequence[ModalGpuSpec]) -> list[str]:
    """Return catalog GPU shapes that can actually back a priced profile.

    Entries without a known VRAM size (e.g. future ``B300`` shapes) or
    without a usable hourly price (e.g. unpriced ``H100!``/``H200`` rows)
    are skipped: offering them produces profiles whose cost math silently
    falls back to RTX pricing and whose fulfillment lands on ``price n/a``
    placements.
    """
    prices = _live_price_by_gpu(modal_gpu_catalog)
    values = [
        entry.value.strip()
        for entry in modal_gpu_catalog
        if entry.value.strip()
        and _GPU_MEMORY_GB.get(entry.value.strip()) is not None
        and (prices.get(entry.value.strip()) or 0) > 0
    ]
    if values:
        return values
    return [
        value
        for value in _GPU_MEMORY_GB
        if value in {"T4", "L4", "A100", "L40S", "RTX-PRO-6000", "H100", "H200", "B200"}
    ]


def _price_by_gpu(modal_gpu_catalog: Sequence[ModalGpuSpec]) -> dict[str, float]:
    prices = dict(_FALLBACK_GPU_PRICE_PER_HOUR)
    for entry in modal_gpu_catalog:
        if entry.price_per_hour_usd is not None and entry.price_per_hour_usd > 0:
            prices[entry.value] = entry.price_per_hour_usd
    return prices


def _live_price_by_gpu(modal_gpu_catalog: Sequence[ModalGpuSpec]) -> dict[str, float]:
    """Return only catalog-reported prices, without static fallbacks.

    Used to decide which GPU shapes are genuinely orderable right now. The
    fallback table in :func:`_price_by_gpu` keeps cost math working for
    shapes Modal omits, but it must not resurrect shapes Modal explicitly
    lists without a price.
    """
    return {
        entry.value.strip(): entry.price_per_hour_usd
        for entry in modal_gpu_catalog
        if entry.value.strip()
        and entry.price_per_hour_usd is not None
        and entry.price_per_hour_usd > 0
    }


def _required_vram_for_quant(
    metadata: GgufQuantMetadata,
    quant: str,
) -> float | None:
    expected = _quant_key(quant)
    for candidate, value in metadata.vram_gb_by_quant.items():
        if _quant_key(candidate) == expected and value > 0:
            return float(value)
    return None


def _display_name(repo_id: str) -> str:
    name = repo_id.split("/", 1)[-1]
    name = re.sub(r"(?i)-gguf$", "", name)
    return re.sub(r"[-_]+", " ", name).strip() or repo_id


def _quant_key(value: str) -> str:
    normalized = value.strip().upper()
    return f"UD-{normalized[3:]}" if normalized.startswith("UD_") else normalized


def _quant_slug(value: str) -> str:
    slug = _quant_key(value).casefold().removeprefix("ud-")
    return (
        slug.replace("_k_xl", "xl")
        .replace("_k_m", "m")
        .replace("_k_s", "s")
        .replace("_", "-")
        .strip("-")
    )


def _join_unique(existing: str, addition: str, *, separator: str = "/") -> str:
    values: list[str] = []
    for value in (existing, addition):
        for part in value.split(separator):
            label = part.strip()
            if label and label not in values:
                values.append(label)
    return separator.join(values)


def _utc_now_iso() -> str:
    return (
        datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
