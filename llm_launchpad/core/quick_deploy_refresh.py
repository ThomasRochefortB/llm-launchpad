"""Build a live Deploy catalog for the TUI."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
from threading import Lock
import time
from typing import Any, Literal, Sequence
from urllib.parse import urlparse

from .artificial_analysis_auth import resolve_artificial_analysis_api_key
from .config import SETTINGS_DIR
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
from .naming import slugify_instance_name
from .quick_deploy import QuickDeployCatalogInfo, QuickDeployProfile
from .llamacpp_planner import (
    compile_server_args,
    estimate_memory,
    serving_requirements,
    tuning_for_objective,
)
from .runtime_support import evaluate_llamacpp_architecture, evaluate_llamacpp_mtp
from ..protocol.enums import ServingObjective, SpeculativeDecodingMethod
from ..protocol.models import (
    MemoryEstimate,
    RuntimeTuning,
    ServingRequirements,
    SpeculativeDecodingConfig,
)

DEFAULT_MODEL_LIMIT = 3
DEFAULT_CANDIDATE_LIMIT = 80
_AA_RESOLUTION_WORKERS = 24
_FALLBACK_TRENDING_LIMIT = 8
_HF_SEARCH_WORKERS = 4
_HF_SEARCH_LIMIT = 10
_HF_SEARCH_TIMEOUT_SECONDS = 10.0
DEFAULT_CONTEXT_TOKENS = 65_536
LOW_VRAM_QUANT = "UD-Q2_K_XL"
AA_PRO_MODELS_URL = "https://artificialanalysis.ai/api/v2/language/models"
AA_FREE_MODELS_URL = "https://artificialanalysis.ai/api/v2/language/models/free"
AA_CACHE_PATH = SETTINGS_DIR / "artificial_analysis_models.json"
AA_CACHE_TTL = timedelta(hours=24)
AA_ATTRIBUTION = "Benchmark data sourced from Artificial Analysis: https://artificialanalysis.ai/"


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
QUICK_DEPLOY_CATALOG_CACHE_SCHEMA_VERSION = 2
_AA_CACHE_LOCK = Lock()

ModelSizeBucket = Literal["compact", "medium", "large"]
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
class AAModelCandidate:
    """Normalized Artificial Analysis Intelligence Index row."""

    aa_model_id: str
    name: str
    slug: str
    creator_name: str
    coding_score: float | None
    intelligence_score: float
    rank: int
    parameter_count_b: float | None = None
    max_context_tokens: int | None = None
    huggingface_url: str | None = None


@dataclass(frozen=True)
class ArtificialAnalysisAuthStatus:
    """Best-effort status for the configured Artificial Analysis API key."""

    authenticated: bool
    tier: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class _AARankings:
    candidates: tuple[AAModelCandidate, ...]
    freshness: Literal["live", "cached"]
    tier: str | None = None


@dataclass(frozen=True)
class _AACacheEntry:
    fetched_at: datetime
    payload: dict[str, Any]
    key_fingerprint: str | None = None


@dataclass(frozen=True)
class _ResolvedAAModel:
    candidate: AAModelCandidate
    repo_id: str
    size_bucket: ModelSizeBucket
    profiles: tuple[QuickDeployProfile, ...]


_AA_AUTH_STATUS_BY_KEY: dict[str, ArtificialAnalysisAuthStatus] = {}


def build_live_quick_deploy_catalog(
    *,
    model_limit: int = DEFAULT_MODEL_LIMIT,
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

    if rankings is not None:
        profiles = _profiles_from_aa_rankings(
            rankings.candidates,
            modal_gpu_catalog,
            model_limit=normalized_model_limit,
            candidate_limit=normalized_candidate_limit,
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
    """Return True when a cached catalog snapshot is fresh enough to trust."""

    generated_at = _clean_string(info.generated_at)
    if not generated_at:
        return False
    try:
        fetched_at = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    current_time = now or datetime.now(timezone.utc)
    return current_time - fetched_at.astimezone(timezone.utc) <= QUICK_DEPLOY_CATALOG_CACHE_TTL


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
                "speculative_gb": memory.speculative_gb,
                "reserve_gb": memory.reserve_gb,
                "total_gb": memory.total_gb,
                "per_device_required_gb": list(memory.per_device_required_gb),
                "confidence": memory.confidence,
                "source": memory.source,
                "total_layer_count": memory.total_layer_count,
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
            speculative_gb=float(payload.get("speculative_gb", 0.0)),
            reserve_gb=float(payload.get("reserve_gb", 0.0)),
            total_gb=float(payload["total_gb"]),
            per_device_required_gb=tuple(
                float(value) for value in payload.get("per_device_required_gb", ())
            ),
            confidence=float(payload.get("confidence", 0.0)),
            source=str(payload.get("source") or "estimated"),
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


def fetch_artificial_analysis_models(
    api_key: str,
    *,
    timeout: float = 20.0,
) -> dict[str, Any]:
    """Fetch every AAI language-model page, using Free when Pro is unavailable."""

    normalized_key = api_key.strip()
    if not normalized_key:
        raise ValueError("An Artificial Analysis API key is required")
    try:
        return _fetch_aa_pages(AA_PRO_MODELS_URL, normalized_key, timeout=timeout)
    except _AAHttpError as exc:
        if exc.status_code != 403:
            raise
    return _fetch_aa_pages(AA_FREE_MODELS_URL, normalized_key, timeout=timeout)


def normalize_aa_model_candidates(payload: Any) -> tuple[AAModelCandidate, ...]:
    """Normalize, deduplicate, and rank AAI rows by Intelligence Index."""

    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return ()

    deduplicated: dict[str, AAModelCandidate] = {}
    for row in rows:
        if not isinstance(row, dict) or _is_explicitly_closed_weight(row):
            continue
        name = _clean_string(row.get("name"))
        slug = _clean_string(row.get("slug"))
        if not name and not slug:
            continue
        evaluations = row.get("evaluations")
        evaluations = evaluations if isinstance(evaluations, dict) else {}
        coding_score = _optional_float(
            evaluations.get("artificial_analysis_coding_index")
        )
        intelligence_score = _optional_float(
            evaluations.get("artificial_analysis_intelligence_index")
        )
        if intelligence_score is None:
            continue
        creator = row.get("model_creator")
        creator_name = (
            _clean_string(creator.get("name")) if isinstance(creator, dict) else ""
        )
        candidate = AAModelCandidate(
            aa_model_id=_clean_string(row.get("id")),
            name=name or slug,
            slug=slug,
            creator_name=creator_name,
            coding_score=coding_score,
            intelligence_score=intelligence_score,
            rank=0,
            parameter_count_b=_extract_parameter_count_b(row),
            max_context_tokens=_extract_context_tokens(row),
            huggingface_url=_clean_string(row.get("huggingface_url")) or None,
        )
        key = slug.casefold() or _model_key(name)
        existing = deduplicated.get(key)
        if existing is None or candidate.intelligence_score > existing.intelligence_score:
            deduplicated[key] = candidate

    ranked = sorted(
        deduplicated.values(),
        key=lambda candidate: (-candidate.intelligence_score, candidate.name.casefold()),
    )
    return tuple(replace(candidate, rank=index + 1) for index, candidate in enumerate(ranked))


def _load_aa_rankings(
    *,
    api_key: str,
    cache_path: Path,
    now: datetime | None = None,
) -> _AARankings | None:
    with _AA_CACHE_LOCK:
        return _load_aa_rankings_locked(
            api_key=api_key,
            cache_path=cache_path,
            now=now,
        )


def _load_aa_rankings_locked(
    *,
    api_key: str,
    cache_path: Path,
    now: datetime | None = None,
) -> _AARankings | None:
    current_time = now or datetime.now(timezone.utc)
    key_fingerprint = _api_key_fingerprint(api_key) if api_key else None
    cached = _read_aa_cache(cache_path)
    if cached is not None:
        cache_is_fresh = current_time - cached.fetched_at <= AA_CACHE_TTL
        cache_matches_key = bool(
            key_fingerprint
            and cached.key_fingerprint == key_fingerprint
        )
        if cache_is_fresh and (not api_key or cache_matches_key):
            candidates = normalize_aa_model_candidates(cached.payload)
            if candidates:
                if key_fingerprint:
                    _AA_AUTH_STATUS_BY_KEY[key_fingerprint] = (
                        ArtificialAnalysisAuthStatus(
                            authenticated=True,
                            tier=_clean_string(cached.payload.get("tier")) or None,
                        )
                    )
                return _AARankings(
                    candidates=candidates,
                    freshness="cached",
                    tier=_clean_string(cached.payload.get("tier")) or None,
                )

    if api_key:
        try:
            payload = fetch_artificial_analysis_models(api_key)
        except Exception as exc:
            if key_fingerprint:
                _AA_AUTH_STATUS_BY_KEY[key_fingerprint] = (
                    ArtificialAnalysisAuthStatus(
                        authenticated=False,
                        error=_artificial_analysis_auth_error(exc),
                    )
                )
            payload = None
        if payload is not None:
            tier = _clean_string(payload.get("tier")) or None
            if key_fingerprint:
                _AA_AUTH_STATUS_BY_KEY[key_fingerprint] = (
                    ArtificialAnalysisAuthStatus(
                        authenticated=True,
                        tier=tier,
                    )
                )
            candidates = normalize_aa_model_candidates(payload)
            if candidates:
                _write_aa_cache(
                    cache_path,
                    payload,
                    fetched_at=current_time,
                    api_key=api_key,
                )
                return _AARankings(
                    candidates=candidates,
                    freshness="live",
                    tier=tier,
                )

    if cached is not None:
        candidates = normalize_aa_model_candidates(cached.payload)
        if candidates:
            return _AARankings(
                candidates=candidates,
                freshness="cached",
                tier=_clean_string(cached.payload.get("tier")) or None,
            )
    return None


def get_artificial_analysis_auth_status(
    *,
    api_key: str | None = None,
    cache_path: Path | None = None,
) -> ArtificialAnalysisAuthStatus:
    """Validate the configured AAI key through the shared catalog cache path."""

    normalized_key = (
        resolve_artificial_analysis_api_key() if api_key is None else api_key
    ).strip()
    if not normalized_key:
        return ArtificialAnalysisAuthStatus(authenticated=False)
    key_fingerprint = _api_key_fingerprint(normalized_key)
    with _AA_CACHE_LOCK:
        cached_status = _AA_AUTH_STATUS_BY_KEY.get(key_fingerprint)
    if cached_status is not None:
        return cached_status

    _load_aa_rankings(
        api_key=normalized_key,
        cache_path=cache_path or AA_CACHE_PATH,
    )
    with _AA_CACHE_LOCK:
        return _AA_AUTH_STATUS_BY_KEY.get(
            key_fingerprint,
            ArtificialAnalysisAuthStatus(
                authenticated=False,
                error="Artificial Analysis auth check failed",
            ),
        )


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
) -> tuple[QuickDeployProfile, ...]:
    """Select the top `model_limit` unique open models in each size bucket."""

    window = _aa_candidate_window(candidates, candidate_limit)
    # Deduplicate variants of the same model *before* fanning out to threads.
    # _resolve_aa_model caches by model key, but the check-then-act on the
    # shared dict races: two variants with the same key can both miss the
    # cache and issue duplicate HF lookups (flaky call counts under xdist).
    # Resolving one representative per key keeps behavior identical (the
    # loser would be dropped by seen_repos anyway) and makes HF call counts
    # deterministic.
    seen_keys: set[str] = set()
    deduped_window: list[AAModelCandidate] = []
    for candidate in window:
        model_key = _model_key(candidate.name) or _model_key(candidate.slug)
        if model_key:
            if model_key in seen_keys:
                continue
            seen_keys.add(model_key)
        deduped_window.append(candidate)
    selected_by_bucket: dict[ModelSizeBucket, list[_ResolvedAAModel]] = {
        bucket: [] for bucket in _MODEL_SIZE_BUCKETS
    }
    seen_repos: set[str] = set()
    repo_by_model_key: dict[str, str] = {}
    resolved_by_candidate: dict[int, _ResolvedAAModel | None] = {}
    try:
        from huggingface_hub import HfApi

        shared_hf_api: Any | None = HfApi()
    except Exception:
        shared_hf_api = None
    executor = ThreadPoolExecutor(max_workers=_AA_RESOLUTION_WORKERS)
    try:
        futures = {
            executor.submit(
                _resolve_aa_model,
                candidate,
                modal_gpu_catalog,
                repo_by_model_key=repo_by_model_key,
                hf_api=shared_hf_api,
            ): candidate
            for candidate in deduped_window
        }
        for completed in as_completed(futures):
            resolved_by_candidate[id(futures[completed])] = completed.result()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    for candidate in deduped_window:
        resolved = resolved_by_candidate.get(id(candidate))
        if resolved is None or not resolved.repo_id or resolved.repo_id in seen_repos:
            continue
        seen_repos.add(resolved.repo_id)
        bucket_models = selected_by_bucket[resolved.size_bucket]
        if len(bucket_models) >= model_limit:
            continue
        bucket_models.append(resolved)
        if all(
            len(selected_by_bucket[bucket]) >= model_limit
            for bucket in _MODEL_SIZE_BUCKETS
        ):
            break

    profiles: list[QuickDeployProfile] = []
    for bucket in _MODEL_SIZE_BUCKETS:
        for resolved in selected_by_bucket[bucket]:
            profiles.extend(resolved.profiles)
    return _with_unique_ids(profiles)


def _aa_candidate_window(
    candidates: Sequence[AAModelCandidate],
    candidate_limit: int,
) -> tuple[AAModelCandidate, ...]:
    """Keep a ranked candidate budget for every known size bucket."""

    candidates_by_bucket: dict[ModelSizeBucket, list[AAModelCandidate]] = {
        bucket: [] for bucket in _MODEL_SIZE_BUCKETS
    }
    unknown_size_candidates: list[AAModelCandidate] = []
    selected_ids: set[int] = set()
    for candidate in candidates:
        bucket = _size_bucket_for_parameters(candidate.parameter_count_b)
        if bucket is None:
            if len(unknown_size_candidates) < candidate_limit:
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
) -> _ResolvedAAModel | None:
    model_key = _model_key(candidate.name) or _model_key(candidate.slug)
    if repo_by_model_key is not None and model_key in repo_by_model_key:
        return _build_resolved_aa_model(
            candidate,
            modal_gpu_catalog,
            repo_by_model_key[model_key],
        )
    try:
        api = hf_api
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi()
        repo_id = _find_unsloth_gguf_match(candidate, api)
    except Exception:
        return None
    if repo_id is None:
        return None
    resolved = _build_resolved_aa_model(candidate, modal_gpu_catalog, repo_id)
    if resolved is not None and repo_by_model_key is not None and model_key:
        repo_by_model_key[model_key] = repo_id
    return resolved


def _build_resolved_aa_model(
    candidate: AAModelCandidate,
    modal_gpu_catalog: Sequence[ModalGpuSpec],
    repo_id: str,
) -> _ResolvedAAModel | None:
    try:
        metadata = _fetch_serving_metadata(repo_id)
    except Exception:
        return None
    size_bucket = _size_bucket_for_parameters(candidate.parameter_count_b)
    if size_bucket is None:
        size_bucket = _size_bucket_from_gguf_metadata(metadata)
    if size_bucket is None:
        return None
    if not evaluate_llamacpp_architecture(metadata.architecture).is_supported:
        return None
    model = ModelCandidate(repo_id=repo_id)
    profiles = _profiles_for_model(
        model,
        modal_gpu_catalog,
        metadata=metadata,
        aa_candidate=candidate,
        size_bucket=size_bucket,
    )
    if not profiles:
        return None
    # MTP heads only affect the draft-model toggle on the confirm screen, so
    # resolve them lazily after the catalog is already visible (see
    # _attach_mtp_recommendations). Fetching 1 MiB range requests here would
    # dominate cold-start latency.
    return _ResolvedAAModel(
        candidate=candidate,
        repo_id=repo_id,
        size_bucket=size_bucket,
        profiles=tuple(profiles),
    )


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


class _AAHttpError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"Artificial Analysis API returned HTTP {status_code}")
        self.status_code = status_code


def _fetch_aa_pages(url: str, api_key: str, *, timeout: float) -> dict[str, Any]:
    import requests

    rows: list[Any] = []
    page = 1
    first_payload: dict[str, Any] | None = None
    while page <= 20:
        response = requests.get(
            url,
            headers={"x-api-key": api_key, "Accept": "application/json"},
            params={"page": page},
            timeout=timeout,
        )
        if response.status_code >= 400:
            raise _AAHttpError(response.status_code)
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Artificial Analysis API returned a non-object payload")
        if first_payload is None:
            first_payload = payload
        page_rows = payload.get("data")
        if not isinstance(page_rows, list):
            raise RuntimeError("Artificial Analysis API returned invalid model data")
        rows.extend(page_rows)
        pagination = payload.get("pagination")
        has_more = (
            bool(pagination.get("has_more"))
            if isinstance(pagination, dict)
            else False
        )
        if not has_more:
            break
        page += 1
    if first_payload is None:
        raise RuntimeError("Artificial Analysis API returned no response")
    combined = dict(first_payload)
    combined["data"] = rows
    combined["pagination"] = {
        "page": 1,
        "page_size": len(rows),
        "total_pages": page,
        "has_more": False,
    }
    return combined


def _read_aa_cache(path: Path) -> _AACacheEntry | None:
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        fetched_at_text = _clean_string(envelope.get("fetched_at"))
        payload = envelope.get("payload")
        if not fetched_at_text or not isinstance(payload, dict):
            return None
        fetched_at = datetime.fromisoformat(fetched_at_text.replace("Z", "+00:00"))
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        return _AACacheEntry(
            fetched_at=fetched_at.astimezone(timezone.utc),
            payload=payload,
            key_fingerprint=_clean_string(envelope.get("key_fingerprint")) or None,
        )
    except Exception:
        return None


def _write_aa_cache(
    path: Path,
    payload: dict[str, Any],
    *,
    fetched_at: datetime,
    api_key: str | None = None,
) -> None:
    envelope = {
        "schema_version": 1,
        "fetched_at": fetched_at.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "payload": payload,
    }
    if api_key:
        envelope["key_fingerprint"] = _api_key_fingerprint(api_key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(envelope, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    except Exception:
        return


def _api_key_fingerprint(api_key: str) -> str:
    return hashlib.sha256(api_key.strip().encode("utf-8")).hexdigest()


def _artificial_analysis_auth_error(exc: Exception) -> str:
    if isinstance(exc, _AAHttpError):
        if exc.status_code == 401:
            return "Invalid Artificial Analysis API key"
        if exc.status_code == 429:
            return "Artificial Analysis API rate limit exceeded"
    detail = " ".join(str(exc).split()).strip()
    return detail or "Artificial Analysis auth check failed"


def _reset_artificial_analysis_auth_status_cache() -> None:
    """Reset in-memory auth state for tests."""

    with _AA_CACHE_LOCK:
        _AA_AUTH_STATUS_BY_KEY.clear()


def _is_explicitly_closed_weight(row: dict[str, Any]) -> bool:
    licensing = row.get("licensing")
    if isinstance(licensing, dict) and licensing.get("is_open_weights") is False:
        return True
    for key, value in _iter_key_values(row):
        normalized_key = key.strip().casefold()
        if normalized_key in {
            "open_weights",
            "open_weight",
            "is_open_weight",
            "is_open_weights",
            "weights_available",
        } and value is False:
            return True
    return False


def _extract_parameter_count_b(row: dict[str, Any]) -> float | None:
    parameters = row.get("parameters")
    if isinstance(parameters, dict):
        parsed = _parameter_value_in_billions(parameters.get("total"))
        if parsed is not None:
            return parsed
    for key in ("parameter_count", "parameter_count_b", "parameters_total"):
        parsed = _parameter_value_in_billions(row.get(key))
        if parsed is not None:
            return parsed
    for value in (
        _clean_string(row.get("name")),
        _clean_string(row.get("slug")),
    ):
        parsed = _parameter_count_from_name(value)
        if parsed is not None:
            return parsed
    return None


def _parameter_value_in_billions(value: Any) -> float | None:
    parsed = _optional_float(value)
    if parsed is not None:
        if parsed <= 0:
            return None
        return parsed / 1_000_000_000.0 if parsed > 1_000_000 else parsed
    return _parameter_count_from_name(_clean_string(value))


def _parameter_count_from_name(value: str) -> float | None:
    if not value:
        return None
    mixture = re.search(
        r"(?i)(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*([bt])(?:illion|rillion)?\b",
        value,
    )
    if mixture:
        multiplier = 1_000.0 if mixture.group(3).casefold() == "t" else 1.0
        return float(mixture.group(1)) * float(mixture.group(2)) * multiplier
    compact_moe = re.search(
        r"(?i)(?<![a-z0-9])(\d+(?:\.\d+)?)\s*a\s*\d+(?:\.\d+)?\s*b\b",
        value,
    )
    if compact_moe:
        return float(compact_moe.group(1))
    match = re.search(
        r"(?i)(?<![a-z0-9])(\d+(?:\.\d+)?)\s*([bt])(?:illion|rillion)?\b",
        value,
    )
    if match is None:
        return None
    multiplier = 1_000.0 if match.group(2).casefold() == "t" else 1.0
    return float(match.group(1)) * multiplier


def _extract_context_tokens(row: dict[str, Any]) -> int | None:
    for key in (
        "context_window_tokens",
        "max_context_tokens",
        "context_window",
        "max_context_length",
    ):
        parsed = _parse_token_count(row.get(key))
        if parsed is not None:
            return parsed
    return None


def _parse_token_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 0
    if parsed > 0:
        return parsed
    text = _clean_string(value).casefold().replace(",", "")
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([km]?)", text)
    if not match:
        return None
    multiplier = {"": 1, "k": 1_000, "m": 1_000_000}[match.group(2)]
    tokens = int(round(float(match.group(1)) * multiplier))
    return tokens if tokens > 0 else None


def _size_bucket_for_parameters(parameter_count_b: float | None) -> ModelSizeBucket | None:
    if parameter_count_b is None or parameter_count_b <= 0:
        return None
    if parameter_count_b <= 40:
        return "compact"
    if parameter_count_b <= 150:
        return "medium"
    return "large"


def _size_bucket_from_gguf_metadata(
    metadata: GgufQuantMetadata,
) -> ModelSizeBucket | None:
    quant_thresholds = (
        (("Q4",), 40.0, 105.0),
        (("Q3",), 32.0, 85.0),
        (("Q2",), 25.0, 70.0),
        (("Q5", "Q6"), 50.0, 135.0),
        (("Q8",), 75.0, 200.0),
    )
    for prefixes, compact_max, medium_max in quant_thresholds:
        for quant, required_vram_gb in metadata.vram_gb_by_quant.items():
            normalized = _quant_key(quant).removeprefix("UD-")
            if required_vram_gb <= 0 or not normalized.startswith(prefixes):
                continue
            if required_vram_gb <= compact_max:
                return "compact"
            if required_vram_gb <= medium_max:
                return "medium"
            return "large"
    return None


def _find_unsloth_gguf_match(
    candidate: AAModelCandidate,
    hf_api: Any,
) -> str | None:
    direct_repo = _repo_id_from_huggingface_url(candidate.huggingface_url)
    if direct_repo and direct_repo.casefold().endswith("-gguf"):
        return direct_repo

    for repo_id in _canonical_unsloth_gguf_repo_ids(candidate):
        try:
            row = hf_api.model_info(repo_id=repo_id)
        except Exception:
            continue
        resolved_repo_id = _repo_id_from_hf_row(row) or repo_id
        if (
            resolved_repo_id.casefold().startswith("unsloth/")
            and resolved_repo_id.casefold().endswith("-gguf")
            and _aa_hf_match_score(candidate, resolved_repo_id) >= 90.0
        ):
            return resolved_repo_id

    rows: list[Any] = []
    search_executor = ThreadPoolExecutor(max_workers=_HF_SEARCH_WORKERS)
    try:
        search_futures = [
            search_executor.submit(
                _list_unsloth_gguf_search,
                hf_api,
                search,
            )
            for search in _ranked_hf_search_terms(candidate)
        ]
        for search_future in as_completed(search_futures):
            try:
                rows.extend(search_future.result())
            except Exception:
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
        return _clean_string(row.get("id") or row.get("modelId"))
    return _clean_string(getattr(row, "id", None) or getattr(row, "modelId", None))


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


def _model_key(value: str) -> str:
    cleaned = re.sub(r"\([^)]*\)", "", value.casefold())
    cleaned = re.sub(r"(?i)gguf", "", cleaned)
    return re.sub(r"[^a-z0-9]+", "", cleaned)


def _iter_key_values(node: Any) -> list[tuple[str, Any]]:
    values: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            values.append((str(key), value))
            values.extend(_iter_key_values(value))
    elif isinstance(node, list):
        for value in node:
            values.extend(_iter_key_values(value))
    return values


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
            server_args = compile_server_args(requirements, tuning)
            memory = profile.memory_estimate
            if memory is not None:
                old_context = max(1, profile.max_context_tokens)
                kv_cache_gb = memory.kv_cache_gb * context_tokens / old_context
                memory = replace(
                    memory,
                    kv_cache_gb=round(kv_cache_gb, 3),
                    total_gb=round(
                        memory.weights_gb
                        + kv_cache_gb
                        + memory.compute_gb
                        + memory.speculative_gb
                        + memory.reserve_gb,
                        3,
                    ),
                )
            upgraded_group.append(
                replace(
                    profile,
                    max_context_tokens=context_tokens,
                    server_args=server_args,
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
) -> tuple[QuickDeployProfile, ...]:
    """Attach MTP recommendations to catalog profiles without blocking.

    Intended as a lazy second pass after the catalog is already active:
    the 1 MiB GGUF range probes dominate cold-start latency but only feed
    the speculative-decoding toggle on the confirm screen. Deploy-time
    preflight revalidates MTP anyway, so a missing recommendation here is
    always safe to recompute later.
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
    return tuple(upgraded)


def _profiles_for_model(
    model: ModelCandidate,
    modal_gpu_catalog: Sequence[ModalGpuSpec],
    *,
    metadata: GgufQuantMetadata | None = None,
    aa_candidate: AAModelCandidate | None = None,
    size_bucket: ModelSizeBucket | None = None,
    skip_context_lookup: bool = False,
) -> list[QuickDeployProfile]:
    if metadata is None:
        try:
            metadata = _fetch_serving_metadata(model.repo_id)
        except Exception:
            return []
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
                speculative_decoding=None,
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
    low_vram = available.get(_quant_key(LOW_VRAM_QUANT))
    if low_vram:
        selected.append(low_vram)
    for preferred in _PREFERRED_QUANT_ORDER:
        quant = available.get(_quant_key(preferred))
        if quant and quant not in selected:
            selected.append(quant)
            break
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
    memory_estimate = estimate_memory(
        metadata,
        weights_gb=weights_gb,
        requirements=requirements,
        tuning=runtime_tuning,
    )
    required_vram_gb = memory_estimate.total_gb
    selections = (
        ("cheap", _select_gpu_shape(quant, required_vram_gb, modal_gpu_catalog)),
        (
            "rtx-pro",
            _select_gpu_shape(
                quant,
                required_vram_gb,
                modal_gpu_catalog,
                gpu_type="RTX-PRO-6000",
            ),
        ),
        (
            "b200",
            _select_gpu_shape(
                quant,
                required_vram_gb,
                modal_gpu_catalog,
                gpu_type="B200",
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
        num_speculative_tokens=3,
        nextn_predict_layers=layers,
    )


def _select_gpu_shape(
    quant: str,
    required_vram_gb: float,
    modal_gpu_catalog: Sequence[ModalGpuSpec],
    *,
    gpu_type: str | None = None,
) -> _GpuSelection | None:
    prices = _price_by_gpu(modal_gpu_catalog)
    available = _available_gpu_types(modal_gpu_catalog)
    candidates: list[tuple[float, int, float, str]] = []
    for candidate_gpu in available:
        if gpu_type is not None and candidate_gpu != gpu_type:
            continue
        memory_gb = _GPU_MEMORY_GB.get(candidate_gpu)
        if memory_gb is None:
            continue
        for gpu_count in range(1, 9):
            reserve_per_gpu = max(2.0, memory_gb * 0.05)
            if required_vram_gb / gpu_count + reserve_per_gpu > memory_gb:
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


def _with_context_length(server_args: Sequence[str], context_tokens: int) -> tuple[str, ...]:
    """Rewrite the ``--ctx-size`` pair so it matches an upgraded context length."""

    args = list(server_args)
    for index, token in enumerate(args[:-1]):
        if token == "--ctx-size":
            args[index + 1] = str(context_tokens)
            return tuple(args)
    return tuple([*args, "--ctx-size", str(context_tokens)])


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


def _gpu_slug(value: str) -> str:
    return value.casefold().replace("_", "-").replace("+", "plus").replace("!", "")


def _join_unique(existing: str, addition: str, *, separator: str = "/") -> str:
    values: list[str] = []
    for value in (existing, addition):
        for part in value.split(separator):
            label = part.strip()
            if label and label not in values:
                values.append(label)
    return separator.join(values)


def _clean_string(value: Any) -> str:
    return str(value or "").strip()


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
