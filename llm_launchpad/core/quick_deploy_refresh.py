"""Build a live Deploy catalog for the TUI."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, UTC
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from statistics import median
from tempfile import NamedTemporaryFile
from threading import Lock
import time
from typing import Any, TypeVar
from collections.abc import Callable, Generator, Sequence
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
from .gguf_metadata import GgufMtpStatus, GgufServingMetadata, scoped_hf_token
from .hf_models import (
    GgufHeaderCache,
    GgufQuantMetadata,
    HubModelPageUnavailable,
    ModelCandidate,
    active_gguf_header_cache,
    fetch_gguf_quant_metadata,
    fetch_model_max_context,
    gguf_metadata_for_artifact,
    list_llamacpp_candidates,
)
from .modal_gpu import (
    MODAL_GPU_MEMORY_GB as _GPU_MEMORY_GB,
    ModalGpuSpec,
    fetch_modal_gpu_catalog,
    modal_gpu_capacity_gib,
    modal_gpu_count_options,
    modal_gpu_supports_fast_deploy,
)
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
    bytes_to_gib,
    memory_fits_device,
    memory_on_topology,
    tuning_for_architecture,
    tuning_for_attention_scratch,
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
    GgufArtifact,
    MemoryEstimate,
    RuntimeTuning,
    ServingRequirements,
    SpeculativeDecodingConfig,
)

DEFAULT_MODEL_LIMIT = 3
DEFAULT_OVERALL_MODEL_LIMIT = 10
DEFAULT_CANDIDATE_LIMIT = 240
_AA_RESOLUTION_WORKERS = 24
_FALLBACK_TRENDING_LIMIT = 12
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


_T = TypeVar("_T")
_UNSLOTH_INDEX_TTL_SECONDS = 600.0
_QUANT_HEADER_WORKERS = 6


@dataclass(frozen=True)
class _UnslothGgufIndex:
    """In-memory index of published ``unsloth/*-GGUF`` repositories."""

    repos: tuple[str, ...]
    repos_lower: frozenset[str]
    entries: tuple[tuple[str, str], ...]
    by_model_key: dict[str, tuple[str, ...]]


_UNSLOTH_INDEX_LOCK = Lock()
_UNSLOTH_INDEX_CACHE: tuple[float, _UnslothGgufIndex] | None = None
_ACTIVE_UNSLOTH_GGUF_INDEX: ContextVar[_UnslothGgufIndex | None] = ContextVar(
    "_ACTIVE_UNSLOTH_GGUF_INDEX", default=None
)
_ACTIVE_SHARED_HF_API: ContextVar[Any | None] = ContextVar(
    "_ACTIVE_SHARED_HF_API", default=None
)
_ORIGINAL_FETCH_GGUF_QUANT_METADATA = fetch_gguf_quant_metadata


def _submit_with_context(
    executor: ThreadPoolExecutor,
    fn: Callable[..., _T],
    /,
    *args: Any,
    **kwargs: Any,
) -> Future[_T]:
    """Submit work to a thread pool while propagating active ContextVars."""
    ctx = copy_context()

    def _run() -> _T:
        return ctx.run(fn, *args, **kwargs)

    return executor.submit(_run)


@contextmanager
def _active_unsloth_gguf_index(index: _UnslothGgufIndex | None) -> Generator[None, None, None]:
    token = _ACTIVE_UNSLOTH_GGUF_INDEX.set(index)
    try:
        yield
    finally:
        _ACTIVE_UNSLOTH_GGUF_INDEX.reset(token)


@contextmanager
def _active_shared_hf_api(api: Any | None) -> Generator[None, None, None]:
    token = _ACTIVE_SHARED_HF_API.set(api)
    try:
        yield
    finally:
        _ACTIVE_SHARED_HF_API.reset(token)


def _reset_unsloth_repo_index_cache() -> None:
    """Clear the process-local Unsloth repository index cache."""
    global _UNSLOTH_INDEX_CACHE
    with _UNSLOTH_INDEX_LOCK:
        _UNSLOTH_INDEX_CACHE = None


def _build_unsloth_gguf_index(repos: Sequence[str]) -> _UnslothGgufIndex:
    unique: list[str] = []
    seen: set[str] = set()
    by_key_lists: dict[str, list[str]] = {}
    entries: list[tuple[str, str]] = []
    for raw in repos:
        repo_id = clean_string(raw)
        lowered = repo_id.casefold()
        if (
            not repo_id
            or lowered in seen
            or not lowered.startswith("unsloth/")
            or not lowered.endswith("-gguf")
        ):
            continue
        seen.add(lowered)
        unique.append(repo_id)
        repo_key = _model_key(_repo_model_name(repo_id))
        entries.append((repo_id, repo_key))
        if repo_key:
            by_key_lists.setdefault(repo_key, []).append(repo_id)
    return _UnslothGgufIndex(
        repos=tuple(unique),
        repos_lower=frozenset(seen),
        entries=tuple(entries),
        by_model_key={key: tuple(values) for key, values in by_key_lists.items()},
    )


def _load_unsloth_gguf_index(hf_api: Any | None) -> _UnslothGgufIndex | None:
    """Fetch all ``unsloth/*-GGUF`` repo IDs once and reuse them across matches."""
    global _UNSLOTH_INDEX_CACHE
    now = time.monotonic()
    with _UNSLOTH_INDEX_LOCK:
        if (
            _UNSLOTH_INDEX_CACHE is not None
            and now - _UNSLOTH_INDEX_CACHE[0] <= _UNSLOTH_INDEX_TTL_SECONDS
        ):
            return _UNSLOTH_INDEX_CACHE[1]
    api = hf_api
    if api is None:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
        except Exception:
            return None
    try:
        # Omit the ``gguf`` tag filter so brand-new untagged Unsloth GGUF repos
        # are still indexed, and omit ``expand`` so responses stay minimal.
        rows = api.list_models(author="unsloth")
        repo_ids = [_repo_id_from_hf_row(row) for row in rows]
    except Exception as exc:
        log_debug(f"Could not prefetch Unsloth repository index: {exc}")
        return None
    index = _build_unsloth_gguf_index(repo_ids)
    if not index.repos:
        return None
    with _UNSLOTH_INDEX_LOCK:
        _UNSLOTH_INDEX_CACHE = (time.monotonic(), index)
    return index


def _seed_header_cache_from_profiles(
    cache: GgufHeaderCache,
    profiles: Sequence[QuickDeployProfile],
) -> None:
    for profile in profiles:
        artifact = profile.gguf_artifact
        layout = profile.serving_metadata
        if (
            artifact is None
            or layout is None
            or not layout.architecture
            or not profile.repo_id
        ):
            continue
        if layout.source_file and layout.source_file != artifact.entrypoint:
            continue
        cache.record(
            profile.repo_id,
            artifact.revision,
            artifact.entrypoint,
            layout,
            mark_dirty=False,
        )


def _fetch_serving_metadata(repo_id: str) -> GgufQuantMetadata:
    """Fetch the Hub's GGUF file listing, retrying transient lookup failures."""

    shared_api = _ACTIVE_SHARED_HF_API.get()
    use_shared_api = (
        shared_api is not None
        and fetch_gguf_quant_metadata is _ORIGINAL_FETCH_GGUF_QUANT_METADATA
    )
    delay = _HUB_RETRY_BASE_SECONDS
    for attempt in range(1, _HUB_FETCH_ATTEMPTS + 1):
        try:
            if use_shared_api:
                return fetch_gguf_quant_metadata(
                    repo_id,
                    inspect_serving=True,
                    require_weight_sizes=True,
                    hf_api=shared_api,
                )
            return fetch_gguf_quant_metadata(
                repo_id, inspect_serving=True, require_weight_sizes=True
            )
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
# Older profiles used the model-page size table and lacked exact weight sets.
QUICK_DEPLOY_CATALOG_CACHE_SCHEMA_VERSION = 15
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
    size_bucket: ModelSizeBucket | None
    profiles: tuple[QuickDeployProfile, ...]


_PARTIAL_PUBLISH_INTERVAL_SECONDS = 0.35


def _merge_partial_catalog_profiles(
    fresh_profiles: Sequence[QuickDeployProfile],
    previous_profiles: Sequence[QuickDeployProfile],
) -> tuple[QuickDeployProfile, ...]:
    """Merge newly resolved profiles onto an existing catalog without shrinking it."""
    if not previous_profiles:
        ordered = sorted(fresh_profiles, key=lambda profile: profile.aa_rank or 10**9)
        return _with_unique_ids(ordered)
    fresh_ids = {profile.id for profile in fresh_profiles}
    fresh_repos = {profile.repo_id.casefold() for profile in fresh_profiles if profile.repo_id}
    kept_previous = [
        profile
        for profile in previous_profiles
        if profile.id not in fresh_ids and profile.repo_id.casefold() not in fresh_repos
    ]
    combined = sorted(
        [*fresh_profiles, *kept_previous],
        key=lambda profile: profile.aa_rank or 10**9,
    )
    return _with_unique_ids(combined)


def build_live_quick_deploy_catalog(
    *,
    model_limit: int = DEFAULT_MODEL_LIMIT,
    overall_model_limit: int = DEFAULT_OVERALL_MODEL_LIMIT,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    on_partial: Callable[[QuickDeployCatalogInfo, tuple[QuickDeployProfile, ...]], None] | None = None,
) -> tuple[QuickDeployCatalogInfo, tuple[QuickDeployProfile, ...]]:
    """Rebuild the Deploy catalog from the top AAI open models and live pricing."""

    normalized_model_limit = max(1, int(model_limit))
    normalized_candidate_limit = max(
        normalized_model_limit,
        int(candidate_limit),
    )
    previous_cached = _read_quick_deploy_catalog_cache(_quick_deploy_catalog_cache_path())
    previous_profiles: tuple[QuickDeployProfile, ...] = (
        previous_cached[1] if previous_cached is not None else ()
    )
    header_cache = GgufHeaderCache.load()
    _seed_header_cache_from_profiles(header_cache, previous_profiles)

    try:
        from huggingface_hub import HfApi, get_token

        shared_hf_api: Any | None = HfApi()
        resolved_hf_token: str | None = (get_token() or "").strip()
    except Exception:
        shared_hf_api = None
        resolved_hf_token = None

    should_prefetch_index = (
        _find_unsloth_gguf_match is _DEFAULT_FIND_UNSLOTH_GGUF_MATCH
        and fetch_gguf_quant_metadata is _ORIGINAL_FETCH_GGUF_QUANT_METADATA
        and _fetch_serving_metadata is _DEFAULT_FETCH_SERVING_METADATA
    )

    def _safe_list_trending() -> tuple[list[ModelCandidate] | None, BaseException | None]:
        try:
            return list_llamacpp_candidates(mode="trending", limit=_FALLBACK_TRENDING_LIMIT), None
        except BaseException as exc:
            return None, exc

    try:
        with (
            active_gguf_header_cache(header_cache),
            scoped_hf_token(resolved_hf_token),
            _active_shared_hf_api(shared_hf_api),
        ):
            with ThreadPoolExecutor(max_workers=4) as startup_executor:
                rankings_future = _submit_with_context(
                    startup_executor,
                    _load_aa_rankings,
                    api_key=resolve_artificial_analysis_api_key(),
                    cache_path=AA_CACHE_PATH,
                    verify_key=False,
                )
                gpu_future = _submit_with_context(
                    startup_executor,
                    _fetch_modal_gpu_catalog_or_fallback,
                )
                index_future = (
                    _submit_with_context(
                        startup_executor,
                        _load_unsloth_gguf_index,
                        shared_hf_api,
                    )
                    if should_prefetch_index
                    else None
                )
                trending_future = _submit_with_context(
                    startup_executor,
                    _safe_list_trending,
                )
                rankings = rankings_future.result()
                modal_gpu_catalog, has_live_modal_pricing = gpu_future.result()
                unsloth_index = index_future.result() if index_future is not None else None

            with _active_unsloth_gguf_index(unsloth_index):
                exclusions: list[CatalogExclusion] = []
                if rankings is not None:
                    tier_suffix = f", {rankings.tier} tier" if rankings.tier else ""
                    aa_source_label = (
                        "Artificial Analysis top open models "
                        f"({rankings.freshness}{tier_suffix}) + "
                        f"{'live' if has_live_modal_pricing else 'fallback'} Modal pricing"
                    )
                    aa_attribution = (
                        f"{AA_ATTRIBUTION} Model metadata sourced from Hugging Face; "
                        "GPU pricing sourced from Modal."
                    )

                    def publish_aa_partial(
                        partial_profiles: tuple[QuickDeployProfile, ...],
                        partial_exclusions: tuple[CatalogExclusion, ...],
                    ) -> None:
                        if on_partial is None or not partial_profiles:
                            return
                        merged = _merge_partial_catalog_profiles(partial_profiles, previous_profiles)
                        if not merged:
                            return
                        partial_info = QuickDeployCatalogInfo(
                            source_label=aa_source_label,
                            generated_at=_utc_now_iso(),
                            attribution=aa_attribution,
                            is_live=True,
                            ready=True,
                            refreshing=True,
                            exclusions=tuple(sorted(partial_exclusions, key=lambda row: row.rank or 10**9)),
                        )
                        on_partial(partial_info, merged)

                    profiles = _profiles_from_aa_rankings(
                        rankings.candidates,
                        modal_gpu_catalog,
                        model_limit=normalized_model_limit,
                        overall_model_limit=max(0, int(overall_model_limit)),
                        candidate_limit=normalized_candidate_limit,
                        exclusions=exclusions,
                        on_partial=publish_aa_partial if on_partial is not None else None,
                    )
                    if profiles:
                        publish_aa_partial(profiles, tuple(exclusions))
                        prefetched_trending, trending_err = trending_future.result()
                        if trending_err is not None:
                            log_debug(f"Trending discovery unavailable: {trending_err}")
                            trending: tuple[QuickDeployProfile, ...] = ()
                        else:
                            try:
                                _, trending = _build_trending_fallback_catalog(
                                    model_limit=_FALLBACK_TRENDING_LIMIT,
                                    modal_gpu_catalog=modal_gpu_catalog,
                                    has_live_modal_pricing=has_live_modal_pricing,
                                    exclusions=exclusions,
                                    skip_repos={p.repo_id.casefold() for p in profiles}
                                    | {row.repo_id.casefold() for row in exclusions},
                                    prefetched_models=prefetched_trending,
                                )
                            except Exception as exc:
                                log_debug(f"Trending discovery unavailable: {exc}")
                                trending = ()
                        profiles = _with_unique_ids([*profiles, *trending])
                        info = QuickDeployCatalogInfo(
                            source_label=aa_source_label,
                            generated_at=_utc_now_iso(),
                            attribution=aa_attribution,
                            is_live=True,
                            exclusions=tuple(sorted(exclusions, key=lambda row: row.rank or 10**9)),
                        )
                        retained = _retained_catalog_for(profiles, exclusions)
                        if retained is not None:
                            _write_quick_deploy_catalog_cache(*retained)
                            return retained
                        _write_quick_deploy_catalog_cache(info, profiles)
                        return (info, profiles)

                def publish_trending_partial(
                    partial_info: QuickDeployCatalogInfo,
                    partial_profiles: tuple[QuickDeployProfile, ...],
                ) -> None:
                    if on_partial is None or not partial_profiles:
                        return
                    merged = _merge_partial_catalog_profiles(partial_profiles, previous_profiles)
                    if not merged:
                        return
                    on_partial(replace(partial_info, refreshing=True), merged)

                prefetched_trending, trending_err = trending_future.result()
                try:
                    if trending_err is not None:
                        raise trending_err
                    fallback = _build_trending_fallback_catalog(
                        model_limit=max(normalized_model_limit, overall_model_limit),
                        modal_gpu_catalog=modal_gpu_catalog,
                        has_live_modal_pricing=has_live_modal_pricing,
                        exclusions=exclusions,
                        on_partial=publish_trending_partial if on_partial is not None else None,
                        prefetched_models=prefetched_trending,
                    )
                except RuntimeError:
                    if not exclusions:
                        raise
                    retained = _retained_catalog_for((), exclusions)
                    fallback = retained or (QuickDeployCatalogInfo(
                        source_label="Model availability", generated_at=_utc_now_iso(),
                        exclusions=tuple(exclusions), is_live=True,
                    ), ())
                else:
                    fallback = _retained_catalog_for(fallback[1], exclusions) or (
                        replace(fallback[0], exclusions=tuple(exclusions)), fallback[1],
                    )
                _write_quick_deploy_catalog_cache(fallback[0], fallback[1])
                return fallback
    finally:
        header_cache.save()


def _retained_catalog_for(
    profiles: Sequence[QuickDeployProfile],
    exclusions: Sequence[CatalogExclusion] = (),
    *,
    cache_path: Path | None = None,
) -> tuple[QuickDeployCatalogInfo, tuple[QuickDeployProfile, ...]] | None:
    """Return the cached catalog when a rebuild is worse than the one it replaces.

    Every model whose Hub metadata cannot be read is dropped from the rebuild,
    so a run of transient failures produces a small but structurally valid
    catalog. Persisting that would replace a good snapshot with a worse one and
    hide models the user deployed yesterday, so the previous snapshot wins and
    the next refresh recovers on its own.

    Profile count alone does not detect that. Discovery walks down the ranking
    until each size category is full, so throttling that removes the top of the
    list is answered by filling the categories from far below it: one observed
    rebuild lost every model ranked above 390 to HTTP 429 and still published
    *twice* as many profiles as the catalog it overwrote. What matters is not
    how many models survived but whether the ones that did not were rejected or
    merely unreachable, so a rebuild that leaves a model unchecked which the
    previous catalog had already resolved is treated as the degraded run it is.
    """

    previous = _read_quick_deploy_catalog_cache(
        cache_path or _quick_deploy_catalog_cache_path()
    )
    if previous is None:
        return None
    previous_info, previous_profiles = previous
    if not previous_profiles:
        return None
    lost = _models_lost_to_unchecked(previous_profiles, exclusions)
    if lost:
        detail = ", ".join(lost[:3]) + (f" and {len(lost) - 3} more" if len(lost) > 3 else "")
        reason = (
            f"Kept the previous catalog: this refresh could not check {len(lost)} "
            f"model(s) it already had ({detail}), which usually means Hugging Face "
            "was rate limiting or unreachable."
        )
    elif len(profiles) < len(previous_profiles) * _CATALOG_RETENTION_RATIO:
        reason = (
            f"Kept the previous catalog: this refresh resolved only "
            f"{len(profiles)} of {len(previous_profiles)} models, which "
            "usually means Hugging Face was rate limiting or unreachable."
        )
    else:
        return None
    by_id = {row.model_id: row for row in previous_info.exclusions}
    by_id.update((row.model_id, row) for row in exclusions)
    blocked = {row.model_id for row in exclusions if not row.unchecked}
    kept = tuple(p for p in previous_profiles if (p.aa_model_id or p.repo_id) not in blocked)
    checked = {p.aa_model_id or p.repo_id for p in profiles}
    incomplete = {row.model_id for row in exclusions if row.unchecked}
    by_id = {key: row for key, row in by_id.items() if key not in checked or key in incomplete}
    new_ids = {p.id for p in profiles}
    new_quants = {(p.repo_id.casefold(), _quant_key(p.quant)) for p in profiles}
    combined = _with_unique_ids([
        *profiles, *(p for p in kept if p.id not in new_ids and
                     (p.repo_id.casefold(), _quant_key(p.quant)) not in new_quants),
    ])
    return replace(previous_info, error=reason, exclusions=tuple(by_id.values())), combined


def _models_lost_to_unchecked(
    previous_profiles: Sequence[QuickDeployProfile],
    exclusions: Sequence[CatalogExclusion],
) -> list[str]:
    """Name the models this rebuild could not check but the last one resolved."""

    known = {profile.aa_model_id or profile.repo_id for profile in previous_profiles}
    if not known:
        return []
    lost: dict[str, str] = {}
    for exclusion in exclusions:
        if exclusion.unchecked and exclusion.model_id in known:
            lost.setdefault(exclusion.model_id, exclusion.display_name or exclusion.model_id)
    return list(lost.values())


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

    A build that could not check every model is never fresh, however recently
    it ran. It is a partial answer, and the whole point of writing down what it
    resolved is that reopening the screen carries on from there instead of
    serving the short list for another six hours.

    This keys off ``CatalogExclusion.unchecked`` rather than the exhausted-budget
    reason alone: a build throttled by Hugging Face records a different sentence
    for the same situation, and matching only the budget's wording let a catalog
    that lost 82 models to HTTP 429 count as fresh for the whole TTL.
    """

    if any(exclusion.unchecked for exclusion in info.exclusions):
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
        "context_known": profile.context_known,
        "parameter_count_b": profile.parameter_count_b,
        "active_parameter_count_b": profile.active_parameter_count_b,
        "serving_metadata": asdict(profile.serving_metadata) if profile.serving_metadata else None,
        "gguf_artifact": asdict(profile.gguf_artifact) if profile.gguf_artifact else None,
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
                "layer_cache_bytes": list(memory.layer_cache_bytes),
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
    layout = payload.get("serving_metadata")
    try:
        if isinstance(layout, dict):
            fields: dict[str, Any] = dict(layout)
            for name in (
                "attention_head_count_kv_by_layer", "recurrent_layers",
                "sliding_window_pattern", "feed_forward_length_by_layer",
            ):
                fields[name] = tuple(fields.get(name) or ())
            layout = GgufServingMetadata(**fields)
        else:
            layout = None
        artifact = payload.get("gguf_artifact")
        return QuickDeployProfile(
            gguf_artifact=GgufArtifact(
                artifact["revision"], tuple((name, size) for name, size in artifact["files"])
            ) if isinstance(artifact, dict) else None,
            serving_metadata=layout,
            context_known=payload.get("context_known") is not False,
            parameter_count_b=float(payload["parameter_count_b"]) if payload.get("parameter_count_b") else None,
            active_parameter_count_b=float(payload["active_parameter_count_b"]) if payload.get("active_parameter_count_b") else None,
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
    except (KeyError, ValueError, TypeError):
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
            layer_cache_bytes=tuple(int(value) for value in payload.get("layer_cache_bytes", ())),
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
            exclusions=tuple(
                CatalogExclusion(
                    model_id=str(row.get("model_id") or ""),
                    display_name=str(row.get("display_name") or ""),
                    repo_id=str(row.get("repo_id") or ""),
                    reason=str(row.get("reason") or ""),
                    rank=row.get("rank"),
                    unchecked=bool(row.get("unchecked", False)),
                )
                for row in payload.get("exclusions", ())
                if isinstance(row, dict)
            ),
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
    temporary_path = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as output:
            temporary_path = Path(output.name)
            json.dump(envelope, output, indent=2)
            output.write("\n")
        temporary_path.replace(path)
    except Exception:
        return
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


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
    if not profiles and not info.exclusions:
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
    on_partial: Callable[[tuple[QuickDeployProfile, ...], tuple[CatalogExclusion, ...]], None] | None = None,
) -> tuple[QuickDeployProfile, ...]:
    """Discover ranked repositories within a request budget, independent of display categories."""
    candidates = _deduped_candidates(candidates)
    match_store = RepoMatchStore.load()
    known = []
    unknown = []
    for candidate in candidates:
        match = match_store.get(_model_key(candidate.name) or _model_key(candidate.slug))
        if match is not None and match.repo_id:
            known.append(candidate)
        elif match is None:
            unknown.append(candidate)
    # Recheck identified open weights before spending requests on speculative matches.
    window = known + list(_aa_candidate_window(unknown, candidate_limit))
    resolved_by_repo: dict[str, _ResolvedAAModel] = {}
    repo_by_model_key: dict[str, str] = {}
    budget = HubRequestBudget(hub_request_budget)
    shared_hf_api = _ACTIVE_SHARED_HF_API.get()
    if shared_hf_api is None:
        try:
            from huggingface_hub import HfApi

            shared_hf_api = HfApi()
        except Exception:
            shared_hf_api = None

    remaining = iter(window)
    pending: dict[Future[_ResolvedAAModel | None], tuple[AAModelCandidate, list[CatalogExclusion]]] = {}
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
            # Workers own their diagnostics. Cancelled/unused lookahead must
            # not mutate an already-published catalog after this call returns.
            reasons: list[CatalogExclusion] = []
            future = _submit_with_context(
                executor,
                _resolve_aa_model,
                candidate,
                modal_gpu_catalog,
                repo_by_model_key=repo_by_model_key,
                hf_api=shared_hf_api,
                exclusions=reasons,
                match_store=match_store,
                budget=budget,
            )
            pending[future] = (candidate, reasons)

    def current_profiles() -> list[QuickDeployProfile]:
        ordered = sorted(
            resolved_by_repo.values(),
            key=lambda item: item.candidate.rank or 10**9,
        )
        collected: list[QuickDeployProfile] = []
        for item in ordered:
            collected.extend(item.profiles)
        return collected

    try:
        fill_pending()
        last_published_at = 0.0
        published_repos = 0
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                _candidate, reasons = pending.pop(future)
                resolved = future.result()
                if exclusions is not None:
                    exclusions.extend(reasons)
                if resolved is not None and resolved.repo_id:
                    existing = resolved_by_repo.get(resolved.repo_id)
                    if existing is None or (resolved.candidate.rank or 10**9) < (
                        existing.candidate.rank or 10**9
                    ):
                        resolved_by_repo[resolved.repo_id] = resolved
                        if on_partial is not None:
                            now = time.monotonic()
                            if (
                                published_repos == 0
                                or now - last_published_at >= _PARTIAL_PUBLISH_INTERVAL_SECONDS
                            ):
                                snapshot = _with_unique_ids(current_profiles())
                                on_partial(snapshot, tuple(exclusions or ()))
                                last_published_at = now
                                published_repos = len(resolved_by_repo)
            fill_pending()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        # Whatever was learned this run is worth keeping even if the run was
        # cut short; that is what makes the next one reach further.
        match_store.save()

    for candidate in remaining:
        match = match_store.get(_model_key(candidate.name) or _model_key(candidate.slug))
        _record_exclusion(exclusions, candidate, (match.repo_id if match else "") or "",
                          UNCHECKED_BUDGET_REASON, unchecked=True)
    profiles = current_profiles()
    profiles.sort(key=lambda profile: profile.aa_rank or 10**9)
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
    """Bound discovery by distinct ranked candidates, independently of size."""
    return tuple(candidates[:max(0, candidate_limit)])


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
            unchecked=True,
        )
        return None
    except HubBudgetExhausted:
        # Not checked, so not an answer. Recording a miss here would poison
        # the store with a conclusion no request was made to support.
        _record_exclusion(exclusions, candidate, "", UNCHECKED_BUDGET_REASON, unchecked=True)
        return None
    except Exception as exc:
        # "The search failed" and "this model has no GGUF weights" are
        # different facts, and only the second is a reason to drop a model
        # without saying so. Hugging Face rate limiting removes whole
        # stretches of the ranking at once, and a silently shorter shortlist
        # gives the reader nothing to notice: a build that lost two thirds of
        # a category to throttling published as though it were complete.
        _record_exclusion(
            exclusions, candidate, "", _hf_failure_reason(exc, "search for weights"), unchecked=True,
        )
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
    # Reserve discovery budget before querying the repository's file listing.
    if budget is not None and not budget.spend(METADATA_REQUEST_COST):
        _record_exclusion(exclusions, candidate, repo_id, UNCHECKED_BUDGET_REASON, unchecked=True)
        return None
    try:
        metadata = _fetch_serving_metadata(repo_id)
    except Exception as exc:
        _record_exclusion(
            exclusions, candidate, repo_id, _hf_failure_reason(exc, "read serving metadata"), unchecked=True,
        )
        return None
    size_bucket = _size_bucket_for_parameters(candidate.parameter_count_b)
    if size_bucket is None:
        size_bucket = _size_bucket_from_gguf_metadata(metadata)
    compatibility = evaluate_llamacpp_architecture(metadata.architecture)
    if metadata.artifacts_by_quant is None and not compatibility.is_supported:
        _record_exclusion(exclusions, candidate, repo_id, compatibility.message)
        return None
    model = ModelCandidate(repo_id=repo_id)
    reasons: list[str] = []
    unchecked: list[str] = []
    profiles = _profiles_for_model(
        model,
        modal_gpu_catalog,
        metadata=metadata,
        aa_candidate=candidate,
        size_bucket=size_bucket,
        rejected=reasons,
        budget=budget,
        unchecked=unchecked,
    )
    if unchecked:
        _record_exclusion(exclusions, candidate, repo_id, " ".join(dict.fromkeys(unchecked)), unchecked=True)
    if not profiles:
        if not unchecked:
            reason = " ".join(dict.fromkeys(reasons)) or "No GGUF quantization has a complete weight set."
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
    if isinstance(exc, HubModelPageUnavailable):
        # The weight sizes have one source, and naming it keeps the reader
        # from reading this as a model that publishes no weights.
        detail = f" (HTTP {status})" if isinstance(status, int) else ""
        return (
            f"Hugging Face did not serve this model's weight-size table{detail}; "
            "the model was not measured. Refresh the catalog to retry."
        )
    return f"Could not {action} on Hugging Face. Refresh the catalog to retry."


def _record_exclusion(
    exclusions: list[CatalogExclusion] | None,
    candidate: AAModelCandidate,
    repo_id: str,
    reason: str,
    *,
    unchecked: bool = False,
) -> None:
    """Record why a discovered model is not in the catalog.

    ``unchecked`` separates "the build could not look" from "the build looked
    and said no". Only the first is a reason to distrust the catalog itself.
    """

    if exclusions is not None:
        exclusions.append(CatalogExclusion(
            model_id=candidate.aa_model_id or candidate.slug,
            display_name=candidate.name, repo_id=repo_id, reason=reason, rank=candidate.rank,
            unchecked=unchecked,
        ))


def _build_trending_fallback_catalog(
    *,
    model_limit: int,
    modal_gpu_catalog: Sequence[ModalGpuSpec],
    has_live_modal_pricing: bool,
    skip_repos: set[str] | None = None,
    exclusions: list[CatalogExclusion] | None = None,
    on_partial: Callable[[QuickDeployCatalogInfo, tuple[QuickDeployProfile, ...]], None] | None = None,
    prefetched_models: Sequence[ModelCandidate] | None = None,
) -> tuple[QuickDeployCatalogInfo, tuple[QuickDeployProfile, ...]]:
    models = (
        list(prefetched_models)
        if prefetched_models is not None
        else list_llamacpp_candidates(mode="trending", limit=_FALLBACK_TRENDING_LIMIT)
    )
    if skip_repos:
        models = [m for m in models if m.repo_id.casefold() not in skip_repos]
    if not models:
        raise RuntimeError(
            "Neither Artificial Analysis rankings nor Hugging Face trending models "
            "were available"
        )

    trending_source_label = (
        "Hugging Face trending GGUF models + "
        f"{'live' if has_live_modal_pricing else 'fallback'} Modal pricing"
    )
    trending_attribution = (
        "Model metadata sourced from Hugging Face; GPU pricing sourced from Modal."
    )

    def build_ordered_profiles(by_repo: dict[str, list[QuickDeployProfile]]) -> list[QuickDeployProfile]:
        ordered: list[QuickDeployProfile] = []
        selected_count = 0
        for candidate_model in models:
            group = by_repo.get(candidate_model.repo_id)
            if not group:
                continue
            ordered.extend(group)
            selected_count += 1
            if selected_count >= model_limit:
                break
        return ordered

    def resolve_trending_model(
        model: ModelCandidate,
    ) -> tuple[list[QuickDeployProfile], CatalogExclusion | None]:
        metadata = _safe_fetch_gguf_metadata(model.repo_id)
        reason = ""
        if metadata is None:
            reason = "Hugging Face serving metadata could not be read. Refresh the catalog to retry."
        else:
            compatibility = evaluate_llamacpp_architecture(metadata.architecture)
            if metadata.artifacts_by_quant is None and not compatibility.is_supported:
                reason = compatibility.message
        if reason:
            return [], CatalogExclusion(
                model.repo_id,
                _display_name(model.repo_id),
                model.repo_id,
                reason,
                unchecked=metadata is None,
            )
        unchecked: list[str] = []
        rejected: list[str] = []
        group = _profiles_for_model(
            model,
            modal_gpu_catalog,
            metadata=metadata,
            skip_context_lookup=True,
            prefer_mtp=not skip_repos,
            unchecked=unchecked,
            rejected=rejected,
        )
        exclusion = None
        if unchecked or not group:
            exclusion = CatalogExclusion(
                model.repo_id,
                _display_name(model.repo_id),
                model.repo_id,
                " ".join(dict.fromkeys(unchecked or rejected))
                or "No GGUF quantization has a complete weight set.",
                unchecked=bool(unchecked),
            )
        return group, exclusion

    profiles_by_repo: dict[str, list[QuickDeployProfile]] = {}
    last_published_at = 0.0
    published_repos = 0
    with ThreadPoolExecutor(max_workers=min(8, len(models))) as executor:
        model_futures = {
            _submit_with_context(executor, resolve_trending_model, model): model
            for model in models
        }
        for model_future in as_completed(model_futures):
            model = model_futures[model_future]
            group, exclusion = model_future.result()
            if exclusion is not None and exclusions is not None:
                exclusions.append(exclusion)
            if group:
                profiles_by_repo[model.repo_id] = group
                if on_partial is not None and len(profiles_by_repo) > published_repos:
                    now = time.monotonic()
                    if published_repos == 0 or now - last_published_at >= _PARTIAL_PUBLISH_INTERVAL_SECONDS:
                        partial_ordered = build_ordered_profiles(profiles_by_repo)
                        if partial_ordered:
                            on_partial(
                                QuickDeployCatalogInfo(
                                    source_label=trending_source_label,
                                    generated_at=_utc_now_iso(),
                                    attribution=trending_attribution,
                                    is_live=True,
                                    is_fallback=True,
                                    exclusions=tuple(exclusions or ()),
                                ),
                                tuple(partial_ordered),
                            )
                            last_published_at = now
                            published_repos = len(profiles_by_repo)

    profiles = build_ordered_profiles(profiles_by_repo)
    if not profiles:
        raise RuntimeError("No deployable profiles could be built from fallback trending models")

    info = QuickDeployCatalogInfo(
        source_label=trending_source_label,
        generated_at=_utc_now_iso(),
        attribution=trending_attribution,
        is_live=True,
        is_fallback=True,
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


def _match_candidate_in_unsloth_index(
    candidate: AAModelCandidate,
    index: _UnslothGgufIndex,
) -> str | None:
    candidate_keys = tuple(
        key
        for key in dict.fromkeys(
            (
                _model_key(candidate.name),
                _model_key(candidate.slug),
                _model_key(
                    _repo_model_name(
                        _repo_id_from_huggingface_url(candidate.huggingface_url) or ""
                    )
                ),
            )
        )
        if key
    )
    if not candidate_keys:
        return None
    scored: list[tuple[float, str]] = []
    seen: set[str] = set()
    for key in candidate_keys:
        for repo_id in index.by_model_key.get(key, ()):
            lowered = repo_id.casefold()
            if lowered in seen:
                continue
            seen.add(lowered)
            score = _aa_hf_match_score(candidate, repo_id)
            if score >= 90.0:
                scored.append((score, repo_id))
    if not scored:
        key_lengths = [(ck, len(ck)) for ck in candidate_keys]
        for repo_id, repo_key in index.entries:
            if not repo_key:
                continue
            repo_len = len(repo_key)
            could_match = False
            for ck, ck_len in key_lengths:
                if ck in repo_key or repo_key in ck:
                    could_match = True
                    break
                # SequenceMatcher.ratio() >= 0.9 requires 2*min/(a+b) >= 0.9 => min/max >= 9/11.
                if min(ck_len, repo_len) * 11 >= max(ck_len, repo_len) * 9:
                    could_match = True
                    break
            if not could_match:
                continue
            score = _aa_hf_match_score(candidate, repo_id)
            if score >= 90.0:
                scored.append((score, repo_id))
    if not scored:
        return None
    best = min(scored, key=lambda item: _aa_hf_match_rank_key(candidate, item[1]))
    return best[1]


def _find_unsloth_gguf_match(
    candidate: AAModelCandidate,
    hf_api: Any,
    budget: HubRequestBudget | None = None,
) -> str | None:
    budget = budget or UnlimitedBudget()
    direct_repo = _repo_id_from_huggingface_url(candidate.huggingface_url)
    if direct_repo and direct_repo.casefold().endswith("-gguf"):
        return direct_repo

    unsloth_index = _ACTIVE_UNSLOTH_GGUF_INDEX.get()
    if unsloth_index is not None:
        return _match_candidate_in_unsloth_index(candidate, unsloth_index)

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
        except Exception as exc:
            if getattr(getattr(exc, "response", None), "status_code", None) != 404:
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
    }
    try:
        return list(hf_api.list_models(**kwargs))
    except TypeError:
        # Older huggingface_hub releases lack author/expand; retry unscoped.
        fallback = {"search": search, "limit": _HF_SEARCH_LIMIT}
        return list(hf_api.list_models(**fallback))


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
    repo_name = _repo_model_name(repo_id)
    # A fuzzy name match must never substitute an older release for the requested one.
    release_pattern = r"(?i)v(\d+(?:\.\d+)*)"
    requested = re.findall(release_pattern, candidate.name)
    offered = re.findall(release_pattern, repo_name)
    if requested and offered and requested != offered:
        return 0.0
    releases = re.findall(r"(?i)(?:v|[- ])(\d+(?:\.\d+)+)(?![\d.])", candidate.name)
    if releases and any(not re.search(rf"(?<!\d){re.escape(version)}(?![.\d])", repo_name) for version in releases):
        return 0.0
    repo_key = _model_key(repo_name)
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
    return (catalog, any(spec.price_per_hour_usd is not None and spec.stale_since is None for spec in catalog))


def _safe_fetch_gguf_metadata(repo_id: str) -> GgufQuantMetadata | None:
    """Resolve the native context before sizing any trending-model profiles."""
    try:
        metadata = _fetch_serving_metadata(repo_id)
    except Exception:
        return None
    if metadata.artifacts_by_quant is None and metadata.context_length is None and metadata.vram_gb_by_quant:
        return replace(metadata, context_length=_safe_fetch_model_max_context(repo_id))
    return metadata


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

    pending = [
        profile for profile in profiles
        if profile.speculative_decoding is None and profile.gguf_artifact is None
    ]
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
        if profile.speculative_decoding is not None or profile.gguf_artifact is not None:
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
        tuning = tuning_for_architecture(tuning, metadata.architecture)
        tuning = tuning_for_attention_scratch(
            tuning,
            context_tokens=requirements.context_tokens,
            attention_head_count=attention_head_count(metadata),
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
    unsloth_index = _ACTIVE_UNSLOTH_GGUF_INDEX.get()
    if unsloth_index is not None and variant_repo.casefold() not in unsloth_index.repos_lower:
        return None
    request_budget = budget if budget is not None else UnlimitedBudget()
    try:
        if not request_budget.spend(METADATA_REQUEST_COST):
            return None
        shared_api = _ACTIVE_SHARED_HF_API.get()
        use_shared_api = (
            shared_api is not None
            and fetch_gguf_quant_metadata is _ORIGINAL_FETCH_GGUF_QUANT_METADATA
        )
        if use_shared_api:
            variant = fetch_gguf_quant_metadata(
                variant_repo,
                inspect_serving=True,
                inspect_mtp=True,
                require_weight_sizes=True,
                hf_api=shared_api,
            )
        else:
            variant = fetch_gguf_quant_metadata(
                variant_repo,
                inspect_serving=True,
                inspect_mtp=True,
                # The variant is only usable if its own quantizations can be
                # sized, so an unreadable weight table is a failed probe to log
                # rather than a variant that offers nothing.
                require_weight_sizes=True,
            )
        if (
            variant.architecture != metadata.architecture
            or _mtp_recommendation(variant) is None
        ):
            return None
        quants = _selected_quants(variant)
        if not quants:
            return None
        if variant.artifacts_by_quant is not None:
            # Profile creation inspects the header of every pinned quant and
            # excludes variants without embedded heads.
            return variant_repo, variant
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
    unchecked: list[str] | None = None,
) -> list[QuickDeployProfile]:
    if metadata is None:
        try:
            metadata = _fetch_serving_metadata(model.repo_id)
        except Exception:
            return []
    variant = _verified_mtp_variant(model.repo_id, metadata, budget) if prefer_mtp else None
    if variant is not None:
        variant_repo, variant_metadata = variant
        unchecked_variant: list[str] = []
        accelerated = _profiles_for_model(
            ModelCandidate(repo_id=variant_repo),
            modal_gpu_catalog,
            metadata=variant_metadata,
            aa_candidate=aa_candidate,
            size_bucket=size_bucket,
            skip_context_lookup=skip_context_lookup,
            budget=budget,
            prefer_mtp=False,
            unchecked=unchecked_variant,
        )
        placed = [
            profile for profile in accelerated
            if profile.gpu_count > 0 and profile.speculative_decoding is not None
        ]
        if placed and not unchecked_variant:
            return placed
    compatibility = evaluate_llamacpp_architecture(metadata.architecture)
    if metadata.artifacts_by_quant is None and not compatibility.is_supported:
        return []
    quants = _selected_quants(metadata)
    if not quants:
        return []
    max_context_tokens = metadata.context_length
    if metadata.artifacts_by_quant is None and max_context_tokens is None and not skip_context_lookup:
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
    unplaced: list[QuickDeployProfile] = []
    layout = metadata.serving_metadata or GgufServingMetadata(
        architecture=metadata.architecture, context_length=max_context_tokens,
        block_count=metadata.block_count, embedding_length=metadata.embedding_length,
        attention_head_count=metadata.attention_head_count, attention_head_count_kv=metadata.attention_head_count_kv,
        attention_key_length=metadata.attention_key_length, attention_value_length=metadata.attention_value_length,
    )
    verified_layout = metadata.serving_metadata
    has_verified_repo_layout = (
        verified_layout is not None
        and bool(verified_layout.architecture)
        and metadata.context_length is not None
    )
    # Within one immutable commit of a GGUF repo, architecture/context/KV layout
    # are shared across quants; per-quant header reads are only needed when no
    # verified header was loaded yet or when MTP heads can vary per quant.
    needs_per_quant_headers = (
        not has_verified_repo_layout
        or model.repo_id.casefold().endswith("-mtp-gguf")
        or _mtp_recommendation(metadata) is not None
        or evaluate_llamacpp_mtp(metadata.architecture, 1).is_supported
    )

    quant_artifacts = [
        (quant, (metadata.artifacts_by_quant or {}).get(_quant_key(quant)))
        for quant in quants
    ]
    fetched_by_quant: dict[str, tuple[GgufQuantMetadata | None, Exception | None]] = {}
    if needs_per_quant_headers:
        to_fetch = [
            (quant, artifact)
            for quant, artifact in quant_artifacts
            if artifact is not None
            and (not layout.source_file or layout.source_file != artifact.entrypoint)
        ]
        if len(to_fetch) > 1:
            with ThreadPoolExecutor(
                max_workers=min(_QUANT_HEADER_WORKERS, len(to_fetch))
            ) as header_executor:
                future_by_quant = {
                    quant: _submit_with_context(
                        header_executor,
                        _fetch_quant_layout,
                        model.repo_id,
                        metadata,
                        artifact,
                    )
                    for quant, artifact in to_fetch
                }
                for quant, future in future_by_quant.items():
                    try:
                        fetched_by_quant[quant] = (future.result(), None)
                    except Exception as exc:
                        fetched_by_quant[quant] = (None, exc)
        elif len(to_fetch) == 1:
            quant, artifact = to_fetch[0]
            try:
                fetched_by_quant[quant] = (
                    _fetch_quant_layout(model.repo_id, metadata, artifact),
                    None,
                )
            except Exception as exc:
                fetched_by_quant[quant] = (None, exc)

    for quant, artifact in quant_artifacts:
        quant_metadata = metadata
        if quant in fetched_by_quant:
            fetched_meta, fetch_exc = fetched_by_quant[quant]
            if fetch_exc is not None or fetched_meta is None:
                if unchecked is not None and fetch_exc is not None:
                    unchecked.append(_hf_failure_reason(fetch_exc, f"read the {quant} GGUF header"))
                continue
            quant_metadata = fetched_meta
        quant_compatibility = (
            compatibility if quant_metadata is metadata
            else evaluate_llamacpp_architecture(quant_metadata.architecture)
        )
        if not quant_compatibility.is_supported:
            if rejected is not None:
                rejected.append(f"{quant}: {quant_compatibility.message}")
            continue
        if artifact is not None and quant_metadata.context_length is None:
            if rejected is not None:
                rejected.append(f"{quant}: the selected GGUF does not declare its context length.")
            continue
        quant_context = quant_metadata.context_length or context_tokens
        built = _profiles_for_quant(
            repo_id=model.repo_id,
            display_name=display_name,
            slug_hint=slug_hint,
            context_tokens=quant_context,
            quant=quant,
            metadata=quant_metadata,
            modal_gpu_catalog=modal_gpu_catalog,
            aa_candidate=aa_candidate,
            size_bucket=size_bucket,
            llamacpp_runtime_id=quant_compatibility.runtime_id,
            speculative_decoding=_mtp_recommendation(quant_metadata),
            rejected=rejected,
            include_unplaced=True,
        )
        for profile in built:
            profile = replace(
                profile, context_known=quant_metadata.context_length is not None or (
                    artifact is None and max_context_tokens is not None
                ), serving_metadata=quant_metadata.serving_metadata or layout, gguf_artifact=artifact,
                parameter_count_b=aa_candidate.parameter_count_b if aa_candidate else None,
                active_parameter_count_b=aa_candidate.active_parameter_count_b if aa_candidate else None,
            )
            (profiles if profile.gpu_count > 0 else unplaced).append(profile)
    # An unplaced recipe is still a recipe: keep at most one per quant so Prime
    # and Vast matching can evaluate it, without duplicating placed profiles.
    seen_unplaced: set[str] = set()
    for profile in unplaced:
        key = _quant_key(profile.quant)
        if key in seen_unplaced:
            continue
        seen_unplaced.add(key)
        profiles.append(profile)
    return profiles


def _fetch_quant_layout(
    repo_id: str, metadata: GgufQuantMetadata, artifact: GgufArtifact,
) -> GgufQuantMetadata:
    delay = _HUB_RETRY_BASE_SECONDS
    for attempt in range(_HUB_FETCH_ATTEMPTS):
        try:
            return gguf_metadata_for_artifact(repo_id, metadata, artifact)
        except Exception as exc:
            if attempt + 1 == _HUB_FETCH_ATTEMPTS or not _is_transient_hub_error(exc):
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"Unreachable GGUF retry loop for {repo_id}")


def _selected_quants(metadata: GgufQuantMetadata) -> list[str]:
    available = {
        _quant_key(quant): quant
        for quant in metadata.quantizations
        if _required_vram_for_quant(metadata, quant) is not None
    }
    if not available:
        return []
    selected: list[str] = []
    # Quality leads. Candidates are chosen in preference order so an
    # intermediate quantization that fits its own hardware can still be
    # offered: previously only the first preference and the low-VRAM fallback
    # were built, and a Q5 or Q6 that fit a GPU exactly was never considered.
    # The preferred width still comes first, so every consumer that reads
    # ``profiles[0]`` gets the better weights.
    for preferred in _PREFERRED_QUANT_ORDER:
        quant = available.get(_quant_key(preferred))
        if quant and quant not in selected:
            selected.append(quant)
    low_vram = available.get(_quant_key(LOW_VRAM_QUANT))
    if low_vram and low_vram not in selected:
        selected.append(low_vram)
    for quant in sorted(available.values()):
        if quant not in selected and len(selected) < 4:
            selected.append(quant)
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
    include_unplaced: bool = False,
) -> list[QuickDeployProfile]:
    artifact = (metadata.artifacts_by_quant or {}).get(_quant_key(quant))
    weights_gb = bytes_to_gib(artifact.size_bytes) if artifact else _required_vram_for_quant(metadata, quant)
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
    if metadata.architecture in {"kimi-k3", "kimi-linear", "glm5next", "glm5-next"} and memory_estimate.source == "conservative-fallback":
        if rejected is not None:
            rejected.append("Hybrid attention memory layout is incomplete; GPU fit cannot be verified. Refresh model metadata to retry.")
        return []
    required_vram_gb = memory_estimate.total_gb
    # An unplaced recipe must reflect genuinely missing capacity, not a
    # fallback GPU list invented for display: without this, an empty catalog
    # could never produce one because the static fallback always fits.
    allow_fallback_gpus = not include_unplaced
    selections = (
        (
            "cheap",
            _select_gpu_shape(
                quant,
                required_vram_gb,
                modal_gpu_catalog,
                memory_estimate=memory_estimate,
                allow_fallback_gpus=allow_fallback_gpus,
            ),
        ),
        (
            "rtx-pro",
            _select_gpu_shape(
                quant,
                required_vram_gb,
                modal_gpu_catalog,
                gpu_type="RTX-PRO-6000",
                memory_estimate=memory_estimate,
                allow_fallback_gpus=allow_fallback_gpus,
            ),
        ),
        (
            "b200",
            _select_gpu_shape(
                quant,
                required_vram_gb,
                modal_gpu_catalog,
                gpu_type="B200",
                memory_estimate=memory_estimate,
                allow_fallback_gpus=allow_fallback_gpus,
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
                gpu_memory_gb=modal_gpu_capacity_gib(selection.gpu_type),
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
    if not profiles:
        if include_unplaced:
            # Keep the recipe discoverable even when the current Modal catalog
            # has no priced topology for it: other providers may still serve
            # it, and dropping it here would hide that. The zeroed shape marks
            # it as unplaced rather than deployable.
            quant_slug = _quant_slug(quant)
            profiles.append(
                _unplaced_profile(
                    repo_id=repo_id,
                    display_name=display_name,
                    slug_hint=slug_hint,
                    context_tokens=context_tokens,
                    quant=quant,
                    quant_slug=quant_slug,
                    metadata=metadata,
                    requirements=requirements,
                    runtime_tuning=runtime_tuning,
                    memory_estimate=memory_estimate,
                    aa_candidate=aa_candidate,
                    size_bucket=size_bucket,
                    llamacpp_runtime_id=llamacpp_runtime_id,
                    speculative_decoding=speculative_decoding,
                )
            )
        elif rejected is not None:
            reason = (
                f"{quant}: estimated {required_vram_gb:,.0f} GiB at {context_tokens:,} tokens "
                "does not fit any priced catalog topology (maximum 8 GPUs)."
            )
            if not runtime_tuning.flash_attention and memory_estimate.attention_scratch_gb >= 1.0:
                # Naming the term matters here: it is the one requirement extra
                # GPUs cannot share, so the shortfall does not look like one more
                # card would close it.
                reason += (
                    f" {memory_estimate.attention_scratch_gb:,.0f} GiB of that is attention"
                    " scratch every GPU needs in full, because this architecture's"
                    " runtime requires flash attention off."
                )
            rejected.append(reason)
    return profiles


def _unplaced_profile(
    *,
    repo_id: str,
    display_name: str,
    slug_hint: str,
    context_tokens: int,
    quant: str,
    quant_slug: str,
    metadata: GgufQuantMetadata,
    requirements: ServingRequirements,
    runtime_tuning: RuntimeTuning,
    memory_estimate: MemoryEstimate,
    aa_candidate: AAModelCandidate | None,
    size_bucket: ModelSizeBucket | None,
    llamacpp_runtime_id: str,
    speculative_decoding: SpeculativeDecodingConfig | None,
) -> QuickDeployProfile:
    """Build a provider-independent recipe with no current Modal placement.

    The catalog used to drop a quantization entirely when the Modal price list
    had no topology for it, so Prime and Vast never saw the recipe even when
    they could serve it. An unplaced profile carries the same serving evidence
    with a zeroed shape; matching treats it as a recipe to evaluate, never as
    a deployable placement.
    """
    return QuickDeployProfile(
        id=_stable_profile_id(repo_id, quant_slug, "unplaced"),
        display_name=display_name,
        repo_id=repo_id,
        quant=quant,
        gpu_type="",
        gpu_count=0,
        profile_label="No priced Modal placement",
        approx_cost_per_hour_usd=0.0,
        max_context_tokens=context_tokens,
        instance_slug_hint=f"{slug_hint}-{quant_slug}-unplaced",
        summary=(
            f"Artificial Analysis-ranked {_MODEL_SIZE_LABELS[size_bucket]} "
            "open-weight model matched to verified Hugging Face GGUF weights."
            if aa_candidate is not None and size_bucket is not None
            else "Live Hugging Face trending GGUF model matched to current "
            "Modal GPU pricing."
        ),
        server_args=compile_server_args(requirements, runtime_tuning),
        required_vram_gb=round(memory_estimate.total_gb, 1),
        gpu_memory_gb=None,
        resource_tier="unplaced",
        resource_tier_label=None,
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
    allow_fallback_gpus: bool = True,
    memory_estimate: MemoryEstimate | None = None,
) -> _GpuSelection | None:
    prices = _price_by_gpu(modal_gpu_catalog)
    available = _available_gpu_types(
        modal_gpu_catalog, allow_fallback=allow_fallback_gpus
    )
    memory = memory_estimate or MemoryEstimate(
        weights_gb=max(0.0, required_vram_gb - per_device_overhead_gb),
        kv_cache_gb=0.0,
        compute_gb=per_device_overhead_gb,
        speculative_gb=0.0,
        reserve_gb=0.0,
        total_gb=required_vram_gb,
        total_layer_count=layer_count,
    )
    candidates: list[tuple[float, int, float, str]] = []
    for candidate_gpu in available:
        if gpu_type is not None and candidate_gpu != gpu_type:
            continue
        memory_gb = modal_gpu_capacity_gib(candidate_gpu)
        if memory_gb is None:
            continue
        for gpu_count in modal_gpu_count_options(candidate_gpu):
            if not memory_fits_device(memory_on_topology(memory, gpu_count, memory_gb), memory_gb):
                continue
            cost = prices.get(
                candidate_gpu,
                _FALLBACK_GPU_PRICE_PER_HOUR["RTX-PRO-6000"],
            ) * gpu_count
            candidates.append((cost, gpu_count, -memory_gb, candidate_gpu))
            break
    if not candidates:
        return None
    min_cost = min(cost for cost, _count, _neg_mem, _gpu in candidates)
    # Within a 5% hourly-cost margin (matching serving_tiers._MEANINGFUL_MARGIN),
    # prefer fewer GPUs per container: Modal multi-GPU scheduling wait times
    # grow sharply past 4 cards, so an 8-GPU shape should not beat a 4-GPU one
    # for a negligible discount.
    comparable = [item for item in candidates if item[0] <= min_cost * 1.05]
    cost, gpu_count, _negative_memory, selected_gpu = min(
        comparable,
        key=lambda item: (item[1], item[0], item[2], item[3]),
    )
    return _GpuSelection(
        quant=quant,
        gpu_type=selected_gpu,
        gpu_count=gpu_count,
        cost_per_hour_usd=cost,
        required_vram_gb=required_vram_gb,
    )


def _available_gpu_types(
    modal_gpu_catalog: Sequence[ModalGpuSpec],
    *,
    allow_fallback: bool = True,
) -> list[str]:
    """Return catalog GPU shapes that can actually back a priced profile.

    Only known prices, supported runtimes, and known memory capacities can form a profile.
    """
    prices = _live_price_by_gpu(modal_gpu_catalog)
    values = [
        entry.value.strip()
        for entry in modal_gpu_catalog
        if entry.value.strip()
        and modal_gpu_supports_fast_deploy(entry.value)
        and _GPU_MEMORY_GB.get(entry.value.strip()) is not None
        and (prices.get(entry.value.strip()) or 0) > 0
    ]
    if values:
        return values
    if not allow_fallback:
        return []
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


_DEFAULT_FIND_UNSLOTH_GGUF_MATCH = _find_unsloth_gguf_match
_DEFAULT_FETCH_SERVING_METADATA = _fetch_serving_metadata
