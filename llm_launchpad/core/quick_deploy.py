"""Quick-deploy catalog for the TUI landing page."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import shlex
from typing import Sequence

from ..protocol.enums import BackendType, ComputeProvider
from ..protocol.models import (
    DeploymentConfig,
    InferencePlan,
    InferenceRecipe,
    SpeculativeDecodingConfig,
    WorkloadProfile,
)
from .inference_options import (
    InferenceProviderAdapter,
    ModalCatalogOption,
    ModalInferenceAdapter,
    recommended_vllm_tool_call_parser,
    resolve_inference_plans,
)
from .naming import build_deployment_name, infer_instance_from_app_name, slugify_instance_name


@dataclass(frozen=True)
class QuickDeployProfile:
    """Typed catalog entry for a curated quick-deploy target."""

    id: str
    display_name: str
    repo_id: str
    quant: str
    gpu_type: str
    gpu_count: int
    profile_label: str
    approx_cost_per_hour_usd: float
    max_context_tokens: int
    instance_slug_hint: str
    summary: str
    server_args: tuple[str, ...]
    required_vram_gb: float | None = None
    resource_tier: str | None = None
    resource_tier_label: str | None = None
    source_label: str = "Curated"
    aa_model_id: str | None = None
    aa_model_name: str | None = None
    aa_model_slug: str | None = None
    aa_coding_score: float | None = None
    aa_intelligence_score: float | None = None
    aa_rank: int | None = None
    model_size_label: str | None = None
    backend: BackendType = BackendType.LLAMACPP
    model_name: str | None = None
    gguf_architecture: str | None = None
    llamacpp_runtime_id: str | None = None
    speculative_decoding: SpeculativeDecodingConfig | None = None


@dataclass(frozen=True)
class QuickDeployModel:
    """Model-first view over one or more deployable inference recipes."""

    id: str
    display_name: str
    recipes: tuple[InferenceRecipe, ...]
    profiles: tuple[QuickDeployProfile, ...]
    max_context_tokens: int
    quality_score: float | None = None
    quality_rank: int | None = None


@dataclass(frozen=True)
class QuickDeployCatalogInfo:
    """Metadata describing the active quick-deploy catalog."""

    source_label: str
    generated_at: str | None = None
    attribution: str | None = None
    is_fallback: bool = False
    is_live: bool = False
    ready: bool = True
    error: str | None = None


_PENDING_CATALOG_INFO = QuickDeployCatalogInfo(
    source_label="Loading live catalog",
    ready=False,
)
_EMPTY_PROFILES: tuple[QuickDeployProfile, ...] = ()
_CATALOG_CACHE: tuple[QuickDeployCatalogInfo, tuple[QuickDeployProfile, ...]] | None = None


def list_quick_deploy_profiles() -> tuple[QuickDeployProfile, ...]:
    """Return the active immutable quick-deploy catalog."""
    return _load_quick_deploy_catalog()[1]


def get_quick_deploy_catalog_info() -> QuickDeployCatalogInfo:
    """Return metadata for the active quick-deploy catalog."""
    return _load_quick_deploy_catalog()[0]


def get_quick_deploy_profile(profile_id: str) -> QuickDeployProfile:
    """Resolve a quick-deploy profile by identifier."""
    profiles_by_id = {profile.id: profile for profile in list_quick_deploy_profiles()}
    try:
        return profiles_by_id[profile_id]
    except KeyError as exc:
        raise KeyError(f"Unknown quick deploy profile: {profile_id}") from exc


def activate_quick_deploy_catalog(
    info: QuickDeployCatalogInfo,
    profiles: Sequence[QuickDeployProfile],
) -> tuple[QuickDeployCatalogInfo, tuple[QuickDeployProfile, ...]]:
    """Activate a catalog rebuilt by the TUI startup worker."""

    global _CATALOG_CACHE
    catalog = (info, tuple(profiles))
    if not catalog[1]:
        raise ValueError("Cannot activate an empty quick-deploy catalog")
    _CATALOG_CACHE = catalog
    return catalog


def _load_quick_deploy_catalog() -> tuple[QuickDeployCatalogInfo, tuple[QuickDeployProfile, ...]]:
    global _CATALOG_CACHE
    if _CATALOG_CACHE is not None:
        return _CATALOG_CACHE
    _CATALOG_CACHE = (_PENDING_CATALOG_INFO, _EMPTY_PROFILES)
    return _CATALOG_CACHE


def record_quick_deploy_catalog_failure(error: str) -> bool:
    """Record a live catalog failure; keeps any existing live catalog."""

    global _CATALOG_CACHE
    if _CATALOG_CACHE is not None and _CATALOG_CACHE[1]:
        return False
    detail = error.strip() or "The live model catalog could not be reached."
    _CATALOG_CACHE = (
        QuickDeployCatalogInfo(
            source_label="Live catalog unavailable",
            ready=False,
            error=detail,
        ),
        _EMPTY_PROFILES,
    )
    return True


def _reset_quick_deploy_catalog_cache() -> None:
    global _CATALOG_CACHE
    _CATALOG_CACHE = None


def format_hourly_cost(value: float) -> str:
    """Render approximate hourly pricing for UI copy."""
    return f"~${value:.2f}/hr"


def format_context_length(value: int) -> str:
    """Render a token context length for UI copy."""
    return f"{value:,} ctx"


def quick_deploy_model_label_parts(profile: QuickDeployProfile) -> tuple[str, str]:
    """Return the base model label and optional quant suffix."""
    quant = profile.quant.strip()
    if not quant:
        return (profile.display_name, "")
    if quant.casefold() in profile.display_name.casefold():
        return (profile.display_name, "")
    return (profile.display_name, f"({quant})")


def quick_deploy_model_key(profile: QuickDeployProfile) -> str:
    """Return a stable model identity independent of provider fulfillment."""

    return (
        (profile.aa_model_id or "").strip()
        or (profile.aa_model_slug or "").strip().casefold()
        or profile.display_name.strip().casefold()
    )


def quick_deploy_quality_score(profile: QuickDeployProfile) -> float | None:
    """Return AAI Intelligence Index, with coding score for legacy catalogs."""

    if profile.aa_intelligence_score is not None:
        return profile.aa_intelligence_score
    return profile.aa_coding_score


def quick_deploy_recipe(profile: QuickDeployProfile) -> InferenceRecipe:
    """Convert a legacy quick-deploy bundle into a provider-neutral recipe."""

    model_key = quick_deploy_model_key(profile)
    model_id = (profile.model_name or profile.repo_id).strip()
    fingerprint = "\0".join(
        (
            model_key,
            profile.backend.value,
            model_id,
            profile.quant,
            str(profile.max_context_tokens),
            profile.speculative_decoding.method.value if profile.speculative_decoding else "",
            str(profile.speculative_decoding.num_speculative_tokens) if profile.speculative_decoding else "",
            *profile.server_args,
        )
    )
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:10]
    recipe_id = f"{slugify_instance_name(profile.display_name)}-{profile.backend.value}-{digest}"
    return InferenceRecipe(
        id=recipe_id,
        model_key=model_key,
        display_name=profile.display_name,
        backend=profile.backend,
        model_id=model_id,
        quant=profile.quant or None,
        max_context_tokens=profile.max_context_tokens,
        required_vram_gb=profile.required_vram_gb,
        server_args=profile.server_args,
        speculative_decoding=profile.speculative_decoding,
        source_label=profile.source_label,
        quality_score=quick_deploy_quality_score(profile),
        quality_rank=profile.aa_rank,
    )


def list_quick_deploy_recipes(
    profiles: Sequence[QuickDeployProfile] | None = None,
) -> tuple[InferenceRecipe, ...]:
    """Return unique provider-neutral recipes represented by the catalog."""

    rows = tuple(profiles) if profiles is not None else list_quick_deploy_profiles()
    recipes: list[InferenceRecipe] = []
    seen: set[str] = set()
    for profile in rows:
        recipe = quick_deploy_recipe(profile)
        if recipe.id in seen:
            continue
        seen.add(recipe.id)
        recipes.append(recipe)
    return tuple(recipes)


def list_quick_deploy_models(
    profiles: Sequence[QuickDeployProfile] | None = None,
) -> tuple[QuickDeployModel, ...]:
    """Group deployable recipes into model-first catalog entries."""

    rows = tuple(profiles) if profiles is not None else list_quick_deploy_profiles()
    grouped: dict[str, list[QuickDeployProfile]] = {}
    order: list[str] = []
    for profile in rows:
        key = quick_deploy_model_key(profile)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(profile)

    models: list[QuickDeployModel] = []
    for key in order:
        model_profiles = tuple(grouped[key])
        recipes = list_quick_deploy_recipes(model_profiles)
        scores = [
            score
            for row in model_profiles
            if (score := quick_deploy_quality_score(row)) is not None
        ]
        ranks = [row.aa_rank for row in model_profiles if row.aa_rank is not None]
        models.append(
            QuickDeployModel(
                id=key,
                display_name=model_profiles[0].display_name,
                recipes=recipes,
                profiles=model_profiles,
                max_context_tokens=max(row.max_context_tokens for row in model_profiles),
                quality_score=max(scores) if scores else None,
                quality_rank=min(ranks) if ranks else None,
            )
        )
    models.sort(
        key=lambda model: (
            model.quality_rank is None,
            model.quality_rank if model.quality_rank is not None else 10**9,
            -(model.quality_score or 0.0),
            model.display_name.casefold(),
        )
    )
    return tuple(models)


def resolve_quick_deploy_plans(
    profiles: Sequence[QuickDeployProfile] | None = None,
    *,
    adapters: Sequence[InferenceProviderAdapter] | None = None,
    workload: WorkloadProfile | None = None,
) -> tuple[InferencePlan, ...]:
    """Resolve provider-agnostic plans while preserving existing bundle IDs."""

    rows = tuple(profiles) if profiles is not None else list_quick_deploy_profiles()
    recipes = list_quick_deploy_recipes(rows)
    if adapters is None:
        options = [
            ModalCatalogOption(
                id=profile.id,
                recipe_id=quick_deploy_recipe(profile).id,
                gpu_type=profile.gpu_type,
                gpu_count=profile.gpu_count,
                price_per_hour_usd=profile.approx_cost_per_hour_usd,
            )
            for profile in rows
        ]
        adapters = (ModalInferenceAdapter(options),)
    return tuple(resolve_inference_plans(recipes, adapters, workload))


def get_quick_deploy_plan(
    plan_id: str,
    profiles: Sequence[QuickDeployProfile] | None = None,
) -> InferencePlan:
    """Resolve one provider plan by its stable option identifier."""

    plans = {plan.quote.id: plan for plan in resolve_quick_deploy_plans(profiles)}
    try:
        return plans[plan_id]
    except KeyError as exc:
        raise KeyError(f"Unknown quick deploy plan: {plan_id}") from exc


def quick_deploy_profile_for_plan(
    plan: InferencePlan,
    profiles: Sequence[QuickDeployProfile] | None = None,
) -> QuickDeployProfile:
    """Return the catalog profile carrying deployment details for a plan."""

    rows = tuple(profiles) if profiles is not None else list_quick_deploy_profiles()
    configuration_id = (plan.quote.configuration_id or "").strip()
    if configuration_id:
        match = next((profile for profile in rows if profile.id == configuration_id), None)
        if match is not None:
            return match
    match = next(
        (profile for profile in rows if quick_deploy_recipe(profile).id == plan.recipe.id),
        None,
    )
    if match is None:
        raise KeyError(f"No quick deploy profile supplies recipe {plan.recipe.id!r}.")
    return match


def build_quick_deploy_config(
    profile: QuickDeployProfile,
    *,
    plan: InferencePlan | None = None,
    instance_name: str = "",
    app_name: str = "",
    do_warmup: bool = True,
    show_debug_logs: bool = False,
    enable_speculative_decoding: bool = True,
) -> DeploymentConfig:
    """Build a deployment config from a recipe bound to a provider quote."""

    recipe = quick_deploy_recipe(profile)
    if plan is not None and (
        plan.recipe.id != recipe.id or plan.quote.recipe_id != recipe.id
    ):
        raise ValueError(
            f"Quick deploy plan {plan.quote.id!r} does not match profile {profile.id!r}."
        )
    provider = plan.quote.provider if plan is not None else ComputeProvider.MODAL
    config = DeploymentConfig(backend=profile.backend, provider=provider)
    if profile.backend == BackendType.LLAMACPP:
        config.repo_id = profile.repo_id
        config.quant = profile.quant
        config.gguf_architecture = profile.gguf_architecture
        config.llamacpp_runtime_id = profile.llamacpp_runtime_id
        config.server_args = shlex.join(profile.server_args)
        if enable_speculative_decoding:
            config.speculative_decoding = profile.speculative_decoding
    else:
        config.model_name = profile.model_name or profile.repo_id
        config.n_gpu = plan.quote.gpu_count if plan is not None else profile.gpu_count
        config.tool_call_parser = recommended_vllm_tool_call_parser(config.model_name)
    config.gpu_type = plan.quote.gpu_type if plan is not None else profile.gpu_type
    config.gpu_count = plan.quote.gpu_count if plan is not None else profile.gpu_count
    config.required_vram_gb = (
        plan.recipe.required_vram_gb
        if plan is not None
        else profile.required_vram_gb
    )
    config.max_context_tokens = (
        plan.recipe.max_context_tokens
        if plan is not None
        else profile.max_context_tokens
    )
    config.provider_options = plan.quote.provider_options if plan is not None else None
    config.preload = True
    config.do_deploy = True
    config.do_warmup = do_warmup
    config.show_debug_logs = show_debug_logs

    instance_override = instance_name.strip()
    app_override = app_name.strip()
    if app_override:
        config.app_name = app_override
        inferred_instance = infer_instance_from_app_name(app_override, config.backend)
        config.instance_name = slugify_instance_name(instance_override or inferred_instance or app_override)
    elif instance_override:
        config.instance_name = slugify_instance_name(instance_override)
        config.app_name = build_deployment_name(provider, config.backend, config.instance_name)
    else:
        config.instance_name = slugify_instance_name(profile.instance_slug_hint)
        config.app_name = build_deployment_name(provider, config.backend, config.instance_name)
    return config
