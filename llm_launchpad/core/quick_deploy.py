"""Quick-deploy catalog for the TUI landing page."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import resources
import json
import shlex
from typing import Any, Sequence

from ..protocol.enums import BackendType, ComputeProvider
from ..protocol.models import (
    DeploymentConfig,
    InferencePlan,
    InferenceRecipe,
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
from .runtime_support import evaluate_llamacpp_architecture


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


_BUNDLED_CATALOG_PACKAGE = "llm_launchpad.data"
_BUNDLED_CATALOG_FILENAME = "quick_deploy_catalog.json"


QWEN35_397B_SERVER_ARGS = (
    "--ctx-size",
    "262144",
    "--threads",
    "16",
    "--temp",
    "0.6",
    "--top-p",
    "0.95",
    "--top-k",
    "20",
    "--min-p",
    "0.00",
)

GLM5_SERVER_ARGS = (
    "--ctx-size",
    "202752",
    "--flash-attn",
    "on",
    "--temp",
    "0.7",
    "--top-p",
    "1.0",
    "--min-p",
    "0.01",
)

KIMI_K25_SERVER_ARGS = (
    "--special",
    "--kv-unified",
    "--ctx-size",
    "98304",
    "--temp",
    "1.0",
    "--top-p",
    "0.95",
    "--min-p",
    "0.01",
)


_STATIC_QUICK_DEPLOY_PROFILES: tuple[QuickDeployProfile, ...] = (
    QuickDeployProfile(
        id="qwen35-397b-rtxpro",
        display_name="Qwen3.5 397B A17B",
        repo_id="unsloth/Qwen3.5-397B-A17B-GGUF",
        quant="UD-Q4_K_XL",
        gpu_type="RTX-PRO-6000",
        gpu_count=3,
        profile_label="Cheap but good",
        approx_cost_per_hour_usd=9.09,
        max_context_tokens=262144,
        instance_slug_hint="qwen35-397b-rtxpro",
        summary="Default curated Qwen3.5 profile for long-context coding workloads on three RTX PRO 6000 GPUs.",
        server_args=QWEN35_397B_SERVER_ARGS,
        source_label="Curated fallback",
    ),
    QuickDeployProfile(
        id="glm5-rtxpro",
        display_name="GLM-5",
        repo_id="unsloth/GLM-5-GGUF",
        quant="UD-Q4_K_XL",
        gpu_type="RTX-PRO-6000",
        gpu_count=4,
        profile_label="Cheap but good",
        approx_cost_per_hour_usd=12.12,
        max_context_tokens=202752,
        instance_slug_hint="glm5-rtxpro",
        summary="Default curated GLM-5 profile for long-context coding and agent workflows on four RTX PRO 6000 GPUs.",
        server_args=GLM5_SERVER_ARGS,
        source_label="Curated fallback",
    ),
    QuickDeployProfile(
        id="kimi25-rtxpro",
        display_name="Kimi K2.5",
        repo_id="unsloth/Kimi-K2.5-GGUF",
        quant="UD-Q4_K_XL",
        gpu_type="RTX-PRO-6000",
        gpu_count=5,
        profile_label="Cheap but good",
        approx_cost_per_hour_usd=15.15,
        max_context_tokens=262144,
        instance_slug_hint="kimi25-rtxpro",
        summary="Default curated Kimi K2.5 profile for long-context coding and agent workflows on five RTX PRO 6000 GPUs.",
        server_args=KIMI_K25_SERVER_ARGS,
        source_label="Curated fallback",
    ),
)

_STATIC_CATALOG_INFO = QuickDeployCatalogInfo(
    source_label="Curated llama.cpp coding profiles",
    is_fallback=True,
)
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

    text = _read_bundled_catalog_text()
    if text is None:
        _CATALOG_CACHE = (_STATIC_CATALOG_INFO, _STATIC_QUICK_DEPLOY_PROFILES)
        return _CATALOG_CACHE

    try:
        payload = json.loads(text)
        loaded = _profiles_from_catalog_payload(payload)
    except Exception:
        loaded = None

    if loaded is None:
        _CATALOG_CACHE = (_STATIC_CATALOG_INFO, _STATIC_QUICK_DEPLOY_PROFILES)
        return _CATALOG_CACHE

    _CATALOG_CACHE = loaded
    return _CATALOG_CACHE


def _read_bundled_catalog_text() -> str | None:
    try:
        resource = resources.files(_BUNDLED_CATALOG_PACKAGE).joinpath(_BUNDLED_CATALOG_FILENAME)
        return resource.read_text(encoding="utf-8")
    except Exception:
        return None


def _profiles_from_catalog_payload(
    payload: Any,
) -> tuple[QuickDeployCatalogInfo, tuple[QuickDeployProfile, ...]] | None:
    if not isinstance(payload, dict):
        return None
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, list):
        return None

    schema_version = _positive_int(payload.get("schema_version")) or 1
    profiles: list[QuickDeployProfile] = []
    seen_ids: set[str] = set()
    for raw_profile in raw_profiles:
        profile = _profile_from_catalog_row(raw_profile)
        if profile is None or profile.id in seen_ids:
            continue
        if profile.backend == BackendType.LLAMACPP and schema_version >= 2:
            compatibility = evaluate_llamacpp_architecture(profile.gguf_architecture)
            if not compatibility.is_supported:
                continue
        seen_ids.add(profile.id)
        profiles.append(profile)

    if not profiles:
        return None

    source = (
        _clean_string(payload.get("source"))
        or "Artificial Analysis Intelligence Index rankings"
    )
    generated_at = _clean_string(payload.get("generated_at")) or None
    attribution = _clean_string(payload.get("attribution")) or None
    info = QuickDeployCatalogInfo(
        source_label=source,
        generated_at=generated_at,
        attribution=attribution,
        is_fallback=False,
    )
    return (info, tuple(profiles))


def _profile_from_catalog_row(payload: Any) -> QuickDeployProfile | None:
    if not isinstance(payload, dict):
        return None

    profile_id = _clean_string(payload.get("id"))
    display_name = _clean_string(payload.get("display_name"))
    repo_id = _clean_string(payload.get("repo_id"))
    model_name = _clean_string(payload.get("model_name"))
    quant = _clean_string(payload.get("quant"))
    gpu_type = _clean_string(payload.get("gpu_type"))
    profile_label = _clean_string(payload.get("profile_label"))
    instance_slug_hint = _clean_string(payload.get("instance_slug_hint"))
    summary = _clean_string(payload.get("summary"))
    server_args = _clean_string_tuple(payload.get("server_args"))
    gpu_count = _positive_int(payload.get("gpu_count"))
    approx_cost = _nonnegative_float(payload.get("approx_cost_per_hour_usd"))
    max_context = _positive_int(payload.get("max_context_tokens"))
    backend_value = _clean_string(payload.get("backend")) or BackendType.LLAMACPP.value
    try:
        backend = BackendType(backend_value)
    except ValueError:
        return None

    common_required = [
        profile_id,
        display_name,
        gpu_type,
        profile_label,
        instance_slug_hint,
        summary,
    ]
    if not all(common_required):
        return None
    if backend == BackendType.LLAMACPP and not (repo_id and quant):
        return None
    if backend == BackendType.VLLM and not (model_name or repo_id):
        return None
    if server_args is None or gpu_count is None or approx_cost is None or max_context is None:
        return None

    return QuickDeployProfile(
        id=profile_id,
        display_name=display_name,
        repo_id=repo_id,
        quant=quant,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        profile_label=profile_label,
        approx_cost_per_hour_usd=approx_cost,
        max_context_tokens=max_context,
        instance_slug_hint=instance_slug_hint,
        summary=summary,
        server_args=server_args or (),
        required_vram_gb=_positive_float(payload.get("required_vram_gb")),
        resource_tier=_clean_string(payload.get("resource_tier")) or None,
        resource_tier_label=_clean_string(payload.get("resource_tier_label")) or None,
        source_label=_clean_string(payload.get("source_label")) or "Artificial Analysis",
        aa_model_id=_clean_string(payload.get("aa_model_id")) or None,
        aa_model_name=_clean_string(payload.get("aa_model_name")) or None,
        aa_model_slug=_clean_string(payload.get("aa_model_slug")) or None,
        aa_coding_score=_optional_float(payload.get("aa_coding_score")),
        aa_intelligence_score=_optional_float(payload.get("aa_intelligence_score")),
        aa_rank=_positive_int(payload.get("aa_rank")),
        model_size_label=_clean_string(payload.get("model_size_label")) or None,
        backend=backend,
        model_name=model_name or None,
        gguf_architecture=_clean_string(payload.get("gguf_architecture")) or None,
        llamacpp_runtime_id=_clean_string(payload.get("llamacpp_runtime_id")) or None,
    )


def _clean_string(value: Any) -> str:
    return str(value or "").strip()


def _clean_string_tuple(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, list):
        return None
    parsed = tuple(str(item).strip() for item in value if str(item).strip())
    return parsed


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


def _nonnegative_float(value: Any) -> float | None:
    parsed = _optional_float(value)
    if parsed is None or parsed < 0:
        return None
    return parsed


def _positive_float(value: Any) -> float | None:
    parsed = _optional_float(value)
    if parsed is None or parsed <= 0:
        return None
    return parsed


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
