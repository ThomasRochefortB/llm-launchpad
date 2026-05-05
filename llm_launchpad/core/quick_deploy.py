"""Quick-deploy catalog for the TUI landing page."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
import json
import shlex
from typing import Any

from ..protocol.enums import BackendType
from ..protocol.models import DeploymentConfig
from .naming import build_app_name, infer_instance_from_app_name, slugify_instance_name


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
    aa_rank: int | None = None


@dataclass(frozen=True)
class QuickDeployCatalogInfo:
    """Metadata describing the active quick-deploy catalog."""

    source_label: str
    generated_at: str | None = None
    attribution: str | None = None
    is_fallback: bool = False


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

QUICK_DEPLOY_PROFILES = _STATIC_QUICK_DEPLOY_PROFILES
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

    profiles: list[QuickDeployProfile] = []
    seen_ids: set[str] = set()
    for raw_profile in raw_profiles:
        profile = _profile_from_catalog_row(raw_profile)
        if profile is None or profile.id in seen_ids:
            continue
        seen_ids.add(profile.id)
        profiles.append(profile)

    if not profiles:
        return None

    source = _clean_string(payload.get("source")) or "Artificial Analysis coding rankings"
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
    quant = _clean_string(payload.get("quant"))
    gpu_type = _clean_string(payload.get("gpu_type"))
    profile_label = _clean_string(payload.get("profile_label"))
    instance_slug_hint = _clean_string(payload.get("instance_slug_hint"))
    summary = _clean_string(payload.get("summary"))
    server_args = _clean_string_tuple(payload.get("server_args"))
    gpu_count = _positive_int(payload.get("gpu_count"))
    approx_cost = _nonnegative_float(payload.get("approx_cost_per_hour_usd"))
    max_context = _positive_int(payload.get("max_context_tokens"))

    if not all([profile_id, display_name, repo_id, quant, gpu_type, profile_label, instance_slug_hint, summary]):
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
        aa_rank=_positive_int(payload.get("aa_rank")),
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


def build_quick_deploy_config(
    profile: QuickDeployProfile,
    *,
    instance_name: str = "",
    app_name: str = "",
    do_warmup: bool = True,
    show_debug_logs: bool = False,
) -> DeploymentConfig:
    """Build a llama.cpp deployment config for a curated quick-deploy profile."""
    config = DeploymentConfig(backend=BackendType.LLAMACPP)
    config.repo_id = profile.repo_id
    config.quant = profile.quant
    config.gpu_type = profile.gpu_type
    config.gpu_count = profile.gpu_count
    config.server_args = shlex.join(profile.server_args)
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
        config.app_name = build_app_name(config.backend, config.instance_name)
    else:
        config.instance_name = slugify_instance_name(profile.instance_slug_hint)
        config.app_name = build_app_name(config.backend, config.instance_name)
    return config
