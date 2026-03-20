"""Curated quick-deploy catalog for the TUI landing page."""

from __future__ import annotations

from dataclasses import dataclass
import shlex

from ..protocol.enums import BackendType
from ..protocol.models import DeploymentConfig
from .naming import build_app_name, slugify_instance_name


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


QUICK_DEPLOY_PROFILES: tuple[QuickDeployProfile, ...] = (
    QuickDeployProfile(
        id="qwen35-397b-rtxpro",
        display_name="Qwen3.5 397B A17B",
        repo_id="unsloth/Qwen3.5-397B-A17B-GGUF",
        quant="UD-Q3_K_XL",
        gpu_type="RTX-PRO-6000",
        gpu_count=3,
        profile_label="Cheap but good",
        approx_cost_per_hour_usd=9.09,
        max_context_tokens=262144,
        instance_slug_hint="qwen35-397b-rtxpro",
        summary="Default curated Qwen3.5 profile for long-context coding workloads on three RTX PRO 6000 GPUs.",
        server_args=QWEN35_397B_SERVER_ARGS,
    ),
    QuickDeployProfile(
        id="glm5-rtxpro",
        display_name="GLM-5",
        repo_id="unsloth/GLM-5-GGUF",
        quant="UD-Q2_K_XL",
        gpu_type="RTX-PRO-6000",
        gpu_count=4,
        profile_label="Cheap but good",
        approx_cost_per_hour_usd=12.12,
        max_context_tokens=202752,
        instance_slug_hint="glm5-rtxpro",
        summary="Default curated GLM-5 profile for long-context coding and agent workflows on four RTX PRO 6000 GPUs.",
        server_args=GLM5_SERVER_ARGS,
    ),
    QuickDeployProfile(
        id="kimi25-rtxpro",
        display_name="Kimi K2.5",
        repo_id="unsloth/Kimi-K2.5-GGUF",
        quant="UD-Q2_K_XL",
        gpu_type="RTX-PRO-6000",
        gpu_count=5,
        profile_label="Cheap but good",
        approx_cost_per_hour_usd=15.15,
        max_context_tokens=262144,
        instance_slug_hint="kimi25-rtxpro",
        summary="Default curated Kimi K2.5 profile for long-context coding and agent workflows on five RTX PRO 6000 GPUs.",
        server_args=KIMI_K25_SERVER_ARGS,
    ),
)

_QUICK_DEPLOY_BY_ID = {profile.id: profile for profile in QUICK_DEPLOY_PROFILES}


def list_quick_deploy_profiles() -> tuple[QuickDeployProfile, ...]:
    """Return the immutable quick-deploy catalog."""
    return QUICK_DEPLOY_PROFILES


def get_quick_deploy_profile(profile_id: str) -> QuickDeployProfile:
    """Resolve a quick-deploy profile by identifier."""
    try:
        return _QUICK_DEPLOY_BY_ID[profile_id]
    except KeyError as exc:
        raise KeyError(f"Unknown quick deploy profile: {profile_id}") from exc


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


def format_quick_deploy_model_label(profile: QuickDeployProfile) -> str:
    """Render a quick-deploy model label with its configured quantization."""
    label, quant_suffix = quick_deploy_model_label_parts(profile)
    if not quant_suffix:
        return label
    return f"{label} {quant_suffix}"


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
        config.instance_name = slugify_instance_name(instance_override or app_override)
    elif instance_override:
        config.instance_name = slugify_instance_name(instance_override)
        config.app_name = build_app_name(config.backend, config.instance_name)
    else:
        config.instance_name = slugify_instance_name(profile.instance_slug_hint)
        config.app_name = build_app_name(config.backend, config.instance_name)
    return config
