"""Shared test fixtures for the now-dynamic quick-deploy catalog."""

from __future__ import annotations

from llm_launchpad.core.quick_deploy import (
    QuickDeployCatalogInfo,
    QuickDeployProfile,
    activate_quick_deploy_catalog,
)

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

STATIC_LIKE_PROFILES: tuple[QuickDeployProfile, ...] = (
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

STATIC_LIKE_CATALOG_INFO = QuickDeployCatalogInfo(
    source_label="Test curated profiles",
    is_live=True,
    ready=True,
)


def activate_static_like_catalog() -> tuple[QuickDeployCatalogInfo, tuple[QuickDeployProfile, ...]]:
    """Activate the static-like catalog used by many screen tests."""
    return activate_quick_deploy_catalog(STATIC_LIKE_CATALOG_INFO, STATIC_LIKE_PROFILES)
