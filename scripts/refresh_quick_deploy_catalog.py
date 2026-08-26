#!/usr/bin/env python3
"""Refresh the bundled maintainer-generated Quick Deploy catalog."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, NamedTuple, Sequence

from llm_launchpad.core.hf_models import GgufQuantMetadata, fetch_gguf_quant_metadata, fetch_model_max_context
from llm_launchpad.core.modal_gpu import ModalGpuSpec, fetch_modal_gpu_catalog
from llm_launchpad.core.naming import slugify_instance_name
from llm_launchpad.core.quick_deploy import (
    GLM5_SERVER_ARGS,
    KIMI_K25_SERVER_ARGS,
    QWEN35_397B_SERVER_ARGS,
)

AA_LLM_MODELS_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
CATALOG_PATH = Path(__file__).resolve().parents[1] / "llm_launchpad" / "data" / "quick_deploy_catalog.json"
ATTRIBUTION = "Benchmark data sourced from Artificial Analysis: https://artificialanalysis.ai/"
POPULAR_ATTRIBUTION = (
    "Model metadata sourced from Hugging Face: https://huggingface.co/unsloth"
)
DEFAULT_MAX_PROFILES = 3
MATCH_THRESHOLD = 90.0
AMBIGUITY_MARGIN = 5.0
RESOURCE_TIER_CHEAP = "cheap"
RESOURCE_TIER_RTX_PRO = "rtx-pro"
RESOURCE_TIER_B200 = "b200"
LOW_VRAM_QUANT = "UD-Q2_K_XL"

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
    RESOURCE_TIER_CHEAP: "$",
    RESOURCE_TIER_RTX_PRO: "$$",
    RESOURCE_TIER_B200: "$$$",
}
_RESOURCE_TIER_DESCRIPTIONS = {
    RESOURCE_TIER_CHEAP: "Slow but cheap",
    RESOURCE_TIER_RTX_PRO: "RTX PRO",
    RESOURCE_TIER_B200: "B200",
}
_MANUAL_OVERRIDES: dict[str, dict[str, object]] = {
    "qwen35397ba17b": {
        "id": "qwen35-397b-rtxpro",
        "display_name": "Qwen3.5 397B A17B",
        "quant": "UD-Q4_K_XL",
        "gpu_type": "RTX-PRO-6000",
        "gpu_count": 3,
        "max_context_tokens": 262144,
        "instance_slug_hint": "qwen35-397b-rtxpro",
        "server_args": QWEN35_397B_SERVER_ARGS,
    },
    "glm5": {
        "id": "glm5-rtxpro",
        "display_name": "GLM-5",
        "quant": "UD-Q4_K_XL",
        "gpu_type": "RTX-PRO-6000",
        "gpu_count": 4,
        "max_context_tokens": 202752,
        "instance_slug_hint": "glm5-rtxpro",
        "server_args": GLM5_SERVER_ARGS,
    },
    "glm51": {
        "id": "glm51-rtxpro",
        "display_name": "GLM-5.1",
        "quant": "UD-Q4_K_XL",
        "gpu_type": "RTX-PRO-6000",
        "gpu_count": 6,
        "max_context_tokens": 202752,
        "instance_slug_hint": "glm51-rtxpro",
        "server_args": GLM5_SERVER_ARGS,
    },
    "kimik25": {
        "id": "kimi25-rtxpro",
        "display_name": "Kimi K2.5",
        "quant": "UD-Q4_K_XL",
        "gpu_type": "RTX-PRO-6000",
        "gpu_count": 5,
        "max_context_tokens": 262144,
        "instance_slug_hint": "kimi25-rtxpro",
        "server_args": KIMI_K25_SERVER_ARGS,
    },
}
_BASE_CONTEXT_REPO_OVERRIDES: dict[str, tuple[str, ...]] = {
    "glm51": ("zai-org/GLM-5.1",),
    "kimik26": ("moonshotai/Kimi-K2.6",),
    "minimaxm27": ("MiniMaxAI/MiniMax-M2.7",),
}


@dataclass(frozen=True)
class AAModelCandidate:
    """Normalized Artificial Analysis model row."""

    aa_model_id: str
    name: str
    slug: str
    creator_name: str
    coding_score: float
    rank: int
    max_context_tokens: int | None = None


@dataclass(frozen=True)
class PopularModelCandidate:
    """Curated, deployment-tested model shown in the Popular Models panel."""

    name: str
    slug: str
    creator_name: str
    repo_id: str
    max_context_tokens: int
    required_vram_floor_gb_by_quant: tuple[tuple[str, float], ...] = ()


POPULAR_MODEL_CANDIDATES: tuple[PopularModelCandidate, ...] = (
    PopularModelCandidate(
        name="Qwen3.8 27B",
        slug="qwen3-8-27b",
        creator_name="Alibaba",
        repo_id="unsloth/Qwen3.8-27B-GGUF",
        max_context_tokens=131072,
        required_vram_floor_gb_by_quant=(
            ("UD-Q2_K_XL", 21.0),
            ("UD-Q4_K_XL", 29.0),
        ),
    ),
    PopularModelCandidate(
        name="DeepSeek V4 Flash 0731",
        slug="deepseek-v4-flash-0731",
        creator_name="DeepSeek",
        repo_id="unsloth/DeepSeek-V4-Flash-0731-GGUF",
        max_context_tokens=131072,
    ),
    PopularModelCandidate(
        name="GLM-5.2",
        slug="glm-5-2",
        creator_name="Z.ai",
        repo_id="unsloth/GLM-5.2-GGUF",
        max_context_tokens=131072,
    ),
)


@dataclass(frozen=True)
class UnslothMatch:
    """High-confidence HF match for one AA model row."""

    repo_id: str
    score: float


class ResourceSelection(NamedTuple):
    """Selected quant plus a viable Modal GPU shape."""

    quant: str
    gpu_type: str
    gpu_count: int
    cost_per_hour_usd: float
    required_vram_gb: float | None


def fetch_aa_llm_models(api_key: str, timeout: float = 20.0) -> dict[str, Any]:
    """Fetch AA LLM benchmark data using the maintainer API key."""
    import requests

    response = requests.get(
        AA_LLM_MODELS_URL,
        headers={"x-api-key": api_key, "Accept": "application/json"},
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Artificial Analysis API returned HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Artificial Analysis API returned a non-object payload")
    return payload


def normalize_aa_candidates(payload: Any, limit: int = 50) -> list[AAModelCandidate]:
    """Return AA rows ranked by Coding Index, excluding known proprietary rows."""
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []

    candidates: list[AAModelCandidate] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        score = _optional_float(_nested_value(row, "evaluations", "artificial_analysis_coding_index"))
        if score is None:
            continue
        open_status = _open_weight_status(row)
        if open_status is False:
            continue
        name = _clean_string(row.get("name"))
        slug = _clean_string(row.get("slug"))
        if not name and not slug:
            continue
        creator = row.get("model_creator")
        creator_name = _clean_string(creator.get("name")) if isinstance(creator, dict) else ""
        candidates.append(
            AAModelCandidate(
                aa_model_id=_clean_string(row.get("id")),
                name=name or slug,
                slug=slug,
                creator_name=creator_name,
                coding_score=score,
                rank=0,
                max_context_tokens=_extract_aa_max_context(row),
            )
        )

    ranked = sorted(candidates, key=lambda candidate: (-candidate.coding_score, candidate.name.casefold()))
    return [
        AAModelCandidate(
            aa_model_id=candidate.aa_model_id,
            name=candidate.name,
            slug=candidate.slug,
            creator_name=candidate.creator_name,
            coding_score=candidate.coding_score,
            rank=index + 1,
            max_context_tokens=candidate.max_context_tokens,
        )
        for index, candidate in enumerate(ranked[: max(1, int(limit))])
    ]


def find_unsloth_gguf_match(candidate: AAModelCandidate, hf_api: Any) -> UnslothMatch | None:
    """Find one high-confidence `unsloth/*-GGUF` repo for an AA model."""
    rows = _list_unsloth_candidates(candidate, hf_api)
    scored: list[UnslothMatch] = []
    seen_repo_ids: set[str] = set()
    for row in rows:
        repo_id = _repo_id_from_hf_row(row)
        if not repo_id or not repo_id.casefold().startswith("unsloth/"):
            continue
        repo_key = repo_id.casefold()
        if repo_key in seen_repo_ids:
            continue
        seen_repo_ids.add(repo_key)
        if not repo_id.casefold().endswith("-gguf"):
            continue
        score = _model_match_score(candidate, repo_id)
        if score >= MATCH_THRESHOLD:
            scored.append(UnslothMatch(repo_id=repo_id, score=score))

    if not scored:
        return None
    scored.sort(key=lambda match: (-match.score, match.repo_id.casefold()))
    if len(scored) > 1 and scored[0].score - scored[1].score < AMBIGUITY_MARGIN:
        return None
    return scored[0]


def build_catalog_payload(
    aa_payload: Any,
    *,
    hf_api: Any,
    modal_gpu_catalog: Sequence[ModalGpuSpec],
    generated_at: str | None = None,
    max_profiles: int = DEFAULT_MAX_PROFILES,
) -> dict[str, Any]:
    """Build the deterministic catalog JSON payload from AA and HF data."""
    profiles: list[dict[str, Any]] = []
    generated_models = 0
    candidates = _latest_family_candidates(normalize_aa_candidates(aa_payload))
    for candidate in candidates:
        match = find_unsloth_gguf_match(candidate, hf_api)
        if match is None:
            continue
        metadata = fetch_gguf_quant_metadata(match.repo_id)
        rows = build_profile_rows(
            candidate,
            match.repo_id,
            metadata=metadata,
            modal_gpu_catalog=modal_gpu_catalog,
        )
        if not rows:
            continue
        profiles.extend(rows)
        generated_models += 1
        if generated_models >= max(1, int(max_profiles)):
            break

    return {
        "schema_version": 1,
        "generated_at": generated_at or _utc_now_iso(),
        "source": "Artificial Analysis coding rankings",
        "attribution": ATTRIBUTION,
        "profiles": profiles,
    }


def build_popular_catalog_payload(
    *,
    modal_gpu_catalog: Sequence[ModalGpuSpec],
    generated_at: str | None = None,
    models: Sequence[PopularModelCandidate] = POPULAR_MODEL_CANDIDATES,
) -> dict[str, Any]:
    """Build the deployment-tested Popular Models catalog from explicit HF repos."""
    profiles: list[dict[str, Any]] = []
    for rank, model in enumerate(models, start=1):
        candidate = AAModelCandidate(
            aa_model_id="",
            name=model.name,
            slug=model.slug,
            creator_name=model.creator_name,
            coding_score=0.0,
            rank=rank,
            max_context_tokens=model.max_context_tokens,
        )
        metadata = fetch_gguf_quant_metadata(model.repo_id)
        metadata = _apply_required_vram_floors(
            metadata,
            model.required_vram_floor_gb_by_quant,
        )
        rows = build_profile_rows(
            candidate,
            model.repo_id,
            metadata=metadata,
            modal_gpu_catalog=modal_gpu_catalog,
        )
        for row in rows:
            for key in (
                "aa_model_id",
                "aa_model_name",
                "aa_model_slug",
                "aa_coding_score",
                "aa_rank",
            ):
                row.pop(key, None)
            row["source_label"] = "Hugging Face"
            row["summary"] = (
                "Curated popular open-weight model with verified llama.cpp GGUF weights."
            )
            row["max_context_tokens"] = model.max_context_tokens
            row["server_args"] = ["--ctx-size", str(model.max_context_tokens)]
        profiles.extend(rows)

    return {
        "schema_version": 1,
        "generated_at": generated_at or _utc_now_iso(),
        "source": "Curated popular open-weight models",
        "attribution": POPULAR_ATTRIBUTION,
        "profiles": profiles,
    }


def _apply_required_vram_floors(
    metadata: GgufQuantMetadata,
    floors: Sequence[tuple[str, float]],
) -> GgufQuantMetadata:
    """Apply deployment-measured runtime floors before selecting GPU shapes."""

    values = dict(metadata.vram_gb_by_quant)
    for quant, floor in floors:
        parsed_floor = _optional_float(floor)
        if parsed_floor is None or parsed_floor <= 0:
            continue
        expected = _quant_key(quant)
        matching_key = next(
            (candidate for candidate in values if _quant_key(candidate) == expected),
            quant,
        )
        values[matching_key] = max(values.get(matching_key, 0.0), parsed_floor)
    return GgufQuantMetadata(
        quantizations=list(metadata.quantizations),
        vram_gb_by_quant=values,
    )


def _latest_family_candidates(candidates: Sequence[AAModelCandidate]) -> list[AAModelCandidate]:
    """Keep only the newest version per model family, then preserve AA score order."""
    latest_by_family: dict[str, AAModelCandidate] = {}
    for candidate in candidates:
        family_key = _candidate_family_key(candidate)
        current = latest_by_family.get(family_key)
        if current is None or _candidate_recency_key(candidate) > _candidate_recency_key(current):
            latest_by_family[family_key] = candidate
    return sorted(
        latest_by_family.values(),
        key=lambda candidate: (-candidate.coding_score, candidate.rank, candidate.name.casefold()),
    )


def _candidate_family_key(candidate: AAModelCandidate) -> str:
    family, _version = _candidate_family_and_version(candidate)
    creator_key = _model_key(candidate.creator_name)
    if creator_key:
        return f"{creator_key}:{family}"
    return family


def _candidate_recency_key(candidate: AAModelCandidate) -> tuple[int, tuple[int, ...], float, int]:
    _family, version = _candidate_family_and_version(candidate)
    return (1 if version else 0, version, candidate.coding_score, -candidate.rank)


def _candidate_family_and_version(candidate: AAModelCandidate) -> tuple[str, tuple[int, ...]]:
    texts = [
        _strip_creator_prefix(candidate.name, candidate.creator_name),
        candidate.name,
        candidate.slug,
    ]
    for text in texts:
        family_version = _extract_family_version(text)
        if family_version is not None:
            return family_version
    fallback = _model_key(candidate.name) or _model_key(candidate.slug) or candidate.aa_model_id
    return (fallback, ())


def _extract_family_version(value: str) -> tuple[str, tuple[int, ...]] | None:
    text = value.strip().casefold()
    if not text:
        return None
    patterns = (
        (r"\bglm[-\s]*(\d+(?:[.-]\d+)*)\b", "glm"),
        (r"\bkimi[-\s]*k[-\s]*(\d+(?:[.-]\d+)*)\b", "kimi-k"),
        (r"\bminimax[-\s]*m[-\s]*(\d+(?:[.-]\d+)*)\b", "minimax-m"),
        (r"\bdeepseek[-\s]*v[-\s]*(\d+(?:[.-]\d+)*)\b", "deepseek-v"),
        (r"\bqwen[-\s]*(\d+(?:[.-]\d+)*)\b", "qwen"),
    )
    for pattern, family in patterns:
        match = re.search(pattern, text)
        if match:
            version = tuple(int(part) for part in re.findall(r"\d+", match.group(1)))
            return (family, version)
    return None


def build_profile_row(
    candidate: AAModelCandidate,
    repo_id: str,
    *,
    metadata: GgufQuantMetadata,
    modal_gpu_catalog: Sequence[ModalGpuSpec],
) -> dict[str, Any] | None:
    """Build one QuickDeployProfile-compatible JSON object."""
    rows = build_profile_rows(
        candidate,
        repo_id,
        metadata=metadata,
        modal_gpu_catalog=modal_gpu_catalog,
    )
    return rows[0] if rows else None


def build_profile_rows(
    candidate: AAModelCandidate,
    repo_id: str,
    *,
    metadata: GgufQuantMetadata,
    modal_gpu_catalog: Sequence[ModalGpuSpec],
) -> list[dict[str, Any]]:
    """Build tiered QuickDeployProfile-compatible JSON rows for one model."""
    override = _manual_override_for(candidate, repo_id)
    price_by_gpu = _price_by_gpu(modal_gpu_catalog)
    display_name = _display_name_for_candidate(candidate, repo_id)
    slug_hint = slugify_instance_name(_repo_model_name(repo_id).removesuffix("-GGUF"))
    if override:
        quant = str(override["quant"])
        cheap = _select_gpu_for_quant(quant, metadata, modal_gpu_catalog, resource_tier=RESOURCE_TIER_CHEAP)
        if cheap is None:
            cheap = ResourceSelection(
                quant=quant,
                gpu_type=str(override["gpu_type"]),
                gpu_count=int(override["gpu_count"]),
                cost_per_hour_usd=_hourly_cost(str(override["gpu_type"]), int(override["gpu_count"]), price_by_gpu),
                required_vram_gb=_required_vram_for_quant(metadata, quant),
            )
            slug_hint = str(override["instance_slug_hint"])
        max_context_tokens = _resolve_max_context_tokens(
            candidate,
            repo_id,
            fallback=int(override["max_context_tokens"]),
        )
        return _profile_rows_for_quant_variants(
            candidate,
            repo_id,
            metadata=metadata,
            modal_gpu_catalog=modal_gpu_catalog,
            display_name=str(override["display_name"]),
            slug_hint=slug_hint,
            max_context_tokens=max_context_tokens,
            server_args=list(override["server_args"]),
            primary_quant=quant,
            primary_cheap=cheap,
        )

    selected = _select_quant_and_gpu(metadata, modal_gpu_catalog)
    if selected is None:
        return []
    max_context_tokens = _resolve_max_context_tokens(candidate, repo_id, fallback=65536)
    return _profile_rows_for_quant_variants(
        candidate,
        repo_id,
        metadata=metadata,
        modal_gpu_catalog=modal_gpu_catalog,
        display_name=display_name,
        slug_hint=slug_hint,
        max_context_tokens=max_context_tokens,
        server_args=["--ctx-size", str(max_context_tokens)],
        primary_quant=selected.quant,
        primary_cheap=selected,
    )


def _profile_rows_for_quant_variants(
    candidate: AAModelCandidate,
    repo_id: str,
    *,
    metadata: GgufQuantMetadata,
    modal_gpu_catalog: Sequence[ModalGpuSpec],
    display_name: str,
    slug_hint: str,
    max_context_tokens: int,
    server_args: list[str],
    primary_quant: str,
    primary_cheap: ResourceSelection,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if _quant_key(primary_quant) != _quant_key(LOW_VRAM_QUANT):
        rows.extend(
            _profile_rows_for_quant(
                candidate,
                repo_id,
                metadata=metadata,
                modal_gpu_catalog=modal_gpu_catalog,
                display_name=display_name,
                slug_hint=slug_hint,
                max_context_tokens=max_context_tokens,
                server_args=server_args,
                quant=LOW_VRAM_QUANT,
            )
        )
    rows.extend(
        _profile_rows_for_quant(
            candidate,
            repo_id,
            metadata=metadata,
            modal_gpu_catalog=modal_gpu_catalog,
            display_name=display_name,
            slug_hint=slug_hint,
            max_context_tokens=max_context_tokens,
            server_args=server_args,
            quant=primary_quant,
            cheap=primary_cheap,
        )
    )
    return rows


def _profile_rows_for_quant(
    candidate: AAModelCandidate,
    repo_id: str,
    *,
    metadata: GgufQuantMetadata,
    modal_gpu_catalog: Sequence[ModalGpuSpec],
    display_name: str,
    slug_hint: str,
    max_context_tokens: int,
    server_args: list[str],
    quant: str,
    cheap: ResourceSelection | None = None,
) -> list[dict[str, Any]]:
    cheap = cheap or _select_gpu_for_quant(quant, metadata, modal_gpu_catalog, resource_tier=RESOURCE_TIER_CHEAP)
    if cheap is None:
        return []
    rtx_pro = _select_gpu_for_quant(cheap.quant, metadata, modal_gpu_catalog, resource_tier=RESOURCE_TIER_RTX_PRO)
    b200 = _select_gpu_for_quant(cheap.quant, metadata, modal_gpu_catalog, resource_tier=RESOURCE_TIER_B200)
    return _profile_rows_from_selections(
        candidate,
        repo_id,
        display_name=display_name,
        slug_hint=slug_hint,
        max_context_tokens=max_context_tokens,
        server_args=server_args,
        cheap=cheap,
        rtx_pro=rtx_pro,
        b200=b200,
    )


def _profile_rows_from_selections(
    candidate: AAModelCandidate,
    repo_id: str,
    *,
    display_name: str,
    slug_hint: str,
    max_context_tokens: int,
    server_args: list[str],
    cheap: ResourceSelection,
    rtx_pro: ResourceSelection | None,
    b200: ResourceSelection | None,
) -> list[dict[str, Any]]:
    selections = [
        (RESOURCE_TIER_CHEAP, cheap),
        (RESOURCE_TIER_RTX_PRO, rtx_pro),
        (RESOURCE_TIER_B200, b200),
    ]
    rows: list[dict[str, Any]] = []
    rows_by_selection: dict[tuple[str, str, int], dict[str, Any]] = {}
    for resource_tier, selection in selections:
        if selection is None:
            continue
        fingerprint = _selection_fingerprint(selection)
        if fingerprint in rows_by_selection:
            _merge_resource_tier_label(rows_by_selection[fingerprint], resource_tier)
            continue
        row = _profile_row_from_selection(
            candidate,
            repo_id,
            display_name=display_name,
            slug_hint=slug_hint,
            max_context_tokens=max_context_tokens,
            server_args=server_args,
            selection=selection,
            resource_tier=resource_tier,
        )
        rows_by_selection[fingerprint] = row
        rows.append(row)
    return rows


def _merge_resource_tier_label(row: dict[str, Any], resource_tier: str) -> None:
    existing_label = _clean_string(row.get("resource_tier_label"))
    next_label = _RESOURCE_TIER_LABELS.get(resource_tier, resource_tier)
    row["resource_tier_label"] = _join_unique_labels(existing_label, next_label)

    existing_description = _clean_string(row.get("profile_label"))
    next_description = _RESOURCE_TIER_DESCRIPTIONS.get(resource_tier, resource_tier)
    row["profile_label"] = _join_unique_labels(existing_description, next_description, separator=" / ")


def _join_unique_labels(existing: str, addition: str, *, separator: str = "/") -> str:
    labels: list[str] = []
    for value in (existing, addition):
        for part in value.split(separator):
            label = part.strip()
            if label and label not in labels:
                labels.append(label)
    return separator.join(labels)


def _profile_row_from_selection(
    candidate: AAModelCandidate,
    repo_id: str,
    *,
    display_name: str,
    slug_hint: str,
    max_context_tokens: int,
    server_args: list[str],
    selection: ResourceSelection,
    resource_tier: str,
) -> dict[str, Any]:
    quant_slug = _quant_slug(selection.quant)
    row = {
        "id": f"{slug_hint}-{quant_slug}-{resource_tier}-{_gpu_slug(selection.gpu_type)}",
        "display_name": display_name,
        "repo_id": repo_id,
        "quant": selection.quant,
        "gpu_type": selection.gpu_type,
        "gpu_count": selection.gpu_count,
        "profile_label": _RESOURCE_TIER_DESCRIPTIONS.get(resource_tier, "AA Coding"),
        "resource_tier": resource_tier,
        "resource_tier_label": _RESOURCE_TIER_LABELS.get(resource_tier, resource_tier),
        "approx_cost_per_hour_usd": round(selection.cost_per_hour_usd, 2),
        "max_context_tokens": max_context_tokens,
        "instance_slug_hint": f"{slug_hint}-{quant_slug}-{resource_tier}",
        "summary": "Artificial Analysis-ranked open-weight coding profile matched to Unsloth GGUF weights.",
        "server_args": list(server_args),
        "source_label": "Artificial Analysis",
        "aa_model_id": candidate.aa_model_id or None,
        "aa_model_name": candidate.name,
        "aa_model_slug": candidate.slug or None,
        "aa_coding_score": candidate.coding_score,
        "aa_rank": candidate.rank,
    }
    _maybe_add_required_vram(row, selection.required_vram_gb)
    return row


def _list_unsloth_candidates(candidate: AAModelCandidate, hf_api: Any) -> list[Any]:
    rows: list[Any] = []
    for search in _hf_search_terms_for_candidate(candidate):
        calls = [
            {"author": "unsloth", "filter": ["text-generation", "gguf"], "search": search, "limit": 25, "full": True},
            {"author": "unsloth", "filter": ["gguf"], "search": search, "limit": 25, "full": True},
        ]
        for kwargs in calls:
            try:
                rows.extend(list(hf_api.list_models(**kwargs)))
            except TypeError:
                fallback_kwargs = dict(kwargs)
                fallback_kwargs.pop("author", None)
                rows.extend(list(hf_api.list_models(**fallback_kwargs)))
    return rows


def _hf_search_terms_for_candidate(candidate: AAModelCandidate) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        text = value.strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            terms.append(text)

    add(candidate.name)
    if candidate.creator_name:
        stripped_name = _strip_creator_prefix(candidate.name, candidate.creator_name)
        add(stripped_name)
    add(candidate.slug)
    for key in {_model_key(candidate.name), _model_key(candidate.slug)}:
        glm_match = re.fullmatch(r"glm(\d)(\d+)", key)
        if glm_match:
            add(f"GLM-{glm_match.group(1)}.{glm_match.group(2)}")
    return terms


def _strip_creator_prefix(name: str, creator_name: str) -> str:
    text = name.strip()
    creator = creator_name.strip()
    if not text or not creator:
        return text
    creator_pattern = re.escape(creator).replace(r"\.", r"\.?")
    return re.sub(rf"(?i)^{creator_pattern}\s*(?:[:/\-|]|\s)\s*", "", text).strip()


def _repo_id_from_hf_row(row: Any) -> str:
    return _clean_string(
        getattr(row, "id", None)
        or getattr(row, "modelId", None)
        or (row.get("id") if isinstance(row, dict) else None)
        or (row.get("modelId") if isinstance(row, dict) else None)
    )


def _model_match_score(candidate: AAModelCandidate, repo_id: str) -> float:
    aa_keys = [_model_key(candidate.name), _model_key(candidate.slug)]
    repo_key = _model_key(_repo_model_name(repo_id).removesuffix("-GGUF"))
    scores: list[float] = []
    for aa_key in aa_keys:
        if not aa_key:
            continue
        if aa_key == repo_key:
            scores.append(100.0)
        elif aa_key in repo_key or repo_key in aa_key:
            scores.append(95.0)
        else:
            scores.append(SequenceMatcher(None, aa_key, repo_key).ratio() * 100.0)
    return max(scores or [0.0])


def _model_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold().replace("gguf", ""))


def _repo_model_name(repo_id: str) -> str:
    return repo_id.split("/", 1)[1] if "/" in repo_id else repo_id


def _resolve_max_context_tokens(
    candidate: AAModelCandidate,
    repo_id: str,
    *,
    fallback: int,
) -> int:
    values: list[int | None] = [candidate.max_context_tokens]
    for base_repo_id in _base_context_repo_candidates(candidate, repo_id):
        values.append(fetch_model_max_context(base_repo_id))
    values.append(fetch_model_max_context(repo_id))
    for value in values:
        if value is not None and value > 0:
            return value
    return max(1, int(fallback))


def _base_context_repo_candidates(candidate: AAModelCandidate, repo_id: str) -> list[str]:
    keys = {
        _model_key(candidate.name),
        _model_key(candidate.slug),
        _model_key(_repo_model_name(repo_id).removesuffix("-GGUF")),
    }
    candidates: list[str] = []
    for key in keys:
        for base_repo_id in _BASE_CONTEXT_REPO_OVERRIDES.get(key, ()):
            if base_repo_id.casefold() not in {candidate.casefold() for candidate in candidates}:
                candidates.append(base_repo_id)
    if repo_id.casefold().startswith("unsloth/") and repo_id.casefold().endswith("-gguf"):
        unsloth_base = repo_id[:-5]
        if unsloth_base.casefold() not in {candidate.casefold() for candidate in candidates}:
            candidates.append(unsloth_base)
    return candidates


def _select_quant_and_gpu(
    metadata: GgufQuantMetadata,
    modal_gpu_catalog: Sequence[ModalGpuSpec],
) -> ResourceSelection | None:
    quantizations = list(metadata.quantizations)
    if not quantizations:
        return None
    price_by_gpu = _price_by_gpu(modal_gpu_catalog)

    candidates: list[tuple[int, float, int, ResourceSelection]] = []
    for quant in sorted(quantizations, key=_quant_sort_key):
        selected = _select_gpu_for_quant(quant, metadata, modal_gpu_catalog)
        if selected is None:
            continue
        quant_rank = _quant_sort_key(quant)
        candidates.append((quant_rank, selected.cost_per_hour_usd, selected.gpu_count, selected))

    if candidates:
        return sorted(candidates)[0][3]

    quant = _preferred_quant(quantizations)
    gpu_type = "RTX-PRO-6000"
    gpu_count = 1
    return ResourceSelection(
        quant=quant,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        cost_per_hour_usd=_hourly_cost(gpu_type, gpu_count, price_by_gpu),
        required_vram_gb=_required_vram_for_quant(metadata, quant),
    )


def _select_gpu_for_quant(
    quant: str,
    metadata: GgufQuantMetadata,
    modal_gpu_catalog: Sequence[ModalGpuSpec],
    *,
    resource_tier: str = RESOURCE_TIER_CHEAP,
) -> ResourceSelection | None:
    required_vram_gb = _required_vram_for_quant(metadata, quant)
    if required_vram_gb is None:
        return None
    if resource_tier == RESOURCE_TIER_RTX_PRO:
        shape = _specific_gpu_shape(required_vram_gb, modal_gpu_catalog, "RTX-PRO-6000")
    elif resource_tier == RESOURCE_TIER_B200:
        shape = _specific_gpu_shape(required_vram_gb, modal_gpu_catalog, "B200")
    else:
        shape = _cost_minimizing_gpu_shape(required_vram_gb, modal_gpu_catalog)
    if shape is None:
        return None
    gpu_type, gpu_count, cost = shape
    return ResourceSelection(
        quant=_metadata_quant_label(metadata, quant),
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        cost_per_hour_usd=cost,
        required_vram_gb=required_vram_gb,
    )


def _required_vram_for_quant(metadata: GgufQuantMetadata, quant: str) -> float | None:
    expected = _quant_key(quant)
    for candidate_quant, value in metadata.vram_gb_by_quant.items():
        if _quant_key(candidate_quant) != expected:
            continue
        parsed = _optional_float(value)
        if parsed is not None and parsed > 0:
            return parsed
    return None


def _metadata_quant_label(metadata: GgufQuantMetadata, quant: str) -> str:
    expected = _quant_key(quant)
    for candidate_quant in metadata.quantizations:
        if _quant_key(candidate_quant) == expected:
            return candidate_quant
    return quant


def _cost_minimizing_gpu_shape(
    required_vram_gb: float,
    modal_gpu_catalog: Sequence[ModalGpuSpec],
) -> tuple[str, int, float] | None:
    """Return the lowest $/hour Modal shape that satisfies required VRAM."""
    price_by_gpu = _price_by_gpu(modal_gpu_catalog)
    available = _available_gpu_types(modal_gpu_catalog)
    candidates: list[tuple[float, int, float, str]] = []
    for gpu_type in available:
        memory_gb = _GPU_MEMORY_GB.get(gpu_type)
        if memory_gb is None:
            continue
        for gpu_count in range(1, 9):
            if memory_gb * gpu_count < required_vram_gb * 1.05:
                continue
            cost = _hourly_cost(gpu_type, gpu_count, price_by_gpu)
            candidates.append((cost, gpu_count, -memory_gb, gpu_type))
            break
    if not candidates:
        return None
    cost, gpu_count, _negative_memory, gpu_type = sorted(candidates)[0]
    return (gpu_type, gpu_count, cost)


def _specific_gpu_shape(
    required_vram_gb: float,
    modal_gpu_catalog: Sequence[ModalGpuSpec],
    gpu_type: str,
) -> tuple[str, int, float] | None:
    """Return a fixed Modal GPU shape that satisfies required VRAM."""
    price_by_gpu = _price_by_gpu(modal_gpu_catalog)
    if gpu_type not in set(_available_gpu_types(modal_gpu_catalog)):
        return None
    memory_gb = _GPU_MEMORY_GB.get(gpu_type)
    if memory_gb is None:
        return None
    for gpu_count in range(1, 9):
        if memory_gb * gpu_count < required_vram_gb * 1.05:
            continue
        return (gpu_type, gpu_count, _hourly_cost(gpu_type, gpu_count, price_by_gpu))
    return None


def _available_gpu_types(modal_gpu_catalog: Sequence[ModalGpuSpec]) -> list[str]:
    values = [_clean_string(getattr(entry, "value", "")) for entry in modal_gpu_catalog]
    values = [value for value in values if value]
    return values or list(_GPU_MEMORY_GB.keys())


def _price_by_gpu(modal_gpu_catalog: Sequence[ModalGpuSpec]) -> dict[str, float]:
    prices = dict(_FALLBACK_GPU_PRICE_PER_HOUR)
    for entry in modal_gpu_catalog:
        value = _clean_string(getattr(entry, "value", ""))
        price = _optional_float(getattr(entry, "price_per_hour_usd", None))
        if value and price is not None and price > 0:
            prices[value] = price
    return prices


def _hourly_cost(gpu_type: str, gpu_count: int, price_by_gpu: dict[str, float]) -> float:
    return price_by_gpu.get(gpu_type, _FALLBACK_GPU_PRICE_PER_HOUR["RTX-PRO-6000"]) * gpu_count


def _preferred_quant(quantizations: Sequence[str]) -> str:
    normalized = {_quant_key(quant): quant for quant in quantizations}
    for preferred in _PREFERRED_QUANT_ORDER:
        key = _quant_key(preferred)
        if key in normalized:
            return normalized[key]
    return quantizations[0]


def _quant_sort_key(quant: str) -> int:
    upper = _quant_key(quant)
    try:
        return _PREFERRED_QUANT_ORDER.index(upper)
    except ValueError:
        return len(_PREFERRED_QUANT_ORDER)


def _quant_key(quant: Any) -> str:
    upper = str(quant or "").strip().upper()
    if upper.startswith("UD_"):
        return f"UD-{upper[3:]}"
    return upper


def _quant_slug(quant: Any) -> str:
    slug = _quant_key(quant).casefold()
    if slug.startswith("ud-"):
        slug = slug[3:]
    return (
        slug.replace("_k_xl", "xl")
        .replace("_k_m", "m")
        .replace("_k_s", "s")
        .replace("_", "-")
        .replace("--", "-")
        .strip("-")
    )


def _maybe_add_required_vram(row: dict[str, Any], required_vram_gb: float | None) -> None:
    if required_vram_gb is None or required_vram_gb <= 0:
        return
    row["required_vram_gb"] = round(required_vram_gb, 1)


def _gpu_slug(gpu_type: str) -> str:
    return gpu_type.casefold().replace("_", "-").replace("+", "plus").replace("!", "")


def _selection_fingerprint(selection: ResourceSelection) -> tuple[str, str, int]:
    return (_quant_key(selection.quant), selection.gpu_type, selection.gpu_count)


def _manual_override_for(candidate: AAModelCandidate, repo_id: str) -> dict[str, object] | None:
    keys = {
        _model_key(candidate.name),
        _model_key(candidate.slug),
        _model_key(_repo_model_name(repo_id).removesuffix("-GGUF")),
    }
    for key in keys:
        override = _MANUAL_OVERRIDES.get(key)
        if override is not None:
            return override
    return None


def _display_name_for_candidate(candidate: AAModelCandidate, repo_id: str) -> str:
    return candidate.name or _repo_model_name(repo_id).removesuffix("-GGUF")


def _open_weight_status(row: dict[str, Any]) -> bool | None:
    status: bool | None = None
    for key, value in _iter_key_values(row):
        key_lower = key.casefold()
        if _is_explicit_weight_status_key(key_lower):
            parsed = _parse_open_weight_value(value, allow_api_only_false=True)
        elif _is_license_key(key_lower):
            parsed = _parse_open_weight_value(value, allow_api_only_false=False)
        elif _is_generic_availability_key(key_lower):
            parsed = _parse_open_weight_value(value, allow_api_only_false=False)
        else:
            continue
        if parsed is False:
            return False
        if parsed is True:
            status = True
    return status


def _extract_aa_max_context(row: dict[str, Any]) -> int | None:
    candidates: list[int] = []
    for key, value in _iter_key_values(row):
        key_lower = key.strip().casefold()
        if key_lower not in {
            "context_window",
            "context_window_tokens",
            "input_context_window",
            "max_context",
            "max_context_length",
            "max_context_tokens",
            "max_position_embeddings",
            "max_seq_len",
            "max_sequence_length",
            "model_max_length",
        }:
            continue
        parsed = _parse_context_value(value)
        if 0 < parsed <= 10_000_000:
            candidates.append(parsed)
    return max(candidates) if candidates else None


def _is_explicit_weight_status_key(key_lower: str) -> bool:
    return (
        key_lower in {"open_weights", "open_weight", "is_open_weight", "weights_available"}
        or ("weight" in key_lower and ("open" in key_lower or "available" in key_lower or "availability" in key_lower))
    )


def _is_license_key(key_lower: str) -> bool:
    return key_lower in {"license", "model_license"} or key_lower.endswith("_license")


def _is_generic_availability_key(key_lower: str) -> bool:
    return key_lower in {"availability", "model_availability"}


def _parse_open_weight_value(value: Any, *, allow_api_only_false: bool) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().casefold()
    if not text:
        return None
    false_tokens = ["proprietary", "closed", "no open weights"]
    if allow_api_only_false:
        false_tokens.append("api only")
    if any(token in text for token in false_tokens):
        return False
    if "open" in text and ("weight" in text or "source" in text):
        return True
    if "commercial use restricted" in text:
        return True
    return None


def _iter_key_values(node: Any) -> Sequence[tuple[str, Any]]:
    pairs: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            pairs.append((str(key), value))
            pairs.extend(_iter_key_values(value))
    elif isinstance(node, list):
        for item in node:
            pairs.extend(_iter_key_values(item))
    return pairs


def _nested_value(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


def _parse_context_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = 0
    if numeric > 0:
        return numeric

    text = str(value or "").strip().casefold().replace(",", "")
    if not text:
        return 0
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([km]?)", text)
    if not match:
        return 0
    number = float(match.group(1))
    unit = match.group(2)
    multiplier = 1
    if unit == "k":
        multiplier = 1000
    elif unit == "m":
        multiplier = 1_000_000
    return int(round(number * multiplier))


def _clean_string(value: Any) -> str:
    return str(value or "").strip()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_catalog(payload: dict[str, Any], path: Path = CATALOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def fetch_modal_gpu_catalog_or_fallback() -> Sequence[ModalGpuSpec]:
    """Fetch Modal GPU metadata, falling back to bundled defaults when docs block automation."""
    try:
        return fetch_modal_gpu_catalog()
    except Exception as exc:
        print(f"Warning: using fallback Modal GPU catalog because live fetch failed: {exc}", file=sys.stderr)
        return []


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CATALOG_PATH)
    parser.add_argument("--max-profiles", type=int, default=DEFAULT_MAX_PROFILES)
    parser.add_argument(
        "--popular",
        action="store_true",
        help="Build the curated Popular Models catalog without an Artificial Analysis API key.",
    )
    args = parser.parse_args(argv)

    modal_gpu_catalog = fetch_modal_gpu_catalog_or_fallback()
    if args.popular:
        catalog = build_popular_catalog_payload(modal_gpu_catalog=modal_gpu_catalog)
    else:
        api_key = os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("ARTIFICIAL_ANALYSIS_API_KEY is required for maintainer catalog refresh")

        from huggingface_hub import HfApi

        aa_payload = fetch_aa_llm_models(api_key)
        catalog = build_catalog_payload(
            aa_payload,
            hf_api=HfApi(),
            modal_gpu_catalog=modal_gpu_catalog,
            max_profiles=args.max_profiles,
        )
    if not catalog["profiles"]:
        raise SystemExit("No Quick Deploy profiles were generated")
    write_catalog(catalog, path=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
