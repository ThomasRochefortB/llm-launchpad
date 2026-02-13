"""Hugging Face model discovery helpers for model picking."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Literal

ModelRankMode = Literal["downloads", "trending"]
ModelDiscoveryTarget = Literal["vllm", "llamacpp"]


@dataclass(frozen=True)
class ModelCandidate:
    """A model row displayed in the vLLM model picker."""

    repo_id: str
    downloads: int | None = None
    likes: int | None = None
    pipeline_tag: str | None = None
    quantizations: tuple[str, ...] = ()


_CACHE_TTL_SECONDS = 300
_CACHE: dict[tuple[str, str, int], tuple[float, list[ModelCandidate]]] = {}
_GGUF_QUANTS_CACHE: dict[tuple[str, str], tuple[float, list[str]]] = {}
_SORT_BY_MODE: dict[ModelRankMode, str] = {
    "downloads": "downloads",
    "trending": "trending_score",
}
_PREFERRED_QUANT_ORDER = [
    "Q4_K_M",
    "Q4_K_S",
    "Q5_K_M",
    "Q5_K_S",
    "Q6_K",
    "Q8_0",
]
_QUANT_PATTERN = re.compile(
    r"(?i)(?<![a-z0-9])(IQ[1-4]_(?:XS|XXS|S|M|NL)|Q[2-8]_(?:K(?:_[MSXL]+)?|0|1))(?![a-z0-9])"
)


def list_vllm_candidates(mode: ModelRankMode = "downloads", limit: int = 10) -> list[ModelCandidate]:
    """List ranked text-generation models suitable for vLLM selection.

    Results are cached in-memory for a short TTL to avoid repeated API calls
    when users switch back and forth between ranking modes.
    """

    normalized_limit = max(1, int(limit))
    cache_key = ("vllm", mode, normalized_limit)
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached and (now - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1]

    models = _fetch_candidates(mode=mode, limit=normalized_limit, target="vllm")
    _CACHE[cache_key] = (now, models)
    return models


def list_llamacpp_candidates(mode: ModelRankMode = "downloads", limit: int = 10) -> list[ModelCandidate]:
    """List ranked GGUF text-generation models for llama.cpp selection."""

    normalized_limit = max(1, int(limit))
    cache_key = ("llamacpp", mode, normalized_limit)
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached and (now - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1]

    models = _fetch_candidates(mode=mode, limit=normalized_limit, target="llamacpp")
    _CACHE[cache_key] = (now, models)
    return models


def fetch_gguf_quantizations(repo_id: str, revision: str | None = None) -> list[str]:
    """Return detected GGUF quantizations for a model repo."""
    normalized_repo = repo_id.strip()
    if not normalized_repo:
        return []
    revision_key = (revision or "").strip()
    cache_key = (normalized_repo, revision_key)
    now = time.time()
    cached = _GGUF_QUANTS_CACHE.get(cache_key)
    if cached and (now - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1]

    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required for model discovery. Install with: pip install huggingface_hub"
        ) from exc

    api = HfApi()
    info = api.model_info(repo_id=normalized_repo, revision=revision or None, files_metadata=False)
    quantizations = _extract_gguf_quantizations(getattr(info, "siblings", None))
    _GGUF_QUANTS_CACHE[cache_key] = (now, quantizations)
    return quantizations


def _fetch_candidates(mode: ModelRankMode, limit: int, target: ModelDiscoveryTarget) -> list[ModelCandidate]:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required for model discovery. Install with: pip install huggingface_hub"
        ) from exc

    sort = _SORT_BY_MODE.get(mode, "downloads")
    api = HfApi()
    if target == "llamacpp":
        rows = api.list_models(
            filter=["text-generation", "gguf"],
            sort=sort,
            limit=limit * 3,
            full=True,
        )
    else:
        rows = api.list_models(
            filter="text-generation",
            sort=sort,
            limit=limit * 3,
            full=True,
        )
    candidates: list[ModelCandidate] = []
    for row in rows:
        candidate = _normalize_candidate(row, target=target)
        if candidate is None:
            continue
        candidates.append(candidate)
        if len(candidates) >= limit:
            break
    return candidates


def _normalize_candidate(row: Any, target: ModelDiscoveryTarget = "vllm") -> ModelCandidate | None:
    repo_id = str(_first_non_empty(getattr(row, "id", None), getattr(row, "modelId", None))).strip()
    if not repo_id:
        return None

    pipeline_tag = _to_optional_str(getattr(row, "pipeline_tag", None))
    tags = getattr(row, "tags", None)
    if not _is_text_generation(pipeline_tag, tags):
        return None
    if target == "llamacpp" and not _has_tag(tags, "gguf"):
        return None
    quantizations = _extract_gguf_quantizations(getattr(row, "siblings", None)) if target == "llamacpp" else []

    return ModelCandidate(
        repo_id=repo_id,
        downloads=_to_optional_int(getattr(row, "downloads", None)),
        likes=_to_optional_int(getattr(row, "likes", None)),
        pipeline_tag=pipeline_tag,
        quantizations=tuple(quantizations),
    )


def _is_text_generation(pipeline_tag: str | None, tags: Any) -> bool:
    if pipeline_tag and pipeline_tag.strip() == "text-generation":
        return True
    if not isinstance(tags, list):
        return False
    lowered = {str(tag).strip().lower() for tag in tags}
    return "text-generation" in lowered


def _has_tag(tags: Any, expected: str) -> bool:
    if not isinstance(tags, list):
        return False
    lowered = {str(tag).strip().lower() for tag in tags}
    return expected in lowered


def _extract_gguf_quantizations(siblings: Any) -> list[str]:
    if not isinstance(siblings, list):
        return []
    detected: set[str] = set()
    for sibling in siblings:
        filename = str(getattr(sibling, "rfilename", "")).strip()
        if not filename or not filename.lower().endswith(".gguf"):
            continue
        for match in _QUANT_PATTERN.findall(filename):
            detected.add(str(match).upper())
    return sorted(detected, key=_quant_sort_key)


def _quant_sort_key(quant: str) -> tuple[int, int, str]:
    upper = quant.upper()
    if upper in _PREFERRED_QUANT_ORDER:
        return (0, _PREFERRED_QUANT_ORDER.index(upper), upper)
    return (1, 0, upper)


def _first_non_empty(*values: object | None) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text.strip():
            return text
    return ""


def _to_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

