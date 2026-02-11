"""Hugging Face model discovery helpers for vLLM model picking."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

ModelRankMode = Literal["downloads", "trending"]


@dataclass(frozen=True)
class ModelCandidate:
    """A model row displayed in the vLLM model picker."""

    repo_id: str
    downloads: int | None = None
    likes: int | None = None
    pipeline_tag: str | None = None


_CACHE_TTL_SECONDS = 300
_CACHE: dict[tuple[str, int], tuple[float, list[ModelCandidate]]] = {}
_SORT_BY_MODE: dict[ModelRankMode, str] = {
    "downloads": "downloads",
    "trending": "trending_score",
}


def list_vllm_candidates(mode: ModelRankMode = "downloads", limit: int = 10) -> list[ModelCandidate]:
    """List ranked text-generation models suitable for vLLM selection.

    Results are cached in-memory for a short TTL to avoid repeated API calls
    when users switch back and forth between ranking modes.
    """

    normalized_limit = max(1, int(limit))
    cache_key = (mode, normalized_limit)
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached and (now - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1]

    models = _fetch_candidates(mode=mode, limit=normalized_limit)
    _CACHE[cache_key] = (now, models)
    return models


def _fetch_candidates(mode: ModelRankMode, limit: int) -> list[ModelCandidate]:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required for model discovery. Install with: pip install huggingface_hub"
        ) from exc

    sort = _SORT_BY_MODE.get(mode, "downloads")
    api = HfApi()
    rows = api.list_models(
        filter="text-generation",
        sort=sort,
        limit=limit * 3,
        full=True,
    )
    candidates: list[ModelCandidate] = []
    for row in rows:
        candidate = _normalize_candidate(row)
        if candidate is None:
            continue
        candidates.append(candidate)
        if len(candidates) >= limit:
            break
    return candidates


def _normalize_candidate(row: Any) -> ModelCandidate | None:
    repo_id = str(_first_non_empty(getattr(row, "id", None), getattr(row, "modelId", None))).strip()
    if not repo_id:
        return None

    pipeline_tag = _to_optional_str(getattr(row, "pipeline_tag", None))
    tags = getattr(row, "tags", None)
    if not _is_text_generation(pipeline_tag, tags):
        return None

    return ModelCandidate(
        repo_id=repo_id,
        downloads=_to_optional_int(getattr(row, "downloads", None)),
        likes=_to_optional_int(getattr(row, "likes", None)),
        pipeline_tag=pipeline_tag,
    )


def _is_text_generation(pipeline_tag: str | None, tags: Any) -> bool:
    if pipeline_tag and pipeline_tag.strip() == "text-generation":
        return True
    if not isinstance(tags, list):
        return False
    lowered = {str(tag).strip().lower() for tag in tags}
    return "text-generation" in lowered


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

