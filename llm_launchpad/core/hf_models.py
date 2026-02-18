"""Hugging Face model discovery helpers for model picking."""

from __future__ import annotations

import html
import json
import math
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


@dataclass(frozen=True)
class GgufQuantMetadata:
    """Quantization list plus optional VRAM estimates in GB."""

    quantizations: list[str]
    vram_gb_by_quant: dict[str, float]


@dataclass(frozen=True)
class VllmMemoryBreakdown:
    """Estimated vLLM memory usage in decimal GB."""

    total_gb: float
    weights_gb: float
    kv_cache_gb: float
    overhead_gb: float
    context_tokens: int


_CACHE_TTL_SECONDS = 300
_VLLM_MEMORY_CACHE_SCHEMA_VERSION = 3
_HF_REQUEST_TIMEOUT_SECONDS = 10.0
_HF_ETAG_TIMEOUT_SECONDS = 10.0
_DEFAULT_CONTEXT_TOKENS = 8192
_CACHE: dict[tuple[str, str, int], tuple[float, list[ModelCandidate]]] = {}
_GGUF_QUANTS_CACHE: dict[tuple[str, str], tuple[float, list[str]]] = {}
_GGUF_QUANT_METADATA_CACHE: dict[tuple[str, str], tuple[float, GgufQuantMetadata]] = {}
_VLLM_MEMORY_CACHE: dict[tuple[int, str, str, int], tuple[float, VllmMemoryBreakdown]] = {}
_HF_JSON_FILE_CACHE: dict[tuple[str, str, str], tuple[float, dict[str, Any] | None]] = {}
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
_MEMORY_PATTERN = re.compile(r"(?i)^\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGTP]?I?B)?\s*$")
_MEMORY_HINTS = ("memory", "vram", "ram", "require", "required", "footprint", "size")
_MEMORY_UNITS = ("b", "kb", "kib", "mb", "mib", "gb", "gib", "tb", "tib", "pb", "pib")
_MODEL_TENSORS_PROPS_RE = re.compile(r'data-target="ModelTensorsParams"[^>]*data-props="([^"]+)"')
_PARAMS_PATTERN = re.compile(r"(?i)(\d+(?:\.\d+)?)\s*([bm])(?=$|[^a-z0-9])")
_HF_PAGE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def fetch_vllm_memory_breakdown(
    repo_id: str,
    revision: str | None = None,
    context_tokens: int | None = None,
) -> VllmMemoryBreakdown | None:
    """Estimate vLLM memory needs using HF metadata with a repo-name fallback."""
    from ..core.backend import ModalBackend
    if ModalBackend.is_shutting_down():
        return None
    normalized_repo = repo_id.strip()
    if not normalized_repo:
        return None
    revision_key = (revision or "").strip()
    context_key = max(1, int(context_tokens)) if context_tokens is not None else 0
    cache_key = (_VLLM_MEMORY_CACHE_SCHEMA_VERSION, normalized_repo, revision_key, context_key)
    now = time.time()
    cached = _VLLM_MEMORY_CACHE.get(cache_key)
    if cached and (now - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1]

    info = None
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        try:
            info = _call_model_info(
                api=api,
                repo_id=normalized_repo,
                revision=revision_key or None,
                expand=["cardData", "config", "safetensors", "tags"],
            )
        except Exception:
            info = _call_model_info(
                api=api,
                repo_id=normalized_repo,
                revision=revision_key or None,
            )
    except Exception:
        info = None

    config = getattr(info, "config", None)
    safetensors = getattr(info, "safetensors", None)
    tags = getattr(info, "tags", None)
    card_data = getattr(info, "cardData", None)
    repo_config: dict[str, Any] | None = None
    tokenizer_config: dict[str, Any] | None = None
    generation_config: dict[str, Any] | None = None

    primary_config = config if isinstance(config, dict) else {}
    if not primary_config:
        repo_config = _load_repo_json_file(normalized_repo, revision_key or None, "config.json")
        if isinstance(repo_config, dict):
            primary_config = repo_config

    weights_bytes = _extract_safetensors_total_bytes(safetensors)
    parameter_count = _extract_parameter_count(repo_id=normalized_repo, config=primary_config, safetensors=safetensors)
    if weights_bytes is None and parameter_count is None:
        return None

    if weights_bytes is None and parameter_count is not None:
        dtype_bytes = _estimate_weight_dtype_bytes(repo_id=normalized_repo, config=primary_config, tags=tags)
        weights_bytes = parameter_count * dtype_bytes
    if weights_bytes is None:
        return None

    num_layers = _extract_int_config(
        primary_config,
        "num_hidden_layers",
        "n_layer",
        "num_layers",
        "n_layers",
        "decoder_layers",
    ) or _extract_int_config(
        repo_config,
        "num_hidden_layers",
        "n_layer",
        "num_layers",
        "n_layers",
        "decoder_layers",
    )
    hidden_size = _extract_int_config(
        primary_config,
        "hidden_size",
        "d_model",
        "n_embd",
        "dim",
        "model_dim",
    )
    if num_layers is None or hidden_size is None:
        if repo_config is None:
            repo_config = _load_repo_json_file(normalized_repo, revision_key or None, "config.json")
        num_layers = num_layers or _extract_int_config(
            repo_config,
            "num_hidden_layers",
            "n_layer",
            "num_layers",
            "n_layers",
            "decoder_layers",
        )
        hidden_size = hidden_size or _extract_int_config(
            repo_config,
            "hidden_size",
            "d_model",
            "n_embd",
            "dim",
            "model_dim",
        )
    if (num_layers is None or hidden_size is None) and parameter_count is not None:
        approx_layers, approx_hidden = _approx_transformer_shape(parameter_count)
        num_layers = num_layers or approx_layers
        hidden_size = hidden_size or approx_hidden

    model_max_context = _pick_max_positive_int(
        _extract_model_max_context(primary_config),
        _extract_model_max_context(card_data),
    )
    if model_max_context is None:
        if repo_config is None:
            repo_config = _load_repo_json_file(normalized_repo, revision_key or None, "config.json")
        tokenizer_config = _load_repo_json_file(normalized_repo, revision_key or None, "tokenizer_config.json")
        generation_config = _load_repo_json_file(normalized_repo, revision_key or None, "generation_config.json")
        model_max_context = _pick_max_positive_int(
            _extract_model_max_context(repo_config),
            _extract_model_max_context(tokenizer_config),
            _extract_model_max_context(generation_config),
            _extract_model_max_context(card_data),
        )
    effective_context = _resolve_context_tokens(requested=context_tokens, model_max=model_max_context)
    if effective_context is None:
        return None

    kv_cache_bytes = 0.0
    if num_layers and hidden_size and effective_context > 0:
        # KV bytes ~= 2 (K,V) * layers * hidden_size * context * 2 bytes (fp16/bf16).
        kv_cache_bytes = float(2 * num_layers * hidden_size * effective_context * 2)

    overhead_bytes = (weights_bytes + kv_cache_bytes) * 0.20
    total_bytes = weights_bytes + kv_cache_bytes + overhead_bytes
    if total_bytes <= 0:
        return None

    estimate = VllmMemoryBreakdown(
        total_gb=total_bytes / 1_000_000_000.0,
        weights_gb=weights_bytes / 1_000_000_000.0,
        kv_cache_gb=kv_cache_bytes / 1_000_000_000.0,
        overhead_gb=overhead_bytes / 1_000_000_000.0,
        context_tokens=effective_context,
    )
    _VLLM_MEMORY_CACHE[cache_key] = (now, estimate)
    return estimate


def _call_model_info(
    api: Any,
    repo_id: str,
    revision: str | None,
    expand: list[str] | None = None,
) -> Any:
    from ..core.backend import ModalBackend
    if ModalBackend.is_shutting_down():
        raise RuntimeError("Shutdown requested")
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "revision": revision,
    }
    if expand is not None:
        kwargs["expand"] = expand
    try:
        return api.model_info(timeout=_HF_REQUEST_TIMEOUT_SECONDS, **kwargs)
    except TypeError:
        if ModalBackend.is_shutting_down():
            raise RuntimeError("Shutdown requested")
        return api.model_info(**kwargs)


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
    """Return detected GGUF quantizations for a model repo.

    Backward-compatible wrapper around `fetch_gguf_quant_metadata`.
    """
    normalized_repo = repo_id.strip()
    if not normalized_repo:
        return []
    revision_key = (revision or "").strip()
    cache_key = (normalized_repo, revision_key)
    now = time.time()
    cached = _GGUF_QUANTS_CACHE.get(cache_key)
    if cached and (now - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1]

    metadata = fetch_gguf_quant_metadata(repo_id=normalized_repo, revision=revision_key or None)
    quantizations = list(metadata.quantizations)
    _GGUF_QUANTS_CACHE[cache_key] = (now, quantizations)
    return quantizations


def fetch_gguf_quant_metadata(repo_id: str, revision: str | None = None) -> GgufQuantMetadata:
    """Return detected GGUF quantizations and per-quant VRAM estimates in GB."""
    from ..core.backend import ModalBackend
    if ModalBackend.is_shutting_down():
        return GgufQuantMetadata(quantizations=[], vram_gb_by_quant={})
    normalized_repo = repo_id.strip()
    if not normalized_repo:
        return GgufQuantMetadata(quantizations=[], vram_gb_by_quant={})
    revision_key = (revision or "").strip()
    cache_key = (normalized_repo, revision_key)
    now = time.time()
    cached = _GGUF_QUANT_METADATA_CACHE.get(cache_key)
    if cached and (now - cached[0]) < _CACHE_TTL_SECONDS:
        cached_metadata = cached[1]
        return GgufQuantMetadata(
            quantizations=list(cached_metadata.quantizations),
            vram_gb_by_quant=dict(cached_metadata.vram_gb_by_quant),
        )

    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required for model discovery. Install with: pip install huggingface_hub"
        ) from exc

    api = HfApi()
    if ModalBackend.is_shutting_down():
        return GgufQuantMetadata(quantizations=[], vram_gb_by_quant={})
    info = api.model_info(
        repo_id=normalized_repo,
        revision=revision_key or None,
        expand=["siblings", "gguf"],
    )
    quantizations = _extract_gguf_quantizations(getattr(info, "siblings", None))
    vram_gb_by_quant = _extract_gguf_vram_by_quant(getattr(info, "gguf", None))
    page_quant_data = _fetch_gguf_quantization_data_from_model_page(normalized_repo)
    page_quantizations, page_vram_gb_by_quant = _extract_quantizations_and_vram_from_quantization_data(page_quant_data)
    if page_quantizations:
        quantizations = page_quantizations
    if page_vram_gb_by_quant:
        vram_gb_by_quant = page_vram_gb_by_quant
    if quantizations:
        allowed = {quant.upper() for quant in quantizations}
        vram_gb_by_quant = {quant: value for quant, value in vram_gb_by_quant.items() if quant in allowed}

    metadata = GgufQuantMetadata(
        quantizations=list(quantizations),
        vram_gb_by_quant=dict(vram_gb_by_quant),
    )
    _GGUF_QUANT_METADATA_CACHE[cache_key] = (now, metadata)
    _GGUF_QUANTS_CACHE[cache_key] = (now, list(quantizations))
    return GgufQuantMetadata(
        quantizations=list(metadata.quantizations),
        vram_gb_by_quant=dict(metadata.vram_gb_by_quant),
    )


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


def _extract_gguf_vram_by_quant(gguf_payload: Any) -> dict[str, float]:
    if gguf_payload is None:
        return {}

    vram_gb_by_quant: dict[str, float] = {}

    def _record(quant: str, value: float) -> None:
        upper_quant = quant.strip().upper()
        if not upper_quant:
            return
        if value <= 0:
            return
        current = vram_gb_by_quant.get(upper_quant)
        if current is None or value > current:
            vram_gb_by_quant[upper_quant] = value

    def _extract_quants(value: Any) -> set[str]:
        if value is None:
            return set()
        text = str(value).strip()
        if not text:
            return set()
        return {str(match).upper() for match in _QUANT_PATTERN.findall(text)}

    def _is_memory_hint(key: str) -> bool:
        lower = key.strip().lower()
        return any(token in lower for token in _MEMORY_HINTS)

    def _collect_memory_scalars(node: Any) -> list[float]:
        values: list[float] = []
        if isinstance(node, dict):
            for key, value in node.items():
                key_text = str(key).strip().lower()
                if _is_memory_hint(key_text):
                    parsed = _parse_memory_to_gb(value)
                    if parsed is not None:
                        values.append(parsed)
                if isinstance(value, (dict, list, tuple)):
                    values.extend(_collect_memory_scalars(value))
                elif isinstance(value, str) and _contains_memory_unit(value):
                    parsed = _parse_memory_to_gb(value)
                    if parsed is not None:
                        values.append(parsed)
        elif isinstance(node, (list, tuple)):
            for item in node:
                values.extend(_collect_memory_scalars(item))
        return values

    def _walk(node: Any, active_quants: set[str] | None = None) -> None:
        inherited_quants = set(active_quants or set())
        if isinstance(node, dict):
            node_quants = set(inherited_quants)
            for key, value in node.items():
                key_text = str(key).strip()
                key_quants = _extract_quants(key_text)
                node_quants.update(key_quants)

                if key_quants:
                    direct = _parse_memory_to_gb(value)
                    if direct is not None:
                        for quant in key_quants:
                            _record(quant, direct)
                    for nested_value in _collect_memory_scalars(value):
                        for quant in key_quants:
                            _record(quant, nested_value)

                if _is_memory_hint(key_text):
                    parsed = _parse_memory_to_gb(value)
                    if parsed is not None:
                        targets = key_quants or node_quants or inherited_quants
                        for quant in targets:
                            _record(quant, parsed)

                if isinstance(value, (dict, list, tuple)):
                    _walk(value, node_quants)
                    continue

                value_quants = _extract_quants(value)
                if value_quants:
                    node_quants.update(value_quants)
                scalar_memory = _parse_memory_to_gb(value)
                if scalar_memory is not None and (key_quants or _is_memory_hint(key_text)):
                    targets = key_quants or value_quants or inherited_quants
                    for quant in targets:
                        _record(quant, scalar_memory)

            contextual_memory: list[float] = []
            for key, value in node.items():
                key_text = str(key).strip().lower()
                if _is_memory_hint(key_text):
                    parsed = _parse_memory_to_gb(value)
                    if parsed is not None:
                        contextual_memory.append(parsed)
                elif isinstance(value, str) and _contains_memory_unit(value):
                    parsed = _parse_memory_to_gb(value)
                    if parsed is not None:
                        contextual_memory.append(parsed)
            if node_quants and contextual_memory:
                for quant in node_quants:
                    for memory_value in contextual_memory:
                        _record(quant, memory_value)
            return

        if isinstance(node, (list, tuple)):
            for item in node:
                _walk(item, inherited_quants)

    _walk(gguf_payload)
    return vram_gb_by_quant


def _contains_memory_unit(value: str) -> bool:
    lowered = value.strip().lower()
    return any(unit in lowered for unit in _MEMORY_UNITS)


def _extract_quantizations_and_vram_from_quantization_data(quantization_data: Any) -> tuple[list[str], dict[str, float]]:
    if not isinstance(quantization_data, dict):
        return ([], {})
    levels = quantization_data.get("variantsByQuantizationLevels")
    if not isinstance(levels, dict):
        return ([], {})

    quantizations: list[str] = []
    seen: set[str] = set()
    vram_gb_by_quant: dict[str, float] = {}

    def _level_sort_key(value: Any) -> tuple[int, int, str]:
        key = str(value).strip()
        if key.isdigit():
            return (0, int(key), key)
        return (1, 0, key)

    for level in sorted(levels.keys(), key=_level_sort_key):
        variants = levels.get(level)
        if not isinstance(variants, list):
            continue
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            label = str(variant.get("label", "")).strip().upper()
            if not label:
                continue
            if label not in seen:
                seen.add(label)
                quantizations.append(label)
            parsed_size = _parse_memory_to_gb(variant.get("size"))
            if parsed_size is not None and parsed_size > 0:
                current = vram_gb_by_quant.get(label)
                if current is None or parsed_size > current:
                    vram_gb_by_quant[label] = parsed_size

    return (quantizations, vram_gb_by_quant)


def _fetch_gguf_quantization_data_from_model_page(repo_id: str, timeout: float = 10.0) -> Any:
    from ..core.backend import ModalBackend
    try:
        import requests
    except Exception:
        return None

    url = f"https://huggingface.co/{repo_id}"
    try:
        response = requests.get(url, timeout=min(timeout, _HF_REQUEST_TIMEOUT_SECONDS), headers=_HF_PAGE_HEADERS)
    except Exception:
        if ModalBackend.is_shutting_down():
            return None
        return None
    if response.status_code >= 400:
        return None
    return _extract_gguf_quantization_data_from_model_page_html(response.text)


def _extract_gguf_quantization_data_from_model_page_html(page_html: str) -> Any:
    for match in _MODEL_TENSORS_PROPS_RE.finditer(page_html):
        encoded = match.group(1)
        if not encoded:
            continue
        try:
            decoded = html.unescape(encoded)
            payload = json.loads(decoded)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        quant_data = payload.get("ggufQuantizationData")
        if quant_data is not None:
            return quant_data
    return None


def _parse_memory_to_gb(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if number <= 0:
            return None
        if number >= 1_000_000_000:
            return number / 1_000_000_000.0
        return number

    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace(",", "").lstrip("~≈")
    match = _MEMORY_PATTERN.fullmatch(normalized)
    if not match:
        return None
    number = float(match.group(1))
    if number <= 0:
        return None
    unit = (match.group(2) or "").upper()
    if not unit:
        if number >= 1_000_000_000:
            return number / 1_000_000_000.0
        return number

    multipliers = {
        "B": 1 / 1_000_000_000.0,
        "KB": 1 / 1_000_000.0,
        "KIB": 1 / float(1024**2),
        "MB": 1 / 1000.0,
        "MIB": 1 / float(1024),
        "GB": 1.0,
        "GIB": 1.0,
        "TB": 1000.0,
        "TIB": float(1024),
        "PB": 1_000_000.0,
        "PIB": float(1024**2),
    }
    multiplier = multipliers.get(unit)
    if multiplier is None:
        return None
    return number * multiplier


def _quant_sort_key(quant: str) -> tuple[int, int, str]:
    upper = quant.upper()
    if upper in _PREFERRED_QUANT_ORDER:
        return (0, _PREFERRED_QUANT_ORDER.index(upper), upper)
    return (1, 0, upper)


def _extract_safetensors_total_bytes(payload: Any) -> float | None:
    if not isinstance(payload, dict):
        return None
    total = payload.get("total")
    try:
        numeric = float(total)
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    return numeric


def _extract_parameter_count(repo_id: str, config: Any, safetensors: Any) -> float | None:
    direct = _extract_int_config(config, "num_parameters", "parameter_count", "n_params")
    if direct is not None and direct > 0:
        return float(direct)

    if isinstance(safetensors, dict):
        parameters = safetensors.get("parameters")
        if isinstance(parameters, dict):
            total = 0.0
            for value in parameters.values():
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if numeric > 0:
                    total += numeric
            if total > 0:
                return total

    return _parse_parameter_count_from_repo_id(repo_id)


def _extract_int_config(config: Any, *keys: str) -> int | None:
    if not isinstance(config, dict):
        return None
    for key in keys:
        value = config.get(key)
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            continue
        if numeric > 0:
            return numeric
    return None


def _extract_model_max_context(config: Any) -> int | None:
    keys = {
        "max_position_embeddings",
        "max_seq_len",
        "max_sequence_length",
        "seq_length",
        "context_length",
        "n_ctx",
        "model_max_length",
    }
    max_found: int | None = None

    def _walk(node: Any) -> None:
        nonlocal max_found
        if isinstance(node, dict):
            for key, value in node.items():
                key_text = str(key).strip().lower()
                if key_text in keys:
                    numeric = _parse_context_value(value)
                    # Guard against tokenizer sentinel values.
                    if 0 < numeric <= 10_000_000:
                        max_found = numeric if max_found is None else max(max_found, numeric)
                if isinstance(value, (dict, list, tuple)):
                    _walk(value)
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                _walk(item)

    _walk(config)
    rope_scaled = _extract_rope_scaled_context(config)
    if rope_scaled is not None:
        max_found = rope_scaled if max_found is None else max(max_found, rope_scaled)
    return max_found


def _extract_rope_scaled_context(config: Any) -> int | None:
    if not isinstance(config, dict):
        return None

    candidates: list[int] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            rope = node.get("rope_scaling")
            if isinstance(rope, dict):
                try:
                    factor = float(rope.get("factor"))
                except (TypeError, ValueError):
                    factor = 0.0
                original = _pick_max_positive_int(
                    _to_positive_int(rope.get("original_max_position_embeddings")),
                    _to_positive_int(rope.get("original_max_position_embedding")),
                )
                if original is None:
                    original = _pick_max_positive_int(
                        _to_positive_int(node.get("max_position_embeddings")),
                        _to_positive_int(node.get("max_seq_len")),
                    )
                if factor > 1 and original is not None:
                    scaled = int(round(original * factor))
                    if 0 < scaled <= 10_000_000:
                        candidates.append(scaled)
            for value in node.values():
                if isinstance(value, (dict, list, tuple)):
                    _walk(value)
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                _walk(item)

    _walk(config)
    return max(candidates) if candidates else None


def _to_positive_int(value: Any) -> int | None:
    numeric = _parse_context_value(value)
    if numeric <= 0:
        return None
    return numeric


def _parse_context_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower().replace(",", "")
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


def _pick_max_positive_int(*values: int | None) -> int | None:
    filtered = [value for value in values if value is not None and value > 0]
    return max(filtered) if filtered else None


def _load_repo_json_file(repo_id: str, revision: str | None, filename: str) -> dict[str, Any] | None:
    from ..core.backend import ModalBackend
    if ModalBackend.is_shutting_down():
        return None
    normalized_repo = repo_id.strip()
    if not normalized_repo or not filename.strip():
        return None
    revision_key = (revision or "").strip()
    cache_key = (normalized_repo, revision_key, filename.strip())
    now = time.time()
    cached = _HF_JSON_FILE_CACHE.get(cache_key)
    if cached and (now - cached[0]) < _CACHE_TTL_SECONDS:
        payload = cached[1]
        return dict(payload) if isinstance(payload, dict) else None

    def _fetch_and_cache_http_fallback() -> dict[str, Any] | None:
        parsed_http = _fetch_repo_json_file_via_http(
            repo_id=normalized_repo,
            revision=revision_key or None,
            filename=filename.strip(),
            timeout=_HF_REQUEST_TIMEOUT_SECONDS,
        )
        payload = dict(parsed_http) if isinstance(parsed_http, dict) else None
        _HF_JSON_FILE_CACHE[cache_key] = (now, payload)
        return dict(payload) if isinstance(payload, dict) else None

    try:
        import huggingface_hub
    except ImportError:
        return _fetch_and_cache_http_fallback()
    except Exception:
        return _fetch_and_cache_http_fallback()

    hf_hub_download = getattr(huggingface_hub, "hf_hub_download", None)
    if not callable(hf_hub_download):
        return _fetch_and_cache_http_fallback()

    if ModalBackend.is_shutting_down():
        return None

    try:
        try:
            path = hf_hub_download(
                repo_id=normalized_repo,
                filename=filename.strip(),
                revision=revision_key or None,
                etag_timeout=_HF_ETAG_TIMEOUT_SECONDS,
            )
        except TypeError:
            path = hf_hub_download(
                repo_id=normalized_repo,
                filename=filename.strip(),
                revision=revision_key or None,
            )
    except Exception:
        if ModalBackend.is_shutting_down():
            return None
        return _fetch_and_cache_http_fallback()

    try:
        if path is None or not isinstance(path, (str, bytes)):
            _HF_JSON_FILE_CACHE[cache_key] = (now, None)
            return None
        with open(path, "r", encoding="utf-8") as handle:
            parsed = json.load(handle)
    except Exception:
        return _fetch_and_cache_http_fallback()

    if not isinstance(parsed, dict):
        return _fetch_and_cache_http_fallback()

    payload = dict(parsed)
    _HF_JSON_FILE_CACHE[cache_key] = (now, payload)
    return dict(payload)


def _fetch_repo_json_file_via_http(
    repo_id: str, revision: str | None, filename: str, timeout: float = _HF_REQUEST_TIMEOUT_SECONDS
) -> dict[str, Any] | None:
    from ..core.backend import ModalBackend
    try:
        import requests
    except Exception:
        return None
    ref = (revision or "main").strip() or "main"
    url = f"https://huggingface.co/{repo_id}/resolve/{ref}/{filename}"
    try:
        response = requests.get(url, timeout=timeout, headers=_HF_PAGE_HEADERS)
    except Exception:
        if ModalBackend.is_shutting_down():
            return None
        return None
    if response.status_code >= 400:
        return None
    try:
        parsed = response.json()
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _parse_parameter_count_from_repo_id(repo_id: str) -> float | None:
    text = repo_id.strip()
    if not text:
        return None
    matches = _PARAMS_PATTERN.findall(text)
    if not matches:
        return None
    largest = 0.0
    for number_text, unit_text in matches:
        try:
            number = float(number_text)
        except (TypeError, ValueError):
            continue
        if number <= 0:
            continue
        unit = unit_text.upper()
        scale = 1_000_000_000.0 if unit == "B" else 1_000_000.0
        largest = max(largest, number * scale)
    return largest if largest > 0 else None


def _estimate_weight_dtype_bytes(repo_id: str, config: Any, tags: Any) -> float:
    samples: list[str] = [repo_id]
    if isinstance(config, dict):
        dtype = config.get("torch_dtype")
        if dtype is not None:
            samples.append(str(dtype))
    if isinstance(tags, list):
        for tag in tags:
            samples.append(str(tag))
    text = " ".join(samples).lower()

    if any(token in text for token in ("int4", "4bit", "4-bit", "nf4", "fp4")):
        return 0.5
    if any(token in text for token in ("fp8", "float8", "int8", "8bit", "8-bit")):
        return 1.0
    if any(token in text for token in ("fp32", "float32", "f32")):
        return 4.0
    if any(token in text for token in ("bf16", "bfloat16", "fp16", "float16", "f16")):
        return 2.0
    return 2.0


def _approx_transformer_shape(parameter_count: float) -> tuple[int, int]:
    params_b = parameter_count / 1_000_000_000.0
    if params_b <= 2:
        num_layers = 24
    elif params_b <= 8:
        num_layers = 32
    elif params_b <= 14:
        num_layers = 40
    elif params_b <= 32:
        num_layers = 64
    elif params_b <= 72:
        num_layers = 80
    else:
        num_layers = 96

    hidden = int(math.sqrt(parameter_count / max(1.0, 12.0 * num_layers)))
    # Round to transformer-friendly multiples.
    hidden = max(1024, int(round(hidden / 128.0) * 128))
    return num_layers, hidden


def _resolve_context_tokens(requested: int | None, model_max: int | None) -> int:
    if model_max is not None and model_max > 0:
        return int(model_max)
    if requested is not None:
        return max(1, int(requested))
    return _DEFAULT_CONTEXT_TOKENS


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
