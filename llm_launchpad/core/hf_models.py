"""Hugging Face model discovery helpers for model picking."""

from __future__ import annotations

import html
import json
import math
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from threading import Event, Lock
from typing import Any, Literal
from collections.abc import Generator

from ..gguf_files import GGUF_SHARD_RE, is_target_gguf
from ..protocol.models import GgufArtifact
from .coerce import optional_str, positive_int
from .diagnostics import log_debug
from .hf_budget import HubLookupIncomplete
from .gguf_metadata import (
    GgufMtpCapability,
    GgufMtpStatus,
    GgufServingMetadata,
    fetch_gguf_mtp_capability,
    fetch_gguf_serving_metadata,
)
from .shutdown import is_shutting_down

ModelRankMode = Literal["downloads", "trending"]
ModelDiscoveryTarget = Literal["vllm", "llamacpp"]


class HubModelPageUnavailable(RuntimeError):
    """The model page could not be read, so its weight sizes are unknown.

    Carries the response where there was one, so callers can tell a rate limit
    apart from a page that is simply gone and retry accordingly.
    """

    def __init__(self, repo_id: str, message: str, *, response: Any = None) -> None:
        super().__init__(f"Could not read the model page for {repo_id}: {message}")
        self.repo_id = repo_id
        self.response = response


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
    """GGUF architecture, quantizations, and optional VRAM estimates in GB."""

    quantizations: list[str]
    vram_gb_by_quant: dict[str, float]
    architecture: str | None = None
    mtp: GgufMtpCapability | None = None
    context_length: int | None = None
    block_count: int | None = None
    embedding_length: int | None = None
    attention_head_count: int | None = None
    attention_head_count_kv: int | None = None
    attention_key_length: int | None = None
    attention_value_length: int | None = None
    serving_metadata: GgufServingMetadata | None = None
    # Set when the weight-size table could not be read. Its presence means the
    # sizes are unknown, never that the model publishes none.
    weight_size_error: str | None = None
    artifacts_by_quant: dict[str, GgufArtifact] | None = None


@dataclass(frozen=True)
class VllmMemoryBreakdown:
    """Estimated vLLM memory usage in decimal GB."""

    total_gb: float
    weights_gb: float
    kv_cache_gb: float
    overhead_gb: float
    context_tokens: int
    vision_working_memory_gb: float | None = 0.0


_CACHE_TTL_SECONDS = 300
_VLLM_MEMORY_CACHE_SCHEMA_VERSION = 5
_HF_REQUEST_TIMEOUT_SECONDS = 10.0
_HF_ETAG_TIMEOUT_SECONDS = 10.0
_DEFAULT_CONTEXT_TOKENS = 8192
# Activations, CUDA graphs, the profiler's own headroom: everything vLLM holds
# beyond the weights and the cache, as a fraction of both.
_MEMORY_OVERHEAD_FRACTION = 0.20
_CACHE: dict[tuple[str, str, int], tuple[float, list[ModelCandidate]]] = {}
_GGUF_QUANT_METADATA_CACHE: dict[
    tuple[str, str, str, bool, bool, bool], tuple[float, GgufQuantMetadata]
] = {}
_VLLM_MEMORY_CACHE: dict[tuple[int, str, str, int], tuple[float, VllmMemoryBreakdown]] = {}
_HF_JSON_FILE_CACHE: dict[tuple[str, str, str], tuple[float, dict[str, Any] | None]] = {}
_SORT_BY_MODE: dict[ModelRankMode, str] = {
    "downloads": "downloads",
    "trending": "trending_score",
}
_PREFERRED_QUANT_ORDER = [
    "UD-Q4_K_XL",
    "Q4_K_M",
    "UD-Q3_K_XL",
    "UD-Q2_K_XL",
    "Q4_K_S",
    "Q5_K_M",
    "Q5_K_S",
    "Q6_K",
    "Q8_0",
]
_QUANT_PATTERN = re.compile(
    r"(?i)(?<![a-z0-9])((?:UD[-_])?(?:IQ[1-4]_(?:XXS|XS|S|M|NL)|Q[2-8]_(?:K(?:_[MSXL]+)?|0|1)))(?![a-z0-9])"
)
_HARDWARE_QUANT_LABEL_RE = re.compile(
    r"(?i)(?<![a-z0-9])((?:DSPARK\s+)?(?:(?:UD[-_])?"
    r"(?:IQ[1-4]_(?:XXS|XS|S|M|NL)|Q[2-8]_(?:K(?:_[MSXL]+)?|0(?:_L)?|1)|TQ[12]_0)"
    r"|MXFP4(?:_MOE)?|BF16|F16|F32))(?![a-z0-9])"
)
_EMBEDDED_MTP_LABEL = re.compile(r"(?i)(?:^|[-_])mtp(?:[-_./]|$)")
_MEMORY_PATTERN = re.compile(r"(?i)^\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGTP]?I?B)?\s*$")
_MEMORY_SEARCH_PATTERN = re.compile(r"(?i)([0-9]+(?:\.[0-9]+)?)\s*([KMGTP]?I?B)")
_MEMORY_HINTS = ("memory", "vram", "ram", "require", "required", "footprint", "size")
_MEMORY_UNITS = ("b", "kb", "kib", "mb", "mib", "gb", "gib", "tb", "tib", "pb", "pib")
_MODEL_TENSORS_PROPS_RE = re.compile(r'data-target="ModelTensorsParams"[^>]*data-props="([^"]+)"')
_HTML_BLOCK_BREAK_RE = re.compile(r"(?i)<\s*(?:br|/p|/div|/li|/tr|/td|/th|/h[1-6])\b[^>]*>")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_PARAMS_PATTERN = re.compile(r"(?i)(\d+(?:\.\d+)?)\s*([bm])(?=$|[^a-z0-9])")
# A mixture-of-experts repository states its size as experts-by-size:
# "Mixtral-8x7B" is not a 7B model. Reading only the second number sized
# the best-known MoE family at a sixth of its weights, which is the unsafe
# direction for a VRAM estimate. The product slightly overshoots the true
# total (experts share attention and embeddings), and overshooting is the
# direction that still fits.
_EXPERT_PARAMS_PATTERN = re.compile(
    r"(?i)(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*([bm])(?=$|[^a-z0-9])"
)
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
    if is_shutting_down():
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

    # ``info.config`` is Hugging Face's trimmed view: architecture, model type
    # and the tokenizer, but never the layer shape, and never the
    # ``text_config``/``vision_config`` split a multimodal repository keeps
    # its shape under. A non-empty trimmed view used to stop the repository's
    # own config.json from being read at all, so every such model was sized
    # from a guessed transformer rather than from its own numbers.
    root_config = config if isinstance(config, dict) else {}
    if _model_shape_config(root_config) is None:
        repo_config = _load_repo_json_file(normalized_repo, revision_key or None, "config.json")
        if isinstance(repo_config, dict):
            root_config = {**repo_config, **root_config}

    multimodal = isinstance(root_config.get("vision_config"), dict)
    primary_config = _model_shape_config(root_config) or root_config
    weights_bytes = _safetensors_weight_bytes(safetensors)
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
        # KV bytes ~= 2 (K,V) * cached layers * KV width * context * 2 bytes
        # (fp16/bf16). The KV width is the grouped-query width -- KV heads by
        # head dimension -- not the model's hidden size: reading the hidden
        # size charges every query head for a cache it shares, which on a
        # 4-KV-head model is eight times the true figure. Only the layers that
        # keep a cache are counted, because a hybrid model's linear-attention
        # layers hold a fixed-size recurrent state instead of growing with the
        # context.
        kv_width = _kv_cache_width(primary_config, hidden_size=hidden_size)
        cached_layers = _kv_cache_layer_count(primary_config, num_layers=num_layers)
        kv_cache_bytes = float(2 * cached_layers * kv_width * effective_context * 2)

    overhead_bytes = (weights_bytes + kv_cache_bytes) * _MEMORY_OVERHEAD_FRACTION
    total_bytes = weights_bytes + kv_cache_bytes + overhead_bytes
    if total_bytes <= 0:
        return None

    estimate = VllmMemoryBreakdown(
        vision_working_memory_gb=None if multimodal else 0.0,
        total_gb=total_bytes / 1_000_000_000.0,
        weights_gb=weights_bytes / 1_000_000_000.0,
        kv_cache_gb=kv_cache_bytes / 1_000_000_000.0,
        overhead_gb=overhead_bytes / 1_000_000_000.0,
        context_tokens=effective_context,
    )
    _VLLM_MEMORY_CACHE[cache_key] = (now, estimate)
    return estimate


def rescale_vllm_memory_breakdown(
    estimate: VllmMemoryBreakdown | None, context_tokens: int | None
) -> VllmMemoryBreakdown | None:
    """Restate an estimate at a different served context length.

    The cache is the only term that moves with the context, and it moves
    linearly, so a breakdown measured at the model's native length answers for
    any other length without another Hugging Face round trip.
    """
    if estimate is None:
        return None
    requested = positive_int(context_tokens)
    if requested is None or estimate.context_tokens <= 0 or requested == estimate.context_tokens:
        return estimate
    kv_cache_gb = estimate.kv_cache_gb * (requested / estimate.context_tokens)
    overhead_gb = (estimate.weights_gb + kv_cache_gb) * _MEMORY_OVERHEAD_FRACTION
    return replace(
        estimate,
        kv_cache_gb=kv_cache_gb,
        overhead_gb=overhead_gb,
        total_gb=estimate.weights_gb + kv_cache_gb + overhead_gb,
        context_tokens=requested,
    )


def fetch_model_max_context(repo_id: str, revision: str | None = None) -> int | None:
    """Return the maximum context length advertised by Hugging Face metadata."""
    return _fetch_model_max_context(repo_id=repo_id, revision=revision, visited=set())


def _fetch_model_max_context(
    repo_id: str,
    revision: str | None = None,
    visited: set[str] | None = None,
) -> int | None:
    if is_shutting_down():
        return None
    normalized_repo = repo_id.strip()
    if not normalized_repo:
        return None
    visited = visited if visited is not None else set()
    repo_key = normalized_repo.casefold()
    if repo_key in visited:
        return None
    visited.add(repo_key)
    revision_key = (revision or "").strip()

    info = None
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        try:
            info = _call_model_info(
                api=api,
                repo_id=normalized_repo,
                revision=revision_key or None,
                expand=["cardData", "config", "tags"],
            )
        except Exception:
            info = _call_model_info(
                api=api,
                repo_id=normalized_repo,
                revision=revision_key or None,
            )
    except Exception:
        info = None

    candidates = [
        _extract_model_max_context(getattr(info, "config", None)),
        _extract_model_max_context(getattr(info, "cardData", None)),
    ]
    if any(value is not None for value in candidates):
        return _pick_max_positive_int(*candidates)

    for base_repo_id in _extract_base_model_repo_ids(info):
        base_context = _fetch_model_max_context(
            repo_id=base_repo_id,
            revision=None,
            visited=visited,
        )
        if base_context is not None:
            return base_context

    config = _load_repo_json_file(normalized_repo, revision_key or None, "config.json")
    tokenizer_config = _load_repo_json_file(normalized_repo, revision_key or None, "tokenizer_config.json")
    generation_config = _load_repo_json_file(normalized_repo, revision_key or None, "generation_config.json")
    return _pick_max_positive_int(
        _extract_model_max_context(config),
        _extract_model_max_context(tokenizer_config),
        _extract_model_max_context(generation_config),
    )


def _extract_base_model_repo_ids(info: Any) -> list[str]:
    candidates: list[str] = []

    def _record(value: Any) -> None:
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("base_model:"):
                text = text.split(":", 1)[1].strip()
            repo_id = _normalize_hf_repo_id(text)
            if repo_id and repo_id.casefold() not in {
                item.casefold() for item in candidates
            }:
                candidates.append(repo_id)
            return
        if isinstance(value, dict):
            for key, nested in value.items():
                key_text = str(key).strip().casefold()
                if key_text in {
                    "base_model",
                    "base_models",
                    "base_model_name",
                    "base_model_name_or_path",
                    "model_name_or_path",
                } or isinstance(nested, (dict, list, tuple)):
                    _record(nested)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                _record(item)

    _record(getattr(info, "cardData", None))
    _record(getattr(info, "tags", None))
    return candidates


def _normalize_hf_repo_id(value: str) -> str | None:
    """Return an ``owner/model`` ID from a Hub repo ID or canonical URL."""
    text = value.strip()
    prefix = "https://huggingface.co/"
    if text.casefold().startswith(prefix):
        text = text[len(prefix) :]
    text = text.strip("/")
    if not text or "/" not in text:
        return None
    if any(char.isspace() for char in text):
        return None
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*",
        text,
    ):
        return None
    return text


def _call_model_info(
    api: Any,
    repo_id: str,
    revision: str | None,
    expand: list[str] | None = None,
    *,
    files_metadata: bool = False,
) -> Any:
    if is_shutting_down():
        raise RuntimeError("Shutdown requested")
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "revision": revision,
    }
    if expand is not None:
        kwargs["expand"] = expand
    if files_metadata:
        kwargs["files_metadata"] = True
    try:
        return api.model_info(timeout=_HF_REQUEST_TIMEOUT_SECONDS, **kwargs)
    except TypeError:
        if is_shutting_down():
            raise RuntimeError("Shutdown requested") from None
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


def fetch_gguf_quant_metadata(
    repo_id: str,
    revision: str | None = None,
    *,
    inspect_mtp: bool = False,
    inspect_serving: bool = False,
    mtp_quant: str | None = None,
    force_refresh: bool = False,
    require_weight_sizes: bool = False,
    hf_api: Any | None = None,
) -> GgufQuantMetadata:
    """Return detected GGUF quantizations and per-quant VRAM estimates in GB.

    Catalog callers requiring both serving metadata and weight sizes get exact,
    revision-pinned Hub file sizes. Other callers can use the model-page hints.
    """
    if is_shutting_down():
        return GgufQuantMetadata(quantizations=[], vram_gb_by_quant={})
    normalized_repo = repo_id.strip()
    if not normalized_repo:
        return GgufQuantMetadata(quantizations=[], vram_gb_by_quant={})
    revision_key = (revision or "").strip()
    quant_key = (mtp_quant or "").strip().casefold()
    cache_key = (normalized_repo, revision_key, quant_key, inspect_mtp, inspect_serving, require_weight_sizes)
    now = time.time()
    cached = _GGUF_QUANT_METADATA_CACHE.get(cache_key)
    if not force_refresh and cached and (now - cached[0]) < _CACHE_TTL_SECONDS:
        return _copy_gguf_quant_metadata(cached[1])

    api = hf_api
    if api is None:
        try:
            from huggingface_hub import HfApi
        except Exception as exc:
            raise RuntimeError(
                "huggingface_hub is required for model discovery. Install with: pip install huggingface_hub"
            ) from exc
        api = HfApi()
    if is_shutting_down():
        return GgufQuantMetadata(quantizations=[], vram_gb_by_quant={})
    if inspect_serving and require_weight_sizes:
        info = _call_model_info(api, normalized_repo, revision_key or None, files_metadata=True)
        artifacts = _gguf_artifacts(getattr(info, "siblings", None), getattr(info, "sha", None))
        quants = sorted(artifacts, key=_quant_sort_key)
        metadata = GgufQuantMetadata(
            quantizations=quants,
            vram_gb_by_quant={quant: artifact.size_bytes / 1e9 for quant, artifact in artifacts.items()},
            artifacts_by_quant=artifacts,
        )
        representative = _normalize_quant_label(mtp_quant) if mtp_quant else next(iter(quants), "")
        if representative in artifacts:
            try:
                metadata = gguf_metadata_for_artifact(normalized_repo, metadata, artifacts[representative])
            except Exception:
                if mtp_quant:
                    raise
                # Keep the file listing so the catalog can retry this header
                # and still size other quantizations if the retry fails.
                return metadata
        _GGUF_QUANT_METADATA_CACHE[cache_key] = (now, metadata)
        return _copy_gguf_quant_metadata(metadata)
    info = _call_model_info(
        api=api,
        repo_id=normalized_repo,
        revision=revision_key or None,
        expand=["siblings", "gguf"],
    )
    siblings = getattr(info, "siblings", None)
    quantizations = _extract_gguf_quantizations(siblings)
    gguf_payload = getattr(info, "gguf", None)
    architecture = _extract_gguf_architecture(gguf_payload)
    context_length = _extract_gguf_positive_int(
        gguf_payload,
        "context_length",
        "max_context_length",
    )
    block_count = _extract_gguf_positive_int(
        gguf_payload,
        "block_count",
        "num_hidden_layers",
    )
    embedding_length = _extract_gguf_positive_int(
        gguf_payload,
        "embedding_length",
        "hidden_size",
    )
    attention_head_count = _extract_gguf_positive_int(
        gguf_payload,
        "attention.head_count",
        "attention_head_count",
        "num_attention_heads",
    )
    attention_head_count_kv = _extract_gguf_positive_int(
        gguf_payload,
        "attention.head_count_kv",
        "attention_head_count_kv",
        "num_key_value_heads",
    )
    attention_key_length = _extract_gguf_positive_int(
        gguf_payload,
        "attention.key_length",
        "attention_key_length",
        "head_dim",
    )
    attention_value_length = _extract_gguf_positive_int(
        gguf_payload,
        "attention.value_length",
        "attention_value_length",
        "head_dim",
    )
    serving_metadata = None
    if inspect_serving:
        serving_metadata = fetch_gguf_serving_metadata(
            normalized_repo,
            siblings,
            revision=revision_key or None,
            quant=mtp_quant,
        )
        if serving_metadata is not None:
            architecture = serving_metadata.architecture or architecture
            context_length = serving_metadata.context_length or context_length
            block_count = serving_metadata.block_count or block_count
            embedding_length = serving_metadata.embedding_length or embedding_length
            attention_head_count = (
                serving_metadata.attention_head_count or attention_head_count
            )
            attention_head_count_kv = (
                serving_metadata.attention_head_count_kv
                if serving_metadata.attention_head_count_kv is not None else attention_head_count_kv
            )
            attention_key_length = (
                serving_metadata.attention_key_length or attention_key_length
            )
            attention_value_length = (
                serving_metadata.attention_value_length or attention_value_length
            )
    vram_gb_by_quant = _extract_gguf_vram_by_quant(gguf_payload)
    page_failure: HubModelPageUnavailable | None = None
    page_quant_data: Any = None
    try:
        page_quant_data = _fetch_gguf_quantization_data_from_model_page(normalized_repo)
    except HubModelPageUnavailable as exc:
        page_failure = exc
    page_quantizations, page_vram_gb_by_quant = _extract_quantizations_and_vram_from_quantization_data(page_quant_data)
    # Model-page and Hub API tables can include quant labels from add-ons.
    target_quants: set[str] = set()
    auxiliary_quants: set[str] = set()
    for filename in _gguf_filenames(siblings):
        labels = target_quants if is_target_gguf(filename) else auxiliary_quants
        labels.update(_normalize_quant_label(label) for label in _HARDWARE_QUANT_LABEL_RE.findall(filename))
    auxiliary_only = auxiliary_quants - target_quants
    page_quantizations = [quant for quant in page_quantizations if quant not in auxiliary_only]
    for quant in auxiliary_only:
        page_vram_gb_by_quant.pop(quant, None)
        vram_gb_by_quant.pop(quant, None)
    if page_quantizations:
        quantizations = page_quantizations
    if page_vram_gb_by_quant:
        vram_gb_by_quant = page_vram_gb_by_quant
    if quantizations:
        allowed = {quant.upper() for quant in quantizations}
        vram_gb_by_quant = {quant: value for quant, value in vram_gb_by_quant.items() if quant in allowed}

    # A page failure only matters where it cost us the sizes: the API payload
    # carries them for some repositories, and there the page is optional.
    weight_size_error: str | None = None
    if page_failure is not None and not vram_gb_by_quant:
        if require_weight_sizes:
            raise page_failure
        weight_size_error = str(page_failure)

    # A complete serving header already contains the target file's NextN count.
    # Keep the same revision and quant key when sharing it with the MTP pass.
    mtp = None
    if serving_metadata is not None and serving_metadata.architecture and serving_metadata.source_file:
        layers = serving_metadata.nextn_predict_layers
        mtp = GgufMtpCapability(
            status=GgufMtpStatus.SUPPORTED if layers > 0 else GgufMtpStatus.UNSUPPORTED,
            nextn_predict_layers=layers, source_file=serving_metadata.source_file,
        )
    elif inspect_mtp:
        mtp = fetch_gguf_mtp_capability(
            normalized_repo, siblings, revision=revision_key or None, quant=mtp_quant,
        )

    metadata = GgufQuantMetadata(
        quantizations=list(quantizations),
        vram_gb_by_quant=dict(vram_gb_by_quant),
        architecture=architecture,
        mtp=mtp if inspect_mtp else None,
        context_length=context_length,
        block_count=block_count,
        embedding_length=embedding_length,
        attention_head_count=attention_head_count,
        attention_head_count_kv=attention_head_count_kv,
        attention_key_length=attention_key_length,
        attention_value_length=attention_value_length,
        serving_metadata=serving_metadata,
        weight_size_error=weight_size_error,
    )
    if weight_size_error is None:
        # Never cache a reading the page failed to supply: one throttled
        # request would otherwise answer for this repository for the whole
        # cache window, including the retries meant to recover from it.
        _GGUF_QUANT_METADATA_CACHE[cache_key] = (now, metadata)
        if inspect_serving and not inspect_mtp and mtp is not None:
            mtp_key = (normalized_repo, revision_key, quant_key, True, True, require_weight_sizes)
            _GGUF_QUANT_METADATA_CACHE[mtp_key] = (now, replace(metadata, mtp=mtp))
    return _copy_gguf_quant_metadata(metadata)


def _copy_gguf_quant_metadata(metadata: GgufQuantMetadata) -> GgufQuantMetadata:
    return replace(
        metadata,
        quantizations=list(metadata.quantizations),
        vram_gb_by_quant=dict(metadata.vram_gb_by_quant),
        artifacts_by_quant=None if metadata.artifacts_by_quant is None else dict(metadata.artifacts_by_quant),
    )


def _gguf_artifacts(siblings: Any, revision: str | None) -> dict[str, GgufArtifact]:
    """Resolve one complete target shard set per unambiguous quant label."""
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise HubLookupIncomplete("Hugging Face did not report the GGUF repository's commit SHA.")
    groups: dict[str, dict[tuple[str, int], dict[int, tuple[str, int | None]]]] = {}
    for sibling in siblings or ():
        name = getattr(sibling, "rfilename", "")
        if not isinstance(name, str) or not is_target_gguf(name):
            continue
        labels = _HARDWARE_QUANT_LABEL_RE.findall(PurePosixPath(name).stem)
        if not labels:
            labels = _HARDWARE_QUANT_LABEL_RE.findall(PurePosixPath(name).parent.name)
        if not labels:
            continue
        quant = _normalize_quant_label(labels[-1])
        split = GGUF_SHARD_RE.fullmatch(name)
        key = (split[1], int(split[3])) if split else (name, 1)
        index = int(split[2]) if split else 1
        parts = groups.setdefault(quant, {}).setdefault(key, {})
        parts[index] = (name, getattr(sibling, "size", None) if index not in parts else None)
    artifacts: dict[str, GgufArtifact] = {}
    for quant, variants in groups.items():
        # Some publishers ship both a plain target and multiple MTP variants
        # under the same quant label. Prefer the unique plain artifact in
        # that case; arbitrary variants with the same label stay ambiguous.
        plain = [key for key in variants if not _EMBEDDED_MTP_LABEL.search(key[0])]
        candidates = plain or list(variants)
        if len(candidates) != 1:
            continue
        key = candidates[0]
        _, count = key
        parts = variants[key]
        if not parts or len(parts) != count or min(parts) != 1 or max(parts) != count:
            continue
        files = tuple(parts[index] for index in sorted(parts))
        if any(not isinstance(size, int) or size <= 0 for _, size in files):
            continue
        try:
            artifacts[quant] = GgufArtifact(
                revision, tuple((name, size) for name, size in files if size is not None),
            )
        except ValueError:
            continue
    return artifacts


def gguf_metadata_for_artifact(
    repo_id: str, metadata: GgufQuantMetadata, artifact: GgufArtifact,
) -> GgufQuantMetadata:
    """Read the selected shard header at the same immutable revision as its size."""
    layout = metadata.serving_metadata
    if layout is not None and layout.source_file == artifact.entrypoint:
        return metadata
    layout = _pinned_gguf_header(repo_id, artifact.revision, artifact.entrypoint)
    layers = layout.nextn_predict_layers
    return replace(
        metadata,
        architecture=layout.architecture,
        mtp=GgufMtpCapability(
            status=GgufMtpStatus.SUPPORTED if layers else GgufMtpStatus.UNSUPPORTED,
            nextn_predict_layers=layers,
            source_file=layout.source_file,
        ),
        context_length=layout.context_length,
        block_count=layout.block_count,
        embedding_length=layout.embedding_length,
        attention_head_count=layout.attention_head_count,
        attention_head_count_kv=layout.attention_head_count_kv,
        attention_key_length=layout.attention_key_length,
        attention_value_length=layout.attention_value_length,
        serving_metadata=layout,
    )


_GGUF_HEADER_CACHE_SCHEMA_VERSION = 1
_GGUF_HEADER_CACHE_MAX_ENTRIES = 2048
_PINNED_HEADER_MEMORY_MAX_ENTRIES = 1024
_PINNED_HEADER_LOCK = Lock()
_PINNED_HEADER_CACHE: dict[tuple[str, str, str], GgufServingMetadata] = {}
_PINNED_HEADER_FLIGHTS: dict[tuple[str, str, str], tuple[Event, list[Any]]] = {}
_ACTIVE_GGUF_HEADER_CACHE: ContextVar[GgufHeaderCache | None] = ContextVar(
    "_ACTIVE_GGUF_HEADER_CACHE", default=None
)


def _gguf_header_cache_path() -> Path:
    from .config import SETTINGS_DIR as current_settings_dir

    return current_settings_dir / "gguf_header_cache.json"


def _serving_metadata_from_dict(payload: Any) -> GgufServingMetadata | None:
    if not isinstance(payload, dict):
        return None
    try:
        fields: dict[str, Any] = dict(payload)
        for name in (
            "attention_head_count_kv_by_layer",
            "recurrent_layers",
            "sliding_window_pattern",
            "feed_forward_length_by_layer",
        ):
            fields[name] = tuple(fields.get(name) or ())
        layout = GgufServingMetadata(**fields)
    except (TypeError, ValueError):
        return None
    return layout if layout.architecture else None


class GgufHeaderCache:
    """Load-once, save-once disk cache of immutable revision-pinned GGUF headers."""

    def __init__(self, entries: dict[str, GgufServingMetadata] | None = None) -> None:
        self._entries: dict[str, GgufServingMetadata] = dict(entries or {})
        self._dirty = False
        self._lock = Lock()

    @staticmethod
    def _key(repo_id: str, revision: str, filename: str) -> str:
        return f"{repo_id}\0{revision}\0{filename}"

    @classmethod
    def load(cls, *, path: Path | None = None) -> GgufHeaderCache:
        target = path or _gguf_header_cache_path()
        try:
            envelope = json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            return cls()
        if (
            not isinstance(envelope, dict)
            or envelope.get("schema_version") != _GGUF_HEADER_CACHE_SCHEMA_VERSION
        ):
            return cls()
        raw_entries = envelope.get("entries")
        if not isinstance(raw_entries, dict):
            return cls()
        entries: dict[str, GgufServingMetadata] = {}
        for key, raw in raw_entries.items():
            if not isinstance(key, str):
                continue
            layout = _serving_metadata_from_dict(raw)
            if layout is not None:
                entries[key] = layout
        return cls(entries)

    def get(self, repo_id: str, revision: str, filename: str) -> GgufServingMetadata | None:
        key = self._key(repo_id, revision, filename)
        with self._lock:
            return self._entries.get(key)

    def record(
        self,
        repo_id: str,
        revision: str,
        filename: str,
        layout: GgufServingMetadata,
        *,
        mark_dirty: bool = True,
    ) -> None:
        if not layout.architecture or not re.fullmatch(r"[0-9a-f]{40}", revision):
            return
        key = self._key(repo_id, revision, filename)
        with self._lock:
            existing = self._entries.get(key)
            if existing == layout:
                return
            self._entries[key] = layout
            if len(self._entries) > _GGUF_HEADER_CACHE_MAX_ENTRIES:
                excess = len(self._entries) - _GGUF_HEADER_CACHE_MAX_ENTRIES
                for stale_key in list(self._entries)[:excess]:
                    self._entries.pop(stale_key, None)
            if mark_dirty:
                self._dirty = True

    def save(self, *, path: Path | None = None) -> None:
        with self._lock:
            if not self._dirty:
                return
            snapshot = dict(self._entries)
        if len(snapshot) > _GGUF_HEADER_CACHE_MAX_ENTRIES:
            trimmed_keys = list(snapshot)[- _GGUF_HEADER_CACHE_MAX_ENTRIES :]
            snapshot = {key: snapshot[key] for key in trimmed_keys}
        envelope = {
            "schema_version": _GGUF_HEADER_CACHE_SCHEMA_VERSION,
            "entries": {key: asdict(layout) for key, layout in snapshot.items()},
        }
        target = path or _gguf_header_cache_path()
        temporary_path: Path | None = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=target.parent, delete=False
            ) as output:
                temporary_path = Path(output.name)
                json.dump(envelope, output)
            temporary_path.replace(target)
            with self._lock:
                self._dirty = False
        except Exception as exc:
            log_debug(f"Could not persist GGUF header cache: {exc}")
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass


@contextmanager
def active_gguf_header_cache(cache: GgufHeaderCache) -> Generator[None, None, None]:
    """Bind a persistent GGUF header cache for the duration of a catalog build."""
    token = _ACTIVE_GGUF_HEADER_CACHE.set(cache)
    try:
        yield
    finally:
        _ACTIVE_GGUF_HEADER_CACHE.reset(token)


def _pinned_gguf_header(repo_id: str, revision: str, filename: str) -> GgufServingMetadata:
    key = (repo_id, revision, filename)
    disk_cache = _ACTIVE_GGUF_HEADER_CACHE.get()
    with _PINNED_HEADER_LOCK:
        cached = _PINNED_HEADER_CACHE.get(key)
        if cached is not None:
            return cached
        if disk_cache is not None:
            disk_hit = disk_cache.get(repo_id, revision, filename)
            if disk_hit is not None:
                _PINNED_HEADER_CACHE[key] = disk_hit
                return disk_hit
        flight = _PINNED_HEADER_FLIGHTS.get(key)
        if flight is None:
            flight = (Event(), [])
            _PINNED_HEADER_FLIGHTS[key] = flight
            is_leader = True
        else:
            is_leader = False

    if not is_leader:
        flight[0].wait()
        outcome = flight[1]
        if outcome and isinstance(outcome[0], GgufServingMetadata):
            return outcome[0]
        if outcome and isinstance(outcome[0], BaseException):
            raise outcome[0]
        raise HubLookupIncomplete(f"Could not read the serving layout of {filename}.")

    try:
        layout = fetch_gguf_serving_metadata(repo_id, None, revision=revision, filename=filename, strict=True)
        if layout is None or not layout.architecture:
            raise HubLookupIncomplete(f"Could not read the serving layout of {filename}.")
    except BaseException as exc:
        flight[1].append(exc)
        raise
    else:
        flight[1].append(layout)
        if disk_cache is not None:
            disk_cache.record(repo_id, revision, filename, layout)
        with _PINNED_HEADER_LOCK:
            if len(_PINNED_HEADER_CACHE) >= _PINNED_HEADER_MEMORY_MAX_ENTRIES:
                _PINNED_HEADER_CACHE.pop(next(iter(_PINNED_HEADER_CACHE)))
            _PINNED_HEADER_CACHE[key] = layout
        return layout
    finally:
        with _PINNED_HEADER_LOCK:
            _PINNED_HEADER_FLIGHTS.pop(key, None)
        flight[0].set()


def _clear_pinned_gguf_header_cache() -> None:
    with _PINNED_HEADER_LOCK:
        _PINNED_HEADER_CACHE.clear()


_pinned_gguf_header.cache_clear = _clear_pinned_gguf_header_cache  # type: ignore[attr-defined]


def _fetch_candidates(mode: ModelRankMode, limit: int, target: ModelDiscoveryTarget) -> list[ModelCandidate]:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required for model discovery. Install with: pip install huggingface_hub"
        ) from exc

    sort = _SORT_BY_MODE.get(mode, "downloads")
    api = HfApi()
    candidates_by_id: dict[str, ModelCandidate] = {}
    best_rank: dict[str, int] = {}
    for task in ("text-generation", "image-text-to-text"):
        rows = api.list_models(
            filter=[task, "gguf"] if target == "llamacpp" else task,
            sort=sort, limit=limit * 3, full=True,
        )
        rank = 0
        for row in rows:
            candidate = _normalize_candidate(row, target=target)
            if candidate is None:
                continue
            # Each task list arrives already ranked, so keep the better
            # placement instead of letting merge order decide the winner.
            if rank < best_rank.get(candidate.repo_id, rank + 1):
                best_rank[candidate.repo_id] = rank
            candidates_by_id.setdefault(candidate.repo_id, candidate)
            rank += 1
    candidates = list(candidates_by_id.values())
    if mode == "downloads":
        candidates.sort(key=lambda row: row.downloads or 0, reverse=True)
    else:
        candidates.sort(key=lambda row: best_rank[row.repo_id])
    return candidates[:limit]


def _normalize_candidate(row: Any, target: ModelDiscoveryTarget = "vllm") -> ModelCandidate | None:
    repo_id = str(_first_non_empty(getattr(row, "id", None), getattr(row, "modelId", None))).strip()
    if not repo_id:
        return None

    pipeline_tag = optional_str(getattr(row, "pipeline_tag", None))
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
    if pipeline_tag and pipeline_tag.strip() in {"text-generation", "image-text-to-text"}:
        return True
    if not isinstance(tags, list):
        return False
    lowered = {str(tag).strip().lower() for tag in tags}
    return bool({"text-generation", "image-text-to-text"} & lowered)


def _has_tag(tags: Any, expected: str) -> bool:
    if not isinstance(tags, list):
        return False
    lowered = {str(tag).strip().lower() for tag in tags}
    return expected in lowered


def _gguf_filenames(siblings: Any) -> list[str]:
    if not isinstance(siblings, list):
        return []
    filenames = [str(getattr(sibling, "rfilename", "")).strip() for sibling in siblings]
    return [filename for filename in filenames if filename.casefold().endswith(".gguf")]


def _extract_gguf_quantizations(siblings: Any) -> list[str]:
    detected: set[str] = set()
    for filename in _gguf_filenames(siblings):
        if not is_target_gguf(filename):
            continue
        for match in _QUANT_PATTERN.findall(filename):
            detected.add(_normalize_quant_label(match))
    return sorted(detected, key=_quant_sort_key)



def _extract_gguf_architecture(gguf_payload: Any) -> str | None:
    """Read the GGUF ``general.architecture`` value exposed by the Hub API."""

    if not isinstance(gguf_payload, dict):
        return None
    candidates = [
        gguf_payload.get("architecture"),
        gguf_payload.get("general.architecture"),
    ]
    metadata = gguf_payload.get("metadata")
    if isinstance(metadata, dict):
        candidates.extend(
            [
                metadata.get("architecture"),
                metadata.get("general.architecture"),
            ]
        )
    for candidate in candidates:
        architecture = str(candidate or "").strip().casefold()
        if architecture:
            return architecture
    return None


def _extract_gguf_positive_int(gguf_payload: Any, *keys: str) -> int | None:
    """Read a positive architecture scalar from the Hub GGUF payload.

    Hugging Face has exposed GGUF metadata in both a flat dotted-key shape and
    nested ``metadata`` objects. Matching suffixes keeps this compatible with
    architecture-prefixed keys such as ``qwen3.attention.head_count_kv``.
    """

    expected = {key.strip().casefold() for key in keys if key.strip()}
    matches: list[int] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for raw_key, value in node.items():
                key = str(raw_key).strip().casefold()
                if key in expected or any(key.endswith(f".{item}") for item in expected):
                    try:
                        numeric = int(value)
                    except (TypeError, ValueError):
                        numeric = 0
                    if numeric > 0 and not isinstance(value, bool):
                        matches.append(numeric)
                if isinstance(value, (dict, list, tuple)):
                    _walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                _walk(value)

    _walk(gguf_payload)
    return max(matches) if matches else None


def _extract_gguf_vram_by_quant(gguf_payload: Any) -> dict[str, float]:
    if gguf_payload is None:
        return {}

    vram_gb_by_quant: dict[str, float] = {}

    def _record(quant: str, value: float) -> None:
        upper_quant = _normalize_quant_label(quant)
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
        return {_normalize_quant_label(match) for match in _QUANT_PATTERN.findall(text)}

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
                if _is_memory_hint(key_text) or isinstance(value, str) and _contains_memory_unit(value):
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
            label = _normalize_quant_label(variant.get("label"))
            if not label or label.startswith("DSPARK "):
                continue
            if label not in seen:
                seen.add(label)
                quantizations.append(label)
            parsed_size = _parse_model_file_size_to_gb(variant.get("size"))
            if parsed_size is not None and parsed_size > 0:
                current = vram_gb_by_quant.get(label)
                if current is None or parsed_size > current:
                    vram_gb_by_quant[label] = parsed_size

    return (quantizations, vram_gb_by_quant)


def _fetch_gguf_quantization_data_from_model_page(repo_id: str, timeout: float = 10.0) -> Any:
    """Read the model page's quantization table, or say why it could not be read.

    "We could not look" and "there is nothing there" are different facts, and
    only the second is a reason to drop a model. This is the only source of
    per-quant weight sizes -- the Hub API lists the GGUF files but not their
    sizes -- so a swallowed failure here surfaced downstream as "no GGUF
    quantization has a usable weight-size estimate", a claim about the model
    rather than about the request that failed. One throttled page took a
    top-ranked model out of Fast Deploy with no trace in the log.
    """

    try:
        import requests
    except Exception as exc:  # pragma: no cover - requests is a hard dependency
        raise HubModelPageUnavailable(
            repo_id, "requests is required to read model weight sizes"
        ) from exc

    url = f"https://huggingface.co/{repo_id}"
    try:
        response = requests.get(url, timeout=min(timeout, _HF_REQUEST_TIMEOUT_SECONDS), headers=_HF_PAGE_HEADERS)
    except Exception as exc:
        if is_shutting_down():
            return None
        log_debug(f"Could not read the model page for {repo_id}: {exc}")
        raise HubModelPageUnavailable(repo_id, str(exc)) from exc
    if response.status_code >= 400:
        log_debug(
            f"Could not read the model page for {repo_id}: HTTP {response.status_code}"
        )
        raise HubModelPageUnavailable(
            repo_id,
            f"HTTP {response.status_code} from huggingface.co",
            response=response,
        )
    hardware_data = _extract_hardware_compatibility_quantization_data_from_model_page_html(response.text)
    if hardware_data is not None:
        return hardware_data
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


def _extract_hardware_compatibility_quantization_data_from_model_page_html(page_html: str) -> Any:
    """Parse the visible HF Hardware compatibility quant rows into quant metadata."""
    lines = _html_to_text_lines(page_html)
    variants: list[dict[str, Any]] = []
    seen: set[str] = set()

    for index, line in enumerate(lines):
        if line.strip().casefold() != "hardware compatibility":
            continue
        section: list[str] = []
        for section_line in lines[index + 1 : index + 160]:
            normalized = section_line.strip().casefold()
            if normalized.startswith(("inference providers", "model tree for", "collection including")):
                break
            section.append(section_line)

        for quant, required_vram_gb in _extract_quant_vram_pairs_from_hardware_section(section):
            normalized_quant = _normalize_quant_label(quant)
            if not normalized_quant or normalized_quant in seen:
                continue
            seen.add(normalized_quant)
            variants.append(
                {
                    "label": normalized_quant,
                    "size": int(round(required_vram_gb * 1_000_000_000)),
                }
            )

    if not variants:
        return None
    return {"variantsByQuantizationLevels": {"0": variants}}


def _html_to_text_lines(page_html: str) -> list[str]:
    with_breaks = _HTML_BLOCK_BREAK_RE.sub("\n", page_html)
    text = html.unescape(_HTML_TAG_RE.sub("\n", with_breaks))
    lines: list[str] = []
    for line in text.splitlines():
        normalized = re.sub(r"\s+", " ", line).strip()
        if normalized:
            lines.append(normalized)
    return lines


def _extract_quant_vram_pairs_from_hardware_section(section_lines: list[str]) -> list[tuple[str, float]]:
    section_text = " ".join(section_lines)
    quant_matches = list(_HARDWARE_QUANT_LABEL_RE.finditer(section_text))
    memory_matches = list(_MEMORY_SEARCH_PATTERN.finditer(section_text))
    pairs: list[tuple[str, float]] = []

    for index, quant_match in enumerate(quant_matches):
        next_quant_start = quant_matches[index + 1].start() if index + 1 < len(quant_matches) else len(section_text)
        memory_match = next(
            (
                match
                for match in memory_matches
                if quant_match.end() <= match.start() < next_quant_start
            ),
            None,
        )
        if memory_match is None:
            continue
        required_vram_gb = _parse_memory_to_gb(memory_match.group(0))
        if required_vram_gb is None:
            continue
        pairs.append((_normalize_quant_label(quant_match.group(1)), required_vram_gb))
    return pairs


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


def _parse_model_file_size_to_gb(value: Any) -> float | None:
    """Parse HF model-page ``size`` fields, whose numeric values are bytes."""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number / 1_000_000_000.0 if number > 0 else None
    return _parse_memory_to_gb(value)


def _quant_sort_key(quant: str) -> tuple[int, int, str]:
    upper = _normalize_quant_label(quant)
    if upper in _PREFERRED_QUANT_ORDER:
        return (0, _PREFERRED_QUANT_ORDER.index(upper), upper)
    return (1, 0, upper)


def _normalize_quant_label(value: Any) -> str:
    upper = str(value or "").strip().upper()
    if upper.startswith("UD_"):
        return f"UD-{upper[3:]}"
    return upper


# Bytes per element for the dtype names Hugging Face reports in a
# repository's safetensors header.
_SAFETENSORS_DTYPE_BYTES: dict[str, float] = {
    "F64": 8.0, "I64": 8.0, "U64": 8.0,
    "F32": 4.0, "I32": 4.0, "U32": 4.0,
    "BF16": 2.0, "F16": 2.0, "I16": 2.0, "U16": 2.0,
    "F8_E4M3": 1.0, "F8_E5M2": 1.0, "F8": 1.0,
    "I8": 1.0, "U8": 1.0, "BOOL": 1.0,
    "F4": 0.5, "I4": 0.5, "U4": 0.5,
}
_DEFAULT_WEIGHT_DTYPE_BYTES = 2.0


def _safetensors_field(payload: Any, name: str) -> Any:
    """Read one field from Hugging Face's safetensors info, dict or object."""
    if isinstance(payload, dict):
        return payload.get(name)
    return getattr(payload, name, None)


def _safetensors_weight_bytes(payload: Any) -> float | None:
    """Return the weight bytes a repository's safetensors header implies.

    ``SafeTensorsInfo.total`` is a **parameter count**, not a byte count:
    reading it as bytes sized every bf16 model at half its weights, which is
    the direction that does not fit. The per-dtype breakdown beside it is what
    converts to bytes, and it is also what makes a mixed-precision checkpoint
    (fp8 experts beside a bf16 embedding) come out right.
    """
    if payload is None:
        return None
    parameters = _safetensors_field(payload, "parameters")
    total_bytes = 0.0
    if isinstance(parameters, dict) and parameters:
        for dtype, count in parameters.items():
            try:
                numeric = float(count)
            except (TypeError, ValueError):
                continue
            if numeric <= 0:
                continue
            width = _SAFETENSORS_DTYPE_BYTES.get(
                str(dtype).strip().upper(), _DEFAULT_WEIGHT_DTYPE_BYTES
            )
            total_bytes += numeric * width
        if total_bytes > 0:
            return total_bytes
    try:
        total = float(_safetensors_field(payload, "total"))
    except (TypeError, ValueError):
        return None
    if total <= 0:
        return None
    return total * _DEFAULT_WEIGHT_DTYPE_BYTES


def _model_shape_config(config: Any) -> dict[str, Any] | None:
    """Return the sub-config carrying the decoder shape, or None if absent.

    A multimodal repository keeps the decoder under ``text_config`` and the
    encoder under ``vision_config``; a text-only one states the shape at the
    top level.
    """
    if not isinstance(config, dict):
        return None
    text_config = config.get("text_config")
    if isinstance(text_config, dict) and _extract_int_config(
        text_config, "num_hidden_layers", "n_layer", "num_layers", "n_layers", "decoder_layers"
    ):
        return text_config
    if _extract_int_config(
        config, "num_hidden_layers", "n_layer", "num_layers", "n_layers", "decoder_layers"
    ):
        return config
    return None


def _kv_cache_width(config: Any, hidden_size: int) -> int:
    """Return the per-layer K (or V) width one cached token occupies."""
    kv_heads = _extract_int_config(
        config, "num_key_value_heads", "num_kv_heads", "n_kv_heads"
    )
    query_heads = _extract_int_config(
        config, "num_attention_heads", "n_head", "num_heads"
    )
    head_dim = _extract_int_config(config, "head_dim", "attention_head_dim", "v_head_dim")
    if head_dim is None and query_heads:
        head_dim = hidden_size // query_heads if hidden_size >= query_heads else None
    if kv_heads is None:
        kv_heads = query_heads
    if not kv_heads or not head_dim:
        # Nothing to narrow the estimate with: keep the conservative width.
        return hidden_size
    return kv_heads * head_dim


# Layer kinds that hold a fixed-size recurrent state rather than a cache that
# grows with the context. Everything else is charged the full context, which
# overstates a sliding-window layer and is the direction that still fits.
_RECURRENT_LAYER_TOKENS = ("linear", "mamba", "recurrent", "conv", "ssm")


def _kv_cache_layer_count(config: Any, num_layers: int) -> int:
    """Return how many layers hold a context-sized KV cache."""
    if isinstance(config, dict):
        layer_types = config.get("layer_types")
        if isinstance(layer_types, list) and layer_types:
            cached = sum(
                1
                for entry in layer_types
                if not any(token in str(entry).lower() for token in _RECURRENT_LAYER_TOKENS)
            )
            if cached:
                return cached
        interval = _extract_int_config(config, "full_attention_interval")
        if interval and interval > 1:
            return max(1, num_layers // interval)
    return num_layers


def _extract_parameter_count(repo_id: str, config: Any, safetensors: Any) -> float | None:
    direct = _extract_int_config(config, "num_parameters", "parameter_count", "n_params")
    if direct is not None and direct > 0:
        return float(direct)

    parameters = _safetensors_field(safetensors, "parameters")
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
    total_raw = _safetensors_field(safetensors, "total")
    try:
        total_numeric = float(total_raw) if total_raw is not None else 0.0
    except (TypeError, ValueError):
        total_numeric = 0.0
    if total_numeric > 0:
        return total_numeric

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
        "context_window",
        "context_window_tokens",
        "input_context_window",
        "max_context",
        "max_context_length",
        "max_context_tokens",
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
    if is_shutting_down():
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

    if is_shutting_down():
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
        if is_shutting_down():
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
    try:
        import requests
    except Exception:
        return None
    ref = (revision or "main").strip() or "main"
    url = f"https://huggingface.co/{repo_id}/resolve/{ref}/{filename}"
    try:
        response = requests.get(url, timeout=timeout, headers=_HF_PAGE_HEADERS)
    except Exception:
        if is_shutting_down():
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
    matches: list[tuple[str, str]] = [
        (str(float(experts) * float(size)), unit)
        for experts, size, unit in _EXPERT_PARAMS_PATTERN.findall(text)
    ]
    # The plain pattern also matches the second half of "8x7B", so its reading
    # of that span is dropped in favour of the product above.
    consumed = [span.span() for span in _EXPERT_PARAMS_PATTERN.finditer(text)]
    for match in _PARAMS_PATTERN.finditer(text):
        if any(start <= match.start() < end for start, end in consumed):
            continue
        matches.append((match.group(1), match.group(2)))
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
    if requested is not None:
        requested_context = max(1, int(requested))
        if model_max is not None and model_max > 0:
            return min(requested_context, int(model_max))
        return requested_context
    if model_max is not None and model_max > 0:
        return int(model_max)
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
