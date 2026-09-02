"""Discover model reasoning controls from revision-pinned Hugging Face sources."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import threading
import time
from typing import Any

from ..protocol.enums import BackendType
from ..protocol.models import DeploymentConfig, ReasoningCapabilities
from .shutdown import is_shutting_down


_SUPPORTED_REQUEST_OPTION_PATHS = {
    "chat_template_kwargs.reasoning_effort",
}
_REQUEST_OPTION_PATH = "chat_template_kwargs.reasoning_effort"
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_REPO_ID_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$"
)
_EFFORT_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_REASONING_EFFORT_IDENTIFIER = (
    r"(?:reasoning_effort|[A-Za-z_]\w*reasoning_effort\w*)"
)
_QUOTED_TOKEN_RE = re.compile(
    r"(?P<quote>['\"])(?P<value>[A-Za-z][A-Za-z0-9_-]{0,31})(?P=quote)"
)
_MEMBERSHIP_RE = re.compile(
    rf"\b{_REASONING_EFFORT_IDENTIFIER}\s+(?:not\s+)?in\s*"
    r"[([{](?P<values>[^\])}]{1,500})[\])}]",
    flags=re.IGNORECASE,
)
_EQUALITY_RE = re.compile(
    rf"\b{_REASONING_EFFORT_IDENTIFIER}\s*==\s*"
    r"(?P<quote>['\"])(?P<value>[A-Za-z][A-Za-z0-9_-]{0,31})(?P=quote)",
    flags=re.IGNORECASE,
)
_REVERSED_EQUALITY_RE = re.compile(
    r"(?P<quote>['\"])(?P<value>[A-Za-z][A-Za-z0-9_-]{0,31})(?P=quote)"
    rf"\s*==\s*\b{_REASONING_EFFORT_IDENTIFIER}",
    flags=re.IGNORECASE,
)
_VARIABLE_EFFORT_RE = re.compile(
    r"\breasoning_effort_(?P<value>[A-Za-z][A-Za-z0-9_-]{0,31})\s*=",
    flags=re.IGNORECASE,
)
_JINJA_DEFAULT_RE = re.compile(
    r"\breasoning_effort\s*\|\s*default\(\s*"
    r"(?P<quote>['\"])(?P<value>[A-Za-z][A-Za-z0-9_-]{0,31})(?P=quote)",
    flags=re.IGNORECASE,
)
_CONSTANT_DEFAULT_RE = re.compile(
    r"\bDEFAULT_REASONING_EFFORT\b(?:\s*:[^=\n]+)?\s*=\s*"
    r"(?P<quote>['\"])(?P<value>[A-Za-z][A-Za-z0-9_-]{0,31})(?P=quote)",
    flags=re.IGNORECASE,
)
_TERNARY_DEFAULT_RE = re.compile(
    rf"(?:set\s+)?{_REASONING_EFFORT_IDENTIFIER}\s*=\s*.{{0,400}}?"
    r"\belse\s+(?P<quote>['\"])(?P<value>[A-Za-z][A-Za-z0-9_-]{0,31})"
    r"(?P=quote)",
    flags=re.IGNORECASE | re.DOTALL,
)
_DOCUMENTED_EFFORTS_RE = re.compile(
    r"reasoning_effort.{0,180}?(?:one\s+of|supported(?:\s+types?)?(?:\s+are)?)"
    r"(?P<values>.{0,240})",
    flags=re.IGNORECASE | re.DOTALL,
)
_DOCUMENTED_DEFAULT_RE = re.compile(
    r"(?P<quote>['\"])(?P<value>[A-Za-z][A-Za-z0-9_-]{0,31})(?P=quote)"
    r"\s+(?:is\s+the\s+)?default\b",
    flags=re.IGNORECASE,
)
_MAX_REASONING_SOURCE_BYTES = 1_000_000
_MAX_REASONING_SOURCE_FILES = 12
_DISCOVERY_CACHE_TTL_SECONDS = 300.0
_HF_REQUEST_TIMEOUT_SECONDS = 10.0
_HF_ETAG_TIMEOUT_SECONDS = 10.0
_EFFORT_ORDER = {
    "none": 0,
    "minimal": 1,
    "low": 2,
    "medium": 3,
    "high": 4,
    "xhigh": 5,
    "max": 6,
}
_RESERVED_EFFORT_TOKENS = {
    "default",
    "effort",
    "false",
    "null",
    "prompts",
    "reasoning",
    "true",
}
_DISCOVERY_CACHE: dict[
    tuple[BackendType, str, str],
    tuple[float, ReasoningCapabilities | None],
] = {}
_DISCOVERY_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class _ReasoningSource:
    repo_id: str
    revision: str
    path: str
    text: str


@dataclass(frozen=True)
class _RepositoryInspection:
    repo_id: str
    revision: str
    sources: tuple[_ReasoningSource, ...]
    quantized_base_models: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ParsedReasoning:
    efforts: tuple[str, ...]
    default_effort: str | None
    source_path: str
    enable_thinking: bool
    interleaved_field: str | None

    @property
    def complete(self) -> bool:
        return (
            len(self.efforts) >= 2
            and self.default_effort is not None
            and self.default_effort in self.efforts
        )


def discover_reasoning_capabilities(
    backend: BackendType,
    repo_id: str,
    revision: str | None = None,
) -> ReasoningCapabilities | None:
    """Inspect a selected Hub revision without executing repository code."""

    normalized_repo = repo_id.strip()
    normalized_revision = (revision or "").strip() or None
    if is_shutting_down() or not _REPO_ID_RE.fullmatch(normalized_repo):
        return None

    cache_key = (backend, normalized_repo, normalized_revision or "")
    now = time.monotonic()
    with _DISCOVERY_CACHE_LOCK:
        cached = _DISCOVERY_CACHE.get(cache_key)
        if cached is not None and now - cached[0] < _DISCOVERY_CACHE_TTL_SECONDS:
            return cached[1]

    capabilities = _discover_reasoning_capabilities(
        backend,
        normalized_repo,
        normalized_revision,
    )
    with _DISCOVERY_CACHE_LOCK:
        _DISCOVERY_CACHE[cache_key] = (now, capabilities)
    return capabilities


def clear_reasoning_discovery_cache() -> None:
    """Clear selection-time discovery results, primarily for explicit refreshes."""

    with _DISCOVERY_CACHE_LOCK:
        _DISCOVERY_CACHE.clear()


def _discover_reasoning_capabilities(
    backend: BackendType,
    normalized_repo: str,
    normalized_revision: str | None,
) -> ReasoningCapabilities | None:
    """Resolve an uncached capability snapshot for one selected revision."""

    selected = _inspect_hf_repository(normalized_repo, normalized_revision)
    selected_reasoning = _parse_repository_reasoning(selected)
    evidence = selected_reasoning if selected_reasoning.complete else None
    evidence_repo = selected

    if evidence is None:
        selected_efforts = set(selected_reasoning.efforts)
        for base_repo_id in selected.quantized_base_models:
            try:
                base = _inspect_hf_repository(base_repo_id, None)
            except Exception:
                continue
            parsed_base = _parse_repository_reasoning(base)
            if not parsed_base.complete:
                continue
            if selected_efforts and not selected_efforts.issubset(parsed_base.efforts):
                continue
            evidence = parsed_base
            evidence_repo = base
            break

    if evidence is None or evidence.default_effort is None:
        return None

    enable_thinking = evidence.enable_thinking or selected_reasoning.enable_thinking
    interleaved_field = (
        selected_reasoning.interleaved_field or evidence.interleaved_field
    )
    profile_seed = "\n".join(
        (
            backend.value,
            selected.repo_id,
            selected.revision,
            evidence_repo.repo_id,
            evidence_repo.revision,
            evidence.source_path,
            ",".join(evidence.efforts),
            evidence.default_effort,
        )
    )
    profile_hash = hashlib.sha256(profile_seed.encode("utf-8")).hexdigest()[:16]
    return ReasoningCapabilities(
        profile_id=f"hf-{profile_hash}",
        canonical_model_id=selected.repo_id,
        model_revision=selected.revision,
        efforts=evidence.efforts,
        default_effort=evidence.default_effort,
        source_repo=evidence_repo.repo_id,
        source_revision=evidence_repo.revision,
        source_path=evidence.source_path,
        request_option_path=_REQUEST_OPTION_PATH,
        enable_thinking=enable_thinking,
        interleaved_field=interleaved_field,
    )


def discover_selected_model_reasoning(
    config: DeploymentConfig,
) -> ReasoningCapabilities | None:
    """Inspect the Hugging Face model selected in a deployment config."""

    if config.backend == BackendType.VLLM:
        repo_id = (config.model_name or "").strip()
        revision = config.model_revision
    else:
        repo_id = (config.repo_id or "").strip()
        revision = config.revision
    if not repo_id:
        return None
    return discover_reasoning_capabilities(config.backend, repo_id, revision)


def reasoning_request_options(
    capabilities: ReasoningCapabilities,
    effort: str,
) -> dict[str, Any]:
    """Build the OpenCode variant options for one repository-verified effort."""

    if effort not in capabilities.efforts:
        raise ValueError(
            f"Unsupported reasoning effort {effort!r} for "
            f"{capabilities.canonical_model_id}"
        )
    if capabilities.request_option_path == _REQUEST_OPTION_PATH:
        template_options: dict[str, Any] = {"reasoning_effort": effort}
        if capabilities.enable_thinking:
            template_options["enable_thinking"] = True
        return {"chat_template_kwargs": template_options}
    raise ValueError(
        f"Unsupported reasoning option path: {capabilities.request_option_path!r}"
    )


def reasoning_variants(
    capabilities: ReasoningCapabilities,
) -> dict[str, dict[str, Any]]:
    """Build OpenCode's default plus every repository-verified variant."""

    variants = {
        "default": reasoning_request_options(
            capabilities,
            capabilities.default_effort,
        )
    }
    variants.update(
        {
            effort: reasoning_request_options(capabilities, effort)
            for effort in capabilities.efforts
        }
    )
    return variants


def reasoning_capabilities_to_dict(
    capabilities: ReasoningCapabilities | None,
) -> dict[str, Any] | None:
    """Serialize verified capabilities for endpoint/registry persistence."""

    if capabilities is None:
        return None
    return {
        "profile_id": capabilities.profile_id,
        "canonical_model_id": capabilities.canonical_model_id,
        "model_revision": capabilities.model_revision,
        "efforts": list(capabilities.efforts),
        "default_effort": capabilities.default_effort,
        "source_repo": capabilities.source_repo,
        "source_revision": capabilities.source_revision,
        "source_path": capabilities.source_path,
        "request_option_path": capabilities.request_option_path,
        "enable_thinking": capabilities.enable_thinking,
        "interleaved_field": capabilities.interleaved_field,
    }


def reasoning_capabilities_from_dict(
    payload: object,
) -> ReasoningCapabilities | None:
    """Parse a persisted revision-pinned capability snapshot."""

    if not isinstance(payload, dict):
        return None
    try:
        profile_id = _required_string(payload, "profile_id", "capability snapshot")
        canonical_model_id = _required_string(
            payload,
            "canonical_model_id",
            "capability snapshot",
        )
        model_revision = _required_string(
            payload,
            "model_revision",
            "capability snapshot",
        ).casefold()
        efforts = _string_tuple(payload.get("efforts"), "capability snapshot efforts")
        default_effort = _required_string(
            payload,
            "default_effort",
            "capability snapshot",
        )
        source_repo = _required_string(payload, "source_repo", "capability snapshot")
        source_revision = _required_string(
            payload,
            "source_revision",
            "capability snapshot",
        ).casefold()
        source_path = _required_string(payload, "source_path", "capability snapshot")
        request_option_path = _required_string(
            payload,
            "request_option_path",
            "capability snapshot",
        )
        enable_thinking = payload.get("enable_thinking", False)
        interleaved_value = payload.get("interleaved_field")
        interleaved_field = (
            interleaved_value.strip()
            if isinstance(interleaved_value, str) and interleaved_value.strip()
            else None
        )
        if not efforts or len(set(efforts)) != len(efforts):
            return None
        if any(
            not _EFFORT_RE.fullmatch(effort) or effort == "default"
            for effort in efforts
        ):
            return None
        if default_effort not in efforts:
            return None
        if request_option_path not in _SUPPORTED_REQUEST_OPTION_PATHS:
            return None
        if not isinstance(enable_thinking, bool):
            return None
        if not _REVISION_RE.fullmatch(model_revision):
            return None
        if not _REVISION_RE.fullmatch(source_revision):
            return None
    except RuntimeError:
        return None
    return ReasoningCapabilities(
        profile_id=profile_id,
        canonical_model_id=canonical_model_id,
        model_revision=model_revision,
        efforts=efforts,
        default_effort=default_effort,
        source_repo=source_repo,
        source_revision=source_revision,
        source_path=source_path,
        request_option_path=request_option_path,
        enable_thinking=enable_thinking,
        interleaved_field=interleaved_field,
    )


def _inspect_hf_repository(
    repo_id: str,
    revision: str | None,
) -> _RepositoryInspection:
    if is_shutting_down():
        raise RuntimeError("Shutdown requested")
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required for reasoning capability discovery"
        ) from exc

    api = HfApi()
    rich_info = _call_model_info(
        api,
        repo_id,
        revision,
        expand=["baseModels", "cardData", "gguf", "sha", "siblings"],
    )
    file_info = _call_model_info(
        api,
        repo_id,
        revision,
        files_metadata=True,
    )
    resolved_revision = str(
        getattr(rich_info, "sha", None) or getattr(file_info, "sha", None) or ""
    ).strip().casefold()
    if not _REVISION_RE.fullmatch(resolved_revision):
        raise RuntimeError(f"Hugging Face did not return a commit SHA for {repo_id}")

    sources: list[_ReasoningSource] = []
    gguf_payload = getattr(rich_info, "gguf", None)
    if isinstance(gguf_payload, dict):
        chat_template = gguf_payload.get("chat_template")
        if (
            isinstance(chat_template, str)
            and "reasoning_effort" in chat_template.casefold()
            and len(chat_template.encode("utf-8")) <= _MAX_REASONING_SOURCE_BYTES
        ):
            sources.append(
                _ReasoningSource(
                    repo_id=repo_id,
                    revision=resolved_revision,
                    path="gguf.chat_template",
                    text=chat_template,
                )
            )

    candidates: list[tuple[int, str]] = []
    for sibling in getattr(file_info, "siblings", None) or []:
        path = str(getattr(sibling, "rfilename", "") or "").strip()
        size = getattr(sibling, "size", None)
        priority = _reasoning_source_priority(path)
        if priority is None or not isinstance(size, int):
            continue
        if size <= 0 or size > _MAX_REASONING_SOURCE_BYTES:
            continue
        candidates.append((priority, path))

    for _priority, source_path in sorted(candidates)[:_MAX_REASONING_SOURCE_FILES]:
        if is_shutting_down():
            break
        try:
            try:
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=source_path,
                    revision=resolved_revision,
                    etag_timeout=_HF_ETAG_TIMEOUT_SECONDS,
                )
            except TypeError:
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=source_path,
                    revision=resolved_revision,
                )
            path = Path(downloaded)
            if path.stat().st_size > _MAX_REASONING_SOURCE_BYTES:
                continue
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        if "reasoning_effort" not in text.casefold():
            continue
        sources.append(
            _ReasoningSource(
                repo_id=repo_id,
                revision=resolved_revision,
                path=source_path,
                text=text,
            )
        )

    return _RepositoryInspection(
        repo_id=repo_id,
        revision=resolved_revision,
        sources=tuple(sources),
        quantized_base_models=_quantized_base_models(rich_info),
    )


def _call_model_info(
    api: Any,
    repo_id: str,
    revision: str | None,
    *,
    expand: list[str] | None = None,
    files_metadata: bool = False,
) -> Any:
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
        return api.model_info(**kwargs)


def _reasoning_source_priority(path: str) -> int | None:
    normalized = path.strip().casefold()
    filename = normalized.rsplit("/", 1)[-1]
    if filename == "chat_template.jinja":
        return 0
    if filename == "tokenizer_config.json":
        return 1
    if "chat_template" in filename and filename.endswith((".jinja", ".json", ".txt")):
        return 2
    if "encoding" in normalized and filename.endswith((".py", ".jinja", ".json")):
        return 3
    if filename in {"generation_config.json", "config.json"}:
        return 4
    if filename in {"readme.md", "readme.txt"}:
        return 5
    return None


def _quantized_base_models(info: Any) -> tuple[str, ...]:
    base_models = getattr(info, "base_models", None)
    relation = ""
    raw_models: object = None
    if isinstance(base_models, dict):
        relation = str(base_models.get("relation") or "").strip().casefold()
        raw_models = base_models.get("models")

    card_data = getattr(info, "card_data", None)
    if card_data is None:
        card_data = getattr(info, "cardData", None)
    card_payload: dict[str, Any] = {}
    to_dict: Any = None
    if isinstance(card_data, dict):
        card_payload = card_data
    else:
        to_dict = getattr(card_data, "to_dict", None)
    if not card_payload and callable(to_dict):
        try:
            value = to_dict()
            if isinstance(value, dict):
                card_payload = value
        except Exception:
            card_payload = {}
    if not relation:
        relation = str(card_payload.get("base_model_relation") or "").casefold()
    if raw_models is None:
        raw_models = card_payload.get("base_model")
    if relation != "quantized":
        return ()

    if isinstance(raw_models, (str, dict)):
        items = [raw_models]
    elif isinstance(raw_models, list):
        items = raw_models
    else:
        items = []
    result: list[str] = []
    for item in items:
        value = item.get("id") if isinstance(item, dict) else item
        repo_id = str(value or "").strip()
        if _REPO_ID_RE.fullmatch(repo_id) and repo_id not in result:
            result.append(repo_id)
    return tuple(result)


def _aggregate_reasoning_pool(parsed_list: tuple[_ParsedReasoning, ...]) -> _ParsedReasoning:
    """Combine one pool of parsed sources into a single reasoning snapshot.

    A single complete source is preferred so the most authoritative file wins
    without diluting its evidence; otherwise efforts, defaults, and optional
    fields are merged across the pool in the order they were inspected.
    """
    for parsed in parsed_list:
        if parsed.complete:
            return parsed

    efforts: list[str] = []
    default_effort: str | None = None
    source_paths: list[str] = []
    enable_thinking = False
    interleaved_field: str | None = None
    for parsed in parsed_list:
        for effort in parsed.efforts:
            if effort not in efforts:
                efforts.append(effort)
        if default_effort is None and parsed.default_effort is not None:
            default_effort = parsed.default_effort
        if parsed.source_path not in source_paths:
            source_paths.append(parsed.source_path)
        enable_thinking = enable_thinking or parsed.enable_thinking
        interleaved_field = interleaved_field or parsed.interleaved_field
    if default_effort is not None and default_effort not in efforts:
        efforts.append(default_effort)
    return _ParsedReasoning(
        efforts=_ordered_efforts(efforts),
        default_effort=default_effort,
        source_path=",".join(source_paths),
        enable_thinking=enable_thinking,
        interleaved_field=interleaved_field,
    )


def _parse_repository_reasoning(
    inspection: _RepositoryInspection,
) -> _ParsedReasoning:
    parsed_sources = tuple(_parse_reasoning_source(source) for source in inspection.sources)
    all_sources = tuple(
        parsed
        for parsed in parsed_sources
        if parsed.efforts or parsed.default_effort
    )
    code_sources = tuple(
        parsed
        for source, parsed in zip(inspection.sources, parsed_sources)
        if source.path.casefold().rsplit("/", 1)[-1] not in {"readme.md", "readme.txt"}
        and (parsed.efforts or parsed.default_effort)
    )
    if not code_sources:
        return _aggregate_reasoning_pool(all_sources)

    # Code templates and configs are authoritative; only reach for README
    # documentation when those sources cannot produce a complete profile on
    # their own. Aggregating with the README keeps evidence monotonic so a
    # partial code signal never discards a documented default.
    code_result = _aggregate_reasoning_pool(code_sources)
    if code_result.complete:
        return code_result
    return _aggregate_reasoning_pool(all_sources)


def _parse_reasoning_source(source: _ReasoningSource) -> _ParsedReasoning:
    efforts: list[str] = []
    defaults: list[str] = []

    def _record(value: object, target: list[str] = efforts) -> None:
        normalized = _normalize_effort(value)
        if normalized is not None and normalized not in target:
            target.append(normalized)

    text = source.text
    if source.path.casefold().endswith(".py"):
        _extract_python_reasoning(text, efforts, defaults)
    elif source.path.casefold().endswith(".json"):
        _extract_json_reasoning(text, efforts, defaults)
    for match in _MEMBERSHIP_RE.finditer(text):
        for value in _quoted_tokens(match.group("values")):
            _record(value)
    for pattern in (_EQUALITY_RE, _REVERSED_EQUALITY_RE, _VARIABLE_EFFORT_RE):
        for match in pattern.finditer(text):
            _record(match.group("value"))
    for pattern in (
        _JINJA_DEFAULT_RE,
        _CONSTANT_DEFAULT_RE,
        _TERNARY_DEFAULT_RE,
    ):
        for match in pattern.finditer(text):
            _record(match.group("value"), defaults)
    if not efforts:
        for match in _DOCUMENTED_EFFORTS_RE.finditer(text):
            for value in _quoted_tokens(match.group("values")):
                _record(value)
    if not defaults:
        for match in _DOCUMENTED_DEFAULT_RE.finditer(text):
            _record(match.group("value"), defaults)
    default_effort = defaults[0] if defaults else None
    if default_effort is not None:
        _record(default_effort)
    return _ParsedReasoning(
        efforts=_ordered_efforts(efforts),
        default_effort=default_effort,
        source_path=source.path,
        enable_thinking=bool(
            re.search(r"\benable_thinking\b", text, flags=re.IGNORECASE)
        ),
        interleaved_field=(
            "reasoning_content"
            if re.search(r"['\"]reasoning_content['\"]", text)
            else None
        ),
    )


def _extract_python_reasoning(
    text: str,
    efforts: list[str],
    defaults: list[str],
) -> None:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [name for target in node.targets for name in _assignment_names(target)]
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            names = _assignment_names(node.target)
            value = node.value
        else:
            continue
        lowered_names = {name.casefold() for name in names}
        if "default_reasoning_effort" in lowered_names and isinstance(value, ast.Constant):
            normalized = _normalize_effort(value.value)
            if normalized is not None and normalized not in defaults:
                defaults.append(normalized)
        if not any("reasoning_effort" in name for name in lowered_names):
            continue
        if isinstance(value, ast.Dict):
            for key in value.keys:
                if isinstance(key, ast.Constant):
                    normalized = _normalize_effort(key.value)
                    if normalized is not None and normalized not in efforts:
                        efforts.append(normalized)


def _extract_json_reasoning(
    text: str,
    efforts: list[str],
    defaults: list[str],
) -> None:
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return

    def _record(value: object, target: list[str]) -> None:
        normalized = _normalize_effort(value)
        if normalized is not None and normalized not in target:
            target.append(normalized)

    def _walk(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                _walk(item)
            return
        if not isinstance(value, dict):
            return
        for raw_key, child in value.items():
            key = str(raw_key).strip().casefold().replace("-", "_")
            if "reasoning_effort" in key:
                if key.startswith("default_"):
                    _record(child, defaults)
                elif isinstance(child, list):
                    for item in child:
                        _record(item, efforts)
                elif isinstance(child, dict):
                    for child_key, child_value in child.items():
                        normalized_key = str(child_key).strip().casefold()
                        if normalized_key in {"default", "default_effort"}:
                            _record(child_value, defaults)
                        elif normalized_key in {
                            "efforts",
                            "levels",
                            "supported",
                            "supported_values",
                            "values",
                        } and isinstance(child_value, list):
                            for item in child_value:
                                _record(item, efforts)
            _walk(child)

    _walk(payload)


def _assignment_names(node: ast.expr) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [name for item in node.elts for name in _assignment_names(item)]
    return []


def _quoted_tokens(fragment: str) -> tuple[str, ...]:
    return tuple(match.group("value") for match in _QUOTED_TOKEN_RE.finditer(fragment))


def _normalize_effort(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold()
    if (
        not _EFFORT_RE.fullmatch(normalized)
        or normalized in _RESERVED_EFFORT_TOKENS
        or normalized == "default"
    ):
        return None
    return normalized


def _ordered_efforts(values: list[str]) -> tuple[str, ...]:
    source_order = {value: index for index, value in enumerate(values)}
    return tuple(
        sorted(
            source_order,
            key=lambda value: (
                _EFFORT_ORDER.get(value, len(_EFFORT_ORDER)),
                source_order[value],
            ),
        )
    )


def _required_string(
    payload: object,
    key: str,
    context: str,
) -> str:
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} must be a JSON object")
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{context} requires a non-empty {key!r}")
    return value.strip()


def _string_tuple(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise RuntimeError(f"{context} must be an array")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise RuntimeError(f"{context} must contain only non-empty strings")
        result.append(item.strip())
    return tuple(result)
