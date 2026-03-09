"""Instance/app naming helpers for launchpad deployments."""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from coolname import generate_slug

from ..protocol.enums import BackendType


_LEGACY_APP_NAMES = {
    BackendType.LLAMACPP: "llamacpp-server",
    BackendType.VLLM: "vllm-server",
}

_BACKEND_PREFIX = {
    BackendType.LLAMACPP: "llamacpp",
    BackendType.VLLM: "vllm",
}


def legacy_app_name(backend: BackendType) -> str:
    """Return the historical single-instance app name for *backend*."""
    return _LEGACY_APP_NAMES[backend]


def slugify_instance_name(raw: str, default: str = "default") -> str:
    """Convert an arbitrary identifier into a safe, deterministic slug."""
    normalized = unicodedata.normalize("NFKD", raw or "")
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    ascii_text = ascii_text.replace("/", "-").replace("_", "-")
    slug = re.sub(r"[^a-z0-9-]+", "-", ascii_text)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or default


def auto_instance_name_for_backend(backend: BackendType, model_hint: Optional[str]) -> str:
    """Build a default instance name based on backend model context."""
    if backend == BackendType.VLLM:
        return slugify_instance_name(model_hint or "default")
    # llama.cpp can use preset name or repo id as a hint; fallback still valid.
    return slugify_instance_name(model_hint or "default")


def default_served_model_name(model_name: Optional[str], default: str = "llm") -> str:
    """Return the default served model alias for a vLLM model id."""
    candidate = (model_name or "").strip()
    if not candidate:
        return default
    tail = candidate.rsplit("/", 1)[-1].strip()
    return tail or default


def default_llamacpp_served_model_name(
    repo_id: Optional[str],
    quant: Optional[str] = None,
    default: str = "default",
) -> str:
    """Return a friendly llama.cpp OpenAI model id derived from repo + quant."""
    alias = default_served_model_name(repo_id, default=default)
    quant_text = (quant or "").strip()
    if not quant_text:
        return alias
    if quant_text.casefold() in alias.casefold():
        return alias
    return f"{alias}-{quant_text}"


def build_app_name(backend: BackendType, instance_name: Optional[str]) -> str:
    """Compose a launchpad app name from backend + instance name."""
    if not instance_name:
        return legacy_app_name(backend)
    return f"{_BACKEND_PREFIX[backend]}-{slugify_instance_name(instance_name)}"


def infer_backend_from_app_name(app_name: str) -> Optional[BackendType]:
    """Infer backend type from legacy or prefixed app names."""
    if app_name == _LEGACY_APP_NAMES[BackendType.VLLM] or app_name.startswith("vllm-"):
        return BackendType.VLLM
    if app_name == _LEGACY_APP_NAMES[BackendType.LLAMACPP] or app_name.startswith("llamacpp-"):
        return BackendType.LLAMACPP
    return None


def infer_instance_from_app_name(app_name: str, backend: Optional[BackendType]) -> Optional[str]:
    """Infer instance name from app name when possible."""
    if backend is None:
        return None
    legacy = legacy_app_name(backend)
    if app_name == legacy:
        return "default"
    prefix = _BACKEND_PREFIX[backend] + "-"
    if app_name.startswith(prefix):
        return app_name[len(prefix) :] or "default"
    return None


def random_function_slug() -> str:
    """Generate a random, URL-safe two-word slug for Modal function names."""
    return slugify_instance_name(generate_slug(2))


def modal_function_name(base_name: str, function_slug: Optional[str]) -> str:
    """Return a Modal function name with optional deployment slug suffix."""
    slug = slugify_instance_name(function_slug or "", default="")
    if not slug:
        return base_name
    return f"{base_name}-{slug}"
