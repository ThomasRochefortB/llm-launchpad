"""Shared helpers for presenting OpenAI-compatible connection details."""

from __future__ import annotations

from ..core.backend import ModalBackend
from ..core.naming import default_llamacpp_served_model_name, default_served_model_name
from ..protocol.enums import BackendType, ComputeProvider
from ..protocol.models import EndpointInfo


def resolve_openai_base_url(row: EndpointInfo, username: str = "") -> tuple[str | None, bool]:
    """Return the OpenAI-compatible base URL for an endpoint and whether it was derived."""
    raw_url = (row.web_url or "").strip()
    if raw_url:
        base_root = raw_url.rstrip("/")
        return (base_root if base_root.endswith("/v1") else f"{base_root}/v1"), False
    if row.provider != ComputeProvider.MODAL or not username.strip() or not row.name.strip():
        return None, False
    derived = ModalBackend.default_server_url(username.strip(), app_name=row.name.strip()).rstrip("/")
    return (derived if derived.endswith("/v1") else f"{derived}/v1"), True


def endpoint_model_summary(row: EndpointInfo) -> tuple[str | None, str | None]:
    """Return the served model ID and human display name for an endpoint."""
    explicit_display_name = (row.display_name or "").strip() or None

    if row.backend == BackendType.VLLM:
        model_id = (row.served_model_name or "").strip()
        if not model_id and (row.model_name or "").strip():
            model_id = default_served_model_name(row.model_name)
        display_name = explicit_display_name or (row.model_name or row.served_model_name or "").strip() or None
        return model_id or None, display_name

    if row.backend == BackendType.LLAMACPP:
        model_id = (row.served_model_name or "").strip()
        if not model_id and (row.repo_id or "").strip():
            model_id = default_llamacpp_served_model_name(row.repo_id, row.quant)
        if explicit_display_name:
            display_name = explicit_display_name
        else:
            repo = (row.repo_id or "").strip()
            quant = (row.quant or "").strip()
            if repo:
                display_name = f"{repo} ({quant})" if quant else repo
            else:
                display_name = (row.served_model_name or "").strip() or None
        return model_id or None, display_name or None

    return (row.served_model_name or "").strip() or None, explicit_display_name


def endpoint_connection_payload(row: EndpointInfo, username: str = "") -> dict[str, str | None]:
    """Return copyable connection fields for one endpoint."""
    base_url, _derived = resolve_openai_base_url(row, username=username)
    model_id, display_name = endpoint_model_summary(row)
    return {
        "base_url": base_url,
        "model_id": model_id,
        "display_name": display_name,
        "api_key": (row.endpoint_api_key or "").strip() or None,
        "provider": row.provider.value,
        "state": (row.state or "").strip().lower(),
    }
