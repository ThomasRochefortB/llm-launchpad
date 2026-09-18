"""Shared helpers for presenting OpenAI-compatible connection details."""

from __future__ import annotations

import json
import shlex

from ..core.backend import ModalBackend
from ..core.deployment_states import is_terminal_deployment_state
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
    # Deriving a URL for an app that is over invents a copyable endpoint that
    # only ever hangs. A row still starting has no URL yet but will get one, so
    # it keeps the derived address.
    if is_terminal_deployment_state(row.state):
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


def connection_curl_example(payload: dict[str, str | None]) -> str | None:
    """Build a copyable curl example from a connection payload.

    Uses JSON serialization plus shell quoting so model IDs with quotes or
    other special characters stay valid. Returns None when the base URL or
    model ID is unavailable.
    """
    base_url = (payload.get("base_url") or "").strip()
    model_id = (payload.get("model_id") or "").strip()
    if not base_url or not model_id:
        return None
    body = json.dumps({"model": model_id, "messages": [{"role": "user", "content": "Hello"}]})
    parts = ["curl", "-sS", f"{base_url.rstrip('/')}/chat/completions"]
    api_key = (payload.get("api_key") or "").strip()
    if api_key:
        parts += ["-H", f"Authorization: Bearer {api_key}"]
    parts += ["-H", "Content-Type: application/json", "-d", body]
    return " ".join(shlex.quote(part) for part in parts)


def connection_json_example(payload: dict[str, str | None]) -> str | None:
    """Build a copyable JSON client config from a connection payload."""
    base_url = (payload.get("base_url") or "").strip()
    model_id = (payload.get("model_id") or "").strip()
    if not base_url or not model_id:
        return None
    config: dict[str, str] = {"base_url": base_url, "model": model_id}
    api_key = (payload.get("api_key") or "").strip()
    if api_key:
        config["api_key"] = api_key
    return json.dumps(config, indent=2)
