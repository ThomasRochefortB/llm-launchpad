"""Local metadata for provider endpoints that cannot be reconstructed remotely."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

from ..protocol.enums import BackendType, ComputeProvider
from ..protocol.models import DeploymentConfig, EndpointInfo
from .config import SETTINGS_DIR
from .opencode import build_openai_connection_payload


CONNECTIONS_PATH = SETTINGS_DIR / "deployment_connection_summaries.json"


def load_connection_entries(path: Path = CONNECTIONS_PATH) -> dict[str, dict[str, Any]]:
    """Load valid connection entries, returning an empty mapping on corruption."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, dict):
        return {}
    return {
        str(key): value
        for key, value in entries.items()
        if str(key).strip() and isinstance(value, dict)
    }


def _write(entries: dict[str, dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"entries": entries}, indent=2), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def save_connection(
    config: DeploymentConfig,
    endpoint: EndpointInfo,
    path: Path = CONNECTIONS_PATH,
) -> None:
    """Persist endpoint URL/model/credential metadata for a deployment."""
    app_name = (endpoint.name or config.app_name or "").strip()
    server_url = (endpoint.web_url or "").strip()
    if not app_name or not server_url:
        return
    payload = build_openai_connection_payload(config, server_url)
    entries = load_connection_entries(path)
    entries[app_name] = {
        "provider": config.provider.value,
        "backend": config.backend.value,
        "resource_id": endpoint.app_id,
        "instance_name": config.instance_name or endpoint.instance_name or "",
        "base_url": payload["base_url"],
        "model_id": payload["model_id"],
        "display_name": payload["display_name"],
        "api_key": config.endpoint_api_key or endpoint.endpoint_api_key or "",
        "cached_at_epoch": time.time(),
    }
    _write(entries, path)


def merge_connections(
    rows: list[EndpointInfo],
    path: Path = CONNECTIONS_PATH,
) -> list[EndpointInfo]:
    """Hydrate provider rows with locally persisted endpoint metadata."""
    entries = load_connection_entries(path)
    for row in rows:
        cached = entries.get((row.name or "").strip())
        if not cached:
            continue
        row.web_url = row.web_url or str(cached.get("base_url") or "").removesuffix("/v1")
        row.served_model_name = row.served_model_name or str(cached.get("model_id") or "") or None
        row.display_name = row.display_name or str(cached.get("display_name") or "") or None
        row.endpoint_api_key = row.endpoint_api_key or str(cached.get("api_key") or "") or None
        row.app_id = row.app_id or str(cached.get("resource_id") or "")
        row.instance_name = row.instance_name or str(cached.get("instance_name") or "") or None
        provider = str(cached.get("provider") or "")
        if provider in {item.value for item in ComputeProvider}:
            row.provider = ComputeProvider(provider)
        backend = str(cached.get("backend") or "")
        if backend in {item.value for item in BackendType}:
            row.backend = BackendType(backend)
    return rows


def remove_connection(app_name: str, path: Path = CONNECTIONS_PATH) -> None:
    """Remove local metadata for a terminated deployment."""
    entries = load_connection_entries(path)
    if entries.pop(app_name, None) is not None:
        _write(entries, path)
