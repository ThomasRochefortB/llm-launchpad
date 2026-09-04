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
from .coerce import positive_int
from .opencode import build_openai_connection_payload
from .llamacpp_planner import runtime_attestation_from_dict, runtime_attestation_to_dict
from .reasoning_profiles import (
    discover_reasoning_capabilities,
    reasoning_capabilities_from_dict,
    reasoning_capabilities_to_dict,
)


CONNECTIONS_PATH = SETTINGS_DIR / "deployment_connection_summaries.json"
STORAGE_SNAPSHOT_PATH = SETTINGS_DIR / "storage_snapshot.json"


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
        "model_name": config.model_name or endpoint.model_name or "",
        "repo_id": config.repo_id or endpoint.repo_id or "",
        "quant": config.quant or endpoint.quant or "",
        "max_context_tokens": payload.get("context_limit"),
        "max_output_tokens": payload.get("output_limit"),
        "reasoning": payload.get("reasoning"),
        "runtime_attestation": runtime_attestation_to_dict(config.runtime_attestation),
        "api_key": config.endpoint_api_key or endpoint.endpoint_api_key or "",
        "cached_at_epoch": time.time(),
    }
    _write(entries, path)


def merge_connections(
    rows: list[EndpointInfo],
    path: Path = CONNECTIONS_PATH,
    storage_path: Path = STORAGE_SNAPSHOT_PATH,
    *,
    persist_backfill: bool = True,
) -> list[EndpointInfo]:
    """Hydrate provider rows with locally persisted endpoint metadata."""
    entries = load_connection_entries(path)
    _backfill_legacy_reasoning(
        entries,
        path=path,
        storage_path=storage_path,
        persist=persist_backfill,
    )
    for row in rows:
        cached = entries.get((row.name or "").strip())
        if not cached:
            continue
        row.web_url = row.web_url or str(cached.get("base_url") or "").removesuffix("/v1")
        row.served_model_name = row.served_model_name or str(cached.get("model_id") or "") or None
        row.display_name = row.display_name or str(cached.get("display_name") or "") or None
        row.model_name = row.model_name or str(cached.get("model_name") or "") or None
        row.repo_id = row.repo_id or str(cached.get("repo_id") or "") or None
        row.quant = row.quant or str(cached.get("quant") or "") or None
        row.endpoint_api_key = row.endpoint_api_key or str(cached.get("api_key") or "") or None
        row.app_id = row.app_id or str(cached.get("resource_id") or "")
        row.instance_name = row.instance_name or str(cached.get("instance_name") or "") or None
        row.max_context_tokens = row.max_context_tokens or positive_int(
            cached.get("max_context_tokens")
        )
        row.max_output_tokens = row.max_output_tokens or positive_int(
            cached.get("max_output_tokens")
        )
        row.reasoning = row.reasoning or reasoning_capabilities_from_dict(
            cached.get("reasoning")
        )
        row.runtime_attestation = row.runtime_attestation or runtime_attestation_from_dict(
            cached.get("runtime_attestation")
        )
        provider = str(cached.get("provider") or "")
        if provider in {item.value for item in ComputeProvider}:
            row.provider = ComputeProvider(provider)
        backend = str(cached.get("backend") or "")
        if backend in {item.value for item in BackendType}:
            row.backend = BackendType(backend)
    return rows


def rows_from_connection_cache(
    path: Path = CONNECTIONS_PATH,
    storage_path: Path = STORAGE_SNAPSHOT_PATH,
    *,
    persist_backfill: bool = True,
) -> list[EndpointInfo]:
    """Build endpoint rows from locally persisted connection metadata."""
    entries = load_connection_entries(path)
    _backfill_legacy_reasoning(
        entries,
        path=path,
        storage_path=storage_path,
        persist=persist_backfill,
    )
    rows: list[EndpointInfo] = []
    for app_name, entry in entries.items():
        base_url = str(entry.get("base_url") or "").removesuffix("/v1")
        if not base_url:
            continue
        provider_value = str(entry.get("provider") or "")
        backend_value = str(entry.get("backend") or "")
        rows.append(
            EndpointInfo(
                name=app_name,
                app_id=str(entry.get("resource_id") or ""),
                backend=(
                    BackendType(backend_value)
                    if backend_value in {item.value for item in BackendType}
                    else None
                ),
                instance_name=str(entry.get("instance_name") or "") or None,
                web_url=base_url,
                served_model_name=str(entry.get("model_id") or "") or None,
                display_name=str(entry.get("display_name") or "") or None,
                model_name=str(entry.get("model_name") or "") or None,
                repo_id=str(entry.get("repo_id") or "") or None,
                quant=str(entry.get("quant") or "") or None,
                provider=(
                    ComputeProvider(provider_value)
                    if provider_value in {item.value for item in ComputeProvider}
                    else ComputeProvider.MODAL
                ),
                endpoint_api_key=str(entry.get("api_key") or "") or None,
                max_context_tokens=positive_int(entry.get("max_context_tokens")),
                max_output_tokens=positive_int(entry.get("max_output_tokens")),
                reasoning=reasoning_capabilities_from_dict(entry.get("reasoning")),
                runtime_attestation=runtime_attestation_from_dict(
                    entry.get("runtime_attestation")
                ),
            )
        )
    return rows


def _backfill_legacy_reasoning(
    entries: dict[str, dict[str, Any]],
    *,
    path: Path,
    storage_path: Path,
    persist: bool,
) -> None:
    """Repair pre-capability connection rows from the cached model inventory.

    Older Launchpad versions saved only the served model label. The storage
    snapshot still records the selected Hugging Face repository and revision,
    which lets us re-run the same source inspection without a model allowlist.
    Failures are intentionally non-fatal so an unavailable Hub never prevents
    deployment listing or OpenCode sync.
    """
    storage_rows = _load_storage_model_rows(storage_path)
    if not storage_rows:
        return

    changed = False
    for entry in entries.values():
        if reasoning_capabilities_from_dict(entry.get("reasoning")) is not None:
            continue
        backend = _backend_from_entry(entry)
        if backend is None:
            continue
        selected = _resolve_storage_model(entry, backend, storage_rows)
        if selected is None:
            continue
        repo_id = str(selected.get("model_id") or "").strip()
        revision = str(selected.get("revision") or "").strip() or None
        if revision and (
            str(entry.get("reasoning_checked_repo") or "").strip() == repo_id
            and str(entry.get("reasoning_checked_revision") or "").strip()
            == revision
        ):
            continue
        try:
            capabilities = discover_reasoning_capabilities(
                backend,
                repo_id,
                revision,
            )
        except Exception:
            continue
        if capabilities is None:
            if revision:
                entry["reasoning_checked_repo"] = repo_id
                entry["reasoning_checked_revision"] = revision
                changed = True
            continue

        if backend == BackendType.LLAMACPP:
            entry["repo_id"] = repo_id
        else:
            entry["model_name"] = repo_id
        selected_quant = str(selected.get("quant") or "").strip()
        if selected_quant and not str(entry.get("quant") or "").strip():
            entry["quant"] = selected_quant
        entry["reasoning"] = reasoning_capabilities_to_dict(capabilities)
        entry["reasoning_checked_repo"] = repo_id
        entry["reasoning_checked_revision"] = capabilities.model_revision
        changed = True

    if changed and persist:
        _write(entries, path)


def _load_storage_model_rows(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    snapshot = payload.get("snapshot") if isinstance(payload, dict) else None
    if not isinstance(snapshot, dict):
        return []
    rows: list[dict[str, Any]] = []
    for key in ("llamacpp_models", "vllm_models"):
        value = snapshot.get(key)
        if not isinstance(value, list):
            continue
        rows.extend(row for row in value if isinstance(row, dict))
    return rows


def _backend_from_entry(entry: dict[str, Any]) -> BackendType | None:
    raw = str(entry.get("backend") or "").strip()
    try:
        return BackendType(raw)
    except ValueError:
        return None


def _resolve_storage_model(
    entry: dict[str, Any],
    backend: BackendType,
    storage_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in storage_rows
        if str(row.get("backend") or "").strip() == backend.value
        and "/" in str(row.get("model_id") or "")
    ]
    if not candidates:
        return None

    explicit_key = "repo_id" if backend == BackendType.LLAMACPP else "model_name"
    explicit_repo = str(entry.get(explicit_key) or "").strip()
    if explicit_repo:
        exact = [
            row
            for row in candidates
            if str(row.get("model_id") or "").strip() == explicit_repo
        ]
        return _unique_storage_revision(exact)

    served_model = str(entry.get("model_id") or "").strip()
    display_name = str(entry.get("display_name") or "").strip()
    display_base = display_name.rsplit(" (", 1)[0].strip()
    identities = {value for value in (served_model, display_base) if value}

    scored: list[tuple[int, dict[str, Any]]] = []
    for row in candidates:
        repo_id = str(row.get("model_id") or "").strip()
        repo_tail = repo_id.rsplit("/", 1)[-1]
        score = 0
        if repo_id in identities:
            score = 300
        elif repo_tail in identities:
            score = 200
        elif (
            backend == BackendType.LLAMACPP
            and any(identity.startswith(f"{repo_tail}-") for identity in identities)
        ):
            score = 100 + min(len(repo_tail), 99)
        if score:
            scored.append((score, row))
    if not scored:
        return None

    best_score = max(score for score, _row in scored)
    best = [row for score, row in scored if score == best_score]
    repo_ids = {str(row.get("model_id") or "").strip() for row in best}
    if len(repo_ids) != 1:
        return None
    return _unique_storage_revision(best)


def _unique_storage_revision(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    revisions = {str(row.get("revision") or "").strip() for row in rows}
    if len(revisions) > 1:
        return None
    return rows[0]


def remove_connection(app_name: str, path: Path = CONNECTIONS_PATH) -> None:
    """Remove local metadata for a terminated deployment."""
    entries = load_connection_entries(path)
    if entries.pop(app_name, None) is not None:
        _write(entries, path)
