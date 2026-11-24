"""Disk persistence and serialization helpers for TUI storage and connection caches."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

from ..core.coerce import positive_int
from ..core.diagnostics import log_exception
from ..protocol.enums import BackendType
from ..protocol.models import StorageSnapshot, StoredModelInfo


def stored_model_to_dict(model: StoredModelInfo) -> dict[str, object]:
    """Serialize one :class:`StoredModelInfo` entry for JSON caching."""
    return {
        "backend": model.backend.value,
        "model_id": model.model_id,
        "revision": model.revision,
        "quant": model.quant,
        "size_bytes": model.size_bytes,
        "file_count": model.file_count,
        "source_volume": model.source_volume,
        "paths": model.paths or [],
        "incomplete": model.incomplete,
    }


def stored_model_from_dict(payload: dict[str, object]) -> StoredModelInfo | None:
    """Deserialize one cached :class:`StoredModelInfo` entry, or ``None`` if invalid."""
    backend_raw = str(payload.get("backend", "")).strip().lower()
    if backend_raw not in {"llamacpp", "vllm"}:
        return None
    paths_value = payload.get("paths")
    if isinstance(paths_value, str):
        paths = [paths_value] if paths_value.strip() else []
    elif isinstance(paths_value, (list, tuple)):
        paths = [str(path) for path in paths_value if str(path).strip()]
    else:
        paths = []
    return StoredModelInfo(
        backend=BackendType(backend_raw),
        model_id=str(payload.get("model_id", "")).strip(),
        revision=str(payload.get("revision")) if payload.get("revision") not in {None, ""} else None,
        quant=str(payload.get("quant")) if payload.get("quant") not in {None, ""} else None,
        size_bytes=positive_int(payload.get("size_bytes")) or 0,
        file_count=positive_int(payload.get("file_count")) or 0,
        source_volume=str(payload.get("source_volume", "") or ""),
        paths=paths,
        incomplete=bool(payload.get("incomplete", False)),
    )


def storage_snapshot_to_dict(snapshot: StorageSnapshot) -> dict[str, object]:
    """Serialize a full :class:`StorageSnapshot` for JSON persistence."""
    return {
        "llamacpp_models": [stored_model_to_dict(model) for model in snapshot.llamacpp_models],
        "vllm_models": [stored_model_to_dict(model) for model in snapshot.vllm_models],
    }


def storage_snapshot_from_dict(payload: dict[str, object]) -> StorageSnapshot | None:
    """Deserialize a cached :class:`StorageSnapshot` payload."""
    llamacpp_payload = payload.get("llamacpp_models")
    vllm_payload = payload.get("vllm_models")
    if not isinstance(llamacpp_payload, list) or not isinstance(vllm_payload, list):
        return None
    llamacpp_models: list[StoredModelInfo] = []
    for row in llamacpp_payload:
        if not isinstance(row, dict):
            continue
        model = stored_model_from_dict(row)
        if model is not None:
            llamacpp_models.append(model)
    vllm_models: list[StoredModelInfo] = []
    for row in vllm_payload:
        if not isinstance(row, dict):
            continue
        model = stored_model_from_dict(row)
        if model is not None:
            vllm_models.append(model)
    return StorageSnapshot(llamacpp_models=llamacpp_models, vllm_models=vllm_models)


def persist_storage_snapshot(path: Path, snapshot: StorageSnapshot) -> None:
    """Write ``snapshot`` to disk, logging on I/O failures."""
    payload = {
        "cached_at_epoch": time.time(),
        "snapshot": storage_snapshot_to_dict(snapshot),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    except Exception:
        log_exception("Failed to persist storage snapshot cache")


def load_persisted_storage_snapshot(path: Path) -> tuple[StorageSnapshot | None, float]:
    """Load the cached :class:`StorageSnapshot` and its epoch timestamp."""
    try:
        if not path.exists():
            return None, 0.0
        payload = json.loads(path.read_text())
        snapshot_payload = payload.get("snapshot")
        if not isinstance(snapshot_payload, dict):
            return None, 0.0
        snapshot = storage_snapshot_from_dict(snapshot_payload)
        if snapshot is None:
            return None, 0.0
        return snapshot, float(payload.get("cached_at_epoch", 0.0) or 0.0)
    except Exception:
        return None, 0.0


def persist_deploy_connection_cache(
    path: Path, entries: dict[str, dict[str, object]]
) -> None:
    """Persist deployment connection summaries with owner-only permissions."""
    payload = {"entries": entries}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        os.chmod(path, 0o600)
    except Exception:
        log_exception("Failed to persist deploy connection cache")


def load_persisted_deploy_connection_cache(path: Path) -> dict[str, dict[str, object]]:
    """Load cached deployment connection summaries from disk."""
    try:
        if not path.exists():
            return {}
        payload = json.loads(path.read_text())
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            return {}
        normalized: dict[str, dict[str, object]] = {}
        for key, value in entries.items():
            app_name = str(key or "").strip()
            if not app_name or not isinstance(value, dict):
                continue
            normalized[app_name] = value
        return normalized
    except Exception:
        return {}
