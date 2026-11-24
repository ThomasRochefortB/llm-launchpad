"""Modal volume storage listing, pre-downloading, and deletion workflows."""

from __future__ import annotations

from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
import json
import re
import time
from typing import Any

from ..protocol.enums import BackendType, DeploymentState, OperationType
from ..protocol.events import (
    BaseEvent,
    ErrorEvent,
    LogEvent,
    OperationCompleteEvent,
    StateChangeEvent,
)
from ..protocol.models import StorageSnapshot, StoredModelInfo
from .backend import ModalBackend
from .naming import utility_app_name
from .operation_events import fail_operation
from .paths import MODAL_LLAMACPP_SCRIPT, MODAL_VLLM_SCRIPT

EventStream = Generator[BaseEvent, None, None]

LLAMACPP_STORAGE_VOLUME = "huggingface-cache"
VLLM_STORAGE_VOLUME = "huggingface-cache"

_GGUF_QUANT_RE = re.compile(r"(Q\d(?:_[A-Z0-9]+)+|IQ\d+_[A-Z0-9_]+)", flags=re.IGNORECASE)
_SIZE_TOKEN_RE = re.compile(r"^\s*(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z]+)?\s*$")
_LLAMACPP_MIN_PLAUSIBLE_GGUF_BYTES = 1024 * 1024
_LLAMACPP_STORAGE_JSON_BEGIN = "LLM_LAUNCHPAD_STORAGE_JSON_BEGIN"
_LLAMACPP_STORAGE_JSON_END = "LLM_LAUNCHPAD_STORAGE_JSON_END"


def entry_value(row: dict[str, Any], *keys: str) -> Any:
    """Case-insensitive lookup for JSON fields from Modal CLI."""
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for key in keys:
        needle = key.strip().lower()
        if needle in lowered:
            return lowered[needle]
    return None


def volume_entry_path(parent: str, row: dict[str, Any]) -> str:
    for key in ("path", "name", "file", "filename", "entry"):
        value = entry_value(row, key)
        if isinstance(value, str) and value.strip():
            candidate = value.strip()
            if candidate.startswith("/"):
                return candidate
            parent_clean = parent.strip("/")
            if parent_clean and candidate.startswith(f"{parent_clean}/"):
                return f"/{candidate.lstrip('/')}"
            if parent == "/" and candidate:
                return f"/{candidate.lstrip('/')}"
            if parent.endswith("/"):
                return f"{parent}{candidate}"
            return f"{parent}/{candidate}"
    return ""


def volume_entry_is_dir(row: dict[str, Any], path: str) -> bool:
    is_dir = entry_value(row, "is_dir", "isdir", "directory")
    if isinstance(is_dir, bool):
        return is_dir
    entry_type = str(entry_value(row, "type", "kind") or "").strip().lower()
    if entry_type in {"dir", "directory", "folder"}:
        return True
    return path.endswith("/")


def parse_size_bytes(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        return 0
    clean = text.replace(",", "")
    if clean.isdigit():
        return int(clean)
    match = _SIZE_TOKEN_RE.match(clean)
    if not match:
        return 0
    number = float(match.group("num"))
    unit = (match.group("unit") or "B").upper()
    factors = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
        "TIB": 1024**4,
    }
    return int(number * factors.get(unit, 1))


def walk_volume_files(
    volume_name: str,
    root: str,
    recursive: bool = True,
) -> list[dict[str, Any]]:
    pending = [root]
    files: list[dict[str, Any]] = []
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        rows = ModalBackend.list_volume(volume_name, current) or []
        for row in rows:
            path = volume_entry_path(current, row)
            if not path:
                continue
            if volume_entry_is_dir(row, path):
                if path == current or not recursive:
                    continue
                pending.append(path)
            else:
                files.append(
                    {
                        "path": path,
                        "size_bytes": parse_size_bytes(
                            entry_value(row, "size", "size_bytes", "bytes", "filesize")
                        ),
                        "is_file": True,
                    }
                )
    return files


def storage_delete_targets(
    model: StoredModelInfo,
    *,
    llamacpp_volume: str = LLAMACPP_STORAGE_VOLUME,
    vllm_volume: str = VLLM_STORAGE_VOLUME,
) -> list[tuple[str, str, bool]]:
    encoded = model.model_id.replace("/", "--")
    if model.backend == BackendType.LLAMACPP:
        return [(model.source_volume or llamacpp_volume, f"/hub/models--{encoded}", True)]
    if model.backend == BackendType.VLLM:
        return [(model.source_volume or vllm_volume, f"/hub/models--{encoded}", True)]
    return []


def list_llamacpp_models_via_backend(
    *,
    llamacpp_volume: str = LLAMACPP_STORAGE_VOLUME,
) -> list[StoredModelInfo] | None:
    captured = ModalBackend.run_modal_script_entrypoint_capture(
        MODAL_LLAMACPP_SCRIPT,
        "list_downloaded_models_json",
        env={"MODAL_APP_NAME": utility_app_name(BackendType.LLAMACPP)},
    )
    if not captured:
        return None
    returncode, stdout, _stderr = captured
    if returncode != 0 or not stdout.strip():
        return None

    lines = stdout.splitlines()
    try:
        start = lines.index(_LLAMACPP_STORAGE_JSON_BEGIN)
        end = lines.index(_LLAMACPP_STORAGE_JSON_END, start + 1)
    except ValueError:
        return None
    if end <= start + 1:
        return []

    payload_text = "\n".join(lines[start + 1 : end]).strip()
    if not payload_text:
        return []
    try:
        payload = json.loads(payload_text)
    except Exception:
        return None
    if not isinstance(payload, list):
        return None

    rows: list[StoredModelInfo] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("model_id", "")).strip()
        if not model_id:
            continue
        revision_raw = item.get("revision")
        quant_raw = item.get("quant")
        source_volume_raw = item.get("source_volume")
        paths_raw = item.get("paths")
        rows.append(
            StoredModelInfo(
                backend=BackendType.LLAMACPP,
                model_id=model_id,
                revision=str(revision_raw).strip() if revision_raw not in (None, "") else None,
                quant=str(quant_raw).strip() if quant_raw not in (None, "") else None,
                size_bytes=max(0, int(item.get("size_bytes", 0) or 0)),
                file_count=max(0, int(item.get("file_count", 0) or 0)),
                source_volume=(
                    str(source_volume_raw).strip()
                    if source_volume_raw not in (None, "")
                    else llamacpp_volume
                ),
                paths=[str(p) for p in paths_raw if isinstance(p, str)]
                if isinstance(paths_raw, list)
                else [],
                incomplete=False,
            )
        )
    return sorted(rows, key=lambda row: (row.model_id, row.revision or ""))


def list_llamacpp_models_from_volume(
    *,
    llamacpp_volume: str = LLAMACPP_STORAGE_VOLUME,
) -> list[StoredModelInfo]:
    grouped: dict[tuple[str, str | None], StoredModelInfo] = {}
    blob_files: dict[str, set[str]] = {}
    incomplete_blob_files: dict[str, set[str]] = {}
    snapshot_gguf_sizes: dict[str, list[int]] = {}
    hub_files = walk_volume_files(llamacpp_volume, "/hub")
    for row in hub_files:
        path = str(row.get("path", "")).strip()
        size = int(row.get("size_bytes", 0) or 0)
        rel = path.removeprefix("/hub/").strip("/")
        if not rel.startswith("models--") or not row.get("is_file", False):
            continue
        parts = rel.split("/")
        if len(parts) < 2:
            continue
        model_id = parts[0][len("models--") :].replace("--", "/")
        lower_path = path.lower()
        if "/blobs/" in lower_path:
            blob_name = path.rsplit("/", 1)[-1].strip()
            if blob_name.endswith(".incomplete"):
                final_blob_name = blob_name[: -len(".incomplete")]
                if final_blob_name:
                    incomplete_blob_files.setdefault(model_id, set()).add(final_blob_name)
            elif blob_name:
                blob_files.setdefault(model_id, set()).add(blob_name)
        if not lower_path.endswith(".gguf"):
            continue
        snapshot_gguf_sizes.setdefault(model_id, []).append(size)
        key = (model_id, None)
        entry = grouped.get(key)
        if entry is None:
            entry = StoredModelInfo(
                backend=BackendType.LLAMACPP,
                model_id=model_id,
                revision=None,
                quant=None,
                size_bytes=0,
                file_count=0,
                source_volume=llamacpp_volume,
                paths=[],
            )
            grouped[key] = entry
        entry.size_bytes += size
        entry.file_count += 1
        if entry.paths is not None and path not in entry.paths:
            entry.paths.append(path)
        if entry.quant is None:
            quant_match = _GGUF_QUANT_RE.search(path.upper())
            if quant_match:
                entry.quant = quant_match.group(1).upper()

    for (model_id, _revision), entry in grouped.items():
        known_blobs = blob_files.get(model_id, set())
        pending_blobs = incomplete_blob_files.get(model_id, set())
        unresolved_incomplete = any(blob_name not in known_blobs for blob_name in pending_blobs)
        gguf_sizes = snapshot_gguf_sizes.get(model_id, [])
        tiny_snapshot_without_blob = (
            bool(gguf_sizes)
            and all(size < _LLAMACPP_MIN_PLAUSIBLE_GGUF_BYTES for size in gguf_sizes)
            and not known_blobs
        )
        if unresolved_incomplete or tiny_snapshot_without_blob:
            entry.incomplete = True

    return sorted(grouped.values(), key=lambda row: (row.model_id, row.revision or ""))


def list_llamacpp_models(
    *,
    llamacpp_volume: str = LLAMACPP_STORAGE_VOLUME,
) -> list[StoredModelInfo]:
    backend_rows = list_llamacpp_models_via_backend(llamacpp_volume=llamacpp_volume)
    if backend_rows is not None:
        return backend_rows
    return list_llamacpp_models_from_volume(llamacpp_volume=llamacpp_volume)


def list_vllm_models(
    *,
    vllm_volume: str = VLLM_STORAGE_VOLUME,
) -> list[StoredModelInfo]:
    files = walk_volume_files(vllm_volume, "/hub")
    grouped: dict[str, StoredModelInfo] = {}
    blob_files: dict[str, set[str]] = {}
    incomplete_blob_files: dict[str, set[str]] = {}
    snapshot_file_counts: dict[str, int] = {}
    ref_file_counts: dict[str, int] = {}
    gguf_models: set[str] = set()
    for row in files:
        path = str(row.get("path", "")).strip()
        size = int(row.get("size_bytes", 0) or 0)
        rel = path.removeprefix("/hub/").strip("/")
        if not rel.startswith("models--"):
            continue
        parts = rel.split("/")
        if not parts:
            continue
        model_id = parts[0][len("models--") :].replace("--", "/")
        entry = grouped.get(model_id)
        if entry is None:
            entry = StoredModelInfo(
                backend=BackendType.VLLM,
                model_id=model_id,
                revision=None,
                quant=None,
                size_bytes=0,
                file_count=0,
                source_volume=vllm_volume,
                paths=[],
            )
            grouped[model_id] = entry
        if row.get("is_file", False):
            entry.size_bytes += size
            entry.file_count += 1
            lower_path = path.lower()
            if "/snapshots/" in lower_path:
                snapshot_file_counts[model_id] = snapshot_file_counts.get(model_id, 0) + 1
            if "/refs/" in lower_path:
                ref_file_counts[model_id] = ref_file_counts.get(model_id, 0) + 1
            if "/blobs/" in lower_path:
                blob_name = path.rsplit("/", 1)[-1].strip()
                if blob_name.endswith(".incomplete"):
                    final_blob_name = blob_name[: -len(".incomplete")]
                    if final_blob_name:
                        incomplete_blob_files.setdefault(model_id, set()).add(final_blob_name)
                elif blob_name:
                    blob_files.setdefault(model_id, set()).add(blob_name)
            if lower_path.endswith(".gguf"):
                gguf_models.add(model_id)
            if entry.paths is not None and path not in entry.paths:
                entry.paths.append(path)

    for model_id in list(grouped.keys()):
        if model_id in gguf_models:
            del grouped[model_id]

    if not grouped:
        return []

    for model_id, entry in grouped.items():
        known_blobs = blob_files.get(model_id, set())
        pending_blobs = incomplete_blob_files.get(model_id, set())
        snapshot_count = snapshot_file_counts.get(model_id, 0)
        ref_count = ref_file_counts.get(model_id, 0)
        unresolved_incomplete = any(blob_name not in known_blobs for blob_name in pending_blobs)
        if unresolved_incomplete and snapshot_count == 0:
            entry.incomplete = True
        if ref_count > 0 and snapshot_count == 0:
            entry.incomplete = True
    return sorted(grouped.values(), key=lambda row: row.model_id)


def stream_list_storage(
    *,
    llamacpp_volume: str = LLAMACPP_STORAGE_VOLUME,
    vllm_volume: str = VLLM_STORAGE_VOLUME,
) -> EventStream:
    """List cached model artifacts for both backends."""
    yield StateChangeEvent(
        current=DeploymentState.RUNNING,
        operation=OperationType.STORAGE_LIST,
        detail="Scanning Modal volumes for cached models",
    )
    scan_started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            llamacpp_future = executor.submit(
                list_llamacpp_models, llamacpp_volume=llamacpp_volume
            )
            vllm_future = executor.submit(list_vllm_models, vllm_volume=vllm_volume)
            snapshot = StorageSnapshot(
                llamacpp_models=llamacpp_future.result(),
                vllm_models=vllm_future.result(),
            )
    except Exception as exc:
        yield from fail_operation(
            OperationType.STORAGE_LIST,
            f"Failed to list storage: {exc}",
            detail=str(exc),
        )
        return

    scan_elapsed_ms = int((time.perf_counter() - scan_started) * 1000)
    yield LogEvent(
        line=(
            f"Storage snapshot: llama.cpp={len(snapshot.llamacpp_models)} models, "
            f"vLLM={len(snapshot.vllm_models)} models, "
            f"total={snapshot.total_models}, bytes={snapshot.total_size_bytes}, "
            f"scan_ms={scan_elapsed_ms}"
        ),
        operation=OperationType.STORAGE_LIST,
    )
    yield OperationCompleteEvent(
        operation=OperationType.STORAGE_LIST,
        success=True,
        data=snapshot,
    )


def stream_predownload_model(
    backend: BackendType,
    model_id: str,
    quant: str | None = None,
    revision: str | None = None,
) -> EventStream:
    """Pre-download a model for a backend without deploying."""
    if backend == BackendType.LLAMACPP:
        script = MODAL_LLAMACPP_SCRIPT
        entrypoint = "predownload_model"
        args = ["--repo-id", model_id]
        if quant:
            args.extend(["--quant", quant])
        if revision:
            args.extend(["--revision", revision])
    else:
        script = MODAL_VLLM_SCRIPT
        entrypoint = "predownload_model"
        args = ["--repo-id", model_id]
        if revision:
            args.extend(["--revision", revision])

    cmd = ModalBackend.build_modal_entrypoint_command(script, entrypoint, args)
    env = {"MODAL_APP_NAME": utility_app_name(backend)}
    yield StateChangeEvent(
        current=DeploymentState.RUNNING,
        operation=OperationType.STORAGE_PREDOWNLOAD,
        detail=" ".join(cmd),
    )
    yield LogEvent(
        line=f"Running: {' '.join(cmd)}",
        operation=OperationType.STORAGE_PREDOWNLOAD,
    )

    for event in ModalBackend.run_modal_script_entrypoint(
        script, entrypoint, args=args, env=env
    ):
        if isinstance(event, OperationCompleteEvent):
            yield OperationCompleteEvent(
                operation=OperationType.STORAGE_PREDOWNLOAD,
                success=event.success,
                exit_code=event.exit_code,
                detail=event.detail,
                data=event.data,
            )
        elif isinstance(event, ErrorEvent):
            yield ErrorEvent(
                message=event.message,
                operation=OperationType.STORAGE_PREDOWNLOAD,
                exit_code=event.exit_code,
                recoverable=event.recoverable,
            )
        elif isinstance(event, LogEvent):
            yield LogEvent(
                line=event.line,
                stream=event.stream,
                operation=OperationType.STORAGE_PREDOWNLOAD,
            )
        else:
            yield event


def stream_delete_stored_model(
    model: StoredModelInfo,
    *,
    llamacpp_volume: str = LLAMACPP_STORAGE_VOLUME,
    vllm_volume: str = VLLM_STORAGE_VOLUME,
) -> EventStream:
    """Delete cached storage artifacts for a selected model."""
    targets = storage_delete_targets(
        model, llamacpp_volume=llamacpp_volume, vllm_volume=vllm_volume
    )
    if not targets:
        yield from fail_operation(
            OperationType.STORAGE_DELETE,
            f"No removable paths found for {model.model_id}",
            detail="No paths resolved for deletion.",
        )
        return

    yield StateChangeEvent(
        current=DeploymentState.RUNNING,
        operation=OperationType.STORAGE_DELETE,
        detail=f"Deleting {model.model_id}",
    )
    for volume_name, remote_path, recursive in targets:
        yield LogEvent(
            line=f"Deleting {remote_path} from {volume_name}",
            operation=OperationType.STORAGE_DELETE,
        )
        for event in ModalBackend.run_volume_remove(
            volume_name, remote_path, recursive=recursive
        ):
            if isinstance(event, OperationCompleteEvent):
                if not event.success:
                    yield OperationCompleteEvent(
                        operation=OperationType.STORAGE_DELETE,
                        success=False,
                        exit_code=event.exit_code,
                        detail=event.detail or f"Failed removing {remote_path}",
                    )
                    return
                continue
            if isinstance(event, ErrorEvent):
                yield from fail_operation(
                    OperationType.STORAGE_DELETE,
                    event.message,
                    exit_code=event.exit_code or 1,
                    recoverable=event.recoverable,
                    error_exit_code=event.exit_code,
                )
                return
            if isinstance(event, LogEvent):
                yield LogEvent(
                    line=event.line,
                    stream=event.stream,
                    operation=OperationType.STORAGE_DELETE,
                )
            else:
                yield event

    yield OperationCompleteEvent(
        operation=OperationType.STORAGE_DELETE,
        success=True,
        detail=f"Deleted storage for {model.model_id}",
    )
