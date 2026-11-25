"""Modal volume storage listing, pre-downloading, and deletion workflows."""

from __future__ import annotations

from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor, wait, FIRST_COMPLETED
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
_VOLUME_WALK_WORKERS = 8
_DEFAULT_LIST_VOLUME = ModalBackend.list_volume


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


def _is_missing_volume_path_error(err: Any) -> bool:
    if err is None or getattr(err, "timed_out", False):
        return False
    lower_msg = str(getattr(err, "message", "") or "").lower()
    return "volume" not in lower_msg and any(
        marker in lower_msg for marker in ("no such file", "path not found")
    )


def _should_skip_volume_dir(root: str, path: str) -> bool:
    """Skip non-model metadata dirs like /hub/.locks during Hub cache scans."""
    if root.rstrip("/") != "/hub":
        return False
    rel = path.removeprefix("/hub/").strip("/")
    if not rel:
        return False
    top = rel.split("/", 1)[0]
    return top.startswith(".") or not top.startswith("models--")


def _extract_snapshot_revision(path: str) -> str | None:
    """Extract the HF snapshot revision segment from ``.../snapshots/<rev>/...``."""
    parts = [part for part in path.strip("/").split("/") if part]
    for index, part in enumerate(parts):
        if part.lower() == "snapshots" and index + 1 < len(parts):
            candidate = parts[index + 1].strip()
            if candidate:
                return candidate
    return None


def _file_record_from_entry(parent: str, row: dict[str, Any], path: str) -> dict[str, Any]:
    entry_type = str(entry_value(row, "type", "kind") or "").strip().lower()
    return {
        "path": path,
        "size_bytes": parse_size_bytes(
            entry_value(row, "size", "size_bytes", "bytes", "filesize")
        ),
        "is_file": True,
        "is_symlink": entry_type == "symlink",
    }


def walk_volume_files(
    volume_name: str,
    root: str,
    recursive: bool = True,
) -> list[dict[str, Any]]:
    # Fast path: one Modal SDK recursive RPC (~0.3s) instead of N sequential CLI
    # subprocesses. Stay on ModalBackend.list_volume whenever it has been mocked
    # or replaced (unit tests), or if the SDK call fails for a non-path reason.
    if recursive and ModalBackend.list_volume is _DEFAULT_LIST_VOLUME:
        sdk_result = ModalBackend.list_volume_recursive_result(volume_name, root)
        if sdk_result.success and sdk_result.entries is not None:
            files: list[dict[str, Any]] = []
            for row in sdk_result.entries:
                path = volume_entry_path(root, row)
                if not path or _should_skip_volume_dir(root, path):
                    continue
                if volume_entry_is_dir(row, path):
                    continue
                files.append(_file_record_from_entry(root, row, path))
            return files
        if _is_missing_volume_path_error(sdk_result.error):
            return []

    def list_dir(current: str) -> tuple[str, list[dict[str, Any]] | None, Any]:
        rows = ModalBackend.list_volume(volume_name, current)
        err = ModalBackend.last_volume_error() if rows is None else None
        return current, rows, err

    files = []
    visited: set[str] = {root}
    in_flight: set[Future[tuple[str, list[dict[str, Any]] | None, Any]]] = set()
    max_workers = _VOLUME_WALK_WORKERS if recursive else 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        in_flight.add(executor.submit(list_dir, root))
        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                current, rows, err = future.result()
                if rows is None:
                    if _is_missing_volume_path_error(err):
                        rows = []
                    else:
                        for remaining in in_flight:
                            remaining.cancel()
                        err_msg = err.message if err is not None else ""
                        raise RuntimeError(
                            err_msg
                            or f"Failed to list Modal volume '{volume_name}' at {current}."
                        )
                for row in rows:
                    path = volume_entry_path(current, row)
                    if not path:
                        continue
                    if volume_entry_is_dir(row, path):
                        if (
                            path == current
                            or not recursive
                            or path in visited
                            or _should_skip_volume_dir(root, path)
                        ):
                            continue
                        visited.add(path)
                        in_flight.add(executor.submit(list_dir, path))
                    else:
                        files.append(_file_record_from_entry(current, row, path))
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


def _parse_llamacpp_models_from_files(
    hub_files: list[dict[str, Any]],
    *,
    llamacpp_volume: str = LLAMACPP_STORAGE_VOLUME,
) -> tuple[list[StoredModelInfo], bool]:
    """Parse llama.cpp GGUF entries from scanned ``/hub`` files.

    Returns ``(models, needs_backend_size_refinement)``. Single-quant repos can
    attribute their real blob size directly without spinning up a remote Modal
    container; multi-quant repos whose snapshots are symlinks still benefit from
    remote ``stat()`` refinement for per-quant byte splits.
    """
    grouped: dict[tuple[str, str | None, str | None], StoredModelInfo] = {}
    blob_files: dict[str, set[str]] = {}
    blob_sizes_by_model: dict[str, int] = {}
    incomplete_blob_files: dict[str, set[str]] = {}
    snapshot_gguf_sizes: dict[str, list[int]] = {}
    snapshot_gguf_has_symlink: dict[str, bool] = {}

    for row in hub_files:
        path = str(row.get("path", "")).strip()
        size = int(row.get("size_bytes", 0) or 0)
        rel = path.removeprefix("/hub/").strip("/")
        if not rel.startswith("models--") or not row.get("is_file", False):
            continue
        parts = rel.split("/")
        if len(parts) < 2:
            continue
        model_id = parts[0][len("models--") :].replace("--", "/", 1)
        lower_path = path.lower()
        if "/blobs/" in lower_path:
            blob_name = path.rsplit("/", 1)[-1].strip()
            if blob_name.endswith(".incomplete"):
                final_blob_name = blob_name[: -len(".incomplete")]
                if final_blob_name:
                    incomplete_blob_files.setdefault(model_id, set()).add(final_blob_name)
            elif blob_name:
                blob_files.setdefault(model_id, set()).add(blob_name)
                blob_sizes_by_model[model_id] = blob_sizes_by_model.get(model_id, 0) + size
            continue
        if not lower_path.endswith(".gguf"):
            continue
        filename = path.rsplit("/", 1)[-1].lower()
        if "mmproj" in filename:
            continue
        snapshot_gguf_sizes.setdefault(model_id, []).append(size)
        if row.get("is_symlink", False):
            snapshot_gguf_has_symlink[model_id] = True
        revision = _extract_snapshot_revision(path)
        quant_match = _GGUF_QUANT_RE.search(path.upper())
        quant = quant_match.group(1).upper() if quant_match else None
        key = (model_id, revision, quant)
        entry = grouped.get(key)
        if entry is None:
            entry = StoredModelInfo(
                backend=BackendType.LLAMACPP,
                model_id=model_id,
                revision=revision,
                quant=quant,
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

    entries_by_model: dict[str, list[StoredModelInfo]] = {}
    for entry in grouped.values():
        entries_by_model.setdefault(entry.model_id, []).append(entry)

    needs_backend_refinement = False
    for model_id, model_entries in entries_by_model.items():
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
            for entry in model_entries:
                entry.incomplete = True

        all_tiny_or_symlink = bool(gguf_sizes) and (
            snapshot_gguf_has_symlink.get(model_id, False)
            or all(size < _LLAMACPP_MIN_PLAUSIBLE_GGUF_BYTES for size in gguf_sizes)
        )
        blob_total = blob_sizes_by_model.get(model_id, 0)
        if all_tiny_or_symlink and blob_total > 0:
            if len(model_entries) == 1:
                model_entries[0].size_bytes = max(model_entries[0].size_bytes, blob_total)
            else:
                needs_backend_refinement = True
                total_files = sum(max(1, entry.file_count) for entry in model_entries)
                allocated = 0
                for index, entry in enumerate(model_entries):
                    if index == len(model_entries) - 1:
                        entry.size_bytes = max(0, blob_total - allocated)
                    else:
                        share = int(blob_total * (max(1, entry.file_count) / total_files))
                        entry.size_bytes = share
                        allocated += share

    models = sorted(
        grouped.values(),
        key=lambda row: (row.model_id, row.revision or "", row.quant or ""),
    )
    return models, needs_backend_refinement


def list_llamacpp_models_from_volume(
    *,
    llamacpp_volume: str = LLAMACPP_STORAGE_VOLUME,
) -> list[StoredModelInfo]:
    hub_files = walk_volume_files(llamacpp_volume, "/hub")
    models, _needs_refinement = _parse_llamacpp_models_from_files(
        hub_files, llamacpp_volume=llamacpp_volume
    )
    return models


def list_llamacpp_models(
    *,
    llamacpp_volume: str = LLAMACPP_STORAGE_VOLUME,
) -> list[StoredModelInfo]:
    hub_files = walk_volume_files(llamacpp_volume, "/hub")
    models, needs_refinement = _parse_llamacpp_models_from_files(
        hub_files, llamacpp_volume=llamacpp_volume
    )
    if needs_refinement:
        backend_rows = list_llamacpp_models_via_backend(llamacpp_volume=llamacpp_volume)
        if backend_rows is not None:
            return _merge_incomplete_flags(backend_rows, models)
    return models


def _merge_incomplete_flags(
    backend_rows: list[StoredModelInfo],
    volume_rows: list[StoredModelInfo],
) -> list[StoredModelInfo]:
    incomplete_models = {row.model_id for row in volume_rows if row.incomplete}
    if not incomplete_models:
        return backend_rows
    for row in backend_rows:
        if row.model_id in incomplete_models:
            row.incomplete = True
    return backend_rows


def _parse_vllm_models_from_files(
    files: list[dict[str, Any]],
    *,
    vllm_volume: str = VLLM_STORAGE_VOLUME,
) -> list[StoredModelInfo]:
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
        model_id = parts[0][len("models--") :].replace("--", "/", 1)
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


def list_vllm_models(
    *,
    vllm_volume: str = VLLM_STORAGE_VOLUME,
) -> list[StoredModelInfo]:
    files = walk_volume_files(vllm_volume, "/hub")
    return _parse_vllm_models_from_files(files, vllm_volume=vllm_volume)


def stream_list_storage(
    *,
    llamacpp_volume: str = LLAMACPP_STORAGE_VOLUME,
    vllm_volume: str = VLLM_STORAGE_VOLUME,
    on_partial: Callable[[StorageSnapshot], None] | None = None,
) -> EventStream:
    """List cached model artifacts for both backends in a single fast pass."""
    yield StateChangeEvent(
        current=DeploymentState.RUNNING,
        operation=OperationType.STORAGE_LIST,
        detail="Scanning Modal volumes for cached models",
    )
    scan_started = time.perf_counter()
    try:
        if llamacpp_volume == vllm_volume:
            shared_files = walk_volume_files(llamacpp_volume, "/hub")
            llamacpp_files = shared_files
            vllm_files = shared_files
        else:
            with ThreadPoolExecutor(max_workers=2) as executor:
                llamacpp_future = executor.submit(walk_volume_files, llamacpp_volume, "/hub")
                vllm_future = executor.submit(walk_volume_files, vllm_volume, "/hub")
                llamacpp_files = llamacpp_future.result()
                vllm_files = vllm_future.result()

        llamacpp_models, needs_refinement = _parse_llamacpp_models_from_files(
            llamacpp_files, llamacpp_volume=llamacpp_volume
        )
        vllm_models = _parse_vllm_models_from_files(vllm_files, vllm_volume=vllm_volume)
        snapshot = StorageSnapshot(
            llamacpp_models=llamacpp_models,
            vllm_models=vllm_models,
        )
        if needs_refinement:
            if on_partial is not None:
                on_partial(snapshot)
            backend_rows = list_llamacpp_models_via_backend(llamacpp_volume=llamacpp_volume)
            if backend_rows is not None:
                snapshot = StorageSnapshot(
                    llamacpp_models=_merge_incomplete_flags(backend_rows, llamacpp_models),
                    vllm_models=vllm_models,
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
