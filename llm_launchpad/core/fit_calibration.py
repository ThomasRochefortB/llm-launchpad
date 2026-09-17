"""Turn llama.cpp's own fit measurements into reusable planning evidence.

The planner predicts device memory by reconstructing what ggml will allocate:
weights and KV cache from GGUF headers, then compute graphs and attention
working memory from formulas. The first two are arithmetic on published
numbers and are reliable. The rest is a model of an allocator we do not own,
and every correction to it so far has come from a deployment failing rather
than from reading the source -- most recently a sparse-attention model whose
indexer masks cost 5 GB per device that no formula accounted for.

llama.cpp computes the true figure before every serve, and says so at trace
verbosity. This module captures that statement and solves it for the two
terms a formula cannot derive:

``G`` -- the graph memory every device pays in full, and
``E`` -- the fixed tensors (output, embeddings) that land on one device.

With the shardable part known from headers, a measurement on any topology
determines both, and every other topology for the same model and context
then follows exactly. A calibration is therefore keyed without the GPU type
or count: it describes the model's graph, not the hardware it was measured
on. Compute buffers follow tensor shapes rather than the device, so the same
numbers carry across cards and across providers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from ..protocol.models import RuntimeTuning, ServingRequirements
from .config import SETTINGS_DIR

CALIBRATION_SCHEMA_VERSION = 1
CALIBRATION_CACHE_PATH = SETTINGS_DIR / "memory_calibrations.json"

_MIB = 1024**2
_GB = 1000**3

# "common_params_fit_impl:   - CUDA0 (NVIDIA A100-SXM4-80GB):  81152 total,
#  78810 used,   1839 free vs. target of   4096". The runtime prefixes its own
# timestamp and level, so the device rows are matched anywhere in the line.
_DEVICE_RE = re.compile(
    r"-\s+(?P<device>\S+)\s*\((?P<description>[^)]*)\):\s*"
    r"(?P<total>\d+)\s+total,\s*"
    r"(?P<used>\d+)\s+used,\s*"
    r"(?P<free>\d+)\s+free\s+vs\.\s+target\s+of\s+(?P<target>\d+)"
)
_PROJECTED_RE = re.compile(
    r"projected to use\s+(?P<used>\d+)\s+MiB of device memory vs\.\s+"
    r"(?P<free>\d+)\s+MiB of free device memory"
)


@dataclass(frozen=True)
class FitMeasurement:
    """What llama.cpp reported it would allocate, in MiB."""

    per_device_used_mib: tuple[int, ...] = ()
    per_device_total_mib: tuple[int, ...] = ()
    device_descriptions: tuple[str, ...] = ()
    margin_mib: int = 0
    projected_used_mib: int = 0
    projected_free_mib: int = 0

    @property
    def is_usable(self) -> bool:
        """Return whether there is enough here to solve a calibration."""

        return bool(self.per_device_used_mib) or self.projected_used_mib > 0

    @property
    def total_used_mib(self) -> int:
        """Total device memory the plan needs, across every device."""

        if self.per_device_used_mib:
            return sum(self.per_device_used_mib)
        return self.projected_used_mib


@dataclass(frozen=True)
class MemoryCalibration:
    """Graph memory measured for one model, context and runtime tuning."""

    key: str
    graph_gb: float
    fixed_extra_gb: float
    shardable_gb: float
    layer_count: int
    gpu_type: str
    gpu_count: int
    measured_at: str
    runtime_id: str | None = None

    def per_device_gb(
        self,
        *,
        shardable_gb: float,
        gpu_count: int,
        layer_count: int | None,
    ) -> tuple[float, ...]:
        """Predict each device's requirement for a topology, from measurement."""

        count = max(1, gpu_count)
        layers = layer_count if layer_count and layer_count > 0 else self.layer_count
        if layers < count:
            shares = [1.0 / count] * count
        else:
            base, remainder = divmod(layers, count)
            shares = [
                (base + (1 if index < remainder else 0)) / layers for index in range(count)
            ]
        # The fixed tensors ride on whichever device takes the first layers,
        # which is the same device that takes the remainder.
        return tuple(
            shardable_gb * share + self.graph_gb + (self.fixed_extra_gb if index == 0 else 0.0)
            for index, share in enumerate(shares)
        )


def parse_fit_measurement(text: str) -> FitMeasurement:
    """Read llama.cpp's fit arithmetic out of deploy log output."""

    used: list[int] = []
    totals: list[int] = []
    descriptions: list[str] = []
    margin = 0
    projected_used = 0
    projected_free = 0
    for line in (text or "").splitlines():
        device = _DEVICE_RE.search(line)
        if device is not None:
            used.append(int(device.group("used")))
            totals.append(int(device.group("total")))
            descriptions.append(device.group("description").strip())
            margin = max(margin, int(device.group("target")))
            continue
        projected = _PROJECTED_RE.search(line)
        if projected is not None:
            projected_used = int(projected.group("used"))
            projected_free = int(projected.group("free"))
    return FitMeasurement(
        per_device_used_mib=tuple(used),
        per_device_total_mib=tuple(totals),
        device_descriptions=tuple(descriptions),
        margin_mib=margin,
        projected_used_mib=projected_used,
        projected_free_mib=projected_free,
    )


def solve_calibration(
    measurement: FitMeasurement,
    *,
    key: str,
    shardable_gb: float,
    layer_count: int,
    gpu_type: str,
    gpu_count: int,
    runtime_id: str | None = None,
) -> MemoryCalibration | None:
    """Solve a measurement for the per-device graph and fixed-tensor terms.

    Two devices give two equations, which separate the graph memory from the
    tensors that sit on one device only. A single device gives one equation,
    so everything above the shardable part is attributed to the graph; that
    still sizes single-GPU plans exactly and only understates the skew of a
    multi-GPU one, which the measured-at provenance makes visible.
    """

    if not measurement.is_usable or shardable_gb <= 0 or layer_count <= 0:
        return None
    shardable_mib = shardable_gb * _GB / _MIB
    used = measurement.per_device_used_mib
    count = max(1, len(used) or gpu_count)
    if count > layer_count:
        return None
    base, remainder = divmod(layer_count, count)
    shares = [(base + (1 if index < remainder else 0)) / layer_count for index in range(count)]

    if len(used) >= 2:
        # The busiest device carries the remainder layers and the fixed
        # tensors; a quieter one carries neither, which separates the terms.
        quietest_index = min(range(len(used)), key=lambda index: used[index])
        graph_mib = used[quietest_index] - shardable_mib * shares[-1]
        fixed_mib = max(used) - shardable_mib * shares[0] - graph_mib
    else:
        graph_mib = measurement.total_used_mib - shardable_mib
        fixed_mib = 0.0
    if graph_mib <= 0:
        # The headers already account for everything measured; a negative
        # graph term would mean the shardable estimate is the wrong one, and
        # guessing which is wrong is not something a calibration can do.
        return None
    return MemoryCalibration(
        key=key,
        graph_gb=round(graph_mib * _MIB / _GB, 4),
        fixed_extra_gb=round(max(0.0, fixed_mib) * _MIB / _GB, 4),
        shardable_gb=round(shardable_gb, 4),
        layer_count=layer_count,
        gpu_type=gpu_type,
        gpu_count=count,
        measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
        runtime_id=runtime_id,
    )


def calibration_key(
    *,
    model_id: str,
    revision: str | None,
    quant: str | None,
    runtime_id: str | None,
    requirements: ServingRequirements,
    tuning: RuntimeTuning,
) -> str:
    """Identify a graph shape, deliberately without the hardware it ran on.

    Everything here changes what ggml allocates. The GPU type and count do
    not: compute buffers are sized from tensor shapes, so one measurement is
    evidence for every topology and every provider serving the same plan.
    """

    payload = {
        "schema": CALIBRATION_SCHEMA_VERSION,
        "model_id": model_id.strip(),
        "revision": (revision or "").strip(),
        "quant": (quant or "").strip(),
        "runtime_id": (runtime_id or "").strip(),
        "context_tokens": requirements.context_tokens,
        "tuning": {
            "batch_size": tuning.batch_size,
            "ubatch_size": tuning.ubatch_size,
            "cache_type_k": tuning.cache_type_k,
            "cache_type_v": tuning.cache_type_v,
            "flash_attention": tuning.flash_attention,
            "parallel_slots": tuning.parallel_slots,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"schema_version": CALIBRATION_SCHEMA_VERSION, "entries": {}}
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CALIBRATION_SCHEMA_VERSION
    ):
        return {"schema_version": CALIBRATION_SCHEMA_VERSION, "entries": {}}
    return payload


def save_memory_calibration(
    calibration: MemoryCalibration,
    path: Path | None = None,
) -> None:
    """Persist a calibration atomically, replacing any earlier measurement."""

    path = path if path is not None else CALIBRATION_CACHE_PATH
    payload = _read_payload(path)
    entries = payload.setdefault("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        payload["entries"] = entries
    entries[calibration.key] = asdict(calibration)
    payload["schema_version"] = CALIBRATION_SCHEMA_VERSION
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.chmod(temp_path, 0o600)
    temp_path.replace(path)


def load_memory_calibration(
    key: str,
    path: Path | None = None,
) -> MemoryCalibration | None:
    """Load one measured calibration, or ``None`` when nothing has run yet."""

    path = path if path is not None else CALIBRATION_CACHE_PATH
    if not path.exists():
        return None
    raw = _read_payload(path).get("entries")
    if not isinstance(raw, dict):
        return None
    entry = raw.get(key)
    if not isinstance(entry, dict):
        return None
    try:
        return MemoryCalibration(
            key=str(entry["key"]),
            graph_gb=float(entry["graph_gb"]),
            fixed_extra_gb=float(entry.get("fixed_extra_gb", 0.0)),
            shardable_gb=float(entry.get("shardable_gb", 0.0)),
            layer_count=int(entry["layer_count"]),
            gpu_type=str(entry.get("gpu_type") or ""),
            gpu_count=int(entry.get("gpu_count", 1)),
            measured_at=str(entry.get("measured_at") or ""),
            runtime_id=entry.get("runtime_id"),
        )
    except (KeyError, TypeError, ValueError):
        return None


class FitCalibrationRecorder:
    """Watch a deploy's log for llama.cpp's fit arithmetic and record it.

    The measurement is printed whether or not the plan is accepted, and the
    rejected ones are the valuable ones: they are the cases where the formula
    was wrong. Both reach the host the same way, so this listens to the log
    rather than to the outcome.

    The topology comes out of the measurement itself -- one device row per
    GPU, each naming its hardware -- so nothing has to be threaded in beside
    the placement the planner already carries.
    """

    def __init__(
        self,
        assessment: Any | None,
        *,
        path: Path | None = None,
    ) -> None:
        self._assessment = assessment
        self._path = path
        self._lines: list[str] = []
        self._saved: MemoryCalibration | None = None

    @property
    def saved(self) -> MemoryCalibration | None:
        """The calibration written, if a complete measurement was seen."""

        return self._saved

    def observe(self, line: str) -> MemoryCalibration | None:
        """Buffer a log line, solving and persisting once one run completes."""

        text = (line or "").strip()
        if not text:
            return None
        if _DEVICE_RE.search(text):
            self._lines.append(text)
            return None
        if _PROJECTED_RE.search(text) is None:
            return None
        # The device rows are printed before this one, so the picture is
        # complete here and a later run starts from an empty buffer.
        self._lines.append(text)
        lines, self._lines = self._lines, []
        return self._record(parse_fit_measurement("\n".join(lines)))

    def _record(self, measurement: FitMeasurement) -> MemoryCalibration | None:
        memory = getattr(self._assessment, "memory", None)
        if memory is None:
            return None
        layer_count = getattr(memory, "total_layer_count", None)
        key = str(getattr(self._assessment, "calibration_key", "") or "")
        if not key or not layer_count:
            return None
        shardable_gb = (
            float(getattr(memory, "weights_gb", 0.0))
            + float(getattr(memory, "kv_cache_gb", 0.0))
            + float(getattr(memory, "speculative_gb", 0.0))
        )
        calibration = solve_calibration(
            measurement,
            key=key,
            shardable_gb=shardable_gb,
            layer_count=int(layer_count),
            gpu_type=_gpu_type_from(measurement),
            gpu_count=len(measurement.per_device_used_mib) or 1,
            runtime_id=getattr(self._assessment, "runtime_id", None),
        )
        if calibration is None:
            return None
        save_memory_calibration(calibration, self._path)
        self._saved = calibration
        return calibration


def _gpu_type_from(measurement: FitMeasurement) -> str:
    """Name the hardware a measurement ran on, for provenance only."""

    for description in measurement.device_descriptions:
        cleaned = description.replace("NVIDIA ", "").strip()
        if cleaned:
            return cleaned
    return ""
