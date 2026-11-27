"""Record the serving runtime's allocation projection for one exact placement."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

from ..protocol.enums import ComputeProvider
from ..protocol.models import LLAMACPP_FIT_START_MARKER, PlacementAssessment, RuntimeTuning, ServingRequirements
from .config import SETTINGS_DIR
from .runtime_support import load_llamacpp_support_manifest

CALIBRATION_SCHEMA_VERSION = 4
CALIBRATION_CACHE_PATH = SETTINGS_DIR / "memory_projections"

_MIB = 1024**2
_DEVICE_RE = re.compile(
    r"-\s+(?P<device>CUDA\d+)\s*\((?P<description>[^)]*)\)\s*:\s*"
    r"(?P<total>\d+)\s+total,\s*(?P<used>\d+)\s+used,\s*"
    r"(?P<free>-?\d+)\s+free\s+vs\.\s+target\s+of\s+(?P<target>\d+)"
)
_DEVICE_INFO_RE = re.compile(r"-\s+(?P<device>CUDA\d+)\s*:\s*(?P<name>.+?)\s+\(\d+ MiB,")
_BUILD_RE = re.compile(r"common_params_print_info: build \d+ \(([0-9a-f]{7,40})\)")
_PROJECTED_RE = re.compile(
    r"projected to use\s+(?P<used>\d+)\s+MiB of device memory vs\.\s*"
    r"(?P<free>\d+)\s+MiB of free device memory"
)


@dataclass(frozen=True)
class MemoryCalibration:
    """Projected bytes by device before adding Launchpad's headroom."""

    key: str
    per_device_bytes: tuple[int, ...]
    projected_at: str


def calibration_key(
    *,
    model_id: str,
    revision: str | None,
    quant: str | None,
    runtime_id: str | None,
    requirements: ServingRequirements,
    tuning: RuntimeTuning,
    gpu_type: str,
    gpu_count: int,
    entrypoint: str | None = None,
    provider: ComputeProvider | None = None,
    capacity_bytes: int | None = None,
) -> str:
    """Identify a pinned model, its runtime settings, and the GPU topology."""
    if not revision or not runtime_id:
        return ""
    payload = {
        "schema": CALIBRATION_SCHEMA_VERSION,
        "model_id": model_id,
        "revision": revision,
        "quant": quant,
        "runtime_id": runtime_id,
        "context_tokens": requirements.context_tokens,
        "gpu_only": requirements.gpu_only,
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "entrypoint": entrypoint,
        "provider": provider,
        "capacity_bytes": capacity_bytes,
        "tuning": asdict(tuning),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _save_memory_calibration(calibration: MemoryCalibration, directory: Path) -> None:
    """Persist one observation atomically without rewriting other jobs' evidence."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{calibration.key}.json"
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as output:
            temporary = Path(output.name)
            json.dump(asdict(calibration), output)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_memory_calibration(key: str, path: Path | None = None) -> MemoryCalibration | None:
    """Load the projection for this exact configuration if one was recorded."""
    if not key:
        return None
    directory = path if path is not None else CALIBRATION_CACHE_PATH
    try:
        entry = json.loads((directory / f"{key}.json").read_text(encoding="utf-8"))
        used = tuple(entry["per_device_bytes"])
        if entry["key"] == key and used and all(type(value) is int and value > 0 for value in used):
            return MemoryCalibration(key, used, str(entry["projected_at"]))
    except (KeyError, OSError, TypeError, ValueError):
        return None
    return None


class FitCalibrationRecorder:
    """Read the first allocation projection after the matching server's start marker."""

    def __init__(self, assessment: PlacementAssessment | None, *, path: Path | None = None) -> None:
        self._assessment = assessment
        self._path = path
        self._devices: list[tuple[str, int, int]] = []
        self._device_names: dict[str, str] = {}
        self._active = False
        self._build: str | None = None
        self.headroom_shortfall_mib: int | None = None
        self.topology_verified = False

    def observe(self, line: str) -> MemoryCalibration | None:
        """Record a complete device vector, including refused fits."""
        assessment = self._assessment
        if assessment is None or not assessment.calibration_key:
            return None
        if LLAMACPP_FIT_START_MARKER in line:
            self._active = line.partition(LLAMACPP_FIT_START_MARKER)[2].strip() == assessment.fingerprint
            self._devices.clear()
            self._device_names.clear()
            self._build = None
            if self._active:
                self.headroom_shortfall_mib = None
                self.topology_verified = False
            return None
        if not self._active:
            return None
        if "failed to measure the memory of the extra model, fitting without it" in line:
            self._active = False
            return None
        if build := _BUILD_RE.search(line):
            self._build = build[1]
            return None
        info = _DEVICE_INFO_RE.search(line)
        if info is not None:
            self._device_names[info["device"]] = info["name"]
            return None
        device = _DEVICE_RE.search(line)
        if device is not None:
            if int(device["target"]) != assessment.tuning.fit_target_mib:
                self._active = False
                return None
            self._device_names[device["device"]] = device["description"]
            self._devices.append((device["device"], int(device["used"]), int(device["free"])))
            return None
        projection = _PROJECTED_RE.search(line)
        if projection is None:
            return None
        self._active = False
        count = len(assessment.memory.per_device_required_gb)
        total_mib = int(projection["used"])
        # b12464 only prints individual rows for multi-GPU configurations.
        if count == 1 and not self._devices:
            self._devices.append(("CUDA0", total_mib, int(projection["free"]) - total_mib))
        if {name for name, _, _ in self._devices} != {f"CUDA{i}" for i in range(count)}:
            return None
        if len(self._devices) != count or count == 0:
            return None
        used = tuple(value for _, value, _ in sorted(self._devices, key=lambda row: int(row[0][4:])))
        # Per-device totals and the aggregate have independent MiB rounding.
        if min(used) <= 0 or not sum(used) <= total_mib < sum(used) + count:
            return None
        # The log rounds down positive free memory and truncates negative
        # values towards zero. Allow 1 MiB for the single-GPU subtraction.
        least_free = min(free for _, _, free in self._devices)
        self.headroom_shortfall_mib = max(0, assessment.tuning.fit_target_mib - least_free - (count == 1))
        # A provider can upgrade hardware, or a custom server binary can
        # change allocation. Use the projection only for what actually ran.
        manifest = load_llamacpp_support_manifest()
        if assessment.runtime_id != manifest.runtime_id:
            manifest = load_llamacpp_support_manifest("glm5next")
        self.topology_verified = bool(
            assessment.runtime_id == manifest.runtime_id
            and self._build and manifest.source_revision.startswith(self._build)
            and assessment.gpu_type and all(
                _matches_gpu(assessment.gpu_type, self._device_names.get(name, ""))
                for name, _, _ in self._devices
            )
        )
        if not self.topology_verified:
            return None
        calibration = MemoryCalibration(
            key=assessment.calibration_key,
            per_device_bytes=tuple((value + 1) * _MIB for value in used),
            projected_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        try:
            _save_memory_calibration(calibration, self._path if self._path is not None else CALIBRATION_CACHE_PATH)
        except OSError:
            return None
        return calibration


def _matches_gpu(requested: str, actual: str) -> bool:
    expected = re.sub(r"[^A-Z0-9]+", " ", requested.upper()).strip()
    observed = re.sub(r"[^A-Z0-9]+", " ", actual.upper()).strip()
    observed = re.sub(r"^(?:NVIDIA |GEFORCE |TESLA )+", "", observed)
    # Catalog names can put the memory suffix where CUDA uses an SXM/PCIe label.
    if expected == "A100":
        expected = "A100 40GB"
    if expected == "H100":
        expected = "H100 80GB"
    if expected.startswith(("A100", "H100", "H200")) and " " in expected:
        family, capacity = expected.split(" ", 1)
        if re.fullmatch(r"\d+GB", capacity):
            if expected == "H100 80GB" and observed in {"H100 PCIE", "H100 SXM5"}:
                return True
            return observed.startswith(family + " ") and capacity in observed.split()
    if expected == "RTX PRO 6000":
        return observed == expected or observed.startswith(expected + " BLACKWELL")
    known_capacity = {"L40S 48GB": "L40S", "L4 24GB": "L4", "A10 24GB": "A10"}
    expected = known_capacity.get(expected, expected)
    return bool(expected) and (observed == expected or observed.startswith(expected + " "))
