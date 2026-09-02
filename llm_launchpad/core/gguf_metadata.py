"""Bounded GGUF metadata inspection for serving capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
import struct
from typing import Any
from urllib.parse import quote


GGUF_METADATA_CHUNK_BYTES = 1024 * 1024
GGUF_METADATA_MAX_BYTES = 32 * 1024 * 1024

_GGUF_MAGIC = b"GGUF"
_GGUF_SCALAR_SIZES = {
    0: 1,  # uint8
    1: 1,  # int8
    2: 2,  # uint16
    3: 2,  # int16
    4: 4,  # uint32
    5: 4,  # int32
    6: 4,  # float32
    7: 1,  # bool
    10: 8,  # uint64
    11: 8,  # int64
    12: 8,  # float64
}
_GGUF_STRING = 8
_GGUF_ARRAY = 9
_CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(?:\d+|\*)$", re.IGNORECASE)
_AUXILIARY_GGUF_MARKERS = ("mmproj", "imatrix", "dspark", "draft", "eagle")


class GgufMtpStatus(str, Enum):
    """Confidence of the MTP capability found in a GGUF header."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GgufMtpCapability:
    """MTP metadata result for one representative target GGUF."""

    status: GgufMtpStatus
    nextn_predict_layers: int | None = None
    source_file: str | None = None
    message: str = ""

    @classmethod
    def unknown(cls, message: str = "MTP metadata was not inspected.") -> GgufMtpCapability:
        return cls(status=GgufMtpStatus.UNKNOWN, message=message)


class _NeedMoreData(Exception):
    pass


class _InvalidGguf(ValueError):
    pass


def select_target_gguf_file(siblings: Any, quant: str | None = None) -> str | None:
    """Choose the first target-model GGUF, excluding known auxiliary artifacts."""

    candidates: list[str] = []
    for sibling in siblings if isinstance(siblings, list) else ():
        if isinstance(sibling, dict):
            filename = str(sibling.get("rfilename", "") or "").strip()
        else:
            filename = str(getattr(sibling, "rfilename", "") or "").strip()
        if not filename or not filename.casefold().endswith(".gguf"):
            continue
        lowered = filename.casefold()
        path_parts = tuple(part for part in lowered.split("/") if part)
        if any(marker in lowered for marker in _AUXILIARY_GGUF_MARKERS):
            continue
        if any(part == "mtp" or part.startswith("mtp-") for part in path_parts):
            continue
        candidates.append(filename)
    if not candidates:
        return None

    normalized_quant = _normalized_token(quant or "")

    def _sort_key(filename: str) -> tuple[int, int, str]:
        normalized_filename = _normalized_token(filename)
        quant_mismatch = int(
            bool(normalized_quant) and normalized_quant not in normalized_filename
        )
        first_shard = int("00001-of-" not in filename.casefold())
        return (quant_mismatch, first_shard, filename.casefold())

    return min(candidates, key=_sort_key)


def parse_gguf_mtp_metadata(data: bytes, source_file: str | None = None) -> GgufMtpCapability:
    """Parse enough GGUF metadata to prove embedded MTP support or absence.

    ``_NeedMoreData`` is intentionally raised for a valid prefix so callers can
    fetch another bounded range without conflating truncation with absence.
    """

    if len(data) < 24:
        raise _NeedMoreData
    if data[:4] != _GGUF_MAGIC:
        raise _InvalidGguf("Missing GGUF magic")
    version = _unpack(data, 4, "<I")[0]
    if version not in {2, 3}:
        raise _InvalidGguf(f"Unsupported GGUF version {version}")
    metadata_count = _unpack(data, 16, "<Q")[0]
    if metadata_count > 10_000_000:
        raise _InvalidGguf("Implausible GGUF metadata count")

    offset = 24
    architecture: str | None = None
    nextn_by_architecture: dict[str, int] = {}
    for _ in range(metadata_count):
        key, offset = _read_string(data, offset, max_length=1024 * 1024)
        if key is None:
            raise _InvalidGguf("GGUF metadata key could not be decoded")
        value_type = _unpack(data, offset, "<I")[0]
        offset += 4
        capture = key == "general.architecture" or key.endswith(
            ".nextn_predict_layers"
        )
        value, offset = _read_value(data, offset, value_type, capture=capture)
        if key == "general.architecture" and isinstance(value, str):
            architecture = value.strip().casefold() or None
        elif key.endswith(".nextn_predict_layers"):
            prefix = key[: -len(".nextn_predict_layers")].strip().casefold()
            if prefix and isinstance(value, int) and not isinstance(value, bool):
                nextn_by_architecture[prefix] = value

        layers = nextn_by_architecture.get(architecture or "")
        if architecture and layers is not None and layers > 0:
            return GgufMtpCapability(
                status=GgufMtpStatus.SUPPORTED,
                nextn_predict_layers=layers,
                source_file=source_file,
                message=(
                    f"{architecture}.nextn_predict_layers={layers} is embedded in "
                    "the target GGUF."
                ),
            )

    layers = nextn_by_architecture.get(architecture or "")
    if layers is not None and layers > 0:
        return GgufMtpCapability(
            status=GgufMtpStatus.SUPPORTED,
            nextn_predict_layers=layers,
            source_file=source_file,
        )
    return GgufMtpCapability(
        status=GgufMtpStatus.UNSUPPORTED,
        nextn_predict_layers=0,
        source_file=source_file,
        message="The complete GGUF metadata table has no embedded MTP heads.",
    )


def fetch_gguf_mtp_capability(
    repo_id: str,
    siblings: Any,
    *,
    revision: str | None = None,
    quant: str | None = None,
    chunk_bytes: int = GGUF_METADATA_CHUNK_BYTES,
    max_bytes: int = GGUF_METADATA_MAX_BYTES,
) -> GgufMtpCapability:
    """Inspect a target GGUF through bounded HTTP range requests."""

    filename = select_target_gguf_file(siblings, quant=quant)
    if filename is None:
        return GgufMtpCapability.unknown(
            "No non-auxiliary target GGUF was available for MTP inspection."
        )
    if chunk_bytes <= 0 or max_bytes <= 0:
        return GgufMtpCapability.unknown("Invalid GGUF metadata byte limit.")

    payload = bytearray()
    try:
        while len(payload) < max_bytes:
            end = min(len(payload) + chunk_bytes, max_bytes) - 1
            chunk = _fetch_hf_file_range(
                repo_id,
                filename,
                revision=revision,
                start=len(payload),
                end=end,
            )
            if not chunk:
                break
            payload.extend(chunk)
            try:
                return parse_gguf_mtp_metadata(bytes(payload), source_file=filename)
            except _NeedMoreData:
                continue
    except Exception as exc:
        return GgufMtpCapability.unknown(
            f"Could not inspect {filename!r}: {exc}"
        )
    return GgufMtpCapability.unknown(
        f"GGUF metadata did not fit within the {max_bytes // (1024 * 1024)} MiB inspection limit."
    )


def _fetch_hf_file_range(
    repo_id: str,
    filename: str,
    *,
    revision: str | None,
    start: int,
    end: int,
) -> bytes:
    import requests

    selected_revision = (revision or "main").strip() or "main"
    url = (
        f"https://huggingface.co/{quote(repo_id.strip(), safe='/')}"
        f"/resolve/{quote(selected_revision, safe='')}/{quote(filename, safe='/')}"
    )
    headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
    try:
        from huggingface_hub import get_token

        token = (get_token() or "").strip()
    except Exception:
        token = ""
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(headers=headers, stream=True, timeout=15.0, url=url)
    try:
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError("Hugging Face ignored the bounded Range request")
        content_range = str(response.headers.get("Content-Range", "") or "").strip()
        match = _CONTENT_RANGE_RE.match(content_range)
        if match is None or int(match.group(1)) != start or int(match.group(2)) > end:
            raise RuntimeError("Hugging Face returned an invalid Content-Range")
        expected_max = end - start + 1
        chunks: list[bytes] = []
        received = 0
        for chunk in response.iter_content(chunk_size=min(expected_max, 64 * 1024)):
            if not chunk:
                continue
            received += len(chunk)
            if received > expected_max:
                raise RuntimeError("Hugging Face returned more than the requested range")
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        response.close()


def _read_value(
    data: bytes,
    offset: int,
    value_type: int,
    *,
    capture: bool,
) -> tuple[Any, int]:
    if value_type == _GGUF_STRING:
        value, next_offset = _read_string(data, offset, max_length=GGUF_METADATA_MAX_BYTES)
        return (value if capture else None, next_offset)
    if value_type == _GGUF_ARRAY:
        element_type = _unpack(data, offset, "<I")[0]
        length = _unpack(data, offset + 4, "<Q")[0]
        offset += 12
        if length > 100_000_000:
            raise _InvalidGguf("Implausible GGUF array length")
        if element_type == _GGUF_ARRAY:
            raise _InvalidGguf("Nested GGUF arrays are invalid")
        if element_type == _GGUF_STRING:
            for _ in range(length):
                _, offset = _read_string(
                    data,
                    offset,
                    max_length=GGUF_METADATA_MAX_BYTES,
                    decode=False,
                )
            return None, offset
        size = _GGUF_SCALAR_SIZES.get(element_type)
        if size is None:
            raise _InvalidGguf(f"Unknown GGUF array element type {element_type}")
        return None, _advance(data, offset, length * size)

    size = _GGUF_SCALAR_SIZES.get(value_type)
    if size is None:
        raise _InvalidGguf(f"Unknown GGUF metadata type {value_type}")
    next_offset = _advance(data, offset, size)
    if not capture:
        return None, next_offset
    formats = {
        0: "<B",
        1: "<b",
        2: "<H",
        3: "<h",
        4: "<I",
        5: "<i",
        6: "<f",
        7: "<?",
        10: "<Q",
        11: "<q",
        12: "<d",
    }
    return _unpack(data, offset, formats[value_type])[0], next_offset


def _read_string(
    data: bytes,
    offset: int,
    *,
    max_length: int,
    decode: bool = True,
) -> tuple[str | None, int]:
    length = _unpack(data, offset, "<Q")[0]
    offset += 8
    if length > max_length:
        raise _InvalidGguf("GGUF string exceeds the metadata inspection limit")
    next_offset = _advance(data, offset, length)
    if not decode:
        return None, next_offset
    try:
        return data[offset:next_offset].decode("utf-8"), next_offset
    except UnicodeDecodeError as exc:
        raise _InvalidGguf("GGUF metadata contains invalid UTF-8") from exc


def _advance(data: bytes, offset: int, length: int) -> int:
    if length < 0 or offset < 0 or offset + length > len(data):
        raise _NeedMoreData
    return offset + length


def _unpack(data: bytes, offset: int, format_string: str) -> tuple[Any, ...]:
    size = struct.calcsize(format_string)
    if offset < 0 or offset + size > len(data):
        raise _NeedMoreData
    return struct.unpack_from(format_string, data, offset)


def _normalized_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())

