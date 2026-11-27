"""Bounded GGUF metadata inspection for serving capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re
import struct
from typing import Any
from urllib.parse import quote

from ..gguf_files import is_target_gguf


GGUF_METADATA_CHUNK_BYTES = 1024 * 1024
GGUF_METADATA_MAX_BYTES = 32 * 1024 * 1024
GGUF_PLE_HEADER_CHUNK_BYTES = 64 * 1024
# Upstream llama.cpp b12233+ assigns type_id 43 to GGML_TYPE_F8_E4M3, while
# Unsloth's PR #30817 DeepSeek-V4.1 MXFP8 Engram shards also wrote type_id 43
# with block-scaled (33/32 bytes per element) data.
GGML_TYPE_F8_E4M3_ID = 43
PLE_TENSOR_NAME = "per_layer_token_embd.weight"

_GGUF_MAGIC = b"GGUF"
_STRING_LENGTH = struct.Struct("<Q")
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


@dataclass(frozen=True)
class GgufServingMetadata:
    """Architecture scalars required for context and KV-cache planning."""

    architecture: str | None = None
    context_length: int | None = None
    block_count: int | None = None
    embedding_length: int | None = None
    attention_head_count: int | None = None
    attention_head_count_kv: int | None = None
    attention_key_length: int | None = None
    attention_value_length: int | None = None
    source_file: str | None = None
    attention_head_count_kv_by_layer: tuple[int, ...] = ()
    attention_key_length_mla: int | None = None
    attention_value_length_mla: int | None = None
    attention_kv_lora_rank: int | None = None
    kda_head_dim: int | None = None
    ssm_conv_kernel: int | None = None
    indexer_key_length: int | None = None
    indexer_kpool: int | None = None
    nextn_predict_layers: int = 0
    full_attention_interval: int | None = None
    recurrent_layers: tuple[bool, ...] = ()
    ssm_inner_size: int | None = None
    ssm_state_size: int | None = None
    ssm_group_count: int | None = None
    sliding_window: int | None = None
    sliding_window_pattern: tuple[bool, ...] = ()
    sliding_window_interval: int | None = None
    shared_kv_layers: int = 0
    attention_key_length_swa: int | None = None
    attention_value_length_swa: int | None = None
    expert_count: int = 0
    feed_forward_length: int | None = None
    feed_forward_length_by_layer: tuple[int, ...] = ()
    shortconv_cache_length: int | None = None


@dataclass(frozen=True)
class GgufPleTensorInfo:
    """Tensor descriptor for a single-tensor per-layer embedding (Engram) shard."""

    name: str
    dims: tuple[int, ...]
    type_id: int
    source_file: str | None = None

    @property
    def num_elements(self) -> int:
        return math.prod(self.dims) if self.dims else 0

    def is_mxfp8_type43_collision(self, size_bytes: int | None = None) -> bool:
        """Return True when this PLE shard collides with upstream F8_E4M3 (type 43).

        Plain F8_E4M3 stores 1 byte per element, whereas PR #30817 MXFP8 stores
        one E8M0 scale byte per 32 elements (33/32 bytes per element). Checking
        the file size when known avoids false positives against real F8_E4M3.
        """
        if self.name != PLE_TENSOR_NAME or self.type_id != GGML_TYPE_F8_E4M3_ID:
            return False
        if size_bytes is None or self.num_elements <= 0:
            return True
        return size_bytes >= (self.num_elements * 33) // 32


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
        if not is_target_gguf(filename):
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
    next_chunk_bytes = chunk_bytes
    try:
        while len(payload) < max_bytes:
            end = min(len(payload) + next_chunk_bytes, max_bytes) - 1
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
                # Large tokenizers need several MiB. Grow geometrically to
                # avoid fetching and reparsing the table one MiB at a time.
                next_chunk_bytes *= 2
                continue
    except Exception as exc:
        return GgufMtpCapability.unknown(
            f"Could not inspect {filename!r}: {exc}"
        )
    return GgufMtpCapability.unknown(
        f"GGUF metadata did not fit within the {max_bytes // (1024 * 1024)} MiB inspection limit."
    )


def parse_gguf_serving_metadata(
    data: bytes,
    source_file: str | None = None,
) -> GgufServingMetadata:
    """Parse the complete GGUF metadata table for serving-relevant scalars."""

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

    values: dict[str, Any] = {}
    architecture: str | None = None
    suffixes = {
        "context_length": "context_length",
        "block_count": "block_count",
        "embedding_length": "embedding_length",
        "attention.head_count": "attention_head_count",
        "attention.head_count_kv": "attention_head_count_kv",
        "attention.key_length": "attention_key_length",
        "attention.value_length": "attention_value_length",
        "attention.key_length_mla": "attention_key_length_mla",
        "attention.value_length_mla": "attention_value_length_mla",
        "attention.kv_lora_rank": "attention_kv_lora_rank",
        "kda.head_dim": "kda_head_dim",
        "ssm.conv_kernel": "ssm_conv_kernel",
        "attention.indexer.key_length": "indexer_key_length",
        "attention.indexer.kpool": "indexer_kpool",
        "nextn_predict_layers": "nextn_predict_layers",
        "full_attention_interval": "full_attention_interval",
        "attention.recurrent_layers": "recurrent_layers",
        "ssm.inner_size": "ssm_inner_size",
        "ssm.state_size": "ssm_state_size",
        "ssm.group_count": "ssm_group_count",
        "attention.sliding_window": "sliding_window",
        "attention.sliding_window_pattern": "sliding_window_interval",
        "attention.shared_kv_layers": "shared_kv_layers",
        "attention.key_length_swa": "attention_key_length_swa",
        "attention.value_length_swa": "attention_value_length_swa",
        "expert_count": "expert_count",
        "feed_forward_length": "feed_forward_length",
        "shortconv.l_cache": "shortconv_cache_length",
    }
    offset = 24
    for _ in range(metadata_count):
        key, offset = _read_string(data, offset, max_length=1024 * 1024)
        if key is None:
            raise _InvalidGguf("GGUF metadata key could not be decoded")
        value_type = _unpack(data, offset, "<I")[0]
        offset += 4
        matched = next(
            (
                field
                for suffix, field in suffixes.items()
                if key == suffix or key.endswith(f".{suffix}")
            ),
            None,
        )
        capture = key == "general.architecture" or matched is not None
        value, offset = _read_value(data, offset, value_type, capture=capture)
        if key == "general.architecture" and isinstance(value, str):
            architecture = value.strip().casefold() or None
        elif matched is not None:
            values[key.casefold()] = value

    # Only the target architecture may supply these fields (not a vision tower).
    scalars: dict[str, int] = {}
    layer_sizes: dict[str, tuple[int, ...]] = {}
    patterns: dict[str, tuple[bool, ...]] = {}
    for suffix, field in suffixes.items():
        fallback = None if field == "nextn_predict_layers" else values.get(suffix)
        value = values.get(f"{architecture}.{suffix}", fallback)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            if field != "recurrent_layers" and (
                value > 0 or field in {
                    "attention_head_count_kv", "feed_forward_length", "nextn_predict_layers", "sliding_window_interval",
                }
            ):
                scalars[field] = value
        elif (
            field in {"attention_head_count_kv", "feed_forward_length"} and isinstance(value, tuple)
            and value and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in value)
        ):
            layer_sizes[field] = value
        elif (
            field in {"recurrent_layers", "sliding_window_interval"}
            and isinstance(value, tuple) and value
            and all(isinstance(entry, bool) for entry in value)
        ):
            patterns[field] = value

    return GgufServingMetadata(
        architecture=architecture,
        **scalars,
        recurrent_layers=patterns.get("recurrent_layers", ()),
        sliding_window_pattern=patterns.get("sliding_window_interval", ()),
        attention_head_count_kv_by_layer=layer_sizes.get("attention_head_count_kv", ()),
        feed_forward_length_by_layer=layer_sizes.get("feed_forward_length", ()),
        source_file=source_file,
    )


def fetch_gguf_serving_metadata(
    repo_id: str,
    siblings: Any,
    *,
    revision: str | None = None,
    quant: str | None = None,
    filename: str | None = None,
    chunk_bytes: int = GGUF_METADATA_CHUNK_BYTES,
    max_bytes: int = GGUF_METADATA_MAX_BYTES,
    strict: bool = False,
) -> GgufServingMetadata | None:
    """Fetch and parse bounded GGUF header ranges for memory planning."""

    filename = filename or select_target_gguf_file(siblings, quant=quant)
    if filename is None or chunk_bytes <= 0 or max_bytes <= 0:
        return None
    payload = bytearray()
    next_chunk_bytes = chunk_bytes
    try:
        while len(payload) < max_bytes:
            end = min(len(payload) + next_chunk_bytes, max_bytes) - 1
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
                return parse_gguf_serving_metadata(bytes(payload), source_file=filename)
            except _NeedMoreData:
                # Large tokenizers need several MiB. Grow geometrically to
                # avoid fetching and reparsing the table one MiB at a time.
                next_chunk_bytes *= 2
                continue
    except Exception:
        if strict:
            raise
    return None


def parse_gguf_ple_tensor_info(
    data: bytes,
    source_file: str | None = None,
) -> GgufPleTensorInfo | None:
    """Parse a GGUF shard header and return PLE tensor info if it is PLE-only.

    Returns ``None`` immediately when the shard holds more (or fewer) than one
    tensor, or when its lone tensor is not ``per_layer_token_embd.weight``.
    """

    if len(data) < 24:
        raise _NeedMoreData
    if data[:4] != _GGUF_MAGIC:
        raise _InvalidGguf("Missing GGUF magic")
    version = _unpack(data, 4, "<I")[0]
    if version not in {2, 3}:
        raise _InvalidGguf(f"Unsupported GGUF version {version}")
    tensor_count = _unpack(data, 8, "<Q")[0]
    if tensor_count != 1:
        return None
    metadata_count = _unpack(data, 16, "<Q")[0]
    if metadata_count > 10_000_000:
        raise _InvalidGguf("Implausible GGUF metadata count")

    offset = 24
    for _ in range(metadata_count):
        key, offset = _read_string(data, offset, max_length=1024 * 1024)
        if key is None:
            raise _InvalidGguf("GGUF metadata key could not be decoded")
        value_type = _unpack(data, offset, "<I")[0]
        offset += 4
        _, offset = _read_value(data, offset, value_type, capture=False)

    tensor_name, offset = _read_string(data, offset, max_length=1024 * 1024)
    if tensor_name is None:
        raise _InvalidGguf("GGUF tensor name could not be decoded")
    n_dims = _unpack(data, offset, "<I")[0]
    offset += 4
    if n_dims > 8:
        raise _InvalidGguf(f"Implausible GGUF tensor rank {n_dims}")
    dims = tuple(int(dim) for dim in _unpack(data, offset, f"<{n_dims}Q")) if n_dims else ()
    offset += 8 * n_dims
    type_id = int(_unpack(data, offset, "<I")[0])
    # Verify the 8-byte tensor data offset is present before trusting the entry.
    _unpack(data, offset + 4, "<Q")
    if tensor_name != PLE_TENSOR_NAME:
        return None
    return GgufPleTensorInfo(
        name=tensor_name,
        dims=dims,
        type_id=type_id,
        source_file=source_file,
    )


def fetch_gguf_ple_tensor_info(
    repo_id: str,
    filename: str,
    *,
    revision: str | None = None,
    chunk_bytes: int = GGUF_PLE_HEADER_CHUNK_BYTES,
    max_bytes: int = GGUF_METADATA_MAX_BYTES,
) -> GgufPleTensorInfo | None:
    """Inspect one GGUF shard through bounded HTTP range requests for PLE info."""

    if not filename or chunk_bytes <= 0 or max_bytes <= 0:
        return None
    payload = bytearray()
    next_chunk_bytes = chunk_bytes
    try:
        while len(payload) < max_bytes:
            end = min(len(payload) + next_chunk_bytes, max_bytes) - 1
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
                return parse_gguf_ple_tensor_info(bytes(payload), source_file=filename)
            except _NeedMoreData:
                next_chunk_bytes *= 2
                continue
    except Exception:
        return None
    return None


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
        if (
            match is None
            or int(match.group(1)) != start
            or not start <= int(match.group(2)) <= end
        ):
            raise RuntimeError("Hugging Face returned an invalid Content-Range")
        expected_max = int(match.group(2)) - start + 1
        chunks: list[bytes] = []
        received = 0
        for chunk in response.iter_content(chunk_size=min(expected_max, 64 * 1024)):
            if not chunk:
                continue
            received += len(chunk)
            if received > expected_max:
                raise RuntimeError("Hugging Face returned more than the requested range")
            chunks.append(chunk)
        if received != expected_max:
            raise RuntimeError("Hugging Face returned a truncated range")
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
        value, next_offset = _read_string(
            data,
            offset,
            max_length=GGUF_METADATA_MAX_BYTES,
            decode=capture,
        )
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
            # Tokenizer arrays may contain hundreds of thousands of strings.
            # Skip their bytes directly. Even empty strings need length words.
            available = len(data)
            if length * _STRING_LENGTH.size > available - offset:
                raise _NeedMoreData
            unpack_length = _STRING_LENGTH.unpack_from
            for _ in range(length):
                if offset + 8 > available:
                    raise _NeedMoreData
                size, = unpack_length(data, offset)
                if size > GGUF_METADATA_MAX_BYTES:
                    raise _InvalidGguf("GGUF string exceeds the metadata inspection limit")
                offset += 8 + size
            if offset > available:
                raise _NeedMoreData
            return None, offset
        size = _GGUF_SCALAR_SIZES.get(element_type)
        if size is None:
            raise _InvalidGguf(f"Unknown GGUF array element type {element_type}")
        if capture:
            if length > 8192:
                raise _InvalidGguf("Serving metadata array exceeds the layer limit")
            values = []
            for _ in range(length):
                value, offset = _read_value(data, offset, element_type, capture=True)
                values.append(value)
            return tuple(values), offset
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
