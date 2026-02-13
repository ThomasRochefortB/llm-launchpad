"""Shared helpers for deployment GPU type/count handling."""

from __future__ import annotations


DEFAULT_GPU_TYPE = "A100-80GB"
DEFAULT_GPU_COUNT = 1


def normalize_gpu_type(value: str) -> str:
    return value.strip().upper()


def parse_gpu_count(value: str, default: int = DEFAULT_GPU_COUNT) -> int:
    raw = value.strip()
    if not raw:
        return default
    try:
        count = int(raw)
    except ValueError:
        return default
    return count if count > 0 else default


def parse_gpu_config(value: str) -> tuple[str, int]:
    """Split GPU_CONFIG into (gpu_type, gpu_count)."""
    raw = value.strip()
    if not raw:
        return DEFAULT_GPU_TYPE, DEFAULT_GPU_COUNT
    if ":" not in raw:
        token = normalize_gpu_type(raw)
        return token or DEFAULT_GPU_TYPE, DEFAULT_GPU_COUNT

    gpu_type_raw, count_raw = raw.rsplit(":", 1)
    gpu_type = normalize_gpu_type(gpu_type_raw) or DEFAULT_GPU_TYPE
    return gpu_type, parse_gpu_count(count_raw, default=DEFAULT_GPU_COUNT)


def build_gpu_type_options(gpu_types: list[str]) -> list[str]:
    """Build GPU type dropdown options from raw values."""
    options: list[str] = []
    seen: set[str] = set()
    for value in gpu_types:
        token = normalize_gpu_type(value)
        if not token or token in seen:
            continue
        seen.add(token)
        options.append(token)
    return options


def build_gpu_config(gpu_type: str, gpu_count: int) -> str:
    return f"{normalize_gpu_type(gpu_type)}:{gpu_count}"
