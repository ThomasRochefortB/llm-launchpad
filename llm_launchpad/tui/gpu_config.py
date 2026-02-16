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
