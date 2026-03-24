"""Shared helpers for deployment GPU type/count handling."""

from __future__ import annotations

from collections.abc import Sequence

from ..core.modal_gpu import ModalGpuSpec

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


def build_gpu_type_options(gpu_types: Sequence[str | ModalGpuSpec]) -> list[tuple[str, str]]:
    """Build `(label, value)` pairs for GPU type dropdown options."""
    options: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value in gpu_types:
        if isinstance(value, ModalGpuSpec):
            token = normalize_gpu_type(value.value)
            label = _format_gpu_type_label(token, value.price_per_hour_usd)
        else:
            token = normalize_gpu_type(value)
            label = token
        if not token or token in seen:
            continue
        seen.add(token)
        options.append((label, token))
    return options


def _format_gpu_type_label(value: str, price_per_hour_usd: float | None) -> str:
    if price_per_hour_usd is None:
        return value
    return f"{value} (${price_per_hour_usd:.2f}/hr)"
