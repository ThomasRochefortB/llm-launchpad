"""Shared Hugging Face download policy for accelerated transfers.

Every provider-managed runtime downloads weights through ``huggingface_hub``,
which now transfers through ``hf-xet``. The knobs are environment variables
read at import time, so they must be set before the interpreter imports the
library -- usually via the container/image env or a fresh subprocess env.

This module is the single place that decides the defaults. Modal entrypoints
stay self-contained for remote execution, so they duplicate the two-line
truth table rather than importing this file; the tests pin both to the same
behavior.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

HF_HUB_ETAG_TIMEOUT_DEFAULT = "30"
HF_HUB_DOWNLOAD_TIMEOUT_DEFAULT = "120"
HF_XET_HIGH_PERFORMANCE_DEFAULT = True

HF_SNAPSHOT_MAX_WORKERS_MIN = 1
HF_SNAPSHOT_MAX_WORKERS_MAX = 64

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def env_value_is_true(value: str | None) -> bool:
    """Return whether a raw env value enables a boolean flag."""
    raw = (value or "").strip()
    if not raw:
        return False
    return raw.lower() in _TRUE_VALUES


def parse_optional_bool(value: Any) -> bool | None:
    """Parse an explicit override, returning None when unset.

    Recognizes the same truthy/falsy tokens as ``huggingface_hub``. Any other
    non-empty value is treated as False to match the library's behavior of
    only honoring the truthy set.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    return False


def _resolve_bool(
    explicit: bool | None,
    env_value: str | None,
    *,
    default: bool,
) -> bool:
    if explicit is not None:
        return explicit
    parsed = parse_optional_bool(env_value)
    return default if parsed is None else parsed


def resolve_xet_transport(
    *,
    disable_xet: bool | None = None,
    high_performance: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Resolve the Xet transport env vars to set.

    Default is Xet enabled with high-performance mode, which is the
    recommended replacement for the legacy ``hf_transfer`` path. An explicit
    ``HF_HUB_DISABLE_XET`` truthy value always wins; high-performance mode is
    only emitted when Xet stays enabled.
    """
    source = os.environ if environ is None else environ
    disable = _resolve_bool(
        disable_xet, source.get("HF_HUB_DISABLE_XET"), default=False
    )
    if disable:
        return {"HF_HUB_DISABLE_XET": "1"}
    perf = _resolve_bool(
        high_performance,
        source.get("HF_XET_HIGH_PERFORMANCE"),
        default=HF_XET_HIGH_PERFORMANCE_DEFAULT,
    )
    if perf:
        return {"HF_XET_HIGH_PERFORMANCE": "1"}
    return {}


def hf_download_env(
    base: Mapping[str, str] | None = None,
    *,
    disable_xet: bool | None = None,
    high_performance: bool | None = None,
    etag_timeout: str | None = None,
    download_timeout: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build download env with timeouts and Xet policy applied.

    Explicit arguments win over ``environ`` (defaulting to ``os.environ``),
    which wins over built-in defaults. Disabling Xet removes any
    high-performance marker so a fallback process cannot mix modes.
    """
    source = os.environ if environ is None else environ
    env = dict(base or {})
    env["HF_HUB_ETAG_TIMEOUT"] = (
        (etag_timeout or "").strip()
        or (source.get("HF_HUB_ETAG_TIMEOUT") or "").strip()
        or HF_HUB_ETAG_TIMEOUT_DEFAULT
    )
    env["HF_HUB_DOWNLOAD_TIMEOUT"] = (
        (download_timeout or "").strip()
        or (source.get("HF_HUB_DOWNLOAD_TIMEOUT") or "").strip()
        or HF_HUB_DOWNLOAD_TIMEOUT_DEFAULT
    )
    transport = resolve_xet_transport(
        disable_xet=disable_xet,
        high_performance=high_performance,
        environ=source,
    )
    if transport.get("HF_HUB_DISABLE_XET"):
        env["HF_HUB_DISABLE_XET"] = "1"
        env.pop("HF_XET_HIGH_PERFORMANCE", None)
    else:
        env.pop("HF_HUB_DISABLE_XET", None)
        if transport.get("HF_XET_HIGH_PERFORMANCE"):
            env["HF_XET_HIGH_PERFORMANCE"] = "1"
        else:
            env.pop("HF_XET_HIGH_PERFORMANCE", None)
    return env


def describe_transport(env: Mapping[str, str]) -> str:
    """Return a short transport label for download logs."""
    if env_value_is_true(env.get("HF_HUB_DISABLE_XET")):
        return "http"
    if env_value_is_true(env.get("HF_XET_HIGH_PERFORMANCE")):
        return "xet-high-performance"
    return "xet"


def clamp_max_workers(value: Any, default: int) -> int:
    """Coerce a worker count into the supported range, falling back to default."""
    parsed: int | None = None
    if isinstance(value, bool):
        parsed = None
    elif isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if text:
            try:
                parsed = int(text)
            except ValueError:
                parsed = None
    if parsed is None:
        parsed = default
    return max(HF_SNAPSHOT_MAX_WORKERS_MIN, min(HF_SNAPSHOT_MAX_WORKERS_MAX, parsed))


def allocated_file_size(path: str | Path) -> int:
    """Return bytes actually allocated for a file, capped at logical size.

    Xet preallocates sparse files whose logical length is not progress, so
    progress accounting must use allocated blocks. Falls back to logical size
    on filesystems without ``st_blocks`` and returns 0 for missing files.
    """
    try:
        stat = Path(path).stat()
    except OSError:
        return 0
    logical = max(0, int(stat.st_size))
    blocks = getattr(stat, "st_blocks", None)
    if not isinstance(blocks, int) or blocks < 0:
        return logical
    try:
        allocated = int(blocks) * 512
    except (OverflowError, ValueError):
        return logical
    return max(0, min(logical, allocated))


def classify_download_failure(output: str) -> str | None:
    """Classify a download failure as non-retryable, or None when retryable.

    Retrying through the other transport only helps for transport-level
    failures. Authentication, missing repos/revisions, and disk exhaustion
    fail identically on both, so callers should surface those immediately.
    """
    text = (output or "").lower()
    if not text.strip():
        return None
    if any(
        marker in text
        for marker in (
            "401",
            "403",
            "unauthorized",
            "forbidden",
            "invalid token",
            "invalid hf_token",
            "gated repo",
            "access denied",
        )
    ):
        return "auth"
    if any(
        marker in text
        for marker in (
            "404",
            "repository not found",
            "repo not found",
            "revision not found",
            "no such revision",
            "entry not found",
            "does not exist",
        )
    ):
        return "not_found"
    if any(
        marker in text
        for marker in (
            "no space left",
            "disk quota",
            "enospc",
            "no space on device",
            "volume out of space",
        )
    ):
        return "no_space"
    return None
