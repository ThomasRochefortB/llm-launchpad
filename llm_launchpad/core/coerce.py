"""Scalar coercion helpers for untrusted JSON, API, and persisted payloads.

Every helper is total: it returns ``None`` rather than raising, so callers can
parse third-party payloads without wrapping each field in ``try``/``except``.
"""

from __future__ import annotations

import math
from typing import Any


def clean_string(value: Any) -> str:
    """Return ``value`` as stripped text, using ``""`` for anything falsy."""
    return str(value or "").strip()


def optional_str(value: Any) -> str | None:
    """Return ``value`` as stripped text, or ``None`` when it is empty."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def optional_float(value: Any) -> float | None:
    """Return ``value`` as a finite float, or ``None`` when it is not one.

    Booleans are rejected rather than coerced to 1.0/0.0, and NaN is treated as
    a missing value so downstream comparisons stay well-behaved.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(parsed) else parsed


def positive_int(value: Any) -> int | None:
    """Return ``value`` as an integer greater than zero, or ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            return None
    else:
        return None
    return parsed if parsed > 0 else None
