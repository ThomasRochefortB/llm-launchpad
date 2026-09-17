"""Last observed runtime health for scale-to-zero providers.

Modal apps stay listed while their GPU containers scale to zero. Probing
``/metrics`` or ``/health`` to learn whether one of those containers is warm
would itself make it warm, so background fleet refreshes stay passive for
Modal: provider state plus whatever was learned by an explicit, user-requested
check.

This module is that memory. Explicit health checks (Manage -> Check status,
deploy warmup) record here; passive refreshes re-attach the last observation
without touching the network. Prime and Vast bill continuously, so they keep
their existing live probing and never consult this store.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from ..protocol.models import EndpointInfo
from .serving_metrics import usage_key


@dataclass(frozen=True)
class RuntimeHealth:
    """What one explicit check learned, and when it learned it."""

    status: str
    detail: str | None = None
    checked_at_epoch: float = 0.0


_lock = threading.Lock()
_entries: dict[str, RuntimeHealth] = {}


def reset() -> None:
    """Drop every stored observation (the test suite's isolation hook)."""
    with _lock:
        _entries.clear()


def record_explicit_health(
    row: EndpointInfo,
    status: str,
    detail: str | None = None,
    *,
    now: float | None = None,
) -> RuntimeHealth:
    """Remember one user-requested health verdict for later passive refreshes."""
    observation = RuntimeHealth(
        status=(status or "").strip().lower() or "unknown",
        detail=detail,
        checked_at_epoch=time.time() if now is None else now,
    )
    with _lock:
        _entries[usage_key(row)] = observation
    # Keep the row the caller is holding in step so a screen that returns
    # without a fleet refresh still shows what was just learned.
    row.runtime_status = observation.status
    row.runtime_status_detail = observation.detail
    row.runtime_checked_at = observation.checked_at_epoch
    return observation


def get_health(row: EndpointInfo) -> RuntimeHealth | None:
    """Return the last explicit verdict for this endpoint, if any."""
    with _lock:
        return _entries.get(usage_key(row))
