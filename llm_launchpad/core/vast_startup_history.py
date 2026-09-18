"""Measured Vast host startup history for evidence-based host selection.

Advertised ``inet_down`` bandwidth is a poor predictor of how long a rental
takes to reach SSH and then a healthy endpoint: host-side image pulls, Vast
provisioning, and SSH-server installation dominate, and none of them scale
with the advertised link. This module records per-machine outcomes of real
deploys in local state and ranks candidate offers by measured startup time,
falling back to price when no machine has evidence yet.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from ..protocol.models import VastOffer
from .config import SETTINGS_DIR

VAST_STARTUP_HISTORY_PATH = SETTINGS_DIR / "vast" / "startup_history.json"

# A stale observation stops being evidence. Thirty days keeps machines that
# reliably start fast while dropping hosts that have long since been retired.
HISTORY_MAX_AGE_SECONDS = 30 * 24 * 60 * 60
# A machine that failed more recently than a success is excluded from the
# fast-host preference so one flake does not poison every future deploy.
MAX_OBSERVATIONS_PER_MACHINE = 50


def machine_startup_stats(
    history: dict[str, Any] | None,
    machine_id: str,
) -> dict[str, float] | None:
    """Return the best recent successful startup time for one machine.

    Returns ``{"ssh_seconds": ..., "healthy_seconds": ...}`` from the most
    recent success, or ``None`` when the machine has no usable evidence
    (never succeeded, or failed after its last success).
    """
    if not history or not machine_id:
        return None
    entries = history.get(machine_id)
    if not isinstance(entries, list):
        return None
    now = time.time()
    latest_success: dict[str, Any] | None = None
    latest_failure_at = 0.0
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        try:
            observed_at = float(raw.get("observed_at_epoch") or 0)
        except (TypeError, ValueError):
            continue
        if observed_at <= 0 or now - observed_at > HISTORY_MAX_AGE_SECONDS:
            continue
        if raw.get("failed"):
            latest_failure_at = max(latest_failure_at, observed_at)
        elif raw.get("healthy_seconds") is not None and (
            latest_success is None
            or observed_at > float(latest_success.get("observed_at_epoch") or 0)
        ):
            latest_success = raw
    if latest_success is None:
        return None
    if latest_failure_at > float(latest_success.get("observed_at_epoch") or 0):
        return None
    try:
        return {
            "ssh_seconds": float(latest_success.get("ssh_seconds") or 0),
            "healthy_seconds": float(latest_success["healthy_seconds"]),
        }
    except (TypeError, ValueError):
        return None


def rank_vast_offers_by_startup(
    offers: list[VastOffer],
    history: dict[str, Any] | None,
) -> list[VastOffer]:
    """Order offers by measured startup time, then price, then offer id.

    Machines with a recent successful observation sort first by their measured
    rental-to-healthy time; everything else keeps its existing relative order
    (callers pass price-sorted lists, which is preserved as the tiebreaker).
    """
    indexed = list(enumerate(offers))
    keyed = []
    for index, offer in indexed:
        stats = machine_startup_stats(history, offer.machine_id or "")
        keyed.append(
            (
                0 if stats is not None else 1,
                stats["healthy_seconds"] if stats is not None else 0.0,
                index,
                offer,
            )
        )
    keyed.sort(key=lambda row: (row[0], row[1], row[2]))
    return [offer for _, _, _, offer in keyed]


def load_vast_startup_history(path: Path | None = None) -> dict[str, Any]:
    """Read the measured startup history; corrupt state raises like VastState."""
    target = path or VAST_STARTUP_HISTORY_PATH
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Cannot read Vast startup history {target}; delete it to start over."
        ) from exc
    if not isinstance(raw, dict):
        raise ValueError(
            f"Cannot read Vast startup history {target}; delete it to start over."
        )
    return {str(key): value for key, value in raw.items()}


def record_vast_startup(
    *,
    machine_id: str,
    offer_id: str,
    ssh_seconds: float | None,
    healthy_seconds: float | None,
    failed: bool = False,
    path: Path | None = None,
) -> None:
    """Append one rental outcome; failures are recorded, never raised."""
    machine = (machine_id or "").strip()
    if not machine:
        return
    target = path or VAST_STARTUP_HISTORY_PATH
    try:
        history = load_vast_startup_history(target)
    except ValueError:
        history = {}
    entries = history.get(machine)
    if not isinstance(entries, list):
        entries = []
    entries.append(
        {
            "observed_at_epoch": time.time(),
            "offer_id": (offer_id or "").strip(),
            "ssh_seconds": ssh_seconds,
            "healthy_seconds": healthy_seconds,
            "failed": bool(failed),
        }
    )
    history[machine] = entries[-MAX_OBSERVATIONS_PER_MACHINE:]
    try:
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(target.parent, 0o700)
        fd, name = tempfile.mkstemp(prefix="startup-history-", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(history, stream, indent=2, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            Path(name).replace(target)
        finally:
            Path(name).unlink(missing_ok=True)
    except OSError:
        return


def startup_evidence_summary(history: dict[str, Any] | None) -> dict[str, Any]:
    """Return the JSON-serializable history for debugging and tests."""
    if not isinstance(history, dict):
        return {}
    return {key: value for key, value in history.items() if isinstance(value, list)}


def fast_machine_ids(
    history: dict[str, Any] | None, *, limit: int = 10
) -> list[str]:
    """Return machine ids with the fastest recent successful startups."""
    if not history:
        return []
    scored: list[tuple[float, str]] = []
    for machine_id in history:
        stats = machine_startup_stats(history, machine_id)
        if stats is not None:
            scored.append((stats["healthy_seconds"], machine_id))
    scored.sort()
    return [machine_id for _, machine_id in scored[: max(0, limit)]]


__all__ = [
    "HISTORY_MAX_AGE_SECONDS",
    "MAX_OBSERVATIONS_PER_MACHINE",
    "VAST_STARTUP_HISTORY_PATH",
    "fast_machine_ids",
    "load_vast_startup_history",
    "machine_startup_stats",
    "rank_vast_offers_by_startup",
    "record_vast_startup",
    "startup_evidence_summary",
]
