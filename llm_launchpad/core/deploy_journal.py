"""Record deployments that are in flight so an interrupted one can be reclaimed.

A provider creates the app before the deploy command returns, so terminating
the client -- quitting the TUI, Ctrl+C, a SIGKILL -- leaves a deployment
running and billing with nothing pointing at it. Certified Fast Deploy widened
that window considerably: provisioning, attestation and calibration all happen
before the endpoint is published, so an impatient quit is far more likely to
land while a real GPU is already allocated.

The journal is written before the deploy starts and cleared once the outcome is
known. An entry surviving into the next session means the previous one did not
finish, which is the only signal that survives a SIGKILL.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

from ..protocol.enums import BackendType, ComputeProvider
from .config import SETTINGS_DIR
from .diagnostics import log_exception

DEPLOY_JOURNAL_PATH = SETTINGS_DIR / "in_flight_deployments.json"
JOURNAL_SCHEMA_VERSION = 1
# Past this, an entry says more about a journal that was never cleaned up than
# about a resource that is still running, so no spend figure is offered.
_STALE_ENTRY_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class InFlightDeployment:
    """One deployment that had been started but not yet resolved."""

    app_name: str
    provider: str
    backend: str
    app_id: str | None = None
    instance_name: str | None = None
    gpu_type: str | None = None
    gpu_count: int = 1
    price_per_hour_usd: float | None = None
    started_at_epoch: float = 0.0

    @property
    def compute_provider(self) -> ComputeProvider:
        return ComputeProvider(self.provider)

    @property
    def backend_type(self) -> BackendType:
        return BackendType(self.backend)

    def elapsed_seconds(self, now: float | None = None) -> float:
        """Seconds since the deployment was recorded as starting."""

        if self.started_at_epoch <= 0:
            return 0.0
        return max(0.0, (now if now is not None else time.time()) - self.started_at_epoch)

    def exposure_usd(self, now: float | None = None) -> float | None:
        """Upper bound on spend if the deployment is in fact still running.

        The journal records that a deploy never resolved, not that the resource
        still exists, so this is an exposure ceiling rather than a bill. Stale
        entries return None: an hourly rate multiplied by an abandoned
        timestamp produces a number large enough to alarm without informing.
        """

        if not self.price_per_hour_usd or self.started_at_epoch <= 0:
            return None
        elapsed = self.elapsed_seconds(now)
        if elapsed > _STALE_ENTRY_SECONDS:
            return None
        return self.price_per_hour_usd * elapsed / 3600.0


def record_in_flight(
    entry: InFlightDeployment,
    path: Path | None = None,
) -> None:
    """Note that a deployment is starting, before the provider is contacted."""

    target = path if path is not None else DEPLOY_JOURNAL_PATH
    entries = {
        row.app_name: row for row in load_in_flight(target) if row.app_name != entry.app_name
    }
    entries[entry.app_name] = entry
    _write(target, entries.values())


def clear_in_flight(app_name: str, path: Path | None = None) -> None:
    """Note that a deployment reached a known outcome and needs no recovery."""

    target = path if path is not None else DEPLOY_JOURNAL_PATH
    name = (app_name or "").strip()
    remaining = [row for row in load_in_flight(target) if row.app_name != name]
    _write(target, remaining)


def load_in_flight(path: Path | None = None) -> tuple[InFlightDeployment, ...]:
    """Return deployments recorded as started but never resolved."""

    target = path if path is not None else DEPLOY_JOURNAL_PATH
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return ()
    if not isinstance(payload, dict) or payload.get("schema_version") != JOURNAL_SCHEMA_VERSION:
        return ()
    rows = payload.get("entries")
    if not isinstance(rows, list):
        return ()
    parsed: list[InFlightDeployment] = []
    for row in rows:
        entry = _entry_from_dict(row)
        if entry is not None:
            parsed.append(entry)
    return tuple(parsed)


def _entry_from_dict(raw: Any) -> InFlightDeployment | None:
    if not isinstance(raw, dict):
        return None
    app_name = str(raw.get("app_name") or "").strip()
    if not app_name:
        return None
    try:
        provider = ComputeProvider(str(raw.get("provider") or "")).value
        backend = BackendType(str(raw.get("backend") or "")).value
    except ValueError:
        return None
    return InFlightDeployment(
        app_name=app_name,
        provider=provider,
        backend=backend,
        app_id=(str(raw["app_id"]) if raw.get("app_id") else None),
        instance_name=(str(raw["instance_name"]) if raw.get("instance_name") else None),
        gpu_type=(str(raw["gpu_type"]) if raw.get("gpu_type") else None),
        gpu_count=int(raw.get("gpu_count") or 1),
        price_per_hour_usd=(
            float(raw["price_per_hour_usd"])
            if raw.get("price_per_hour_usd") is not None
            else None
        ),
        started_at_epoch=float(raw.get("started_at_epoch") or 0.0),
    )


def _write(path: Path, entries: Any) -> None:
    payload = {
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "entries": [asdict(entry) for entry in entries],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        try:
            os.chmod(temporary_path, 0o600)
        except OSError:
            pass
        temporary_path.replace(path)
    except Exception:
        # Losing the journal must never break a deployment; the worst case is
        # the recovery prompt this exists to provide.
        log_exception("Failed to write the in-flight deployment journal")
