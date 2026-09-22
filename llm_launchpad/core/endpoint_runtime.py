"""Current-run uptime and cumulative compute cost per endpoint.

A deployment existing is not the same as its GPU running. Modal serves fleet
HTTP from the GPU container itself, so a deployed app may be asleep; Prime and
Vast rent continuously, so a listed pod/rental bills until it is stopped.

This tracker keeps two identities separate:

- the *run*: one container/process lifetime, identified by the runtime's own
  ``process_start_time_seconds`` gauge when a live ``/metrics`` reading exists.
  The runclock resets when that gauge changes and is unknown when no live
  reading has established it.
- the *endpoint*: the logical deployment across restarts, keyed like the
  serving-usage store. Cumulative cost accumulates across runs and persists
  across TUI restarts.

Billing rules:

- Prime/Vast bill while the provider reports a non-terminal state, at the
  row's ``hourly_cost_usd`` when known. Deltas between observations are billed
  in full (the rental bills with the TUI closed), but any gap wider than the
  rate window marks the total estimated/partial.
- Modal bills only across intervals with live evidence the container answered.
  Passive refreshes deliberately never probe, so they never accrue Modal cost
  and never invent a run start. An explicit live fetch may wake the container;
  only then does Modal accrue.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from typing import Any

from ..protocol.enums import ComputeProvider
from ..protocol.models import EndpointInfo
from .coerce import optional_float
from .config import SETTINGS_DIR
from .diagnostics import log_exception
from .serving_metrics import MAX_RATE_WINDOW_SECONDS, usage_key


RUNTIME_PATH = SETTINGS_DIR / "endpoint_runtime.json"

_TERMINAL_STATES = frozenset(
    {
        "stopped",
        "stopping",
        "terminated",
        "archived",
        "failed",
        "error",
        "crashed",
        "destroyed",
        "deleted",
    }
)


def _normalized_state(state: str) -> str:
    return (state or "").strip().lower()


def is_billable_provider_state(state: str) -> bool:
    """Whether the provider considers this deployment allocated/billing."""
    return _normalized_state(state) not in _TERMINAL_STATES


def runtime_key(row: EndpointInfo) -> str:
    """Endpoint identity for run/cost tracking (stable across restarts)."""
    return usage_key(row)


def _live_process_start(row: EndpointInfo) -> float | None:
    serving = row.serving
    if serving is None:
        return None
    started = serving.stats.runtime_started_at
    if started is None or started <= 0:
        return None
    return float(started)


def _row_rate(row: EndpointInfo) -> float | None:
    rate = row.hourly_cost_usd
    if rate is None or rate < 0:
        return None
    return float(rate)


class EndpointRuntimeTracker:
    """Persist run starts and cumulative cost across refreshes and restarts."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] | None = None

    def reset(self) -> None:
        """Drop cached entries (tests only)."""
        with self._lock:
            self._entries = None

    def _runtime_path(self) -> Path:
        return RUNTIME_PATH if self._path is None else self._path

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._entries is not None:
            return self._entries
        entries: dict[str, dict[str, Any]] = {}
        try:
            payload = json.loads(self._runtime_path().read_text(encoding="utf-8"))
        except FileNotFoundError:
            payload = None
        except Exception:
            log_exception(f"Could not read endpoint runtime from {self._runtime_path()}")
            payload = None
        stored = payload.get("entries") if isinstance(payload, dict) else None
        if isinstance(stored, dict):
            entries = {
                str(key): value
                for key, value in stored.items()
                if str(key).strip() and isinstance(value, dict)
            }
        self._entries = entries
        return entries

    def _save(self, entries: dict[str, dict[str, Any]]) -> None:
        path = self._runtime_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"entries": entries}, indent=2), encoding="utf-8")
        except Exception:
            log_exception(f"Could not write endpoint runtime to {path}")
            return
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    def observe_rows(
        self,
        rows: list[EndpointInfo],
        *,
        now: float | None = None,
        explicit: bool = False,
    ) -> None:
        """Update run starts and cumulative cost, then attach them to rows.

        ``explicit`` marks a user-requested live pass that may wake a
        scaled-to-zero container. Passive passes never accrue Modal cost and
        never invent a Modal run start.
        """
        moment = time.time() if now is None else now
        with self._lock:
            entries = self._load()
            changed = False
            for row in rows:
                key = runtime_key(row)
                entry = dict(entries.get(key) or {})
                if self._observe_one(row, entry, moment, explicit=explicit):
                    changed = True
                entries[key] = entry
                self._attach(row, entry, moment)
            if changed:
                self._save(entries)

    def _observe_one(
        self,
        row: EndpointInfo,
        entry: dict[str, Any],
        moment: float,
        *,
        explicit: bool,
    ) -> bool:
        before = json.dumps(entry, sort_keys=True, default=str)
        state = _normalized_state(row.state)
        terminal = not is_billable_provider_state(row.state)
        live_start = _live_process_start(row)
        rate = _row_rate(row)
        stored_rate = optional_float(entry.get("hourly_cost_usd"))

        # A provider-supplied rate wins; otherwise keep the last known rate so
        # a redeploy that drops metadata does not silently zero the estimate.
        # A changed rate marks the total estimated from here on.
        if rate is not None:
            if stored_rate is None or abs(rate - stored_rate) > 1e-9:
                if entry.get("cumulative_cost_usd") not in (None, 0, 0.0):
                    entry["cost_estimated"] = True
                entry["hourly_cost_usd"] = rate
            stored_rate = rate
        elif stored_rate is not None:
            rate = stored_rate

        if not entry.get("tracking_since_epoch"):
            entry["tracking_since_epoch"] = moment

        last_obs = optional_float(entry.get("last_observation_epoch"))
        last_billable = bool(entry.get("last_billable", False))

        if terminal:
            # The run ends, but the endpoint total survives. Keep the rate and
            # tracking window so a later redeploy under the same name continues
            # the endpoint story rather than starting a second ledger.
            entry["run_started_at_epoch"] = None
            entry["run_id"] = None
            entry["last_billable"] = False
            entry["last_observation_epoch"] = moment
            after = json.dumps(entry, sort_keys=True, default=str)
            return before != after

        scale_to_zero = row.provider == ComputeProvider.MODAL

        if scale_to_zero:
            # Passive passes must not wake or bill: without a live gauge this
            # observation contributes no runtime and no cost.
            if live_start is None:
                entry["last_billable"] = False
                entry["last_observation_epoch"] = moment
                after = json.dumps(entry, sort_keys=True, default=str)
                return before != after
            # A live gauge exists only on an explicit pass (or a Prime/Vast
            # row mislabelled Modal). Passive Modal totals never carry one.
            if not explicit and row.serving is not None and row.serving.observed_at is not None:
                # Defensive: a banked total must never accrue Modal cost.
                # Only accrue when this observation itself produced a live
                # reading (observed_at == this pass is approximated by
                # explicit=True from callers; explicit probes set it).
                entry["last_billable"] = False
                entry["last_observation_epoch"] = moment
                after = json.dumps(entry, sort_keys=True, default=str)
                return before != after

        # Establish or roll the current run.
        stored_run_id = entry.get("run_id")
        run_started = optional_float(entry.get("run_started_at_epoch"))
        if live_start is not None:
            run_id = f"process:{live_start:.0f}"
            if stored_run_id != run_id:
                # New container/process: the clock restarts, the endpoint
                # total does not.
                entry["run_id"] = run_id
                entry["run_started_at_epoch"] = live_start
                run_started = live_start
        elif run_started is None:
            # Provisioned rentals bill scheduled uptime even without a metrics
            # gauge, so the first billable observation opens the run. Modal
            # never reaches here without a live gauge (handled above).
            entry["run_id"] = f"observed:{int(moment)}"
            entry["run_started_at_epoch"] = moment
            run_started = moment

        # Accrue cost across the interval since the last observation.
        billable_now = True
        if rate is not None and last_obs is not None and last_billable:
            delta = max(0.0, moment - last_obs)
            if delta > 0:
                if scale_to_zero:
                    # The container may have slept between probes: only bridge
                    # short gaps, and mark the total estimated whenever a gap
                    # was skipped or capped.
                    if delta > MAX_RATE_WINDOW_SECONDS:
                        entry["cost_estimated"] = True
                        entry["coverage_gap_seconds"] = float(
                            entry.get("coverage_gap_seconds") or 0.0
                        ) + delta
                    else:
                        self._accrue(entry, rate, delta)
                else:
                    self._accrue(entry, rate, delta)
                    if delta > MAX_RATE_WINDOW_SECONDS:
                        # The rental billed while nobody watched; the amount
                        # is right at the current rate, but the continuity is
                        # inferred rather than observed.
                        entry["cost_estimated"] = True
                        entry["coverage_gap_seconds"] = float(
                            entry.get("coverage_gap_seconds") or 0.0
                        ) + delta
        elif last_obs is not None and rate is None and last_billable:
            # Billable time with no known rate: the total stays unknown but
            # the gap is recorded so a later rate does not backfill fiction.
            entry["coverage_gap_seconds"] = float(
                entry.get("coverage_gap_seconds") or 0.0
            ) + max(0.0, moment - last_obs)
            entry["cost_estimated"] = True

        # First billable observation opens the ledger without backfilling.
        if entry.get("cumulative_cost_usd") is None and rate is not None:
            entry["cumulative_cost_usd"] = 0.0

        entry["last_billable"] = billable_now
        entry["last_observation_epoch"] = moment
        if state:
            entry["last_state"] = state
        after = json.dumps(entry, sort_keys=True, default=str)
        return before != after

    @staticmethod
    def _accrue(entry: dict[str, Any], rate: float, delta_seconds: float) -> None:
        current = optional_float(entry.get("cumulative_cost_usd")) or 0.0
        billed = float(entry.get("billed_seconds") or 0.0)
        entry["cumulative_cost_usd"] = current + rate * delta_seconds / 3600.0
        entry["billed_seconds"] = billed + delta_seconds

    def _attach(self, row: EndpointInfo, entry: dict[str, Any], moment: float) -> None:
        _ = moment
        run_started = optional_float(entry.get("run_started_at_epoch"))
        # A live process gauge is fresher than a stored observation: prefer it
        # so an explicit fetch moves the clock without waiting for the next
        # passive pass to re-attach.
        live_start = _live_process_start(row)
        if live_start is not None:
            row.run_started_at_epoch = live_start
        elif run_started is not None and row.run_started_at_epoch is None:
            row.run_started_at_epoch = run_started
        elif run_started is None and row.run_started_at_epoch is None:
            row.run_started_at_epoch = None

        stored_rate = optional_float(entry.get("hourly_cost_usd"))
        if row.hourly_cost_usd is None and stored_rate is not None:
            row.hourly_cost_usd = stored_rate
        cumulative = optional_float(entry.get("cumulative_cost_usd"))
        row.cumulative_cost_usd = cumulative
        row.cost_estimated = bool(entry.get("cost_estimated", False))
        tracked_since = optional_float(entry.get("tracking_since_epoch"))
        row.cost_tracked_since_epoch = tracked_since


_DEFAULT_RUNTIME_TRACKER = EndpointRuntimeTracker()


def default_runtime_tracker() -> EndpointRuntimeTracker:
    """The process-wide run/cost tracker the fleet passes annotate into."""
    return _DEFAULT_RUNTIME_TRACKER


def attach_endpoint_runtime(
    rows: list[EndpointInfo],
    *,
    now: float | None = None,
    explicit: bool = False,
) -> None:
    """Observe runs/costs and attach them to rows (passive by default)."""
    default_runtime_tracker().observe_rows(rows, now=now, explicit=explicit)
