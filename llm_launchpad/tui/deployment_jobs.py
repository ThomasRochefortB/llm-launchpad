"""Session-owned deployment jobs, independent of the visible screen."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event

from ..protocol.models import DeploymentConfig
from .screens.monitor import MonitorScreen


class DeploymentCancelled(Exception):
    """Stop advancing a deployment after the current provider call returns."""


@dataclass
class DeploymentJob:
    """Retain a deployment's monitor and cancellation state for this session."""

    id: str
    config: DeploymentConfig
    monitor: MonitorScreen
    cancel_requested: Event = field(default_factory=Event)
    finished: Event = field(default_factory=Event)
    accepting_cancellation: bool = True
    outcome: str = "Running"
    persistent_id: str | None = None
    last_seen_seq: int = 0
    opencode_synced: bool = False


@dataclass
class OperationRecord:
    """A finished-or-running non-deploy operation this session can reopen.

    Status checks, benchmarks, logs, stops and storage work each open a monitor
    and used to vanish with it: leaving the screen lost the result. Keeping the
    monitor (installed as a named screen) lets Operations reopen it.
    """

    id: str
    title: str
    subject: str
    monitor: MonitorScreen
    started_at: float

    @property
    def outcome(self) -> str:
        if not getattr(self.monitor, "_done", False):
            return "Running"
        return "Done" if getattr(self.monitor, "_success", False) else "Failed"
