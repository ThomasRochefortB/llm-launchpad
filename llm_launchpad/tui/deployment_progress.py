"""Typed deployment progress tracking for the monitor screen.

The monitor used to represent progress only inside scrolling logs, so a user
had to interpret backend output to answer "what is happening now?". This
module tracks a small, typed stage machine driven by lifecycle/state events
and renders it as a persistent summary above the logs.

Only measured totals become percentages; quiet stages get explanations, never
fabricated progress. Fallback attempts reset the stage row but keep the
attempt count so a fresh placement never inherits completed stages.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..protocol.enums import DeploymentState, OperationType

DEPLOY_STAGES: tuple[str, ...] = ("Validate", "Allocate", "Prepare", "Load", "Verify")
GENERIC_STAGES: tuple[str, ...] = ("Running",)

_LONG_QUIET_THRESHOLD_SECONDS = 60.0

_QUIET_EXPLANATIONS: dict[str, str] = {
    "Prepare": "Building the serving image or downloading weights. This can take several minutes without new output.",
    "Load": "Loading weights into GPU memory. Large models can sit here quietly for minutes.",
    "Verify": "Running readiness checks against the live endpoint.",
}


@dataclass
class DeploymentProgress:
    """Mutable stage tracker; the monitor owns one per operation."""

    title: str = "Operation"
    operation: OperationType | None = None
    stages: list[str] = field(default_factory=lambda: list(DEPLOY_STAGES))
    stage_states: list[str] = field(default_factory=lambda: ["active"] + ["pending"] * 4)
    started_at: float = field(default_factory=time.monotonic)
    last_progress_at: float = field(default_factory=time.monotonic)
    current_detail: str = "Starting"
    attempt: int = 1
    done: bool = False
    success: bool = False
    error: str = ""
    resource_status: str = "none"
    endpoint_known: bool = False

    def start(self, operation: OperationType | None, title: str) -> None:
        """Begin tracking a new operation, choosing the stage row it needs."""
        self.operation = operation
        self.title = title or "Operation"
        if operation in (OperationType.DEPLOY, OperationType.WARMUP, OperationType.SMOKE_TEST):
            self.stages = list(DEPLOY_STAGES)
            self.stage_states = ["active"] + ["pending"] * (len(self.stages) - 1)
        elif operation is not None:
            self.stages = [operation.value.capitalize()]
            self.stage_states = ["active"]
        else:
            self.stages = list(GENERIC_STAGES)
            self.stage_states = ["active"]
        now = time.monotonic()
        self.started_at = now
        self.last_progress_at = now
        self.current_detail = "Starting"
        self.attempt = 1
        self.done = False
        self.success = False
        self.error = ""
        self.resource_status = "none"
        self.endpoint_known = False

    def _touch(self, detail: str = "") -> None:
        self.last_progress_at = time.monotonic()
        if detail:
            self.current_detail = detail

    def _set_stage(self, index: int, state: str) -> None:
        if 0 <= index < len(self.stage_states) and self.stage_states[index] != state:
            self.stage_states[index] = state
            self.last_progress_at = time.monotonic()

    def _activate(self, index: int, detail: str = "") -> None:
        # Exactly one stage is active at a time. Advancing completes every
        # earlier stage (we moved past them) and activates the new one;
        # later stages stay pending. A failed stage is never marked done by
        # advancing past it.
        for i in range(len(self.stage_states)):
            if i < index and self.stage_states[i] == "pending":
                self.stage_states[i] = "done"
            elif i == index and self.stage_states[i] in ("pending", "active"):
                self.stage_states[i] = "active"
            elif i < index and self.stage_states[i] == "active":
                self.stage_states[i] = "done"
        self._touch(detail)

    def on_state(self, state: DeploymentState, detail: str = "") -> None:
        """Advance stages from a lifecycle state transition."""
        cleaned = (detail or "").strip()
        if self.operation not in (OperationType.DEPLOY, OperationType.WARMUP, OperationType.SMOKE_TEST):
            self._touch(cleaned or self.current_detail)
            return
        mapping: dict[DeploymentState, int] = {
            DeploymentState.QUEUED: 0,
            DeploymentState.RUNNING: 1,
            DeploymentState.DEPLOYING: 2,
            DeploymentState.WARMING_UP: 3,
            DeploymentState.CALIBRATING: 3,
            DeploymentState.VERIFYING: 4,
            DeploymentState.PUBLISHING: 4,
        }
        if state == DeploymentState.HEALTHY:
            self.stage_states = ["done"] * len(self.stage_states)
            self._touch(cleaned or "Healthy")
            return
        if state in (DeploymentState.STOPPED, DeploymentState.IDLE):
            self._touch(cleaned or self.current_detail)
            return
        index = mapping.get(state)
        if index is None:
            self._touch(cleaned or self.current_detail)
            return
        # A fresh QUEUED after a failure is a fallback attempt, not a rewind:
        # reset the row and count the attempt so completed stages never carry
        # over into the new placement.
        if state == DeploymentState.QUEUED and self.error:
            self.attempt += 1
            self.error = ""
            self.stage_states = ["pending"] * len(self.stage_states)
        self._activate(index, cleaned or self.current_detail)

    def on_resource_allocated(self, detail: str = "Resource allocated") -> None:
        """A billable resource exists, even before its endpoint is reachable."""
        self.resource_status = "allocated"
        if self.operation in (OperationType.DEPLOY, OperationType.WARMUP, OperationType.SMOKE_TEST):
            self._set_stage(0, "done")
            self._set_stage(1, "done")
            self._activate(2, detail)
        else:
            self._touch(detail)

    def on_endpoint_available(self, detail: str = "Endpoint URL known; runtime may still be loading") -> None:
        """The public URL is known; weights may still be loading."""
        self.endpoint_known = True
        if self.operation in (OperationType.DEPLOY, OperationType.WARMUP, OperationType.SMOKE_TEST):
            self._set_stage(0, "done")
            self._set_stage(1, "done")
            self._set_stage(2, "done")
            self._activate(3, detail)
        else:
            self._touch(detail)

    def on_milestone(self, detail: str) -> None:
        """Record meaningful log-derived progress without inventing totals."""
        cleaned = (detail or "").strip()
        if not cleaned:
            return
        lowered = cleaned.lower()
        if self.operation in (OperationType.DEPLOY, OperationType.WARMUP, OperationType.SMOKE_TEST):
            if any(k in lowered for k in ("endpoint published", "server is ready", "runtime ready")):
                self._set_stage(3, "done")
                self._activate(4, cleaned)
                return
            if any(k in lowered for k in ("loading", "weights", "checkpoint", "hydrat")):
                self._activate(3, cleaned)
                return
            if any(k in lowered for k in ("building image", "download", "fetching", "prepar")):
                self._activate(2, cleaned)
                return
            if "gpu allocated" in lowered or "machine ready" in lowered:
                self._set_stage(0, "done")
                self._set_stage(1, "done")
                self._activate(2, cleaned)
                return
        self._touch(cleaned)

    def on_connection_ready(self, detail: str = "Endpoint verified") -> None:
        self._touch(detail)

    def on_error(self, message: str) -> None:
        cleaned = (message or "").strip()
        if cleaned:
            self.error = cleaned
            self.current_detail = cleaned
        self.last_progress_at = time.monotonic()

    def on_done(self, success: bool, detail: str = "", *, resource_status: str = "") -> None:
        """Finalize the row after the full deploy/warmup/cleanup sequence."""
        self.done = True
        self.success = success
        if resource_status:
            self.resource_status = resource_status
        elif success and self.resource_status == "none":
            self.resource_status = "allocated"
        cleaned = (detail or "").strip()
        if success:
            self.stage_states = ["done"] * len(self.stage_states)
            self._touch(cleaned or "Complete")
        else:
            for i in range(len(self.stage_states) - 1, -1, -1):
                if self.stage_states[i] == "active":
                    self.stage_states[i] = "failed"
                    break
            else:
                for i in range(len(self.stage_states)):
                    if self.stage_states[i] == "pending":
                        self.stage_states[i] = "failed"
                        break
            if cleaned:
                self.error = cleaned
                self.current_detail = cleaned
            self.last_progress_at = time.monotonic()

    def elapsed_seconds(self, *, now: float | None = None) -> float:
        base = time.monotonic() if now is None else now
        return max(0.0, base - self.started_at)

    def seconds_since_progress(self, *, now: float | None = None) -> float:
        base = time.monotonic() if now is None else now
        return max(0.0, base - self.last_progress_at)

    def quiet_explanation(self, *, now: float | None = None) -> str:
        """Explain a long quiet stage without claiming progress happened."""
        if self.done:
            return ""
        if self.seconds_since_progress(now=now) < _LONG_QUIET_THRESHOLD_SECONDS:
            return ""
        for stage, state in zip(self.stages, self.stage_states, strict=False):
            if state == "active":
                return _QUIET_EXPLANATIONS.get(stage, "")
        return ""

    def active_stage(self) -> str:
        for stage, state in zip(self.stages, self.stage_states, strict=False):
            if state == "active":
                return stage
        if self.done:
            return "Done"
        return self.stages[-1] if self.stages else ""
