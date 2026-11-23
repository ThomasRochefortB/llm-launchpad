"""Track the deployment stages shown above the monitor's Summary."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from ..core.deploy_log_summary import summary_progress_parts
from ..protocol.enums import BackendType, ComputeProvider, DeploymentState, OperationIntent, OperationType
from ..protocol.events import DeploymentAttemptStartedEvent


class Stage(StrEnum):
    CHECK = "Checks"
    PREPARE = "Prepare"
    DEPLOY = "Deploy app"
    GPU = "Get GPUs"
    START = "Start server"
    VERIFY = "Verify"


_DEPLOY_OPERATIONS = (OperationType.DEPLOY, OperationType.WARMUP, OperationType.SMOKE_TEST)
_STAGE_HEADLINES = {
    Stage.CHECK: "Checking deployment plan",
    Stage.PREPARE: "Preparing runtime and model",
    Stage.DEPLOY: "Deploying app",
    Stage.GPU: "Getting GPUs",
    Stage.START: "Starting server",
    Stage.VERIFY: "Verifying endpoint",
}
_PREPARATION_STEPS = {
    "Preparing model cache",
    "Building runtime image",
    "Building container image",
    "Pulling runtime image",
    "Pulling container image",
    "Installing runtime",
    "Downloading model",
}
_RUNTIME_STEPS = {
    "Starting server",
    "Initializing GPU",
    "Loading model metadata",
    "Loading model",
    "Loading model weights",
    "Loading weights on GPU",
    "Reading cached weights",
    "Warming up model",
    "Capturing CUDA graphs",
    "Initializing KV cache",
    "Compiling kernels",
}


def _monotonic_at(timestamp: datetime | None) -> float:
    now = time.monotonic()
    if timestamp is None:
        return now
    return now - max(0.0, time.time() - timestamp.timestamp())


@dataclass
class DeploymentProgress:
    """A stage cursor and the latest measured activity for the deployment."""

    title: str = "Operation"
    operation: OperationType | None = None
    stages: list[str] = field(default_factory=list)
    stage_states: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)
    stage_started_at: float = field(default_factory=time.monotonic)
    last_progress_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    warmup_deadline: float | None = None
    headline: str = "Starting"
    current_detail: str = ""
    percent: int | None = None
    attempt: int = 1
    done: bool = False
    success: bool = False
    error: str = ""
    resource_status: str = "none"
    endpoint_known: bool = False
    provider: ComputeProvider = ComputeProvider.MODAL
    backend: BackendType | None = None
    intent: OperationIntent = OperationIntent.SERVE
    warmup: bool = True
    gpu_type: str | None = None
    gpu_count: int | None = None

    def start(self, operation: OperationType | None, title: str) -> None:
        """Initialize the monitor for the requested operation."""
        self.operation = operation
        self.title = title or "Operation"
        self.attempt = 1
        self.started_at = time.monotonic()
        self._reset_attempt(self.started_at)

    def on_attempt_started(self, event: DeploymentAttemptStartedEvent) -> None:
        """Reset the stage strip for an approved placement, including on replay."""
        at = _monotonic_at(event.timestamp)
        self.provider = event.provider
        self.backend = event.backend or self.backend
        self.intent = event.intent
        self.warmup = event.warmup
        self.gpu_type = event.gpu_type
        self.gpu_count = event.gpu_count
        self.attempt = event.attempt
        if event.attempt == 1:
            self.started_at = at
        self._reset_attempt(at)

    def _reset_attempt(self, at: float) -> None:
        if self.operation not in _DEPLOY_OPERATIONS:
            self.stages = [self.operation.value.capitalize() if self.operation else "Running"]
        elif self.intent == OperationIntent.PRELOAD:
            self.stages = [Stage.CHECK, Stage.PREPARE]
        elif self.intent == OperationIntent.SMOKE:
            self.stages = [Stage.CHECK, Stage.PREPARE, Stage.GPU, Stage.START, Stage.VERIFY]
        elif self.provider == ComputeProvider.MODAL:
            self.stages = [Stage.CHECK]
            if self.backend != BackendType.VLLM:
                self.stages.append(Stage.PREPARE)
            self.stages.append(Stage.DEPLOY)
            if self.warmup:
                self.stages.extend([Stage.GPU, Stage.START, Stage.VERIFY])
        else:
            self.stages = [Stage.CHECK, Stage.GPU, Stage.PREPARE, Stage.START]
            if self.warmup:
                self.stages.append(Stage.VERIFY)
        self.stage_states = ["active"] + ["pending"] * (len(self.stages) - 1)
        self.stage_started_at = self.last_progress_at = at
        self.finished_at = self.warmup_deadline = None
        self.headline = _STAGE_HEADLINES[Stage.CHECK] if self.stages[0] == Stage.CHECK else "Running"
        self.current_detail = self.error = ""
        self.percent = None
        self.done = self.success = self.endpoint_known = False
        self.resource_status = "none"

    def _activate(
        self,
        stage: Stage,
        at: float,
        *,
        headline: str = "",
        detail: str = "",
        percent: int | None = None,
    ) -> None:
        if self.done or stage not in self.stages:
            return
        index = self.stages.index(stage)
        active_index = self.stage_states.index("active")
        if index < active_index:
            return
        headline = headline or _STAGE_HEADLINES[stage]
        if index != active_index or headline != self.headline:
            self.stage_started_at = at
        self.stage_states[:index + 1] = ["done"] * index + ["active"]
        if stage == Stage.VERIFY:
            self.warmup_deadline = None
        if (headline, detail, percent) != (self.headline, self.current_detail, self.percent):
            self.last_progress_at = at
        self.headline = headline
        self.current_detail = detail
        self.percent = percent

    def on_state(
        self,
        state: DeploymentState,
        detail: str = "",
        *,
        operation: OperationType | None = None,
        timestamp: datetime | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        """Advance from provider and lifecycle state events."""
        if self.done:
            return
        at = _monotonic_at(timestamp)
        operation = operation or self.operation
        if self.operation not in _DEPLOY_OPERATIONS:
            self.current_detail = detail
            return
        if operation == OperationType.STOP and state != DeploymentState.STOPPED:
            self.on_cleanup(at)
        elif state == DeploymentState.QUEUED and operation == OperationType.WARMUP:
            self.on_gpu_wait(at, detail)
        elif state == DeploymentState.QUEUED and self.error:
            # Historical jobs predate the explicit attempt-start event.
            self.attempt += 1
            self._reset_attempt(at)
        elif state == DeploymentState.QUEUED and "GPU availability" in detail:
            self._activate(Stage.GPU, at, headline="Finding GPUs")
        elif state == DeploymentState.DEPLOYING:
            if "modal deploy" in detail:
                self._activate(Stage.DEPLOY, at)
            elif "Provisioning" in detail or detail.startswith("Renting Vast"):
                self._activate(Stage.GPU, at, headline="Provisioning machine")
            elif "Waiting for the " in detail:
                self._activate(Stage.START, at, headline="Waiting for server response")
            elif "secure Prime Tunnel" not in detail:
                self._activate(Stage.PREPARE, at, detail=detail if not detail.startswith("modal ") else "")
        elif state == DeploymentState.RUNNING:
            if self.intent == OperationIntent.PRELOAD:
                self._activate(Stage.PREPARE, at)
            elif self.intent == OperationIntent.SMOKE:
                self._activate(Stage.START, at, headline="Running smoke test")
        elif state == DeploymentState.WARMING_UP:
            if timeout_seconds is not None:
                self.warmup_deadline = at + timeout_seconds
            if "GPU allocated" in detail:
                self._activate(Stage.START, at)
            else:
                self.on_waiting_for_server(at)
        elif state in (DeploymentState.VERIFYING, DeploymentState.CALIBRATING):
            self._activate(Stage.VERIFY, at, headline=detail)
        elif state == DeploymentState.PUBLISHING:
            self._activate(Stage.VERIFY, at, headline="Publishing connection details")
        elif state == DeploymentState.HEALTHY and self.active_stage() != Stage.VERIFY:
            self._activate(Stage.VERIFY, at)

    def on_waiting_for_server(self, at: float) -> None:
        if self.provider == ComputeProvider.MODAL and self.active_stage() not in (Stage.START, Stage.VERIFY):
            if not self.headline.startswith("Waiting for ") or self.active_stage() != Stage.GPU:
                self._activate(Stage.GPU, at, headline="Waiting for server response")
        elif self.active_stage() not in (Stage.START, Stage.VERIFY):
            self._activate(Stage.START, at, headline="Waiting for server response")

    def on_gpu_wait(self, at: float, detail: str) -> None:
        gpu_type = self.gpu_type
        if not gpu_type and "(GPU_" in detail:
            gpu_type = detail.split("(GPU_", 1)[1].split(")", 1)[0].split(",", 1)[0]
        gpu = f"{self.gpu_count} × {gpu_type}" if self.gpu_count and gpu_type else gpu_type
        unit = "GPU" if self.gpu_count == 1 else "GPUs"
        headline = f"Waiting for {gpu} {unit}" if gpu else "Waiting for GPUs"
        self._activate(
            Stage.GPU, at, headline=headline, detail="Modal is looking for capacity. No ETA yet."
        )

    def on_cleanup(self, at: float) -> None:
        if self.headline != "Cleaning up deployment":
            self.stage_started_at = at
        self.headline = "Cleaning up deployment"
        self.current_detail = ""
        self.warmup_deadline = None
        self.percent = None

    def on_phase_done(
        self, operation: OperationType, success: bool, detail: str, timestamp: datetime | None
    ) -> None:
        """Record a provider step completing while the overall job continues."""
        if self.done:
            return
        at = _monotonic_at(timestamp)
        if operation == OperationType.STOP:
            self.resource_status = "stopped" if success else "failed"
        if not success:
            self.on_error(detail)
        elif operation == OperationType.DEPLOY and self.intent == OperationIntent.SERVE:
            self.resource_status = "allocated"
            if self.warmup:
                if self.provider == ComputeProvider.MODAL:
                    self.on_waiting_for_server(at)
                else:
                    self._activate(Stage.VERIFY, at)
        elif operation == OperationType.WARMUP:
            self._activate(Stage.VERIFY, at, headline="Endpoint responding")

    def on_resource_allocated(self) -> None:
        """Record the resource for failure and cleanup reporting."""
        self.resource_status = "allocated"

    def on_endpoint_available(self) -> None:
        """Record the endpoint URL becoming known."""
        self.endpoint_known = True
        if self.headline == "Opening endpoint":
            self._activate(Stage.START, time.monotonic(), headline="Waiting for server response")

    def on_milestone(self, detail: str, timestamp: datetime | None = None) -> None:
        """Map normalized provider activity to the current stage."""
        if self.done or self.operation not in _DEPLOY_OPERATIONS:
            return
        at = _monotonic_at(timestamp)
        label, percent = summary_progress_parts(detail)
        if label.startswith("Waiting for GPU scheduling"):
            self.on_gpu_wait(at, detail)
        elif label.startswith(("Certification failed; cleaning up", "Cancelled; cleaning up", "Cleaning up")):
            self.on_cleanup(at)
        elif label == "Endpoint published":
            self.endpoint_known = True
            if self.warmup and self.provider == ComputeProvider.MODAL:
                self.on_waiting_for_server(at)
        elif label == "Publishing endpoint" and self.provider == ComputeProvider.MODAL:
            self._activate(Stage.DEPLOY, at)
        elif label in ("Machine ready", "GPU allocated"):
            stage = Stage.START if self.provider == ComputeProvider.MODAL else Stage.PREPARE
            self._activate(stage, at)
        elif label in ("Finding a GPU", "Provisioning machine", "Waiting for SSH"):
            self._activate(Stage.GPU, at, headline=label, percent=percent)
        elif label in _PREPARATION_STEPS:
            stage = Stage.PREPARE
            if self.active_stage() == Stage.DEPLOY and "image" in label:
                stage = Stage.DEPLOY
            elif self.provider == ComputeProvider.MODAL and self.active_stage() in (Stage.GPU, Stage.START):
                stage = Stage.START
            transfer = detail.partition(" — ")[2]
            self._activate(stage, at, headline=label, detail=transfer, percent=percent)
        elif label == "Model cached":
            self._activate(Stage.PREPARE, at, headline="Model cached")
        elif label in _RUNTIME_STEPS or label == "Loading cached model":
            stage = Stage.START
            if label == "Loading cached model" and self.active_stage() == Stage.PREPARE:
                stage = Stage.PREPARE
            self._activate(stage, at, headline=label, percent=percent)
        elif label == "Runtime ready":
            headline = "Waiting for server response" if self.endpoint_known else "Opening endpoint"
            self._activate(Stage.START, at, headline=headline)
        elif label in ("Model loaded", "Server listening"):
            self._activate(Stage.START, at, headline="Waiting for server response")
        elif label == "Server is ready!":
            self._activate(Stage.VERIFY, at)
        elif label.startswith(("Verifying tool calling", "Verifying image input")):
            self._activate(Stage.VERIFY, at, headline=label.rstrip("."))

    def on_error(self, message: str) -> None:
        if message.strip():
            self.error = message.strip()
        self.last_progress_at = time.monotonic()

    def on_done(
        self, success: bool, detail: str = "", *,
        resource_status: str = "", timestamp: datetime | None = None,
    ) -> None:
        """Finish the stage strip after the whole job completes."""
        self.done = True
        self.success = success
        self.finished_at = _monotonic_at(timestamp)
        self.percent = None
        if resource_status:
            self.resource_status = resource_status
        if success:
            self.stage_states = ["done"] * len(self.stage_states)
            self.headline = "Complete"
            self.current_detail = detail
            if self.operation in _DEPLOY_OPERATIONS:
                if self.intent == OperationIntent.PRELOAD:
                    self.headline = "Model cached"
                elif self.intent == OperationIntent.SMOKE:
                    self.headline = "Smoke test passed"
                else:
                    self.headline = "Ready" if self.warmup else "App deployed"
        else:
            if "active" in self.stage_states:
                self.stage_states[self.stage_states.index("active")] = "failed"
            self.error = self.error or detail
            self.headline = "Cancelled" if detail.lower().startswith("cancel") else "Failed"

    def elapsed_seconds(self, *, now: float | None = None) -> float:
        end = self.finished_at if self.finished_at is not None else now
        return max(0.0, (time.monotonic() if end is None else end) - self.started_at)

    def stage_elapsed_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.stage_started_at)

    def warmup_seconds_remaining(self) -> float | None:
        if self.warmup_deadline is None:
            return None
        return max(0.0, self.warmup_deadline - time.monotonic())

    def seconds_since_progress(self, *, now: float | None = None) -> float:
        return max(0.0, (time.monotonic() if now is None else now) - self.last_progress_at)

    def quiet_explanation(self, *, now: float | None = None) -> str:
        """Explain an unmeasured delay without assigning progress to it."""
        if self.done or self.seconds_since_progress(now=now) < 60:
            return ""
        if self.headline == "Waiting for server response":
            return "Waiting for the first successful response."
        if self.active_stage() == Stage.PREPARE:
            return "Preparing the runtime or model cache can take several minutes."
        return ""

    def active_stage(self) -> str:
        if "active" in self.stage_states:
            return self.stages[self.stage_states.index("active")]
        return ""

    def next_stage(self) -> str:
        if "pending" in self.stage_states:
            return self.stages[self.stage_states.index("pending")]
        return ""
