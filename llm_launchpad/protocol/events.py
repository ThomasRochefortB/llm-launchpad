"""Protocol events emitted by the Core layer and consumed by UI layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .enums import BackendType, ComputeProvider, DeploymentState, OperationIntent, OperationType
from .models import EndpointInfo


@dataclass(frozen=True)
class BaseEvent:
    """Base for all protocol events."""

    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class LogEvent(BaseEvent):
    """A single log line emitted by a running operation."""

    line: str = ""
    stream: str = "stdout"  # "stdout" | "stderr"
    operation: OperationType | None = None
    is_milestone: bool = False


@dataclass(frozen=True)
class StateChangeEvent(BaseEvent):
    """Signals a transition in the deployment state machine."""

    current: DeploymentState = DeploymentState.IDLE
    operation: OperationType | None = None
    detail: str = ""
    timeout_seconds: int | None = None


@dataclass(frozen=True)
class DeploymentAttemptStartedEvent(BaseEvent):
    """The serving placement and steps requested for this attempt."""

    attempt: int = 1
    provider: ComputeProvider = ComputeProvider.MODAL
    backend: BackendType | None = None
    intent: OperationIntent = OperationIntent.SERVE
    warmup: bool = True
    gpu_type: str | None = None
    gpu_count: int | None = None


@dataclass(frozen=True)
class ErrorEvent(BaseEvent):
    """An error occurred during an operation."""

    message: str = ""
    operation: OperationType | None = None
    exit_code: int | None = None
    recoverable: bool = True


@dataclass(frozen=True)
class OperationCompleteEvent(BaseEvent):
    """Signals that an operation finished (success or failure)."""

    operation: OperationType = OperationType.DEPLOY
    success: bool = True
    exit_code: int = 0
    detail: str = ""
    data: Any | None = None


@dataclass(frozen=True)
class ResourceAllocatedEvent(BaseEvent):
    """A billable resource exists, before its endpoint is available."""

    app_id: str = ""
    operation: OperationType = OperationType.DEPLOY


@dataclass(frozen=True)
class EndpointAvailableEvent(BaseEvent):
    """Public inference URL is known. The runtime may still be loading weights."""

    endpoint: EndpointInfo = field(default_factory=EndpointInfo)
    operation: OperationType | None = None
