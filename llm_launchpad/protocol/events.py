"""Protocol events emitted by the Core layer and consumed by UI layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .enums import DeploymentState, OperationType
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
    operation: Optional[OperationType] = None
    is_milestone: bool = False


@dataclass(frozen=True)
class StateChangeEvent(BaseEvent):
    """Signals a transition in the deployment state machine."""

    current: DeploymentState = DeploymentState.IDLE
    operation: Optional[OperationType] = None
    detail: str = ""


@dataclass(frozen=True)
class ErrorEvent(BaseEvent):
    """An error occurred during an operation."""

    message: str = ""
    operation: Optional[OperationType] = None
    exit_code: Optional[int] = None
    recoverable: bool = True


@dataclass(frozen=True)
class OperationCompleteEvent(BaseEvent):
    """Signals that an operation finished (success or failure)."""

    operation: OperationType = OperationType.DEPLOY
    success: bool = True
    exit_code: int = 0
    detail: str = ""
    data: Optional[Any] = None


@dataclass(frozen=True)
class EndpointAvailableEvent(BaseEvent):
    """Public inference URL is known. The runtime may still be loading weights."""

    endpoint: EndpointInfo = field(default_factory=EndpointInfo)
    operation: Optional[OperationType] = None
