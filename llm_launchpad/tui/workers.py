"""Worker bridge: Core orchestrator generators -> Textual messages.

Each worker runs a Core generator in a thread and posts protocol events
as Textual messages to the app/screen.
"""

from __future__ import annotations

from textual.message import Message
from ..core.hf_models import ModelCandidate

from ..protocol.enums import DeploymentState, OperationType
from ..protocol.events import (
    BaseEvent,
    ErrorEvent,
    LogEvent,
    OperationCompleteEvent,
    StateChangeEvent,
)


# -----------------------------------------------------------------------
# Textual Messages (thin wrappers around protocol events)
# -----------------------------------------------------------------------


class LogMessage(Message):
    """A log line from an operation."""

    def __init__(self, line: str, stream: str = "stdout") -> None:
        super().__init__()
        self.line = line
        self.stream = stream


class StateChanged(Message):
    """Deployment state transition."""

    def __init__(
        self,
        state: DeploymentState,
        operation: OperationType | None = None,
        detail: str = "",
    ) -> None:
        super().__init__()
        self.state = state
        self.operation = operation
        self.detail = detail


class OperationDone(Message):
    """An operation finished."""

    def __init__(
        self,
        operation: OperationType,
        success: bool,
        exit_code: int = 0,
        detail: str = "",
        data: object = None,
    ) -> None:
        super().__init__()
        self.operation = operation
        self.success = success
        self.exit_code = exit_code
        self.detail = detail
        self.data = data


class OperationError(Message):
    """An error occurred."""

    def __init__(self, message: str, recoverable: bool = True) -> None:
        super().__init__()
        self.message = message
        self.recoverable = recoverable


class VllmModelsLoaded(Message):
    """Top ranked vLLM-capable models were loaded."""

    def __init__(self, mode: str, models: list[ModelCandidate]) -> None:
        super().__init__()
        self.mode = mode
        self.models = models


class VllmModelsFailed(Message):
    """Model discovery failed for a ranking mode."""

    def __init__(self, mode: str, error: str) -> None:
        super().__init__()
        self.mode = mode
        self.error = error


# -----------------------------------------------------------------------
# Event dispatcher: protocol event -> Textual message
# -----------------------------------------------------------------------


def _dispatch_event(app_or_widget: object, event: BaseEvent) -> None:
    """Convert a protocol event into a Textual message and post it."""
    poster = getattr(app_or_widget, "post_message", None)
    if poster is None:
        return

    if isinstance(event, LogEvent):
        poster(LogMessage(line=event.line, stream=event.stream))
    elif isinstance(event, StateChangeEvent):
        poster(StateChanged(state=event.current, operation=event.operation, detail=event.detail))
    elif isinstance(event, OperationCompleteEvent):
        poster(
            OperationDone(
                operation=event.operation,
                success=event.success,
                exit_code=event.exit_code,
                detail=event.detail,
                data=event.data,
            )
        )
    elif isinstance(event, ErrorEvent):
        poster(OperationError(message=event.message, recoverable=event.recoverable))
