"""Protocol layer: pure data models, enums, and event contracts.

This package is shared by Core, TUI, and CLI layers. It has zero UI
dependencies and defines the strict communication contract.
"""

from .enums import BackendType, DeploymentState, OperationType
from .events import (
    BaseEvent,
    ErrorEvent,
    LogEvent,
    OperationCompleteEvent,
    StateChangeEvent,
)
from .models import DeploymentConfig, EndpointInfo, EndpointStatus, LaunchpadSettings

__all__ = [
    "BackendType",
    "BaseEvent",
    "DeploymentConfig",
    "DeploymentState",
    "EndpointInfo",
    "EndpointStatus",
    "ErrorEvent",
    "LaunchpadSettings",
    "LogEvent",
    "OperationCompleteEvent",
    "OperationType",
    "StateChangeEvent",
]
