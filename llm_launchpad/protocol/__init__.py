"""Protocol layer: pure data models, enums, and event contracts.

This package is shared by Core, TUI, and CLI layers. It has zero UI
dependencies and defines the strict communication contract.
"""

from .enums import BackendType, DeploymentState, OperationType, SpeculativeDecodingMethod
from .events import (
    BaseEvent,
    EndpointAvailableEvent,
    ErrorEvent,
    LogEvent,
    OperationCompleteEvent,
    StateChangeEvent,
)
from .models import (
    DeploymentConfig,
    EndpointInfo,
    LaunchpadSettings,
    ReasoningCapabilities,
    SpeculativeDecodingConfig,
    StoredModelInfo,
    StorageSnapshot,
)

__all__ = [
    "BackendType",
    "BaseEvent",
    "DeploymentConfig",
    "DeploymentState",
    "EndpointAvailableEvent",
    "EndpointInfo",
    "ErrorEvent",
    "LaunchpadSettings",
    "LogEvent",
    "OperationCompleteEvent",
    "OperationType",
    "ReasoningCapabilities",
    "SpeculativeDecodingConfig",
    "SpeculativeDecodingMethod",
    "StateChangeEvent",
    "StoredModelInfo",
    "StorageSnapshot",
]
