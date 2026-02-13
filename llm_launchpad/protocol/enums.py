"""Enumerations shared across the entire launchpad stack."""

from __future__ import annotations

from enum import Enum


class BackendType(str, Enum):
    """Supported serving backends."""

    LLAMACPP = "llamacpp"
    VLLM = "vllm"

    @property
    def display_name(self) -> str:
        return {
            BackendType.LLAMACPP: "llama.cpp (GGUF)",
            BackendType.VLLM: "vLLM (OpenAI-compatible)",
        }[self]

    @property
    def script(self) -> str:
        from ..core.paths import MODAL_LLAMACPP_SCRIPT, MODAL_VLLM_SCRIPT

        return {
            BackendType.LLAMACPP: MODAL_LLAMACPP_SCRIPT,
            BackendType.VLLM: MODAL_VLLM_SCRIPT,
        }[self]


class DeploymentState(str, Enum):
    """Lifecycle states for a deployment operation."""

    IDLE = "idle"
    QUEUED = "queued"
    RUNNING = "running"
    DEPLOYING = "deploying"
    WARMING_UP = "warming_up"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"
    ERROR = "error"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {
            DeploymentState.HEALTHY,
            DeploymentState.UNHEALTHY,
            DeploymentState.STOPPED,
            DeploymentState.ERROR,
            DeploymentState.CANCELLED,
            DeploymentState.IDLE,
        }


class OperationType(str, Enum):
    """Types of operations the orchestrator can run."""

    DEPLOY = "deploy"
    SMOKE_TEST = "smoke_test"
    WARMUP = "warmup"
    LOGS = "logs"
    STATUS = "status"
    STOP = "stop"
    LIST = "list"
    STORAGE_LIST = "storage_list"
    STORAGE_PREDOWNLOAD = "storage_predownload"
    STORAGE_DELETE = "storage_delete"
