"""Enumerations shared across the entire launchpad stack."""

from __future__ import annotations

from enum import Enum

MODAL_VLLM_SCRIPT = "llm_launchpad.backends.modal_vllm_app"
MODAL_LLAMACPP_SCRIPT = "llm_launchpad.backends.modal_llamacpp_app"


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
        return {
            BackendType.LLAMACPP: MODAL_LLAMACPP_SCRIPT,
            BackendType.VLLM: MODAL_VLLM_SCRIPT,
        }[self]


class SpeculativeDecodingMethod(str, Enum):
    """Speculative decoding methods understood by launchpad runtimes."""

    MTP = "mtp"


class ComputeProvider(str, Enum):
    """Infrastructure providers used to host serving backends."""

    MODAL = "modal"
    PRIME = "prime"

    @property
    def display_name(self) -> str:
        return {
            ComputeProvider.MODAL: "Modal",
            ComputeProvider.PRIME: "Prime Intellect",
        }[self]


class BillingModel(str, Enum):
    """How a provider bills an inference resource."""

    SCALE_TO_ZERO = "scale_to_zero"
    PROVISIONED = "provisioned"


class QuoteAvailability(str, Enum):
    """Normalized availability state for a provider quote."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class DeploymentState(str, Enum):
    """Lifecycle states for a deployment operation."""

    IDLE = "idle"
    QUEUED = "queued"
    RUNNING = "running"
    DEPLOYING = "deploying"
    WARMING_UP = "warming_up"
    HEALTHY = "healthy"
    STOPPED = "stopped"


class OperationType(str, Enum):
    """Types of operations the orchestrator can run."""

    DEPLOY = "deploy"
    SMOKE_TEST = "smoke_test"
    WARMUP = "warmup"
    LOGS = "logs"
    STATUS = "status"
    STOP = "stop"
    LIST = "list"
    BENCHMARK = "benchmark"
    STORAGE_LIST = "storage_list"
    STORAGE_PREDOWNLOAD = "storage_predownload"
    STORAGE_DELETE = "storage_delete"
