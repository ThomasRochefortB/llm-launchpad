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
        """Modal entrypoint module for this backend.

        Provider-neutral callers should resolve scripts through the Modal
        adapter instead; this stays as the compatibility accessor while the
        migration completes.
        """
        return {
            BackendType.LLAMACPP: MODAL_LLAMACPP_SCRIPT,
            BackendType.VLLM: MODAL_VLLM_SCRIPT,
        }[self]


class SpeculativeDecodingMethod(str, Enum):
    """Speculative decoding methods understood by launchpad runtimes."""

    MTP = "mtp"


class ServingObjective(str, Enum):
    """Optimization target for a planned inference endpoint."""

    GENERAL_PURPOSE = "general_purpose"
    INTERACTIVE = "interactive"
    THROUGHPUT = "throughput"
    BENCHMARK = "benchmark"

    @property
    def display_name(self) -> str:
        return {
            ServingObjective.GENERAL_PURPOSE: "General purpose",
            ServingObjective.INTERACTIVE: "Interactive",
            ServingObjective.THROUGHPUT: "Throughput",
            ServingObjective.BENCHMARK: "Benchmark",
        }[self]


class CertificationState(str, Enum):
    """Confidence level for a planned or running serving configuration."""

    ESTIMATED = "estimated"
    CERTIFIED = "certified"
    REJECTED = "rejected"


class EvidenceLevel(str, Enum):
    """Strength of evidence behind a placement claim.

    Missing evidence is unknown, never success: a claim without one of these
    levels has not been observed.
    """

    # A planner formula computed this; nothing ran.
    PREDICTED = "predicted"
    # One live run observed this exact configuration.
    OBSERVED = "observed"


class MemoryObservationSource(str, Enum):
    """Where one per-device memory reading came from."""

    # Sampled device memory during a phase of the run.
    DEVICE_SAMPLE = "device_sample"


class OutcomeKind(str, Enum):
    """How an evaluation run ended, kept separate from what it taught."""

    SUCCESS = "success"
    OUT_OF_MEMORY = "out_of_memory"
    RUNTIME_INCOMPATIBLE = "runtime_incompatible"
    PROVISIONING_FAILED = "provisioning_failed"
    CANCELLED = "cancelled"
    TELEMETRY_MISSING = "telemetry_missing"
    UNKNOWN = "unknown"


class CachePolicy(str, Enum):
    """Prompt-cache behavior required by one workload scenario."""

    UNCACHED = "uncached"
    UNSPECIFIED = "unspecified"


class ComputeProvider(str, Enum):
    """Infrastructure providers used to host serving backends."""

    MODAL = "modal"
    PRIME = "prime"
    VAST = "vast"

    @property
    def display_name(self) -> str:
        return {
            ComputeProvider.MODAL: "Modal",
            ComputeProvider.PRIME: "Prime Intellect",
            ComputeProvider.VAST: "Vast.ai",
        }[self]


class OfferProvider(str, Enum):
    """Marketplaces with offer discovery, independently of deployment support."""

    PRIME = "prime"
    VAST = "vast"


class BillingModel(str, Enum):
    """How a provider bills an inference resource."""

    SCALE_TO_ZERO = "scale_to_zero"
    PROVISIONED = "provisioned"


class QuoteAvailability(str, Enum):
    """Normalized availability state for a provider quote."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"

    @property
    def sort_rank(self) -> int:
        """Rank for ordering quotes best-first: available, unknown, unavailable."""
        return _QUOTE_AVAILABILITY_RANK[self]


_QUOTE_AVAILABILITY_RANK = {
    QuoteAvailability.AVAILABLE: 0,
    QuoteAvailability.UNKNOWN: 1,
    QuoteAvailability.UNAVAILABLE: 2,
}


class DeploymentState(str, Enum):
    """Lifecycle states for a deployment operation."""

    IDLE = "idle"
    QUEUED = "queued"
    RUNNING = "running"
    DEPLOYING = "deploying"
    WARMING_UP = "warming_up"
    VERIFYING = "verifying"
    CALIBRATING = "calibrating"
    PUBLISHING = "publishing"
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


class OperationIntent(str, Enum):
    """What a deployment request is for, replacing overlapping booleans.

    ``do_deploy``/``run_smoke``/``preload`` mixed serving, preload-only, and
    smoke runs in three interacting flags (for vLLM ``do_deploy`` silently won
    over ``run_smoke``). Requests and jobs carry this explicit mode instead.
    """

    SERVE = "serve"
    PRELOAD = "preload"
    SMOKE = "smoke"


class AttemptDisposition(str, Enum):
    """How one deployment attempt ended for lifecycle/fallback decisions."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETAINED = "retained"


class CleanupDisposition(str, Enum):
    """Confirmed outcome of stopping a failed or cancelled resource."""

    CONFIRMED = "confirmed"
    RETAINED = "retained"
    FAILED = "failed"
    UNKNOWN = "unknown"
    # Nothing was allocated, so no stop was needed. Distinct from CONFIRMED
    # so "cancelled before allocating" never reads as "a stop succeeded".
    NOTHING_TO_CLEAN = "nothing_to_clean"


class WarmupStatus(str, Enum):
    """How a requested warmup/certification pass ended.

    Missing evidence is never success: a warmup that produced no explicit
    successful completion is FAILED, even when the deploy itself succeeded.
    SKIPPED means warmup was intentionally disabled, not verified.
    """

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class VisionMode(str, Enum):
    """Requested image input behavior."""

    AUTO = "auto"
    ON = "on"
    OFF = "off"


class VisionVerification(str, Enum):
    """Result of exercising image input on a particular deployment."""

    UNTESTED = "untested"
    PASSED = "passed"
    FAILED = "failed"
