"""Data models shared across the launchpad stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .enums import (
    BackendType,
    BillingModel,
    ComputeProvider,
    QuoteAvailability,
)


@dataclass(frozen=True)
class ModalProviderOptions:
    """Modal-specific deployment options kept behind the provider boundary."""


@dataclass(frozen=True)
class PrimeProviderOptions:
    """Prime-specific deployment options kept behind the provider boundary."""

    offer_id: Optional[str] = None
    region: Optional[str] = None
    disk_id: Optional[str] = None
    keep_failed_resource: bool = False
    allow_insecure_http: bool = False


ProviderOptions = ModalProviderOptions | PrimeProviderOptions


@dataclass
class LaunchpadSettings:
    """Persisted user settings (scaledown, etc.)."""

    scaledown_window: int = 1800  # seconds (30 minutes)

    def to_env(self) -> Dict[str, str]:
        """Derive Modal environment variables from settings."""
        env: Dict[str, str] = {}
        if self.scaledown_window > 0:
            env["SCALEDOWN_WINDOW"] = str(self.scaledown_window)
        return env

    def to_dict(self) -> Dict[str, Any]:
        return {"SCALEDOWN_WINDOW": self.scaledown_window}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LaunchpadSettings:
        raw_value = data.get("SCALEDOWN_WINDOW", data.get("scaledown_window", 1800))
        return cls(scaledown_window=int(raw_value))


@dataclass
class DeploymentConfig:
    """All parameters needed to execute a deployment."""

    backend: BackendType = BackendType.LLAMACPP
    provider: ComputeProvider = ComputeProvider.MODAL

    # llama.cpp specific
    preset: Optional[str] = None
    repo_id: Optional[str] = None
    quant: Optional[str] = None
    revision: Optional[str] = None
    preload: bool = True
    server_args: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    n_gpu_layers: Optional[int] = None
    llamacpp_image_no_cache: Optional[bool] = None
    gpu_type: Optional[str] = None
    gpu_count: Optional[int] = None
    required_vram_gb: Optional[float] = None

    # vLLM specific
    model_name: Optional[str] = None
    model_revision: Optional[str] = None
    served_model_name: Optional[str] = None
    fast_boot: Optional[bool] = None
    n_gpu: Optional[int] = None
    trust_remote_code: Optional[bool] = None
    reasoning_parser: Optional[str] = None
    tool_call_parser: Optional[str] = None
    default_chat_template_kwargs: Optional[str] = None

    # Deployment options
    do_deploy: bool = True
    run_smoke: bool = False
    do_warmup: bool = True
    show_debug_logs: bool = False

    # Instance identity
    instance_name: Optional[str] = None
    app_name: Optional[str] = None
    function_slug: Optional[str] = None

    # Provider-specific settings are typed and interpreted only by the adapter.
    provider_options: Optional[ProviderOptions] = None
    endpoint_api_key: Optional[str] = None


@dataclass(frozen=True)
class ProviderCapabilities:
    """Provider features used to filter recipes before requesting quotes."""

    provider: ComputeProvider
    supported_backends: frozenset[BackendType]
    billing_model: BillingModel
    live_availability: bool = False
    supports_regions: bool = False
    supports_spot: bool = False
    supports_secure_cloud: bool = False

    def supports_backend(self, backend: BackendType) -> bool:
        """Return whether this provider can serve the requested runtime."""
        return backend in self.supported_backends


@dataclass(frozen=True)
class InferenceRecipe:
    """Provider-neutral description of a runnable inference configuration."""

    id: str
    model_key: str
    display_name: str
    backend: BackendType
    model_id: str
    quant: Optional[str] = None
    max_context_tokens: Optional[int] = None
    required_vram_gb: Optional[float] = None
    server_args: tuple[str, ...] = ()
    source_label: str = "Curated"
    quality_score: Optional[float] = None
    quality_rank: Optional[int] = None


@dataclass(frozen=True)
class WorkloadProfile:
    """Small workload description used to normalize unlike billing models."""

    paid_hours_per_day: float = 8.0
    utilization: float = 0.25
    output_tokens_per_month: Optional[int] = None


@dataclass(frozen=True)
class ProviderQuote:
    """Normalized provider fulfillment option for an inference recipe."""

    id: str
    recipe_id: str
    provider: ComputeProvider
    provider_reference: str
    gpu_type: str
    gpu_count: int
    price_per_hour_usd: Optional[float]
    billing_model: BillingModel
    availability: QuoteAvailability = QuoteAvailability.UNKNOWN
    region: Optional[str] = None
    security: Optional[str] = None
    is_estimate: bool = True
    estimated_output_tokens_per_second: Optional[float] = None
    configuration_id: Optional[str] = None
    provider_options: Optional[ProviderOptions] = None


@dataclass(frozen=True)
class InferencePlan:
    """One deployable recipe bound to a compatible provider quote."""

    recipe: InferenceRecipe
    quote: ProviderQuote
    estimated_monthly_cost_usd: Optional[float] = None
    estimated_cost_per_million_output_tokens_usd: Optional[float] = None
    recommendation_reason: Optional[str] = None


@dataclass
class EndpointInfo:
    """A single row from `modal app list`."""

    name: str = ""
    app_id: str = ""
    state: str = "unknown"
    backend: Optional[BackendType] = None
    instance_name: Optional[str] = None
    web_url: Optional[str] = None
    served_model_name: Optional[str] = None
    display_name: Optional[str] = None
    model_name: Optional[str] = None
    repo_id: Optional[str] = None
    quant: Optional[str] = None
    runtime_status: Optional[str] = None
    runtime_status_detail: Optional[str] = None
    provider: ComputeProvider = ComputeProvider.MODAL
    endpoint_api_key: Optional[str] = None


@dataclass
class BenchmarkConfig:
    """Parameters for benchmarking a deployed OpenAI-compatible endpoint."""

    backend: BackendType = BackendType.LLAMACPP
    provider: ComputeProvider = ComputeProvider.MODAL
    app_name: Optional[str] = None
    instance_name: Optional[str] = None
    server_url: Optional[str] = None
    model_name: Optional[str] = None
    concurrency: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    request_count: Optional[int] = None
    input_tokens: int = 550
    output_tokens: int = 256
    tokenizer: str = "gpt2"
    request_timeout_seconds: int = 300
    output_dir: Optional[str] = None
    aiperf_args: list[str] = field(default_factory=list)
    api_key: Optional[str] = None


@dataclass(frozen=True)
class ComputeOffer:
    """Normalized GPU offer exposed by a compute provider."""

    id: str
    cloud_id: str
    provider_name: str
    gpu_type: str
    gpu_count: int
    gpu_memory_gb: Optional[float] = None
    region: Optional[str] = None
    data_center: Optional[str] = None
    country: Optional[str] = None
    socket: Optional[str] = None
    security: Optional[str] = None
    price_per_hour: Optional[float] = None
    is_spot: bool = False
    is_variable_price: bool = False
    stock_status: Optional[str] = None
    disk_default_gb: Optional[int] = None
    vcpu_default: Optional[int] = None
    memory_default_gb: Optional[int] = None
    images: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass
class BenchmarkConcurrencyResult:
    """Benchmark result for one concurrency value."""

    concurrency: int
    command: list[str]
    artifact_dir: str
    exit_code: int = 0
    success: bool = True
    detail: str = ""
    json_export_path: Optional[str] = None
    csv_export_path: Optional[str] = None
    metrics: dict[str, Optional[float]] = field(default_factory=dict)


@dataclass
class BenchmarkRunSummary:
    """Summary of a benchmark sweep."""

    config: BenchmarkConfig
    run_dir: str
    results: list[BenchmarkConcurrencyResult] = field(default_factory=list)
    success: bool = True
    best_concurrency: Optional[int] = None
    best_output_token_throughput: Optional[float] = None


@dataclass
class StoredModelInfo:
    """Single cached model entry in Modal storage."""

    backend: BackendType
    model_id: str
    revision: Optional[str] = None
    quant: Optional[str] = None
    size_bytes: int = 0
    file_count: int = 0
    source_volume: str = ""
    paths: list[str] | None = None
    incomplete: bool = False


@dataclass
class StorageSnapshot:
    """Cached model inventory grouped by backend."""

    llamacpp_models: list[StoredModelInfo]
    vllm_models: list[StoredModelInfo]

    @property
    def total_size_bytes(self) -> int:
        return sum(row.size_bytes for row in self.llamacpp_models + self.vllm_models)

    @property
    def total_models(self) -> int:
        return len(self.llamacpp_models) + len(self.vllm_models)
