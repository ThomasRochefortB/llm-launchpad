"""Data models shared across the launchpad stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .enums import BackendType


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


@dataclass
class BenchmarkConfig:
    """Parameters for benchmarking a deployed OpenAI-compatible endpoint."""

    backend: BackendType = BackendType.LLAMACPP
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
