"""Data models shared across the launchpad stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .enums import BackendType


@dataclass
class LaunchpadSettings:
    """Persisted user settings (GPU config, scaledown, etc.)."""

    gpu_config: str = "A100-80GB:1"
    scaledown_window: int = 1800  # seconds (30 minutes)

    def to_env(self) -> Dict[str, str]:
        """Derive Modal environment variables from settings."""
        env: Dict[str, str] = {}
        if self.gpu_config.strip():
            env["GPU_CONFIG"] = self.gpu_config.strip()
        if self.scaledown_window > 0:
            env["SCALEDOWN_WINDOW"] = str(self.scaledown_window)
        return env

    def to_dict(self) -> Dict[str, Any]:
        return {
            "GPU_CONFIG": self.gpu_config,
            "SCALEDOWN_WINDOW": self.scaledown_window,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LaunchpadSettings:
        return cls(
            gpu_config=str(data.get("GPU_CONFIG", "A100-80GB:1")),
            scaledown_window=int(data.get("SCALEDOWN_WINDOW", 1800)),
        )


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

    # vLLM specific
    model_name: Optional[str] = None
    model_revision: Optional[str] = None
    served_model_name: Optional[str] = None
    fast_boot: Optional[bool] = None
    n_gpu: Optional[int] = None
    reasoning_parser: Optional[str] = None
    default_chat_template_kwargs: Optional[str] = None

    # Deployment options
    do_deploy: bool = True
    run_smoke: bool = False
    do_warmup: bool = True

    # Instance identity
    instance_name: Optional[str] = None
    app_name: Optional[str] = None


@dataclass
class EndpointInfo:
    """A single row from `modal app list`."""

    name: str = ""
    app_id: str = ""
    state: str = "unknown"
    backend: Optional[BackendType] = None
    instance_name: Optional[str] = None


