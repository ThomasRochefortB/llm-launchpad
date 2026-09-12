"""Pinned llama.cpp runtime and streaming verification for Vast rentals."""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
import json
import shlex
import time

from huggingface_hub import get_token
import requests

from typing import Any

from ..protocol.enums import BackendType
from ..protocol.models import DeploymentConfig
from .runtime_support import load_llamacpp_support_manifest
from .serving_runtime import projector_setup, vllm_serve_args

VAST_RUNTIME_DIR = "/root/.llm-launchpad"
# CUDA_VERSION in the bundled b10689 image's immutable OCI config. Require
# native driver support instead of assuming CUDA forward compatibility on
# arbitrary marketplace GPUs: a paid run on a CUDA 12.2 RTX 3060 (offer
# 45598047, 2026-09-11) served chat, tools and a 60s stream, but nvidia-smi
# reported 25 MiB used on the device, so the weights were not on the GPU.
# Renting a GPU to run on CPU is worse than refusing the host.
# Recheck this alongside runtime image updates.
VAST_MIN_CUDA_VERSION = 12.8
# The bundled llama.cpp image is a CUDA 12.x build, which still supports
# Maxwell and newer. Read this from the image alongside its CUDA version.
VAST_MIN_COMPUTE_CAPABILITY = 5.0
# Vast bundles at most eight GPUs; the offer search already asks for that range.
VAST_MAX_GPU_COUNT = 8

GPU_INVENTORY_COMMAND = (
    "nvidia-smi --query-gpu=index,name,memory.total,memory.used "
    "--format=csv,noheader,nounits"
)


@dataclass(frozen=True)
class GpuDevice:
    """One GPU as the rented host actually reports it."""

    index: int
    name: str
    memory_total_mib: int
    memory_used_mib: int

    @property
    def memory_free_gib(self) -> float:
        return max(0, self.memory_total_mib - self.memory_used_mib) / 1024


def parse_gpu_inventory(output: str) -> tuple[GpuDevice, ...]:
    """Parse nvidia-smi CSV. Malformed rows are dropped, never guessed at."""
    devices: list[GpuDevice] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        try:
            devices.append(
                GpuDevice(
                    index=int(fields[0]),
                    name=fields[1],
                    memory_total_mib=int(fields[2]),
                    memory_used_mib=int(fields[3]),
                )
            )
        except ValueError:
            continue
    return tuple(sorted(devices, key=lambda device: device.index))


def verify_gpu_topology(
    devices: Sequence[GpuDevice], *, gpu_count: int, per_device_required_gb: float
) -> None:
    """Refuse a rental whose real devices do not match the approved placement.

    An offer advertises one GPU model and one memory size for every device it
    bundles. That is an assumption the memory plan divides by, not a promise,
    so confirm it against the host before serving anything.
    """

    if not devices:
        raise ValueError("The Vast host did not report any GPUs.")
    if len(devices) != gpu_count:
        raise ValueError(
            f"Vast rented {gpu_count} GPUs but the host exposes {len(devices)}."
        )
    names = {device.name.strip().casefold() for device in devices}
    if len(names) > 1:
        listed = ", ".join(sorted(device.name.strip() for device in devices))
        raise ValueError(
            f"This Vast host mixes GPU models ({listed}); the memory plan assumes identical devices."
        )
    if per_device_required_gb > 0:
        starved = min(devices, key=lambda device: device.memory_free_gib)
        if starved.memory_free_gib < per_device_required_gb:
            raise ValueError(
                f"GPU {starved.index} has {starved.memory_free_gib:.1f} GiB free of "
                f"{starved.memory_total_mib / 1024:.1f} GiB; the plan needs "
                f"{per_device_required_gb:.1f} GiB per device."
            )


@dataclass(frozen=True)
class VastRuntime:
    """A pinned image with the driver and architecture floors it was built for."""

    image: str
    min_cuda_version: float
    min_compute_capability: float


@lru_cache(maxsize=1)
def _vast_runtime_catalog() -> dict[str, Any]:
    """Load the bundled non-llama.cpp runtime metadata."""
    resource = resources.files("llm_launchpad.data").joinpath("vast_runtime.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("runtimes"), dict):
        raise RuntimeError("Vast runtime manifest must be a JSON object with runtimes")
    return payload["runtimes"]


def load_vast_runtime(backend: BackendType) -> VastRuntime:
    """Return the pinned image and driver floor for a non-llama.cpp runtime."""
    entry = _vast_runtime_catalog().get(backend.value)
    if not isinstance(entry, dict):
        raise ValueError(f"Vast has no published runtime for {backend.display_name}.")
    digest = str(entry.get("image_digest") or "")
    reference = str(entry.get("image_ref") or "")
    floor = entry.get("min_cuda_version")
    architecture = entry.get("min_compute_capability")
    if (
        not digest.startswith("sha256:")
        or not reference
        or not isinstance(floor, (int, float))
        or not isinstance(architecture, (int, float))
    ):
        raise ValueError(f"The Vast {backend.value} runtime entry is incomplete.")
    return VastRuntime(
        image=f"{reference.split('@')[0]}@{digest}",
        min_cuda_version=float(floor),
        min_compute_capability=float(architecture),
    )


# vLLM shards attention heads across devices, so the count must divide them.
# Powers of two are the shapes that hold for every supported architecture.
VLLM_TENSOR_PARALLEL_COUNTS = frozenset({1, 2, 4, 8})


def vast_refusal(config: DeploymentConfig) -> str | None:
    """Explain why a Vast rental cannot serve this configuration, or None.

    Provider-level checks (backend, vision, revision, GPU count) live in
    ``core/providers.py``; this covers what only the Vast runtime knows.
    """

    if config.backend == BackendType.VLLM:
        if not (config.model_name or "").strip():
            return "Vast vLLM rentals require a model name."
        count = config.gpu_count or 1
        if count not in VLLM_TENSOR_PARALLEL_COUNTS:
            return (
                f"vLLM tensor parallelism needs 1, 2, 4, or 8 GPUs; this rental has {count}."
            )
        if (config.n_gpu or count) != count:
            return (
                "A Vast rental bills every GPU it bundles, so tensor parallelism "
                f"must use all {count}."
            )
        try:
            load_vast_runtime(BackendType.VLLM)
        except ValueError as exc:
            return str(exc)
        return None
    if not config.repo_id or not config.quant:
        return "Vast requires a GGUF repository and quant."
    manifest = load_llamacpp_support_manifest(config.gguf_architecture)
    if manifest.build_recipe or not manifest.image_digest.startswith("sha256:"):
        return "Vast requires a published, digest-pinned runtime image for this architecture."
    return None


def vast_runtime(config: DeploymentConfig) -> VastRuntime:
    """Resolve a digest-pinned runtime before allocating a rental."""
    from .providers import refuse

    reason = refuse(config)
    if reason:
        raise ValueError(reason)
    if config.backend == BackendType.VLLM:
        return load_vast_runtime(BackendType.VLLM)
    manifest = load_llamacpp_support_manifest(config.gguf_architecture)
    return VastRuntime(
        image=manifest.image_ref.split("@")[0] + "@" + manifest.image_digest,
        min_cuda_version=VAST_MIN_CUDA_VERSION,
        min_compute_capability=VAST_MIN_COMPUTE_CAPABILITY,
    )


def vast_runtime_image(config: DeploymentConfig) -> str:
    """Return only the pinned image reference for this configuration."""
    return vast_runtime(config).image


def vast_runtime_script(config: DeploymentConfig) -> str:
    """Build a private startup script; callers transfer it only over SSH stdin."""
    vast_runtime_image(config)
    if not config.endpoint_api_key:
        raise ValueError("Vast endpoints require an API key.")
    if config.backend == BackendType.VLLM:
        return _vllm_script(config)
    arguments = [
        "/app/llama-server", "--hf-repo", f"{config.repo_id}:{config.quant}",
        *shlex.split(config.server_args or ""),
        "--host", "127.0.0.1", "--port", "8000",
        "--alias", config.served_model_name or "model",
    ]
    # The same pinned image Prime uses, so the projector stages identically.
    setup = ""
    if config.vision is not None and config.vision.enabled:
        setup, projector_path = projector_setup(config, root=VAST_RUNTIME_DIR)
        arguments.extend(["--mmproj", projector_path])
    else:
        arguments.append("--no-mmproj")
    if config.n_gpu_layers is not None:
        arguments.extend(["--n-gpu-layers", str(config.n_gpu_layers)])
    env = {
        "LLAMA_API_KEY": config.endpoint_api_key,
        "LLAMA_CACHE": f"{VAST_RUNTIME_DIR}/models",
        # SSH sessions do not preserve the image's library search environment.
        "LD_LIBRARY_PATH": "/app:/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
    }
    token = get_token()
    if token:
        env["HF_TOKEN"] = token
    lines = ["#!/bin/sh", "set -eu", "umask 077", f"mkdir -p {VAST_RUNTIME_DIR}/models"]
    lines.extend(f"export {name}={shlex.quote(value)}" for name, value in env.items())
    if setup:
        lines.append(setup)
    lines.append("exec " + shlex.join(arguments))
    return "\n".join(lines) + "\n"


def _vllm_script(config: DeploymentConfig) -> str:
    """Serve vLLM on the rental's loopback interface, keyed by environment."""
    arguments = list(vllm_serve_args(config, host="127.0.0.1", port=8000))
    env = {
        # vLLM reads this natively, so the key never reaches argv where anything
        # able to list processes on the host could read it.
        "VLLM_API_KEY": config.endpoint_api_key or "",
        "HF_HOME": f"{VAST_RUNTIME_DIR}/hf",
        "VLLM_CACHE_ROOT": f"{VAST_RUNTIME_DIR}/vllm",
        # Tensor-parallel workers are separate processes; spawn avoids the fork
        # interaction with CUDA that hangs multi-GPU startup.
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }
    token = get_token()
    if token:
        env["HF_TOKEN"] = token
    lines = [
        "#!/bin/sh", "set -eu", "umask 077",
        f"mkdir -p {VAST_RUNTIME_DIR}/hf {VAST_RUNTIME_DIR}/vllm",
    ]
    lines.extend(f"export {name}={shlex.quote(value)}" for name, value in env.items())
    lines.append("exec " + shlex.join(arguments))
    return "\n".join(lines) + "\n"


def endpoint_healthy(url: str, api_key: str) -> bool:
    """Probe loopback directly, without forwarding credentials to HTTP proxies."""
    try:
        with requests.Session() as session:
            session.trust_env = False
            with session.get(url + "/health", headers={"Authorization": f"Bearer {api_key}"}, timeout=5) as response:
                return response.status_code == 200
    except requests.RequestException:
        return False


def verify_endpoint_auth(url: str, model: str) -> None:
    """Reject endpoints that accept missing or incorrect bearer credentials."""
    try:
        with requests.Session() as session:
            session.trust_env = False
            for headers in ({}, {"Authorization": "Bearer invalid-launchpad-probe"}):
                with session.post(
                    url + "/v1/chat/completions", headers=headers,
                    json={"model": model, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1},
                    timeout=(5, 30),
                ) as response:
                    if response.status_code not in {401, 403}:
                        raise RuntimeError("Vast endpoint did not reject unauthorized chat requests.")
    except requests.RequestException:
        raise RuntimeError("Vast endpoint authentication verification failed.") from None


def verify_streaming(url: str, api_key: str, model: str) -> None:
    """Require a valid OpenAI SSE response and terminal marker before success."""
    try:
        with requests.Session() as session:
            session.trust_env = False
            with session.post(
                url + "/v1/chat/completions", headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "messages": [{"role": "user", "content": "Reply OK."}], "max_tokens": 8, "stream": True},
                timeout=(5, 30), stream=True,
            ) as response:
                response.raise_for_status()
                if "text/event-stream" not in response.headers.get("Content-Type", ""):
                    raise ValueError("Endpoint did not return an SSE stream.")
                saw_chunk = False
                # Say which limit ended the stream. This check is stricter than
                # the shared warmup calibration, which tolerates a missing
                # [DONE] entirely, so when it rejects a rental the user has
                # already paid for, the reason needs to be in the message.
                ended = "the server closed it"
                deadline = time.monotonic() + 60
                for index, line in enumerate(response.iter_lines()):
                    if index > 256:
                        ended = "it exceeded 256 lines"
                        break
                    if len(line) > 65536:
                        ended = "a line exceeded 64 KiB"
                        break
                    if time.monotonic() >= deadline:
                        ended = "it passed the 60s deadline"
                        break
                    if not line.startswith(b"data:"):
                        continue
                    value = line[5:].strip()
                    if value == b"[DONE]":
                        if saw_chunk:
                            return
                        ended = "[DONE] arrived before any content"
                        break
                    data = json.loads(value)
                    if isinstance(data, dict) and isinstance(data.get("choices"), list) and data["choices"]:
                        saw_chunk = True
    except (requests.RequestException, ValueError):
        raise RuntimeError("Vast endpoint failed streaming chat verification.") from None
    raise RuntimeError(
        "Vast chat stream ended without a valid completion marker: "
        f"{ended} after {'some' if saw_chunk else 'no'} content."
    )
