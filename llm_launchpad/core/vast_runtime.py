"""Pinned llama.cpp runtime and streaming verification for Vast rentals."""

import json
import shlex
import time

from huggingface_hub import get_token
import requests

from ..protocol.enums import BackendType
from ..protocol.models import DeploymentConfig
from .runtime_support import load_llamacpp_support_manifest

VAST_RUNTIME_DIR = "/root/.llm-launchpad"
# CUDA_VERSION in the bundled b10689 image's immutable OCI config. Require
# native driver support instead of assuming CUDA forward compatibility on
# arbitrary marketplace GPUs. Recheck this alongside runtime image updates.
VAST_MIN_CUDA_VERSION = 12.8


def vast_runtime_image(config: DeploymentConfig) -> str:
    """Resolve a digest-pinned runtime before allocating a rental."""
    if config.backend != BackendType.LLAMACPP or config.gpu_count != 1:
        raise ValueError("Vast deployment currently supports single-GPU llama.cpp only.")
    if not config.repo_id or not config.quant or config.revision:
        raise ValueError("Vast requires a GGUF repository and quant on the default revision.")
    if config.vision is not None and config.vision.enabled:
        raise ValueError("Vast deployment currently supports text models only.")
    manifest = load_llamacpp_support_manifest(config.gguf_architecture)
    if manifest.build_recipe or not manifest.image_digest.startswith("sha256:"):
        raise ValueError("Vast requires a published, digest-pinned runtime image for this architecture.")
    return manifest.image_ref.split("@")[0] + "@" + manifest.image_digest


def vast_runtime_script(config: DeploymentConfig) -> str:
    """Build a private startup script; callers transfer it only over SSH stdin."""
    vast_runtime_image(config)
    if not config.endpoint_api_key:
        raise ValueError("Vast endpoints require an API key.")
    arguments = [
        "/app/llama-server", "--hf-repo", f"{config.repo_id}:{config.quant}",
        *shlex.split(config.server_args or ""), "--no-mmproj",
        "--host", "127.0.0.1", "--port", "8000",
        "--alias", config.served_model_name or "model",
    ]
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
                deadline = time.monotonic() + 60
                for index, line in enumerate(response.iter_lines()):
                    if index > 256 or len(line) > 65536 or time.monotonic() >= deadline:
                        break
                    if not line.startswith(b"data:"):
                        continue
                    value = line[5:].strip()
                    if value == b"[DONE]":
                        if saw_chunk:
                            return
                        break
                    data = json.loads(value)
                    if isinstance(data, dict) and isinstance(data.get("choices"), list) and data["choices"]:
                        saw_chunk = True
    except (requests.RequestException, ValueError):
        raise RuntimeError("Vast endpoint failed streaming chat verification.") from None
    raise RuntimeError("Vast chat stream ended without a valid completion marker.")
