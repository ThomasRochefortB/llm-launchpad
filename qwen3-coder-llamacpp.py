from pathlib import Path
from typing import List, Optional

import modal


# --- App configuration
APP_NAME = "qwen3-coder-llamacpp"
app = modal.App(APP_NAME)


# --- Hardware / build settings
GPU_CONFIG = "A100-80GB:6"  # Qwen3 480B is huge; recommend 8x H100-80GB for VRAM headroom
MINUTES = 60
LLAMA_CPP_RELEASE = None  # build from latest HEAD for newer arch support (e.g., qwen3moe)


# --- Model configuration (from Hugging Face)
REPO_ID = "unsloth/Qwen3-Coder-480B-A35B-Instruct-1M-GGUF"
QUANT = "Q4_K_M"


# --- Persistent cache for model weights
cache_dir = "/root/.cache/llama.cpp"
model_cache = modal.Volume.from_name("llamacpp-cache", create_if_missing=True)


# --- Build llama.cpp with CUDA support
cuda_version = "12.4.0"  # should be <= host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"


image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("git", "build-essential", "cmake", "curl", "libcurl4-openssl-dev")
    .run_commands("git clone https://github.com/ggerganov/llama.cpp")
    .run_commands(
        "cmake llama.cpp -B llama.cpp/build "
        "-DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON "
    )
    .run_commands(
        # build cli and server binaries
        "cmake --build llama.cpp/build --config Release -j --clean-first --target llama-quantize llama-cli llama-server"
    )
    .run_commands("cp llama.cpp/build/bin/llama-* llama.cpp")
    .entrypoint([])  # remove NVIDIA base container entrypoint
)


# --- Separate lightweight image for downloading models
download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_transfer]==0.26.2")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(image=download_image, volumes={cache_dir: model_cache}, timeout=30 * MINUTES)
def download_model(
    repo_id: str = REPO_ID,
    allow_patterns: Optional[List[str]] = None,
    revision: Optional[str] = None,
) -> List[str]:
    """Download model files from Hugging Face into a persistent Modal Volume.

    Returns a list of relative paths of GGUF files matching the quantization.
    """
    from huggingface_hub import snapshot_download

    if allow_patterns is None:
        allow_patterns = [f"*{QUANT}*.gguf"]

    print(f"🦙 downloading {repo_id} (patterns: {allow_patterns}, revision: {revision})")

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=cache_dir,
        allow_patterns=allow_patterns,
    )

    # Ensure other functions can see the writes before we quit
    model_cache.commit()

    # Discover matching GGUF files
    matches: List[str] = []
    for gguf in Path(cache_dir).glob("**/*.gguf"):
        if any(gguf.name.find(pat.strip("*")) != -1 for pat in allow_patterns):
            matches.append(str(gguf.relative_to(cache_dir)))

    print(f"🦙 found GGUF entries: {matches}")
    return matches


DEFAULT_SERVER_ARGS = [
    "--ctx-size",
    "32768",
    "--threads",
    "16",
]


def _resolve_model_entrypoint(quant: str) -> Path:
    """Pick a GGUF matching the quantization, preferring the largest file."""
    candidates = list(Path(cache_dir).glob(f"**/*{quant}*.gguf"))
    if not candidates:
        raise RuntimeError(f"No GGUF files matching '*{quant}*.gguf' in {cache_dir}")
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


@app.function(
    image=image,
    volumes={cache_dir: model_cache},
    gpu=GPU_CONFIG,
    timeout=60 * MINUTES,  # allow long cold starts
    scaledown_window=30 * MINUTES,  # keep container warm after requests
)
@modal.web_server(8080, startup_timeout=30 * 60)
def serve():
    """Run llama.cpp's HTTP server for the specified model.

    This exposes an OpenAI-compatible API by default at /v1/chat/completions and /v1/completions.
    """
    import subprocess

    # Resolve configuration from module constants
    model_repo_id = REPO_ID
    quant = QUANT
    revision = None
    extra_server_args: Optional[List[str]] = None

    # Ensure the weights are present (download once and persist in Volume)
    download_model.remote(model_repo_id, [f"*{quant}*.gguf"], revision)

    model_path = _resolve_model_entrypoint(quant)
    print(f"🦙 using model file: {model_path}")

    # offload all layers to GPU if configured
    n_gpu_layers = 9999 if GPU_CONFIG is not None else 0

    command = [
        "/llama.cpp/llama-server",
        "--model",
        str(model_path),
        "--n-gpu-layers",
        str(n_gpu_layers),
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
    ] + (extra_server_args or DEFAULT_SERVER_ARGS)

    print("🦙 starting llama-server:")
    print(" ", " ".join(command))

    # Start server process; the web_server decorator will keep the container alive
    subprocess.Popen(command)


@app.local_entrypoint()
def main(
    preload: bool = True,
    repo_id: str = REPO_ID,
    quant: str = QUANT,
    revision: Optional[str] = None,
):
    """Optionally preload model weights into the cache Volume, then print deploy info."""
    if preload:
        download_model.remote(repo_id, [f"*{quant}*.gguf"], revision)
        print("✅ Weights cached in Modal Volume.")

    this_file = Path(__file__).resolve()
    print("\nNext steps:")
    print(f"1) Deploy the server: modal deploy {this_file}")
    print("2) Once deployed, curl the server (OpenAI-compatible):")
    print("   Use the URL printed by modal deploy, e.g.: https://<user>--qwen3-coder-llamacpp-serve.modal.run")
    print(
        "   curl -s -X POST "
        "-H 'Content-Type: application/json' "
        "-d '{\"model\": \"default\", \"prompt\": \"Hello Qwen!\"}' "
        "https://<user>--qwen3-coder-llamacpp-serve.modal.run/v1/completions"
    )


