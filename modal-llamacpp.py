from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import os

import modal


# --- App configuration
APP_NAME = "llamacpp-server"
app = modal.App(APP_NAME)


# --- Hardware / build settings
# You can override this at deploy time with environment variable GPU_CONFIG
GPU_CONFIG = os.environ.get("GPU_CONFIG", "A100-80GB:1")
MINUTES = 60
LLAMA_CPP_RELEASE = None  # build from latest HEAD for newer arch support (e.g., qwen3moe)


# --- Model configuration (from Hugging Face)
# Defaults can be overridden via the CLI local entrypoint or by writing config
REPO_ID = "unsloth/Qwen3-Coder-480B-A35B-Instruct-1M-GGUF"
QUANT = "Q4_K_M"


# --- Persistent cache for model weights
cache_dir = "/root/.cache/llama.cpp"
model_cache = modal.Volume.from_name("llamacpp-cache", create_if_missing=True)
CONFIG_PATH = f"{cache_dir}/serve_config.json"

# --- Simple presets for convenience (model-agnostic)
# Note: Import presets lazily inside the local entrypoint to avoid container import issues.


def _save_config(config: Dict[str, Any]) -> None:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f)
    # Ensure other functions can see the writes before we quit
    model_cache.commit()


def _load_config() -> Dict[str, Any]:
    if Path(CONFIG_PATH).exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    # Fall back to module defaults
    return {
        "repo_id": REPO_ID,
        "quant": QUANT,
        "revision": None,
        "server_args": None,  # use DEFAULT_SERVER_ARGS
        "port": 8080,
        "host": "0.0.0.0",
        "n_gpu_layers": None,
    }


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
        allow_patterns = [f"*{QUANT}*.gguf"] if QUANT else ["*.gguf"]

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


def _resolve_model_entrypoint(quant: Optional[str]) -> Path:
    """Pick a GGUF file, optionally filtered by quant, preferring the largest file."""
    pattern = f"**/*{quant}*.gguf" if quant else "**/*.gguf"
    candidates = list(Path(cache_dir).glob(pattern))
    if not candidates:
        if quant:
            raise RuntimeError(f"No GGUF files matching '*{quant}*.gguf' in {cache_dir}")
        raise RuntimeError(f"No GGUF files found in {cache_dir}")
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


try:
    SCALEDOWN_WINDOW = int(os.environ.get("SCALEDOWN_WINDOW", str(30 * MINUTES)))
except Exception:
    SCALEDOWN_WINDOW = 30 * MINUTES


@app.function(
    image=image,
    volumes={cache_dir: model_cache},
    gpu=GPU_CONFIG,
    timeout=60 * MINUTES,  # allow long cold starts
    scaledown_window=SCALEDOWN_WINDOW,  # keep container warm after requests (overridable via env)
    max_containers=1,  # cap number of containers (replicas) to 1
)
@modal.web_server(8080, startup_timeout=30 * 60)
def serve():
    """Run llama.cpp's HTTP server for the specified model.

    This exposes an OpenAI-compatible API by default at /v1/chat/completions and /v1/completions.
    """
    import subprocess

    # Resolve configuration from persisted config (with sensible defaults)
    cfg = _load_config()
    model_repo_id = cfg.get("repo_id", REPO_ID)
    quant = cfg.get("quant", QUANT)
    revision = cfg.get("revision", None)
    extra_server_args: Optional[List[str]] = cfg.get("server_args")
    host = str(cfg.get("host", "0.0.0.0"))
    port = int(cfg.get("port", 8080))

    # Ensure the weights are present (download once and persist in Volume)
    allow_patterns = [f"*{quant}*.gguf"] if quant else ["*.gguf"]
    download_model.remote(model_repo_id, allow_patterns, revision)

    model_path = _resolve_model_entrypoint(quant)
    print(f"🦙 using model file: {model_path}")

    # offload all layers to GPU if configured, or use explicit override
    n_gpu_layers_cfg = cfg.get("n_gpu_layers")
    n_gpu_layers = int(n_gpu_layers_cfg) if n_gpu_layers_cfg is not None else (9999 if GPU_CONFIG else 0)

    command = [
        "/llama.cpp/llama-server",
        "--model",
        str(model_path),
        "--n-gpu-layers",
        str(n_gpu_layers),
        "--host",
        host,
        "--port",
        str(port),
    ] + (extra_server_args or DEFAULT_SERVER_ARGS)

    print("🦙 starting llama-server:")
    print(" ", " ".join(command))

    # Start server process; the web_server decorator will keep the container alive
    subprocess.Popen(command)


@app.function(image=download_image, volumes={cache_dir: model_cache})
def save_config_remote(config: Dict[str, Any]) -> None:
    """Persist configuration inside the Modal Volume so web server can read it.

    This avoids attempting to write to /root locally when running the local entrypoint.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f)
    model_cache.commit()


@app.local_entrypoint()
def main(
    preload: bool = True,
    preset: Optional[str] = None,
    repo_id: Optional[str] = None,
    quant: Optional[str] = None,
    revision: Optional[str] = None,
    server_args: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8080,
    n_gpu_layers: Optional[int] = None,
    deploy: bool = False,
):
    """Configure, optionally preload weights, and optionally deploy the server.

    Usage examples:
      modal run modal-llamacpp.py::main --preset qwen2.5-coder-7b --preload True --deploy True
      modal run modal-llamacpp.py::main --repo-id Qwen/Qwen2.5-Coder-7B-Instruct-GGUF --quant Q4_K_M --deploy True
    """
    # Lazy import here so containers importing this module don't need the package
    from llm_launchpad.presets import PRESETS

    # Merge preset with explicit arguments
    cfg: Dict[str, Any] = {}
    if preset:
        if preset not in PRESETS:
            print(f"⚠️ Unknown preset '{preset}'. Available: {', '.join(PRESETS.keys())}")
        else:
            cfg.update(PRESETS[preset])

    if repo_id is not None:
        cfg["repo_id"] = repo_id
    if quant is not None:
        cfg["quant"] = quant
    if revision is not None:
        cfg["revision"] = revision

    # Fill missing with defaults
    cfg.setdefault("repo_id", REPO_ID)
    cfg.setdefault("quant", QUANT)
    cfg.setdefault("revision", None)
    cfg["host"] = host
    cfg["port"] = int(port)
    cfg["n_gpu_layers"] = n_gpu_layers if n_gpu_layers is not None else None

    if server_args:
        # Tokenize simple space-separated string to list
        cfg["server_args"] = [arg for arg in server_args.split(" ") if arg]

    # Save the config inside the Modal Volume (remote) to avoid local filesystem issues
    save_config_remote.remote(cfg)
    print(f"📝 Saved config (remote volume): {json.dumps(cfg, indent=2)}")

    if preload:
        allow_patterns = [f"*{cfg['quant']}*.gguf"] if cfg.get("quant") else ["*.gguf"]
        download_model.remote(cfg["repo_id"], allow_patterns, cfg.get("revision"))
        print("✅ Weights cached in Modal Volume.")

    this_file = Path(__file__).resolve()
    print("\nNext steps:")
    print(f"1) Deploy the server: modal deploy {this_file}")
    print("2) Once deployed, curl the server (OpenAI-compatible):")
    print("   Use the URL printed by modal deploy, e.g.: https://<user>--llamacpp-server-serve.modal.run")
    print(
        "   curl -s -X POST "
        "-H 'Content-Type: application/json' "
        "-d '{\"model\": \"default\", \"prompt\": \"Hello!\"}' "
        f"https://<user>--llamacpp-server-serve.modal.run/v1/completions"
    )

    if deploy:
        try:
            import subprocess
            print("\n🚀 Deploying...")
            subprocess.run(["modal", "deploy", str(this_file)], check=True)
            print("✅ Deploy triggered. Check the Modal dashboard for status.")
        except Exception as e:
            print(f"⚠️ Failed to deploy automatically: {e}")


