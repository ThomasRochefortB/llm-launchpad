from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import os
import time
import fnmatch
from uuid import uuid4

import modal


# --- App configuration
APP_NAME = os.environ.get("MODAL_APP_NAME", "llamacpp-server").strip() or "llamacpp-server"
app = modal.App(APP_NAME)


# --- Hardware / build settings
# You can override this at deploy time with environment variable GPU_CONFIG
GPU_CONFIG = os.environ.get("GPU_CONFIG", "A100-80GB:1")
MINUTES = 60


def _slugify_name(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in raw.lower()).strip("-")


def _read_function_slug() -> str:
    raw = os.environ.get("MODAL_FUNCTION_SLUG", "").strip()
    if raw:
        slug = _slugify_name(raw)
        if slug:
            return slug
    return _slugify_name(uuid4().hex[:8])


FUNCTION_SLUG = _read_function_slug()


def _function_name(base_name: str) -> str:
    return f"{base_name}-{FUNCTION_SLUG}"


def _current_modal_username() -> Optional[str]:
    """Best-effort lookup of the active Modal profile username."""
    try:
        import subprocess

        result = subprocess.run(
            ["modal", "profile", "current"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        value = (result.stdout or "").strip()
        if result.returncode == 0 and value:
            return value
    except Exception:
        pass
    return None


def _read_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(f"Environment variable {name} must be an integer, got: {raw!r}") from None


PREDOWNLOAD_TIMEOUT_MINUTES = _read_int_env("PREDOWNLOAD_TIMEOUT_MINUTES", 6 * 60)


# --- Model configuration (from Hugging Face)
# Defaults can be overridden via the CLI local entrypoint or by writing config
REPO_ID = "unsloth/Qwen3-Coder-480B-A35B-Instruct-1M-GGUF"
QUANT = "Q4_K_M"


# --- Persistent cache for model weights (shared with vLLM)
cache_dir = "/root/.cache/huggingface"
model_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
CONFIG_PATH = f"{cache_dir}/serve_config.json"
MODELS_ROOT = Path(cache_dir) / "models"
INDEX_PATH = Path(cache_dir) / "model_index.json"

# --- Simple presets for convenience (model-agnostic)
# Note: Import presets lazily inside the local entrypoint to avoid container import issues.


def _load_config() -> Dict[str, Any]:
    """Load configuration from the volume, with sane defaults."""
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


def _repo_slug(repo_id: str) -> str:
    return repo_id.strip().replace("/", "__")


def _revision_slug(revision: Optional[str]) -> str:
    value = (revision or "").strip()
    return value if value else "main"


def _model_dir(repo_id: str, revision: Optional[str]) -> Path:
    return MODELS_ROOT / _repo_slug(repo_id) / _revision_slug(revision)


def _load_index() -> list[dict[str, Any]]:
    if not INDEX_PATH.exists():
        return []
    try:
        with open(INDEX_PATH) as handle:
            payload = json.load(handle)
    except Exception:
        return []
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    return []


def _save_index(rows: list[dict[str, Any]]) -> None:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "w") as handle:
        json.dump(rows, handle)
    model_cache.commit()


def _upsert_index_entry(
    repo_id: str,
    revision: Optional[str],
    allow_patterns: list[str],
    relative_matches: list[str],
) -> None:
    rows = _load_index()
    key_repo = repo_id.strip()
    key_revision = (revision or "").strip() or None
    others: list[dict[str, Any]] = []
    for row in rows:
        if (
            str(row.get("repo_id", "")).strip() == key_repo
            and ((row.get("revision") or None) == key_revision)
        ):
            continue
        others.append(row)

    size_bytes = 0
    for rel in relative_matches:
        candidate = Path(cache_dir) / rel
        if candidate.exists():
            size_bytes += candidate.stat().st_size

    quant_hint = None
    for pat in allow_patterns:
        stripped = pat.strip("*").strip()
        if stripped and "gguf" not in stripped.lower():
            quant_hint = stripped
            break

    others.append(
        {
            "repo_id": key_repo,
            "revision": key_revision,
            "quant": quant_hint,
            "size_bytes": size_bytes,
            "file_count": len(relative_matches),
            "paths": relative_matches,
            "updated_at_unix": int(time.time()),
        }
    )
    _save_index(others)


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
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface-hub==0.36.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


@app.function(
    image=download_image,
    volumes={cache_dir: model_cache},
    timeout=PREDOWNLOAD_TIMEOUT_MINUTES * MINUTES,
)
def download_model(
    repo_id: str = REPO_ID,
    allow_patterns: Optional[List[str]] = None,
    revision: Optional[str] = None,
) -> List[str]:
    """Download model files from Hugging Face into a persistent Modal Volume.

    Returns a list of relative paths of GGUF files matching the quantization.
    """
    return _download_model_files(repo_id, allow_patterns, revision)


def _download_model_files(
    repo_id: str,
    allow_patterns: Optional[List[str]],
    revision: Optional[str],
) -> List[str]:
    from huggingface_hub import snapshot_download  # type: ignore

    if allow_patterns is None:
        allow_patterns = [f"*{QUANT}*.gguf"] if QUANT else ["*.gguf"]

    target_dir = _model_dir(repo_id, revision)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"🦙 downloading {repo_id} (patterns: {allow_patterns}, revision: {revision}) "
        f"into {target_dir}"
    )

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(target_dir),
        allow_patterns=allow_patterns,
    )

    # Ensure other functions can see the writes before we quit
    model_cache.commit()

    # Discover matching GGUF files
    matches: List[str] = []
    for gguf in target_dir.glob("**/*.gguf"):
        if any(fnmatch.fnmatch(gguf.name, pat) for pat in allow_patterns):
            matches.append(str(gguf.relative_to(cache_dir)))

    _upsert_index_entry(repo_id, revision, allow_patterns, matches)

    print(f"🦙 found GGUF entries: {matches}")
    return matches


@app.function(
    image=download_image,
    volumes={cache_dir: model_cache},
    timeout=PREDOWNLOAD_TIMEOUT_MINUTES * MINUTES,
)
def predownload_model(
    repo_id: str,
    quant: Optional[str] = None,
    revision: Optional[str] = None,
) -> List[str]:
    """Pre-download model files for storage management workflows."""
    allow_patterns = [f"*{quant}*.gguf"] if quant else ["*.gguf"]
    return _download_model_files(repo_id, allow_patterns, revision)


@app.function(
    image=download_image,
    volumes={cache_dir: model_cache},
)
def list_downloaded_models() -> List[Dict[str, Any]]:
    """List cached llama.cpp models with lightweight metadata."""
    rows = _load_index()
    items: List[Dict[str, Any]] = []
    for row in rows:
        items.append(
            {
                "backend": "llamacpp",
                "model_id": str(row.get("repo_id", "")).strip(),
                "revision": row.get("revision"),
                "quant": row.get("quant"),
                "size_bytes": int(row.get("size_bytes", 0) or 0),
                "file_count": int(row.get("file_count", 0) or 0),
                "source_volume": "huggingface-cache",
                "paths": list(row.get("paths", []) or []),
            }
        )

    # Compatibility: detect legacy flat-cache GGUF files not indexed yet.
    known_paths = {path for entry in items for path in entry.get("paths", [])}
    for gguf in Path(cache_dir).glob("**/*.gguf"):
        try:
            gguf.relative_to(MODELS_ROOT)
            continue
        except ValueError:
            pass
        rel = str(gguf.relative_to(cache_dir))
        if rel in known_paths:
            continue
        items.append(
            {
                "backend": "llamacpp",
                "model_id": f"legacy:{gguf.stem}",
                "revision": None,
                "quant": None,
                "size_bytes": int(gguf.stat().st_size),
                "file_count": 1,
                "source_volume": "huggingface-cache",
                "paths": [rel],
            }
        )
    return items


DEFAULT_SERVER_ARGS = [
    "--ctx-size",
    "32768",
    "--threads",
    "16",
]


def _resolve_model_entrypoint(
    repo_id: Optional[str],
    revision: Optional[str],
    quant: Optional[str],
) -> Path:
    """Pick a GGUF file, preferring model-specific cache directories."""
    pattern = f"**/*{quant}*.gguf" if quant else "**/*.gguf"

    # Primary: per-model multi-cache layout.
    if repo_id:
        scoped_dir = _model_dir(repo_id, revision)
        scoped_candidates = list(scoped_dir.glob(pattern))
        if scoped_candidates:
            scoped_candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
            return scoped_candidates[0]

    # Compatibility: legacy flat layout in cache root.
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

    model_path = _resolve_model_entrypoint(model_repo_id, revision, quant)
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
        "--metrics",  # Enable metrics endpoint
    ] + (extra_server_args or DEFAULT_SERVER_ARGS)

    print("🦙 starting llama-server:")
    print(" ", " ".join(command))

    # Start server process; the web_server decorator will keep the container alive
    subprocess.Popen(command)


@app.function(
    image=download_image,
    volumes={cache_dir: model_cache},
)
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
      modal run llm_launchpad/backends/modal_llamacpp_app.py::main --preset qwen2.5-coder-7b --preload True --deploy True
      modal run llm_launchpad/backends/modal_llamacpp_app.py::main --repo-id Qwen/Qwen2.5-Coder-7B-Instruct-GGUF --quant Q4_K_M --deploy True
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
        # Tokenize simple space-separated string to list; respect quotes
        import shlex
        cfg["server_args"] = shlex.split(server_args)

    # Save the config inside the Modal Volume (remote) to avoid local filesystem issues
    save_config_remote.remote(cfg)
    print(f"📝 Saved config (remote volume): {json.dumps(cfg, indent=2)}")

    if preload:
        allow_patterns = [f"*{cfg['quant']}*.gguf"] if cfg.get("quant") else ["*.gguf"]
        download_model.remote(cfg["repo_id"], allow_patterns, cfg.get("revision"))
        print("✅ Weights cached in Modal Volume.")

    this_file = Path(__file__).resolve()
    username = _current_modal_username()
    base_url = (
        f"https://{username}--{APP_NAME}-serve.modal.run"
        if username
        else f"https://$(modal profile current)--{APP_NAME}-serve.modal.run"
    )
    print("\nNext steps:")
    print(f"1) Deploy the server: modal deploy {this_file}")
    print("2) Once deployed, curl the server (OpenAI-compatible):")
    print(f"   Endpoint base URL: {base_url}")
    print(
        "   curl -sS -X POST "
        "-H 'Content-Type: application/json' "
        "-d '{\"model\": \"default\", \"prompt\": \"Hello!\", \"max_tokens\": 64, \"temperature\": 0.7}' "
        f"{base_url}/v1/completions"
    )

    if deploy:
        try:
            import subprocess
            print("\n🚀 Deploying...")
            subprocess.run(["modal", "deploy", str(this_file)], check=True)
            print("✅ Deploy triggered. Check the Modal dashboard for status.")
        except Exception as e:
            print(f"⚠️ Failed to deploy automatically: {e}")
