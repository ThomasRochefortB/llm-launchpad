from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import os
import time
import fnmatch
import re
import subprocess
import sys
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
SNAPSHOT_MAX_WORKERS = _read_int_env("HF_SNAPSHOT_MAX_WORKERS", 16)


# --- Model configuration (from Hugging Face)
# Defaults can be overridden via the CLI local entrypoint or by writing config
REPO_ID = "unsloth/Qwen3-Coder-480B-A35B-Instruct-1M-GGUF"
QUANT = "Q4_K_M"


# --- Persistent cache for model weights (shared with vLLM)
cache_dir = "/root/.cache/huggingface"
model_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
CONFIG_PATH = f"{cache_dir}/serve_config.json"
HF_HUB_DIR = Path(cache_dir) / "hub"
_GGUF_QUANT_RE = re.compile(r"(Q\d(?:_[A-Z0-9]+)+|IQ\d+_[A-Z0-9_]+)", flags=re.IGNORECASE)

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


def _hub_repo_slug(repo_id: str) -> str:
    return repo_id.strip().replace("/", "--")


def _hub_model_dir(repo_id: str) -> Path:
    return HF_HUB_DIR / f"models--{_hub_repo_slug(repo_id)}"


def _read_hub_ref(model_dir: Path, revision: Optional[str]) -> Optional[str]:
    ref_name = (revision or "").strip() or "main"
    ref_path = model_dir / "refs" / ref_name
    if not ref_path.exists() or not ref_path.is_file():
        return None
    try:
        value = ref_path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    return value or None


def _resolve_hub_snapshot_dir(repo_id: str, revision: Optional[str]) -> Optional[Path]:
    model_dir = _hub_model_dir(repo_id)
    snapshots_root = model_dir / "snapshots"
    if not snapshots_root.exists() or not snapshots_root.is_dir():
        return None

    preferred = (revision or "").strip()
    candidate_names: list[str] = []
    if preferred:
        candidate_names.append(preferred)
        ref_sha = _read_hub_ref(model_dir, preferred)
        if ref_sha:
            candidate_names.append(ref_sha)
    else:
        ref_sha = _read_hub_ref(model_dir, "main")
        if ref_sha:
            candidate_names.append(ref_sha)

    seen: set[str] = set()
    for name in candidate_names:
        if name in seen:
            continue
        seen.add(name)
        candidate = snapshots_root / name
        if candidate.exists() and candidate.is_dir():
            return candidate

    snapshots = [entry for entry in snapshots_root.iterdir() if entry.is_dir()]
    if not snapshots:
        return None
    snapshots.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return snapshots[0]


def _collect_hub_gguf_matches(
    repo_id: str,
    revision: Optional[str],
    allow_patterns: list[str],
) -> list[str]:
    snapshot_dir = _resolve_hub_snapshot_dir(repo_id, revision)
    if snapshot_dir is None:
        return []
    matches: list[str] = []
    for gguf in snapshot_dir.glob("**/*.gguf"):
        if any(fnmatch.fnmatch(gguf.name, pat) for pat in allow_patterns):
            matches.append(str(gguf.relative_to(cache_dir)))
    matches.sort()
    return matches


def _extract_quant(path: str) -> Optional[str]:
    match = _GGUF_QUANT_RE.search(path.upper())
    if match:
        return match.group(1).upper()
    return None


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
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_ETAG_TIMEOUT": "30",
            "HF_HUB_DOWNLOAD_TIMEOUT": "120",
        }
    )
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
    if allow_patterns is None:
        allow_patterns = [f"*{QUANT}*.gguf"] if QUANT else ["*.gguf"]

    print(
        f"🦙 downloading {repo_id} (patterns: {allow_patterns}, revision: {revision}) "
        f"into {HF_HUB_DIR}"
    )

    _snapshot_download_with_keepalive(
        repo_id=repo_id,
        revision=revision,
        cache_dir=str(HF_HUB_DIR),
        allow_patterns=allow_patterns,
    )

    # Ensure other functions can see the writes before we quit
    model_cache.commit()

    # Discover matching GGUF files in the Hub cache snapshot.
    matches = _collect_hub_gguf_matches(repo_id, revision, allow_patterns)

    print(f"🦙 found GGUF entries: {matches}")
    return matches


def _snapshot_download_with_keepalive(
    repo_id: str,
    revision: Optional[str],
    cache_dir: str,
    allow_patterns: List[str],
    max_workers: int = SNAPSHOT_MAX_WORKERS,
) -> None:
    """Run snapshot_download in a subprocess and print periodic keepalives."""
    payload = {
        "repo_id": repo_id,
        "revision": revision,
        "cache_dir": cache_dir,
        "allow_patterns": allow_patterns,
        "max_workers": max_workers,
    }
    worker_code = (
        "import json, os\n"
        "from huggingface_hub import snapshot_download\n"
        "cfg = json.loads(os.environ['LLM_LAUNCHPAD_SNAPSHOT_CFG'])\n"
        "snapshot_download("
        "repo_id=cfg['repo_id'],"
        "revision=cfg.get('revision'),"
        "cache_dir=cfg['cache_dir'],"
        "allow_patterns=cfg['allow_patterns'],"
        "max_workers=cfg.get('max_workers', 8)"
        ")\n"
    )
    env = {
        **os.environ,
        "LLM_LAUNCHPAD_SNAPSHOT_CFG": json.dumps(payload),
    }
    process = subprocess.Popen(
        [sys.executable, "-c", worker_code],
        env=env,
    )
    keepalive_interval_seconds = 20
    started = time.time()
    last_bytes = -1
    stalled_intervals = 0
    while process.poll() is None:
        elapsed = int(time.time() - started)
        size_bytes, file_count = _estimate_matched_snapshot_size(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=allow_patterns,
        )
        size_gib = size_bytes / (1024**3)
        rate_mib_s = (size_bytes / (1024**2)) / elapsed if elapsed > 0 else 0.0
        if size_bytes > last_bytes:
            stalled_intervals = 0
        else:
            stalled_intervals += 1
        last_bytes = size_bytes
        stall_note = " (no growth detected)" if stalled_intervals >= 3 else ""
        print(
            "🦙 download in progress... "
            f"elapsed={elapsed}s files={file_count} size={size_gib:.2f}GiB "
            f"avg_rate={rate_mib_s:.2f}MiB/s{stall_note}"
        )
        time.sleep(keepalive_interval_seconds)
    if process.returncode != 0:
        raise RuntimeError(f"snapshot_download failed with exit code {process.returncode}")


def _estimate_matched_snapshot_size(
    repo_id: str,
    revision: Optional[str],
    allow_patterns: list[str],
) -> tuple[int, int]:
    """Estimate size of files matching allow_patterns in the resolved snapshot."""
    snapshot_dir = _resolve_hub_snapshot_dir(repo_id, revision)
    if snapshot_dir is None:
        return 0, 0

    total_bytes = 0
    file_count = 0
    for path in snapshot_dir.glob("**/*"):
        if not path.is_file():
            continue
        if allow_patterns and not any(fnmatch.fnmatch(path.name, pat) for pat in allow_patterns):
            continue
        file_count += 1
        try:
            total_bytes += path.stat().st_size
        except OSError:
            continue
    return total_bytes, file_count


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
    """List cached llama.cpp GGUF entries from Hugging Face Hub cache layout."""
    items: List[Dict[str, Any]] = []
    grouped: dict[tuple[str, Optional[str], Optional[str]], dict[str, Any]] = {}
    if HF_HUB_DIR.exists():
        for model_dir in sorted(HF_HUB_DIR.glob("models--*")):
            if not model_dir.is_dir():
                continue
            encoded = model_dir.name[len("models--") :]
            model_id = encoded.replace("--", "/")
            snapshots_root = model_dir / "snapshots"
            if not snapshots_root.exists() or not snapshots_root.is_dir():
                continue
            for snapshot_dir in snapshots_root.iterdir():
                if not snapshot_dir.is_dir():
                    continue
                revision = snapshot_dir.name
                for gguf in snapshot_dir.glob("**/*.gguf"):
                    rel = str(gguf.relative_to(cache_dir))
                    quant = _extract_quant(gguf.name)
                    key = (model_id, revision, quant)
                    entry = grouped.get(key)
                    if entry is None:
                        entry = {
                            "backend": "llamacpp",
                            "model_id": model_id,
                            "revision": revision,
                            "quant": quant,
                            "size_bytes": 0,
                            "file_count": 0,
                            "source_volume": "huggingface-cache",
                            "paths": [],
                        }
                        grouped[key] = entry
                    entry["size_bytes"] += int(gguf.stat().st_size)
                    entry["file_count"] += 1
                    entry["paths"].append(rel)

    return list(grouped.values())


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
    """Pick a GGUF file from HF hub cache."""
    pattern = f"**/*{quant}*.gguf" if quant else "**/*.gguf"

    if repo_id:
        snapshot_dir = _resolve_hub_snapshot_dir(repo_id, revision)
        scoped_candidates = list(snapshot_dir.glob(pattern)) if snapshot_dir else []
        if scoped_candidates:
            scoped_candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
            return scoped_candidates[0]

    candidates = list(HF_HUB_DIR.glob(f"models--*/snapshots/{pattern}"))
    if not candidates:
        if quant:
            raise RuntimeError(
                f"No GGUF files matching '*{quant}*.gguf' in HF cache {HF_HUB_DIR}"
            )
        raise RuntimeError(f"No GGUF files found in HF cache {HF_HUB_DIR}")
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
