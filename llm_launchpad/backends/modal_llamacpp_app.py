from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import os
import time
import fnmatch
import re
import shutil
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


def _read_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(f"Environment variable {name} must be an integer, got: {raw!r}") from None


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _read_optional_bool_env(name: str) -> Optional[bool]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return raw.lower() in {"1", "true", "yes", "on"}


def _default_llamacpp_served_model_name(
    repo_id: Optional[str],
    quant: Optional[str] = None,
    default: str = "default",
) -> str:
    candidate = (repo_id or "").strip()
    if not candidate:
        alias = default
    else:
        alias = candidate.rsplit("/", 1)[-1].strip() or default
    quant_text = (quant or "").strip()
    if not quant_text:
        return alias
    if quant_text.casefold() in alias.casefold():
        return alias
    return f"{alias}-{quant_text}"


def _server_args_define_alias(args: Optional[List[str]]) -> bool:
    if not args:
        return False
    return any(token in {"--alias", "-a"} for token in args)


PREDOWNLOAD_TIMEOUT_MINUTES = _read_int_env("PREDOWNLOAD_TIMEOUT_MINUTES", 6 * 60)
SNAPSHOT_MAX_WORKERS = _read_int_env("HF_SNAPSHOT_MAX_WORKERS", 32)
DOWNLOAD_CPU = _read_int_env("HF_DOWNLOAD_CPU", 4)
LLAMA_CPP_IMAGE_REF = (
    os.environ.get("LLAMA_CPP_IMAGE_REF", "ghcr.io/ggml-org/llama.cpp:server-cuda").strip()
    or "ghcr.io/ggml-org/llama.cpp:server-cuda"
)
LLAMA_CPP_IMAGE_NO_CACHE = _read_optional_bool_env("LLAMA_CPP_IMAGE_NO_CACHE")
# Prefer cached image layers by default. Set LLAMA_CPP_IMAGE_NO_CACHE=true (or LLAMA_CPP_IMAGE_FORCE_BUILD=true)
# to force a fresh pull/build from the latest tag.
_llama_cpp_force_build_override = _read_optional_bool_env("LLAMA_CPP_IMAGE_FORCE_BUILD")
if _llama_cpp_force_build_override is not None:
    LLAMA_CPP_IMAGE_FORCE_BUILD = _llama_cpp_force_build_override
elif LLAMA_CPP_IMAGE_NO_CACHE is not None:
    LLAMA_CPP_IMAGE_FORCE_BUILD = LLAMA_CPP_IMAGE_NO_CACHE
else:
    LLAMA_CPP_IMAGE_FORCE_BUILD = False
LLAMA_CPP_SERVER_BIN = os.environ.get("LLAMA_CPP_SERVER_BIN", "/app/llama-server").strip() or "/app/llama-server"


# --- Model configuration (from Hugging Face)
# Defaults can be overridden via the CLI local entrypoint or by writing config
REPO_ID = "unsloth/Qwen3-Coder-480B-A35B-Instruct-1M-GGUF"
QUANT = "Q4_K_M"


# --- Persistent cache for model weights (shared with vLLM)
HF_CACHE_VOLUME_NAME = "huggingface-cache"
HF_CACHE_DIR = "/root/.cache/huggingface"
cache_dir = HF_CACHE_DIR  # backwards-compatible alias used throughout this module
model_cache = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
CONFIG_PATH = f"{HF_CACHE_DIR}/serve_config.json"
HF_HUB_DIR = Path(HF_CACHE_DIR) / "hub"
_GGUF_QUANT_RE = re.compile(r"(Q\d(?:_[A-Z0-9]+)+|IQ\d+_[A-Z0-9_]+)", flags=re.IGNORECASE)
_GGUF_SPLIT_RE = re.compile(r"-(\d+)-of-(\d+)\.gguf$", flags=re.IGNORECASE)

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
        if _matches_any_pattern(gguf.name, allow_patterns):
            matches.append(str(gguf.relative_to(cache_dir)))
    for gguf in snapshot_dir.glob("**/*.GGUF"):
        if _matches_any_pattern(gguf.name, allow_patterns):
            matches.append(str(gguf.relative_to(cache_dir)))
    matches.sort()
    return matches


def _matches_any_pattern(name: str, patterns: list[str]) -> bool:
    """Case-insensitive fnmatch to tolerate mixed-case quant/file naming."""
    folded = name.casefold()
    return any(fnmatch.fnmatch(folded, pat.casefold()) for pat in patterns)


def _gguf_allow_patterns(quant: Optional[str]) -> list[str]:
    """Build HF allow_patterns that tolerate common case variants."""
    if not quant:
        return ["*.gguf", "*.GGUF"]

    quant_variants: list[str] = []
    for value in (quant, quant.upper(), quant.lower()):
        if value not in quant_variants:
            quant_variants.append(value)

    patterns: list[str] = []
    for q in quant_variants:
        for ext in (".gguf", ".GGUF"):
            pattern = f"*{q}*{ext}"
            if pattern not in patterns:
                patterns.append(pattern)
    return patterns


def _snapshot_gguf_candidates(snapshot_dir: Path, quant: Optional[str]) -> list[Path]:
    quant_folded = quant.casefold() if quant else None
    candidates: list[Path] = []
    for path in snapshot_dir.glob("**/*"):
        if not path.is_file():
            continue
        if path.suffix.casefold() != ".gguf":
            continue
        if quant_folded and quant_folded not in path.name.casefold():
            continue
        candidates.append(path)
    return candidates


def _raise_missing_gguf_match(
    repo_id: str,
    allow_patterns: list[str],
    revision: Optional[str],
) -> None:
    patterns = ", ".join(repr(pat) for pat in allow_patterns) if allow_patterns else "'*.gguf'"
    revision_text = f", revision={revision}" if revision else ""
    raise RuntimeError(
        "No GGUF files matched the requested repo/quant for llama.cpp "
        f"(repo_id={repo_id!r}, patterns=[{patterns}]{revision_text}). "
        "Use a GGUF repo and the correct quant name, or omit the quant filter."
    )


def _resolve_or_download_model_entrypoint(
    repo_id: str,
    revision: Optional[str],
    quant: Optional[str],
) -> Path:
    """Resolve a cached GGUF path first; download only on cache miss."""
    try:
        model_path = _resolve_model_entrypoint(repo_id, revision, quant)
        print(f"🦙 cache hit: using cached GGUF for {repo_id}")
        return model_path
    except RuntimeError as exc:
        print(f"🦙 cache miss for requested model ({exc}); downloading...")

    allow_patterns = _gguf_allow_patterns(quant)
    matches = download_model.remote(repo_id, allow_patterns, revision)
    if not matches:
        _raise_missing_gguf_match(repo_id, allow_patterns, revision)
    return _resolve_model_entrypoint(repo_id, revision, quant)


def _extract_quant(path: str) -> Optional[str]:
    match = _GGUF_QUANT_RE.search(path.upper())
    if match:
        return match.group(1).upper()
    return None


def _gguf_split_index(path: Path) -> Optional[int]:
    match = _GGUF_SPLIT_RE.search(path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _pick_preferred_gguf_entrypoint(candidates: list[Path]) -> Path:
    """Choose a loadable GGUF path, preferring the first shard for split GGUF sets."""
    split_first: list[Path] = []
    unsplit: list[Path] = []
    split_nonfirst: list[Path] = []

    for candidate in candidates:
        split_idx = _gguf_split_index(candidate)
        if split_idx is None:
            unsplit.append(candidate)
        elif split_idx == 1:
            split_first.append(candidate)
        else:
            split_nonfirst.append(candidate)

    def _largest(paths: list[Path]) -> Path:
        paths.sort(key=lambda p: (p.stat().st_size, str(p)), reverse=True)
        return paths[0]

    if split_first:
        return _largest(split_first)
    if unsplit:
        return _largest(unsplit)
    return _largest(split_nonfirst)


# --- Use official llama.cpp GHCR server image (CUDA build) and add Python for Modal functions
image = (
    modal.Image.from_registry(
        LLAMA_CPP_IMAGE_REF,
        add_python="3.12",
        force_build=LLAMA_CPP_IMAGE_FORCE_BUILD,
    )
    .entrypoint([])  # clear image entrypoint so Modal can run Python function code
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
    cpu=DOWNLOAD_CPU,
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
        allow_patterns = _gguf_allow_patterns(QUANT)

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
        complete_bytes, complete_files = _estimate_matched_snapshot_size(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=allow_patterns,
        )
        inflight_bytes, inflight_files = _estimate_incomplete_blob_size(repo_id=repo_id)
        observed_bytes = complete_bytes + inflight_bytes
        observed_files = complete_files + inflight_files
        size_gib = observed_bytes / (1024**3)
        rate_mib_s = (observed_bytes / (1024**2)) / elapsed if elapsed > 0 else 0.0
        if observed_bytes > last_bytes:
            stalled_intervals = 0
        else:
            stalled_intervals += 1
        last_bytes = observed_bytes
        stall_note = " (no growth detected)" if stalled_intervals >= 3 else ""
        print(
            "🦙 download in progress... "
            f"elapsed={elapsed}s files={observed_files} size={size_gib:.2f}GiB "
            f"complete={complete_files} inflight={inflight_files} "
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
        if allow_patterns and not _matches_any_pattern(path.name, allow_patterns):
            continue
        file_count += 1
        try:
            total_bytes += path.stat().st_size
        except OSError:
            continue
    return total_bytes, file_count


def _estimate_incomplete_blob_size(repo_id: str) -> tuple[int, int]:
    """Estimate active download size from .incomplete blob files."""
    blobs_dir = _hub_model_dir(repo_id) / "blobs"
    if not blobs_dir.exists() or not blobs_dir.is_dir():
        return 0, 0

    total_bytes = 0
    file_count = 0
    for path in blobs_dir.glob("*.incomplete"):
        if not path.is_file():
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
    cpu=DOWNLOAD_CPU,
)
def predownload_model(
    repo_id: str,
    quant: Optional[str] = None,
    revision: Optional[str] = None,
) -> List[str]:
    """Pre-download model files for storage management workflows."""
    allow_patterns = _gguf_allow_patterns(quant)
    return _download_model_files(repo_id, allow_patterns, revision)


@app.function(
    image=download_image,
    volumes={cache_dir: model_cache},
)
def list_downloaded_models() -> List[Dict[str, Any]]:
    """List cached llama.cpp GGUF entries from Hugging Face Hub cache layout."""
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
                            "source_volume": HF_CACHE_VOLUME_NAME,
                            "paths": [],
                        }
                        grouped[key] = entry
                    entry["size_bytes"] += int(gguf.stat().st_size)
                    entry["file_count"] += 1
                    entry["paths"].append(rel)

    return list(grouped.values())


@app.local_entrypoint()
def list_downloaded_models_json() -> None:
    """Print cached llama.cpp models as JSON for local tooling."""
    rows = list_downloaded_models.remote()
    print("LLM_LAUNCHPAD_STORAGE_JSON_BEGIN")
    print(json.dumps(rows, separators=(",", ":")))
    print("LLM_LAUNCHPAD_STORAGE_JSON_END")


DEFAULT_SERVER_ARGS = [
    "--ctx-size",
    "32768",
    "--threads",
    "16",
]


def _resolve_llama_server_binary() -> str:
    """Resolve llama-server binary path for official docker image with PATH fallback."""
    configured = os.environ.get("LLAMA_CPP_SERVER_BIN", LLAMA_CPP_SERVER_BIN).strip() or LLAMA_CPP_SERVER_BIN
    candidates: list[str] = []
    for candidate in (configured, "llama-server"):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        if os.path.isabs(candidate):
            if Path(candidate).exists():
                return candidate
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    raise RuntimeError(
        "Could not find llama-server binary. "
        f"Tried {candidates}. Set LLAMA_CPP_SERVER_BIN to the correct path in the image."
    )


def _llama_server_runtime_env(server_bin: str) -> tuple[dict[str, str], Optional[str]]:
    """Prepare env/cwd so official docker image shared libs resolve under Modal."""
    env = dict(os.environ)

    ld_entries: list[str] = []
    if os.path.isabs(server_bin):
        bin_dir = str(Path(server_bin).parent)
        if bin_dir:
            ld_entries.append(bin_dir)
    if "/app" not in ld_entries:
        ld_entries.append("/app")

    existing_ld = env.get("LD_LIBRARY_PATH", "").strip()
    if existing_ld:
        ld_entries.append(existing_ld)
    env["LD_LIBRARY_PATH"] = ":".join(entry for entry in ld_entries if entry)

    cwd: Optional[str] = "/app" if Path("/app").is_dir() else None
    return env, cwd


def _resolve_model_entrypoint(
    repo_id: Optional[str],
    revision: Optional[str],
    quant: Optional[str],
) -> Path:
    """Pick a GGUF file from HF hub cache."""
    if repo_id:
        snapshot_dir = _resolve_hub_snapshot_dir(repo_id, revision)
        if snapshot_dir is None:
            raise RuntimeError(
                f"No cached snapshot found for repo {repo_id!r} in HF cache {HF_HUB_DIR}"
            )
        scoped_candidates = _snapshot_gguf_candidates(snapshot_dir, quant)
        if scoped_candidates:
            return _pick_preferred_gguf_entrypoint(scoped_candidates)
        if quant:
            raise RuntimeError(
                f"No GGUF files matching '*{quant}*.gguf' found for repo {repo_id!r} "
                f"in cached snapshot {snapshot_dir}"
            )
        raise RuntimeError(
            f"No GGUF files found for repo {repo_id!r} in cached snapshot {snapshot_dir}"
        )

    candidates: list[Path] = []
    for snapshot_dir in HF_HUB_DIR.glob("models--*/snapshots/*"):
        if snapshot_dir.is_dir():
            candidates.extend(_snapshot_gguf_candidates(snapshot_dir, quant))
    if not candidates:
        if quant:
            raise RuntimeError(
                f"No GGUF files matching '*{quant}*.gguf' in HF cache {HF_HUB_DIR}"
            )
        raise RuntimeError(f"No GGUF files found in HF cache {HF_HUB_DIR}")
    return _pick_preferred_gguf_entrypoint(candidates)


try:
    SCALEDOWN_WINDOW = int(os.environ.get("SCALEDOWN_WINDOW", str(30 * MINUTES)))
except Exception:
    SCALEDOWN_WINDOW = 30 * MINUTES


@app.function(
    name=_function_name("serve"),
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
    served_model_name = str(
        cfg.get("served_model_name") or _default_llamacpp_served_model_name(model_repo_id, quant)
    ).strip() or _default_llamacpp_served_model_name(model_repo_id, quant)
    extra_server_args: Optional[List[str]] = cfg.get("server_args")
    host = str(cfg.get("host", "0.0.0.0"))
    port = int(cfg.get("port", 8080))

    # Prefer cached GGUFs already in the shared volume; download only on cache miss.
    model_path = _resolve_or_download_model_entrypoint(model_repo_id, revision, quant)
    print(f"🦙 using model file: {model_path}")

    # offload all layers to GPU if configured, or use explicit override
    n_gpu_layers_cfg = cfg.get("n_gpu_layers")
    n_gpu_layers = int(n_gpu_layers_cfg) if n_gpu_layers_cfg is not None else (9999 if GPU_CONFIG else 0)
    server_bin = _resolve_llama_server_binary()
    server_env, server_cwd = _llama_server_runtime_env(server_bin)
    print(f"🦙 using llama-server binary: {server_bin}")
    if server_cwd:
        print(f"🦙 using llama-server cwd: {server_cwd}")

    server_args = extra_server_args or DEFAULT_SERVER_ARGS
    command = [
        server_bin,
        "--model",
        str(model_path),
        "--n-gpu-layers",
        str(n_gpu_layers),
        "--host",
        host,
        "--port",
        str(port),
        "--metrics",  # Enable metrics endpoint
    ]
    if not _server_args_define_alias(server_args):
        command += ["--alias", served_model_name]
    command += server_args

    print("🦙 starting llama-server:")
    print(" ", " ".join(command))

    # Start server process; the web_server decorator will keep the container alive
    subprocess.Popen(command, env=server_env, cwd=server_cwd)


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
    cfg["served_model_name"] = _default_llamacpp_served_model_name(
        cfg.get("repo_id"),
        cfg.get("quant"),
    )
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
        allow_patterns = _gguf_allow_patterns(cfg.get("quant"))
        matches = download_model.remote(cfg["repo_id"], allow_patterns, cfg.get("revision"))
        if not matches:
            _raise_missing_gguf_match(cfg["repo_id"], allow_patterns, cfg.get("revision"))
        print(f"✅ Weights cached in Modal Volume ({len(matches)} GGUF file(s)).")

    this_file = Path(__file__).resolve()
    print("\nNext steps:")
    print(f"1) Deploy the server: modal deploy {this_file}")
    print("2) Once deployed, curl the server (OpenAI-compatible):")
    print("   Use the exact URL from the `Created web function serve => ...` line above.")
    print("   (Modal may truncate long labels and append a hash, so guessed hostnames can be wrong.)")
    print(
        "   curl -sS -X POST "
        "-H 'Content-Type: application/json' "
        f"-d '{{\"model\": \"{cfg['served_model_name']}\", \"prompt\": \"Hello!\", \"max_tokens\": 64, \"temperature\": 0.7}}' "
        "https://<ACTUAL_MODAL_WEB_URL>/v1/completions"
    )
    print("3) Faster iteration (avoids nested deploy after preload):")
    print(f"   modal run {this_file}::main --preload True")
    print(f"   modal deploy {this_file}")
    print("4) Image cache behavior (llama.cpp backend):")
    print("   Default: reuse cached image layers for faster runs.")
    print("   Force fresh latest image pull/build: set LLAMA_CPP_IMAGE_NO_CACHE=true")
    print("   Example: LLAMA_CPP_IMAGE_NO_CACHE=true modal deploy " + str(this_file))

    if deploy:
        try:
            import subprocess
            print(
                "\nℹ️ `--deploy` runs `modal deploy` after this `modal run`, so a second image build is expected."
            )
            print("\n🚀 Deploying...")
            subprocess.run(["modal", "deploy", str(this_file)], check=True)
            print("✅ Deploy triggered. Check the Modal dashboard for status.")
        except Exception as e:
            print(f"⚠️ Failed to deploy automatically: {e}")
