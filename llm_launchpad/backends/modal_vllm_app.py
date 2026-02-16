import json
import os
from pathlib import Path
from typing import Any

import aiohttp
from coolname import generate_slug
import modal


def default_served_model_name(model_name: str | None, default: str = "llm") -> str:
    """Local helper so this file is self-contained in Modal runtimes."""
    candidate = (model_name or "").strip()
    if not candidate:
        return default
    tail = candidate.rsplit("/", 1)[-1].strip()
    return tail or default


APP_NAME = os.environ.get("MODAL_APP_NAME", "vllm-server").strip() or "vllm-server"
app = modal.App(APP_NAME)

MINUTES = 60
VLLM_PORT = 8000
HF_CACHE_DIR = "/root/.cache/huggingface"
HF_HUB_DIR = Path(HF_CACHE_DIR) / "hub"


def _slugify_name(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in raw.lower()).strip("-")


def _read_function_slug() -> str:
    raw = os.environ.get("MODAL_FUNCTION_SLUG", "").strip()
    if raw:
        slug = _slugify_name(raw)
        if slug:
            return slug
    return _slugify_name(generate_slug(2))


FUNCTION_SLUG = _read_function_slug()


def _function_name(base_name: str) -> str:
    return f"{base_name}-{FUNCTION_SLUG}"


def _read_str_env(name: str, default: str) -> str:
    raw = os.environ.get(name, "").strip()
    return raw or default


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
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# These values are captured at deploy/run time by the Modal CLI process.
DEPLOY_N_GPU = _read_int_env("N_GPU", 1)
DEPLOY_GPU_CONFIG = _read_str_env("GPU_CONFIG", "A100-80GB:1")
DEPLOY_MODEL_NAME = _read_str_env("MODEL_NAME", "Qwen/Qwen3-4B-Thinking-2507-FP8")
DEPLOY_MODEL_REVISION = os.environ.get("MODEL_REVISION", "").strip() or None
DEPLOY_SERVED_MODEL_NAME = _read_str_env("SERVED_MODEL_NAME", default_served_model_name(DEPLOY_MODEL_NAME))
DEPLOY_FAST_BOOT = _read_bool_env("FAST_BOOT", False)
DEPLOY_TRUST_REMOTE_CODE = _read_bool_env("TRUST_REMOTE_CODE", False)
DEPLOY_REASONING_PARSER = os.environ.get("REASONING_PARSER", "").strip() or None
DEPLOY_DEFAULT_CHAT_TEMPLATE_KWARGS = os.environ.get("DEFAULT_CHAT_TEMPLATE_KWARGS", "").strip() or None
PREDOWNLOAD_TIMEOUT_MINUTES = _read_int_env("PREDOWNLOAD_TIMEOUT_MINUTES", 6 * 60)

RUNTIME_ENV = {
    "MODEL_NAME": DEPLOY_MODEL_NAME,
    "SERVED_MODEL_NAME": DEPLOY_SERVED_MODEL_NAME,
    "FAST_BOOT": "true" if DEPLOY_FAST_BOOT else "false",
    "N_GPU": str(DEPLOY_N_GPU),
    "TRUST_REMOTE_CODE": "true" if DEPLOY_TRUST_REMOTE_CODE else "false",
}
if DEPLOY_MODEL_REVISION:
    RUNTIME_ENV["MODEL_REVISION"] = DEPLOY_MODEL_REVISION
if DEPLOY_REASONING_PARSER:
    RUNTIME_ENV["REASONING_PARSER"] = DEPLOY_REASONING_PARSER
if DEPLOY_DEFAULT_CHAT_TEMPLATE_KWARGS:
    RUNTIME_ENV["DEFAULT_CHAT_TEMPLATE_KWARGS"] = DEPLOY_DEFAULT_CHAT_TEMPLATE_KWARGS


hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.13.0",
        "huggingface-hub==0.36.0",
        "aiohttp>=3.9.5",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


@app.function(
    name=_function_name("predownload-model"),
    serialized=True,
    image=vllm_image,
    timeout=PREDOWNLOAD_TIMEOUT_MINUTES * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
def predownload_model(
    repo_id: str,
    revision: str | None = None,
) -> dict[str, Any]:
    """Download model weights into the shared HF cache volume."""
    from huggingface_hub import snapshot_download  # type: ignore

    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=HF_CACHE_DIR,
        allow_patterns=None,
    )
    hf_cache_vol.commit()
    return {"repo_id": repo_id, "revision": revision, "path": path}


@app.function(
    name=_function_name("list-downloaded-models"),
    serialized=True,
    image=vllm_image,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
def list_downloaded_models() -> list[dict[str, Any]]:
    """List model repos present in the huggingface-cache volume."""
    if not HF_HUB_DIR.exists():
        return []

    items: list[dict[str, Any]] = []
    for model_dir in sorted(HF_HUB_DIR.glob("models--*")):
        if not model_dir.is_dir():
            continue
        encoded = model_dir.name[len("models--") :]
        model_id = encoded.replace("--", "/")

        file_count = 0
        size_bytes = 0
        for file_path in model_dir.glob("snapshots/**/*"):
            if file_path.is_file():
                file_count += 1
                size_bytes += file_path.stat().st_size

        items.append(
            {
                "backend": "vllm",
                "model_id": model_id,
                "revision": None,
                "quant": None,
                "size_bytes": size_bytes,
                "file_count": file_count,
                "source_volume": "huggingface-cache",
                "paths": [str(model_dir.relative_to(HF_CACHE_DIR))],
            }
        )
    return items


@app.function(
    name=_function_name("serve"),
    serialized=True,
    image=vllm_image,
    gpu=DEPLOY_GPU_CONFIG,
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    secrets=[modal.Secret.from_dict(RUNTIME_ENV)],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve() -> None:
    import shlex
    import subprocess

    model_name = os.environ.get("MODEL_NAME", DEPLOY_MODEL_NAME).strip() or DEPLOY_MODEL_NAME
    model_revision = os.environ.get("MODEL_REVISION", "").strip() or None
    served_model_name = os.environ.get("SERVED_MODEL_NAME", DEPLOY_SERVED_MODEL_NAME).strip() or DEPLOY_SERVED_MODEL_NAME
    fast_boot = _read_bool_env("FAST_BOOT", DEPLOY_FAST_BOOT)
    n_gpu = _read_int_env("N_GPU", DEPLOY_N_GPU)
    trust_remote_code = _read_bool_env("TRUST_REMOTE_CODE", DEPLOY_TRUST_REMOTE_CODE)
    reasoning_parser = os.environ.get("REASONING_PARSER", "").strip() or DEPLOY_REASONING_PARSER
    default_chat_template_kwargs = (
        os.environ.get("DEFAULT_CHAT_TEMPLATE_KWARGS", "").strip() or DEPLOY_DEFAULT_CHAT_TEMPLATE_KWARGS
    )

    cmd = [
        "vllm",
        "serve",
        model_name,
        "--uvicorn-log-level=info",
        "--served-model-name",
        served_model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--tensor-parallel-size",
        str(n_gpu),
    ]

    if model_revision:
        cmd += ["--revision", model_revision]
    if reasoning_parser:
        cmd += ["--reasoning-parser", reasoning_parser]
    if default_chat_template_kwargs:
        cmd += ["--default-chat-template-kwargs", default_chat_template_kwargs]
    if trust_remote_code:
        cmd += ["--trust-remote-code"]
    cmd += ["--enforce-eager" if fast_boot else "--no-enforce-eager"]

    print("Starting vLLM command:")
    print(" ", shlex.join(cmd))
    subprocess.Popen(cmd)


@app.local_entrypoint()
async def test(
    test_timeout: int = 10 * MINUTES,
    content: str | None = None,
    twice: bool = True,
) -> None:
    url = serve.get_web_url()
    served_model_name = os.environ.get("SERVED_MODEL_NAME", DEPLOY_SERVED_MODEL_NAME).strip() or DEPLOY_SERVED_MODEL_NAME

    system_prompt = {
        "role": "system",
        "content": "You are a pirate who can't help but drop sly reminders that he went to Harvard.",
    }
    if content is None:
        content = "Explain the singular value decomposition."

    messages = [
        system_prompt,
        {"role": "user", "content": content},
    ]

    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Running health check for server at {url}")
        async with session.get("/health", timeout=test_timeout - 1 * MINUTES) as resp:
            up = resp.status == 200
        assert up, f"Failed health check for server at {url}"
        print(f"Successful health check for server at {url}")

        print(f"Sending messages to {url}:", *messages, sep="\n\t")
        await _send_request(session, served_model_name, messages)
        if twice:
            messages[0]["content"] = "You are Jar Jar Binks."
            print(f"Sending messages to {url}:", *messages, sep="\n\t")
            await _send_request(session, served_model_name, messages)


async def _send_request(
    session: aiohttp.ClientSession,
    model: str,
    messages: list[dict[str, str]],
) -> None:
    payload: dict[str, Any] = {"messages": messages, "model": model, "stream": True}
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    async with session.post("/v1/chat/completions", json=payload, headers=headers) as resp:
        async for raw in resp.content:
            resp.raise_for_status()
            line = raw.decode().strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):
                line = line[len("data: ") :]

            chunk = json.loads(line)
            assert chunk["object"] == "chat.completion.chunk"
            print(chunk["choices"][0]["delta"].get("content", ""), end="")
    print()
