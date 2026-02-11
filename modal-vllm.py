import json
import os
from typing import Any

import aiohttp
import modal


APP_NAME = "vllm-server"
app = modal.App(APP_NAME)

MINUTES = 60
VLLM_PORT = 8000


def _read_required_str_env(name: str) -> str:
    raw = os.environ.get(name, "").strip()
    if not raw:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return raw


def _read_required_int_env(name: str) -> int:
    raw = _read_required_str_env(name)
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(f"Environment variable {name} must be an integer, got: {raw!r}") from None


def _read_required_bool_env(name: str) -> bool:
    raw = _read_required_str_env(name)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# These values are captured at deploy/run time by the Modal CLI process.
DEPLOY_N_GPU = _read_required_int_env("N_GPU")
DEPLOY_GPU_CONFIG = _read_required_str_env("GPU_CONFIG")
DEPLOY_MODEL_NAME = _read_required_str_env("MODEL_NAME")
DEPLOY_MODEL_REVISION = os.environ.get("MODEL_REVISION", "").strip() or None
DEPLOY_SERVED_MODEL_NAME = _read_required_str_env("SERVED_MODEL_NAME")
DEPLOY_FAST_BOOT = _read_required_bool_env("FAST_BOOT")

RUNTIME_ENV = {
    "MODEL_NAME": DEPLOY_MODEL_NAME,
    "SERVED_MODEL_NAME": DEPLOY_SERVED_MODEL_NAME,
    "FAST_BOOT": "true" if DEPLOY_FAST_BOOT else "false",
    "N_GPU": str(DEPLOY_N_GPU),
}
if DEPLOY_MODEL_REVISION:
    RUNTIME_ENV["MODEL_REVISION"] = DEPLOY_MODEL_REVISION


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
    fast_boot = _read_required_bool_env("FAST_BOOT")
    n_gpu = _read_required_int_env("N_GPU")

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
