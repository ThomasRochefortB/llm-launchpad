import json
import os
from typing import Any

import aiohttp
import modal


APP_NAME = "vllm-server"
app = modal.App(APP_NAME)

N_GPU = int(os.environ.get("N_GPU", "1"))
GPU_CONFIG = os.environ.get("GPU_CONFIG", f"H100:{N_GPU}")
MINUTES = 60
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Thinking-2507-FP8")
MODEL_REVISION = os.environ.get(
    "MODEL_REVISION",
    "953532f942706930ec4bb870569932ef63038fdf",
)
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "llm")
FAST_BOOT = os.environ.get("FAST_BOOT", "true").lower() in {"1", "true", "yes", "on"}


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
    gpu=GPU_CONFIG,
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
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

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--uvicorn-log-level=info",
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--tensor-parallel-size",
        str(N_GPU),
    ]

    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

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
        await _send_request(session, SERVED_MODEL_NAME, messages)
        if twice:
            messages[0]["content"] = "You are Jar Jar Binks."
            print(f"Sending messages to {url}:", *messages, sep="\n\t")
            await _send_request(session, SERVED_MODEL_NAME, messages)


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
