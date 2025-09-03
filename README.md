# llm-launchpad

One-click personal LLM deployment with coding agent + chat UI.

## Qwen3‑Coder GGUF on Modal (llama.cpp)

Run `unsloth/Qwen3-Coder-480B-A35B-Instruct-1M-GGUF` on Modal using llama.cpp's HTTP server.

### Prerequisites
- Python 3.11+ and Modal CLI installed: `pip install modal`
- Login/configure Modal: `modal setup`
- Optional (if HF rate-limited/private): `huggingface-cli login` or set `HUGGINGFACE_HUB_TOKEN`

### Files
- Server entrypoint: `qwen3-coder-llamacpp.py`

### 1) Preload/download model weights (optional, recommended)
This downloads GGUF weights into a persistent Volume (`llamacpp-cache`).

```bash
modal run qwen3-coder-llamacpp.py
```

Common flags (defaults shown):
- `--preload True`
- `--repo-id "unsloth/Qwen3-Coder-480B-A35B-Instruct-1M-GGUF"`
- `--quant "Q4_K_M"`
- `--revision None`

Example without preloading:
```bash
modal run qwen3-coder-llamacpp.py --preload False
```

### 2) Deploy the HTTP server
Builds llama.cpp with CUDA and serves an OpenAI-compatible API on port 8080.

```bash
modal deploy qwen3-coder-llamacpp.py
```

Notes:
- First cold start can take many minutes; long timeouts are configured.
- During warmup you may see 503 responses; retry after a few minutes.

Get the public URL:
- Copy the web function URL printed by `modal deploy` (e.g. `https://<user>--qwen3-coder-llamacpp-serve.modal.run`).

Tail logs:
```bash
modal logs -f qwen3-coder-llamacpp.serve
```

### 3) Call the API
Set the server URL (replace with yours):
```bash
export SERVER_URL="https://<user>--qwen3-coder-llamacpp-serve.modal.run"
```
Completions endpoint:
```bash
curl -s -X POST \
  -H 'Content-Type: application/json' \
  -d '{"model": "default", "prompt": "Hello Qwen!"}' \
  "$SERVER_URL"/v1/completions
```

Chat completions endpoint:
```bash
curl -s -X POST \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "default",
        "messages": [
          {"role": "user", "content": "Write a Python function that reverses a string."}
        ]
      }' \
  "$SERVER_URL"/v1/chat/completions
```

### Tuning and configuration
- GPU type: edit `GPU_CONFIG` in `qwen3-coder-llamacpp.py`.
- Quantization: edit `QUANT` (default: `"Q4_K_M"`).
- Server args: edit `DEFAULT_SERVER_ARGS` (e.g., `--ctx-size`, `--threads`).
- If VRAM is insufficient, reduce GPU offload by lowering `--n-gpu-layers` or set `GPU_CONFIG = None` for CPU.

### Volumes
- Weights cache volume: `llamacpp-cache`
  - List files: `modal volume ls llamacpp-cache`
  - Explore: `modal shell --volume llamacpp-cache` (then `cd /mnt`)

### Troubleshooting
- Slow downloads: ensure `HF_HUB_ENABLE_HF_TRANSFER=1`.
- HF auth errors: login with `huggingface-cli login`.
- Build errors: ensure host CUDA >= 12.4, or switch to CPU.
