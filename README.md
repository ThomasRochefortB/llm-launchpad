## Qwen3-Coder GGUF on Modal via llama.cpp

Run `unsloth/Qwen3-Coder-480B-A35B-Instruct-1M-GGUF:Q4_K_M` on Modal using llama.cpp's HTTP server.

### Prerequisites
- Python 3.11+ and Modal CLI installed: `pip install modal`
- Login/configure Modal: `modal setup`
- Optional (if HF rate-limited/private): `huggingface-cli login` or set `HUGGINGFACE_HUB_TOKEN`

### Files
- Server: `/Users/thomas.rochefort-be/GitHub/modal_exp/llama_qwen_server.py`

### 1) Preload/download model weights (optional, recommended)
This downloads the GGUF once into a persistent Volume (`llamacpp-cache`).

```bash
modal run /Users/thomas.rochefort-be/GitHub/modal_exp/llama_qwen_server.py
```

Flags you can pass (defaults shown):
- `--preload=True`
- `--repo-id="unsloth/Qwen3-Coder-480B-A35B-Instruct-1M-GGUF"`
- `--quant="Q4_K_M"`
- `--revision=None`

Example without preloading:
```bash
modal run /Users/thomas.rochefort-be/GitHub/modal_exp/llama_qwen_server.py --preload=False
```

### 2) Deploy the HTTP server
Builds llama.cpp with CUDA and serves an OpenAI-compatible API on port 8080.

```bash
modal deploy /Users/thomas.rochefort-be/GitHub/modal_exp/llama_qwen_server.py
```

Notes on cold start & timeouts:
- Initial load can take a long time (tens of minutes). The function uses `startup_timeout=1800s`, `timeout=3600s`, and `scaledown_window=3600s` to allow long warmups and keep the container hot.
- During warmup you may see 503 responses. Retry after a few minutes or poll `/health`-like endpoints if added.

Get the public URL:
- Copy the web function URL printed by `modal deploy` (e.g., `https://<user>--qwen3-coder-llamacpp-serve.modal.run`).
  - From your log example: `https://thomasrochefortb--qwen3-coder-llamacpp-serve.modal.run`

Tail logs (replace with your app/function if needed):
```bash
modal logs -f qwen3-coder-llamacpp.serve
```

### 3) Call the API
Set the server URL (replace with yours):
```bash
export SERVER_URL="https://thomasrochefortb--qwen3-coder-llamacpp-serve.modal.run"
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
- GPU type: edit `GPU_CONFIG` in `llama_qwen_server.py` (default: `"L40S:1"`).
- Quantization: edit `QUANT` (default: `"Q4_K_M"`).
- Server args: edit `DEFAULT_SERVER_ARGS` (e.g., `--ctx-size`, `--threads`).
- If VRAM is insufficient, reduce GPU offload by passing fewer layers (edit code to add `"--n-gpu-layers", "<n>"` in the command) or switch to CPU by setting `GPU_CONFIG = None`.

### Volumes
- Weights cache volume: `llamacpp-cache`
    - List files: `modal volume ls llamacpp-cache`
    - Explore: `modal shell --volume llamacpp-cache` (then `cd /mnt`)

### Troubleshooting
- Slow downloads: ensure `HF_HUB_ENABLE_HF_TRANSFER=1` (already set in downloader image).
- HF auth errors: login with `huggingface-cli login`.
- Build errors: ensure host has CUDA >= 12.4 or reduce to CPU by setting `GPU_CONFIG = None`.


