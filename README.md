# llm-launchpad

One-click personal LLM deployment with coding agent + chat UI.

## Install

Using uv (recommended):
```bash
uv pip install llm-launchpad
```

Standard pip:
```bash
pip install llm-launchpad
```

## GGUF on Modal with llama.cpp

Deploy any GGUF model on Modal using llama.cpp's HTTP server. Includes presets for popular coding models.

### Prerequisites
- Python 3.11+ and Modal CLI installed: `pip install modal`
- Login/configure Modal: `modal setup`
- Optional (if HF rate-limited/private): `huggingface-cli login` or set `HUGGINGFACE_HUB_TOKEN`

### Files
- Server entrypoint: `modal-llamacpp.py`

### 1) Preload/download model weights (optional, recommended)
This downloads GGUF weights into a persistent Volume (`llamacpp-cache`).

Presets (recommended):
```bash
modal run modal-llamacpp.py::main --preset qwen3-coder-30b --preload
```

Custom repo/quant:
```bash
modal run modal-llamacpp.py::main \
  --repo-id Qwen/Qwen2.5-Coder-7B-Instruct-GGUF \
  --quant Q4_K_M --preload
```

Common flags:
- `--preload` (use `--no-preload` to disable)
- `--preset <name>` (see Presets below)
- `--repo-id <org/model>`
- `--quant <pattern>` (e.g., `Q4_K_M`)
- `--revision <hf-revision>`
- `--server_args "--ctx-size 65536 --threads 24"`
- `--host 0.0.0.0` `--port 8080`
- `--n_gpu_layers <int>`

### 2) Deploy the HTTP server
Builds llama.cpp with CUDA and serves an OpenAI-compatible API on port 8080.

```bash
modal deploy modal-llamacpp.py
```

Alternatively, one-click deploy directly from CLI (configure, preload, deploy):
```bash
modal run modal-llamacpp.py::main \
  --preset qwen3-coder-30b \
  --preload \
  --deploy
```

Notes:
- First cold start can take many minutes; long timeouts are configured.
- During warmup you may see 503 responses; retry after a few minutes.

Get the public URL:
- Copy the web function URL printed by `modal deploy` (e.g. `https://<user>--llamacpp-server-serve.modal.run`).

Tail logs:
```bash
modal app logs -f llamacpp-server.serve
```

### 3) Call the API
Set the server URL (replace with yours):
```bash
export SERVER_URL="https://<user>--llamacpp-server-serve.modal.run"
```
Completions endpoint:
```bash
curl -s -X POST \
  -H 'Content-Type: application/json' \
  -d '{"model": "default", "prompt": "Hello!"}' \
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
- **GPU shape**: set environment variable before deploy/run, e.g. `export GPU_CONFIG="A100-80GB:2"`.
- **Quantization**: pass `--quant` (default: `Q4_K_M`) or adjust presets.
- **Server args**: pass `--server_args "--ctx-size 65536 --threads 24"`.
- **GPU offload**: override with `--n_gpu_layers <int>` or rely on auto (all layers if GPU provided).
- **Persisted config**: settings are saved to `/root/.cache/llama.cpp/serve_config.json` and read by the server.

### Presets
Built-in examples (adjust as needed):
- `qwen3-coder-480b` → `unsloth/Qwen3-Coder-480B-A35B-Instruct-1M-GGUF`
- `qwen2.5-coder-7b` → `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF`
- `deepseek-coder-lite` → `TheBloke/deepseek-coder-6.7b-instruct-GGUF`

### Volumes
- Weights cache volume: `llamacpp-cache`
  - List files: `modal volume ls llamacpp-cache`
  - Explore: `modal shell --volume llamacpp-cache` (then `cd /mnt`)

### Troubleshooting
- Slow downloads: ensure `HF_HUB_ENABLE_HF_TRANSFER=1`.
- HF auth errors: login with `huggingface-cli login`.
- Build errors: ensure host CUDA >= 12.4, or switch to CPU.
