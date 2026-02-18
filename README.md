# llm-launchpad

One-click personal LLM deployment with coding agent + chat UI.

## Install

Using uv (recommended for CLI usage):
```bash
uv tool install llm-launchpad
```

## Endpoint Management From CLI

Manage deployed launchpad endpoints directly from `llm-launchpad`.
Launchpad now supports multi-instance deployments per backend. By default,
instance names are auto-derived from the model identifier.

List launchpad deployments:
```bash
llm-launchpad list
```

Deploy independent instances:
```bash
llm-launchpad deploy --backend vllm --model-name Qwen/Qwen3-4B-Thinking-2507-FP8
llm-launchpad deploy --backend vllm --model-name Qwen/Qwen2.5-7B-Instruct
```

Use explicit names when needed:
```bash
llm-launchpad deploy --backend vllm --model-name Qwen/Qwen3-4B-Thinking-2507-FP8 --instance-name qwen3
llm-launchpad deploy --backend llamacpp --app-name llamacpp-prod-coder
```

Check endpoint readiness:
```bash
llm-launchpad status --backend llamacpp
llm-launchpad status --backend vllm
llm-launchpad status --backend vllm --instance-name qwen3
```

Tail backend logs:
```bash
llm-launchpad logs --backend llamacpp
llm-launchpad logs --backend vllm
llm-launchpad logs --backend vllm --app-name vllm-qwen3
```

Stop a deployed backend:
```bash
llm-launchpad stop --backend llamacpp
llm-launchpad stop --backend vllm
llm-launchpad stop --backend vllm --instance-name qwen2-5
```

When multiple instances exist for one backend, `status`, `logs`, `stop`, and
`warmup` require `--instance-name` or `--app-name` to avoid ambiguous targeting.

## GGUF on Modal with llama.cpp

Deploy any GGUF model on Modal using llama.cpp's HTTP server. Includes presets for popular coding models.

### Prerequisites
- Python 3.11+ and Modal CLI installed: `pip install modal`
- Login/configure Modal: `modal setup`
- Optional (if HF rate-limited/private): `huggingface-cli login` or set `HUGGINGFACE_HUB_TOKEN`

### Files
- Server entrypoint: `llm_launchpad/backends/modal_llamacpp_app.py`

### 1) Preload/download model weights (optional, recommended)
This downloads GGUF weights into a persistent Volume (`llamacpp-cache`).

Presets (recommended):
```bash
modal run llm_launchpad/backends/modal_llamacpp_app.py::main --preset qwen3-coder-30b --preload
```

Custom repo/quant:
```bash
modal run llm_launchpad/backends/modal_llamacpp_app.py::main \
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
modal deploy llm_launchpad/backends/modal_llamacpp_app.py
```

Alternatively, one-click deploy directly from CLI (configure, preload, deploy):
```bash
modal run llm_launchpad/backends/modal_llamacpp_app.py::main \
  --preset qwen3-coder-30b \
  --preload \
  --deploy
```

Notes:
- First cold start can take many minutes; long timeouts are configured.
- During warmup you may see 503 responses; retry after a few minutes.

Get the public URL:
- Copy the web function URL printed by `modal deploy` (e.g. `https://<user>--llamacpp-qwen3-coder-serve.modal.run`).

Tail logs:
```bash
modal app logs -f llamacpp-server.serve
```

Shutdown / stop:
- If you run any streaming/local dev command (for example `modal app logs -f ...` or `modal serve ...`), stop it with `Ctrl+C`.
- To fully stop the deployed app (destructive, cannot be resumed), run:
  ```bash
  modal app stop llamacpp-server
  ```
- If needed, discover the exact app identifier first with `modal app list`.

### 3) Call the API
Set the server URL (replace with yours):
```bash
export SERVER_URL="https://<user>--llamacpp-qwen3-coder-serve.modal.run"
```
Completions endpoint:
```bash
curl -s -X POST \
  -H 'Content-Type: application/json' \
  -d '{"model": "default", "prompt": "Hello! How are you?"}' \
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

Metrics endpoint:
```bash
curl -s "$SERVER_URL"/metrics
```

### Tuning and configuration
- **GPU shape**: in `llm-launchpad wizard`, choose GPU type/count on each deploy screen so each instance can use its own shape. For raw `modal deploy/run`, set `GPU_CONFIG` manually.
- **Quantization**: pass `--quant` (default: `Q4_K_M`) or adjust presets.
- **Server args**: pass `--server_args "--ctx-size 65536 --threads 24"`.
- **GPU offload**: override with `--n_gpu_layers <int>` or rely on auto (all layers if GPU provided).
- **Persisted config**: settings are saved to `/root/.cache/huggingface/serve_config.json` and read by the server.

### Presets
Built-in examples (adjust as needed):
- `qwen3-coder-480b` → `unsloth/Qwen3-Coder-480B-A35B-Instruct-1M-GGUF`
- `qwen2.5-coder-7b` → `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF`
- `deepseek-coder-lite` → `TheBloke/deepseek-coder-6.7b-instruct-GGUF`

### Volumes
- Weights cache volume: `huggingface-cache` (HF default hub cache under `/root/.cache/huggingface/hub`)
  - List files: `modal volume ls huggingface-cache`
  - Explore: `modal shell --volume huggingface-cache` (then `cd /mnt`)

### Troubleshooting
- Slow downloads: prefer Xet (`hf-xet`) and set `HF_XET_HIGH_PERFORMANCE=1` (legacy `HF_HUB_ENABLE_HF_TRANSFER` is deprecated).
- HF auth errors: login with `huggingface-cli login`.
- Build errors: ensure host CUDA >= 12.4, or switch to CPU.

## vLLM on Modal (OpenAI-compatible)

Deploy a vLLM server on Modal using an OpenAI-compatible API, based on Modal's `vllm_inference` example.

### Files
- Server entrypoint: `llm_launchpad/backends/modal_vllm_app.py`

### 1) Deploy the server
```bash
modal deploy llm_launchpad/backends/modal_vllm_app.py
```

Or deploy via launchpad CLI with an explicit reasoning parser:
```bash
llm-launchpad deploy \
  --backend vllm \
  --model-name Qwen/Qwen3-8B \
  --reasoning-parser qwen3
```

Get the public URL:
- Copy the web function URL printed by `modal deploy`, e.g. `https://<user>--vllm-qwen3-4b-thinking-2507-fp8-serve.modal.run`.

### 2) Optional: run local smoke test against a fresh replica
This starts a replica, checks `/health`, then streams a chat completion.

```bash
modal run llm_launchpad/backends/modal_vllm_app.py
```

### 3) Call the API
Set the server URL (replace with yours):
```bash
export SERVER_URL="https://<user>--vllm-qwen3-4b-thinking-2507-fp8-serve.modal.run"
```

Chat completions endpoint:
```bash
curl -s -X POST \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "llm",
        "messages": [
          {"role": "user", "content": "Write a Python function that reverses a string."}
        ]
      }' \
  "$SERVER_URL"/v1/chat/completions
```

Swagger docs:
```bash
open "$SERVER_URL"/docs
```

### Configuration
In `llm-launchpad wizard`, these are set from deployment form fields (per deployment). If you use raw `modal deploy`, you can still set them via environment variables:
- `GPU_CONFIG` (default: `A100-80GB:1`)
- `N_GPU` (default: `1`, tensor parallel size; intentionally separate from `GPU_CONFIG` count)
- `MODEL_NAME` (default: `Qwen/Qwen3-4B-Thinking-2507-FP8`)
- `MODEL_REVISION` (default pinned revision in `llm_launchpad/backends/modal_vllm_app.py`)
- `SERVED_MODEL_NAME` (default: model id suffix, e.g. `Qwen3-4B-Thinking-2507-FP8`)
- `FAST_BOOT` (`true`/`false`, default: `false`)
- `TRUST_REMOTE_CODE` (`true`/`false`, default: `false`; required for some HF repos with custom modeling code)
- `REASONING_PARSER` (optional, e.g. `qwen3`, `deepseek_r1`, `granite`)
- `DEFAULT_CHAT_TEMPLATE_KWARGS` (optional JSON object passed to `--default-chat-template-kwargs`)
- `VLLM_PORT` (default: `8000`)

Model-specific thinking defaults:
- Qwen3 reasoning is enabled by default; disable server-wide with:
  ```bash
  export DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
  ```
- Granite and DeepSeek-V3.1 reasoning are disabled by default; enable with:
  ```bash
  export DEFAULT_CHAT_TEMPLATE_KWARGS='{"thinking": true}'
  ```

Request-level `chat_template_kwargs` continue to override server defaults.

Cached Modal volumes used by this backend:
- `huggingface-cache`
- `vllm-cache`
