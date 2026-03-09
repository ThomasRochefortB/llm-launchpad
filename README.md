# llm-launchpad

Launch and manage personal LLM inference on Modal from a Textual TUI, with headless CLI commands available for automation.

## What is this?
- **Who it’s for:** developers who want an OpenAI-compatible endpoint for local or personal use without wiring up infrastructure by hand.
- **What it does:** provisions vLLM or llama.cpp backends on Modal, manages multiple named instances, and ships a TUI plus headless CLI.
- **Common uses:** spin up a coding model for your editor, test new quantizations, or manage multiple model variants behind clean endpoints.

## Why a TUI?

Launching LLM inference on Modal usually means juggling model names, GPU choices, warmup checks, logs, and endpoint details across several commands. The TUI keeps that flow in one place.

Use the TUI when you want to:
- launch a model without memorizing Modal or backend-specific commands
- compare `vLLM` and `llama.cpp` from one interface
- manage multiple deployed instances and inspect their status
- copy the final OpenAI-compatible base URL and model ID after deployment

`llm-launchpad` is designed so the terminal UI is the default experience for humans, while the headless CLI commands remain available for automation and scripting.

If you see the command `llm-launchpad wizard`, that is an older alias for the same TUI.

## Prerequisites
- Python **3.12+**
- **uv** for environment + CLI management (`pip install uv` or `curl -Ls https://astral.sh/uv/install.sh | sh`)
- **Modal account + CLI**: `pip install modal` then run `modal setup` and paste your token
- Optional: Hugging Face auth for gated/private weights: `huggingface-cli login` or set `HUGGINGFACE_HUB_TOKEN`
- cURL (for quick API checks)

## Install

CLI/TUI (global):
```bash
pip install uv
uv tool install llm-launchpad
llm-launchpad --help
```

From a clone (recommended for contributing):
```bash
git clone https://github.com/ThomasRochefortB/llm-launchpad.git
cd llm-launchpad
uv sync
uv run llm-launchpad --help
```

## Quickstart
1) Make sure Modal is configured:
```bash
pip install modal
modal setup   # follow the prompt to authenticate
```

2) Launch the TUI:
```bash
uv run llm-launchpad
```
Explicit TUI command:
```bash
uv run llm-launchpad tui
```
Select a backend (vLLM or llama.cpp), choose a model/preset, and deploy. The TUI will show the instance name, endpoint URL, and final connection details.

Prefer headless? Deploy a default vLLM instance with one command:
```bash
uv run llm-launchpad deploy \
  --backend vllm \
  --instance-name quickstart \
  --model-name Qwen/Qwen3-4B-Thinking-2507-FP8
```

3) Check readiness and grab the URL:
```bash
uv run llm-launchpad status --backend vllm --instance-name quickstart
```

4) Call the endpoint (replace with your URL from the status/logs output):
```bash
export SERVER_URL="https://<user>--vllm-quickstart-serve.modal.run"
curl -s -X POST \
  -H 'Content-Type: application/json' \
  -d '{"model": "llm", "messages": [{"role": "user", "content": "Hello!"}]}' \
  "$SERVER_URL"/v1/chat/completions
```

## Minimal end-to-end example (vLLM)
```bash
# 1) Authenticate Modal once
modal setup

# 2) Optional: auth to HF for private or rate-limited repos
huggingface-cli login  # or export HUGGINGFACE_HUB_TOKEN=<token>

# 3) Deploy with sensible defaults
uv run llm-launchpad deploy \
  --backend vllm \
  --instance-name demo \
  --model-name Qwen/Qwen3-4B-Thinking-2507-FP8

# 4) Wait for readiness, then copy the serve URL
uv run llm-launchpad status --backend vllm --instance-name demo

# 5) Call it
export SERVER_URL="https://<user>--vllm-demo-serve.modal.run"
curl -s "$SERVER_URL"/v1/health   # quick smoke
curl -s -X POST -H 'Content-Type: application/json' \
  -d '{"model": "llm", "messages": [{"role": "user", "content": "Write a haiku about Modal."}]}' \
  "$SERVER_URL"/v1/chat/completions
```

## Common configuration (env vars)
- `GPU_CONFIG`: GPU type/count (default `A100-80GB:1`; also accepts `A10G:1`, etc.)
- `N_GPU`: tensor parallel size for vLLM (default `1`, independent of `GPU_CONFIG` count)
- `MODEL_NAME`: HF repo id (default `Qwen/Qwen3-4B-Thinking-2507-FP8`)
- `MODEL_REVISION`: optional pinned revision for vLLM
- `SERVED_MODEL_NAME`: override exposed model name (defaults to model id suffix)
- `FAST_BOOT`: `true`/`false`, skips warm caches where possible (vLLM)
- `TRUST_REMOTE_CODE`: `true`/`false`, required for some HF repos with custom code
- `REASONING_PARSER`: `qwen3`, `deepseek_r1`, `granite`, etc. (vLLM)
- `DEFAULT_CHAT_TEMPLATE_KWARGS`: JSON for server-wide chat template args (vLLM)
- `VLLM_PORT`: port inside the Modal container (default `8000`)
- `HUGGINGFACE_HUB_TOKEN`: auth for private/gated HF models
- llama.cpp extras: `PRESET`, `REPO_ID`, `QUANT`, `SERVER_ARGS`, `N_GPU_LAYERS`

## Endpoint management from the CLI
- List deployments: `llm-launchpad list`
- Deploy instances: `llm-launchpad deploy --backend vllm --model-name Qwen/Qwen2.5-7B-Instruct`
- Name instances explicitly: `llm-launchpad deploy --backend vllm --model-name ... --instance-name qwen3`
- Check readiness: `llm-launchpad status --backend vllm --instance-name qwen3`
- Tail logs: `llm-launchpad logs --backend vllm --instance-name qwen3`
- Stop: `llm-launchpad stop --backend vllm --instance-name qwen3`

When multiple instances exist for one backend, `status`, `logs`, `stop`, and `warmup` require `--instance-name` or `--app-name` to avoid ambiguity.

## Troubleshooting
- **Modal CLI errors / auth failed:** rerun `modal setup` and ensure your token is valid.
- **HF 403 or slow downloads:** run `huggingface-cli login` and optionally set `HF_XET_HIGH_PERFORMANCE=1`.
- **The TUI fails to open:** install Textual (`pip install textual`) or use headless commands.
- **GPU unavailable / quota:** pick a smaller GPU in the TUI or change `GPU_CONFIG`.
- **Endpoints return 503 during warmup:** wait a few minutes after first deploy; cold starts are expected.
- **Llama.cpp build errors:** ensure host CUDA ≥ 12.4 or switch to CPU/offload configs.

## GGUF on Modal with llama.cpp

Deploy any GGUF model on Modal using llama.cpp's HTTP server. Includes presets for popular coding models.

### Prerequisites
- Python 3.12+ and Modal CLI installed: `pip install modal`
- Login/configure Modal: `modal setup`
- Optional (if HF rate-limited/private): `huggingface-cli login` or set `HUGGINGFACE_HUB_TOKEN`

### Files
- Server entrypoint: `llm_launchpad/backends/modal_llamacpp_app.py`

### 1) Preload/download model weights (optional, recommended)
This downloads GGUF weights into the shared persistent Volume (`huggingface-cache`) used by both llama.cpp and vLLM, so models appear in the storage menu.

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
Uses the official `ghcr.io/ggml-org/llama.cpp:server-cuda` image (plus a small Modal runtime layer for Python/Modal deps) and serves an OpenAI-compatible API on port 8080.

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
- `modal run ...::main --deploy` triggers a second `modal deploy` build by design (you will see two image-build phases).
- llama.cpp image cache behavior defaults to reuse for faster iteration. To force a fresh pull/build from the latest image tag:
  ```bash
  LLAMA_CPP_IMAGE_NO_CACHE=true modal deploy llm_launchpad/backends/modal_llamacpp_app.py
  ```
- For reproducible builds, pin `LLAMA_CPP_IMAGE_REF` to a digest and keep cache reuse enabled.

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
- **GPU shape**: in the `llm-launchpad` TUI, choose GPU type/count on each deploy screen so each instance can use its own shape. For raw `modal deploy/run`, set `GPU_CONFIG` manually.
- **Quantization**: pass `--quant` (default: `Q4_K_M`) or adjust presets.
- **Server args**: pass `--server_args "--ctx-size 65536 --threads 24"`.
- **GPU offload**: override with `--n_gpu_layers <int>` or rely on auto (all layers if GPU provided).
- **Persisted config**: settings are saved to `/root/.cache/huggingface/serve_config.json` and read by the server.
- **Image source**: override `LLAMA_CPP_IMAGE_REF` (default: `ghcr.io/ggml-org/llama.cpp:server-cuda`) or use `LLAMA_CPP_SERVER_BIN` for an alternate binary path if the upstream image layout changes.

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
- If `llama-server` fails with a missing shared library in the official image, make sure you are using a recent `llm-launchpad` version (the backend now sets the runtime library path for `/app` on Modal's legacy image builder).

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
In the `llm-launchpad` TUI, these are set from deployment form fields (per deployment). If you use raw `modal deploy`, you can still set them via environment variables:
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
