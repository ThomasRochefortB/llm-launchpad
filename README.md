<h1 align="center">LLM-Launchpad</h1>
<p align="center">Spin up LLM endpoints on Modal for local and personal use</p>
<p align="center">
  <img src="docs/assets/llm_launchpad_header.png" alt="LLM Launchpad header" width="900" />
</p>

- Deploy any open-source models from the Hugging Face model hub.
- OpenAI-compatible endpoints via llama.cpp (preferred) and vLLM backends.
- Direct integration with [OpenCode](https://github.com/anomalyco/opencode).

## Prerequisites
- **uv** for Python, environment, and CLI tool management (install with `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **Modal account**
- **Hugging Face account**
- Optional: [OpenCode](https://github.com/anomalyco/opencode) (install with `curl -fsSL https://opencode.ai/install | bash`)

## Quickstart

Get up and running in four steps:

1. Install the CLI so `llm-launchpad` is available in your shell:
   ```bash
   uv tool install llm-launchpad
   llm-launchpad --help
   ```

2. Authenticate Modal:
   ```bash
   modal setup
   ```

3. Authenticate Hugging Face:
   ```bash
   huggingface-cli login
   ```

4. Launch the TUI:
   ```bash
   llm-launchpad
   ```

## Why a TUI?

Setting up LLM endpoints usually means juggling model names, container images, GPU choices, warmup checks, logs, and endpoint details across several commands. The TUI keeps that flow in one place.

From the TUI you can:
- Launch any open-source model on the Hugging Face model hub without memorizing Modal or backend-specific commands
- Manage multiple deployed instances and inspect their status
- Integrate the final OpenAI-compatible base URL and model ID into your workflows like OpenCode after deployment.

## Headless CLI examples

The TUI is the recommended path for interactive use, but the same workflows are available from the command line for scripts and repeatable operations.

Deploy a vLLM endpoint and wait until it is ready:
```bash
llm-launchpad deploy \
  --backend vllm \
  --model-name Qwen/Qwen3-4B \
  --instance-name qwen3 \
  --do-warmup
```

Switch a llama.cpp instance to a Hugging Face GGUF model, redeploy it, and warm it up:
```bash
llm-launchpad switch \
  --backend llamacpp \
  --repo-id unsloth/Qwen3-4B-GGUF \
  --quant '*Q4_K_M.gguf' \
  --instance-name qwen3
```

Inspect and manage deployed apps:
```bash
llm-launchpad list
llm-launchpad status --backend llamacpp --instance-name qwen3
llm-launchpad logs --backend llamacpp --instance-name qwen3
llm-launchpad stop --backend llamacpp --instance-name qwen3 --yes
```

Sync existing Launchpad deployments into OpenCode without changing files first:
```bash
llm-launchpad opencode sync --dry-run
```

## Storage and cleanup

Downloaded model weights are cached in the Modal `huggingface-cache` volume so repeated deploys can start faster. Use the TUI Storage screen to refresh the cache inventory, predownload a model, or delete selected cached weights when they are no longer needed.

Stopping an app and deleting cached weights are separate operations: `llm-launchpad stop` stops a deployed Modal app, while the Storage screen manages cached model files. If storage size looks stale after a deployment or delete, refresh the Storage screen to reload the Modal volume snapshot.

## Costs and scaledown

GPU costs are controlled by the Modal resources selected for each deployment and by how long containers stay warm. LLM-Launchpad defaults the scaledown window to 1800 seconds, and you can change it in the Settings screen or with `SCALEDOWN_WINDOW` before deploying.

For predictable costs:
- Stop apps you no longer need with `llm-launchpad stop`.
- Prefer smaller GPU layouts for quick tests before moving to larger models.
- Use the warmup command only when you actually need the endpoint ready immediately.
- Treat displayed cost estimates as guidance and confirm current pricing in Modal for production workloads.

## OpenCode integration

LLM-Launchpad automatically detects local installation of OpenCode and will set up your OpenCode config with the final OpenAI-compatible base URL and model ID after deployment.

## Troubleshooting

- `Modal CLI not found`: reinstall or upgrade the package, then confirm `modal --help` works in the same shell.
- `Modal authentication missing`: run `modal setup`.
- Hugging Face download errors: run `huggingface-cli login` and verify the model license or gated-repo access in your Hugging Face account.
- Warmup stays queued: Modal may still be scheduling the requested GPU. Try a smaller GPU configuration or wait for capacity.
- Endpoint status fails after deploy: inspect `llm-launchpad logs --backend <backend> --instance-name <name>` for backend startup errors.
- SSH copy or selection feels wrong in the TUI: start with `llm-launchpad tui --no-mouse` to let the terminal handle native text selection.

## Development setup

If you are working from a clone and want the command available directly while editing the source:
```bash
git clone https://github.com/ThomasRochefortB/llm-launchpad.git
cd llm-launchpad
uv tool install --editable .
llm-launchpad --help
```

If you need the full project environment for tests or local development workflows:
```bash
uv sync
uv run pytest
```
