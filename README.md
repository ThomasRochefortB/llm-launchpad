<h1 align="center">LLM-Launchpad</h1>
<p align="center">Spin up personal LLM endpoints on Modal or Prime Intellect</p>
<p align="center">
  <img src="docs/assets/llm_launchpad_header.png" alt="LLM Launchpad header" width="900" />
</p>

- Deploy any open-source models from the Hugging Face model hub.
- OpenAI-compatible endpoints via llama.cpp (preferred) and vLLM backends.
- Direct integration with [OpenCode](https://github.com/anomalyco/opencode).

## Prerequisites
- **uv** for Python, environment, and CLI tool management (install with `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- A **Modal account**, a **Prime Intellect account**, or both
- **Hugging Face account**
- Optional: [OpenCode](https://github.com/anomalyco/opencode) (install with `curl -fsSL https://opencode.ai/install | bash`)

## Quickstart

Get up and running in four steps:

1. Install the CLI so `llm-launchpad` is available in your shell:
   ```bash
   uv tool install llm-launchpad
   llm-launchpad --help
   ```

2. Authenticate at least one compute provider:
   ```bash
   modal setup
   # or
   prime login
   ```

3. Authenticate Hugging Face:
   ```bash
   huggingface-cli login
   ```

4. Verify your setup:
   ```bash
   llm-launchpad doctor
   ```

5. Launch the TUI:
   ```bash
   llm-launchpad
   ```

## Why a TUI?

Setting up LLM endpoints usually means juggling model names, container images, GPU choices, warmup checks, logs, and endpoint details across several commands. The TUI keeps that flow in one place.

From the TUI you can:
- Deploy a popular model by picking it, choosing live GPU placement, and confirming
- Use Custom deploy for arbitrary Hugging Face llama.cpp or vLLM setups
- Manage multiple deployed instances and inspect their status
- Copy the OpenAI-compatible base URL, model ID, and API key after a successful deploy

Deploy is model-first: each model can have multiple runtime recipes and each
recipe can receive quotes from any compatible provider adapter. For details on
recommendations, the Artificial Analysis ranking, caching, and llama.cpp
runtime support, see the [deploy catalog guide](docs/catalog.md).

## Headless CLI examples

The TUI is the recommended path for interactive use, but the same workflows are available from the command line for scripts and repeatable operations. See the [full CLI reference](docs/cli.md) for every command and option.

Before anything else, verify your local setup:

```bash
llm-launchpad doctor
```

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

## Prime Intellect

Prime deploys use the cheapest matching fixed-price, secure on-demand offer,
provision a portable Ubuntu runtime over SSH, and publish the endpoint through
an HTTPS tunnel with bearer authentication:

```bash
llm-launchpad offers --gpu-type H100_80GB --gpu-count 1
llm-launchpad deploy \
  --provider prime \
  --backend vllm \
  --model-name Qwen/Qwen3-4B \
  --gpu-type H100_80GB \
  --gpu-count 1 \
  --do-warmup
```

See the [Prime Intellect provider guide](docs/prime.md) for the tunnel and
security model, persistent cache disks, and maintainer certification.

## OpenCode integration

LLM-Launchpad automatically detects a local OpenCode installation and syncs
your deployments into its config. Preview a sync without changing files:
```bash
llm-launchpad opencode sync --dry-run
```

## Documentation

- [Deploy catalog and recommendations](docs/catalog.md)
- [Prime Intellect provider](docs/prime.md)
- [Storage and costs](docs/storage-and-costs.md)
- [OpenCode integration](docs/opencode.md)
- [Troubleshooting and debug log](docs/troubleshooting.md)
- [Full CLI reference](docs/cli.md)

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
