# llm-launchpad

Launch and manage personal LLM inference on Modal from a Textual TUI.

## What is this?
- **Who it’s for:** developers who want an OpenAI-compatible endpoint for local or personal use without wiring up infrastructure by hand.
- **What it does:** provisions vLLM or llama.cpp backends on Modal, manages multiple named instances, and ships a TUI for deployment and monitoring.
- **Common uses:** spin up a coding model for your editor, test new quantizations, or manage multiple model variants behind clean endpoints.

## Why a TUI?

Launching LLM inference on Modal usually means juggling model names, GPU choices, warmup checks, logs, and endpoint details across several commands. The TUI keeps that flow in one place.

Use the TUI when you want to:
- launch a model without memorizing Modal or backend-specific commands
- compare `vLLM` and `llama.cpp` from one interface
- manage multiple deployed instances and inspect their status
- copy the final OpenAI-compatible base URL and model ID after deployment

`llm-launchpad` is designed around the terminal UI as the primary user experience.

If you see the command `llm-launchpad wizard`, that is an older alias for the same TUI.

## Prerequisites
- Python **3.12+**
- **uv** for environment + CLI management (`pip install uv` or `curl -Ls https://astral.sh/uv/install.sh | sh`)
- **Modal account + CLI**: `pip install modal` then run `modal setup` and paste your token
- Optional: Hugging Face auth for gated/private weights: `huggingface-cli login` or set `HUGGINGFACE_HUB_TOKEN`

## Install

Install the CLI so `llm-launchpad` is available directly in your shell:
```bash
pip install uv
uv tool install llm-launchpad
llm-launchpad --help
```

If uv says its tool directory is not on your `PATH`, run:
```bash
uv tool update-shell
```

## Quickstart
1) Make sure Modal is configured:
```bash
pip install modal
modal setup   # follow the prompt to authenticate
```

2) Launch the TUI:
```bash
llm-launchpad
```
Explicit TUI command:
```bash
llm-launchpad tui
```
Select a backend (vLLM or llama.cpp), choose a model/preset, and deploy. The TUI will show the instance name, endpoint URL, and final connection details.

3) Use the deployment details shown in the TUI to connect your client to the final OpenAI-compatible endpoint.

## Troubleshooting
- **Modal CLI errors / auth failed:** rerun `modal setup` and ensure your token is valid.
- **HF 403 or slow downloads:** run `huggingface-cli login` and optionally set `HF_XET_HIGH_PERFORMANCE=1`.
- **The TUI fails to open:** make sure the package installed cleanly and that Textual is available in the environment uv created for the tool.
- **GPU unavailable / quota:** pick a smaller GPU in the TUI or change `GPU_CONFIG`.
- **Endpoints return 503 during warmup:** wait a few minutes after first deploy; cold starts are expected.
- **Llama.cpp build errors:** ensure host CUDA ≥ 12.4 or switch to CPU/offload configs.

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
