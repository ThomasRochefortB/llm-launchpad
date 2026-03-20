# llm-launchpad

Easily launch and manage inference of open-source LLMs on Modal from a Python terminal UI (TUI) for local and personal use.

## What is this?
- A Python-only CLI tool built with Textual, a Python library for building UIs.
- **Who it’s for:** developers who want to quickly spin up an OpenAI-compatible endpoint for local or personal use without wiring up infrastructure by hand.
- **What it does:** provisions vLLM or llama.cpp backends on Modal, manages multiple named instances, and ships a TUI for deployment and monitoring.
- **Common uses:** spin up a coding model for your editor, test new quantizations, or manage multiple model variants behind clean endpoints.

## Project docs

- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)

## Why a TUI?

Launching LLM inference on Modal usually means juggling model names, GPU choices, warmup checks, logs, and endpoint details across several commands. The TUI keeps that flow in one place.

From the TUI you can:
- launch any open-source model on the Hugging Face model hub without memorizing Modal or backend-specific commands
- manage multiple deployed instances and inspect their status
- Integrate the final OpenAI-compatible base URL and model ID into your workflows like OpenCode after deployment


## Prerequisites
- Python **3.12+**
- **uv** for environment + CLI management (`curl -Ls https://astral.sh/uv/install.sh | sh`)
- **Modal account**
- Optional: Hugging Face account with token for Hub access.

## Install

Install the CLI so `llm-launchpad` is available directly in your shell:
```bash
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
Select a backend (vLLM or llama.cpp), choose a model/preset, and deploy. The TUI will show the instance name, endpoint URL, and final connection details.

3) Use the deployment details shown in the TUI to connect your client to the final OpenAI-compatible endpoint.
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
