# Repository Guidelines

## Project Structure & Module Organization
- Core package code lives in `llm_launchpad/`.
- CLI entrypoints are in `llm_launchpad/cli/` (`main.py` is the current command router).
- Deployment and orchestration logic is under `llm_launchpad/core/`.
- Shared protocol/event models are in `llm_launchpad/protocol/`.
- Textual UI code is in `llm_launchpad/tui/` (`screens/`, `widgets/`, `theme.tcss`).
- Modal app entrypoints are in `llm_launchpad/backends/`: `modal_llamacpp_app.py` and `modal_vllm_app.py`.
- Tests live in `tests/` and follow `test_*.py` naming.

## Build, Test, and Development Commands
- `uv sync`: install project dependencies from `pyproject.toml`/`uv.lock`.
- `uv run pytest`: run the full test suite.
- `uv run pytest tests/test_modal_gpu.py`: run a focused test file during iteration.
- `uv run llm-launchpad --help`: inspect CLI commands.
- `uv run llm-launchpad tui`: launch the Textual TUI locally.
- `modal deploy llm_launchpad/backends/modal_vllm_app.py` or `modal deploy llm_launchpad/backends/modal_llamacpp_app.py`: deploy a backend to Modal.

## Coding Style & Naming Conventions
- Follow Python 3.12+ style with 4-space indentation and PEP 8 spacing.
- Use type hints consistently (`Optional[...]`, concrete return types).
- Keep functions/modules `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE_CASE`.
- Prefer small, focused helpers in `core/` and keep CLI command functions thin.
- Use concise docstrings for public helpers and command behavior.

## Testing Guidelines
- Framework: `pytest` (tests currently use `unittest.TestCase` patterns and `unittest.mock`).
- Add tests in `tests/test_<feature>.py` and keep names behavior-focused, e.g. `test_fetch_modal_gpu_types_reads_docs_response`.
- Cover success and failure paths for CLI/core changes, especially Modal command construction and parsing logic.

## Commit & Pull Request Guidelines
- Existing history uses short, direct subjects (examples: `bugfix gpu dropdown`, `Adding HF hub cli`).
- Prefer imperative, specific commit messages, ideally under 72 chars (e.g., `Fix vLLM instance name slugging`).
- PRs should include: what changed, why, and local validation commands run.
- Link related issues, and include screenshots/GIFs for TUI screen changes.

## Security & Configuration Tips
- Never commit credentials (`modal setup`, HF tokens).
- Use environment variables for runtime config (`GPU_CONFIG`, `MODEL_NAME`, `HUGGINGFACE_HUB_TOKEN`).
