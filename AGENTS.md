# Repository Guidelines

## Project Structure & Module Organization
- Core package code lives in `llm_launchpad/`.
- CLI entrypoints are in `llm_launchpad/cli/`; `main.py` owns both the default TUI launch path and the headless Typer commands (`deploy`, `warmup`, `list`, `status`, `logs`, `stop`, `switch`, `gpu-types`).
- Deployment and orchestration logic is under `llm_launchpad/core/`; keep subprocess and Modal CLI integration in `backend.py`, high-level workflows in `orchestrator.py`, auth helpers in `modal_auth.py`/`hf_auth.py`, naming logic in `naming.py`, and canonical backend script paths in `paths.py`.
- Shared protocol/event models are in `llm_launchpad/protocol/` (`enums.py`, `events.py`, `models.py`) and should remain the contract between CLI, orchestrator, and TUI workers.
- Textual UI code is in `llm_launchpad/tui/`; `app.py` is the entrypoint, `screens/` contains deploy/manage/monitor/settings/storage flows, `widgets/` holds reusable components, and `theme.tcss` defines styling.
- Llama.cpp preset definitions live in `llm_launchpad/presets.py`.
- Modal app entrypoints are in `llm_launchpad/backends/`: `modal_llamacpp_app.py` and `modal_vllm_app.py`.
- Tests live in `tests/` and follow `test_*.py` naming.

## Build, Test, and Development Commands
- `uv sync`: install project dependencies from `pyproject.toml`/`uv.lock`.
- `uv run pytest`: run the full test suite.
- `uv run pytest tests/test_cli_main.py`: run a focused test file during iteration.
- `uv run llm-launchpad --help`: inspect CLI commands.
- `uv run llm-launchpad`: launch the default Textual TUI flow.
- `uv run llm-launchpad tui --no-mouse`: launch the TUI with native terminal text selection enabled.
- `uv run llm-launchpad list` or `uv run llm-launchpad status --backend vllm --instance-name <name>`: exercise the headless management commands locally.
- `modal deploy llm_launchpad/backends/modal_vllm_app.py` or `modal deploy llm_launchpad/backends/modal_llamacpp_app.py`: deploy a backend to Modal.

## Coding Style & Naming Conventions
- Follow Python 3.12+ style with 4-space indentation and PEP 8 spacing.
- Use type hints consistently (`Optional[...]`, concrete return types).
- Keep functions/modules `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE_CASE`.
- Prefer small, focused helpers in `core/` and keep CLI command functions thin; push long-running behavior into orchestrator/backend helpers rather than growing Typer handlers.
- Keep cross-layer data shapes in `protocol/` dataclasses/enums instead of ad hoc dictionaries.
- Reuse `llm_launchpad/core/paths.py` and `BackendType.script` instead of hardcoding backend script paths in new code.
- Use concise docstrings for public helpers and command behavior.

## Testing Guidelines
- Framework: `pytest`; the suite primarily uses `unittest.TestCase` plus `unittest.mock`.
- Add tests in `tests/test_<feature>.py` and keep names behavior-focused, e.g. `test_fetch_modal_gpu_types_reads_docs_response`.
- Cover success and failure paths for CLI/core changes, especially Modal command construction, app naming, auth detection, warmup/status probing, and storage parsing.
- Prefer mocks/fakes over real Modal or Hugging Face calls; keep tests hermetic and fast.

## CI Checks
- Before opening a PR, ensure all CI checks pass locally:
  - `uv run ruff check .` — linting
  - `uv run ty check` — type checking
  - `uv run pytest` — test suite
  - `uv build --no-sources` — package build verification
- CI runs on every pull_request and push to main; failing checks block merges.

## Commit & Pull Request Guidelines
- Existing history uses short, direct subjects (examples: `bugfix gpu dropdown`, `Adding HF hub cli`).
- Prefer imperative, specific commit messages, ideally under 72 chars (e.g., `Fix vLLM instance name slugging`).
- PRs should include: what changed, why, and local validation commands run.
- When opening a PR, always start from `.github/pull_request_template.md` and fill it out instead of writing an ad hoc body.
- Link related issues, and include screenshots/GIFs for TUI screen changes.

## Security & Configuration Tips
- Never commit credentials (`modal setup`, HF tokens).
- User settings and local caches live under `~/.llm_launchpad/` (`settings.json`, `storage_snapshot.json`, `deployment_connection_summaries.json`); treat them as local state, not repo artifacts.
- Use environment variables for runtime config (`GPU_CONFIG`, `SCALEDOWN_WINDOW`, `MODEL_NAME`, `MODEL_REVISION`, `SERVED_MODEL_NAME`, `TRUST_REMOTE_CODE`, `REASONING_PARSER`, `TOOL_CALL_PARSER`, `DEFAULT_CHAT_TEMPLATE_KWARGS`).
- Hugging Face auth is handled by the local Hub login/token state used by `huggingface_hub`; do not hardcode or commit tokens in repo files or tests.
- `LLM_LAUNCHPAD_TUI_MOUSE` controls the default mouse/copy behavior for the TUI; prefer the CLI flag for one-off local testing.
- llama.cpp Modal backend image selection/refresh is env-driven (`LLAMA_CPP_IMAGE_REF`, `LLAMA_CPP_IMAGE_NO_CACHE`, `LLAMA_CPP_SERVER_BIN`); prefer cache reuse by default and use `LLAMA_CPP_IMAGE_NO_CACHE=true` only when forcing a fresh latest-image pull.
