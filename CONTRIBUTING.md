# Contributing

Thanks for contributing to `llm-launchpad`.

## Development setup

```bash
git clone https://github.com/ThomasRochefortB/llm-launchpad.git
cd llm-launchpad
uv sync --group dev
```

Run the main local validation commands before opening a pull request:

```bash
uv run pytest
uv run ruff check .
uv run ty check
```

Useful focused commands during iteration:

```bash
uv run pytest tests/test_cli_main.py
uv run llm-launchpad --help
uv run llm-launchpad tui --no-mouse
```

## Coding guidelines

- Target Python 3.12+.
- Keep CLI entrypoints thin and move long-running logic into `llm_launchpad/core/`.
- Keep cross-layer contracts in `llm_launchpad/protocol/`.
- Reuse `llm_launchpad/core/paths.py` and `BackendType.script` instead of hardcoding backend paths.
- Add behavior-focused tests for both success and failure paths when changing CLI or core logic.
- When adding or changing a CLI command, regenerate the reference with
  `uv run python scripts/generate_cli_reference.py` and commit the updated
  `docs/cli.md`.

## Pull requests

- Start from `.github/pull_request_template.md` and fill in every section.
- Keep commit messages short, imperative, and specific.
- Include the exact validation commands you ran locally.
- Add screenshots or GIFs for TUI changes.
- Link related issues or follow-up work when relevant.

## Release checklist

For a release tag such as `v0.0.3`:

1. Update `llm_launchpad/_version.py`.
2. Move the relevant notes from `Unreleased` into a new `CHANGELOG.md` release section.
3. Run `uv run pytest`, `uv run ruff check .`, and `uv run ty check`.
4. Commit the release changes and create the matching Git tag (`vX.Y.Z`).
5. Push the branch and tag so the publish workflow can build, smoke test, and publish the package.
