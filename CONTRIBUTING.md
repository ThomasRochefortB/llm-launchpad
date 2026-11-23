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
uv run llm-launchpad tui
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

## CI costs and local live checks

CI uses standard GitHub-hosted Ubuntu runners in this public repository and
offline provider tests. Build artifacts expire after one day. Keep provider
credentials out of workflows and keep the account's Actions budget at $0 with
"Stop usage when budget limit is reached" enabled to prevent paid overages.

The scheduled live canary has been removed because it rents billable compute.
`scripts/canary_live.py` rejects execution in GitHub Actions, even with `--live`.

Developers who explicitly want a paid integration check can run it locally:

```bash
uv run python scripts/canary_live.py --live --provider vast
```

It deploys `bartowski/Qwen2.5-0.5B-Instruct-GGUF` (Q4_K_M), requires a chat
answer during warmup, and stops the deployment. It uses local provider
credentials and skips providers without credentials. This optional check is
billable and is not a CI or release requirement.

## Release checklist

For a release tag such as `v0.0.3`:

1. Update `llm_launchpad/_version.py`.
2. Move the relevant notes from `Unreleased` into a new `CHANGELOG.md` release section.
3. Run `uv run pytest`, `uv run ruff check .`, and `uv run ty check`.
4. Commit the release changes and create the matching Git tag (`vX.Y.Z`).
5. Push the branch and tag so the publish workflow can build, smoke test, and publish the package.
