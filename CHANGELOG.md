# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Vast.ai rentals (beta): Fast Deploy quotes, GPU filters, and serving-tier competition for verified on-demand NVIDIA offers. Supported text-only llama.cpp GGUF placements on one to eight GPUs can be rented; the endpoint is a loopback SSH tunnel on this computer. Offers whose runtime is unsupported remain comparisons. A rented host's GPUs are inventoried over SSH before serving, so a bundle with a different device count, mixed GPU models, or too little free memory per device is refused and destroyed. Stop destroys the rental and its disk.
- vLLM on Vast.ai rentals, on a digest-pinned image whose CUDA floor is read from its own OCI config (13.0, above llama.cpp's 12.8). Tensor parallelism uses every GPU the rental bundles and is limited to counts that can shard attention heads. The API key is passed by environment, never on the command line.
- Image input on Vast.ai rentals through Advanced deploy. The projector is staged on the rental over the same pinned llama.cpp image Prime uses. Fast Deploy still refuses vision for every provider, because vision working memory is not calibrated for guaranteed-fit placement.
- `llm-launchpad vast-auth` (login/status/logout), `vast connect`, `offers --provider vast`, and `--provider vast` on deploy/list/status/logs/stop. `doctor` reports a local Vast key as optional and does not authenticate it over the network.
- Advanced deploy can bind a selected Vast rental on llama.cpp. Disk size is quoted with the GPU.
- Docs for the beta and the broader release plan: `docs/vast.md`, `docs/vast-integration-plan.md`.
- Added an always-on rotating debug log under `~/.llm_launchpad/logs/` and routed previously silent failure paths (cache persistence, auth probing, log-tail cleanup, SSH key permissions) to it.
- Added `llm-launchpad doctor`, a self-check command that verifies the Modal CLI, Modal/Prime/Hugging Face authentication, the optional Artificial Analysis key, and state-directory writability, with fix hints per failure.
- Added `llm-launchpad --version`.
- Added documentation pages under `docs/` (deploy catalog, Prime provider, storage and costs, OpenCode, troubleshooting) and a generated CLI reference (`docs/cli.md`) via `scripts/generate_cli_reference.py`.
- Added GitHub issue templates for bug reports and feature requests.

### Changed
- Multi-GPU llama.cpp placements no longer predict faster single-request decoding. The estimate previously assumed roughly 1.6x more decode speed per extra GPU, but llama.cpp splits layers across devices by default, so one request walks them in sequence: extra GPUs buy memory, not speed. **This affects Modal and Prime as well as Vast** -- a multi-GPU placement that previously appeared in the Fastest or Balanced tier on that estimate may now rank lower. Measured attestations are unaffected and still take precedence.
- Deploy forms disable controls their provider ignores and say why, instead of greying them out silently. Vast.ai host/port are fixed by its SSH tunnel, image rebuilds are Modal-only, and llama.cpp cannot pin an HF revision on any marketplace provider.
- Deploy forms and `llm-launchpad deploy` now refuse a configuration its provider cannot run before anything is allocated, rather than failing after the deployment has been routed. Vast.ai previously accepted vLLM, vision, and pinned-revision configurations in the Advanced deploy form and rejected them later.
- Vision option fields keep visible labels, and projector/image details hide when vision is set to text only.

### Fixed
- Vast rental startup requires a successful per-instance SSH key attachment and omits the reverted container startup hook, which caused SSH failures. Live vLLM serving retesting is still required.
- Vast live validation uses production-style SSH startup, checks total runtime and transfer estimates with cleanup headroom, and retains recovery records after uncertain image-probe creation. The Advanced deploy vision check supports vLLM without requiring a GGUF projector.
- Uncertified Vast vLLM and image-input deployments now require `LLM_LAUNCHPAD_VAST_EXPERIMENTAL=1` before renting. Text-only llama.cpp remains enabled normally.
- Vast cleanup retries rate-limited account and instance identity checks, including reconciliation after uncertain creation, before destroying the rental.
- First llama.cpp Modal deploy after a large Hugging Face download no longer dies at the 30-minute web-server startup timeout. The GPU container sequentially hydrates GGUF shards (and a projector, if any) before `llama-server` starts, and the bind wait defaults to 90 minutes (`LLAMACPP_SERVE_STARTUP_TIMEOUT_MINUTES`).

## [1.1.1] - 2026-08-22

### Changed
- Updated runtime dependencies: `huggingface-hub` to 1.27.0, `requests` to 2.34.2, and `modal` to 1.5.0; the optional `benchmark` extra now uses `aiperf` 0.12.0.

### Security
- Upgraded locked transitive dependencies to clear Dependabot advisories: aiohttp, cbor2, h2 (with hpack), idna, pydantic-settings, Pygments, setuptools, starlette, and urllib3.
- Remaining aiohttp and Pillow advisories are pending an upstream `aiperf` release that relaxes its dependency caps (tracked in #63).

## [1.1.0] - 2026-06-17

### Added
- Added Modal Volume storage cost estimates to the TUI billing panel and Storage screen, using Modal's `$0.09 / GiB / month` list price and `1 TiB / month` free tier.
- Added storage cost helpers and tests for binary GiB conversion, list-rate estimates, billable storage after the free tier, and monthly cost estimates.

### Changed
- Refreshed Quick Deploy catalog data and ranking display, including AA coding index context and score-based sorting.
- Improved backend orchestration, configuration handling, and CI hygiene.
- Refactored deployment flow internals for cleaner launch and management behavior.

### Fixed
- Handled Modal metadata fetch failures during Quick Deploy catalog refreshes.

## [1.0.1] - 2026-03-25

### Fixed
- Fixed Modal backend invocation for installed PyPI and `uv tool` environments by switching llama.cpp and vLLM entrypoints from source-tree file paths to Python module references.
- Fixed a llama.cpp startup race after on-demand GGUF downloads by reloading the shared Hugging Face cache volume before resolving the freshly downloaded snapshot.

## [1.0.0] - 2026-03-24

### Added
- Curated llama.cpp quick deploy profiles for large coding models, with recommended GPU layouts, estimated hourly cost, and max context shown directly in the TUI.
- OpenCode integration that syncs Launchpad-managed deployments into your local OpenCode config and prunes stale provider entries when apps stop or disappear.
- Public PyPI packaging with validated wheel and source-distribution smoke tests for a standard `uv tool install llm-launchpad` workflow.

### Changed
- Installation and first-run onboarding were simplified around `uv tool install llm-launchpad`, bundled `modal`, and explicit Modal/Hugging Face setup steps.
- The deployment UI now surfaces Modal GPU hourly pricing, trims noisy llama.cpp toggles, and makes quick deploy and main-menu navigation faster to use.
- llama.cpp deployment and monitoring flows were hardened with safer download/app-management behavior and cheaper health checks during status polling.

### Fixed
- Improved TUI behavior for local and SSH sessions, including more reliable mouse selection, copy actions, footer shortcuts, and `uv tool` startup.
- Improved Modal auth detection and OpenCode provider naming so synced connections appear cleaner and stay aligned with active deployments.

### Notes
- This is the first stable public `1.0.0` release of `llm-launchpad`.

## [0.0.2] - 2026-03-19

### Added
- Initial packaged release of `llm-launchpad` with the Textual TUI, headless
  CLI management commands, and Modal vLLM / llama.cpp backends.

[Unreleased]: https://github.com/ThomasRochefortB/llm-launchpad/compare/v1.1.1...HEAD
[1.1.1]: https://github.com/ThomasRochefortB/llm-launchpad/releases/tag/v1.1.1
[1.1.0]: https://github.com/ThomasRochefortB/llm-launchpad/releases/tag/v1.1.0
[1.0.1]: https://github.com/ThomasRochefortB/llm-launchpad/releases/tag/v1.0.1
[1.0.0]: https://github.com/ThomasRochefortB/llm-launchpad/releases/tag/v1.0.0
[0.0.2]: https://github.com/ThomasRochefortB/llm-launchpad/releases/tag/v0.0.2
