# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
