# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- The main menu's Provider Billing panel reports Vast.ai credit beside Modal workspace spend and the Prime Intellect wallet. Vast bills against credit rather than invoicing, so the number that decides whether a rental can start is the one that was missing. An unconfigured key names `llm-launchpad vast-auth login` instead of reaching the network, and an owed balance is called out separately.
- Vast.ai as a third compute provider, alongside Modal and Prime Intellect: Fast Deploy quotes, GPU filters, and serving-tier competition for verified on-demand NVIDIA offers. Supported llama.cpp GGUF placements on one to eight GPUs can be rented; the endpoint is a loopback SSH tunnel on this computer. Offers a rental cannot actually serve are left out of Fast Deploy rather than listed as unselectable rows. A rented host's GPUs are inventoried over SSH before serving, so a bundle with a different device count, mixed GPU models, or too little free memory per device is refused and destroyed. Stop destroys the rental and its disk.
- vLLM on Vast.ai rentals, on a digest-pinned image whose CUDA floor is read from its own OCI config (13.0, above llama.cpp's 12.8). Tensor parallelism uses every GPU the rental bundles and is limited to counts that can shard attention heads. The API key is passed by environment, never on the command line.
- Image input on Vast.ai rentals through Advanced deploy, on both runtimes. The llama.cpp projector is staged on the rental over the same pinned image Prime uses; vLLM serves images through its native multimodal path. Fast Deploy still refuses vision for every provider, because vision working memory is not calibrated for guaranteed-fit placement.
- `llm-launchpad vast-auth` (login/status/logout), `vast connect`, `offers --provider vast`, and `--provider vast` on deploy/list/status/logs/stop. `doctor` checks for a local Vast key the way it checks Modal and Prime, without authenticating it over the network.
- Advanced deploy can bind a selected Vast rental on either runtime. Disk size is quoted with the GPU.
- Docs for the provider and the broader release plan: `docs/vast.md`, `docs/vast-integration-plan.md`.
- Added an always-on rotating debug log under `~/.llm_launchpad/logs/` and routed previously silent failure paths (cache persistence, auth probing, log-tail cleanup, SSH key permissions) to it.
- Added `llm-launchpad doctor`, a self-check command that verifies the Modal CLI, Modal/Prime/Vast/Hugging Face authentication, that at least one compute provider is usable, the optional Artificial Analysis key, and state-directory writability, with fix hints per failure.
- Added `llm-launchpad --version`.
- Added documentation pages under `docs/` (deploy catalog, Prime provider, storage and costs, OpenCode, troubleshooting) and a generated CLI reference (`docs/cli.md`) via `scripts/generate_cli_reference.py`.
- Added GitHub issue templates for bug reports and feature requests.

### Changed
- A Vast rental now gets 1800s to answer SSH on either runtime, instead of 900s on llama.cpp and 1800s on vLLM. The old split was scaled on image size, but the pinned images are only 2.59 GB and 8.67 GB compressed — seconds to a minute of download — while the wait is dominated by unpacking them and by Vast's own provisioning, which installs the SSH server the upstream images do not carry. Two of three test hosts exceeded 900s, and the faster links were the ones that failed.
- A rental that reports no provisioning progress for 360s before SSH is reachable is now destroyed early rather than billed out to the deadline. Progress is read from Vast's `status_msg`, ignoring BuildKit's elapsed-time prefix, which keeps ticking while a step is wedged. The deploy log reports what the host is doing every 30s instead of leaving a multi-minute silence after one "loading" line.
- `doctor` requires one authenticated compute provider rather than all of them, which is what the TUI already gates on. Each provider reports for itself and only warns, and a new **Compute provider** check fails when none is configured. Previously a user deploying happily on Modal still got a failing run and a non-zero exit for not having a Prime key.
- Multi-GPU llama.cpp placements no longer predict faster single-request decoding. The estimate previously assumed roughly 1.6x more decode speed per extra GPU, but llama.cpp splits layers across devices by default, so one request walks them in sequence: extra GPUs buy memory, not speed. **This affects Modal and Prime as well as Vast** -- a multi-GPU placement that previously appeared in the Fastest or Balanced tier on that estimate may now rank lower. Measured attestations are unaffected and still take precedence.
- Deploy forms disable controls their provider ignores and say why, instead of greying them out silently. Vast.ai host/port are fixed by its SSH tunnel, image rebuilds are Modal-only, and llama.cpp cannot pin an HF revision on any marketplace provider.
- Deploy forms and `llm-launchpad deploy` now refuse a configuration its provider cannot run before anything is allocated, rather than failing after the deployment has been routed. Vast.ai previously accepted vLLM, vision, and pinned-revision configurations in the Advanced deploy form and rejected them later.
- Vision option fields keep visible labels, and projector/image details hide when vision is set to text only.

### Fixed
- Destroying a Vast rental now removes its SSH key along with its recovery record. Only `record.json` was deleted, so every rental left its private key, public key and `known_hosts` behind indefinitely; a local account that had rented 59 times was holding 47 private keys for hosts that no longer existed. The lock file is deliberately kept, because removal runs while that lock is held.
- A TUI crash now leaves a traceback in the debug log and exits non-zero. Textual renders an unhandled exception to a terminal it is about to restore and lets `run()` return normally, so the crash reached neither the log nor the exit status: a session that died mid-screen reported success and left an empty log behind. The log also records one line per session, so an empty file means nothing failed rather than that logging is broken.
- Vast rental startup requires a successful per-instance SSH key attachment and omits the reverted container startup hook, which caused SSH failures. Live rentals then certified vLLM serving on one GPU and on two-way tensor parallelism: the remaining `Connection refused` on a large image is Vast's sshd binding after the image pull, which the deploy loop already waits out.
- Vast live validation uses production-style SSH startup, checks total runtime and transfer estimates with cleanup headroom, and retains recovery records after uncertain image-probe creation. The Advanced deploy vision check supports vLLM without requiring a GGUF projector.
- Vast live certification covers llama.cpp text (with 1245 MiB verified resident on the device), llama.cpp image input, vLLM text, and two-way vLLM tensor parallelism against real rentals. Image input through vLLM is certified too, on an RTX A5000 through Advanced deploy. The vision stages verify the request path and a non-empty answer rather than visual accuracy, which `docs/vast-certification.md` states for both engines.
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
