# TUI feature audit — 2026-09-06

The audit exercised every screen family and the main deployment and management workflows with Textual Pilot, the production stylesheet, keyboard input, button activation, simulated provider responses, and failure injection. Reproducible interaction, validation, refresh, and layout defects were fixed and covered by regression tests.

This is a local functional audit. It does **not** certify live Modal/Prime provisioning, actual GPU capacity, billing accuracy, model inference, or the host terminal's clipboard integration. Provider operations and clipboard commands were mocked; test settings and caches were isolated from personal configuration.

## What was exercised

| Feature | Checks and outcome |
| --- | --- |
| Setup | Missing-provider screen, authentication recheck success/failure, keyboard and button Quit, narrow layout. Fixed unawaited shutdown and clipped buttons. |
| Home | Menu routing, keyboard shortcuts, auth/billing/deployment rendering and error states, deferred refresh, details panel, resume behavior, compact layout. Existing focused tests cover the panel data; the combined journey uses simulated data. |
| Help and navigation | Help open/dismiss, inherited bindings, arrows, Tab/focus, Escape/back, runtime resize and minimum-size gate. Priority Deploy shortcuts now respect the size gate. |
| Fast deploy | Model catalog, text/GPU filtering, live-availability responses, placement ordering/grouping, compare-all, empty/error/fallback states, stale responses after backing out, transition to confirmation. |
| Quick deploy | Profile and placement selection, serving objectives, advanced options, warmup, speculative decoding, Prime options and fallback plans; deploy through the actual app worker bridge to a simulated success. |
| Backend selection | Keyboard routing into both custom deployment forms and back. |
| llama.cpp deploy | Cached/downloaded/trending selection, quant metadata and stale responses, compatibility, GPU/provider options, naming, advanced values, pre-download, warmup and deployment routing. Fixed numeric validation and Deploy while editing. |
| vLLM deploy | Model selection, memory estimates/debounce, GPU/tensor settings, tool/reasoning parsers, JSON validation, alias/revision, smoke mode, Prime options and deployment routing. Fixed tensor validation and Deploy while editing. |
| Fleet and endpoint actions | Empty/populated fleet, refresh/resume, exact endpoint identity, duplicate names, lifecycle-dependent actions, status/logs/benchmark/stop/connection routes. |
| Connection information | Base URL, model and key rendering/copy, missing fields, result-card actions and return navigation. Fixed Enter on focused Copy buttons. |
| Status | URL override, timeout validation, Enter submission, provider routing, stored key forwarding and successful result completion. |
| Benchmark | Concurrency, request/token inputs, defaults, successful result card and return. Invalid values now leave the form and its edits available for correction. Benchmark execution uses simulated results, not a live AIPerf load test. |
| Monitor and logs | Success/failure, deployment/warmup sequence, connection/result cards, raw/summary views, ANSI handling, retention, reflow, follow/pause, search/match navigation, clear and copy. Fixed Done, search Enter, button Enter and populated-card overflow. |
| Storage | Both backend inventories, filter, row selection/prefill, refresh/resume, cost formatting, pre-download, delete confirmation/cancellation and worker completion. Fixed refresh subscribers, unexpected worker errors and hidden delete selection. |
| Settings | Persistence, invalid input, save failure, theme/density, mouse/quit preferences and unsaved-change behavior. Error text is now escaped. |
| Shared behavior | Clipboard fallbacks and terminal escape sequences, mouse toggle, quit confirmation, interrupted-deployment cleanup/journal, responsive tables and focus restoration. Terminal-specific behavior is tested with fakes. |

The viewport suite covers **140×45, 100×30, 80×24, 60×20, 50×40, 50×35, 40×12, 120×16, 200×15 and 220×60**, plus below-minimum resize/focus checks. Added cases cover setup, connection information with an API key, deletion confirmation, and a **populated** deployment result; empty-screen geometry alone missed the result-card defects.

## Fixed defects

| Problem before the audit | Result after the fix |
| --- | --- |
| Ctrl+D in a focused deployment input invoked Textual's delete-character action. | The advertised Deploy shortcut works in custom and quick-deploy fields, with the minimum-size gate preserved. |
| Enter in status/benchmark inputs did nothing. | Input submission reaches the form's validation and action. |
| Invalid benchmark concurrency closed the form; nonpositive request/token counts reached the worker. | Validation stays on the form and preserves entered values. |
| Setup Quit created an unawaited coroutine. | Both button and keyboard routes await app shutdown. |
| Done after a successful operation without a connection payload did nothing. | Completed status, benchmark and storage operations return to their originating flow. |
| Enter in log search could leave a completed monitor; Enter on Copy could trigger Done. | Search closes independently, and focused buttons receive Enter. |
| A storage refresh served only its first subscriber. | One refresh delivers its result to every waiting screen, with duplicate subscriptions suppressed. |
| Unexpected storage exceptions escaped the worker. | Waiting screens receive a failure message and can retry. |
| An empty storage filter retained a hidden model as the deletion target. | A model removed from the visible inventory is no longer selected for deletion. |
| Modal status/warmup omitted configured endpoint API keys. | Keys are forwarded to the orchestrator for authenticated probes. |
| Invalid llama.cpp port/GPU-layer text and vLLM tensor values were silently replaced or accepted. | Invalid values block submission with an explanation; valid values retain their existing behavior. |
| Markup-like text in backend errors, naming previews or status details could break rendering. | Dynamic text in the affected paths is escaped with Rich's shared helper. |
| Setup buttons and populated connection cards overflowed supported terminals. | Setup actions stack; connection actions adapt; result cards scroll while leaving room for logs. |

Regression coverage is in [test_tui_audit.py](../tests/test_tui_audit.py), [test_tui_feature_journey.py](../tests/test_tui_feature_journey.py) and the expanded [viewport suite](../tests/test_tui_responsive_layout.py). The two journeys retain real screen routing and threaded event delivery while replacing provider operations. They complete custom llama.cpp, custom vLLM, quick deploy, status, benchmark, logs, stop, pre-download and deletion flows.

## Prioritized improvements

These are remaining product/workflow improvements, ordered by impact. Sizes are relative: **S** is a focused change, **M** spans a few components, **L** needs a lifecycle or data-contract change.

| Priority | Improvement and evidence | Acceptance criterion | Size |
| --- | --- | --- | --- |
| **P1 — first** | **Track and reopen active operations.** Back currently pops a monitor while app-owned work can continue; the app tracks a single in-flight deployment. See `MonitorScreen.action_go_back` and `TuiApp._in_flight_deploy`. | Home exposes each active job; users can reopen logs and deliberately cancel or leave it running. Multiple deployments cannot overwrite each other's tracking. | L |
| **P1 — second** | **Distinguish provider failure from an empty fleet.** `_visible_rows_and_prune_scope` suppresses provider discovery errors and can return an empty or partial list as a normal refresh. | Retain the last successful rows for an unavailable provider, show its error and data age, and keep actions for healthy providers usable. | M |
| **P1 — third** | **Make provider readiness explicit without delaying first paint.** Startup's `_provider_is_configured` treats an installed Modal CLI as configured; the setup copy describes authentication. | Show installed/checking/authenticated/failed separately, check asynchronously, and offer a clear recovery action before a deploy attempt fails. Prime-only users get accurate provider-specific choices. | M |
| **P2** | **Keep model search available in normal terminals.** CSS hides `#fast-deploy-model-search` below 120 columns or in short terminals; `/` currently explains that it is unavailable. | `/` opens a temporary search field at 80×24 and 60×20, preserves the query through resize, and returns focus to the filtered list. | S |
| **P2** | **Make form actions and validation easier to discover.** Storage pre-download has no submit button; `p` types text while an input is focused. Several deployment errors use transient notifications. | Provide a visible pre-download button and an input-safe shortcut; identify and focus invalid fields with persistent inline feedback. Clarify scaledown boundary semantics and GPU-count versus tensor-parallel constraints. | M |
| **P2** | **Clarify storage provider scope.** The inventory and estimates describe Modal volumes while the application also supports Prime disks. | Label the current provider/scope explicitly, show what is unavailable for Prime, and separate provider-specific inventory and cost estimates when disk management is added. | M |
| **P2** | **Use a clear final operation state.** Deployment publication emits `PUBLISHING` just before completion; the header can continue showing that transient state after the connection card appears. Several newer states have no specific status icon. | Completion replaces transitional wording with an unambiguous outcome; status, deploy, benchmark and storage results use consistent final summaries and navigation. | S |
| **P3** | **Improve help and release coverage.** Help collects class bindings rather than the effective focused-widget bindings, and TUI modules remain excluded from `ty` checks. | Help shows effective user-facing shortcuts and conflicts; CI retains combined keyboard journeys and populated-state layout checks, and adds TUI typing incrementally. | M |

## Validation and review artifacts

| Check | Final result |
| --- | --- |
| Full suite: `uv run --no-sync pytest -n 4 --cov=llm_launchpad.tui` | **936 passed** in 61.53 seconds; two Modal SDK warnings about locally invoked functions lacking mounted volumes. |
| TUI statement coverage | **82%**: 5,274 of 6,406 statements exercised. This measures code execution, not completeness of live-provider verification. |
| `uv run --no-sync ruff check .` | Passed. |
| `uv run --no-sync ty check` | Passed for the repository's configured scope, which excludes TUI modules. |
| `uv build --no-sources --offline` | Source distribution and wheel built successfully. Build dependencies had been fetched during the earlier successful build. |
| `git diff --check` | Passed. |

The initial test attempts exposed two unrelated test-isolation gaps: vLLM screen metadata lookup and catalog context lookup could contact Hugging Face. Both are now stubbed at their test boundaries; focused lookup tests still supply their own responses.

The sandbox also stalled a minimal `asyncio.run(asyncio.to_thread(lambda: 1))` example. That same example passed outside the sandbox, so the complete suite was run outside it rather than changing application shutdown logic to accommodate the environment.

Eighteen SVG captures were generated during the combined journeys. They use synthetic endpoints and model data. Representative captures:

- [Setup at 60×20](assets/tui-audit/setup-60x20.svg)
- [Completed deployment](assets/tui-audit/deployed.svg)
- [Status result](assets/tui-audit/status-result.svg)

To regenerate the full screenshot set:

```bash
LLM_LAUNCHPAD_AUDIT_ARTIFACTS=/tmp/launchpad-tui-audit \
  uv run pytest -n 0 tests/test_tui_feature_journey.py
```

Before a release, the remaining external verification is a real deployment/warmup/status/inference/stop cycle for each supported provider/backend combination, real pre-download/delete against disposable storage, AIPerf execution, and clipboard/mouse checks in the supported terminal environments. Those checks were not performed in this audit.
