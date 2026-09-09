# Vast.ai integration plan

Status: implementation started, 2026-09-08. No Vast resources have been created
or live-certified.

The implemented increments cover auth and quotes; model-aware Fast Deploy
prices and filters; and a deployable single-GPU, text-only llama.cpp beta. Vast
placements now compete in serving tiers and accepted fallbacks. The lifecycle
includes repricing, durable rental intent, create reconciliation, private SSH
startup, health/stream verification, listing, logs, reconnect, and confirmed
destruction. See [current behavior](vast.md).

The beta's explicit scope decision is a persistent loopback SSH endpoint, usable
on this computer only. **Launchpad Stop destroys the rental and its disk.** It
does not implement Vast suspend/resume. This supersedes the broader lifecycle
and public-ingress proposals below for the current increment. These differences
are displayed at deployment and stop confirmation.

Pending live certification: pinned image startup on a real host, authenticated
streaming and tool calls, long requests, startup/cost measurements, and final
cleanup. The broader vLLM, multi-GPU, vision, model-switching, persistent-storage,
and public HTTPS work remains planned. Mocked tests are not live certification.

The work packages below retain the broader release plan; their completion
criteria are not a description of currently implemented support.

## Decision and motivation

Prioritize Vast.ai as the next cloud integration. Runpod
[issue #6](https://github.com/ThomasRochefortB/llm-launchpad/issues/6) is closed as
not planned: direct Runpod account support is deferred while Launchpad evaluates
a different marketplace. Prime already exposes some Runpod supply; that is not
a claim that every Runpod offer works with Launchpad's current Prime runtime.

The intended benefit is a broader choice of suitable rentals for personal
inference. Success means finding an appropriate machine, understanding its
cost, and operating an endpoint from Launchpad. Lower advertised GPU prices
alone do not establish that Vast is better than Prime or Modal.

## First release scope

- Add `--provider vast` and a connected Vast account to the existing model-first
  TUI placement flow. Cover curated and custom deployments.
- Support llama.cpp GGUF and vLLM on verified, on-demand Linux/x86-64 NVIDIA
  instances. Certify single-GPU deployments first; expose multi-GPU placements
  only after their runtime and topology have been validated.
- Include auth/doctor, live offers, deploy, warmup, list, status, logs, model
  switch, stop/resume, explicit destroy, and OpenCode synchronization.
- Include instance-local model caching, disk sizing, and visible residual
  storage costs. Cross-instance volumes are a later increment.
- Defer interruptible bidding, serverless endpoints, autoscaling, clusters,
  unverified hosts, AMD, arbitrary SSH hosts, and direct Runpod integration.

Vast distinguishes verified machines from its datacenter tier. Present that
distinction accurately; verification must not be labeled as equivalent to
Prime's secure-cloud policy. Offer a datacenter-only filter.
[Source: instance selection](https://docs.vast.ai/guides/instances/choosing/find-and-rent)

## Existing architecture and required changes

The quote boundary exists in `core/inference_options.py` as
`InferenceProviderAdapter`. Recipes, quotes, placements, deployment config, and
endpoint records already live in `protocol/models.py`. Reuse these structures.

The lifecycle boundary is less complete: `core/orchestrator.py`,
`core/compute_availability.py`, and several CLI/TUI paths explicitly choose
between Modal and Prime. Some non-Modal branches implicitly mean Prime.

| Area | Planned change |
| --- | --- |
| `protocol/enums.py`, `protocol/models.py` | Add `ComputeProvider.VAST`, `VastProviderOptions`, normalized marketplace offer/cost data, and operation capabilities with backward-compatible defaults. |
| New `core/providers.py` | Centralize provider lookup and connected-provider enumeration. Expose the lifecycle operations required by callers; preserve existing Modal and Prime implementations behind wrappers. |
| New `core/vast_auth.py` | Resolve credentials, validate the account, and report useful auth errors. |
| New `core/vast_backend.py` | Own HTTP requests, offer parsing, instance lifecycle, mapped-port discovery, and log retrieval. |
| `core/inference_options.py`, `core/compute_availability.py` | Add Vast quotes and placements through the same normalization and selection path. |
| New `core/serving_runtime.py` and Vast runtime metadata | Extract reusable image/argument/environment construction from Prime only where both providers need it. Keep SSH bootstrap and Prime Tunnel logic provider-specific. |
| `core/orchestrator.py`, `core/warmup.py` | Consume provider operations, retain shared preflight/certification, emit existing protocol events, and support provider log polling during warmup. |
| `core/naming.py`, `core/connection_store.py`, `core/deploy_journal.py` | Persist provider identity, remote instance IDs, ownership labels, connection details, and interrupted deployments. |
| CLI, TUI setup/deploy/manage/monitor/storage, `core/doctor.py`, `core/opencode.py` | Wire account setup, placement details, management capabilities, recovery, and connection refresh. |

Keep transport details in core and cross-layer types in protocol. Provider
dispatch must reject unsupported providers rather than falling through to
another provider. Avoid creating a general plugin framework for this change.

## Work package 1: validate the provider contract

Timebox: 2-3 engineering days before broad CLI/TUI implementation. First use
documentation and fixtures; any paid experiment belongs in a separately invoked,
budgeted live-validation script.

1. Verify request/response contracts for account validation, offer search,
   creation, lookup/list, start/stop, destruction, SSH keys, and logs. Prefer
   REST through the existing `requests` dependency; do not require the Vast CLI.
2. Confirm offer field units, per-instance versus per-GPU rates, allocated disk
   pricing, transfer charges, rental expiration, and availability race behavior.
   Capture redacted fixtures; documentation examples are not measured offers.
3. Prove that pinned runtime images start correctly in Vast's container launch
   modes. Do not assume Prime's Docker-over-SSH bootstrap works unchanged.
   Build/publish any required images before GPU rental, with immutable references.
4. Resolve secure endpoint publication and test OpenAI streaming, bearer auth,
   tool calls, slow first tokens, and requests lasting more than five minutes.
5. Verify cache persistence, stop/resume behavior, final destruction, and which
   charges remain at each lifecycle state.

The REST API exposes [offer search](https://docs.vast.ai/api-reference/search/search-offers)
and [instance creation](https://docs.vast.ai/api-reference/instances/create-instance).
The latter supports startup commands and image arguments. Treat creation as a
billable mutation even though its HTTP method is PUT.

### Endpoint decision gate

Vast maps internal ports to external ports on generally shared IP addresses.
Port mapping alone does not establish a publicly trusted HTTPS endpoint.
[Source: networking](https://docs.vast.ai/guides/instances/connect/networking)

The target for full release is an authenticated HTTPS URL that continues working
after the CLI/TUI exits, matching the existing deployment experience. Verify a
supported ingress or tunnel arrangement, certificate renewal, URL lifetime,
streaming limits, and whether another account/service is required before
selecting it. Do not invent a Runpod-style proxy URL or reuse Prime Tunnel
credentials for Vast.

If that target cannot be met within the timebox, document an explicit scope
choice before work package 3: either defer publication or ship a beta using a
managed SSH forward. The SSH option must bind locally, mark the endpoint as
available only on this computer, preserve/recover the tunnel independently of
the TUI, and distinguish disconnecting the tunnel from stopping paid compute.
Public plaintext HTTP or disabled certificate verification is not the default
fallback. Networking is the principal unresolved release dependency.

## Work package 2: provider registry, authentication, and quotes

Land provider dispatch changes with Modal/Prime regression coverage first.
Then implement Vast account resolution with this precedence: `VAST_API_KEY`,
Launchpad's owner-only credential file, and the existing Vast CLI key file if
present (confirm its supported path in work package 1). Add a masked TUI input
and `vast-auth login/status/logout` following existing auth command conventions.
Keep local key detection distinct from a successful API validation. Never copy
the account API key into a rented instance or OpenCode configuration.

Add typed options for offer ID, location, disk size, reliability floor,
datacenter-only selection, and keeping a failed resource for diagnosis.
Default selection to verified, rentable, non-rented, on-demand offers with a
proposed reliability floor of 0.99. Keep the floor configurable; it is a product
policy to validate, not an uptime guarantee from Vast.

Filter against runtime CUDA/driver compatibility, per-device VRAM, GPU count,
CPU RAM, model download disk requirements, required ports, and usable rental
duration. Preserve exact offer/machine IDs; GPU-family grouping must not erase
topology or per-device memory constraints. Do not substitute total machine
VRAM for memory available to one GPU.

Search and fulfillment must use the same constraints. Refresh a selected offer
immediately before renting. If it disappears, return fresh choices or follow an
already accepted fallback within the user's price and placement constraints.

### Cost presentation

Add a typed cost breakdown rather than overloading `price_per_hour_usd` with an
inconsistent mix of GPU, disk, and transfer charges. Preserve existing compute
estimates and display the following separately:

- Compute per hour and disk per hour for the selected allocation.
- Estimated initial image/model download charge, with size assumptions.
- Transfer rates and unknown traffic costs; unknown must not mean zero.
- Remaining storage cost after stopping and the rental expiration when supplied.

Estimate a session as `hours * (compute/hour + disk/hour) + transfer costs`.
Do not apply Modal's utilization discount to a provisioned Vast instance. Rank
comparable estimates using the existing workload profile, show their assumptions,
and avoid declaring a cheapest total when material costs are unknown.
Vast bills transfer separately, including model downloads.
[Source: cost differences](https://docs.vast.ai/examples/migrations/runpod-to-vast)

## Work package 3: serving and reliable lifecycle

Use pinned images and shared model/runtime preflight for both engines. Preserve
model revision, served model name, context limits, reasoning/tool parsers, and
GGUF quant/projector selection. Enable vision or speculative decoding only where
the runtime compatibility checks and certification support the selected recipe.

Deployment sequence: validate config/auth/model -> resolve and refresh offer ->
record intent -> create instance -> persist remote ID -> wait for transport ->
wait for model -> certify inference/auth -> publish connection -> finish journal.
Emit progress and failures using the existing protocol events and `fail_operation`.

Use a unique ownership label and deployment attempt identifier. If creation
times out, reconcile by that identifier before attempting another rental. Never
blindly retry a mutation that may have succeeded. Retry read operations and
rate-limit responses with bounded backoff; respect cancellation.

Cleanup must touch only resources owned by the current attempt. Failed cleanup
keeps the journal entry and remote ID visible for recovery. Publish endpoints
to OpenCode only after readiness/certification succeeds; a transient provider
listing failure must not prune cached connections.

Vast log retrieval returns a generated download URL. Poll with bounded retries,
deduplicate repeated log content, and fetch that URL without forwarding the
Vast Authorization header. Redact runtime credentials and expiring log URLs.
[Source: logs API](https://docs.vast.ai/api-reference/instances/show-logs)

### Stop, resume, destroy, and storage

For Vast, `stop` stops compute and retains instance storage; `resume` restarts
and rechecks networking/readiness. Add an explicit `destroy` operation for
permanent instance removal, with provider capabilities controlling availability.
In the TUI use separate Stop and Destroy actions with concrete storage effects;
keep the current Modal/Prime behaviors intact. A stop may affect the ability to
reacquire GPUs, so resume must handle unavailable capacity clearly.

Default cache scope is the instance. Reuse weights for model switches and
stop/resume where confirmed; destroying the instance removes its cache. Size
disk before creation to include weights, image/runtime overhead, and temporary
download files. Do not advertise a portable cache across arbitrary offers.
Vast's documented separate volumes are tied to a physical machine; their
attachment and retained billing need a later design.
[Source: volumes](https://docs.vast.ai/guides/instances/storage/volumes)

## Work package 4: CLI and TUI integration

Keep model -> runtime recipe -> placement as the primary flow. Connected Vast
offers join the existing placement list. Put reliability, location, transfer
rates, disk size, and rental duration in placement details and confirmation;
show unsupported recipes as unavailable with a reason.

Add `--provider vast` to management and deployment commands, provider-aware
offers/GPU discovery, and advanced Vast options for explicit offer selection,
disk size, reliability, and datacenter filtering. Illustrative target commands
(not implemented yet):

```bash
llm-launchpad vast-auth login
llm-launchpad offers --provider vast
llm-launchpad deploy --provider vast --backend vllm \
  --model-name Qwen/Qwen3-4B --instance-name qwen3 \
  --gpu-type RTX_4090 --gpu-count 1 --do-warmup
llm-launchpad stop --provider vast --backend vllm --instance-name qwen3 --yes
llm-launchpad resume --provider vast --backend vllm --instance-name qwen3
llm-launchpad destroy --provider vast --backend vllm --instance-name qwen3 --yes
```

Account setup must work without Modal/Prime credentials. Manage/Monitor must
distinguish provider instance state, model readiness, and connection health.
Refresh OpenCode after a model switch, endpoint change, resume, or destruction.
Storage views must show supported Vast cache information or an explicit
capability limitation, never fall through to Modal storage operations.

## Work package 5: tests, certification, and documentation

Hermetic tests cover credential precedence/errors; malformed/partial offers and
units; VRAM/topology/runtime filtering; disk/transfer estimates; stale offers;
auth, quota and rate-limit failures; ambiguous create responses; cancellation;
cleanup failures and restart recovery; stop/resume/destroy effects; log retrieval;
and provider routing in CLI/TUI, connection storage, and OpenCode.

Extend existing provider-selection tests to three providers and prove that
missing Vast credentials do not affect Modal/Prime use. Add focused Vast test
files and fake HTTP responses rather than making paid calls from pytest.

Create `scripts/validate_vast_live.py` following the Prime certification pattern:
explicit live flag, spend budget, wall-clock cutoff, owned-resource cleanup in
failure paths, and a redacted report. Certify a small vLLM model and a small GGUF
model through the actual TUI, including auth, streaming, tool calls, cold/warm
cache, long requests, stop/resume, model switching, and final destruction.
Verify provider state after cleanup and report any remaining storage charges.

Before declaring general support, repeat representative deployment on a second
verified host and compare measured session cost and startup time against an
equivalent Prime/Modal recipe. No claimed price or throughput advantage without
measurements. Multi-GPU exposure requires a separate live test.

Update README, `docs/catalog.md`, CLI reference, auth/doctor help, storage/cost
docs, troubleshooting, the feature-request provider options, and a new
`docs/vast.md`. Include TUI screenshots with the implementation PR. Run all CI
checks before each PR: `uv run ruff check .`, `uv run ty check`, `uv run pytest`,
and `uv build --no-sources`.

## Delivery order and completion criteria

Land reviewable changes in order: (1) contract findings and networking decision,
(2) provider dispatch refactor, (3) auth/quotes/costs, (4) runtime/lifecycle and
headless commands, (5) TUI integration, (6) certification and release docs.
Do not expose Vast as generally supported while its endpoint or cleanup gate
remains unresolved.

Planning estimate: 3-5 engineer-weeks for this scoped release, including the
initial investigation and certification. This is not a commitment; custom image
distribution or a new ingress service could extend it. Re-estimate after work
package 1 rather than expanding the scope silently.

Done means both engines deploy from CLI and TUI, cost assumptions are visible,
connections survive the documented lifecycle, OpenCode works, failures leave
recoverable ownership records, destruction is verified, and Modal/Prime checks
continue to pass. A working creation API call alone does not satisfy the plan.
