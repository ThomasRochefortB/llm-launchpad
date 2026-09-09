# Vast.ai rentals (beta)

Vast offers participate in Fast Deploy's prices, GPU filters, serving tiers,
and eligible deployment fallbacks. The deployable beta supports **text-only
llama.cpp GGUF models on one to eight GPUs** with a published, digest-pinned
runtime. Offers whose runtime is unsupported remain comparisons.

A rented host's GPUs are inventoried over SSH before anything is served: a
bundle that reports a different device count, mixed GPU models, or too little
free memory per device is refused and destroyed rather than served.

Endpoints use a managed SSH tunnel bound to `127.0.0.1`: they work on this
computer only. OpenSSH and a POSIX system are required. Closing Launchpad leaves
a completed rental and its tunnel running. **Stop destroys the rental and its
disk**, including cached models. Rentals bill continuously until destroyed.

The lifecycle and UI have hermetic test coverage. No real Vast host has been
live-certified yet; runtime startup, long streams, tool calls, and measured costs
remain release validation work. This is not general provider certification.

## Account setup

Use the hidden prompt to validate and save your key:

```bash
llm-launchpad vast-auth login
llm-launchpad vast-auth status
llm-launchpad vast-auth status --local
llm-launchpad vast-auth logout
```

For automation, `vast-auth login --key-stdin` reads the key from standard input.
The effective key is resolved in this order:

1. `VAST_API_KEY`.
2. `~/.llm_launchpad/vast_auth.json`, created with owner-only permissions.
3. `$XDG_CONFIG_HOME/vastai/vast_api_key`, defaulting to
   `~/.config/vastai/vast_api_key` when XDG_CONFIG_HOME is unset.

Login validates the provided key before saving it. Status verifies the effective
key; `--local` only reports its source. Logout removes Launchpad's file without
revoking the key or changing environment variables or the Vast CLI file. It
reports when a key remains configured elsewhere. `doctor` reports local Vast
configuration as optional and does not authenticate it over the network.
The Vast CLI is not required. Deployment requires `ssh` and `ssh-keygen`.

## Browse offers

```bash
llm-launchpad offers --provider vast
llm-launchpad offers --provider vast --gpu-type RTX_4090 --region US \
  --disk-gb 120 --min-reliability 0.99 --limit 50
llm-launchpad offers --provider vast --secure-only --json
```

In the TUI, open **Settings -> Vast.ai rentals (beta)**, or use the Vast button
on the setup-required screen. Enter and validate a key, or use an existing
configured key, then refresh offers. Browsing does not rent a GPU.

## Fast Deploy

After saving a Vast key, open **Deploy model**. Fast Deploy fetches Vast offers
alongside Modal and Prime. Press **r** to refresh offers and pick up a newly
configured key.

- A cheaper fitting Vast offer changes the model's **from** price and can win a
  serving tier. Vast GPU types also appear in the GPU filter.
- Supported placements are selectable in step 2 and appear in the
  confirmation's fulfillment choices. Confirmation explains local connectivity,
  continuous billing, disk deletion, and separate transfer charges.
- Multi-GPU and unsupported-runtime comparisons are labeled **Vast preview**.
  Selecting one shows cost and memory fit but cannot rent it. **a** shows all
  comparison rows.
- An equivalent Vast placement can enter the accepted fallback list within its
  price ceiling. Uncertain creation or unconfirmed destruction blocks further
  fallback rentals.
- Failed refreshes report partial results and discard stale Vast prices while
  leaving other providers available.

## Advanced deploy

**Advanced deploy** offers Vast.ai as a compute provider on both the llama.cpp
and vLLM forms. Selecting it swaps the Modal GPU picker for a **Vast.ai rental**
list, sorted by hourly total and filtered to rentals whose combined GPU memory
fits the model's estimate. The bound rental supplies the GPU shape and count, so
the GPU fields become read-only and tensor sharding follows the rented count.
At equal price the simpler topology is offered first.

- **Vast disk size (GB)** lives under **Advanced options** and defaults to 100.
  It is rented with the GPU and included in the quoted hourly price.
- The approved hourly ceiling is exactly the price shown on the selected row.
  If the rental re-quotes higher at deploy time the deployment is refused;
  reselect the provider to refresh rentals.
- Smoke-test-only mode is Modal-only, as it is for Prime.

Fast Deploy requests a bounded snapshot of up to 500 verified, on-demand NVIDIA
offers with one through eight GPUs and reliability of at least 0.99. It checks
each catalog quant's estimated full-context memory, including GPU reserve, on
the offered topology. Profiles without the required memory/context metadata are
excluded from Vast comparisons. This establishes estimated memory fit, not host,
runtime, interconnect, or streaming-endpoint certification.

Disk sizing uses the larger of 100 GiB and estimated model weights plus 10%
headroom and 10 GiB for runtime files. Offers lacking that disk capacity are
excluded. When a model needs more than the quoted disk allocation, the comparison
estimates the new hourly total from the quoted per-GB disk rate. If that rate is
unknown, the resized total stays unknown. Transfer charges remain separate; a
multi-GPU offer's whole-machine price is not multiplied by its GPU count.

The cheapest host per GPU topology and quant is displayed. This snapshot is
neither a reservation nor an exhaustive marketplace search. Vast prices do not change
the catalog's model quality ranking. Supported placements compete in serving
tiers; their exact offer, machine, disk allocation, memory, and hourly price are
checked again immediately before rental.

## Offer query details

The default query requests up to 100 verified, on-demand NVIDIA
offers on x86-64 hosts, with a reliability score of at least 0.99 and space for
100 GiB of disk. The TUI deploys supported offers; the CLI can inspect other
GPU counts with `--gpu-count`, without implying deployment support. `--region`
accepts a two-letter country code for Vast. `--secure-only` adds Vast's
datacenter filter; verified hosts alone are not labeled as secure cloud.

The API receives the selected disk allocation for pricing. Output separates
compute/hour, disk/hour, total/hour excluding traffic, and download/upload
charges per GB. Missing values are shown as unknown, including in JSON as null.
These are rental quotes, not total-session estimates or performance predictions.
Model downloads incur transfer charges and stopped instances retain storage
charges. The JSON also includes machine identity and maximum rental duration
when supplied. A search is a bounded snapshot, not a reservation or an exhaustive
inventory.

The default `offers` command continues to use Prime, including its default
secure-cloud filter. Vast-specific pricing filters are rejected for Prime;
Prime disk IDs and interruptible selection are rejected for Vast offers.

## Deploy and manage from the CLI

Select a current offer ID and set a maximum hourly compute-plus-disk charge.
The following IDs and price cap are illustrative; substitute your selection:

```bash
llm-launchpad deploy --provider vast --backend llamacpp \
  --repo-id Qwen/Qwen3-0.6B-GGUF --quant Q4_K_M --vision off \
  --gpu-type RTX_4090 --gpu-count 1 --vast-offer-id 123456 \
  --vast-disk-gb 100 --max-hourly-cost 0.50 --instance-name vast-test
llm-launchpad list --provider vast
llm-launchpad status --provider vast --instance-name vast-test
llm-launchpad logs --provider vast --instance-name vast-test --no-follow
llm-launchpad vast connect 987654
llm-launchpad stop --provider vast --instance-name vast-test
```

`vast connect` takes the **instance ID**, restores its original local port, and
verifies streaming without renting another GPU. If another application owns the
port, reconnect fails rather than publishing an unrelated endpoint. The SSH
master survives CLI/TUI exit but may disconnect after sleep or network loss.

Launchpad generates a per-rental SSH key and endpoint bearer token. Startup
transfers the private runtime script through SSH stdin; the Vast account key is
never sent to the host. An existing Hugging Face login is used for model download.
SSH uses a private known-hosts file, accepts the first host key, and rejects
changed keys. Remote model logs redact endpoint and Hugging Face tokens.

Deployment requires health and streaming-chat checks before it reports success.
Fast Deploy additionally runs the existing serving-plan certification before
publishing to OpenCode.

Image input is available on **Advanced deploy** only, and that limit is not a
Vast one: Fast Deploy refuses vision for every provider because vision working
memory is not calibrated for guaranteed-fit placement. The projector is staged
on the rental exactly as it is on Prime, over the same pinned image.

Model switching, custom-build runtimes, non-default revisions, suspend/resume,
and persistent Vast volumes are outside this beta. Unsupported configurations
are refused in the form, before anything is rented.

### Recovery and billing

Private recovery records live under `~/.llm_launchpad/vast/`. Launchpad writes a
unique ownership label before creation and saves the returned instance ID before
continuing. An ambiguous response is reconciled by label; creation is never
blindly retried. Account and machine identity are checked before destruction.
Failed deployment and cancellation attempt to destroy their own rental. If
cleanup cannot be confirmed, the record remains visible and automatic fallback
is disabled. Keep these files until the rental is destroyed.

Use Manage -> Stop, or `stop --provider vast --app-name <recorded-name>`, to retry
cleanup after an interrupted attempt. The confirmation explicitly describes disk
deletion; `--yes` skips it. Launchpad only removes the recovery record after the
provider reports the instance absent. If creation is still uncertain, inspect the
recorded ownership label in Vast's console before taking further action. API
outages can delay cleanup while billing continues.

The hourly cap excludes traffic and does not impose a maximum session duration.
Inspect the offer's download/upload rates before deployment. A budgeted live test
must account for image/model download charges as well as runtime, and verify
final destruction. No paid test has been run by this implementation.

## API and transport contracts

Discovery uses `/users/current/` and `/bundles/` on the v0 REST API. Creation uses
`PUT /asks/{id}/` with SSH launch mode; instance lookup, per-instance SSH key
attachment, and destruction also use v0. Recovery paginates the v1 instances
list and checks the exact ownership label. Requests have timeouts, reject
redirects, and sanitize provider errors.
[Account](https://docs.vast.ai/api-reference/accounts/show-user),
[offers](https://docs.vast.ai/api-reference/search/search-offers),
[creation](https://docs.vast.ai/api-reference/instances/create-instance),
[lookup](https://docs.vast.ai/api-reference/instances/show-instance),
[listing](https://docs.vast.ai/api-reference/instances/show-instances),
[SSH keys](https://docs.vast.ai/api-reference/instances/attach-ssh-key),
[destruction](https://docs.vast.ai/api-reference/instances/destroy-instance).

The offer browser follows Vast's GB display convention (raw memory / 1000).
Fast Deploy uses raw GPU MiB / 1024 for the GiB-based memory planner. Rental
duration is normalized to hours and disk allocation is included in search pricing.
[Official CLI](https://github.com/vast-ai/vast-python/blob/master/vast.py).

The beta uses SSH forwarding rather than Instance Portal quick tunnels, which
have SSE limitations. It does not expose a public plaintext server or require a
separate ingress account. A public HTTPS endpoint remains future work.
[SSH connections](https://docs.vast.ai/guides/instances/connect/ssh),
[Cloudflare tunnel limitations](https://developers.cloudflare.com/tunnel/setup/).

See the [integration plan](vast-integration-plan.md) for the broader release scope.
