<h1 align="center">LLM-Launchpad</h1>
<p align="center">Spin up personal LLM endpoints on Modal or Prime Intellect</p>
<p align="center">
  <img src="docs/assets/llm_launchpad_header.png" alt="LLM Launchpad header" width="900" />
</p>

- Deploy any open-source models from the Hugging Face model hub.
- OpenAI-compatible endpoints via llama.cpp (preferred) and vLLM backends.
- Direct integration with [OpenCode](https://github.com/anomalyco/opencode).

## Prerequisites
- **uv** for Python, environment, and CLI tool management (install with `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- A **Modal account**, a **Prime Intellect account**, or both
- **Hugging Face account**
- Optional: [OpenCode](https://github.com/anomalyco/opencode) (install with `curl -fsSL https://opencode.ai/install | bash`)

## Quickstart

Get up and running in four steps:

1. Install the CLI so `llm-launchpad` is available in your shell:
   ```bash
   uv tool install llm-launchpad
   llm-launchpad --help
   ```

2. Authenticate at least one compute provider:
   ```bash
   modal setup
   # or
   prime login
   ```

3. Authenticate Hugging Face:
   ```bash
   huggingface-cli login
   ```

4. Launch the TUI:
   ```bash
   llm-launchpad
   ```

## Why a TUI?

Setting up LLM endpoints usually means juggling model names, container images, GPU choices, warmup checks, logs, and endpoint details across several commands. The TUI keeps that flow in one place.

From the TUI you can:
- Browse popular models, compare compatible inference options, and deploy without memorizing provider-specific commands
- Manage multiple deployed instances and inspect their status
- Integrate the final OpenAI-compatible base URL and model ID into your workflows like OpenCode after deployment.

The Popular Models panel is model-first: each model can have multiple runtime
recipes and each recipe can receive quotes from any compatible provider adapter.
Quotes normalize GPU shape, availability, hourly price, billing model, and a
workload-based monthly estimate. Existing Quick Deploy bundles supply the
curated recipes and Modal estimates; both llama.cpp and vLLM recipes can also
use live Prime Intellect offers through the same plan-to-deployment path. Prime
CPU rows are excluded, and live GPU options are filtered per model using its
estimated VRAM requirement plus safety headroom.

## Headless CLI examples

The TUI is the recommended path for interactive use, but the same workflows are available from the command line for scripts and repeatable operations.

Deploy a vLLM endpoint and wait until it is ready:
```bash
llm-launchpad deploy \
  --backend vllm \
  --model-name Qwen/Qwen3-4B \
  --instance-name qwen3 \
  --do-warmup
```

Deploy vLLM on Prime Intellect using the cheapest matching fixed-price,
secure on-demand offer:
```bash
llm-launchpad offers --gpu-type H100_80GB --gpu-count 1
llm-launchpad deploy \
  --provider prime \
  --backend vllm \
  --model-name Qwen/Qwen3-4B \
  --gpu-type H100_80GB \
  --gpu-count 1 \
  --do-warmup
```

Deploy a llama.cpp GGUF endpoint on Prime with the same offer selection:
```bash
llm-launchpad deploy \
  --provider prime \
  --backend llamacpp \
  --repo-id unsloth/Qwen3-4B-GGUF \
  --quant Q4_K_M \
  --gpu-type H100_80GB \
  --gpu-count 1 \
  --do-warmup
```

Prime support uses the REST API and the credentials written by `prime login`
(`PRIME_API_KEY` takes precedence). Launchpad resolves the Prime runtime behind
the provider adapter. Its default path provisions Prime's `ubuntu_22_cuda_12`
image, creates and registers a dedicated Launchpad SSH key, and starts the
configured upstream vLLM or llama.cpp container over SSH. The key is reused from
`~/.llm_launchpad/prime/bootstrap_ed25519`; users only need their normal Prime
login.

The runtime is bound only to the pod's loopback interface. Launchpad registers a
Prime Tunnel locally, sends only its ephemeral tunnel credentials to the pod,
and returns the generated HTTPS URL. The Prime account API key never needs to be
stored on rented compute. Standard OpenAI-compatible bearer authentication remains
enabled behind the tunnel. `--allow-insecure-http` explicitly bypasses the tunnel
and publishes the runtime directly; it is intended only as a troubleshooting
fallback. Stopping a pod also removes every Launchpad tunnel associated with it.
Prime currently returns tunnel registrations with an expiry timestamp (seven days
in live validation). Launchpad shows that timestamp in deploy logs; redeploy before
it expires if the serving session needs to continue.

Prime llama.cpp uses the default Hugging Face revision and accepts a quant label
such as `Q4_K_M`.

The generated endpoint bearer key is stored with mode `0600` in
`~/.llm_launchpad/deployment_connection_summaries.json` and is reused by status,
warmup, benchmark, and OpenCode sync. Prime Tunnel's public URL is HTTPS, but the
URL itself is not an access-control mechanism; vLLM or llama.cpp validates the
stored bearer key on every inference request.

Maintainers can run the Prime certification suite against the same portable
Ubuntu path used by every deployment. It drives the real Textual deploy form,
verifies HTTPS and bearer authentication, enforces a spend cutoff, removes test
pods and tunnels, and writes a redacted JSON report:

```bash
uv run python scripts/validate_prime_live.py \
  --confirm-live \
  --budget-usd 3 \
  --stage portable_vllm_and_auth
```

Switch a llama.cpp instance to a Hugging Face GGUF model, redeploy it, and warm it up:
```bash
llm-launchpad switch \
  --backend llamacpp \
  --repo-id unsloth/Qwen3-4B-GGUF \
  --quant '*Q4_K_M.gguf' \
  --instance-name qwen3
```

Inspect and manage deployed apps:
```bash
llm-launchpad list
llm-launchpad status --backend llamacpp --instance-name qwen3
llm-launchpad logs --backend llamacpp --instance-name qwen3
llm-launchpad stop --backend llamacpp --instance-name qwen3 --yes
llm-launchpad list --provider prime
llm-launchpad logs --provider prime --backend llamacpp --instance-name qwen3
llm-launchpad stop --provider prime --backend llamacpp --instance-name qwen3 --yes
```

Sync existing Launchpad deployments into OpenCode without changing files first:
```bash
llm-launchpad opencode sync --dry-run
```

## Storage and cleanup

Downloaded model weights are cached in the Modal `huggingface-cache` volume so repeated deploys can start faster. Use the TUI Storage screen to refresh the cache inventory, predownload a model, or delete selected cached weights when they are no longer needed.

Stopping an app and deleting cached weights are separate operations: `llm-launchpad stop` stops a deployed Modal app, while the Storage screen manages cached model files. If storage size looks stale after a deployment or delete, refresh the Storage screen to reload the Modal volume snapshot.

## Costs and scaledown

GPU costs depend on the selected provider's billing model. Modal inference can
scale to zero, while a Prime pod is billed for the full provisioned serving
window. LLM-Launchpad shows both the provider's hourly price and a normalized
monthly estimate so those options can be compared without pretending their idle
costs are equivalent. Modal deployments default the scaledown window to 1800
seconds; change it in Settings or with `SCALEDOWN_WINDOW` before deploying.

For predictable costs:
- Stop apps you no longer need with `llm-launchpad stop`.
- Prefer smaller GPU layouts for quick tests before moving to larger models.
- Use the warmup command only when you actually need the endpoint ready immediately.
- Treat displayed cost estimates as guidance and confirm current provider pricing for production workloads.

## OpenCode integration

LLM-Launchpad automatically detects local installation of OpenCode and will set up your OpenCode config with the final OpenAI-compatible base URL and model ID after deployment.

## Troubleshooting

- `Modal CLI not found`: reinstall or upgrade the package, then confirm `modal --help` works in the same shell.
- `Modal authentication missing`: run `modal setup`.
- Hugging Face download errors: run `huggingface-cli login` and verify the model license or gated-repo access in your Hugging Face account.
- Warmup stays queued: Modal may still be scheduling the requested GPU. Try a smaller GPU configuration or wait for capacity.
- Endpoint status fails after deploy: inspect `llm-launchpad logs --backend <backend> --instance-name <name>` for backend startup errors.
- SSH copy or selection feels wrong in the TUI: start with `llm-launchpad tui --no-mouse` to let the terminal handle native text selection.

## Development setup

If you are working from a clone and want the command available directly while editing the source:
```bash
git clone https://github.com/ThomasRochefortB/llm-launchpad.git
cd llm-launchpad
uv tool install --editable .
llm-launchpad --help
```

If you need the full project environment for tests or local development workflows:
```bash
uv sync
uv run pytest
```
