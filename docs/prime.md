# Prime Intellect provider

Prime Intellect support uses the REST API and the credentials written by
`prime login` (`PRIME_API_KEY` takes precedence). Launchpad resolves the Prime
runtime behind the provider adapter. Its default path provisions Prime's
`ubuntu_22_cuda_12` image, creates and registers a dedicated Launchpad SSH key,
and starts the configured upstream vLLM or llama.cpp container over SSH. The
key is reused from `~/.llm_launchpad/prime/bootstrap_ed25519`; users only need
their normal Prime login.

## Networking and security

The runtime is bound only to the pod's loopback interface. Launchpad registers
a Prime Tunnel locally, sends only its ephemeral tunnel credentials to the pod,
and returns the generated HTTPS URL. The Prime account API key never needs to
be stored on rented compute. Standard OpenAI-compatible bearer authentication
remains enabled behind the tunnel. `--allow-insecure-http` explicitly bypasses
the tunnel and publishes the runtime directly; it is intended only as a
troubleshooting fallback. Stopping a pod also removes every Launchpad tunnel
associated with it.

Prime currently returns tunnel registrations with an expiry timestamp (seven
days in live validation). Launchpad shows that timestamp in deploy logs;
redeploy before it expires if the serving session needs to continue.

The generated endpoint bearer key is stored with mode `0600` in
`~/.llm_launchpad/deployment_connection_summaries.json` and is reused by
status, warmup, benchmark, and OpenCode sync. Prime Tunnel's public URL is
HTTPS, but the URL itself is not an access-control mechanism; vLLM or
llama.cpp validates the stored bearer key on every inference request.

## Models and quants

Prime llama.cpp uses the default Hugging Face revision and accepts a quant
label such as `Q4_K_M`.

Launchpad repeats the llama.cpp architecture preflight before allocating Prime
compute and blocks known-incompatible models with the architecture and pinned
runtime build in the error. Run `llm-launchpad llamacpp-support` to list every
supported GGUF architecture.

## Persistent cache disks

Prime deploys create or reuse a 100 GB persistent cache disk by default. The
disk stores Hugging Face and llama.cpp caches so later pods can reuse
downloaded model weights. Use `--no-prime-disk` to opt out, or provide
`--prime-disk-id` to attach an existing disk. Persistent disks remain billable
after a pod stops; remove an unused Launchpad cache disk from the Prime
dashboard. Launchpad also caches its pinned, checksum-verified Prime Tunnel
client locally and on the persistent disk, so pods do not repeatedly download
it from GitHub.

## Headless examples

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
llm-launchpad llamacpp-support --repo-id unsloth/Qwen3-4B-GGUF
llm-launchpad deploy \
  --provider prime \
  --backend llamacpp \
  --repo-id unsloth/Qwen3-4B-GGUF \
  --quant Q4_K_M \
  --gpu-type H100_80GB \
  --gpu-count 1 \
  --do-warmup
```

```bash
llm-launchpad list --provider prime
llm-launchpad logs --provider prime --backend llamacpp --instance-name qwen3
llm-launchpad stop --provider prime --backend llamacpp --instance-name qwen3 --yes
```

## Maintainer certification suite

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
