# Deploy catalog and recommendations

## Model-first deploy

Deploy is model-first: each model can have multiple runtime recipes and each
recipe can receive quotes from any compatible provider adapter. Quotes
normalize GPU shape, availability, hourly price, billing model, and a
scenario-based monthly estimate: every plan carries its cost assumptions
(schedule, active hours, idle tails), so ranking and display share one
calculation. A GPU filter on the Deploy screen narrows the
catalog to models that fit a selected GPU type. The live catalog supplies the
curated recipes and Modal estimates; both llama.cpp and vLLM recipes can also
use live Prime Intellect offers through the same plan-to-deployment path.
Prime CPU rows are excluded, and live GPU options are filtered per model using
its estimated VRAM requirement plus safety headroom.

## Artificial Analysis ranking

When the TUI opens, it builds the Deploy catalog in the background. With an
Artificial Analysis API key configured (`ARTIFICIAL_ANALYSIS_API_KEY` or
`llm-launchpad aai-auth login`), it selects the top three deployable
open-weight models in each of three size bands (Compact ≤40B, Medium
40–150B, and Large >150B), plus the top ten overall, ranked by the Artificial
Analysis Intelligence Index. Press **v** in the model picker to switch between
**By size** and **Top 10 by AA score · all sizes**. The overall view has no
per-size quota; search and GPU filters still apply. It shows up to ten scored
models from the accessible catalog, with fewer rows when fewer qualify.
Unranked trending models remain available in the size view.

Hugging Face verifies matching GGUF weights and memory requirements,
while Modal supplies current GPU availability and pricing. GGUF
recommendations are also filtered by their `general.architecture` against a
generated support manifest for the exact pinned llama.cpp image. Models with
missing or unsupported architecture metadata are not recommended.

### MTP speculative decoding

For Unsloth models whose architecture supports native MTP in the pinned runtime,
the catalog checks for a matching `-MTP-GGUF` repository before choosing GPUs.
It verifies embedded MTP heads in every offered quantization and uses the
variant's weight sizes plus speculative decoding memory overhead for placement.
This also applies when the ordinary repository match was cached previously.
If the variant cannot be verified within the request budget or does not fit,
the ordinary model remains available. Separate draft-head files are not selected
by this discovery path.

## Caching and fallbacks

Artificial Analysis responses are cached for 24 hours under
`~/.llm_launchpad/artificial_analysis_models.json`; the API key resolves from
`ARTIFICIAL_ANALYSIS_API_KEY` first, then from an owner-only file written by
`llm-launchpad aai-auth login` under `~/.llm_launchpad/`. A one-way key
fingerprint lets the TUI reuse a recent successful validation safely. The
authentication footer shows the AAI status and account tier alongside Modal,
Prime Intellect, and Hugging Face. Modal prices and Hugging Face GGUF metadata
are still refreshed on each launch. Free AAI keys are supported, with model
size inferred when the free response omits parameter counts. Without a key or
cached AAI response, startup falls back to Hugging Face trending GGUF models.
If that also fails the Deploy screen shows a clear unavailable state that
points to Custom deploy.

## llama.cpp runtime support

Launchpad verifies GGUF `general.architecture` compatibility before every
deploy, on Modal and Prime alike:

```bash
llm-launchpad llamacpp-support                       # list supported architectures
llm-launchpad llamacpp-support --repo-id unsloth/Qwen3-4B-GGUF   # preflight one repo
```

Setting a custom `LLAMA_CPP_IMAGE_REF` is still supported, but its
compatibility is reported as unverified because the bundled manifest only
describes the pinned default image.
