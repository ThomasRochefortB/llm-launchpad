# Fast deploy: hybrid model support

Kimi K3 and GLM-5.3-Flash were already in the Artificial Analysis feed and
matched real Unsloth GGUF repositories. They disappeared during eligibility
checks, before the three-model shortlist for each size category was chosen.

## Changes

- Preserve GGUF per-layer KV-head arrays, including zero entries. Zero means a
  recurrent layer for these models; it must not become the query-head count.
- Plan Kimi's KDA state and compressed MLA cache separately. GLM Flash also
  needs its pooled indexer cache. Exclude NextN layers from normal decoding.
- Resize recurrent state when concurrency changes, without multiplying the
  entire shared token cache. Keep the breakdown in the catalog cache.
- Select a pinned, model-specific source runtime for GLM Flash on both Modal
  and Prime. Other models retain the existing b10689 runtime. Explicit custom
  container overrides remain overrides.
- Apply flash attention off and `NVIDIA_TF32_OVERRIDE=0` for the GLM runtime.
  Native MTP remains unavailable on this runtime until separately validated.
- Record exclusions in the catalog and expose them with **x — Excluded models**.
  Unsupported architectures, missing hybrid layout data, insufficient catalog
  GPU capacity, and models outside the category quota now have explanations.
- Invalidate old catalog and planner certificates so the old memory estimates
  are not reused. Preserve the Small / Medium / Large presentation.

Models with no matched GGUF repository are not added to the exclusions list:
the benchmark feed also includes API-only models. This view covers candidates
whose repositories were resolved and checked, not every benchmark entry.
Discovery now stops checking lower-ranked candidates in a filled category;
uninspected candidates are not labelled as incompatible.

## Memory accounting

Sources:

- [Kimi implementation at the existing runtime revision](https://github.com/ggml-org/llama.cpp/blob/57291f2644af8c9df0dd8d44395881c5bdcf0ecd/src/models/kimi-k3.cpp)
- [GLM Flash implementation at the selected revision](https://github.com/unslothai/llama.cpp/blob/629b50552801912b3e2078f9799e4d77213197d7/src/models/glm5next.cpp)
- [Runtime cache allocation](https://github.com/unslothai/llama.cpp/blob/629b50552801912b3e2078f9799e4d77213197d7/src/llama-memory-hybrid.cpp)
- [Recurrent-state dimensions](https://github.com/unslothai/llama.cpp/blob/629b50552801912b3e2078f9799e4d77213197d7/src/llama-hparams.cpp)

For each attention layer, the token cache uses its GGUF KV-head count and
compressed key width. The runtime stores the absorbed MLA latent once; it does
not allocate a separate expanded value cache. KDA layers instead allocate fp32
recurrent state plus convolution history per serving slot, including rollback
copies when requested.

GLM's indexer adds three vectors per token per attention layer: the key, the
compressor gate, and the pooled key. Indexer precision never falls below f16.
The existing compute-buffer estimate and per-GPU reserve remain additional
costs. These are planning estimates, not measurements or certifications.

The inspected Kimi header has 69 KDA and 24 MLA layers. GLM Flash has 34 KDA
and 11 attention layers plus one NextN layer. The public metadata fixtures in
`tests/fixtures/hybrid_gguf_metadata.json` reproduce the original failures.

## Runtime and validation

GLM support comes from [llama.cpp PR #27754](https://github.com/ggml-org/llama.cpp/pull/27754),
which was unmerged when inspected. The bundled Dockerfile downloads a specific
source commit and verifies the archive checksum. Modal builds that recipe;
Prime uploads the same recipe and builds it locally on the pod. Known GPU
families compile only their CUDA target instead of every supported target. The first
deployment therefore takes longer than pulling an existing server image.

The source server and `llama-fit-params` build successfully with the local CPU
toolchain, and their CLI help confirms the required serving flags. Tests cover
metadata parsing, cache sizing, catalog selection, placement, objective changes,
both provider paths, overrides, cache round trips, and the exclusion screen.
A rebuild using the cached benchmark feed and live Hub metadata/GPU prices
includes both models in the Large shortlist.

No full-model GPU run was performed. The local GPU driver is unavailable; the
installed CUDA compiler also rejects the host GCC version. The CUDA container
build and generation quality therefore remain unverified here. Existing runtime
fit and warmup attestation remain required before treating an endpoint as certified.

## Next design step

Keep the shortlist small, but retain discovery results independently of the
recommendations. This change starts that separation through persisted exclusion
records. A later model browser can expose all compatible candidates and apply
budget, context, and workload constraints before selecting the three leaders
per category. Architecture support and GPU availability should remain separate
facts, with their evidence and freshness visible to the user.

## Catalog build speed

The resolver keeps at most 24 candidates in flight, consumes their results in
benchmark order, and stops scheduling known-size candidates once that category
has its three eligible models. Unknown-size candidates still receive metadata
inspection while any category needs models. This preserves rank order even
when lower-ranked requests finish first. Unused lookahead requests cannot hold
up the picker or change its published exclusion records.

GGUF header reads start at 1 MiB and double on incomplete metadata, staying
within the existing 32 MiB limit. The complete metadata table is still parsed,
including serving fields after tokenizer data. This reduces both HTTP round
trips and repeated parsing of large tokenizers. Canonical repository probes
also have an explicit ten-second request timeout.

In a September 7 local comparison using the same cached benchmark feed and
live Hub metadata/Modal prices, catalog build time fell from **65.8 seconds to
13.9 seconds**, with a repeat run at **16.3 seconds**. Both faster builds
returned all nine category recommendations,
including Kimi K3 and GLM-5.3-Flash. Network and upstream-cache conditions vary;
this is a measured sample, not a latency guarantee. Existing disk snapshots
continue to make repeat opens immediate while the live refresh runs in the
background.
