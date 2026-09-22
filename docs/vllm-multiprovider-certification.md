# Advanced vLLM certification across every provider

This records what was measured, not what was implemented, for one model served
through the **Advanced deploy** vLLM form on Modal, Prime Intellect and
Vast.ai on 2026-09-19. The model is `Qwen/Qwen3.8-27B`: 55.6 GB of bf16
safetensors, a hybrid attention stack (full attention every fourth of 64
layers, linear attention between), 4 KV heads at head dimension 256, a vision
tower, and a 262,144-token native context. Every run was driven through the
real Textual form with Textual's Pilot, allocated real compute, and destroyed
it afterwards.

Each stage served a **32,768-token cap** at **256 concurrent sequences** on a
single 80–98 GB GPU. Tensor parallelism is 1 throughout: the model shards
cleanly only at 1, 2, 4 or 8 (24 query heads, 16 linear key heads), and one
GPU holds it. The form's own estimate for that configuration is **69.3 GB**;
the runtimes served it in 80 GB.

| Provider | Placement | Recorded evidence |
| --- | --- | --- |
| Modal | 1× A100-80GB, $2.50/hr | 323.3s deploy and warmup with the weights already in the shared volume; 401 to a missing bearer token and 401 to a wrong one; 26 completion tokens in 1.24s; 203 stream chunks in 7.46s; OpenCode returned its sentinel with exit 0; app stopped and confirmed absent |
| Prime Intellect | 1× A100 80GB in `us-central-3`, $1.23/hr | 1844.4s deploy and warmup, of which 20 minutes was the 55.6 GB download with no cache disk available; 401 and 401; 26 completion tokens in 1.08s; 199 stream chunks in 7.15s; OpenCode sentinel, exit 0; pod terminated and confirmed absent |
| Vast.ai | 1× RTX PRO 6000 Server Edition (95.6 GiB free) in Switzerland, $1.678/hr incl. disk | 391.2s deploy and warmup, including the 55.6 GB download at up to 713 MB/s; 401 and 401; 29 completion tokens in 1.23s of which **25 were reported as `reasoning_tokens`**; 256 stream chunks in 9.75s; OpenCode sentinel, exit 0; rental and disk destroyed, nothing left billable |

The three stages cost **$3.39** in total across all attempts, read from the
accounts before and after: Modal $1.63, Vast $1.18, Prime $0.58.

Prime's row was measured before `--max-num-seqs` became explicit. The flag
did not change what Prime serves — its pinned vLLM already defaulted to the
256 now passed — and the command it builds comes from the same
`vllm_serve_args` that the later Vast run exercised with the flag in place.

## What "validated" means here

Every row cleared the same five checks, in order:

1. The Advanced form built the deployment — model, provider, GPU, context
   cap, concurrency and both parsers — with Deploy pressed in the real screen.
2. Deploy and warmup completed and the endpoint published a connection.
3. A request with no bearer token and a request with the wrong one were both
   **refused**, and a request with the right one returned completion tokens.
4. A streaming request delivered chunks.
5. OpenCode listed the synced provider and **processed tokens through it**:
   `opencode run` returned a per-run sentinel with exit code 0.

Then the deployment was stopped and its absence confirmed.

## The failures were the point

Nine live attempts produced three certified rows. Each failure named a defect
that unit tests and screenshots could not have reached:

| Attempt | What it exposed |
| --- | --- |
| Modal 1–2 | The harness built `/v1/v1/chat/completions`. **The harness was wrong**; the unauthenticated Modal endpoint it seemed to prove was real, but was established by reading the code, not by this probe. |
| Modal 3 | The connection summary is published *after* the operation completes, so reading it the instant Deploy reports success returns nothing. |
| Modal 4 | `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set` — OpenCode's first message refused outright, on an endpoint that had just streamed 200 chunks. Qwen 3.5/3.8 name themselves past the `qwen3-` prefix the recommendation matched. |
| Modal 5 | Passed, but the answer carried the model's own `</think>`: no reasoning parser, while the synced OpenCode provider declared `reasoning_content`. |
| Vast 1 | `no_such_ask` — the offer was taken between listing and renting, reported as a bare `HTTP 400`. |
| Vast 2 | A host below the image's CUDA floor, refused before renting (the floor working as designed). |
| Vast 3–4 | `RuntimeError: Engine core initialization failed. See root cause above.` with the above already scrolled out of a 25-line window. The cause, once the window was widened: `max_num_seqs (1024) exceeds available Mamba cache blocks (674)` — Vast pins a newer vLLM whose default concurrency a hybrid model cannot hold. |

## What this does not certify

Image input, tensor parallelism above 1, the full 262k context, sustained
throughput, or answer quality. A stage measures the path, not the hardware's
best case. The llama.cpp path is out of scope here, and the Modal llama.cpp
endpoint still has the authentication gap its vLLM sibling no longer does.
