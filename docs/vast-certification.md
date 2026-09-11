# Vast certification status

This records the boundary between implemented behavior and measured serving
results for PR #77. Unit tests and screenshots do not certify rented hardware.
Rows marked 2026-09-10 were measured on the release configuration, after the
`onstart` hook was reverted and the validation harness was repaired.

| Path | Recorded evidence | Release behavior |
| --- | --- | --- |
| Single-GPU text-only llama.cpp | 48.7s cold start; unauthorized requests rejected; structured tool call; 310s stream; reconnect; 8.8s warm restart; confirmed destruction | Enabled |
| Two-GPU text-only llama.cpp | 15.7 GB Q8_0 model split over two RTX 3060s, with 7729/7791 MiB used; chat, tools, 60s stream; confirmed destruction | Enabled; other topologies are not individually certified |
| Fast Deploy TUI | Real llama.cpp rental deployed and warmed through UI/worker path; confirmed destruction | Enabled |
| Single-GPU vLLM | 2026-09-10, RTX 3060: 876.6s cold start; unauthorized requests rejected (401, 401); structured tool call; 60s stream; confirmed destruction | Enabled |
| vLLM tensor parallelism | 2026-09-10, two RTX 3060s: 531.1s cold start with 11641 MiB resident on both devices; 60s stream over 1563 chunks; post-cancel chat; reconnect to the same URL; confirmed destruction | Enabled for two-way; four- and eight-way are not individually certified |
| Advanced deploy on a real rental, with llama.cpp image input | 2026-09-10, RTX 3060: the Advanced form drove a real rental; a 108.8 MB GGUF projector staged at a pinned revision; the model answered a question about an image; 97.7s deploy and warmup; confirmed destruction | Enabled |
| vLLM image input | No live run has served an image through vLLM | Enabled; implemented on vLLM's native multimodal path but not live-certified |
| Hosts below the image's CUDA build version | 2026-09-11, RTX 3060 reporting `cuda_max_good` 12.2 (offer 45598047): 403.2s cold start, 401 on unauthorized requests, structured tool call, 60s stream over 1849 chunks, post-cancel chat, reconnect, 6.2s warm restart, confirmed destruction — but `nvidia-smi` reported **25 MiB** used on the device | Refused. The server ran; the weights did not reach the GPU |

The 2026-09-10 stages cost $0.050 in total, measured as reported credit before
the first stage and after billing settled ($9.2250 to $9.1752). Each stage
confirmed destruction and absence before the next one started.

Read that total from the account, not by adding the stages up. Each report's
`credit_delta_usd` is taken the moment its rental is destroyed, and the three
deltas sum to $0.035 — about 30% short, because Vast keeps charging transfer
against the account for several minutes afterwards. The earlier llama.cpp
certification cost $0.78. These figures are historical, not an estimate for the
next run.

## The CUDA floor is measured, not assumed

`VAST_MIN_CUDA_VERSION` is 12.8 because that is the CUDA version in the pinned
llama.cpp image's OCI config. CUDA minor version compatibility suggests a 12.8
image should run on any 12.x driver, and on that reasoning the floor excludes
51 of 463 live offers on otherwise capable hardware, so it was tested.

The test passed every functional check and still failed the one that matters:
on a CUDA 12.2 host the device held 25 MiB while serving, against 7729/7791 MiB
on the certified two-GPU 12.8 run using the same RTX 3060 hardware and harness.
The image falls back to CPU rather than refusing to start, which is why a
driver check before rental is the only thing standing between a user and GPU
prices for CPU inference.

That run settled at $0.0082 ($9.1752 to $9.1670), against the $0.0041 its own
report recorded at destruction: transfer billing landing minutes later again.
Read the account, not the report.

Two harness gaps this exposed, both now closed:

- `devices_after_load` did not exist when single-GPU llama.cpp was certified, so
  that run never verified GPU residency at all.
- The idle-device guard only raised for multi-GPU rentals, so a single idle GPU
  reported `success: true`. It now fails at any GPU count.

`logs.gpu_offload_reported` is **not** evidence either way: it is false on the
certified 12.8 run too, because the log line it greps for is not emitted.

## Before another paid run

1. Choose an explicit total spend limit and reserve part for cleanup. Check
   current offers, per-device memory, driver/architecture floors, and transfer
   prices. A quoted hourly cap is not a total-spend cap.
2. Run the hermetic checks. The image probe now omits `onstart`, like production,
   and uses durable intent plus confirmed destruction. The custom vision probe
   requires a separate GGUF projector only for llama.cpp.
3. Run one stage at a time and confirm absence before starting another. The
   scripts estimate selected runtime plus five cleanup minutes, inbound traffic,
   and 1 GB outbound. Actual transfer volume and provider delays can differ;
   reconcile reported credit changes between stages, and read the account again
   a few minutes after the last rental, because transfer charges keep landing
   after destruction is confirmed.
4. Save the report and screenshots. Do not remove the recovery record while
   creation or destruction is uncertain. Do not publish credentials or private
   runtime scripts with the evidence.

The commands that produced the 2026-09-10 rows, in the order they were run.
Replace the limits and report paths; the offer filters re-select at rental time,
because an offer id read minutes earlier is usually gone.

```bash
uv run python scripts/validate_vast_live.py \
  --live --stage vllm_single_gpu --model Qwen/Qwen3-0.6B \
  --gpu-count 1 --min-gpu-memory-gb 12 --min-cuda 13 --min-compute 7.5 \
  --min-inet-down 800 --western-only \
  --max-hourly-cost 0.20 --budget-usd 0.40 --max-minutes 20 \
  --transfer-gb 30 --stream-seconds 60 --report /tmp/vast-vllm-single.json

uv run python scripts/validate_vast_live.py \
  --live --stage vllm_tensor_parallel --model Qwen/Qwen3-0.6B \
  --gpu-count 2 --min-gpu-memory-gb 12 --min-cuda 13 --min-compute 7.5 \
  --min-inet-down 800 --western-only \
  --max-hourly-cost 0.18 --budget-usd 0.35 --max-minutes 35 \
  --transfer-gb 30 --stream-seconds 60 --report /tmp/vast-vllm-tp.json

uv run python scripts/validate_vast_custom_live.py \
  --live --backend llamacpp \
  --model ggml-org/SmolVLM-500M-Instruct-GGUF --quant Q8_0 \
  --vision --projector-file mmproj-SmolVLM-500M-Instruct-Q8_0.gguf \
  --offer-id OFFER_ID \
  --max-hourly-cost 0.15 --budget-usd 0.30 --max-minutes 25 \
  --transfer-gb 10 --stream-seconds 60 --report /tmp/vast-llama-vision.json
```

The tensor-parallel stage verifies that every GPU carries weights, so it fails
rather than passing a rental that silently used one card. The custom script
drives the real Advanced Deploy form, which is why the third command certifies
advanced deploy and image input together; `--rehearse` drives that form without
renting. `scripts/validate_vast_custom_live.py` requires `--offer-id` and
re-prices it at launch, so read a current offer immediately before running it.
Use the normal product configuration and preserve the no-hook startup.

`vllm` with image input is the one release path with no live run behind it;
certifying it is the obvious next paid stage.

## The large-image SSH refusal was a startup race, not a rejected key

Instance 50546520 (machine 120972, `vllm/vllm-openai@sha256:c291476760…`)
reproduced the refusal on 2026-09-10 and then recovered on its own. The
sequence, which closes the question this section used to leave open:

1. Vast reported `running` while the container had been up for 1.1 seconds.
2. The deploy loop's first SSH attempt failed with `connect to host
   ssh4.vast.ai port 26520: Connection refused`.
3. About a minute later the same address answered a plain TCP connect with
   `SSH-2.0-OpenSSH_9.6p1 Ubuntu-3ubuntu13.19`.
4. The loop's next attempt authenticated with the per-rental key, enumerated
   the GPU, and served the model.

`Connection refused` means nothing was listening yet, so this is Vast's sshd
binding late after a long image pull, not the container refusing a key. The
instance carried `onstart = null`, confirming the reverted hook is absent from
the release configuration. Do not reintroduce the `onstart` experiment or a key
re-attach: neither addresses a listener that has not started.

The deploy loop already handles this correctly by backing off and retrying
instead of failing fast, so the only requirement is a deadline long enough to
cover the pull and the bind that follows it.

## Deadlines and host bandwidth

Both certified vLLM stages spent most of their cold start pulling the image, so
the deadline is the constraint that decides whether a stage finishes. Advertised
bandwidth predicts that wait only loosely: the two hosts below both advertised
about 868 Mbps and still differed by 345 seconds.

| Stage | Advertised `inet_down_mbps` | Cold start |
| --- | --- | --- |
| vLLM single GPU | 868 | 876.6s |
| vLLM two-way tensor parallelism | 868 | 531.1s |

Treat `--min-inet-down` as a floor that excludes hosts too slow to finish, not
as a predictor, and set `--max-minutes` with room for the slower outcome. A
20-minute deadline left the single-GPU stage about four minutes of margin, which
is thinner than it looks for a stage whose pull dominates the run.

Transfer price deserves more attention than the hourly rate, because
`select_offer` filters on `--min-inet-down` and then sorts by hourly price
alone. Among eligible two-GPU offers on 2026-09-10, download prices ranged from
$0.0026 to $0.039 per GB — enough to move a 30 GB estimate from $0.08 to $1.17,
and enough for the cheapest host per hour to be the most expensive run. When the
budget guard refuses a stage, check the selected offer's download price before
raising `--budget-usd`; lowering `--max-hourly-cost` to exclude a
cheap-per-hour, expensive-per-GB host is usually the better fix.

Further host and GPU-architecture coverage remains necessary before describing
the provider as generally certified; each row above is evidence from the hosts
named in it, not from the marketplace as a whole.
