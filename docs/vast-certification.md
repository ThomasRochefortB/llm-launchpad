# Vast certification status

This records the boundary between implemented behavior and measured serving
results for PR #77. Unit tests and screenshots do not certify rented hardware.
Rows marked 2026-09-10 were measured on the release configuration, after the
`onstart` hook was reverted and the validation harness was repaired.

| Path | Recorded evidence | Release behavior |
| --- | --- | --- |
| Single-GPU text-only llama.cpp | 2026-09-11, RTX 3060 (offer 45601619): 99.8s cold start; **1245 MiB resident on the device**; 401 on unauthorized requests; structured tool call; 60s stream over 5882 chunks; 6.1s warm restart; confirmed destruction. Supersedes the original 48.7s run, which predated `devices_after_load` | Enabled |
| Two-GPU text-only llama.cpp | 15.7 GB Q8_0 model split over two RTX 3060s, with 7729/7791 MiB used; chat, tools, 60s stream; confirmed destruction | Enabled; other topologies are not individually certified |
| Fast Deploy TUI | Real llama.cpp rental deployed and warmed through UI/worker path; confirmed destruction | Enabled |
| Single-GPU vLLM | 2026-09-10, RTX 3060: 876.6s cold start; unauthorized requests rejected (401, 401); structured tool call; 60s stream; confirmed destruction | Enabled |
| vLLM tensor parallelism | 2026-09-10, two RTX 3060s: 531.1s cold start with 11641 MiB resident on both devices; 60s stream over 1563 chunks; post-cancel chat; reconnect to the same URL; confirmed destruction | Enabled for two-way; four- and eight-way are not individually certified |
| Advanced deploy on a real rental, with llama.cpp image input | 2026-09-10, RTX 3060: the Advanced form drove a real rental; a 108.8 MB GGUF projector staged at a pinned revision; the model answered a question about an image; 97.7s deploy and warmup; confirmed destruction | Enabled |
| vLLM image input | 2026-09-11, RTX A5000 (offer 48465829): Advanced deploy drove a real rental; 232.2s deploy and warmup; 19309 MiB resident; the model answered an image request and chatted afterwards; confirmed destruction | Enabled. Verifies the request path and a non-empty answer, not visual accuracy |
| Hosts below the image's CUDA build version | 2026-09-11, RTX 3060 reporting `cuda_max_good` 12.2 (offer 45598047): 403.2s cold start, 401 on unauthorized requests, structured tool call, 60s stream over 1849 chunks, post-cancel chat, reconnect, 6.2s warm restart, confirmed destruction — but `nvidia-smi` reported **25 MiB** used on the device | Refused. The server ran; the weights did not reach the GPU |

The single-GPU llama.cpp rerun settled at $0.0083 and the vLLM image-input stage at $0.0097. The 2026-09-10 stages cost
$0.050 in total, measured as reported credit before
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
on a CUDA 12.2 host the device held 25 MiB while serving. A later run put the
same model and quant on the same GPU model above the floor, which settles it:

| | CUDA 12.2 (45598047) | CUDA 12.8 (45601619) |
| --- | --- | --- |
| GPU memory in use | 25 MiB | **1245 MiB** |
| `every_device_holds_weights` | false | **true** |
| 60s stream | 1849 chunks | **5882 chunks** |
| Cold start | 403.2s | 99.8s |

Identical RTX 3060 hardware, model, quant and harness, 3.2x the tokens, and
weights actually resident.
The image falls back to CPU rather than refusing to start, which is why a
driver check before rental is the only thing standing between a user and GPU
prices for CPU inference.

That run settled at $0.0082 ($9.1752 to $9.1670), against the $0.0041 its own
report recorded at destruction: transfer billing landing minutes later again.
Read the account, not the report.

Two harness gaps this exposed, both now closed:

- `devices_after_load` did not exist when single-GPU llama.cpp was certified, so
  that run never verified GPU residency at all. The row above is now a rerun
  that does.
- The idle-device guard only raised for multi-GPU rentals, so a single idle GPU
  reported `success: true`. It now fails at any GPU count.

`logs.gpu_offload_reported` is **not** evidence either way: it is false on the
certified 12.8 run too, because the log line it greps for is not emitted.

## Why a rental gets 1800s to answer SSH

The deploy path used to allow 900s on llama.cpp and 1800s on vLLM, scaled on
image size. Three attempts on 2026-09-11 showed that is the wrong variable:

| Host | Link | Result |
| --- | --- | --- |
| RTX 3060, Poland (45598047) | 891 Mbps | SSH in time; 403.2s cold start |
| RTX 3060, New Brunswick (48529480) | 1890 Mbps | reached `running`, refused SSH past 900s |
| RTX 3060, New Jersey (44022327) | 2065 Mbps | still `loading` at 900s |

Two of three failed and the faster links were the ones that failed, at $0.0273
for no result. The pull cannot explain it: the pinned images are 2.59 GB
(llama.cpp, 13 layers) and 8.67 GB (vLLM, 37 layers) compressed, which is 11s
and 37s at 1890 Mbps. What actually runs before sshd binds is unpacking those
layers and Vast's own provisioning, which installs the SSH server the pinned
upstream images do not carry — visible in the instance `status_msg` as BuildKit
steps fetching from `archive.ubuntu.com`. Neither scales with the image, and
neither is measured by `inet_down`, which is a speedtest figure.

So both runtimes now get the same 1800s (`VAST_READY_DEADLINE_SECONDS`), and a
host that stops reporting progress for 360s before SSH is reachable is
abandoned early (`VAST_PROVISION_STALL_SECONDS`) rather than billed to the
deadline. Progress is read from `status_msg`, with BuildKit's `#step elapsed`
prefix stripped first: that counter keeps ticking while a step is wedged, so
the raw string is not a progress signal.

This does not make `inet_down` predictive. `disk_bw` is not either — both failed
hosts had NVMe-class disks (2538 and 3679 MB/s). `cpu_cores_effective` remains
an untested candidate, since layer decompression is CPU-bound and the New
Brunswick host was allotted 4 of its 16 cores.

The rerun that closed this also exercised the new wait: the host reported a
docker pull line at 32s, nothing at 64s, and reached SSH without the stall
window closing. Cold start was 99.8s against 403.2s on the slower host.

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
4. Remember that `--max-minutes` is the harness's own clock, not the deploy
   path's. A rental now gets 1800s to answer SSH on either runtime, and is
   abandoned sooner if it stops reporting progress. See the section below.
5. Save the report and screenshots. Do not remove the recovery record while
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

Every runtime and feature pairing the provider offers now has a live run behind
it. The vision stages verify the request path and a non-empty answer: the probe
sends a 64x64 solid-red PNG and `verify_image_request` does not judge visual
accuracy, so neither engine's row claims the model perceived the image well.

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

The 2026-09-11 runs put a number on how weak a predictor it is: hosts
advertising 1890 and 2065 Mbps both failed to answer SSH inside 900s, while the
891 Mbps host connected. Raising the floor would have selected *against* the
host that worked. It bounds the pull, which is 11-78s of the wait; it says
nothing about unpacking or provisioning, which are the rest.

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
