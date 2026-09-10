# Vast certification status

This records the boundary between implemented behavior and measured serving
results for PR #77. Unit tests and screenshots do not certify rented hardware.
Historical results below come from the PR's recorded live runs; they were not
repeated by the follow-up that fixes the validation harness.

| Path | Recorded evidence | Release behavior |
| --- | --- | --- |
| Single-GPU text-only llama.cpp | 48.7s cold start; unauthorized requests rejected; structured tool call; 310s stream; reconnect; 8.8s warm restart; confirmed destruction | Enabled |
| Two-GPU text-only llama.cpp | 15.7 GB Q8_0 model split over two RTX 3060s, with 7729/7791 MiB used; chat, tools, 60s stream; confirmed destruction | Enabled; other topologies are not individually certified |
| Fast Deploy TUI | Real llama.cpp rental deployed and warmed through UI/worker path; confirmed destruction | Enabled |
| Current vLLM startup and serving | SSH reached a vLLM image, but no complete serving certification after the hook was reverted | Experimental opt-in required |
| vLLM tensor parallelism | No completed live certification | Experimental opt-in required |
| Advanced deploy on a real rental | Harness available; no completed recorded live run | llama.cpp text enabled; vLLM/vision require opt-in |
| Image inference / GGUF projector staging | No completed live certification | Experimental opt-in required |

The historical certification total was $0.78. All rentals were eventually
destroyed and confirmed absent, including one recovered after cleanup was rate
limited. These figures are historical, not an estimate for the next run.

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
   reconcile reported credit changes between stages.
4. Save the report and screenshots. Do not remove the recovery record while
   creation or destruction is uncertain. Do not publish credentials or private
   runtime scripts with the evidence.

Example serving stage (replace the offer, model, limits, and report path):

```bash
LLM_LAUNCHPAD_VAST_EXPERIMENTAL=1 uv run python scripts/validate_vast_live.py \
  --live --stage vllm_single_gpu --model Qwen/Qwen3-0.6B \
  --offer-id OFFER_ID --gpu-count 1 --min-cuda 13 --min-compute 7.5 \
  --max-hourly-cost 0.20 --budget-usd 0.50 --max-minutes 20 \
  --transfer-gb 30 --stream-seconds 60 --report /tmp/vast-vllm-single.json
```

Use `vllm_tensor_parallel` with a two-GPU offer for the next stage, verifying
that every GPU carries weights. Then run `scripts/validate_vast_custom_live.py`
with `--backend`, `--model`, the exact offer and budget, and `--vision` for image
inference. `--rehearse` drives the custom form without renting. Use the normal
product configuration and preserve the no-hook startup for certification.

## SSH investigation still open

Earlier large-image vLLM rentals reported `running` but rejected the per-rental
key. The `onstart` key-installation experiment caused additional refusals and
was reverted. Reattaching the same key was also tried without fixing the issue.
Do not reintroduce either as an assumed solution.

If the current configuration still fails, collect the instance ID, machine ID,
image digest, create/attach/running timestamps, and sanitized SSH error before
cleanup. The question for Vast support is whether per-instance key attachment
persists across a long image pull, and how custom `onstart` interacts with the
image's startup script and sshd provisioning. A successful attachment response
alone is not evidence that the container accepts the key.

The experimental gate should be removed only after the corresponding serving
and cleanup stages pass on the actual release configuration. Further host/GPU
coverage remains necessary before describing the provider as generally certified.
