# Image input (vision)

LLM-Launchpad enables image input automatically for models that prove they
support it, on both backends and both providers. Image input means images in,
text out; audio, video, and image generation are out of scope.

## Modes

`--vision` (and the *Image input* control on the deploy screens) accepts:

| Mode | Behavior |
| --- | --- |
| `auto` (default) | Enable images when the pinned model revision shows an image encoder, an image processor, or a projector. Unknown models stay text-only. |
| `on` | Require images. Fails before provisioning if the model is known to be text-only, or if a required projector cannot be resolved. |
| `off` | Text-only. llama.cpp is launched with `--no-mmproj`, and vLLM's multimodal limits are set to zero. |

Capability is read from revision-pinned Hugging Face metadata — `config.json`,
`preprocessor_config.json`, GGUF headers, and the repository file list. A model's
name or pipeline tag alone never establishes support.

Three states are tracked separately, and the TUI's *Connection Info* screen
shows all of them:

- whether the **model** supports images,
- whether images are **enabled** for this deployment,
- whether an image request has actually been **verified** against it.

## Verification

When warmup runs for a vision-enabled deployment, Launchpad sends a bundled
64×64 PNG to `/v1/chat/completions` after the ordinary readiness probe. A
failure fails deployment validation and returns a non-zero exit code.

Because the server already answered the readiness probe, an image-probe
failure is reported distinctly from a failed rollout: the deployment is left
running so you can inspect it, on both providers and from both the TUI and the
CLI. Every other warmup failure still tears the deployment down as before.

This verifies that the request path accepts an image and returns assistant
text — `content`, content parts, or a thinking model's `reasoning_content`. It
is not a benchmark of visual accuracy.

```bash
# Verify an already-running deployment on demand, and republish the result.
llm-launchpad warmup --instance-name my-vl-model --image-test
```

`--image-test` only works on a deployment that already enabled image input; it
will not invent a capability for a text-only or pre-vision endpoint. On success
it re-syncs OpenCode, so a verified deployment starts advertising image input
immediately. Interrupting a probe leaves verification untouched rather than
recording a failure.

*Connection Info* offers **Copy image request**, which yields the same request
as a `curl` command with the API key left as `$LLM_LAUNCHPAD_API_KEY`.

Skipping warmup leaves vision `untested`. Verification is bound to the
deployment's identity (model, revision, quant, projector, runtime, and vision
settings) and resets whenever any of those change.

## llama.cpp projectors

llama.cpp needs a separate `mmproj` GGUF alongside the model weights. Launchpad
resolves it from the selected repository and revision, downloads it
independently of the model's quantization filter, and passes it explicitly with
`--mmproj` on both providers rather than relying on automatic discovery.

If a repository ships more than one projector, deployment stops and asks you to
choose:

```bash
llm-launchpad deploy \
  --repo-id unsloth/Qwen3-VL-8B-Instruct-GGUF \
  --quant Q4_K_M \
  --projector-file mmproj-F16.gguf
```

`--projector-repo` and `--projector-revision` pull the projector from a
different repository or pin it to a different commit. Pass projector settings
through these flags only — raw `--mmproj` arguments inside `--server-args` are
rejected before any compute is allocated, so that planning and serving cannot
diverge.

Projector files are never selected as the main model, and are excluded from the
quantization lists shown in the model pickers.

## vLLM

Both providers serve vLLM `v0.19.1`. Image requests use vLLM's native
multimodal path, controlled by two vLLM-specific options:

- `--image-limit` — maximum images per prompt (default `1`).
- `--mm-processor-kwargs` — a JSON object forwarded to the image processor.

Video and audio are always disabled.

## Limitations

- **Fast Deploy skips vision models.** Image working memory cannot be estimated
  from GGUF headers, so vision-capable models are excluded from guaranteed-fit
  recommendations. Deploy them from the manual deploy screens or the CLI
  instead.
- **Manual vLLM estimates exclude image working memory.** The estimate line says
  so when the model is multimodal; leave headroom accordingly.
- **OpenCode advertises `image` input only after verification passes.** Until
  then the model is registered as text-only, so an unverified deployment will
  refuse attachments in OpenCode rather than fail mid-request.
- **Fast Deploy is text-only.** Its plans certify a placement that does not
  model image memory, so the image-input control is not offered there.
- **Existing deployments are untouched** until redeployed. Records saved before
  image support load with unknown vision status and text-only advertising.
