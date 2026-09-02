# Storage and costs

## Model weight cache (Modal)

Downloaded model weights are cached in the Modal `huggingface-cache` volume so
repeated deploys can start faster. Use the TUI Storage screen to refresh the
cache inventory, predownload a model, or delete selected cached weights when
they are no longer needed.

Stopping an app and deleting cached weights are separate operations:
`llm-launchpad stop` stops a deployed Modal app, while the Storage screen
manages cached model files. If storage size looks stale after a deployment or
delete, refresh the Storage screen to reload the Modal volume snapshot.

## Prime persistent disks

Prime deploys use a 100 GB persistent cache disk by default so model weights
survive across pods. Disks remain billable after a pod stops; see
[Prime Intellect provider](prime.md#persistent-cache-disks) for details.

## Costs and scaledown

GPU costs depend on the selected provider's billing model. Modal inference can
scale to zero, while a Prime pod is billed for the full provisioned serving
window. LLM-Launchpad shows both the provider's hourly price and a normalized
monthly estimate so those options can be compared without pretending their
idle costs are equivalent. Modal deployments default the scaledown window to
1800 seconds; change it in Settings or with `SCALEDOWN_WINDOW` before
deploying.

Modal Volume storage costs use Modal's `$0.09 / GiB / month` list price with a
`1 TiB / month` free tier; the TUI billing panel and Storage screen show the
estimated billable amount.

For predictable costs:

- Stop apps you no longer need with `llm-launchpad stop`.
- Prefer smaller GPU layouts for quick tests before moving to larger models.
- Use the warmup command only when you actually need the endpoint ready immediately.
- Treat displayed cost estimates as guidance and confirm current provider pricing for production workloads.
