# Storage and costs

## Storage by provider

The Storage screen is provider-scoped: Modal shows its shared volume cache
with file inventory, Prime shows persistent cache disks with per-disk billing,
and Vast explains that rental-local disks are destroyed with the rental and
have no separate inventory.

Stopping compute and deleting storage are separate operations, with different
consequences per provider:

- Modal: Stop app; shared volume/cache remains for faster redeploys.
- Prime: Terminate pod; persistent cache disk remains and keeps billing until
  deleted with `llm-launchpad prime-disks delete <id>`.
- Vast: Destroy rental; rental disk and cached models are deleted.

`llm-launchpad stop` prints this consequence before confirming.

## Model weight cache (Modal)

Downloaded model weights are cached in the Modal `huggingface-cache` volume so
repeated deploys can start faster. Use the TUI Storage screen to refresh the
cache inventory, predownload a model, or delete selected cached weights when
they are no longer needed.

If storage size looks stale after a deployment or delete, refresh the Storage
screen to reload the Modal volume snapshot.

## Prime persistent disks

Prime deploys use a persistent cache disk (100 GB floor, larger when the
planner's weight estimate needs headroom) so model weights survive across
pods. Disks remain billable after a pod stops; see
[Prime Intellect provider](prime.md#persistent-cache-disks) for details.

## Passive monitoring and scaledown

Home and Manage never wake a Modal container to ask how it is doing.
Background refreshes read provider metadata plus locally banked totals only:
a deployed Modal app without an explicit check shows "health not checked",
traffic shows last-observed totals with no live rate, and no `/metrics` or
`/health` request is sent. Leaving either screen open cannot extend the idle
timeout or create GPU billings.

To check a Modal endpoint explicitly, use Manage -> Check status. That
one-shot probe may start the GPU; its verdict is remembered and later
passive refreshes show its age ("checked 5m ago") instead of reprobing.
Deploy warmup records the same observation when certification succeeds.
Prime and Vast bill continuously, so their rows keep live probing.

## Costs and scaledown

GPU costs depend on the selected provider's billing model. Every placement
leads with its hourly rate while billed and its 24/7 ceiling; monthly usage
is an explicit scenario, never an invented schedule:

- Always on: running 24/7 with no shutdown.
- Workday 8h: up 8h/day with shutdown outside the window.
- Sparse vs clustered: identical active time spread over many or few sessions.

Modal scale-to-zero bills active compute plus one idle timeout per session, so
sparse requests bill more than clustered ones with the same active time.
Modal deployments default the scaledown window to 1800 seconds; change it in
Settings or with `SCALEDOWN_WINDOW` before deploying. Vast hourly totals
already include disk rent; other providers report storage separately, and
unknown storage cost is not $0.

Modal Volume storage costs use Modal's `$0.09 / GiB / month` list price with a
`1 TiB / month` free tier; the TUI billing panel and Storage screen show the
estimated billable amount.

For predictable costs:

- Stop apps you no longer need with `llm-launchpad stop`.
- Prefer smaller GPU layouts for quick tests before moving to larger models.
- Use the warmup command only when you actually need the endpoint ready immediately.
- Treat displayed cost estimates as guidance and confirm current provider pricing for production workloads.
