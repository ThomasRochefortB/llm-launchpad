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

## Idle shutdown for rentals

Vast rentals and Prime pods bill until they are deleted, so Launchpad deletes
one after it has served nothing for an hour. Change the window, or turn it off,
under Settings -> Providers ("Delete idle Vast and Prime rentals after"). The
confirm screen states the rule for the placement you are about to rent.

Idle means the runtime's own token counters stopped moving and no request is
in flight. Reading `/metrics` does not count, so Manage and the home screen
never keep a rental alive; any served request does. The clock starts when the
model starts serving, so a long first download is not idle time. A runtime
that never serves is deleted after three hours.

- **Vast** runs the watchdog on the rental itself, so it works with this
  computer asleep or off. It deletes the rental with `CONTAINER_API_KEY`, which
  Vast scopes to that one instance; your account key never leaves this
  computer.
- **Prime** has no instance-scoped key. By default Launchpad watches from this
  computer in a detached process, which only acts while the computer is awake.
  Turn on "Let Prime pods stop themselves" to run the watchdog on the pod
  instead; that stores your Prime API key on the pod (`/opt/llm-launchpad`,
  root-only, mode 600).

Deleting a rental removes its disk and cached weights on Vast; Prime keeps its
persistent cache disk, which still bills (see above). A rental that deleted
itself shows as `destroyed` in Manage until you stop it, which removes the
local record. The watchdog logs to `idle-watchdog.log` next to the runtime, or
`~/.llm_launchpad/logs/idle-watchdog-<name>.log` for the local Prime watchdog.

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
