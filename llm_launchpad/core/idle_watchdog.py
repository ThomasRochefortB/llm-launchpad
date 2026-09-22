"""An on-machine watchdog that destroys an idle rental.

Vast rentals and Prime pods bill until they are destroyed, and the moment
people forget one is when the laptop that started it is closed. So the
watchdog runs on the rented machine itself, not here: it reads the serving
runtime's own ``/metrics`` and, once no tokens have moved for the idle window,
asks the provider to delete the machine it runs on.

Activity is what the runtime counted, not what reached the port. Launchpad's
fleet probe reads ``/metrics`` too, and reading counters does not change them,
so watching an endpoint never keeps it alive; serving a request always does.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass

# Written next to the runtime on the machine; ``watchdog.state`` is what
# ``vast connect``/logs can read to see when the endpoint was last busy.
WATCHDOG_SCRIPT_NAME = "idle-watchdog.sh"
WATCHDOG_LOG_NAME = "idle-watchdog.log"
WATCHDOG_STATE_NAME = "idle-watchdog.state"

DEFAULT_IDLE_SHUTDOWN_SECONDS = 3600
# A runtime that never answers is billing for nothing too, but first starts
# legitimately download tens of GB; this only ends a rental that has shown no
# sign of serving for far longer than any certified startup took.
STARTUP_GRACE_SECONDS = 3 * 3600
POLL_SECONDS = 60

# Counters that move whenever the runtime serves anything, and gauges that are
# nonzero while a request is in flight (a long prefill moves no counter).
ACTIVITY_COUNTERS = (
    "llamacpp:prompt_tokens_total",
    "llamacpp:tokens_predicted_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
)
IN_FLIGHT_GAUGES = (
    "llamacpp:requests_processing",
    "llamacpp:requests_deferred",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
)


@dataclass(frozen=True)
class WatchdogSpec:
    """Everything the on-machine script needs, rendered into it at deploy."""

    metrics_url: str
    endpoint_api_key: str
    idle_seconds: int
    # Shell that deletes this machine; runs with ``set -e`` semantics off so a
    # failed attempt is retried rather than ending the watchdog.
    destroy_command: str
    runtime_dir: str
    startup_grace_seconds: int = STARTUP_GRACE_SECONDS
    poll_seconds: int = POLL_SECONDS


def _pattern(names: tuple[str, ...]) -> str:
    return "^(" + "|".join(name.replace(".", "\\.") for name in names) + ")([{ ])"


def watchdog_script(spec: WatchdogSpec) -> str:
    """POSIX sh + curl: both runtime images and Prime's host carry them."""
    if spec.idle_seconds <= 0:
        raise ValueError("An idle watchdog needs a positive idle window.")
    state = f"{spec.runtime_dir}/{WATCHDOG_STATE_NAME}"
    return "\n".join(
        [
            "#!/bin/sh",
            "# llm-launchpad idle watchdog: destroys this machine once its model",
            "# has served nothing for the idle window. Delete this file and kill",
            "# the process to keep the machine running.",
            "umask 077",
            f"METRICS_URL={shlex.quote(spec.metrics_url)}",
            f"ENDPOINT_KEY={shlex.quote(spec.endpoint_api_key)}",
            f"IDLE={int(spec.idle_seconds)}",
            f"GRACE={int(spec.startup_grace_seconds)}",
            f"POLL={int(spec.poll_seconds)}",
            f"STATE={shlex.quote(state)}",
            f"COUNTERS={shlex.quote(_pattern(ACTIVITY_COUNTERS))}",
            f"GAUGES={shlex.quote(_pattern(IN_FLIGHT_GAUGES))}",
            "destroy() {",
            '  echo "$(date -u +%FT%TZ) destroying: $1"',
            "  attempt=0",
            "  while [ $attempt -lt 30 ]; do",
            "    attempt=$((attempt + 1))",
            f"    if {spec.destroy_command}; then",
            '      echo "$(date -u +%FT%TZ) destroy requested"',
            "      exit 0",
            "    fi",
            '    echo "$(date -u +%FT%TZ) destroy attempt $attempt failed"',
            "    sleep 60",
            "  done",
            '  echo "$(date -u +%FT%TZ) giving up; the machine is still billing"',
            "  exit 1",
            "}",
            "started=$(date +%s)",
            "last_active=$started",
            "last_counters=''",
            "healthy=0",
            'echo "$(date -u +%FT%TZ) watching $METRICS_URL; idle window ${IDLE}s"',
            "while :; do",
            "  now=$(date +%s)",
            '  if body=$(curl -fsS -m 10 -H "Authorization: Bearer $ENDPOINT_KEY" "$METRICS_URL" 2>/dev/null); then',
            "    if [ $healthy -eq 0 ]; then",
            '      echo "$(date -u +%FT%TZ) runtime is serving; idle clock started"',
            "      healthy=1",
            "      last_active=$now",
            "    fi",
            '    counters=$(printf "%s\\n" "$body" | grep -E "$COUNTERS" | sort)',
            '    in_flight=$(printf "%s\\n" "$body" | grep -E "$GAUGES" '
            "| awk '{ s += $NF } END { print (s > 0) ? 1 : 0 }')",
            '    if [ "$counters" != "$last_counters" ] || [ "$in_flight" = 1 ]; then',
            "      last_active=$now",
            '      last_counters="$counters"',
            "    fi",
            "  fi",
            '  printf "%s %s %s\\n" "$healthy" "$last_active" "$IDLE" > "$STATE"',
            "  if [ $healthy -eq 0 ] && [ $((now - started)) -ge $GRACE ]; then",
            '    destroy "the model never started serving"',
            "  fi",
            "  if [ $healthy -eq 1 ] && [ $((now - last_active)) -ge $IDLE ]; then",
            '    destroy "idle for $((now - last_active))s"',
            "  fi",
            "  sleep $POLL",
            "done",
            "",
        ]
    )


def vast_destroy_command() -> str:
    """Delete this Vast instance with the key Vast scopes to it.

    Vast gives every instance ``CONTAINER_ID`` and a ``CONTAINER_API_KEY``
    limited to managing itself, so the user's account key never leaves their
    computer. SSH sessions do not inherit the container environment, so the
    values are read from PID 1's.
    """
    lookup = (
        "tr '\\0' '\\n' < /proc/1/environ 2>/dev/null | sed -n \"s/^$1=//p\" | head -n 1"
    )
    return (
        "{ "
        f'id="${{CONTAINER_ID:-$(set -- CONTAINER_ID; {lookup})}}"; '
        f'key="${{CONTAINER_API_KEY:-$(set -- CONTAINER_API_KEY; {lookup})}}"; '
        '[ -n "$id" ] && [ -n "$key" ] && '
        'curl -fsS -m 30 -X DELETE -H "Authorization: Bearer $key" '
        '"https://console.vast.ai/api/v0/instances/$id/" >/dev/null; }'
    )


def prime_destroy_command(api_base_url: str, pod_id: str, key_path: str) -> str:
    """Delete this Prime pod with the account key the user opted to place here."""
    url = f"{api_base_url.rstrip('/')}/pods/{pod_id}"
    return (
        "{ "
        f'key="$(cat {shlex.quote(key_path)} 2>/dev/null)"; '
        '[ -n "$key" ] && '
        'curl -fsS -m 30 -X DELETE -H "Authorization: Bearer $key" '
        f"{shlex.quote(url)} >/dev/null; }}"
    )


def start_command(runtime_dir: str) -> str:
    """Detach the watchdog from the SSH session that starts it."""
    script = f"{runtime_dir}/{WATCHDOG_SCRIPT_NAME}"
    log = f"{runtime_dir}/{WATCHDOG_LOG_NAME}"
    return f"nohup sh {shlex.quote(script)} >> {shlex.quote(log)} 2>&1 < /dev/null &"


def resolve_idle_shutdown(config: object, settings: object | None = None) -> int:
    """The deployment's own window, else the user's setting."""
    explicit = getattr(config, "idle_shutdown_seconds", None)
    if explicit is not None:
        return max(0, int(explicit))
    if settings is None:
        from .config import ConfigStore

        try:
            settings = ConfigStore().load()
        except Exception:
            return DEFAULT_IDLE_SHUTDOWN_SECONDS
    return max(0, int(getattr(settings, "rental_idle_shutdown", DEFAULT_IDLE_SHUTDOWN_SECONDS)))


def describe_idle_shutdown(seconds: int | None) -> str:
    """Plain wording for the confirm screen and logs."""
    if not seconds or seconds <= 0:
        return "off; bills until you stop it"
    if seconds % 3600 == 0:
        window = f"{seconds // 3600}h"
    elif seconds % 60 == 0:
        window = f"{seconds // 60}m"
    else:
        window = f"{seconds}s"
    return f"deleted after {window} with no requests"
