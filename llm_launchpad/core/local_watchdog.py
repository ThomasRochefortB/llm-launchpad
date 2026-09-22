"""Idle shutdown for a Prime pod, run from this computer.

The fallback when the user has not agreed to place their Prime API key on the
pod: the same idle rule as the on-machine watchdog, but it can only act while
this computer is awake. It is a detached process so closing the TUI does not
end it, and it exits as soon as the pod is gone for any reason.

Run as ``python -m llm_launchpad.core.local_watchdog <args>``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

from .config import SETTINGS_DIR
from .idle_watchdog import ACTIVITY_COUNTERS, IN_FLIGHT_GAUGES

LOCAL_WATCHDOG_LOG_DIR = SETTINGS_DIR / "logs"


def activity_signature(metrics_text: str) -> tuple[tuple[str, ...], bool]:
    """Counter lines that move with any served token, and whether one is in flight."""
    counters: list[str] = []
    in_flight = False
    for line in metrics_text.splitlines():
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in ACTIVITY_COUNTERS:
            counters.append(line.strip())
        elif name in IN_FLIGHT_GAUGES:
            try:
                in_flight = in_flight or float(line.rsplit(" ", 1)[-1]) > 0
            except ValueError:
                pass
    return tuple(sorted(counters)), in_flight


@dataclass
class IdleClock:
    """The watchdog's rule, kept apart from I/O so it can be tested exactly."""

    idle_seconds: int
    started_at: float
    last_active: float | None = None
    _counters: tuple[str, ...] | None = None

    def observe(self, now: float, metrics_text: str | None) -> None:
        if metrics_text is None:
            return
        counters, in_flight = activity_signature(metrics_text)
        if self.last_active is None or in_flight or counters != self._counters:
            self.last_active = now
        self._counters = counters

    def expired(self, now: float) -> bool:
        # Idle time only counts once the runtime has served: a long download is
        # not idleness, and the on-pod startup deadline covers a hung start.
        return self.last_active is not None and now - self.last_active >= self.idle_seconds


def run(
    *,
    metrics_url: str,
    endpoint_api_key: str,
    idle_seconds: int,
    pod_exists: Callable[[], bool],
    stop: Callable[[], None],
    poll_seconds: float = 60.0,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Watch until the pod is idle (then stop it) or gone; returns why it ended."""
    import requests

    idle = IdleClock(idle_seconds=idle_seconds, started_at=clock())
    while True:
        if not pod_exists():
            return "pod is gone"
        text: str | None = None
        try:
            response = requests.get(
                metrics_url,
                headers={"Authorization": f"Bearer {endpoint_api_key}"},
                timeout=15,
            )
            if response.ok:
                text = response.text
        except requests.RequestException:
            text = None
        idle.observe(clock(), text)
        if idle.expired(clock()):
            stop()
            return "stopped after idle window"
        sleep(poll_seconds)


def spawn_local_watchdog(
    *,
    app_name: str,
    backend: str,
    pod_id: str,
    endpoint_url: str,
    endpoint_api_key: str,
    idle_seconds: int,
) -> None:
    """Start the watchdog detached from this process and its terminal."""
    LOCAL_WATCHDOG_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOCAL_WATCHDOG_LOG_DIR / f"idle-watchdog-{app_name}.log"
    root = endpoint_url.rstrip("/").removesuffix("/v1")
    with open(log_path, "a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable, "-m", "llm_launchpad.core.local_watchdog",
                "--app-name", app_name,
                "--backend", backend,
                "--pod-id", pod_id,
                "--metrics-url", f"{root}/metrics",
                "--idle-seconds", str(idle_seconds),
            ],
            # The key goes by stdin so it never appears in the process list.
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    # Hand over the key and let go; the watchdog outlives this process.
    assert process.stdin is not None
    process.stdin.write((endpoint_api_key + "\n").encode())
    process.stdin.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-launchpad-local-watchdog")
    parser.add_argument("--app-name", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--metrics-url", required=True)
    parser.add_argument("--idle-seconds", type=int, required=True)
    args = parser.parse_args(argv)
    endpoint_api_key = sys.stdin.readline().strip()
    sys.stdin.close()

    from ..protocol.enums import BackendType, ComputeProvider
    from .orchestrator import Orchestrator
    from .prime_backend import PrimeApiError, PrimeBackend

    prime = PrimeBackend()

    def pod_exists() -> bool:
        try:
            pod = prime.get_pod(args.pod_id)
        except PrimeApiError as exc:
            # Only a definite "not found" ends the watch; a network blip must
            # not leave a pod unwatched for the rest of its life. Prime moves
            # terminated pods to history, where Get Pod no longer finds them.
            return exc.status_code != 404
        except Exception:
            return True
        return str(pod.get("status") or "").upper() not in {"TERMINATED", "TERMINATING"}

    def stop() -> None:
        events = Orchestrator().stop_app(
            BackendType(args.backend),
            app_name=args.app_name,
            provider=ComputeProvider.PRIME,
            app_id=args.pod_id,
        )
        for event in events:
            line = getattr(event, "line", None) or getattr(event, "message", None)
            if line:
                print(line, flush=True)

    print(
        f"{time.strftime('%Y-%m-%dT%H:%M:%S')} watching {args.app_name} "
        f"({args.pod_id}); idle window {args.idle_seconds}s",
        flush=True,
    )
    reason = run(
        metrics_url=args.metrics_url,
        endpoint_api_key=endpoint_api_key,
        idle_seconds=args.idle_seconds,
        pod_exists=pod_exists,
        stop=stop,
    )
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {reason}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
