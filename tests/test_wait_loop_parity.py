"""Every long wait has to keep saying it is still waiting.

Vast learned this first: "A minutes-long wait with one log line reads as a
hang." The lesson was never carried across to Prime, whose runtime wait and
pod-provisioning wait both emitted a line only when their *detail text*
changed -- and both details are constant for minutes at a time. A 109 GB model
download produced one line and then silence, on a screen billing by the hour.

Two things are pinned here. Every poll-until-ready loop in core is classified,
so a new provider cannot add a silent one without this failing; and the loops
that promise progress are driven against a fake clock to prove they deliver it.
"""

from __future__ import annotations

import pathlib
import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm_launchpad.core import orchestrator as orchestrator_module
from llm_launchpad.core.orchestrator import Orchestrator, PrimeApiError
from llm_launchpad.protocol.events import LogEvent

# Loops that poll a remote until it is ready. Each is either expected to report
# progress on an interval, or is short enough that silence is defensible -- and
# the bound that makes it defensible is written down.
REPORTS_PROGRESS = {
    ("orchestrator.py", "_await_prime_pod_active"),
    ("orchestrator.py", "_establish_prime_endpoint"),
    ("vast_deployment.py", "_deploy_locked"),
}
BOUNDED_SILENT = {
    # Cannot report by construction: not a generator, so it has no channel to
    # emit on. Capped at 120s, which is the whole justification.
    ("orchestrator.py", "_await_prime_public_endpoint"): 120,
}
# Not poll-until-ready at all: these wait a known interval before retrying, so
# there is no unknown remaining time to report on.
BACKOFF_SLEEPS = {
    ("warmup.py", "run"),
}

_POLL_LOOP_RE = re.compile(r"^\s*while time\.(?:monotonic|time)\(\) < ", re.MULTILINE)
_DEF_RE = re.compile(r"^\s*(?:async )?def (\w+)", re.MULTILINE)

# The cadence Vast set and Prime now matches, with room for one missed tick.
MAX_SILENCE_SECONDS = 60.0


def _enclosing_function(source: str, offset: int) -> str:
    last = ""
    for match in _DEF_RE.finditer(source, 0, offset):
        last = match.group(1)
    return last


def _discover_poll_loops() -> set[tuple[str, str]]:
    core = pathlib.Path(__file__).resolve().parents[1] / "llm_launchpad" / "core"
    found: set[tuple[str, str]] = set()
    for path in sorted(core.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for match in _POLL_LOOP_RE.finditer(source):
            found.add((path.name, _enclosing_function(source, match.start())))
    return found


class PollLoopRegistryTests(unittest.TestCase):
    def test_every_poll_loop_is_classified(self) -> None:
        discovered = _discover_poll_loops()
        classified = REPORTS_PROGRESS | set(BOUNDED_SILENT) | BACKOFF_SLEEPS

        unclassified = discovered - classified
        self.assertEqual(
            unclassified,
            set(),
            "A new poll-until-ready loop must declare whether it reports "
            "progress or is short enough to stay silent: "
            f"{sorted(unclassified)}",
        )

    def test_silent_loops_stay_short(self) -> None:
        # Silence is only defensible while it is brief.
        for loop, bound in BOUNDED_SILENT.items():
            with self.subTest(loop=loop):
                self.assertLessEqual(bound, 120)


class _FakeClock:
    """A clock that only advances when the code under test sleeps."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleeper(self, timeout: float | None = None) -> bool:
        self.now += timeout or 5.0
        return False


class PrimePodWaitReportsProgressTests(unittest.TestCase):
    """The pod-provisioning wait, driven against a pod that never becomes ready."""

    def _run(self) -> list[float]:
        clock = _FakeClock()
        backend = SimpleNamespace(
            get_pod=lambda _pod_id: {
                "status": "PROVISIONING",
                "installationStatus": "INSTALLING",
                "sshConnection": None,
            }
        )
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator._prime_backend = backend  # type: ignore[attr-defined]

        emitted_at: list[float] = []
        with (
            patch.object(orchestrator_module.time, "monotonic", clock.monotonic),
            patch.object(
                orchestrator_module,
                "shutdown_event",
                lambda: SimpleNamespace(wait=clock.sleeper),
            ),
        ):
            generator = orchestrator._await_prime_pod_active("pod-1", {})
            try:
                for event in generator:
                    if isinstance(event, LogEvent):
                        emitted_at.append(clock.now)
            except PrimeApiError:
                pass
        return emitted_at

    def test_it_reports_while_a_pod_sits_in_one_state(self) -> None:
        emitted_at = self._run()

        self.assertGreater(len(emitted_at), 1, "one line then silence reads as a hang")

    def test_no_gap_exceeds_the_cadence(self) -> None:
        emitted_at = self._run()
        gaps = [
            later - earlier
            for earlier, later in zip(emitted_at, emitted_at[1:], strict=False)
        ]

        self.assertTrue(gaps)
        self.assertLessEqual(max(gaps), MAX_SILENCE_SECONDS)

    def test_the_wait_is_named_in_the_line(self) -> None:
        # A repeated identical line is indistinguishable from a stuck one; the
        # elapsed time is what shows it is still counting.
        clock = _FakeClock()
        backend = SimpleNamespace(
            get_pod=lambda _pod_id: {
                "status": "PROVISIONING",
                "installationStatus": "INSTALLING",
                "sshConnection": None,
            }
        )
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator._prime_backend = backend  # type: ignore[attr-defined]
        lines: list[str] = []
        with (
            patch.object(orchestrator_module.time, "monotonic", clock.monotonic),
            patch.object(
                orchestrator_module,
                "shutdown_event",
                lambda: SimpleNamespace(wait=clock.sleeper),
            ),
        ):
            try:
                for event in orchestrator._await_prime_pod_active("pod-1", {}):
                    if isinstance(event, LogEvent):
                        lines.append(event.line)
            except PrimeApiError:
                pass

        self.assertTrue(any("waiting" in line for line in lines[1:]))
