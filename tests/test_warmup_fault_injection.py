"""Fault-injection tests for WarmupRunner, calibration, and lifecycle cleanup.

Simulates real cloud failure modes (blocking cold-start proxies, starved GPU
scheduling, cold first-request latency spikes, mid-load log stream drops) in
hermetic unit tests with zero network or GPU spend.
"""

from __future__ import annotations

import json
import threading
import types
import unittest
from unittest.mock import patch

from llm_launchpad.core.deployment_lifecycle import (
    LifecycleAttempt,
    LifecycleCallbacks,
    LifecycleOptions,
    run_lifecycle,
)
from llm_launchpad.core.warmup import (
    WarmupRunner,
    _calibrate_endpoint,
    _calibration_is_acceptable,
)
from llm_launchpad.protocol.enums import (
    AttemptDisposition,
    BackendType,
    CleanupDisposition,
    ComputeProvider,
    DeploymentState,
    OperationType,
    ServingObjective,
)
from llm_launchpad.protocol.events import (
    LogEvent,
    OperationCompleteEvent,
    StateChangeEvent,
)
from llm_launchpad.protocol.models import DeploymentConfig


class _Response:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text

    def json(self) -> object:
        return json.loads(self.text)


class WarmupFaultInjectionTests(unittest.TestCase):
    def test_slow_cold_start_proxy_streams_logs_then_becomes_ready(self) -> None:
        """A cold Modal probe blocks until the container boots, then transitions 503 -> 200."""
        first_probe_started = threading.Event()
        release_first_probe = threading.Event()
        self.addCleanup(release_first_probe.set)
        calls = {"n": 0}

        def _post(*_args: object, **_kwargs: object) -> _Response:
            calls["n"] += 1
            if calls["n"] == 1:
                first_probe_started.set()
                release_first_probe.wait(timeout=5.0)
                return _Response(503, "loading model")
            return _Response(200, '{"choices":[{"text":"ok"}]}')

        class _StreamingTail:
            def __init__(self, app_name: str) -> None:
                self.app_name = app_name
                self.seen_lines: set[str] = set()
                self._sent_boot = False

            def may_attach(self) -> bool:
                return False

            def attach(self):  # type: ignore[no-untyped-def]
                yield from ()

            def drain(self):  # type: ignore[no-untyped-def]
                if first_probe_started.is_set() and not self._sent_boot:
                    self._sent_boot = True
                    yield LogEvent(line="🦙 warming 5 volume file(s)", operation=OperationType.WARMUP)
                    release_first_probe.set()

            def stop(self, _reason: str) -> None:
                pass

        with (
            patch.dict("sys.modules", {"requests": types.SimpleNamespace(post=_post)}),
            patch("llm_launchpad.core.warmup._ModalLogTail", _StreamingTail),
            patch(
                "llm_launchpad.core.warmup.shutdown_event",
                return_value=types.SimpleNamespace(wait=lambda **_kwargs: False),
            ),
            patch("llm_launchpad.core.warmup.ModalBackend.test_curl_command", return_value="curl ok"),
        ):
            events = list(
                WarmupRunner().run(
                    backend=BackendType.LLAMACPP,
                    server_url="https://user--llp-lc-abc123456789.us-east.modal.run",
                    timeout=1800,
                    tail_logs=True,
                    app_name="llamacpp-test",
                )
            )

        lines = [e.line for e in events if isinstance(e, LogEvent)]
        self.assertIn("🦙 warming 5 volume file(s)", lines)
        self.assertIn("Server is ready!", lines)
        completion = next(e for e in events if isinstance(e, OperationCompleteEvent))
        self.assertTrue(completion.success)
        self.assertEqual(calls["n"], 2)

    def test_starved_scheduling_surfaces_system_hint_and_confirms_lifecycle_cleanup(self) -> None:
        """Timeout with read timeouts surfaces Modal system logs and tears down the app."""
        clock = {"t": 0.0}

        def _tick() -> float:
            clock["t"] += 25.0
            return clock["t"]

        def _always_timeout(*_args: object, **_kwargs: object) -> _Response:
            raise TimeoutError("Read timed out. (read timeout=60)")

        class _FakeOrchestrator:
            def __init__(self) -> None:
                self.stopped_apps: list[str | None] = []

            def deploy(self, _config: DeploymentConfig):  # type: ignore[no-untyped-def]
                yield LogEvent(
                    line="├── ✓ Created Function serve-test",
                    operation=OperationType.DEPLOY,
                )
                yield LogEvent(
                    line="│   └── https://user--llp-lc-111111111111.us-east.modal.run ",
                    operation=OperationType.DEPLOY,
                )
                yield OperationCompleteEvent(operation=OperationType.DEPLOY, success=True)

            def warmup(self, backend, url, timeout, tail_logs, **kwargs):  # type: ignore[no-untyped-def]
                return WarmupRunner().run(
                    backend=backend,
                    server_url=url,
                    timeout=timeout,
                    tail_logs=tail_logs,
                    **kwargs,
                )

            def stop_app(self, _backend, *, app_name=None, **_kwargs):  # type: ignore[no-untyped-def]
                self.stopped_apps.append(app_name)
                yield OperationCompleteEvent(operation=OperationType.STOP, success=True)

        orchestrator = _FakeOrchestrator()
        observed: list[object] = []
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.MODAL,
            app_name="llamacpp-starved-test",
            do_deploy=True,
            do_warmup=True,
        )

        with (
            patch.dict("sys.modules", {"requests": types.SimpleNamespace(post=_always_timeout)}),
            patch("llm_launchpad.core.warmup._ModalLogTail") as tail_cls,
            patch("llm_launchpad.core.warmup.time.time", side_effect=_tick),
            patch(
                "llm_launchpad.core.warmup.shutdown_event",
                return_value=types.SimpleNamespace(wait=lambda **_kwargs: False),
            ),
            patch(
                "llm_launchpad.core.warmup.ModalBackend.recent_system_logs",
                return_value=[
                    "Function 'serve-test' is waiting to be scheduled on a GPU_H200 worker."
                ],
            ),
        ):
            tail_cls.return_value.may_attach.return_value = False
            tail_cls.return_value.attach.return_value = iter(())
            tail_cls.return_value.drain.return_value = iter(())
            result = run_lifecycle(
                orchestrator,
                [LifecycleAttempt(config=config, retry_allowed=False)],
                options=LifecycleOptions(warmup_timeout_seconds=100, tail_logs=True),
                callbacks=LifecycleCallbacks(on_event=observed.append),
            )

        self.assertFalse(result.succeeded)
        self.assertEqual(orchestrator.stopped_apps, ["llamacpp-starved-test"])
        outcome = result.attempts[0]
        self.assertEqual(outcome.disposition, AttemptDisposition.FAILED)
        self.assertEqual(outcome.cleanup, CleanupDisposition.CONFIRMED)
        self.assertIn("Waiting for GPU scheduling (GPU_H200)", outcome.failure_detail or "")
        self.assertTrue(
            any(isinstance(e, StateChangeEvent) and e.current == DeploymentState.QUEUED for e in observed)
        )

    def test_cold_first_request_spike_is_absorbed_by_untimed_calibration_warmup(self) -> None:
        """A 4 tok/s cold first request must not fail certification when steady-state is fast."""
        call_index = {"n": 0}

        def _fake_request(_requests_module: object, **kwargs: object) -> dict[str, float]:
            call_index["n"] += 1
            if kwargs.get("prompt_seed") == -1:
                # Untimed cold warmup request hits cold mmap pages / CUDA graph compile.
                return {
                    "completion_tokens": 64.0,
                    "actual_prompt_tokens": 256.0,
                    "latency": 16.0,
                    "ttft": 1.0,
                    "prompt_tps": 256.0,
                    "output_tps": 4.0,
                }
            # Steady-state measured scenarios run at 28 tok/s.
            return {
                "completion_tokens": 128.0,
                "actual_prompt_tokens": float(kwargs["prompt_tokens"]),  # type: ignore[arg-type]
                "latency": 4.8,
                "ttft": 0.2,
                "prompt_tps": 2000.0,
                "output_tps": 28.0,
            }

        with patch(
            "llm_launchpad.core.warmup._streaming_calibration_request",
            side_effect=_fake_request,
        ):
            points = _calibrate_endpoint(
                object(),
                server_url="https://user--llp-lc-abc.us-east.modal.run",
                model="test-model",
                headers={},
                parallel_slots=2,
                price_per_hour_usd=18.0,
                budget_seconds=30.0,
            )

        accepted, detail = _calibration_is_acceptable(points, ServingObjective.GENERAL_PURPOSE)
        self.assertTrue(accepted, detail)
        self.assertGreater(call_index["n"], 1)
        self.assertTrue(all((p.output_tokens_per_second or 0.0) >= 28.0 for p in points))

    def test_dropped_live_stream_recovers_terminal_crash_from_historical_logs(self) -> None:
        """If `modal app logs -f` disconnects before a crash, historical fetch still fast-fails."""
        detached = {"flag": False}

        class _DroppingTail:
            def __init__(self, app_name: str) -> None:
                self.app_name = app_name
                self.seen_lines: set[str] = set()
                self.stopped: str | None = None

            def may_attach(self) -> bool:
                return detached["flag"]

            def attach(self):  # type: ignore[no-untyped-def]
                detached["flag"] = False
                yield from ()

            def drain(self):  # type: ignore[no-untyped-def]
                detached["flag"] = True
                yield from ()

            def stop(self, reason: str) -> None:
                self.stopped = reason

        crash_stdout = (
            "llama.cpp rejected the full-context GPU-only serving plan: "
            "need 4096 MiB less in total (fit exit 1).\n"
        )
        with (
            patch.dict(
                "sys.modules",
                {"requests": types.SimpleNamespace(post=lambda *_a, **_k: _Response(503, "restarting"))},
            ),
            patch("llm_launchpad.core.warmup._ModalLogTail", _DroppingTail),
            patch(
                "llm_launchpad.core.warmup.subprocess.run",
                return_value=types.SimpleNamespace(returncode=0, stdout=crash_stdout),
            ),
            patch(
                "llm_launchpad.core.warmup.shutdown_event",
                return_value=types.SimpleNamespace(wait=lambda **_kwargs: False),
            ),
            patch("llm_launchpad.core.warmup.time.time", side_effect=[float(i) for i in range(100)]),
        ):
            events = list(
                WarmupRunner().run(
                    backend=BackendType.LLAMACPP,
                    server_url="https://user--llp-lc-crash.us-east.modal.run",
                    timeout=1800,
                    tail_logs=True,
                    app_name="llamacpp-crash",
                )
            )

        completion = next(e for e in events if isinstance(e, OperationCompleteEvent))
        self.assertFalse(completion.success)
        self.assertIn("rejected the full-context", completion.detail)


if __name__ == "__main__":
    unittest.main()
