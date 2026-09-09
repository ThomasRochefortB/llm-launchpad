"""Cover tail_logs for Prime and Vast providers (currently untested)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm_launchpad.core.orchestrator import Orchestrator
from llm_launchpad.protocol.enums import BackendType, ComputeProvider, DeploymentState, OperationType
from llm_launchpad.protocol.events import LogEvent, OperationCompleteEvent, StateChangeEvent


def _orchestrator(**backends: object) -> Orchestrator:
    orch = Orchestrator()
    for name, backend in backends.items():
        setattr(orch, f"_{name}", backend)
    return orch


class OrchestratorTailLogsProvidersTests(unittest.TestCase):
    def test_prime_logs_stream_without_follow(self) -> None:
        backend = SimpleNamespace(
            get_pod_logs=lambda pod_id, tail=500: ["line-1", "line-2"],
        )
        orch = _orchestrator(prime_backend=backend)

        events = list(
            orch.tail_logs(
                BackendType.VLLM,
                follow=False,
                app_id="pod-1",
                provider=ComputeProvider.PRIME,
            )
        )

        states = [e for e in events if isinstance(e, StateChangeEvent)]
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].current, DeploymentState.RUNNING)
        self.assertIn("pod-1", states[0].detail or "")
        lines = [e.line for e in events if isinstance(e, LogEvent)]
        self.assertEqual(lines, ["line-1", "line-2"])
        completion = next(e for e in events if isinstance(e, OperationCompleteEvent))
        self.assertTrue(completion.success)
        self.assertEqual(completion.operation, OperationType.LOGS)

    def test_prime_logs_require_pod_id(self) -> None:
        orch = _orchestrator(prime_backend=SimpleNamespace())
        events = list(
            orch.tail_logs(
                BackendType.VLLM,
                follow=False,
                app_id="",
                provider=ComputeProvider.PRIME,
            )
        )
        completion = next(e for e in events if isinstance(e, OperationCompleteEvent))
        self.assertFalse(completion.success)
        self.assertEqual(completion.exit_code, 2)
        self.assertIn("pod ID", completion.detail)

    def test_prime_logs_surface_backend_errors(self) -> None:
        def _raise(pod_id: str, tail: int = 500) -> list[str]:
            raise RuntimeError("prime api down")

        orch = _orchestrator(prime_backend=SimpleNamespace(get_pod_logs=_raise))
        events = list(
            orch.tail_logs(
                BackendType.VLLM,
                follow=False,
                app_id="pod-1",
                provider=ComputeProvider.PRIME,
            )
        )
        completion = next(e for e in events if isinstance(e, OperationCompleteEvent))
        self.assertFalse(completion.success)
        self.assertIn("prime api down", completion.detail)

    def test_prime_logs_follow_dedupes_overlap(self) -> None:
        batches = iter([["a", "b"], ["b", "c"], ["b", "c"]])
        backend = SimpleNamespace(get_pod_logs=lambda pod_id, tail=500: next(batches))
        orch = _orchestrator(prime_backend=backend)

        waits = iter([False, False, True])
        with patch("llm_launchpad.core.orchestrator.shutdown_event") as shutdown:
            shutdown.return_value.wait.side_effect = lambda **kwargs: next(waits)
            events = list(
                orch.tail_logs(
                    BackendType.VLLM,
                    follow=True,
                    app_id="pod-1",
                    provider=ComputeProvider.PRIME,
                )
            )

        lines = [e.line for e in events if isinstance(e, LogEvent)]
        self.assertEqual(lines, ["a", "b", "c"])

    def test_vast_logs_without_follow(self) -> None:
        backend = SimpleNamespace(logs=lambda instance_id: ["v-line-1"])
        orch = _orchestrator(vast_backend=backend)

        events = list(
            orch.tail_logs(
                BackendType.LLAMACPP,
                follow=False,
                app_id="inst-1",
                provider=ComputeProvider.VAST,
            )
        )
        lines = [e.line for e in events if isinstance(e, LogEvent)]
        self.assertEqual(lines, ["v-line-1"])
        completion = next(e for e in events if isinstance(e, OperationCompleteEvent))
        self.assertTrue(completion.success)

    def test_vast_logs_surface_backend_errors(self) -> None:
        def _raise(instance_id: str) -> list[str]:
            raise ValueError("instance gone")

        orch = _orchestrator(vast_backend=SimpleNamespace(logs=_raise))
        events = list(
            orch.tail_logs(
                BackendType.LLAMACPP,
                follow=False,
                app_id="inst-1",
                provider=ComputeProvider.VAST,
            )
        )
        completion = next(e for e in events if isinstance(e, OperationCompleteEvent))
        self.assertFalse(completion.success)
        self.assertIn("instance gone", completion.detail)

    def test_vast_logs_follow_emits_only_new_lines(self) -> None:
        batches = iter([["a", "b"], ["a", "b", "c"], ["a", "b", "c"]])
        orch = _orchestrator(vast_backend=SimpleNamespace(logs=lambda instance_id: next(batches)))

        waits = iter([False, False, True])
        with patch("llm_launchpad.core.orchestrator.shutdown_event") as shutdown:
            shutdown.return_value.wait.side_effect = lambda **kwargs: next(waits)
            events = list(
                orch.tail_logs(
                    BackendType.LLAMACPP,
                    follow=True,
                    app_id="inst-1",
                    provider=ComputeProvider.VAST,
                )
            )

        lines = [e.line for e in events if isinstance(e, LogEvent)]
        self.assertEqual(lines, ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
