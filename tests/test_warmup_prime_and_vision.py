"""Cover WarmupRunner's Prime pod-log path and vision-verified warmup persistence."""

from __future__ import annotations

import types
import unittest
from unittest.mock import patch

from llm_launchpad.core.orchestrator import Orchestrator
from llm_launchpad.protocol.enums import BackendType, ComputeProvider, OperationType
from llm_launchpad.protocol.events import ErrorEvent, LogEvent, OperationCompleteEvent
from llm_launchpad.protocol.models import VisionCapabilities, VisionVerification


class _Response:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


def _hermetic_warmup(test: unittest.TestCase, fake_requests: object):  # type: ignore[no-untyped-def]
    stack = test._warmup_stack  # type: ignore[attr-defined]
    stack.enter_context(patch.dict("sys.modules", {"requests": fake_requests}))
    stack.enter_context(
        patch("llm_launchpad.core.warmup.shutdown_event",
              return_value=types.SimpleNamespace(wait=lambda **_kwargs: False))
    )
    stack.enter_context(
        patch("llm_launchpad.core.warmup.ModalBackend.test_curl_command", return_value="curl ok")
    )


class WarmupPrimeLogsTests(unittest.TestCase):
    def setUp(self) -> None:
        import contextlib

        self._warmup_stack = contextlib.ExitStack()
        self.addCleanup(self._warmup_stack.close)

    def test_prime_warmup_streams_pod_logs_during_probing(self) -> None:
        responses = iter([_Response(503, "warming"), _Response(200, "ok")])
        fake_requests = types.SimpleNamespace(get=lambda *_a, **_k: next(responses))
        _hermetic_warmup(self, fake_requests)
        prime_backend = types.SimpleNamespace(get_pod_logs=lambda pod_id, tail=200: ["pod booting"])
        clock = {"t": 0.0}

        def _tick() -> float:
            clock["t"] += 0.5
            return clock["t"]

        self._warmup_stack.enter_context(
            patch("llm_launchpad.core.warmup.time.time", side_effect=_tick)
        )

        events = list(
            Orchestrator(prime_backend=prime_backend).warmup(
                backend=BackendType.VLLM,
                server_url="https://tunnel.example.com",
                timeout=10,
                tail_logs=True,
                provider=ComputeProvider.PRIME,
                pod_id="pod-1",
            )
        )

        lines = [e.line for e in events if isinstance(e, LogEvent)]
        self.assertIn("pod booting", lines)
        self.assertTrue(any(
            isinstance(e, OperationCompleteEvent) and e.success for e in events
        ))

    def test_prime_warmup_tolerates_log_fetch_errors(self) -> None:
        fake_requests = types.SimpleNamespace(get=lambda *_a, **_k: _Response(200, "ok"))
        _hermetic_warmup(self, fake_requests)

        def _raise(pod_id: str, tail: int = 200) -> list[str]:
            raise RuntimeError("logs unavailable")

        prime_backend = types.SimpleNamespace(get_pod_logs=_raise)
        events = list(
            Orchestrator(prime_backend=prime_backend).warmup(
                backend=BackendType.VLLM,
                server_url="https://tunnel.example.com",
                timeout=10,
                tail_logs=True,
                provider=ComputeProvider.PRIME,
                pod_id="pod-1",
            )
        )
        self.assertTrue(any(
            isinstance(e, OperationCompleteEvent) and e.success for e in events
        ))

    def test_prime_warmup_without_pod_id_skips_log_fetch(self) -> None:
        fake_requests = types.SimpleNamespace(get=lambda *_a, **_k: _Response(200, "ok"))
        _hermetic_warmup(self, fake_requests)

        def _fail(pod_id: str, tail: int = 200) -> list[str]:
            raise AssertionError("no pod id, must not fetch logs")

        prime_backend = types.SimpleNamespace(get_pod_logs=_fail)
        events = list(
            Orchestrator(prime_backend=prime_backend).warmup(
                backend=BackendType.VLLM,
                server_url="https://tunnel.example.com",
                timeout=10,
                tail_logs=True,
                provider=ComputeProvider.PRIME,
                pod_id=None,
            )
        )
        self.assertTrue(any(
            isinstance(e, OperationCompleteEvent) and e.success for e in events
        ))

    def test_vision_verified_warmup_persists_verification(self) -> None:
        fake_requests = types.SimpleNamespace(
            post=lambda *_a, **_k: _Response(200, '{"choices":[{"text":"ok"}]}')
        )
        _hermetic_warmup(self, fake_requests)
        vision = VisionCapabilities(enabled=True, verification=VisionVerification.PASSED)

        with patch(
            "llm_launchpad.core.warmup.verify_image_request", return_value=None
        ), patch(
            "llm_launchpad.core.connection_store.update_vision_verification"
        ) as persist:
            events = list(
                Orchestrator().warmup(
                    backend=BackendType.LLAMACPP,
                    server_url="https://example.modal.run",
                    timeout=10,
                    tail_logs=False,
                    app_name="llamacpp-test",
                    vision=vision,
                )
            )

        self.assertTrue(any(
            isinstance(e, OperationCompleteEvent) and e.success for e in events
        ))
        persist.assert_called_once_with("llamacpp-test", "https://example.modal.run", vision)
        lines = [e.line for e in events if isinstance(e, LogEvent)]
        self.assertIn("Image request verified (not a visual accuracy benchmark).", lines)

    def test_vision_failed_probe_marks_completion_data(self) -> None:
        from llm_launchpad.core.vision_probe import VISION_PROBE_FAILED

        fake_requests = types.SimpleNamespace(
            post=lambda *_a, **_k: _Response(200, '{"choices":[{"text":"ok"}]}')
        )
        _hermetic_warmup(self, fake_requests)
        vision = VisionCapabilities(enabled=True, verification=VisionVerification.PASSED)

        with patch(
            "llm_launchpad.core.warmup.verify_image_request",
            side_effect=RuntimeError("bad pixels"),
        ):
            events = list(
                Orchestrator().warmup(
                    backend=BackendType.LLAMACPP,
                    server_url="https://example.modal.run",
                    timeout=10,
                    tail_logs=False,
                    app_name="llamacpp-test",
                    vision=vision,
                )
            )

        completion = next(e for e in events if isinstance(e, OperationCompleteEvent))
        self.assertFalse(completion.success)
        self.assertEqual(completion.operation, OperationType.WARMUP)
        self.assertEqual(completion.data, {VISION_PROBE_FAILED: True})

    def test_warmup_loads_vision_from_connection_store(self) -> None:
        fake_requests = types.SimpleNamespace(
            post=lambda *_a, **_k: _Response(200, '{"choices":[{"text":"ok"}]}')
        )
        _hermetic_warmup(self, fake_requests)
        stored = VisionCapabilities(enabled=True, verification=VisionVerification.PASSED)

        with patch(
            "llm_launchpad.core.connection_store.load_connection_entries",
            return_value={"llamacpp-test": {
                "base_url": "https://example.modal.run/v1",
                "vision": {"enabled": True},
            }},
        ), patch(
            "llm_launchpad.core.vision.vision_from_dict", return_value=stored
        ) as from_dict, patch(
            "llm_launchpad.core.warmup.verify_image_request", return_value=None
        ), patch(
            "llm_launchpad.core.connection_store.update_vision_verification"
        ) as persist:
            events = list(
                Orchestrator().warmup(
                    backend=BackendType.LLAMACPP,
                    server_url="https://example.modal.run",
                    timeout=10,
                    tail_logs=False,
                    app_name="llamacpp-test",
                )
            )

        from_dict.assert_called_once()
        persist.assert_called_once()
        self.assertTrue(any(
            isinstance(e, OperationCompleteEvent) and e.success for e in events
        ))


if __name__ == "__main__":
    unittest.main()


class WarmupTerminalFailureTests(unittest.TestCase):
    """A crash that every container restart reproduces must not be waited out."""

    def setUp(self) -> None:
        import contextlib

        self._warmup_stack = contextlib.ExitStack()
        self.addCleanup(self._warmup_stack.close)

    def test_rejected_serving_plan_ends_warmup_without_waiting_for_the_timeout(self) -> None:
        crash = (
            "Runner failed with exception: RuntimeError('llama.cpp rejected the "
            "full-context GPU-only serving plan: the model, its full-context cache "
            "and the compute graph do not fit this placement's GPUs with the "
            "requested margin (fit exit 1).')"
        )

        class _Tail:
            """A Modal log stream that has already seen the container die."""

            def __init__(self, app_name: str) -> None:
                self.app_name = app_name
                self.seen_lines: set[str] = set()
                self.stopped: str | None = None
                self._sent = False

            def may_attach(self) -> bool:
                return False

            def attach(self):  # type: ignore[no-untyped-def]
                yield from ()

            def drain(self):  # type: ignore[no-untyped-def]
                if not self._sent:
                    self._sent = True
                    yield LogEvent(line=crash, operation=OperationType.WARMUP)

            def stop(self, reason: str) -> None:
                self.stopped = reason

        fake_requests = types.SimpleNamespace(
            post=lambda *_a, **_k: _Response(503, "still warming"),
            get=lambda *_a, **_k: _Response(503, "still warming"),
        )
        _hermetic_warmup(self, fake_requests)
        self._warmup_stack.enter_context(
            patch("llm_launchpad.core.warmup._ModalLogTail", _Tail)
        )

        events = list(
            Orchestrator().warmup(
                backend=BackendType.LLAMACPP,
                server_url="https://example.modal.run",
                timeout=1800,
                tail_logs=True,
                app_name="llp-lc-test",
            )
        )

        completion = [e for e in events if isinstance(e, OperationCompleteEvent)]
        self.assertEqual(len(completion), 1)
        self.assertFalse(completion[0].success)
        # The failure is reported as llama.cpp's verdict, not as a 30-minute
        # readiness timeout that says nothing about why.
        self.assertTrue(
            completion[0].detail == ""
            or "rejected the full-context" in completion[0].detail
        )
        errors = [e for e in events if isinstance(e, ErrorEvent)]
        self.assertTrue(errors[0].message.startswith("llama.cpp rejected the full-context"))
        self.assertFalse(errors[0].recoverable)
