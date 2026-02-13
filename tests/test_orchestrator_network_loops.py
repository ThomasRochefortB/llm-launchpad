from __future__ import annotations

import types
import unittest
from unittest.mock import patch

from llm_launchpad.core.orchestrator import Orchestrator
from llm_launchpad.protocol.enums import BackendType, DeploymentState, OperationType
from llm_launchpad.protocol.events import ErrorEvent, LogEvent, OperationCompleteEvent, StateChangeEvent


class _Response:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class _FakeStdout:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def readline(self) -> str:
        if not self._lines:
            return ""
        return self._lines.pop(0)


class _FakeProc:
    def __init__(self, lines: list[str]) -> None:
        self.stdout = _FakeStdout(lines)
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True


class OrchestratorNetworkLoopTests(unittest.TestCase):
    def test_warmup_success_vllm(self) -> None:
        fake_requests = types.SimpleNamespace(get=lambda *_args, **_kwargs: _Response(200, "ok"))
        with patch.dict("sys.modules", {"requests": fake_requests}):
            with patch("llm_launchpad.core.orchestrator.time.sleep", return_value=None):
                with patch("llm_launchpad.core.orchestrator.ModalBackend.test_curl_command", return_value="curl ok"):
                    events = list(
                        Orchestrator().warmup(
                            backend=BackendType.VLLM,
                            server_url="https://example.modal.run",
                            timeout=10,
                            tail_logs=False,
                        )
                    )

        self.assertTrue(any(isinstance(e, StateChangeEvent) and e.current == DeploymentState.HEALTHY for e in events))
        self.assertTrue(
            any(
                isinstance(e, OperationCompleteEvent)
                and e.operation == OperationType.WARMUP
                and e.success
                for e in events
            )
        )

    def test_warmup_timeout_after_non_200_responses(self) -> None:
        fake_requests = types.SimpleNamespace(get=lambda *_args, **_kwargs: _Response(503, "not ready"))
        with patch.dict("sys.modules", {"requests": fake_requests}):
            with patch("llm_launchpad.core.orchestrator.time.time", side_effect=[0.0, 0.1, 999.0]):
                with patch("llm_launchpad.core.orchestrator.time.sleep", return_value=None):
                    events = list(
                        Orchestrator().warmup(
                            backend=BackendType.VLLM,
                            server_url="https://example.modal.run",
                            timeout=1,
                            tail_logs=False,
                        )
                    )

        self.assertTrue(any(isinstance(e, ErrorEvent) and "Timed out after 1s" in e.message for e in events))
        self.assertTrue(
            any(
                isinstance(e, OperationCompleteEvent)
                and e.operation == OperationType.WARMUP
                and not e.success
                for e in events
            )
        )

    def test_warmup_with_log_tailing_drains_modal_logs(self) -> None:
        fake_proc = _FakeProc(lines=["modal-log-line\n"])
        fake_requests = types.SimpleNamespace(get=lambda *_args, **_kwargs: _Response(200, "ok"))
        fake_select = types.SimpleNamespace(
            select=lambda *_args, **_kwargs: ([fake_proc.stdout], [], [])
        )
        with patch.dict("sys.modules", {"requests": fake_requests, "select": fake_select}):
            with patch("llm_launchpad.core.orchestrator.ModalBackend.logs_follow_args", return_value=["--follow"]):
                with patch("llm_launchpad.core.orchestrator.subprocess.Popen", return_value=fake_proc):
                    with patch(
                        "llm_launchpad.core.orchestrator.ModalBackend.test_curl_command",
                        return_value="curl ok",
                    ):
                        events = list(
                            Orchestrator().warmup(
                                backend=BackendType.VLLM,
                                server_url="https://example.modal.run",
                                timeout=10,
                                tail_logs=True,
                                app_name="vllm-test",
                            )
                        )

        self.assertTrue(
            any(
                isinstance(e, LogEvent)
                and e.line == "modal-log-line"
                and e.operation == OperationType.WARMUP
                for e in events
            )
        )
        self.assertTrue(fake_proc.terminated)

    def test_check_status_success(self) -> None:
        fake_requests = types.SimpleNamespace(get=lambda *_args, **_kwargs: _Response(200, "ok"))
        with patch.dict("sys.modules", {"requests": fake_requests}):
            with patch("llm_launchpad.core.orchestrator.ModalBackend.test_curl_command", return_value="curl ok"):
                events = list(
                    Orchestrator().check_status(
                        backend=BackendType.VLLM,
                        server_url="https://example.modal.run",
                        timeout=5,
                    )
                )
        self.assertTrue(
            any(
                isinstance(e, OperationCompleteEvent)
                and e.operation == OperationType.STATUS
                and e.success
                for e in events
            )
        )

    def test_check_status_timeout(self) -> None:
        fake_requests = types.SimpleNamespace(get=lambda *_args, **_kwargs: _Response(503, "nope"))
        with patch.dict("sys.modules", {"requests": fake_requests}):
            with patch("llm_launchpad.core.orchestrator.time.time", side_effect=[0.0, 0.1, 999.0]):
                with patch("llm_launchpad.core.orchestrator.time.sleep", return_value=None):
                    events = list(
                        Orchestrator().check_status(
                            backend=BackendType.VLLM,
                            server_url="https://example.modal.run",
                            timeout=1,
                        )
                    )

        self.assertTrue(any(isinstance(e, ErrorEvent) and "Unhealthy (timed out)" in e.message for e in events))
        self.assertTrue(
            any(
                isinstance(e, OperationCompleteEvent)
                and e.operation == OperationType.STATUS
                and not e.success
                for e in events
            )
        )

    def test_check_status_error_flow_with_request_exception(self) -> None:
        def _raise(*_args, **_kwargs):
            raise RuntimeError("network down")

        fake_requests = types.SimpleNamespace(get=_raise)
        with patch.dict("sys.modules", {"requests": fake_requests}):
            with patch("llm_launchpad.core.orchestrator.time.time", side_effect=[0.0, 0.1, 999.0]):
                with patch("llm_launchpad.core.orchestrator.time.sleep", return_value=None):
                    events = list(
                        Orchestrator().check_status(
                            backend=BackendType.VLLM,
                            server_url="https://example.modal.run",
                            timeout=1,
                        )
                    )

        self.assertTrue(any(isinstance(e, ErrorEvent) and "network down" in e.message for e in events))


if __name__ == "__main__":
    unittest.main()
