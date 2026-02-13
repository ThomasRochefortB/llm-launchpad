from __future__ import annotations

import unittest
from unittest.mock import patch

from llm_launchpad.core.backend import ModalBackend, _extract_modal_app_rows
from llm_launchpad.core.orchestrator import Orchestrator
from llm_launchpad.protocol.enums import BackendType
from llm_launchpad.protocol.events import LogEvent


class BackendMultiInstanceTests(unittest.TestCase):
    def test_extract_modal_app_rows_infers_backend_for_multi_instance_names(self) -> None:
        payload = [
            {"name": "vllm-qwen3", "app_id": "ap-1", "state": "running"},
            {"name": "llamacpp-qwen2-5", "app_id": "ap-2", "state": "running"},
        ]
        rows = _extract_modal_app_rows(payload)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].backend, BackendType.VLLM)
        self.assertEqual(rows[0].instance_name, "qwen3")
        self.assertEqual(rows[1].backend, BackendType.LLAMACPP)
        self.assertEqual(rows[1].instance_name, "qwen2-5")

    def test_default_server_url_accepts_explicit_app_name(self) -> None:
        url = ModalBackend.default_server_url("alice", app_name="vllm-qwen3")
        self.assertEqual(url, "https://alice--vllm-qwen3-serve.modal.run")


class OrchestratorMultiInstanceTests(unittest.TestCase):
    def test_tail_logs_targets_explicit_app_name(self) -> None:
        orch = Orchestrator()
        captured: list[list[str]] = []

        def _fake_run_streaming(command: list[str], env=None):  # type: ignore[no-untyped-def]
            captured.append(command)
            if False:
                yield None
            return
            yield  # pragma: no cover

        with patch("llm_launchpad.core.backend.ModalBackend.logs_follow_args", return_value=[]):
            with patch("llm_launchpad.core.backend.ModalBackend.run_streaming", side_effect=_fake_run_streaming):
                list(orch.tail_logs(BackendType.VLLM, follow=False, app_name="vllm-qwen3"))
        self.assertTrue(captured)
        self.assertEqual(captured[0], ["modal", "app", "logs", "vllm-qwen3"])

    def test_stop_app_targets_explicit_app_name(self) -> None:
        orch = Orchestrator()
        captured: list[list[str]] = []

        def _fake_run_streaming(command: list[str], env=None):  # type: ignore[no-untyped-def]
            captured.append(command)
            if False:
                yield None
            return
            yield  # pragma: no cover

        with patch("llm_launchpad.core.backend.ModalBackend.run_streaming", side_effect=_fake_run_streaming):
            events = list(orch.stop_app(BackendType.VLLM, app_name="vllm-qwen2-5"))
        self.assertTrue(captured)
        self.assertEqual(captured[0], ["modal", "app", "stop", "vllm-qwen2-5"])
        self.assertTrue(any(isinstance(event, LogEvent) for event in events))


if __name__ == "__main__":
    unittest.main()

