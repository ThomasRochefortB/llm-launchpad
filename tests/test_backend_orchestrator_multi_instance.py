from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm_launchpad.core.backend import ModalBackend, _extract_modal_app_rows
from llm_launchpad.core.orchestrator import Orchestrator
from llm_launchpad.protocol.enums import BackendType
from llm_launchpad.protocol.events import LogEvent, OperationCompleteEvent
from llm_launchpad.protocol.models import DeploymentConfig, LaunchpadSettings


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

    def test_default_server_url_accepts_function_slug(self) -> None:
        url = ModalBackend.default_server_url(
            "alice",
            app_name="vllm-qwen3",
            function_slug="alpha-bravo",
        )
        self.assertEqual(url, "https://alice--vllm-qwen3-serve-alpha-bravo.modal.run")


class OrchestratorMultiInstanceTests(unittest.TestCase):
    def test_deploy_assigns_and_forwards_function_slug(self) -> None:
        orch = Orchestrator(config_store=SimpleNamespace(load=lambda: LaunchpadSettings()))
        captured_env: list[dict[str, str] | None] = []

        def _fake_run_streaming(command: list[str], env=None):  # type: ignore[no-untyped-def]
            captured_env.append(env)
            yield OperationCompleteEvent(success=True, exit_code=0)

        config = DeploymentConfig(
            backend=BackendType.VLLM,
            app_name="vllm-qwen3",
            do_deploy=True,
        )
        with patch("llm_launchpad.core.orchestrator.random_function_slug", return_value="alpha-bravo"):
            with patch("llm_launchpad.core.backend.ModalBackend.run_streaming", side_effect=_fake_run_streaming):
                list(orch.deploy(config))
        self.assertEqual(config.function_slug, "alpha-bravo")
        self.assertTrue(captured_env)
        self.assertEqual(captured_env[0]["MODAL_FUNCTION_SLUG"], "alpha-bravo")

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
