from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm_launchpad.core.backend import ModalBackend, ModalListAppsResult, _extract_modal_app_rows
from llm_launchpad.core.orchestrator import Orchestrator
from llm_launchpad.protocol.enums import BackendType, OperationType
from llm_launchpad.protocol.events import ErrorEvent, LogEvent, OperationCompleteEvent
from llm_launchpad.protocol.models import DeploymentConfig, EndpointInfo, LaunchpadSettings


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

    def test_extract_modal_web_url_parses_deploy_line(self) -> None:
        line = (
            "└── 🔨 Created web function serve-discerning-tapir => "
            "https://alice--vllm-minimax-m2-5-serve-disc-42c728.modal.run (label truncated)"
        )
        parsed = ModalBackend.extract_modal_web_url(line)
        self.assertEqual(parsed, "https://alice--vllm-minimax-m2-5-serve-disc-42c728.modal.run")

    def test_default_llamacpp_url_uses_dns_safe_custom_label(self) -> None:
        url = ModalBackend.default_server_url(
            "thomasrochefortb",
            app_name="llamacpp-glm-5-3-flash-q2xl-cheap",
            function_slug="adaptable-cockatrice",
        )

        self.assertEqual(
            url,
            "https://thomasrochefortb--llp-lc-19d04acfa30d.modal.run",
        )
        dns_label = url.removeprefix("https://").split(".", 1)[0]
        self.assertLessEqual(len(dns_label), 63)


class OrchestratorMultiInstanceTests(unittest.TestCase):
    def test_deploy_assigns_and_forwards_function_slug_for_non_vllm(self) -> None:
        orch = Orchestrator(config_store=SimpleNamespace(load=lambda: LaunchpadSettings()))
        captured_env: list[dict[str, str] | None] = []

        def _fake_run_streaming(command: list[str], env=None):  # type: ignore[no-untyped-def]
            captured_env.append(env)
            yield OperationCompleteEvent(success=True, exit_code=0)

        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            app_name="llamacpp-qwen3",
            do_deploy=True,
        )
        with patch("llm_launchpad.core.orchestrator.random_function_slug", return_value="alpha-bravo"):
            with patch("llm_launchpad.core.backend.ModalBackend.run_streaming", side_effect=_fake_run_streaming):
                list(orch.deploy(config))
        self.assertEqual(config.function_slug, "alpha-bravo")
        self.assertTrue(captured_env)
        self.assertEqual(captured_env[0]["MODAL_FUNCTION_SLUG"], "alpha-bravo")

    def test_deploy_does_not_auto_assign_function_slug_for_vllm(self) -> None:
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
        self.assertIsNone(config.function_slug)
        self.assertTrue(captured_env)
        self.assertNotIn("MODAL_FUNCTION_SLUG", captured_env[0] or {})

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

    def test_tail_logs_prefers_explicit_app_id(self) -> None:
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
                list(orch.tail_logs(BackendType.VLLM, follow=False, app_name="vllm-qwen3", app_id="ap-123"))
        self.assertTrue(captured)
        self.assertEqual(captured[0], ["modal", "app", "logs", "ap-123"])

    def test_tail_logs_tags_generic_subprocess_events_as_logs(self) -> None:
        stream = iter(
            [
                LogEvent(line="line"),
                OperationCompleteEvent(success=True, exit_code=0),
            ]
        )
        with (
            patch("llm_launchpad.core.backend.ModalBackend.logs_follow_args", return_value=[]),
            patch("llm_launchpad.core.backend.ModalBackend.run_streaming", return_value=stream),
        ):
            events = list(
                Orchestrator().tail_logs(
                    BackendType.VLLM,
                    follow=False,
                    app_name="vllm-qwen3",
                )
            )

        subprocess_events = [
            event for event in events if isinstance(event, (LogEvent, OperationCompleteEvent))
        ]
        self.assertTrue(subprocess_events)
        self.assertTrue(
            all(event.operation == OperationType.LOGS for event in subprocess_events)
        )

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
        self.assertEqual(captured[0], ["modal", "app", "stop", "--yes", "vllm-qwen2-5"])
        self.assertTrue(any(isinstance(event, LogEvent) for event in events))

    def test_stop_app_prefers_explicit_app_id(self) -> None:
        orch = Orchestrator()
        captured: list[list[str]] = []

        def _fake_run_streaming(command: list[str], env=None):  # type: ignore[no-untyped-def]
            captured.append(command)
            if False:
                yield None
            return
            yield  # pragma: no cover

        with patch("llm_launchpad.core.backend.ModalBackend.run_streaming", side_effect=_fake_run_streaming):
            list(orch.stop_app(BackendType.VLLM, app_name="vllm-qwen2-5", app_id="ap-123"))
        self.assertTrue(captured)
        self.assertEqual(captured[0], ["modal", "app", "stop", "--yes", "ap-123"])

    def test_stop_app_tags_generic_subprocess_events_as_stop(self) -> None:
        stream = iter(
            [
                LogEvent(line="stopping"),
                ErrorEvent(message="failed", exit_code=9),
                OperationCompleteEvent(success=False, exit_code=9),
            ]
        )
        with patch(
            "llm_launchpad.core.backend.ModalBackend.run_streaming",
            return_value=stream,
        ):
            events = list(
                Orchestrator().stop_app(
                    BackendType.VLLM,
                    app_name="vllm-qwen3",
                )
            )

        subprocess_events = [
            event
            for event in events
            if isinstance(event, (LogEvent, ErrorEvent, OperationCompleteEvent))
        ]
        self.assertTrue(subprocess_events)
        self.assertTrue(
            all(event.operation == OperationType.STOP for event in subprocess_events)
        )

    def test_list_deployments_handles_empty_json_as_success(self) -> None:
        orch = Orchestrator()
        with patch(
            "llm_launchpad.core.backend.ModalBackend.list_apps_result",
            return_value=ModalListAppsResult(rows=[]),
        ):
            with patch("llm_launchpad.core.backend.ModalBackend.list_apps_raw_result") as raw_mock:
                events = list(orch.list_deployments())
        raw_mock.assert_not_called()
        self.assertTrue(
            any(isinstance(event, LogEvent) and event.line == "No launchpad deployments found." for event in events)
        )
        self.assertTrue(
            any(
                isinstance(event, OperationCompleteEvent)
                and event.success
                and isinstance(event.data, list)
                and len(event.data) == 0
                for event in events
            )
        )

    def test_list_deployments_hides_historical_duplicates_for_same_app(self) -> None:
        orch = Orchestrator()
        rows = [
            EndpointInfo(
                name="llamacpp-edge-quant-nanbeige4",
                app_id="ap-live",
                state="deployed",
                backend=BackendType.LLAMACPP,
                instance_name="edge-quant-nanbeige4",
            ),
            EndpointInfo(
                name="llamacpp-edge-quant-nanbeige4",
                app_id="ap-old-1",
                state="stopped",
                backend=BackendType.LLAMACPP,
                instance_name="edge-quant-nanbeige4",
            ),
            EndpointInfo(
                name="llamacpp-edge-quant-nanbeige4",
                app_id="ap-old-2",
                state="stopped",
                backend=BackendType.LLAMACPP,
                instance_name="edge-quant-nanbeige4",
            ),
            EndpointInfo(
                name="llamacpp-server",
                app_id="ap-default",
                state="stopped",
                backend=BackendType.LLAMACPP,
                instance_name="default",
            ),
        ]

        with patch(
            "llm_launchpad.core.backend.ModalBackend.list_apps_result",
            return_value=ModalListAppsResult(rows=rows),
        ):
            events = list(orch.list_deployments())

        log_lines = [event.line for event in events if isinstance(event, LogEvent)]
        deployment_lines = [line for line in log_lines if line.startswith("  backend=")]
        self.assertEqual(len(deployment_lines), 2)
        self.assertTrue(any("ap-live" in line and "state=deployed" in line for line in deployment_lines))
        self.assertFalse(any("ap-old-1" in line for line in deployment_lines))
        self.assertFalse(any("ap-old-2" in line for line in deployment_lines))
        self.assertTrue(any("hidden 2 historical duplicate app rows" in line for line in log_lines))

        done = next(
            event
            for event in events
            if isinstance(event, OperationCompleteEvent) and event.operation.value == "list"
        )
        self.assertTrue(done.success)
        self.assertIsInstance(done.data, list)
        self.assertEqual(len(done.data), 2)

    def test_list_deployments_keeps_concurrent_active_duplicates(self) -> None:
        orch = Orchestrator()
        rows = [
            EndpointInfo(
                name="llamacpp-glm5-rtxpro",
                app_id="ap-first",
                state="ephemeral",
                backend=BackendType.LLAMACPP,
                instance_name="glm5-rtxpro",
            ),
            EndpointInfo(
                name="llamacpp-glm5-rtxpro",
                app_id="ap-second",
                state="ephemeral",
                backend=BackendType.LLAMACPP,
                instance_name="glm5-rtxpro",
            ),
            EndpointInfo(
                name="llamacpp-glm5-rtxpro",
                app_id="ap-old",
                state="stopped",
                backend=BackendType.LLAMACPP,
                instance_name="glm5-rtxpro",
            ),
        ]

        with patch(
            "llm_launchpad.core.backend.ModalBackend.list_apps_result",
            return_value=ModalListAppsResult(rows=rows),
        ):
            events = list(orch.list_deployments())

        log_lines = [event.line for event in events if isinstance(event, LogEvent)]
        deployment_lines = [line for line in log_lines if line.startswith("  backend=")]
        self.assertEqual(len(deployment_lines), 2)
        self.assertTrue(any("ap-first" in line and "state=ephemeral" in line for line in deployment_lines))
        self.assertTrue(any("ap-second" in line and "state=ephemeral" in line for line in deployment_lines))
        self.assertFalse(any("ap-old" in line for line in deployment_lines))


if __name__ == "__main__":
    unittest.main()
