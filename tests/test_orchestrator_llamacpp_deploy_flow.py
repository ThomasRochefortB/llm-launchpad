from __future__ import annotations

import unittest
from unittest.mock import patch

from llm_launchpad.core.orchestrator import Orchestrator
from llm_launchpad.protocol.enums import BackendType, OperationType
from llm_launchpad.protocol.events import LogEvent, OperationCompleteEvent
from llm_launchpad.protocol.models import DeploymentConfig, LaunchpadSettings


class _FakeConfigStore:
    def load(self) -> LaunchpadSettings:
        return LaunchpadSettings()


class OrchestratorLlamaCppDeployFlowTests(unittest.TestCase):
    def test_llamacpp_deploy_splits_prepare_and_deploy_commands(self) -> None:
        seen_commands: list[list[str]] = []

        def _run_streaming(cmd, env=None):  # type: ignore[no-untyped-def]
            _ = env
            seen_commands.append(list(cmd))
            if cmd[:2] == ["modal", "run"]:
                return iter(
                    [
                        LogEvent(line="prep"),
                        OperationCompleteEvent(success=True, exit_code=0),
                    ]
                )
            if cmd[:2] == ["modal", "deploy"]:
                return iter(
                    [
                        LogEvent(line="deploy"),
                        OperationCompleteEvent(success=True, exit_code=0),
                    ]
                )
            return iter([])

        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            repo_id="Edge-Quant/Nanbeige4.1-3B-Q4_K_M-GGUF",
            quant="Q4_K_M",
            preload=True,
            do_deploy=True,
            do_warmup=False,
            app_name="llamacpp-test",
            function_slug="slug-1",
        )

        with patch("llm_launchpad.core.orchestrator.ModalBackend.run_streaming", side_effect=_run_streaming):
            events = list(Orchestrator(config_store=_FakeConfigStore()).deploy(config))

        self.assertEqual(len(seen_commands), 2)
        prep_cmd, deploy_cmd = seen_commands

        self.assertEqual(prep_cmd[:3], ["modal", "run", "llm_launchpad/backends/modal_llamacpp_app.py::main"])
        self.assertIn("--preload", prep_cmd)
        self.assertNotIn("--deploy", prep_cmd)

        self.assertEqual(deploy_cmd[:2], ["modal", "deploy"])
        self.assertIn("llm_launchpad/backends/modal_llamacpp_app.py", deploy_cmd)
        self.assertIn("--name", deploy_cmd)
        self.assertIn("llamacpp-test", deploy_cmd)

        completions = [e for e in events if isinstance(e, OperationCompleteEvent)]
        self.assertEqual(len(completions), 1)
        self.assertEqual(completions[0].operation, OperationType.DEPLOY)
        self.assertTrue(completions[0].success)

    def test_llamacpp_deploy_does_not_run_modal_deploy_when_prepare_fails(self) -> None:
        seen_commands: list[list[str]] = []

        def _run_streaming(cmd, env=None):  # type: ignore[no-untyped-def]
            _ = env
            seen_commands.append(list(cmd))
            return iter(
                [
                    LogEvent(line="prep failed"),
                    OperationCompleteEvent(success=False, exit_code=7, detail="prep failed"),
                ]
            )

        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            repo_id="repo/model",
            quant="Q4_K_M",
            preload=True,
            do_deploy=True,
            do_warmup=False,
            app_name="llamacpp-test",
            function_slug="slug-1",
        )

        with patch("llm_launchpad.core.orchestrator.ModalBackend.run_streaming", side_effect=_run_streaming):
            events = list(Orchestrator(config_store=_FakeConfigStore()).deploy(config))

        self.assertEqual(len(seen_commands), 1)
        self.assertEqual(seen_commands[0][:2], ["modal", "run"])

        completions = [e for e in events if isinstance(e, OperationCompleteEvent)]
        self.assertEqual(len(completions), 1)
        self.assertFalse(completions[0].success)
        self.assertEqual(completions[0].exit_code, 7)


if __name__ == "__main__":
    unittest.main()
