from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from llm_launchpad.cli import main as cli_main
from llm_launchpad.protocol.enums import BackendType, OperationType
from llm_launchpad.protocol.events import LogEvent, OperationCompleteEvent


class CliMainHelperTests(unittest.TestCase):
    def test_resolve_deploy_target_prefers_explicit_app_name(self) -> None:
        instance, app_name = cli_main._resolve_deploy_target(
            BackendType.VLLM,
            model_hint="Qwen/Qwen3-8B",
            instance_name=None,
            app_name="vllm-prod",
        )
        self.assertEqual(instance, "vllm-prod")
        self.assertEqual(app_name, "vllm-prod")

    def test_resolve_deploy_target_slugifies_explicit_instance(self) -> None:
        instance, app_name = cli_main._resolve_deploy_target(
            BackendType.VLLM,
            model_hint=None,
            instance_name="Qwen 3",
            app_name=None,
        )
        self.assertEqual(instance, "qwen-3")
        self.assertEqual(app_name, "vllm-qwen-3")

    def test_resolve_deploy_target_uses_auto_name_when_missing(self) -> None:
        instance, app_name = cli_main._resolve_deploy_target(
            BackendType.LLAMACPP,
            model_hint="my model",
            instance_name=None,
            app_name=None,
        )
        self.assertEqual(instance, "my-model")
        self.assertEqual(app_name, "llamacpp-my-model")

    def test_resolve_manage_app_name_prefers_explicit_app_name(self) -> None:
        resolved = cli_main._resolve_manage_app_name(
            BackendType.VLLM,
            app_name="vllm-override",
            instance_name="ignored",
        )
        self.assertEqual(resolved, "vllm-override")

    def test_resolve_manage_app_name_builds_from_instance_name(self) -> None:
        resolved = cli_main._resolve_manage_app_name(
            BackendType.VLLM,
            app_name=None,
            instance_name="Qwen 3",
        )
        self.assertEqual(resolved, "vllm-qwen-3")

    def test_resolve_manage_app_name_uses_single_match(self) -> None:
        row = SimpleNamespace(name="vllm-qwen3")
        with patch("llm_launchpad.cli.main._backend_instances", return_value=[row]):
            resolved = cli_main._resolve_manage_app_name(BackendType.VLLM, None, None)
        self.assertEqual(resolved, "vllm-qwen3")

    def test_resolve_manage_app_name_fails_when_multiple_matches(self) -> None:
        rows = [SimpleNamespace(name="vllm-a"), SimpleNamespace(name="vllm-b")]
        with patch("llm_launchpad.cli.main._backend_instances", return_value=rows):
            with self.assertRaises(cli_main.typer.Exit):
                cli_main._resolve_manage_app_name(BackendType.VLLM, None, None)

    def test_resolve_manage_app_name_falls_back_to_legacy(self) -> None:
        with patch("llm_launchpad.cli.main._backend_instances", return_value=[]):
            resolved = cli_main._resolve_manage_app_name(BackendType.LLAMACPP, None, None)
        self.assertEqual(resolved, "llamacpp-server")


class CliMainCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_deploy_success_prints_target(self) -> None:
        orch = SimpleNamespace(
            deploy=lambda _config: [
                LogEvent(line="deploying"),
                OperationCompleteEvent(operation=OperationType.DEPLOY, success=True, exit_code=0),
            ]
        )
        with patch("llm_launchpad.cli.main._preflight", return_value=(orch, "alice")):
            with patch("llm_launchpad.cli.main._print_banner", return_value=None):
                result = self.runner.invoke(
                    cli_main.app, ["deploy", "--backend", "vllm", "--model-name", "Qwen/Qwen3-8B"]
                )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Deploy target: backend=vllm", result.output)

    def test_deploy_failure_propagates_operation_exit_code(self) -> None:
        orch = SimpleNamespace(
            deploy=lambda _config: [
                OperationCompleteEvent(
                    operation=OperationType.DEPLOY,
                    success=False,
                    exit_code=9,
                    detail="boom",
                ),
            ]
        )
        with patch("llm_launchpad.cli.main._preflight", return_value=(orch, "alice")):
            with patch("llm_launchpad.cli.main._print_banner", return_value=None):
                result = self.runner.invoke(cli_main.app, ["deploy", "--backend", "vllm"])
        self.assertEqual(result.exit_code, 9)

    def test_deploy_vllm_maps_trust_remote_code_flag(self) -> None:
        captured = {}

        def _deploy(config):  # type: ignore[no-untyped-def]
            captured["config"] = config
            return [OperationCompleteEvent(operation=OperationType.DEPLOY, success=True, exit_code=0)]

        orch = SimpleNamespace(deploy=_deploy)
        with patch("llm_launchpad.cli.main._preflight", return_value=(orch, "alice")):
            with patch("llm_launchpad.cli.main._print_banner", return_value=None):
                result = self.runner.invoke(
                    cli_main.app,
                    [
                        "deploy",
                        "--backend",
                        "vllm",
                        "--model-name",
                        "MiniMaxAI/MiniMax-M2.5",
                        "--trust-remote-code",
                    ],
                )
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(captured["config"].trust_remote_code)

    def test_status_failure_returns_exit_code_1(self) -> None:
        orch = SimpleNamespace(
            check_status=lambda *_args, **_kwargs: [
                OperationCompleteEvent(operation=OperationType.STATUS, success=False, exit_code=3),
            ]
        )
        with patch("llm_launchpad.cli.main._preflight", return_value=(orch, "alice")):
            with patch("llm_launchpad.cli.main._print_banner", return_value=None):
                result = self.runner.invoke(
                    cli_main.app, ["status", "--backend", "vllm", "--app-name", "vllm-test"]
                )
        self.assertEqual(result.exit_code, 1)

    def test_status_uses_function_slug_for_default_url(self) -> None:
        calls: list[tuple[BackendType, str, int]] = []

        def _check_status(backend: BackendType, url: str, timeout: int):
            calls.append((backend, url, timeout))
            return [OperationCompleteEvent(operation=OperationType.STATUS, success=True, exit_code=0)]

        orch = SimpleNamespace(check_status=_check_status)
        with patch("llm_launchpad.cli.main._preflight", return_value=(orch, "alice")):
            with patch("llm_launchpad.cli.main._print_banner", return_value=None):
                result = self.runner.invoke(
                    cli_main.app,
                    [
                        "status",
                        "--backend",
                        "vllm",
                        "--app-name",
                        "vllm-test",
                        "--function-slug",
                        "alpha-bravo",
                    ],
                )
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(
            calls,
            [(BackendType.VLLM, "https://alice--vllm-test-serve-alpha-bravo.modal.run", 60)],
        )

    def test_warmup_uses_function_slug_for_default_url(self) -> None:
        calls: list[tuple[BackendType, str, int, bool, str | None]] = []

        def _warmup(
            backend: BackendType,
            url: str,
            timeout: int,
            tail_logs: bool,
            app_name: str | None = None,
        ):
            calls.append((backend, url, timeout, tail_logs, app_name))
            return []

        orch = SimpleNamespace(warmup=_warmup)
        with patch("llm_launchpad.cli.main._preflight", return_value=(orch, "alice")):
            with patch("llm_launchpad.cli.main._print_banner", return_value=None):
                result = self.runner.invoke(
                    cli_main.app,
                    [
                        "warmup",
                        "--backend",
                        "vllm",
                        "--app-name",
                        "vllm-test",
                        "--function-slug",
                        "alpha-bravo",
                    ],
                )
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(
            calls,
            [
                (
                    BackendType.VLLM,
                    "https://alice--vllm-test-serve-alpha-bravo.modal.run",
                    1800,
                    True,
                    "vllm-test",
                )
            ],
        )

    def test_logs_passes_follow_and_app_name(self) -> None:
        calls: list[tuple[BackendType, bool, str | None]] = []

        def _tail_logs(backend: BackendType, follow: bool, app_name: str | None = None):
            calls.append((backend, follow, app_name))
            return []

        orch = SimpleNamespace(tail_logs=_tail_logs)
        with patch("llm_launchpad.cli.main._preflight", return_value=(orch, "alice")):
            with patch("llm_launchpad.cli.main._print_banner", return_value=None):
                result = self.runner.invoke(
                    cli_main.app,
                    ["logs", "--backend", "vllm", "--no-follow", "--app-name", "vllm-qwen3"],
                )
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(calls, [(BackendType.VLLM, False, "vllm-qwen3")])

    def test_stop_without_yes_can_abort(self) -> None:
        orch = SimpleNamespace(stop_app=lambda *_args, **_kwargs: [])
        with patch("llm_launchpad.cli.main._preflight", return_value=(orch, "alice")):
            with patch("llm_launchpad.cli.main._print_banner", return_value=None):
                result = self.runner.invoke(
                    cli_main.app,
                    ["stop", "--backend", "vllm", "--app-name", "vllm-qwen3"],
                    input="n\n",
                )
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Aborted.", result.output)

    def test_stop_with_yes_calls_orchestrator(self) -> None:
        calls: list[tuple[BackendType, str | None]] = []

        def _stop_app(backend: BackendType, app_name: str | None = None):
            calls.append((backend, app_name))
            return []

        orch = SimpleNamespace(stop_app=_stop_app)
        with patch("llm_launchpad.cli.main._preflight", return_value=(orch, "alice")):
            with patch("llm_launchpad.cli.main._print_banner", return_value=None):
                result = self.runner.invoke(
                    cli_main.app,
                    ["stop", "--backend", "vllm", "--yes", "--app-name", "vllm-qwen3"],
                )
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(calls, [(BackendType.VLLM, "vllm-qwen3")])

    def test_switch_llamacpp_requires_preset_or_repo(self) -> None:
        orch = SimpleNamespace(deploy=lambda *_args, **_kwargs: [])
        with patch("llm_launchpad.cli.main._preflight", return_value=(orch, "alice")):
            with patch("llm_launchpad.cli.main._print_banner", return_value=None):
                result = self.runner.invoke(
                    cli_main.app,
                    ["switch", "--backend", "llamacpp", "--no-redeploy"],
                )
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Provide --preset or --repo-id to switch.", result.output)

    def test_switch_vllm_no_redeploy_exits_cleanly(self) -> None:
        orch = SimpleNamespace(deploy=lambda *_args, **_kwargs: [])
        with patch("llm_launchpad.cli.main._preflight", return_value=(orch, "alice")):
            with patch("llm_launchpad.cli.main._print_banner", return_value=None):
                result = self.runner.invoke(
                    cli_main.app,
                    ["switch", "--backend", "vllm", "--no-redeploy"],
                )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No deploy performed. Use --redeploy", result.output)


if __name__ == "__main__":
    unittest.main()
