from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from llm_launchpad.cli import main as cli_main
from llm_launchpad.protocol.enums import BackendType, OperationType
from llm_launchpad.protocol.events import LogEvent, OperationCompleteEvent


class CliMainHelperTests(unittest.TestCase):
    def test_parse_bool_env_accepts_truthy_values(self) -> None:
        with patch.dict("os.environ", {"LLM_LAUNCHPAD_TUI_MOUSE": "yes"}):
            self.assertTrue(cli_main._parse_bool_env("LLM_LAUNCHPAD_TUI_MOUSE", False))

    def test_default_tui_mouse_enabled_defaults_off_over_ssh(self) -> None:
        with patch.dict("os.environ", {"SSH_CONNECTION": "1"}, clear=True):
            self.assertFalse(cli_main._default_tui_mouse_enabled())

    def test_default_tui_mouse_enabled_can_be_forced_on_by_env(self) -> None:
        with patch.dict(
            "os.environ",
            {"SSH_CONNECTION": "1", "LLM_LAUNCHPAD_TUI_MOUSE": "true"},
            clear=True,
        ):
            self.assertTrue(cli_main._default_tui_mouse_enabled())

    def test_default_tui_mouse_enabled_can_be_forced_off_by_env(self) -> None:
        with patch.dict("os.environ", {"LLM_LAUNCHPAD_TUI_MOUSE": "false"}, clear=True):
            self.assertFalse(cli_main._default_tui_mouse_enabled())

    def test_default_tui_mouse_enabled_stays_on_for_known_remote_clipboard_terminal(self) -> None:
        with patch.dict(
            "os.environ",
            {"SSH_CONNECTION": "1", "LC_TERMINAL": "iTerm2"},
            clear=True,
        ):
            self.assertTrue(cli_main._default_tui_mouse_enabled())

    def test_ensure_tui_runtime_requires_tty(self) -> None:
        with (
            patch("llm_launchpad.cli.main.sys.stdin.isatty", return_value=False),
            patch("llm_launchpad.cli.main.sys.stdout.isatty", return_value=True),
        ):
            with self.assertRaises(cli_main.typer.Exit) as ctx:
                cli_main._ensure_tui_runtime()
        self.assertEqual(ctx.exception.exit_code, 1)

    def test_ensure_tui_runtime_requires_modal_cli(self) -> None:
        with (
            patch("llm_launchpad.cli.main.sys.stdin.isatty", return_value=True),
            patch("llm_launchpad.cli.main.sys.stdout.isatty", return_value=True),
            patch("llm_launchpad.cli.main.ModalBackend.is_cli_available", return_value=False),
        ):
            with self.assertRaises(cli_main.typer.Exit) as ctx:
                cli_main._ensure_tui_runtime()
        self.assertEqual(ctx.exception.exit_code, 1)

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

    def test_main_defaults_to_tui_subcommand_when_no_args(self) -> None:
        fake_app = Mock()
        argv = ["llm-launchpad"]
        with (
            patch("llm_launchpad.cli.main.app", fake_app),
            patch("llm_launchpad.cli.main.sys.argv", argv),
        ):
            cli_main.main()
        self.assertEqual(argv, ["llm-launchpad", "tui"])
        fake_app.assert_called_once_with()

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

    def test_tui_defaults_to_terminal_selection_over_ssh(self) -> None:
        app_instance = Mock()
        with (
            patch.dict("os.environ", {"SSH_CONNECTION": "1"}, clear=True),
            patch("llm_launchpad.tui.app.TuiApp", return_value=app_instance) as app_cls,
            patch("llm_launchpad.core.backend.ModalBackend.terminate_all", return_value=None),
            patch("llm_launchpad.cli.main.sys.stdin.isatty", return_value=True),
            patch("llm_launchpad.cli.main.sys.stdout.isatty", return_value=True),
            patch("llm_launchpad.cli.main.ModalBackend.is_cli_available", return_value=True),
        ):
            cli_main.tui()
        app_cls.assert_called_once_with(mouse_enabled=False)
        app_instance.run.assert_called_once_with(mouse=False)

    def test_tui_allows_mouse_override(self) -> None:
        app_instance = Mock()
        with (
            patch("llm_launchpad.tui.app.TuiApp", return_value=app_instance) as app_cls,
            patch("llm_launchpad.core.backend.ModalBackend.terminate_all", return_value=None),
            patch("llm_launchpad.cli.main.sys.stdin.isatty", return_value=True),
            patch("llm_launchpad.cli.main.sys.stdout.isatty", return_value=True),
            patch("llm_launchpad.cli.main.ModalBackend.is_cli_available", return_value=True),
        ):
            cli_main.tui(mouse=True)
        app_cls.assert_called_once_with(mouse_enabled=True)
        app_instance.run.assert_called_once_with(mouse=True)

    def test_tui_allows_no_mouse_override(self) -> None:
        app_instance = Mock()
        with (
            patch("llm_launchpad.tui.app.TuiApp", return_value=app_instance) as app_cls,
            patch("llm_launchpad.core.backend.ModalBackend.terminate_all", return_value=None),
            patch("llm_launchpad.cli.main.sys.stdin.isatty", return_value=True),
            patch("llm_launchpad.cli.main.sys.stdout.isatty", return_value=True),
            patch("llm_launchpad.cli.main.ModalBackend.is_cli_available", return_value=True),
        ):
            cli_main.tui(mouse=False)
        app_cls.assert_called_once_with(mouse_enabled=False)
        app_instance.run.assert_called_once_with(mouse=False)

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

    def test_deploy_warmup_prefers_deployed_web_url_from_logs(self) -> None:
        warmup_calls: list[str] = []

        def _deploy(_config):  # type: ignore[no-untyped-def]
            return [
                LogEvent(
                    line=(
                        "└── 🔨 Created web function serve-discerning-tapir => "
                        "https://alice--vllm-test-serve-disc-42c728.modal.run (label truncated)"
                    )
                ),
                OperationCompleteEvent(operation=OperationType.DEPLOY, success=True, exit_code=0),
            ]

        def _warmup(
            _backend: BackendType,
            url: str,
            _timeout: int,
            _tail_logs: bool,
            app_name: str | None = None,
            served_model_name: str | None = None,
        ):
            warmup_calls.append(url)
            return []

        orch = SimpleNamespace(deploy=_deploy, warmup=_warmup)
        with patch("llm_launchpad.cli.main._preflight", return_value=(orch, "alice")):
            with patch("llm_launchpad.cli.main._print_banner", return_value=None):
                result = self.runner.invoke(
                    cli_main.app,
                    ["deploy", "--backend", "vllm", "--app-name", "vllm-test", "--do-warmup"],
                )
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(
            warmup_calls,
            ["https://alice--vllm-test-serve-disc-42c728.modal.run"],
        )

    def test_deploy_warmup_uses_function_slug_when_url_not_in_logs(self) -> None:
        warmup_calls: list[str] = []

        def _deploy(config):  # type: ignore[no-untyped-def]
            config.function_slug = "alpha-bravo"
            return [OperationCompleteEvent(operation=OperationType.DEPLOY, success=True, exit_code=0)]

        def _warmup(
            _backend: BackendType,
            url: str,
            _timeout: int,
            _tail_logs: bool,
            app_name: str | None = None,
            served_model_name: str | None = None,
        ):
            warmup_calls.append(url)
            return []

        orch = SimpleNamespace(deploy=_deploy, warmup=_warmup)
        with patch("llm_launchpad.cli.main._preflight", return_value=(orch, "alice")):
            with patch("llm_launchpad.cli.main._print_banner", return_value=None):
                result = self.runner.invoke(
                    cli_main.app,
                    ["deploy", "--backend", "llamacpp", "--app-name", "llamacpp-test", "--do-warmup"],
                )
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(
            warmup_calls,
            ["https://alice--llamacpp-test-serve-alpha-bravo.modal.run"],
        )

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

        def _check_status(
            backend: BackendType,
            url: str,
            timeout: int,
            served_model_name: str | None = None,
        ):
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
            served_model_name: str | None = None,
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

    def test_warmup_failure_returns_nonzero_exit_code(self) -> None:
        orch = SimpleNamespace(
            warmup=lambda *_args, **_kwargs: [
                OperationCompleteEvent(
                    operation=OperationType.WARMUP,
                    success=False,
                    exit_code=7,
                    detail="timeout",
                ),
            ]
        )
        with patch("llm_launchpad.cli.main._preflight", return_value=(orch, "alice")):
            with patch("llm_launchpad.cli.main._print_banner", return_value=None):
                result = self.runner.invoke(
                    cli_main.app, ["warmup", "--backend", "vllm", "--app-name", "vllm-test"]
                )
        self.assertEqual(result.exit_code, 7)

    def test_deploy_warmup_failure_returns_nonzero_exit_code(self) -> None:
        orch = SimpleNamespace(
            deploy=lambda _config: [
                OperationCompleteEvent(operation=OperationType.DEPLOY, success=True, exit_code=0),
            ],
            warmup=lambda *_args, **_kwargs: [
                OperationCompleteEvent(
                    operation=OperationType.WARMUP,
                    success=False,
                    exit_code=11,
                    detail="cold start failed",
                ),
            ],
        )
        with patch("llm_launchpad.cli.main._preflight", return_value=(orch, "alice")):
            with patch("llm_launchpad.cli.main._print_banner", return_value=None):
                result = self.runner.invoke(
                    cli_main.app,
                    [
                        "deploy",
                        "--backend",
                        "vllm",
                        "--app-name",
                        "vllm-test",
                        "--do-warmup",
                    ],
                )
        self.assertEqual(result.exit_code, 11)

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

    def test_switch_llamacpp_redeploy_warmup_uses_generated_function_slug(self) -> None:
        warmup_calls: list[str] = []
        orch = SimpleNamespace(
            deploy=lambda *_args, **_kwargs: [
                OperationCompleteEvent(operation=OperationType.DEPLOY, success=True, exit_code=0)
            ],
            warmup=lambda _backend, url, *_args, **_kwargs: warmup_calls.append(url) or [],
        )
        with patch("llm_launchpad.cli.main._preflight", return_value=(orch, "alice")):
            with patch("llm_launchpad.cli.main._print_banner", return_value=None):
                with patch("llm_launchpad.cli.main.random_function_slug", return_value="alpha-bravo"):
                    with patch("llm_launchpad.cli.main.ModalBackend.run_blocking", return_value=0):
                        result = self.runner.invoke(
                            cli_main.app,
                            [
                                "switch",
                                "--backend",
                                "llamacpp",
                                "--preset",
                                "qwen2.5-coder-7b-instruct-q4km",
                                "--do-warmup",
                            ],
                        )
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(
            warmup_calls,
            ["https://alice--llamacpp-qwen2-5-coder-7b-instruct-q4km-serve-alpha-bravo.modal.run"],
        )

    def test_switch_llamacpp_redeploy_sets_instance_name_and_app_name(self) -> None:
        deploy_configs: list[object] = []

        def _deploy(config):  # type: ignore[no-untyped-def]
            deploy_configs.append(config)
            return [OperationCompleteEvent(operation=OperationType.DEPLOY, success=True, exit_code=0)]

        orch = SimpleNamespace(deploy=_deploy, warmup=lambda *_args, **_kwargs: [])
        with patch("llm_launchpad.cli.main._preflight", return_value=(orch, "alice")):
            with patch("llm_launchpad.cli.main._print_banner", return_value=None):
                with patch("llm_launchpad.cli.main.ModalBackend.run_blocking", return_value=0):
                    with patch("llm_launchpad.cli.main.random_function_slug", return_value="test-slug"):
                        result = self.runner.invoke(
                            cli_main.app,
                            [
                                "switch",
                                "--backend",
                                "llamacpp",
                                "--repo-id",
                                "Qwen/Qwen2.5-Coder-7B-Instruct",
                                "--quant",
                                "Q4_K_M",
                                "--no-do-warmup",
                            ],
                        )
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(len(deploy_configs), 1)  # Only the switch call
        swtch_config = deploy_configs[0]
        self.assertIsNotNone(swtch_config.instance_name)
        self.assertEqual(swtch_config.instance_name, "qwen-qwen2-5-coder-7b-instruct")
        self.assertEqual(swtch_config.app_name, "llamacpp-qwen-qwen2-5-coder-7b-instruct")


if __name__ == "__main__":
    unittest.main()
