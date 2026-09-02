from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from llm_launchpad.cli import main as cli_main
from llm_launchpad.core.artificial_analysis_auth import save_artificial_analysis_api_key
from llm_launchpad.core.quick_deploy_refresh import ArtificialAnalysisAuthStatus
from llm_launchpad.protocol.enums import BackendType, ComputeProvider, OperationType
from llm_launchpad.protocol.events import EndpointAvailableEvent, LogEvent, OperationCompleteEvent
from llm_launchpad.protocol.models import EndpointInfo, PrimeProviderOptions


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

    def test_ensure_tui_runtime_requires_configured_provider(self) -> None:
        with (
            patch("llm_launchpad.cli.main.sys.stdin.isatty", return_value=True),
            patch("llm_launchpad.cli.main.sys.stdout.isatty", return_value=True),
            patch("llm_launchpad.cli.main.ModalBackend.is_cli_available", return_value=False),
            patch(
                "llm_launchpad.cli.main.get_prime_auth_status",
                return_value=SimpleNamespace(authenticated=False),
            ),
        ):
            with self.assertRaises(cli_main.typer.Exit) as ctx:
                cli_main._ensure_tui_runtime()
        self.assertEqual(ctx.exception.exit_code, 1)

    def test_ensure_tui_runtime_accepts_prime_without_modal_cli(self) -> None:
        with (
            patch("llm_launchpad.cli.main.sys.stdin.isatty", return_value=True),
            patch("llm_launchpad.cli.main.sys.stdout.isatty", return_value=True),
            patch("llm_launchpad.cli.main.ModalBackend.is_cli_available", return_value=False),
            patch(
                "llm_launchpad.cli.main.get_prime_auth_status",
                return_value=SimpleNamespace(authenticated=True),
            ),
        ):
            cli_main._ensure_tui_runtime()

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
        for target in (
            "llm_launchpad.cli.main._provider_instances",
            "llm_launchpad.cli.main._load_visible_launchpad_rows",
        ):
            discovery_patch = patch(target, return_value=[])
            discovery_patch.start()
            self.addCleanup(discovery_patch.stop)
        save_connection_patch = patch("llm_launchpad.cli.main.save_connection")
        save_connection_patch.start()
        self.addCleanup(save_connection_patch.stop)

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

    def test_invalid_provider_is_rejected_without_a_traceback(self) -> None:
        result = self.runner.invoke(
            cli_main.app,
            ["status", "--provider", "bogus", "--app-name", "vllm-test"],
        )

        self.assertEqual(result.exit_code, 2)
        self.assertIn("Invalid value for '--provider'", result.output)
        self.assertIn("modal", result.output)
        self.assertIn("prime", result.output)
        self.assertNotIn("Traceback", result.output)
        self.assertNotIsInstance(result.exception, ValueError)

    def test_invalid_backend_is_rejected_without_a_traceback(self) -> None:
        result = self.runner.invoke(
            cli_main.app,
            ["status", "--backend", "bogus", "--app-name", "vllm-test"],
        )

        self.assertEqual(result.exit_code, 2)
        self.assertIn("Invalid value for '--backend'", result.output)
        self.assertIn("llamacpp", result.output)
        self.assertIn("vllm", result.output)
        self.assertNotIn("Traceback", result.output)
        self.assertNotIsInstance(result.exception, ValueError)

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
        app_instance.mouse_enabled = False
        with (
            patch.dict("os.environ", {"SSH_CONNECTION": "1"}, clear=True),
            patch("llm_launchpad.tui.app.TuiApp", return_value=app_instance) as app_cls,
            patch("llm_launchpad.core.backend.ModalBackend.terminate_all", return_value=None),
            patch("llm_launchpad.cli.main.sys.stdin.isatty", return_value=True),
            patch("llm_launchpad.cli.main.sys.stdout.isatty", return_value=True),
            patch("llm_launchpad.cli.main.ModalBackend.is_cli_available", return_value=True),
        ):
            cli_main.tui()
        app_cls.assert_called_once_with(mouse_enabled=None)
        app_instance.run.assert_called_once_with(mouse=False)

    def test_tui_allows_mouse_override(self) -> None:
        app_instance = Mock()
        app_instance.mouse_enabled = True
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
        app_instance.mouse_enabled = False
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

    def test_deploy_cli_summarizes_verbose_backend_logs_by_default(self) -> None:
        orch = SimpleNamespace(
            deploy=lambda _config: [
                LogEvent(
                    line="Running: modal run -m llm_launchpad.backends.modal_llamacpp_app::main --preload"
                ),
                LogEvent(line="  env: SCALEDOWN_WINDOW=1800, GPU_CONFIG=T4:1"),
                LogEvent(line="print_info: n_embd = 896"),
                LogEvent(line="Server is ready!"),
                OperationCompleteEvent(operation=OperationType.DEPLOY, success=True, exit_code=0),
            ]
        )
        with patch("llm_launchpad.cli.main._preflight", return_value=(orch, "alice")):
            with patch("llm_launchpad.cli.main._print_banner", return_value=None):
                result = self.runner.invoke(
                    cli_main.app,
                    ["deploy", "--backend", "llamacpp", "--repo-id", "org/model"],
                )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Preparing model cache", result.output)
        self.assertIn("Server is ready!", result.output)
        self.assertNotIn("env: SCALEDOWN_WINDOW", result.output)
        self.assertNotIn("print_info:", result.output)
        self.assertNotIn("Running: modal run", result.output)

    def test_deploy_cli_debug_logs_keep_raw_backend_output(self) -> None:
        orch = SimpleNamespace(
            deploy=lambda _config: [
                LogEvent(line="  env: SCALEDOWN_WINDOW=1800, GPU_CONFIG=T4:1"),
                LogEvent(line="print_info: n_embd = 896"),
                OperationCompleteEvent(operation=OperationType.DEPLOY, success=True, exit_code=0),
            ]
        )
        with patch("llm_launchpad.cli.main._preflight", return_value=(orch, "alice")):
            with patch("llm_launchpad.cli.main._print_banner", return_value=None):
                result = self.runner.invoke(
                    cli_main.app,
                    [
                        "deploy",
                        "--backend",
                        "llamacpp",
                        "--repo-id",
                        "org/model",
                        "--debug-logs",
                    ],
                )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("env: SCALEDOWN_WINDOW=1800, GPU_CONFIG=T4:1", result.output)
        self.assertIn("print_info: n_embd = 896", result.output)

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

    def test_deploy_prime_llamacpp_maps_gguf_and_provider_options(self) -> None:
        captured = {}

        def _deploy(config):  # type: ignore[no-untyped-def]
            captured["config"] = config
            return [
                OperationCompleteEvent(
                    operation=OperationType.DEPLOY,
                    success=True,
                    exit_code=0,
                )
            ]

        orch = SimpleNamespace(deploy=_deploy)
        with (
            patch("llm_launchpad.cli.main._preflight", return_value=(orch, "prime-user")),
            patch("llm_launchpad.cli.main._print_banner", return_value=None),
        ):
            result = self.runner.invoke(
                cli_main.app,
                [
                    "deploy",
                    "--provider",
                    "prime",
                    "--backend",
                    "llamacpp",
                    "--repo-id",
                    "org/Model-GGUF",
                    "--quant",
                    "Q4_K_M",
                    "--gpu-type",
                    "H100_80GB",
                    "--gpu-count",
                    "1",
                    "--prime-offer-id",
                    "abc123",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        config = captured["config"]
        self.assertEqual(config.provider, ComputeProvider.PRIME)
        self.assertEqual(config.backend, BackendType.LLAMACPP)
        self.assertEqual(config.repo_id, "org/Model-GGUF")
        self.assertEqual(config.quant, "Q4_K_M")
        self.assertEqual(config.app_name, "llp-prime-llamacpp-org-model-gguf")
        self.assertEqual(
            config.provider_options,
            PrimeProviderOptions(
                offer_id="abc123",
            ),
        )

    def test_deploy_prime_llamacpp_requires_repo_id(self) -> None:
        orch = SimpleNamespace(deploy=lambda _config: [])
        with (
            patch("llm_launchpad.cli.main._preflight", return_value=(orch, "prime-user")),
            patch("llm_launchpad.cli.main._print_banner", return_value=None),
        ):
            result = self.runner.invoke(
                cli_main.app,
                ["deploy", "--provider", "prime", "--backend", "llamacpp"],
            )

        self.assertEqual(result.exit_code, 2)
        self.assertIn("requires --repo-id", result.output)

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

    def test_deploy_syncs_opencode_after_success_without_warmup(self) -> None:
        orch = SimpleNamespace(
            deploy=lambda _config: [
                OperationCompleteEvent(operation=OperationType.DEPLOY, success=True, exit_code=0),
            ]
        )
        with (
            patch("llm_launchpad.cli.main._preflight", return_value=(orch, "alice")),
            patch("llm_launchpad.cli.main._print_banner", return_value=None),
            patch("llm_launchpad.cli.main._load_visible_launchpad_rows", return_value=[]),
            patch("llm_launchpad.cli.main._sync_opencode_cli") as sync_mock,
        ):
            result = self.runner.invoke(
                cli_main.app,
                ["deploy", "--backend", "vllm", "--app-name", "vllm-test"],
            )
        self.assertEqual(result.exit_code, 0)
        sync_mock.assert_called_once()
        self.assertEqual(sync_mock.call_args.kwargs["target_app_name"], "vllm-test")
        self.assertEqual(sync_mock.call_args.kwargs["target_url"], "https://alice--vllm-test-serve.modal.run")
        self.assertEqual(sync_mock.call_args.kwargs["username"], "alice")

    def test_deploy_syncs_opencode_when_prime_endpoint_url_is_available(self) -> None:
        endpoint = EndpointInfo(
            name="llp-prime-vllm-qwen",
            app_id="pod-1",
            web_url="https://t-0-abc.tunnel.pinfra.io",
            backend=BackendType.VLLM,
            provider=ComputeProvider.PRIME,
            endpoint_api_key="secret",
        )
        orch = SimpleNamespace(
            deploy=lambda _config: [
                EndpointAvailableEvent(endpoint=endpoint, operation=OperationType.DEPLOY),
                OperationCompleteEvent(
                    operation=OperationType.DEPLOY,
                    success=True,
                    exit_code=0,
                    data=endpoint,
                ),
            ],
            warmup=lambda *_args, **_kwargs: [
                OperationCompleteEvent(
                    operation=OperationType.WARMUP,
                    success=True,
                    data={"url": "https://t-0-abc.tunnel.pinfra.io"},
                )
            ],
        )
        with (
            patch("llm_launchpad.cli.main._preflight", return_value=(orch, "alice")),
            patch("llm_launchpad.cli.main._print_banner", return_value=None),
            patch("llm_launchpad.cli.main._load_visible_launchpad_rows", return_value=[]),
            patch("llm_launchpad.cli.main.save_connection") as save_mock,
            patch("llm_launchpad.cli.main._sync_opencode_cli") as sync_mock,
        ):
            result = self.runner.invoke(
                cli_main.app,
                [
                    "deploy",
                    "--provider",
                    "prime",
                    "--backend",
                    "vllm",
                    "--model-name",
                    "Qwen/Qwen3-4B",
                    "--app-name",
                    "llp-prime-vllm-qwen",
                    "--do-warmup",
                ],
            )
        self.assertEqual(result.exit_code, 0, result.output)
        sync_mock.assert_called_once()
        self.assertEqual(sync_mock.call_args.kwargs["target_app_name"], "llp-prime-vllm-qwen")
        self.assertEqual(
            sync_mock.call_args.kwargs["target_url"],
            "https://t-0-abc.tunnel.pinfra.io",
        )
        save_mock.assert_called()

    def test_deploy_no_prime_disk_sets_auto_disk_false(self) -> None:
        captured: dict[str, object] = {}

        def _deploy(config):  # type: ignore[no-untyped-def]
            captured["config"] = config
            return [
                OperationCompleteEvent(operation=OperationType.DEPLOY, success=True, exit_code=0),
            ]

        orch = SimpleNamespace(deploy=_deploy)
        with (
            patch("llm_launchpad.cli.main._preflight", return_value=(orch, "alice")),
            patch("llm_launchpad.cli.main._print_banner", return_value=None),
            patch("llm_launchpad.cli.main._load_visible_launchpad_rows", return_value=[]),
            patch("llm_launchpad.cli.main._sync_opencode_cli"),
        ):
            result = self.runner.invoke(
                cli_main.app,
                [
                    "deploy",
                    "--provider",
                    "prime",
                    "--backend",
                    "vllm",
                    "--model-name",
                    "Qwen/Qwen3-4B",
                    "--no-prime-disk",
                ],
            )
        self.assertEqual(result.exit_code, 0, result.output)
        config = captured["config"]
        assert isinstance(config.provider_options, PrimeProviderOptions)
        self.assertFalse(config.provider_options.auto_disk)

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
            ["https://alice--llp-lc-c0e8ad06e105.modal.run"],
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

    def test_list_command_triggers_opencode_prune_with_visible_rows(self) -> None:
        visible_rows = [EndpointInfo(name="vllm-qwen3", backend=BackendType.VLLM, state="running")]
        orch = SimpleNamespace(
            list_deployments=lambda: [
                OperationCompleteEvent(operation=OperationType.LIST, success=True, data=visible_rows),
            ]
        )
        with (
            patch("llm_launchpad.cli.main._preflight", return_value=(orch, "alice")),
            patch("llm_launchpad.cli.main._print_banner", return_value=None),
            patch("llm_launchpad.cli.main._sync_opencode_cli") as sync_mock,
        ):
            result = self.runner.invoke(cli_main.app, ["list"])
        self.assertEqual(result.exit_code, 0)
        sync_mock.assert_called_once_with(
            current_rows=visible_rows,
            username="alice",
            prune_providers=(ComputeProvider.MODAL,),
        )

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

    def test_logs_failure_propagates_operation_exit_code(self) -> None:
        orch = SimpleNamespace(
            tail_logs=lambda *_args, **_kwargs: [
                OperationCompleteEvent(
                    operation=OperationType.LOGS,
                    success=False,
                    exit_code=8,
                    detail="log command failed",
                ),
            ]
        )
        with (
            patch("llm_launchpad.cli.main._preflight", return_value=(orch, "alice")),
            patch("llm_launchpad.cli.main._print_banner", return_value=None),
        ):
            result = self.runner.invoke(
                cli_main.app,
                ["logs", "--backend", "vllm", "--app-name", "vllm-qwen3"],
            )

        self.assertEqual(result.exit_code, 8)

    def test_benchmark_maps_cli_options_to_benchmark_config(self) -> None:
        captured = {}
        row = EndpointInfo(
            name="vllm-qwen3",
            backend=BackendType.VLLM,
            state="running",
            web_url="https://alice--vllm-qwen3-serve.modal.run",
            served_model_name="Qwen3-4B",
        )

        def _benchmark(config):  # type: ignore[no-untyped-def]
            captured["config"] = config
            return [OperationCompleteEvent(operation=OperationType.BENCHMARK, success=True, exit_code=0)]

        orch = SimpleNamespace(benchmark=_benchmark)
        with (
            patch("llm_launchpad.cli.main._preflight", return_value=(orch, "alice")),
            patch("llm_launchpad.cli.main._print_banner", return_value=None),
            patch("llm_launchpad.cli.main._load_visible_launchpad_rows", return_value=[row]),
        ):
            result = self.runner.invoke(
                cli_main.app,
                [
                    "benchmark",
                    "--backend",
                    "vllm",
                    "--app-name",
                    "vllm-qwen3",
                    "--concurrency",
                    "1,4",
                    "--request-count",
                    "33",
                    "--input-tokens",
                    "1024",
                    "--output-tokens",
                    "128",
                    "--tokenizer",
                    "gpt2",
                    "--request-timeout-seconds",
                    "45",
                    "--output-dir",
                    "/tmp/bench",
                    "--aiperf-arg",
                    "--warmup-request-count",
                    "--aiperf-arg",
                    "2",
                ],
            )
        self.assertEqual(result.exit_code, 0)
        config = captured["config"]
        self.assertEqual(config.backend, BackendType.VLLM)
        self.assertEqual(config.app_name, "vllm-qwen3")
        self.assertEqual(config.server_url, "https://alice--vllm-qwen3-serve.modal.run")
        self.assertEqual(config.model_name, "Qwen3-4B")
        self.assertEqual(config.concurrency, [1, 4])
        self.assertEqual(config.request_count, 33)
        self.assertEqual(config.input_tokens, 1024)
        self.assertEqual(config.output_tokens, 128)
        self.assertEqual(config.request_timeout_seconds, 45)
        self.assertEqual(config.output_dir, "/tmp/bench")
        self.assertEqual(config.aiperf_args, ["--warmup-request-count", "2"])

    def test_benchmark_invalid_concurrency_returns_error(self) -> None:
        orch = SimpleNamespace(benchmark=lambda _config: [])
        with (
            patch("llm_launchpad.cli.main._preflight", return_value=(orch, "alice")),
            patch("llm_launchpad.cli.main._print_banner", return_value=None),
            patch("llm_launchpad.cli.main._load_visible_launchpad_rows", return_value=[]),
        ):
            result = self.runner.invoke(
                cli_main.app,
                ["benchmark", "--backend", "vllm", "--app-name", "vllm-qwen3", "--concurrency", "bad"],
            )
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Invalid concurrency", result.output)

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

    def test_stop_triggers_opencode_sync_with_removed_app_name(self) -> None:
        orch = SimpleNamespace(
            stop_app=lambda *_args, **_kwargs: [
                OperationCompleteEvent(operation=OperationType.STOP, success=True, exit_code=0),
            ]
        )
        with (
            patch("llm_launchpad.cli.main._preflight", return_value=(orch, "alice")),
            patch("llm_launchpad.cli.main._print_banner", return_value=None),
            patch("llm_launchpad.cli.main._load_visible_launchpad_rows", return_value=[]),
            patch("llm_launchpad.cli.main._sync_opencode_cli") as sync_mock,
        ):
            result = self.runner.invoke(
                cli_main.app,
                ["stop", "--backend", "vllm", "--yes", "--app-name", "vllm-test"],
            )
        self.assertEqual(result.exit_code, 0)
        sync_mock.assert_called_once_with(
            current_rows=[],
            remove_app_names=["vllm-test"],
            prune_providers=(ComputeProvider.MODAL,),
            username="alice",
        )

    def test_stop_failure_propagates_operation_exit_code_without_cleanup(self) -> None:
        orch = SimpleNamespace(
            stop_app=lambda *_args, **_kwargs: [
                OperationCompleteEvent(
                    operation=OperationType.STOP,
                    success=False,
                    exit_code=9,
                    detail="stop command failed",
                ),
            ]
        )
        with (
            patch("llm_launchpad.cli.main._preflight", return_value=(orch, "alice")),
            patch("llm_launchpad.cli.main._print_banner", return_value=None),
            patch("llm_launchpad.cli.main.remove_connection") as remove_mock,
            patch("llm_launchpad.cli.main._sync_opencode_cli") as sync_mock,
        ):
            result = self.runner.invoke(
                cli_main.app,
                ["stop", "--backend", "vllm", "--yes", "--app-name", "vllm-test"],
            )

        self.assertEqual(result.exit_code, 9)
        remove_mock.assert_not_called()
        sync_mock.assert_not_called()

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
            ["https://alice--llp-lc-f16f2cc918bf.modal.run"],
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

    def test_opencode_sync_command_uses_target_rows_and_dry_run(self) -> None:
        rows = [SimpleNamespace(name="vllm-qwen3")]
        with (
            patch("llm_launchpad.cli.main._print_banner", return_value=None),
            patch(
                "llm_launchpad.cli.main._connected_compute_providers",
                return_value=(ComputeProvider.MODAL, ComputeProvider.PRIME),
            ),
            patch("llm_launchpad.cli.main.ModalBackend.get_username", return_value=""),
            patch(
                "llm_launchpad.cli.main._load_rows_for_opencode_sync",
                return_value=(rows, (ComputeProvider.MODAL, ComputeProvider.PRIME), []),
            ) as load_rows,
            patch("llm_launchpad.cli.main._sync_opencode_cli") as sync_mock,
        ):
            result = self.runner.invoke(
                cli_main.app,
                ["opencode", "sync", "--app-name", "vllm-qwen3", "--dry-run"],
            )
        self.assertEqual(result.exit_code, 0)
        load_rows.assert_called_once_with(None, persist_backfill=False)
        sync_mock.assert_called_once_with(
            target_app_name="vllm-qwen3",
            current_rows=rows,
            prune_providers=(ComputeProvider.MODAL, ComputeProvider.PRIME),
            username="",
            dry_run=True,
            fail_on_error=True,
        )

    def test_opencode_sync_command_can_limit_to_one_provider(self) -> None:
        rows = [SimpleNamespace(name="vllm-qwen3")]
        with (
            patch("llm_launchpad.cli.main._preflight", return_value=(SimpleNamespace(), "alice")),
            patch("llm_launchpad.cli.main._print_banner", return_value=None),
            patch(
                "llm_launchpad.cli.main._load_rows_for_opencode_sync",
                return_value=(rows, (ComputeProvider.MODAL,), []),
            ),
            patch("llm_launchpad.cli.main._sync_opencode_cli") as sync_mock,
        ):
            result = self.runner.invoke(
                cli_main.app,
                ["opencode", "sync", "--provider", "modal", "--dry-run"],
            )
        self.assertEqual(result.exit_code, 0)
        sync_mock.assert_called_once_with(
            target_app_name=None,
            current_rows=rows,
            prune_providers=(ComputeProvider.MODAL,),
            username="alice",
            dry_run=True,
            fail_on_error=True,
        )

    def test_opencode_sync_upserts_cached_connections_when_listing_fails(self) -> None:
        rows = [SimpleNamespace(name="llamacpp-qwen")]
        with (
            patch("llm_launchpad.cli.main._print_banner", return_value=None),
            patch(
                "llm_launchpad.cli.main._connected_compute_providers",
                return_value=(ComputeProvider.MODAL, ComputeProvider.PRIME),
            ),
            patch("llm_launchpad.cli.main.ModalBackend.get_username", return_value=""),
            patch(
                "llm_launchpad.cli.main._load_rows_for_opencode_sync",
                return_value=(rows, (), ["Modal listing failed; using cached connection summaries."]),
            ),
            patch("llm_launchpad.cli.main._sync_opencode_cli") as sync_mock,
        ):
            result = self.runner.invoke(cli_main.app, ["opencode", "sync"])
        self.assertEqual(result.exit_code, 0)
        sync_mock.assert_called_once_with(
            target_app_name=None,
            current_rows=rows,
            # Cached rows must never become prune authorization when no live
            # provider listing succeeded.
            prune_providers=(),
            username="",
            dry_run=False,
            fail_on_error=True,
        )

    def test_opencode_sync_fails_loudly_when_no_deployment_source_is_available(self) -> None:
        with (
            patch("llm_launchpad.cli.main._print_banner", return_value=None),
            patch(
                "llm_launchpad.cli.main._connected_compute_providers",
                return_value=(ComputeProvider.MODAL, ComputeProvider.PRIME),
            ),
            patch("llm_launchpad.cli.main.ModalBackend.get_username", return_value=""),
            patch(
                "llm_launchpad.cli.main._load_rows_for_opencode_sync",
                return_value=(None, (), ["Modal listing failed and no cached connections are available."]),
            ),
        ):
            result = self.runner.invoke(cli_main.app, ["opencode", "sync"])
        self.assertEqual(result.exit_code, 1)

    def test_load_rows_for_opencode_sync_falls_back_to_cached_connections(self) -> None:
        cached = [
            EndpointInfo(
                name="llamacpp-qwen",
                backend=BackendType.LLAMACPP,
                provider=ComputeProvider.MODAL,
                web_url="https://cached.modal.run",
                served_model_name="Qwen3-4B",
            )
        ]
        with (
            patch(
                "llm_launchpad.cli.main._connected_compute_providers",
                return_value=(ComputeProvider.MODAL,),
            ),
            patch("llm_launchpad.cli.main.ModalBackend.list_apps", return_value=None),
            patch(
                "llm_launchpad.cli.main.merge_connections",
                side_effect=lambda rows, **_kwargs: rows,
            ),
            patch(
                "llm_launchpad.cli.main.rows_from_connection_cache",
                return_value=cached,
            ),
        ):
            rows, listed, notes = cli_main._load_rows_for_opencode_sync()
        self.assertIsNotNone(rows)
        self.assertEqual([row.name for row in rows or []], ["llamacpp-qwen"])
        self.assertEqual(listed, ())
        self.assertTrue(any("using cached connection summaries" in note for note in notes))

    def test_load_rows_for_opencode_sync_keeps_live_listed_providers_as_prune_scope(self) -> None:
        live_prime = [
            EndpointInfo(
                name="llp-prime-vllm-qwen",
                backend=BackendType.VLLM,
                provider=ComputeProvider.PRIME,
                web_url="https://prime.example",
            )
        ]
        cached_modal = [
            EndpointInfo(
                name="llamacpp-qwen",
                backend=BackendType.LLAMACPP,
                provider=ComputeProvider.MODAL,
                web_url="https://cached.modal.run",
            )
        ]
        with (
            patch(
                "llm_launchpad.cli.main._connected_compute_providers",
                return_value=(ComputeProvider.MODAL, ComputeProvider.PRIME),
            ),
            patch("llm_launchpad.cli.main.ModalBackend.list_apps", return_value=None),
            patch("llm_launchpad.cli.main.PrimeBackend") as prime_mock,
            patch(
                "llm_launchpad.cli.main.merge_connections",
                side_effect=lambda rows, **_kwargs: rows,
            ),
            patch(
                "llm_launchpad.cli.main.rows_from_connection_cache",
                return_value=cached_modal,
            ),
        ):
            prime_mock.return_value.list_deployments.return_value = live_prime
            rows, listed, notes = cli_main._load_rows_for_opencode_sync()
        self.assertEqual(listed, (ComputeProvider.PRIME,))
        self.assertEqual(
            sorted(row.name for row in rows or []),
            ["llamacpp-qwen", "llp-prime-vllm-qwen"],
        )
        self.assertTrue(any("using cached connection summaries" in note for note in notes))

    def test_load_rows_for_opencode_sync_reports_no_source_when_listing_fails_and_cache_is_empty(self) -> None:
        with (
            patch(
                "llm_launchpad.cli.main._connected_compute_providers",
                return_value=(ComputeProvider.MODAL,),
            ),
            patch("llm_launchpad.cli.main.ModalBackend.list_apps", return_value=None),
            patch("llm_launchpad.cli.main.merge_connections", side_effect=lambda rows: rows),
            patch("llm_launchpad.cli.main.rows_from_connection_cache", return_value=[]),
        ):
            rows, listed, notes = cli_main._load_rows_for_opencode_sync()
        self.assertIsNone(rows)
        self.assertEqual(listed, ())
        self.assertTrue(any("no cached connections are available" in note for note in notes))

    def test_sync_opencode_cli_backfills_visible_rows_when_target_is_omitted(self) -> None:
        rows = [
            EndpointInfo(
                name="vllm-qwen3",
                state="running",
                backend=BackendType.VLLM,
                instance_name="qwen3",
                web_url="https://alice--vllm-qwen3-serve.modal.run",
                model_name="Qwen/Qwen3-4B",
                served_model_name="Qwen3-4B",
            )
        ]
        with patch("llm_launchpad.cli.main.sync_opencode_config") as sync_mock:
            sync_mock.return_value = SimpleNamespace(messages=[])
            cli_main._sync_opencode_cli(current_rows=rows, username="alice")

        sync_mock.assert_called_once()
        targets = sync_mock.call_args.kwargs["targets"]
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].app_name, "vllm-qwen3")
        self.assertEqual(targets[0].base_url, "https://alice--vllm-qwen3-serve.modal.run/v1")


class CliMainAaiAuthCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_login_stores_validated_key(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            key_path = Path(temporary_directory) / "aa.json"
            with (
                patch(
                    "llm_launchpad.core.artificial_analysis_auth.AAI_AUTH_PATH",
                    key_path,
                ),
                patch(
                    "llm_launchpad.core.quick_deploy_refresh.get_artificial_analysis_auth_status",
                    return_value=ArtificialAnalysisAuthStatus(authenticated=True, tier="pro"),
                ),
            ):
                result = self.runner.invoke(cli_main.app, ["aai-auth", "login", "secret-key"])

            self.assertEqual(result.exit_code, 0)
            self.assertIn("Authenticated (pro tier)", result.output)
            self.assertTrue(key_path.exists())

    def test_login_rejects_invalid_key_without_storing(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            key_path = Path(temporary_directory) / "aa.json"
            with (
                patch(
                    "llm_launchpad.core.artificial_analysis_auth.AAI_AUTH_PATH",
                    key_path,
                ),
                patch(
                    "llm_launchpad.core.quick_deploy_refresh.get_artificial_analysis_auth_status",
                    return_value=ArtificialAnalysisAuthStatus(
                        authenticated=False,
                        error="Invalid Artificial Analysis API key",
                    ),
                ),
            ):
                result = self.runner.invoke(cli_main.app, ["aai-auth", "login", "bad-key"])

            self.assertEqual(result.exit_code, 1)
            self.assertIn("Invalid Artificial Analysis API key", result.output)
            self.assertFalse(key_path.exists())

    def test_status_reports_stored_key_source_and_tier(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            key_path = Path(temporary_directory) / "aa.json"
            save_artificial_analysis_api_key("secret-key", path=key_path)
            with (
                patch.dict("os.environ", {}, clear=True),
                patch(
                    "llm_launchpad.core.artificial_analysis_auth.AAI_AUTH_PATH",
                    key_path,
                ),
                patch(
                    "llm_launchpad.core.quick_deploy_refresh.get_artificial_analysis_auth_status",
                    return_value=ArtificialAnalysisAuthStatus(authenticated=True, tier="free"),
                ),
            ):
                result = self.runner.invoke(cli_main.app, ["aai-auth", "status"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("AAI API key source: stored. Authenticated (free tier).", result.output)

    def test_status_reports_missing_key(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "llm_launchpad.core.artificial_analysis_auth.load_saved_artificial_analysis_api_key",
                return_value="",
            ),
        ):
            result = self.runner.invoke(cli_main.app, ["aai-auth", "status"])

        self.assertEqual(result.exit_code, 1)
        self.assertIn("llm-launchpad aai-auth login", result.output)

    def test_clear_removes_stored_key(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            key_path = Path(temporary_directory) / "aa.json"
            save_artificial_analysis_api_key("secret-key", path=key_path)
            with patch(
                "llm_launchpad.core.artificial_analysis_auth.AAI_AUTH_PATH",
                key_path,
            ):
                result = self.runner.invoke(cli_main.app, ["aai-auth", "clear"])

            self.assertEqual(result.exit_code, 0)
            self.assertIn("removed", result.output)
            self.assertFalse(key_path.exists())

    def test_clear_without_stored_key_fails(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            key_path = Path(temporary_directory) / "aa.json"
            with patch(
                "llm_launchpad.core.artificial_analysis_auth.AAI_AUTH_PATH",
                key_path,
            ):
                result = self.runner.invoke(cli_main.app, ["aai-auth", "clear"])

        self.assertEqual(result.exit_code, 1)
        self.assertIn("No stored Artificial Analysis API key to remove.", result.output)


if __name__ == "__main__":
    unittest.main()
