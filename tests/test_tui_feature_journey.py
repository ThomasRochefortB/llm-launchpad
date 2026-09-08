"""Exercise real TUI routing and workers against deterministic provider events."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from textual.widgets import Button, Input, OptionList, Select

from llm_launchpad.core.compute_availability import aggregate_compute_availability
from llm_launchpad.core.hf_models import GgufQuantMetadata
from llm_launchpad.protocol.enums import BackendType, DeploymentState, OperationType
from llm_launchpad.protocol.events import LogEvent, OperationCompleteEvent, StateChangeEvent
from llm_launchpad.tui.app import TuiApp
from llm_launchpad.tui.screens.deploy import LlamaCppDeployScreen, VllmDeployScreen
from llm_launchpad.tui.screens.fast_deploy import FastDeployScreen
from llm_launchpad.tui.screens.main_menu import MainMenuScreen
from llm_launchpad.tui.screens.manage import ConnectionInfoScreen, ManageScreen, StopConfirmScreen
from llm_launchpad.tui.screens.monitor import MonitorScreen
from llm_launchpad.tui.screens.quick_deploy import QuickDeployScreen
from llm_launchpad.tui.screens.settings import SettingsScreen
from llm_launchpad.tui.screens.setup import SetupRequiredScreen
from llm_launchpad.tui.screens.storage import StorageScreen
from tests.catalog_fixtures import activate_static_like_catalog
from tests.test_manage_screen_routing import _endpoint
from tests.test_storage_screen import _sample_snapshot


class _JourneyApp(TuiApp):
    def on_mount(self) -> None:
        self._username = "audit"
        self.push_screen(MainMenuScreen(username="audit"))


class TuiFeatureJourneyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        # Keep screen composition, input handling, app routing and worker
        # delivery real. Independent home-panel fetches have focused tests.
        for method in (
            "_refresh_modal_auth_status", "_refresh_prime_auth_status",
            "_refresh_hf_auth_status", "_refresh_aai_auth_status",
            "_refresh_panels", "_refresh_quick_deploy_catalog",
            "_refresh_storage_estimate", "_refresh_billing_panels",
        ):
            self.patches.enter_context(patch.object(MainMenuScreen, method))
        self.patches.enter_context(patch.object(TuiApp, "_sync_opencode"))
        self.patches.enter_context(patch("llm_launchpad.tui.app.write_system_clipboard"))
        self.patches.enter_context(patch(
            "llm_launchpad.tui.app.fetch_gguf_quant_metadata",
            return_value=GgufQuantMetadata(
                quantizations=["Q4_K_M"], vram_gb_by_quant={"Q4_K_M": 4.0}, architecture="llama"
            ),
        ))
        self.patches.enter_context(patch(
            "llm_launchpad.tui.screens.fast_deploy.load_compute_availability",
            return_value=aggregate_compute_availability(),
        ))
        activate_static_like_catalog()
        self.app = _JourneyApp()
        self.row = _endpoint("vllm-audit", "ap-audit")
        self.row.model_name = "org/audit-model"
        self.row.served_model_name = "audit-model"
        self.patches.enter_context(patch.object(
            self.app, "_visible_rows_and_prune_scope", return_value=([self.row], ())
        ))
        self.orchestrator = Mock()
        self.app._orchestrator = self.orchestrator
        snapshot = _sample_snapshot()
        snapshot.llamacpp_models = [replace(snapshot.llamacpp_models[0], incomplete=False)]
        for method, operation, data in (
            ("list_storage", OperationType.STORAGE_LIST, snapshot),
            ("predownload_model", OperationType.STORAGE_PREDOWNLOAD, None),
            ("delete_stored_model", OperationType.STORAGE_DELETE, None),
            ("stop_app", OperationType.STOP, None),
            ("deploy", OperationType.DEPLOY, self.row),
            ("warmup", OperationType.WARMUP, None),
            ("benchmark", OperationType.BENCHMARK, SimpleNamespace(
                best_concurrency=2, best_output_token_throughput=42.0, run_dir="/tmp/audit-benchmark"
            )),
        ):
            getattr(self.orchestrator, method).return_value = [
                OperationCompleteEvent(operation=operation, success=True, data=data)
            ]
        self.orchestrator.check_status.return_value = [
            StateChangeEvent(current=DeploymentState.RUNNING, operation=OperationType.STATUS),
            LogEvent(line="Status: healthy (backend=vllm)"),
            LogEvent(line="Test command: curl https://example.test/v1/models"),
            OperationCompleteEvent(operation=OperationType.STATUS, success=True),
        ]
        self.orchestrator.tail_logs.return_value = [
            LogEvent(line="Ready for requests"),
            LogEvent(line="Example error for search"),
            OperationCompleteEvent(operation=OperationType.LOGS, success=True),
        ]

    async def _settle(self, pilot) -> None:
        await pilot.pause()
        await self.app.workers.wait_for_complete()
        await pilot.pause()

    def _capture(self, name: str) -> None:
        """Optionally export review artifacts without changing normal test runs."""
        directory = os.environ.get("LLM_LAUNCHPAD_AUDIT_ARTIFACTS")
        if directory:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            self.app.save_screenshot(f"{name}.svg", path=str(path))

    async def test_deploy_routes_deliver_worker_results_and_return_home(self) -> None:
        async with self.app.run_test(size=(140, 45)) as pilot:
            await self._settle(pilot)
            self._capture("home")
            await pilot.press("?")
            self._capture("help")
            await pilot.press("escape", "s")
            self.assertIsInstance(self.app.screen, SettingsScreen)
            self.app.screen.query_one("#scaledown-window", Input).value = "900"
            self.app.screen.query_one("#tui-theme", Select).value = "launchpad-high-contrast"
            await pilot.press("ctrl+s")
            self.assertEqual(self.app.theme, "launchpad-high-contrast")
            self._capture("settings")
            await pilot.press("escape")

            for screen_type, backend_keys, model_field in (
                (LlamaCppDeployScreen, ("enter",), "repo-id"),
                (VllmDeployScreen, ("down", "enter"), "model-name"),
            ):
                await pilot.press("c", *backend_keys)
                await self._settle(pilot)
                self.assertIsInstance(self.app.screen, screen_type)
                screen = self.app.screen
                # Cached model selection must prefill the correct backend form.
                model_list = screen.query_one(
                    "#llama-model-list" if screen_type is LlamaCppDeployScreen else "#vllm-model-list",
                    OptionList,
                )
                model_list.focus()
                await pilot.press("home", "enter")
                self.assertTrue(screen.query_one(f"#{model_field}", Input).value)
                self._capture(screen_type.__name__)
                await pilot.press("ctrl+d")
                await self._settle(pilot)
                self.assertIsInstance(
                    self.app.screen, MonitorScreen,
                    [notice.message for notice in self.app._notifications],
                )
                self.assertTrue(self.app.screen._success)
                self._capture("deployed")
                await pilot.press("enter")
                self.assertIsInstance(self.app.screen, MainMenuScreen)

            await pilot.press("d")
            await self._settle(pilot)
            self.assertIsInstance(self.app.screen, FastDeployScreen)
            self._capture("model-picker")
            await pilot.press("enter")
            await self._settle(pilot)
            self._capture("placement-picker")
            await pilot.press("enter")
            await self._settle(pilot)
            self.assertIsInstance(self.app.screen, QuickDeployScreen)
            self._capture("quick-deploy")
            await pilot.press("ctrl+d")
            await self._settle(pilot)
            self.assertIsInstance(self.app.screen, MonitorScreen)
            self.assertTrue(self.app.screen._success)
            await pilot.press("enter")
            self.assertIsInstance(self.app.screen, MainMenuScreen)
            self.assertEqual(self.orchestrator.deploy.call_count, 3)
            self.assertEqual(self.orchestrator.warmup.call_count, 3)
            self.assertEqual(
                [call.args[0].backend for call in self.orchestrator.deploy.call_args_list[:2]],
                [BackendType.LLAMACPP, BackendType.VLLM],
            )

    async def test_management_storage_and_dialogs_complete_through_keyboard(self) -> None:
        async with self.app.run_test(size=(100, 30)) as pilot:
            await self._settle(pilot)
            self.app.push_screen(SetupRequiredScreen())
            await pilot.resize_terminal(60, 20)
            self._capture("setup-60x20")
            self.app.pop_screen()
            await pilot.resize_terminal(100, 30)
            await pilot.press("m")
            await self._settle(pilot)
            self.assertIsInstance(self.app.screen, ManageScreen)
            self._capture("manage")
            await pilot.press("enter")
            actions = self.app.screen.query_one("#manage-actions", OptionList)
            actions.highlighted = next(
                index for index in range(actions.option_count)
                if actions.get_option_at_index(index).id == "connection"
            )
            await pilot.press("enter")
            self.assertIsInstance(self.app.screen, ConnectionInfoScreen)
            self._capture("connection")
            await pilot.press("u")
            self.assertIn("example.test", self.app.clipboard)
            await pilot.press("escape", "s", "enter")
            await self._settle(pilot)
            self.assertIsInstance(self.app.screen, MonitorScreen)
            self.assertTrue(self.app.screen._result_rows)
            self._capture("status-result")
            await pilot.press("enter", "b", "enter")
            await self._settle(pilot)
            self.assertIsInstance(self.app.screen, MonitorScreen)
            self.assertTrue(self.app.screen._result_rows)
            self._capture("benchmark-result")
            await pilot.press("enter", "l")
            await self._settle(pilot)
            await pilot.press("/", "e", "r", "r", "o", "r", "enter", "n")
            self.assertIsInstance(self.app.screen, MonitorScreen)
            self._capture("logs")
            await pilot.press("escape", "x")
            self.assertIsInstance(self.app.screen, StopConfirmScreen)
            self._capture("stop-confirmation")
            await pilot.press("escape")
            self.orchestrator.stop_app.assert_not_called()
            await pilot.press("x", "right", "enter")
            await self._settle(pilot)
            self.orchestrator.stop_app.assert_called_once()
            await pilot.press("enter", "escape", "t")
            await self._settle(pilot)
            self.assertIsInstance(self.app.screen, StorageScreen)
            self._capture("storage")
            self.app.screen.query_one("#storage-table").focus()
            await pilot.press("p")
            await self._settle(pilot)
            self.orchestrator.predownload_model.assert_called_once()
            await pilot.press("enter")
            await self._settle(pilot)
            self.app.screen.query_one("#storage-table").focus()
            await pilot.press("x")
            self._capture("delete-confirmation")
            self.app.screen.query_one("#delete-confirm", Button).focus()
            await pilot.press("enter")
            await self._settle(pilot)
            self.orchestrator.delete_stored_model.assert_called_once()
            await pilot.press("enter", "escape")
            self.assertIsInstance(self.app.screen, MainMenuScreen)
