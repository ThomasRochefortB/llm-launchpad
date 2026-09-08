"""Interaction and failure-path regressions from the TUI feature audit."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from rich.markup import render as render_markup
from textual.app import App
from textual.widgets import Button, Input, Static

from llm_launchpad.protocol.enums import BackendType, ComputeProvider, OperationType
from llm_launchpad.protocol.events import OperationCompleteEvent
from llm_launchpad.protocol.models import DeploymentConfig
from llm_launchpad.tui.app import TuiApp
from llm_launchpad.tui.screens.manage import BenchmarkOptionsScreen, StatusOptionsScreen
from llm_launchpad.tui.screens.deploy import LlamaCppDeployScreen, VllmDeployScreen
from llm_launchpad.tui.screens.monitor import MonitorScreen
from llm_launchpad.tui.screens.setup import SetupRequiredScreen
from llm_launchpad.tui.screens.storage import StorageScreen
from llm_launchpad.tui.workers import ConnectionSummaryReady, OperationDone, StorageFailed, StorageLoaded
from tests.test_manage_screen_routing import _TestApp as ManageApp, _endpoint
from tests.test_llamacpp_deploy_screen import _TestApp as LlamaApp
from tests.test_vllm_deploy_screen import _TestApp as VllmApp
from tests.test_storage_screen import _TestApp as StorageApp, _sample_snapshot
from tests.test_tui_app_storage_cache import _MessageReceiver


class _StyledApp(App):
    CSS_PATH = TuiApp.CSS_PATH


class TuiAuditInteractionTests(unittest.IsolatedAsyncioTestCase):
    async def test_priority_deploy_shortcut_is_blocked_below_minimum_size(self) -> None:
        app = LlamaApp()
        async with app.run_test(size=(39, 12)) as pilot:
            screen = LlamaCppDeployScreen()
            app.push_screen(screen)
            await pilot.pause()
            await pilot.press("ctrl+d")
            self.assertIsNone(app.deployed_config)
            await pilot.resize_terminal(80, 24)
            await pilot.press("ctrl+d")
            self.assertIsNotNone(app.deployed_config)

    async def test_enter_activates_focused_connection_copy_button(self) -> None:
        app = _StyledApp()
        async with app.run_test(size=(120, 40)) as pilot:
            screen = MonitorScreen()
            app.push_screen(screen)
            await pilot.pause()
            screen.on_connection_summary_ready(ConnectionSummaryReady({
                "base_url": "https://example.test/v1", "model_id": "test",
                "display_name": "Test", "api_key": "audit-example-key",
            }))
            screen.on_operation_done(OperationDone(OperationType.DEPLOY, success=True))
            screen.query_one("#copy-url-btn", Button).focus()
            await pilot.press("enter")
            self.assertEqual(app.clipboard, "https://example.test/v1")
            self.assertIs(app.screen, screen)

    async def test_invalid_advanced_llamacpp_numbers_do_not_start_deploy(self) -> None:
        for field, value in (
            ("port-input", "abc"), ("port-input", "0"),
            ("port-input", "65536"), ("n-gpu-layers", "abc"),
        ):
            with self.subTest(field=field, value=value):
                app = LlamaApp()
                async with app.run_test() as pilot:
                    screen = LlamaCppDeployScreen()
                    app.push_screen(screen)
                    await pilot.pause()
                    screen.query_one(f"#{field}", Input).value = value
                    await pilot.press("ctrl+d")
                    self.assertIsNone(app.deployed_config)
                    self.assertTrue(app.notifications)

    async def test_invalid_tensor_parallel_does_not_start_deploy(self) -> None:
        for value in ("abc", "0", "-1"):
            with self.subTest(value=value):
                app = VllmApp()
                async with app.run_test() as pilot:
                    screen = VllmDeployScreen()
                    app.push_screen(screen)
                    await pilot.pause()
                    screen.query_one("#n-gpu", Input).value = value
                    await pilot.press("ctrl+d")
                    self.assertIsNone(app.deployed_config)
                    self.assertTrue(app.notifications)

    async def test_storage_error_is_displayed_as_literal_text(self) -> None:
        app = StorageApp()
        async with app.run_test() as pilot:
            screen = StorageScreen()
            app.push_screen(screen)
            await pilot.pause()
            screen.on_storage_failed(StorageFailed("provider returned [/unexpected]"))
            text = str(screen.query_one("#storage-status", Static).content)
            self.assertIn("[/unexpected]", render_markup(text).plain)

    async def test_setup_quit_keyboard_awaits_app_shutdown(self) -> None:
        app = _StyledApp()
        app.action_request_quit = AsyncMock()
        async with app.run_test() as pilot:
            app.push_screen(SetupRequiredScreen())
            await pilot.pause()
            await pilot.press("q")
            app.action_request_quit.assert_awaited_once()

    async def test_setup_quit_button_awaits_app_shutdown(self) -> None:
        app = _StyledApp()
        app.action_request_quit = AsyncMock()
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(SetupRequiredScreen())
            await pilot.pause()
            await pilot.click("#setup-quit-btn")
            app.action_request_quit.assert_awaited_once()

    async def test_done_returns_from_successful_operation_without_connection(self) -> None:
        app = _StyledApp()
        async with app.run_test() as pilot:
            screen = MonitorScreen("Status")
            app.push_screen(screen)
            await pilot.pause()
            screen.on_operation_done(OperationDone(OperationType.STATUS, success=True))
            await pilot.press("enter")
            self.assertIsNot(app.screen, screen)

    async def test_enter_closes_log_search_without_leaving_completed_monitor(self) -> None:
        app = _StyledApp()
        async with app.run_test() as pilot:
            screen = MonitorScreen()
            app.push_screen(screen)
            await pilot.pause()
            screen.on_operation_done(OperationDone(OperationType.STATUS, success=False))
            await pilot.press("/")
            await pilot.press("e", "r", "r", "o", "r", "enter")
            self.assertIs(app.screen, screen)
            self.assertFalse(screen.query_one("#monitor-search", Input).display)

    async def test_storage_empty_filter_clears_delete_target(self) -> None:
        app = StorageApp()
        async with app.run_test() as pilot:
            screen = StorageScreen()
            app.push_screen(screen)
            await pilot.pause()
            screen.query_one("#storage-table").focus()
            await pilot.pause()
            self.assertIsNotNone(screen._selected_model)
            screen.query_one("#storage-filter", Input).value = "no-such-model"
            await pilot.pause()
            screen.action_delete_selected_model()
            await pilot.pause()
            self.assertIs(app.screen, screen)
            self.assertIsNone(screen._selected_model)

    async def test_status_enter_submits_from_input(self) -> None:
        app = ManageApp()
        async with app.run_test() as pilot:
            app.push_screen(StatusOptionsScreen(_endpoint("test", "ap-test")))
            await pilot.pause()
            await pilot.press("enter")
            self.assertEqual(len(app.status_calls), 1)

    async def test_benchmark_enter_submits_from_input(self) -> None:
        app = ManageApp()
        async with app.run_test() as pilot:
            app.push_screen(BenchmarkOptionsScreen(_endpoint("test", "ap-test")))
            await pilot.pause()
            await pilot.press("enter")
            self.assertEqual(len(app.benchmark_calls), 1)

    async def test_invalid_benchmark_preserves_form_and_values(self) -> None:
        app = ManageApp()
        async with app.run_test() as pilot:
            screen = BenchmarkOptionsScreen(_endpoint("test", "ap-test"))
            app.push_screen(screen)
            await pilot.pause()
            screen.query_one("#benchmark-concurrency", Input).value = "1,nope"
            await pilot.press("ctrl+b")
            self.assertIs(app.screen, screen)
            self.assertEqual(app.benchmark_calls, [])
            self.assertTrue(str(screen.query_one("#benchmark-feedback", Static).content))


class TuiAuditStorageRefreshTests(unittest.TestCase):
    def test_modal_warmup_forwards_configured_endpoint_api_key(self) -> None:
        app = TuiApp()
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP, app_name="llamacpp-audit",
            endpoint_api_key="audit-example-key", do_warmup=True,
        )
        row = _endpoint("llamacpp-audit", "ap-audit", backend=BackendType.LLAMACPP)
        with (
            patch.object(app._orchestrator, "deploy", return_value=[
                OperationCompleteEvent(operation=OperationType.DEPLOY, success=True, data=row)
            ]),
            patch.object(app._orchestrator, "warmup", return_value=[]) as warmup,
        ):
            app._run_deploy(config, _MessageReceiver())
        self.assertEqual(warmup.call_args.kwargs.get("api_key"), "audit-example-key")

    def test_modal_status_forwards_stored_endpoint_api_key(self) -> None:
        app = TuiApp()
        with patch.object(app._orchestrator, "check_status", return_value=[]) as check:
            app._run_status(
                BackendType.LLAMACPP, "https://example.test", 60,
                "llamacpp-test", "test", ComputeProvider.MODAL,
                "audit-example-key", "ap-test", _MessageReceiver(),
            )
        self.assertEqual(check.call_args.kwargs.get("api_key"), "audit-example-key")

    def test_shared_storage_refresh_delivers_to_every_waiting_screen_once(self) -> None:
        app = TuiApp()
        first, second = _MessageReceiver(), _MessageReceiver()
        callbacks = []
        with patch.object(app, "run_worker", side_effect=lambda callback, **kw: callbacks.append(callback)):
            app.begin_storage_refresh(first)
            app.begin_storage_refresh(second)
            app.begin_storage_refresh(second)
        self.assertEqual(len(callbacks), 1)
        with patch.object(app._orchestrator, "list_storage", return_value=iter([
            OperationCompleteEvent(operation=OperationType.STORAGE_LIST, success=True, data=_sample_snapshot())
        ])):
            callbacks[0]()
        for receiver in (first, second):
            self.assertEqual(len(receiver.messages), 1)
            self.assertIsInstance(receiver.messages[0], StorageLoaded)

    def test_storage_exception_notifies_waiters_and_allows_retry(self) -> None:
        app = TuiApp()
        receiver = _MessageReceiver()
        callbacks = []
        with patch.object(app, "run_worker", side_effect=lambda callback, **kw: callbacks.append(callback)):
            app.begin_storage_refresh(receiver)
            with patch.object(app._orchestrator, "list_storage", side_effect=RuntimeError("offline")):
                callbacks[0]()
            self.assertIsInstance(receiver.messages[0], StorageFailed)
            self.assertEqual(receiver.messages[0].error, "offline")
            app.begin_storage_refresh(receiver)
        self.assertEqual(len(callbacks), 2)
