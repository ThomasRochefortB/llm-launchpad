from __future__ import annotations

import unittest
from unittest.mock import patch

from rich.cells import cell_len
from textual.app import App
from textual.screen import Screen
from textual.widgets import Button, Input, Log, Static

from llm_launchpad.protocol.enums import BackendType, DeploymentState, OperationType
from llm_launchpad.tui.deploy_log_summary import SUMMARY_SPINNER_FRAMES
from llm_launchpad.tui.screens.monitor import MonitorScreen, _connection_copy_text
from llm_launchpad.tui.widgets.log_viewer import (
    MAX_RETAINED_LOG_LINES,
    SelectableLog,
    prune_retained_items,
)
from llm_launchpad.tui.workers import (
    ConnectionSummaryReady,
    LogMessage,
    OperationDone,
    OperationError,
    StateChanged,
)


class _TestApp(App[None]):
    pass


class _HomeApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.home_calls = 0

    def pop_to_main_menu(self) -> None:
        self.home_calls += 1
        self.pop_screen()


class MonitorScreenTests(unittest.IsolatedAsyncioTestCase):
    def test_log_history_prunes_in_chunks_at_the_retention_limit(self) -> None:
        lines = list(range(MAX_RETAINED_LOG_LINES + 1))

        pruned_count = prune_retained_items(lines)

        self.assertEqual(pruned_count, 1_000)
        self.assertEqual(len(lines), MAX_RETAINED_LOG_LINES - 999)
        self.assertEqual(lines[0], 1_000)

    async def test_log_lines_reflow_to_terminal_width_without_horizontal_scroll(
        self,
    ) -> None:
        app = _TestApp()
        async with app.run_test(size=(70, 30)) as pilot:
            app.push_screen(MonitorScreen(title="Logs"))
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, MonitorScreen)
            source_line = (
                "Provisioning provider instance with "
                "an-unbroken-deployment-identifier-that-must-fold-at-the-edge"
            )
            screen.on_log_message(LogMessage(source_line))
            await pilot.pause()

            log = screen.log_viewer.log_widget
            wide_rows = log.wrapped_lines
            self.assertEqual(list(log.lines), [source_line])
            self.assertGreater(len(wide_rows), 1)
            self.assertTrue(
                all(cell_len(line) <= log.wrap_width for line in wide_rows)
            )
            self.assertEqual(log.max_scroll_x, 0)

            await pilot.resize_terminal(45, 30)
            await pilot.pause()

            narrow_rows = log.wrapped_lines
            self.assertGreater(len(narrow_rows), len(wide_rows))
            self.assertTrue(
                all(cell_len(line) <= log.wrap_width for line in narrow_rows)
            )
            self.assertLessEqual(
                log.virtual_size.width,
                log.scrollable_content_region.width,
            )
            self.assertEqual(log.max_scroll_x, 0)

    async def test_log_lines_are_appended_as_plain_text(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(MonitorScreen(title="Logs"))
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, MonitorScreen)
            self.assertIsInstance(screen.log_viewer.log_widget, Log)
            screen.on_log_message(LogMessage("first line"))
            screen.on_log_message(LogMessage("stderr line", stream="stderr"))
            await pilot.pause()

            content = "\n".join(screen.log_viewer.log_widget.lines)
            self.assertIn("first line", content)
            self.assertIn("stderr | stderr line", content)
            title = screen.query_one("#monitor-title", Static)
            self.assertIn("Logs", str(title.content))
            self.assertIn("MOUSE", str(title.content))
            view_status = screen.query_one("#monitor-view-status", Static)
            self.assertIn("FOLLOWING", str(view_status.content))
            self.assertIn("2 lines", str(view_status.content))

    async def test_title_mentions_terminal_selection_when_mouse_disabled(self) -> None:
        app = _TestApp()
        app.mouse_enabled = False
        async with app.run_test() as pilot:
            app.push_screen(MonitorScreen(title="Logs"))
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, MonitorScreen)
            title = screen.query_one("#monitor-title", Static)
            self.assertIn("TERMINAL SELECT", str(title.content))

    async def test_title_shows_paused_follow_state_and_unseen_lines(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(MonitorScreen(title="Logs"))
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, MonitorScreen)
            for index in range(40):
                screen.on_log_message(LogMessage(f"line {index}"))
            await pilot.pause()

            log = screen.log_viewer.log_widget
            log.scroll_home(animate=False, immediate=True, x_axis=False)
            await pilot.pause()
            screen.on_log_message(LogMessage("new while paused"))
            await pilot.pause()

            view_status = screen.query_one("#monitor-view-status", Static)
            self.assertIn("PAUSED", str(view_status.content))
            self.assertIn("1 new line", str(view_status.content))
            self.assertIn("41 lines", str(view_status.content))

            screen.action_resume_follow()
            await pilot.pause()
            self.assertIn("FOLLOWING", str(view_status.content))
            self.assertNotIn("new line", str(view_status.content))

    async def test_log_lines_strip_ansi_escape_sequences(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(MonitorScreen(title="Logs"))
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, MonitorScreen)
            screen.on_log_message(LogMessage("\x1b[0;36m(APIServer pid=4)\x1b[0;0m INFO booted"))
            await pilot.pause()

            content = "\n".join(screen.log_viewer.log_widget.lines)
            self.assertIn("(APIServer pid=4) INFO booted", content)
            self.assertNotIn("\x1b[0;36m", content)
            self.assertNotIn("\x1b[0;0m", content)

    async def test_clear_action_empties_log(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(MonitorScreen(title="Logs"))
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, MonitorScreen)
            screen.on_log_message(LogMessage("line to clear"))
            await pilot.pause()
            self.assertGreater(screen.log_viewer.log_widget.line_count, 0)

            screen.action_clear_log()
            await pilot.pause()
            self.assertEqual(screen.log_viewer.log_widget.line_count, 0)
            view_status = screen.query_one("#monitor-view-status", Static)
            self.assertIn("FOLLOWING", str(view_status.content))
            self.assertIn("0 lines", str(view_status.content))

    async def test_operation_messages_render_without_markup(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(MonitorScreen(title="Deploy"))
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, MonitorScreen)
            screen.on_operation_done(
                OperationDone(
                    operation=OperationType.DEPLOY,
                    success=False,
                    exit_code=9,
                    detail="backend error",
                )
            )
            screen.on_operation_error(OperationError("worker failed"))
            await pilot.pause()

            content = "\n".join(screen.log_viewer.log_widget.lines)
            self.assertIn("Operation failed (exit code 9).", content)
            self.assertIn("Detail: backend error", content)
            self.assertIn("Press esc or q to return, or enter to retry.", content)
            self.assertIn("Error: worker failed", content)
            self.assertNotIn("[red", content)
            self.assertNotIn("[green", content)
            self.assertNotIn("[dim]", content)

    async def test_summary_mode_normalizes_backend_log_lines(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(
                MonitorScreen(
                    title="Deploy",
                    deploy_backend=BackendType.VLLM,
                    summarize_backend_logs=True,
                    show_debug_logs=False,
                )
            )
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, MonitorScreen)
            screen.on_state_changed(
                StateChanged(
                    state=DeploymentState.WARMING_UP,
                    operation=OperationType.WARMUP,
                    detail="warming",
                )
            )
            screen.on_log_message(LogMessage("(APIServer pid=4) INFO 02-25 03:08:40 vLLM API server version 0.13.0"))
            await pilot.pause()

            content = "\n".join(screen.log_viewer.log_widget.lines)
            self.assertIn("Preparing deployment", content)
            self.assertIn("warming", content)
            self.assertIn("Starting server", content)
            self.assertNotIn("vLLM API server version", content)
            self.assertNotIn("Log view: summary", content)
            self.assertEqual(screen.status_header.backend, "vllm")
            self.assertEqual(screen.status_header.state, "warming_up")
            self.assertIs(screen.status_header.parent, screen.query_one("#monitor-layout"))
            title = screen.query_one("#monitor-title", Static)
            self.assertGreater(screen.status_header.region.height, 0)
            self.assertLess(screen.status_header.region.y, title.region.y)

    async def test_summary_mode_shows_explicit_milestones_but_hides_raw_logs(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(
                MonitorScreen(
                    title="Deploy",
                    deploy_backend=BackendType.VLLM,
                    summarize_backend_logs=True,
                    show_debug_logs=False,
                )
            )
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, MonitorScreen)
            screen.on_log_message(
                LogMessage(
                    "Selected provider offer: 1x GPU ($0.50/hr)",
                    is_milestone=True,
                )
            )
            screen.on_log_message(LogMessage("unclassified backend chatter"))
            await pilot.pause()

            content = "\n".join(screen.log_viewer.log_widget.lines)
            self.assertIn("GPU ready:", content)
            self.assertIn("1× GPU", content)
            self.assertIn("$0.50/hr", content)
            self.assertNotIn("Selected provider offer:", content)
            self.assertNotIn("unclassified backend chatter", content)

            screen.action_toggle_log_view()
            await pilot.pause()
            raw_content = "\n".join(screen.log_viewer.log_widget.lines)
            self.assertIn("unclassified backend chatter", raw_content)
            self.assertIn("RAW", str(screen.query_one("#monitor-view-status", Static).content))

            screen.action_toggle_log_view()
            await pilot.pause()
            summary_content = "\n".join(screen.log_viewer.log_widget.lines)
            self.assertNotIn("unclassified backend chatter", summary_content)

    async def test_log_search_tracks_matches_and_preserves_query(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(MonitorScreen(title="Logs"))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MonitorScreen)
            for line in ("starting", "error one", "healthy", "error two"):
                screen.on_log_message(LogMessage(line))
            await pilot.pause()

            screen.action_search_logs()
            search = screen.query_one("#monitor-search", Input)
            search.value = "error"
            await pilot.pause()

            self.assertTrue(search.display)
            self.assertEqual(screen._search_total, 2)
            self.assertEqual(screen._search_current, 1)
            screen.action_next_search_match()
            await pilot.pause()
            self.assertEqual(screen._search_current, 2)
            self.assertIn("/error 2/2", str(screen.query_one("#monitor-view-status", Static).content))

    async def test_debug_mode_preserves_raw_backend_log_lines(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(
                MonitorScreen(
                    title="Deploy",
                    deploy_backend=BackendType.VLLM,
                    summarize_backend_logs=True,
                    show_debug_logs=True,
                )
            )
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, MonitorScreen)
            raw_line = "(APIServer pid=4) INFO 02-25 03:08:40 vLLM API server version 0.13.0"
            screen.on_log_message(LogMessage(raw_line))
            await pilot.pause()

            content = "\n".join(screen.log_viewer.log_widget.lines)
            self.assertIn(raw_line, content)
            self.assertNotIn("Preparing deployment", content)

    async def test_debug_mode_preserves_ansi_escape_sequences(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(
                MonitorScreen(
                    title="Deploy",
                    deploy_backend=BackendType.VLLM,
                    summarize_backend_logs=True,
                    show_debug_logs=True,
                )
            )
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, MonitorScreen)
            raw_line = "\x1b[0;36m(APIServer pid=4)\x1b[0;0m INFO booted"
            with patch.object(screen.log_viewer, "write_line") as write_line:
                screen.on_log_message(LogMessage(raw_line))

            write_line.assert_called_once_with(raw_line)

    async def test_summary_mode_failure_appends_debug_hint(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(
                MonitorScreen(
                    title="Deploy",
                    deploy_backend=BackendType.LLAMACPP,
                    summarize_backend_logs=True,
                    show_debug_logs=False,
                )
            )
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, MonitorScreen)
            screen.on_operation_done(
                OperationDone(
                    operation=OperationType.WARMUP,
                    success=False,
                    exit_code=1,
                    detail="timed out",
                )
            )
            await pilot.pause()

            content = "\n".join(screen.log_viewer.log_widget.lines)
            self.assertIn('Tip: re-run with "Show debug logs" enabled to see full backend logs.', content)

    async def test_log_widget_is_selectable_log(self) -> None:
        """LogViewer should use SelectableLog (not plain Log) for line selection."""
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(MonitorScreen(title="Logs"))
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, MonitorScreen)
            self.assertIsInstance(screen.log_viewer.log_widget, SelectableLog)

    async def test_double_click_line_selection_copies_line_to_clipboard(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(MonitorScreen(title="Logs"))
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, MonitorScreen)
            screen.on_log_message(LogMessage("first line"))
            await pilot.pause()

            copied = screen.log_viewer.log_widget._select_line(0)
            self.assertTrue(copied)
            await pilot.pause()
            await pilot.pause()

            self.assertEqual(app.clipboard, "first line")

    def test_copy_binding_includes_terminal_safe_variants(self) -> None:
        copy_binding = next(b for b in MonitorScreen.BINDINGS if b.action == "copy_text")
        self.assertEqual(copy_binding.key, "y")

    def test_direct_clipboard_binding_keeps_terminal_copy_aliases(self) -> None:
        copy_binding = next(
            b for b in MonitorScreen.BINDINGS if b.action == "copy_text_to_clipboard"
        )
        self.assertEqual(
            copy_binding.key,
            "ctrl+shift+c,super+c,meta+c,cmd+c,command+c",
        )

    async def test_connection_card_shows_copy_actions_and_returns_home(self) -> None:
        app = _HomeApp()
        async with app.run_test() as pilot:
            app.push_screen(MonitorScreen(title="Deploy"))
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, MonitorScreen)
            card = screen.query_one("#connection-card")
            self.assertTrue(card.has_class("hidden"))

            screen.on_connection_summary_ready(
                ConnectionSummaryReady(
                    {
                        "base_url": "https://example.modal.run/v1",
                        "model_id": "Qwen3-4B",
                        "display_name": "Qwen3-4B",
                        "api_key": "sk-test-key",
                    }
                )
            )
            screen.on_operation_done(
                OperationDone(operation=OperationType.DEPLOY, success=True)
            )
            await pilot.pause()

            self.assertFalse(card.has_class("hidden"))
            body = str(screen.query_one("#connection-card-body", Static).content)
            self.assertIn("https://example.modal.run/v1", body)
            self.assertIn("sk-test-key", body)
            self.assertTrue(screen.query_one("#copy-key-btn", Button).display)
            content = "\n".join(screen.log_viewer.log_widget.lines)
            self.assertIn("Press enter or esc to return home.", content)

            screen.action_copy_base_url()
            self.assertEqual(app.clipboard, "https://example.modal.run/v1")
            screen.action_copy_connection()
            self.assertIn("API key: sk-test-key", app.clipboard)
            self.assertEqual(
                _connection_copy_text(
                    {
                        "base_url": "https://example.modal.run/v1",
                        "model_id": "Qwen3-4B",
                        "display_name": "Qwen3-4B",
                        "api_key": "sk-test-key",
                    }
                ).splitlines()[0],
                "Base URL: https://example.modal.run/v1",
            )

            screen.action_finish_success()
            await pilot.pause()
            self.assertEqual(app.home_calls, 1)

    async def test_failed_operation_does_not_open_connection_card(self) -> None:
        app = _HomeApp()
        async with app.run_test() as pilot:
            app.push_screen(MonitorScreen(title="Deploy"))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MonitorScreen)
            screen.on_operation_done(
                OperationDone(operation=OperationType.DEPLOY, success=False, exit_code=1)
            )
            await pilot.pause()
            self.assertTrue(screen.query_one("#connection-card").has_class("hidden"))
            screen.action_go_back()
            await pilot.pause()
            self.assertEqual(app.home_calls, 0)

    async def test_failed_operation_enter_pops_back_for_retry(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(Screen())
            app.push_screen(MonitorScreen(title="Status Check"))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MonitorScreen)
            self.assertEqual(len(app.screen_stack), 3)
            screen.on_operation_done(
                OperationDone(operation=OperationType.STATUS, success=False, exit_code=1)
            )
            await pilot.pause()

            screen.action_finish_success()
            await pilot.pause()

            self.assertEqual(len(app.screen_stack), 2)

    async def test_status_success_shows_result_card_with_curl(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(MonitorScreen(title="Status Check"))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MonitorScreen)
            card = screen.query_one("#result-card")
            self.assertTrue(card.has_class("hidden"))

            screen.on_state_changed(
                StateChanged(DeploymentState.RUNNING, operation=OperationType.STATUS)
            )
            screen.on_log_message(LogMessage("Status: healthy (backend=vllm, url=https://x.test)"))
            screen.on_log_message(LogMessage("Test command:\ncurl https://x.test/v1/chat"))
            screen.on_operation_done(
                OperationDone(operation=OperationType.STATUS, success=True)
            )
            await pilot.pause()

            self.assertFalse(card.has_class("hidden"))
            body = str(screen.query_one("#result-card-body", Static).content)
            self.assertIn("Healthy", body)
            self.assertIn("curl https://x.test/v1/chat", body)

    async def test_connection_card_manage_button_routes_to_manage(self) -> None:
        class _RouteApp(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self.home_calls = 0
                self.manage_calls = 0

            def pop_to_main_menu(self) -> None:
                self.home_calls += 1
                self.pop_screen()

            def action_push_manage(self) -> None:
                self.manage_calls += 1

        app = _RouteApp()
        async with app.run_test() as pilot:
            app.push_screen(MonitorScreen(title="Deploy"))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MonitorScreen)
            screen.on_connection_summary_ready(
                ConnectionSummaryReady(
                    {
                        "base_url": "https://example.modal.run/v1",
                        "model_id": "Qwen3-4B",
                        "display_name": "Qwen3-4B",
                    }
                )
            )
            screen.on_operation_done(
                OperationDone(operation=OperationType.DEPLOY, success=True)
            )
            await pilot.pause()

            screen.action_open_manage()

        self.assertEqual(app.home_calls, 1)
        self.assertEqual(app.manage_calls, 1)

    async def test_connection_card_stays_hidden_until_operation_succeeds(self) -> None:
        app = _HomeApp()
        async with app.run_test() as pilot:
            app.push_screen(MonitorScreen(title="Deploy"))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MonitorScreen)
            screen.on_connection_summary_ready(
                ConnectionSummaryReady(
                    {
                        "base_url": "https://example.modal.run/v1",
                        "model_id": "Qwen3-4B",
                        "display_name": "Qwen3-4B",
                        "api_key": "sk-test-key",
                    }
                )
            )
            await pilot.pause()
            self.assertTrue(screen.query_one("#connection-card").has_class("hidden"))
            screen.action_finish_success()
            await pilot.pause()
            self.assertEqual(app.home_calls, 0)

            screen.on_operation_done(
                OperationDone(operation=OperationType.DEPLOY, success=True)
            )
            await pilot.pause()
            self.assertFalse(screen.query_one("#connection-card").has_class("hidden"))
            screen.action_finish_success()
            await pilot.pause()
            self.assertEqual(app.home_calls, 1)

    async def test_summary_mode_marks_finished_steps_and_updates_progress_in_place(
        self,
    ) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(
                MonitorScreen(
                    title="Deploy",
                    deploy_backend=BackendType.LLAMACPP,
                    summarize_backend_logs=True,
                    show_debug_logs=False,
                )
            )
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MonitorScreen)
            screen.on_log_message(
                LogMessage("Prime pod state: ACTIVE/FINISHED", is_milestone=True)
            )
            screen.on_log_message(
                LogMessage(
                    "🦙 download in progress... elapsed=20s files=1 size=2.28GiB/10.00GiB "
                    "pct=22% complete=0 inflight=1 avg_rate=12.34MiB/s"
                )
            )
            screen.on_log_message(
                LogMessage(
                    "🦙 download in progress... elapsed=40s files=1 size=3.28GiB/10.00GiB "
                    "pct=32% complete=0 inflight=1 avg_rate=12.34MiB/s"
                )
            )
            await pilot.pause()

            lines = list(screen.log_viewer.log_widget.lines)
            downloading = [line for line in lines if "Downloading model" in line]
            self.assertEqual(len(downloading), 1)
            self.assertIn("(32%)", downloading[0])
            self.assertTrue(
                downloading[0].startswith(tuple(f"{frame} " for frame in SUMMARY_SPINNER_FRAMES))
            )
            preparing = [line for line in lines if "Preparing deployment" in line]
            self.assertEqual(preparing, ["✓ Preparing deployment"])
            self.assertIn("✓ Machine ready", lines)

    def test_footer_includes_log_navigation_bindings(self) -> None:
        visible_bindings = {
            binding.key: binding.description
            for binding in MonitorScreen.BINDINGS
            if binding.show
        }
        self.assertEqual(visible_bindings["pageup"], "Page up")
        self.assertEqual(visible_bindings["pagedown"], "Page down")
        self.assertEqual(visible_bindings["end"], "Follow")


if __name__ == "__main__":
    unittest.main()
