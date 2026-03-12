from __future__ import annotations

import unittest

from textual.app import App
from textual.widgets import Log, Static

from llm_launchpad.protocol.enums import BackendType, DeploymentState, OperationType
from llm_launchpad.tui.screens.monitor import MonitorScreen
from llm_launchpad.tui.widgets.log_viewer import SelectableLog
from llm_launchpad.tui.workers import LogMessage, OperationDone, OperationError, StateChanged


class _TestApp(App[None]):
    pass


class MonitorScreenTests(unittest.IsolatedAsyncioTestCase):
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
            self.assertIn(
                "ctrl+shift+c copy  y fallback  ctrl+c exits  ctrl+l clear",
                str(title.content),
            )

    async def test_title_mentions_terminal_selection_when_mouse_disabled(self) -> None:
        app = _TestApp()
        app.mouse_enabled = False
        async with app.run_test() as pilot:
            app.push_screen(MonitorScreen(title="Logs"))
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, MonitorScreen)
            title = screen.query_one("#monitor-title", Static)
            self.assertIn(
                "terminal selection mode  use ctrl+shift+c to copy  ctrl+c exits",
                str(title.content),
            )

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
            self.assertIn("Press esc or q to return.", content)
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
            self.assertIn("Log view: summary (normalized milestones; raw backend logs hidden)", content)
            self.assertIn("Starting server", content)
            self.assertNotIn("vLLM API server version", content)

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
            self.assertNotIn("Log view: summary (normalized milestones; raw backend logs hidden)", content)

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


if __name__ == "__main__":
    unittest.main()
