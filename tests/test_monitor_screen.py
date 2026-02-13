from __future__ import annotations

import unittest

from textual.app import App
from textual.widgets import Log, Static

from llm_launchpad.protocol.enums import OperationType
from llm_launchpad.tui.screens.monitor import MonitorScreen
from llm_launchpad.tui.widgets.log_viewer import SelectableLog
from llm_launchpad.tui.workers import LogMessage, OperationDone, OperationError


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
            self.assertIn("cmd+c/ctrl+c/c to copy  ctrl+l clear", str(title.content))

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

    async def test_log_widget_is_selectable_log(self) -> None:
        """LogViewer should use SelectableLog (not plain Log) for line selection."""
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(MonitorScreen(title="Logs"))
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, MonitorScreen)
            self.assertIsInstance(screen.log_viewer.log_widget, SelectableLog)

    def test_copy_binding_includes_ctrl_and_cmd_variants(self) -> None:
        copy_binding = next(b for b in MonitorScreen.BINDINGS if b.action == "copy_text")
        self.assertIn("ctrl+c", copy_binding.key)
        self.assertIn("meta+c", copy_binding.key)
        self.assertIn("super+c", copy_binding.key)


if __name__ == "__main__":
    unittest.main()
