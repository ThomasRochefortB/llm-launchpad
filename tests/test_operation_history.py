"""Status checks, benchmarks and storage work stay reachable from Operations."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from textual.screen import Screen
from textual.widgets import OptionList

from llm_launchpad.tui.app import TuiApp
from llm_launchpad.tui.screens.monitor import MonitorScreen, OperationDone
from llm_launchpad.tui.screens.operations import OperationsScreen
from llm_launchpad.protocol.enums import OperationType


class _App(TuiApp):
    def on_mount(self) -> None:
        self.push_screen(Screen())


class OperationHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_finished_status_check_can_be_reopened(self) -> None:
        """Leaving a status check used to discard its result for good."""
        app = _App()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            monitor = MonitorScreen(title="Status Check")
            app.push_screen(app._track_operation(monitor, "vllm-demo"))
            await pilot.pause()
            monitor.on_operation_done(OperationDone(OperationType.STATUS, True))
            await pilot.pause()
            app.pop_screen()
            await pilot.pause()

            with patch.object(OperationsScreen, "on_mount", lambda self: None):
                app.push_screen(OperationsScreen())
                await pilot.pause()
            panel_list = app.screen.query_one("#deployment-jobs", OptionList)
            panel = panel_list.parent
            panel._render_jobs()  # type: ignore[union-attr]
            await pilot.pause()
            prompts = [
                str(panel_list.get_option_at_index(index).prompt)
                for index in range(panel_list.option_count)
            ]
            self.assertIn("Status Check · vllm-demo · Done", prompts)

            op_id = next(iter(app.operation_history))
            app.reopen_operation(op_id)
            await pilot.pause()
            self.assertIs(app.screen, monitor)

    async def test_only_deployments_can_be_cancelled(self) -> None:
        app = _App()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.push_screen(app._track_operation(MonitorScreen(title="Logs"), "vllm-demo"))
            await pilot.pause()
            app.pop_screen()
            with patch.object(OperationsScreen, "on_mount", lambda self: None):
                app.push_screen(OperationsScreen())
                await pilot.pause()
            options = app.screen.query_one("#deployment-jobs", OptionList)
            panel = options.parent
            panel._render_jobs()  # type: ignore[union-attr]
            await pilot.pause()
            options.highlighted = 0
            with patch.object(app, "notify") as notify:
                panel.action_cancel_selected()  # type: ignore[union-attr]
            notify.assert_called_once()
            self.assertIn("Only deployments", notify.call_args.args[0])

    async def test_the_history_is_bounded(self) -> None:
        app = _App()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            for index in range(app._OPERATION_HISTORY_LIMIT + 5):
                app.push_screen(app._track_operation(MonitorScreen(title="Logs"), f"e{index}"))
                await pilot.pause()
                app.pop_screen()
            await pilot.pause()
            self.assertEqual(len(app.operation_history), app._OPERATION_HISTORY_LIMIT)
            subjects = [record.subject for record in app.operation_history.values()]
            self.assertEqual(subjects[0], "e5")


if __name__ == "__main__":
    unittest.main()
