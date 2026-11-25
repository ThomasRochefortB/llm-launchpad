"""Tests for the offline demo TUI mode used for local UI iteration."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from textual.widgets import OptionList, Static, TabbedContent

from llm_launchpad.cli import main as cli_main
from llm_launchpad.tui.demo import DemoTuiApp
from llm_launchpad.tui.screens.main_menu import MainMenuScreen
from llm_launchpad.tui.screens.manage import (
    ConnectionInfoScreen,
    EndpointActionsScreen,
    ManageScreen,
    StopConfirmScreen,
)
from llm_launchpad.tui.screens.monitor import MonitorScreen
from llm_launchpad.tui.widgets.adaptive_table import AdaptiveDataTable


class DemoTuiAppTests(unittest.IsolatedAsyncioTestCase):
    async def test_demo_tui_opens_directly_on_manage_and_renders_fake_fleet(self) -> None:
        app = DemoTuiApp(open_manage=True)
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()

            screen = app.screen
            self.assertIsInstance(screen, ManageScreen)
            assert isinstance(screen, ManageScreen)
            table = screen.query_one("#manage-endpoint-table", AdaptiveDataTable)
            # 3 active demo rows (Prime running, Modal ready, Vast starting); 1 stopped hidden.
            self.assertEqual(table.row_count, 3)
            status = str(screen.query_one("#manage-status", Static).renderable)
            self.assertIn("3 endpoints", status)
            self.assertIn("1 stopped hidden", status)

            # Pressing 'a' reveals the stopped Modal deployment.
            await pilot.press("a")
            await pilot.pause()
            self.assertEqual(table.row_count, 4)

            # Pressing Escape returns to Home underneath.
            await pilot.press("escape")
            await pilot.pause()
            self.assertIsInstance(app.screen, MainMenuScreen)

    async def test_demo_manage_actions_work_without_cloud_calls(self) -> None:
        app = DemoTuiApp(open_manage=True)
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ManageScreen)
            # Sorted alphabetically: deepseek-r1-32b (Vast starting), glm-4-7-flash (Modal ready), qwen3-32b (Prime running).
            await pilot.press("down")  # select glm-4-7-flash (Modal)
            await pilot.pause()

            # Open EndpointActionsScreen -> Connection info.
            await pilot.press("enter")
            await pilot.pause()
            self.assertIsInstance(app.screen, EndpointActionsScreen)
            await pilot.press("enter")
            await pilot.pause()
            self.assertIsInstance(app.screen, ConnectionInfoScreen)
            await pilot.press("escape")
            await pilot.pause()

            # Explicit live metrics fetch wakes the fake Modal endpoint and advances tokens.
            self.assertIsInstance(app.screen, ManageScreen)
            await pilot.press("f")
            await pilot.pause()
            await app.screen.workers.wait_for_complete()
            await pilot.pause()
            detail = str(app.screen.query_one("#manage-selection-detail", Static).renderable)
            self.assertIn("65K tokens served", detail)

            # Check status probes the fake Modal endpoint and marks it Healthy on return.
            await pilot.press("enter")
            await pilot.pause()
            self.assertIsInstance(app.screen, EndpointActionsScreen)
            await pilot.press("down")  # Check status — may start GPU
            await pilot.press("enter")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertIsInstance(app.screen, MonitorScreen)
            await pilot.press("escape")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            detail_after_status = str(
                app.screen.query_one("#manage-selection-detail", Static).renderable
            )
            self.assertIn("Healthy", detail_after_status)

            # View logs opens MonitorScreen and streams demo logs.
            await pilot.press("l")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertIsInstance(app.screen, MonitorScreen)
            await pilot.press("escape")
            await pilot.pause()

            # Stop endpoint marks the fake endpoint stopped and removes it from active rows.
            self.assertIsInstance(app.screen, ManageScreen)
            await pilot.press("x")
            await pilot.pause()
            self.assertIsInstance(app.screen, StopConfirmScreen)
            await pilot.press("x")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertIsInstance(app.screen, MonitorScreen)
            await pilot.press("escape")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()

            manage = app.screen
            assert isinstance(manage, ManageScreen)
            table = manage.query_one("#manage-endpoint-table", AdaptiveDataTable)
            self.assertEqual(table.row_count, 2)

    async def test_demo_jobs_tab_lists_and_reopens_fake_jobs(self) -> None:
        app = DemoTuiApp(open_manage=True)
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()

            manage = app.screen
            assert isinstance(manage, ManageScreen)
            manage.query_one("#manage-tabs", TabbedContent).active = "manage-jobs"
            await pilot.pause()
            await manage.workers.wait_for_complete()
            await pilot.pause()

            jobs_list = manage.query_one("#deployment-jobs", OptionList)
            self.assertEqual(jobs_list.option_count, 3)

            # Selecting the first job reopens its MonitorScreen with history.
            await pilot.press("enter")
            await pilot.pause()
            self.assertIsInstance(app.screen, MonitorScreen)


class DemoCliWiringTests(unittest.TestCase):
    def test_tui_demo_flag_constructs_demo_app_without_modal_cleanup(self) -> None:
        demo_instance = Mock()
        demo_instance.return_code = 0
        with (
            patch("llm_launchpad.cli.main.sys.stdin.isatty", return_value=True),
            patch("llm_launchpad.cli.main.sys.stdout.isatty", return_value=True),
            patch("llm_launchpad.tui.demo.DemoTuiApp", return_value=demo_instance) as demo_cls,
            patch("llm_launchpad.core.backend.ModalBackend.terminate_all") as terminate_all,
        ):
            cli_main.tui(demo=True, manage=True)

        demo_cls.assert_called_once_with(mouse_enabled=None, open_manage=True)
        demo_instance.run.assert_called_once_with(mouse=False)
        terminate_all.assert_not_called()

    def test_tui_manage_without_demo_fails_fast(self) -> None:
        with self.assertRaises(cli_main.typer.BadParameter):
            cli_main.tui(demo=False, manage=True)


if __name__ == "__main__":
    unittest.main()
