"""Operations refreshes must not block input or erase known results."""

import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from textual.app import App
from textual.widgets import OptionList, Static, TabbedContent

from llm_launchpad.tui.app import TuiApp
from llm_launchpad.tui.screens.manage import ManageScreen
from llm_launchpad.tui.screens.operations import (
    DeploymentJobsPanel,
    OperationsScreen,
    PersistentCancelDeploymentScreen,
)


class OperationsApp(App):
    def __init__(self, store: object) -> None:
        super().__init__()
        self.store = store
        self.deployment_jobs = {}

    def _get_job_store(self) -> object:
        return self.store

    def on_mount(self) -> None:
        self.push_screen(OperationsScreen())


def record(job_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=job_id, app_name=job_id, outcome="running",
        config=SimpleNamespace(provider=SimpleNamespace(display_name="Modal")),
    )


class OperationsResponsivenessTests(unittest.IsolatedAsyncioTestCase):
    async def test_manage_jobs_reopen_cancel_and_keep_endpoint_actions_scoped(self) -> None:
        class ManageApp(App):
            CSS_PATH = TuiApp.CSS_PATH
            deployment_jobs = {}

            def _get_job_store(self) -> object:
                return SimpleNamespace(list_jobs=lambda: [record("saved")])

            def on_mount(self) -> None:
                self.push_screen(ManageScreen())

        app = ManageApp()
        app.reopen_persistent_deployment = Mock()
        with patch.object(ManageScreen, "_refresh_endpoints"):
            async with app.run_test(size=(80, 24)) as pilot:
                await app.workers.wait_for_complete()
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, ManageScreen)
                self.assertTrue(screen.check_action("stop_selected", ()))

                # The old global shortcut opens Jobs in this same Manage screen.
                stack_size = len(app.screen_stack)
                TuiApp.action_push_operations(app)
                await pilot.pause()
                self.assertEqual(len(app.screen_stack), stack_size)
                self.assertEqual(screen.query_one(TabbedContent).active, "manage-jobs")
                self.assertFalse(screen.check_action("stop_selected", ()))
                self.assertFalse(screen.check_action("open_actions", ()))
                self.assertIs(app.focused, screen.query_one("#deployment-jobs", OptionList))
                self.assertGreater(app.focused.region.height, 0)
                self.assertLess(app.focused.region.bottom, app.size.height)

                await pilot.press("enter")
                app.reopen_persistent_deployment.assert_called_once_with("saved")
                await pilot.press("x")
                self.assertIsInstance(app.screen, PersistentCancelDeploymentScreen)
                await pilot.press("escape")
                self.assertIs(app.screen, screen)

                screen.query_one(TabbedContent).active = "manage-endpoints"
                await pilot.pause()
                self.assertTrue(screen.check_action("stop_selected", ()))
                self.assertEqual(app.focused.id, "manage-endpoint-table")

    async def test_slow_refresh_keeps_input_live_and_coalesces_reconciliation(self) -> None:
        started = threading.Event()
        release = threading.Event()
        calls: list[str] = []
        main_thread = threading.get_ident()

        class Store:
            def reconcile_workers(self) -> None:
                calls.append("reconcile")

            def list_jobs(self) -> list[object]:
                assert threading.get_ident() != main_thread
                calls.append("list")
                started.set()
                release.wait(timeout=5)
                return [record("saved")]

        app = OperationsApp(Store())
        try:
            async with app.run_test() as pilot:
                await asyncio.to_thread(started.wait, 2)
                screen = app.screen.query_one(DeploymentJobsPanel)
                for _ in range(5):
                    screen._refresh_jobs(reconcile=True)
                await pilot.press("tab")
                self.assertFalse(release.is_set())
                self.assertEqual(calls, ["reconcile", "list"])
                self.assertNotIn("No deployments", str(screen.query_one("#operations-status", Static).content))
                release.set()
                await app.workers.wait_for_complete()
                await pilot.pause()
                await app.workers.wait_for_complete()
                await pilot.pause()
                self.assertEqual(calls.count("reconcile"), 2)
                self.assertEqual(screen.query_one(OptionList).get_option_at_index(0).id, "persistent:saved")
        finally:
            release.set()

    async def test_failure_retains_rows_and_recovery_preserves_selected_identity(self) -> None:
        class Store:
            rows = [record("one"), record("two")]
            fail = False

            def list_jobs(self) -> list[object]:
                if self.fail:
                    raise OSError("database busy")
                return self.rows

        store = Store()
        app = OperationsApp(store)
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            screen = app.screen.query_one(DeploymentJobsPanel)
            options = screen.query_one(OptionList)
            options.highlighted = 1
            store.fail = True
            screen._refresh_jobs()
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertEqual(options.option_count, 2)
            self.assertIn("Showing last saved results", str(screen.query_one("#operations-status", Static).content))
            store.fail = False
            store.rows = [record("new"), record("one"), record("two")]
            screen._refresh_jobs()
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertEqual(options.highlighted_option.id, "persistent:two")
            self.assertEqual(str(screen.query_one("#operations-status", Static).content), "")
