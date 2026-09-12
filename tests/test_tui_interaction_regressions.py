"""Regressions from driving the TUI rather than reading it.

These cover focus, failure reporting, and what a background refresh is allowed
to do to work the user is in the middle of -- none of which is visible in a
rendered screenshot.
"""

from __future__ import annotations

import unittest
from contextlib import ExitStack
from unittest.mock import patch

from textual.app import App
from textual.containers import ScrollableContainer
from textual.scroll_view import ScrollView
from textual.widgets import Input, Static

from llm_launchpad.protocol.enums import BackendType, ComputeProvider, OperationType
from llm_launchpad.protocol.models import EndpointInfo
from llm_launchpad.tui.app import TuiApp
from llm_launchpad.tui.screens.deploy import LlamaCppDeployScreen, VllmDeployScreen
from llm_launchpad.tui.screens.main_menu import MainMenuScreen
from llm_launchpad.tui.screens.manage import ManageScreen
from llm_launchpad.tui.screens.monitor import MonitorScreen
from llm_launchpad.tui.screens.settings import SettingsScreen
from llm_launchpad.tui.screens.setup import SetupRequiredScreen
from llm_launchpad.tui.screens.storage import StorageScreen
from llm_launchpad.tui.widgets.status_header import StatusHeader
from llm_launchpad.tui.workers import EndpointsLoaded, OperationDone, StorageLoaded
from tests.test_storage_screen import _sample_snapshot


class _StyledApp(App[None]):
    CSS_PATH = TuiApp.CSS_PATH


def _focus_cycle(app: App[object], limit: int = 30) -> list[str]:
    """Widget ids reachable by repeated Tab, in order, until it wraps."""
    order: list[str] = []
    seen: set[int] = set()
    for _ in range(limit):
        focused = app.focused
        if focused is None or id(focused) in seen:
            break
        seen.add(id(focused))
        order.append(f"{type(focused).__name__}#{focused.id or '-'}")
        app.screen.focus_next()
    return order


class ScrollContainerFocusTests(unittest.IsolatedAsyncioTestCase):
    """Every form screen had a tab stop where nothing appeared focused.

    Textual makes VerticalScroll focusable so a keyboard user can scroll a
    region with nothing in it to focus. A form is not that region.
    """

    SCREENS = (
        ("MainMenu", lambda: MainMenuScreen(username="probe")),
        ("LlamaCppDeploy", LlamaCppDeployScreen),
        ("VllmDeploy", VllmDeployScreen),
        ("Storage", StorageScreen),
        ("Settings", SettingsScreen),
        ("SetupRequired", SetupRequiredScreen),
    )

    async def test_no_scroll_container_sits_in_the_tab_cycle(self) -> None:
        for name, factory in self.SCREENS:
            with self.subTest(screen=name):
                app = _StyledApp()
                # The menu's on_mount is what focuses its list, so it has to
                # run; only its network probes are stubbed.
                with ExitStack() as stack:
                    for method in (
                        "_refresh_modal_auth_status", "_refresh_prime_auth_status",
                        "_refresh_hf_auth_status", "_refresh_aai_auth_status",
                        "_refresh_panels", "_refresh_quick_deploy_catalog",
                        "_refresh_storage_estimate", "_refresh_billing_panels",
                    ):
                        stack.enter_context(patch.object(MainMenuScreen, method))
                    stack.enter_context(
                        patch.object(StorageScreen, "_refresh_storage_snapshot", lambda self: None)
                    )
                    async with app.run_test(size=(140, 45)) as pilot:
                        app.push_screen(factory())
                        await pilot.pause()
                        await pilot.pause()
                        containers = {
                            f"{type(c).__name__}#{c.id or '-'}"
                            for c in app.screen.query(ScrollableContainer)
                            if not isinstance(c, ScrollView)
                        }
                        cycle = _focus_cycle(app)
                        self.assertTrue(cycle, f"{name} has no focusable widget")
                        self.assertFalse(
                            containers & set(cycle),
                            f"{name} tabs into a scroll container: {cycle}",
                        )

    async def test_a_scroll_region_with_nothing_to_focus_stays_reachable(self) -> None:
        """The exemption has to keep working where it earns its place."""
        app = _StyledApp()
        async with app.run_test(size=(80, 24)) as pilot:
            container = ScrollableContainer(Static("just text"))
            await app.screen.mount(container)
            await pilot.pause()
            await pilot.pause()
            self.assertTrue(container.can_focus)


class SettingsFocusTests(unittest.IsolatedAsyncioTestCase):
    async def test_it_opens_on_its_first_field(self) -> None:
        app = _StyledApp()
        async with app.run_test(size=(140, 45)) as pilot:
            app.push_screen(SettingsScreen())
            await pilot.pause()
            await pilot.pause()
            self.assertEqual(getattr(app.focused, "id", None), "scaledown-window")


class StatusHeaderFailureTests(unittest.IsolatedAsyncioTestCase):
    """A failed operation left the one status line reading `state: idle`."""

    async def test_a_failed_operation_is_named_in_the_status_bar(self) -> None:
        app = _StyledApp()
        async with app.run_test(size=(120, 34)) as pilot:
            app.push_screen(MonitorScreen(title="Deploy"))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MonitorScreen)
            header = screen.query_one("#monitor-status-header", StatusHeader)
            screen.on_operation_done(
                OperationDone(operation=OperationType.DEPLOY, success=False, exit_code=1)
            )
            await pilot.pause()

            self.assertTrue(header.failed)
            rendered = str(header.render())
            self.assertIn("failed", rendered)
            self.assertTrue(header.has_class("state-error"))

    async def test_the_state_the_deployment_reached_is_kept(self) -> None:
        """Both facts matter: where it got to, and that the run failed."""
        from llm_launchpad.protocol.enums import DeploymentState

        app = _StyledApp()
        async with app.run_test(size=(120, 34)) as pilot:
            app.push_screen(MonitorScreen(title="Deploy"))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, MonitorScreen)
            header = screen.query_one("#monitor-status-header", StatusHeader)
            header.update_from_event(
                state=DeploymentState.DEPLOYING,
                operation=OperationType.DEPLOY,
                detail="Provisioning machine",
            )
            screen.on_operation_done(
                OperationDone(operation=OperationType.DEPLOY, success=False, exit_code=1)
            )
            await pilot.pause()

            self.assertEqual(header.state, "deploying")
            self.assertIn("deploying", str(header.render()))
            self.assertIn("failed", str(header.render()))
            # The detail described work that has stopped.
            self.assertEqual(header.detail, "")

    async def test_a_new_attempt_clears_the_previous_failure(self) -> None:
        from llm_launchpad.protocol.enums import DeploymentState

        header = StatusHeader()
        app = _StyledApp()
        async with app.run_test(size=(120, 34)) as pilot:
            await app.screen.mount(header)
            await pilot.pause()
            header.report_failure()
            self.assertTrue(header.failed)
            header.update_from_event(state=DeploymentState.DEPLOYING)
            await pilot.pause()
            self.assertFalse(header.failed)
            self.assertNotIn("failed", str(header.render()))

    def test_every_state_marker_stays_safe_to_interpolate(self) -> None:
        from rich.markup import render as render_markup

        from llm_launchpad.tui.widgets.status_header import FAILED_STATE, _state_icon

        self.assertEqual(render_markup(_state_icon(FAILED_STATE)).plain.strip(), "XX")


class StorageRefreshTests(unittest.IsolatedAsyncioTestCase):
    """A snapshot refresh arrives on a timer, not because the user asked."""

    async def _storage(self, pilot, app) -> StorageScreen:
        screen = StorageScreen()
        with patch.object(StorageScreen, "_refresh_storage_snapshot", lambda self: None):
            app.push_screen(screen)
            await pilot.pause()
        return screen

    async def test_a_refresh_does_not_overwrite_what_is_being_typed(self) -> None:
        app = _StyledApp()
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await self._storage(pilot, app)
            screen.on_storage_loaded(StorageLoaded(_sample_snapshot()))
            await pilot.pause()

            model_id = screen.query_one("#storage-model-id", Input)
            model_id.focus()
            model_id.value = "org/half-typed"
            await pilot.pause()

            screen.on_storage_loaded(StorageLoaded(_sample_snapshot()))
            await pilot.pause()

            self.assertEqual(model_id.value, "org/half-typed")

    async def test_the_first_snapshot_still_prefills_the_form(self) -> None:
        """Prefilling is what makes p and x act on the highlighted row."""
        app = _StyledApp()
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await self._storage(pilot, app)
            screen.on_storage_loaded(StorageLoaded(_sample_snapshot()))
            await pilot.pause()

            self.assertTrue(screen.query_one("#storage-model-id", Input).value)
            self.assertIsNotNone(screen._selected_model)

    async def test_moving_to_another_row_still_prefills(self) -> None:
        app = _StyledApp()
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await self._storage(pilot, app)
            screen.on_storage_loaded(StorageLoaded(_sample_snapshot()))
            await pilot.pause()
            table = screen.query_one("#storage-table")
            first = screen.query_one("#storage-model-id", Input).value

            table.focus()
            table.move_cursor(row=1)
            await pilot.pause()

            self.assertNotEqual(
                screen.query_one("#storage-model-id", Input).value, first
            )


class ManageRefreshTests(unittest.IsolatedAsyncioTestCase):
    def _rows(self, count: int) -> list[EndpointInfo]:
        return [
            EndpointInfo(
                name=f"llp-app-{i}", app_id=f"ap-{i}", backend=BackendType.VLLM,
                instance_name=f"inst-{i}", provider=ComputeProvider.MODAL,
                state="running",
            )
            for i in range(count)
        ]

    async def test_a_refresh_keeps_the_selected_endpoint(self) -> None:
        app = _StyledApp()
        with patch.object(ManageScreen, "_refresh_endpoints", lambda self: None):
            async with app.run_test(size=(120, 40)) as pilot:
                screen = ManageScreen()
                app.push_screen(screen)
                await pilot.pause()
                screen.on_endpoints_loaded(EndpointsLoaded(self._rows(4)))
                await pilot.pause()
                screen.query_one("#manage-endpoint-table").move_cursor(row=2)
                await pilot.pause()
                selected = screen._selected_key

                screen.on_endpoints_loaded(EndpointsLoaded(self._rows(4)))
                await pilot.pause()

                self.assertEqual(screen._selected_key, selected)


class ScreenStackTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_app_boots_onto_exactly_one_screen(self) -> None:
        for configured, expected in ((True, MainMenuScreen), (False, SetupRequiredScreen)):
            with self.subTest(provider_configured=configured):
                with patch.object(TuiApp, "_provider_is_configured", return_value=configured), \
                     patch.object(TuiApp, "_warn_about_abandoned_deployments", lambda self: None), \
                     patch.object(MainMenuScreen, "on_mount", lambda self: None), \
                     patch("llm_launchpad.tui.app.ModalBackend.is_cli_available", return_value=False):
                    app = TuiApp()
                    async with app.run_test(size=(120, 40)) as pilot:
                        await pilot.pause()
                        self.assertEqual(len(app.screen_stack), 2)
                        self.assertIsInstance(app.screen, expected)

    async def test_navigating_in_and_out_does_not_grow_the_stack(self) -> None:
        with ExitStack() as stack:
            stack.enter_context(patch.object(TuiApp, "_provider_is_configured", return_value=True))
            stack.enter_context(patch.object(TuiApp, "_warn_about_abandoned_deployments", lambda self: None))
            stack.enter_context(patch("llm_launchpad.tui.app.ModalBackend.is_cli_available", return_value=False))
            # Every screen this walks through would otherwise reach the network
            # on mount; the stack is what is under test, not discovery.
            for method in (
                "_refresh_modal_auth_status", "_refresh_prime_auth_status",
                "_refresh_hf_auth_status", "_refresh_aai_auth_status",
                "_refresh_panels", "_refresh_quick_deploy_catalog",
                "_refresh_storage_estimate", "_refresh_billing_panels",
            ):
                stack.enter_context(patch.object(MainMenuScreen, method))
            stack.enter_context(patch.object(ManageScreen, "_refresh_endpoints", lambda self: None))
            stack.enter_context(
                patch.object(StorageScreen, "_refresh_storage_snapshot", lambda self: None)
            )

            app = TuiApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                baseline = len(app.screen_stack)
                for _ in range(3):
                    for key in ("m", "t", "s", "?"):
                        await pilot.press(key)
                        await pilot.pause()
                        await pilot.press("escape")
                        await pilot.pause()
                        self.assertEqual(
                            len(app.screen_stack),
                            baseline,
                            f"{key!r} left a screen on the stack",
                        )
                self.assertIsInstance(app.screen, MainMenuScreen)


if __name__ == "__main__":
    unittest.main()
