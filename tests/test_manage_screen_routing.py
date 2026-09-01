from __future__ import annotations

import unittest

from textual.app import App
from textual.widgets import Button, Input, Static

from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import EndpointInfo
from llm_launchpad.tui.screens.manage import (
    BenchmarkOptionsScreen,
    EndpointActionsScreen,
    ManageScreen,
    StatusOptionsScreen,
    StopConfirmScreen,
)
from llm_launchpad.tui.widgets.adaptive_table import AdaptiveDataTable
from llm_launchpad.tui.workers import EndpointsLoaded


class _TestApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.pushed: list[object] = []
        self.refresh_forces: list[bool] = []
        self.instances: list[EndpointInfo] = []
        self.status_calls: list[tuple[EndpointInfo, str | None, int]] = []
        self.logs_calls: list[tuple[EndpointInfo, bool]] = []
        self.benchmark_calls: list[
            tuple[EndpointInfo, str, int | None, int, int, str, str | None]
        ] = []
        self.stop_calls: list[EndpointInfo] = []

    def begin_endpoint_refresh(self, receiver: object, force: bool = False) -> None:
        self.refresh_forces.append(force)
        receiver.post_message(EndpointsLoaded(rows=list(self.instances)))  # type: ignore[attr-defined]

    def push_screen(self, screen, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.pushed.append(screen)
        return super().push_screen(screen, *args, **kwargs)

    def begin_status(
        self,
        endpoint: EndpointInfo,
        url_override: str | None = None,
        timeout: int = 60,
    ) -> None:
        self.status_calls.append((endpoint, url_override, timeout))

    def begin_logs(self, endpoint: EndpointInfo, follow: bool = True) -> None:
        self.logs_calls.append((endpoint, follow))

    def begin_benchmark(
        self,
        endpoint: EndpointInfo,
        *,
        concurrency: str = "1,2,4,8,16",
        request_count: int | None = None,
        input_tokens: int = 550,
        output_tokens: int = 256,
        tokenizer: str = "gpt2",
        output_dir: str | None = None,
    ) -> None:
        self.benchmark_calls.append(
            (
                endpoint,
                concurrency,
                request_count,
                input_tokens,
                output_tokens,
                tokenizer,
                output_dir,
            )
        )

    def begin_stop(self, endpoint: EndpointInfo) -> None:
        self.stop_calls.append(endpoint)


def _endpoint(
    name: str,
    app_id: str,
    *,
    state: str = "running",
    backend: BackendType = BackendType.VLLM,
    provider: ComputeProvider = ComputeProvider.MODAL,
) -> EndpointInfo:
    return EndpointInfo(
        name=name,
        app_id=app_id,
        state=state,
        backend=backend,
        provider=provider,
        web_url=f"https://example.test/{app_id}" if state in {"running", "deployed", "active"} else None,
    )


class ManageScreenRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_manage_screen_loads_once_and_focuses_endpoint_table(self) -> None:
        app = _TestApp()
        app.instances = [
            _endpoint("vllm-beta", "ap-vllm", state="deployed"),
            _endpoint(
                "llamacpp-alpha",
                "ap-llama",
                backend=BackendType.LLAMACPP,
            ),
            _endpoint("failed-app", "ap-failed", state="failed"),
        ]

        async with app.run_test() as pilot:
            app.push_screen(ManageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ManageScreen)
            table = screen.query_one("#manage-endpoint-table", AdaptiveDataTable)

            self.assertTrue(table.has_focus)
            self.assertEqual(table.row_count, 3)
            self.assertEqual(app.refresh_forces, [False])
            self.assertIn("3 managed endpoints", str(screen.query_one("#manage-status", Static).content))

    async def test_selected_endpoint_routes_status_logs_benchmark_and_stop(self) -> None:
        first = _endpoint("alpha", "ap-first")
        second = _endpoint("beta", "ap-second")
        app = _TestApp()
        app.instances = [first, second]

        async with app.run_test() as pilot:
            app.push_screen(ManageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ManageScreen)
            await pilot.press("down")
            await pilot.pause()

            screen.action_logs_selected()
            self.assertEqual(app.logs_calls, [(second, True)])

            screen.action_status_selected()
            await pilot.pause()
            self.assertIsInstance(app.screen, StatusOptionsScreen)
            assert isinstance(app.screen, StatusOptionsScreen)
            self.assertIs(app.screen.endpoint, second)
            app.pop_screen()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ManageScreen)
            screen.action_benchmark_selected()
            await pilot.pause()
            self.assertIsInstance(app.screen, BenchmarkOptionsScreen)
            assert isinstance(app.screen, BenchmarkOptionsScreen)
            self.assertIs(app.screen.endpoint, second)
            app.pop_screen()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ManageScreen)
            screen.action_stop_selected()
            await pilot.pause()
            self.assertIsInstance(app.screen, StopConfirmScreen)
            assert isinstance(app.screen, StopConfirmScreen)
            self.assertIs(app.screen.endpoint, second)

    async def test_enter_opens_action_menu_and_menu_routes_logs(self) -> None:
        endpoint = _endpoint("active-endpoint", "ap-active")
        app = _TestApp()
        app.instances = [endpoint]

        async with app.run_test() as pilot:
            app.push_screen(ManageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ManageScreen)
            await pilot.press("enter")
            await pilot.pause()

            self.assertIsInstance(app.screen, EndpointActionsScreen)
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

        self.assertEqual(app.logs_calls, [(endpoint, True)])

    async def test_failed_endpoint_keeps_logs_but_blocks_runtime_actions(self) -> None:
        failed = _endpoint("failed-app", "ap-failed", state="failed")
        app = _TestApp()
        app.instances = [failed]

        async with app.run_test() as pilot:
            app.push_screen(ManageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ManageScreen)

            screen.action_logs_selected()
            screen.action_status_selected()
            screen.action_benchmark_selected()
            screen.action_stop_selected()
            await pilot.pause()

            self.assertEqual(app.logs_calls, [(failed, True)])
            self.assertIs(app.screen, screen)
            self.assertEqual(app.stop_calls, [])

    async def test_status_options_submits_override_for_preselected_endpoint(self) -> None:
        endpoint = _endpoint("llamacpp-alpha", "ap-llama", backend=BackendType.LLAMACPP)
        app = _TestApp()

        async with app.run_test() as pilot:
            app.push_screen(StatusOptionsScreen(endpoint))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StatusOptionsScreen)
            screen.query_one("#status-url", Input).value = "https://override.test"
            screen.query_one("#status-timeout", Input).value = "15"
            screen.action_do_submit()
            await pilot.pause()

        self.assertEqual(app.status_calls, [(endpoint, "https://override.test", 15)])

    async def test_status_options_rejects_invalid_timeout_without_submitting(self) -> None:
        endpoint = _endpoint("alpha", "ap-alpha")
        app = _TestApp()

        async with app.run_test() as pilot:
            app.push_screen(StatusOptionsScreen(endpoint))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StatusOptionsScreen)
            screen.query_one("#status-timeout", Input).value = "later"
            screen.action_do_submit()
            await pilot.pause()

            self.assertEqual(app.status_calls, [])
            self.assertIn("integer", str(screen.query_one("#status-feedback", Static).content))

    async def test_benchmark_options_submits_selected_endpoint_and_defaults(self) -> None:
        endpoint = _endpoint("vllm-beta", "ap-vllm", state="deployed")
        app = _TestApp()

        async with app.run_test() as pilot:
            app.push_screen(BenchmarkOptionsScreen(endpoint))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, BenchmarkOptionsScreen)
            screen.action_do_submit()
            await pilot.pause()

        self.assertEqual(
            app.benchmark_calls,
            [(endpoint, "1,2,4,8,16", None, 550, 256, "gpt2", None)],
        )

    async def test_stop_confirmation_is_arrow_navigable_and_defaults_to_cancel(self) -> None:
        endpoint = _endpoint("vllm-beta", "ap-vllm")
        app = _TestApp()

        async with app.run_test() as pilot:
            app.push_screen(StopConfirmScreen(endpoint))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StopConfirmScreen)

            self.assertEqual(app.stop_calls, [])
            self.assertTrue(screen.query_one("#stop-cancel", Button).has_focus)
            await pilot.press("right")
            await pilot.pause()
            self.assertTrue(screen.query_one("#stop-confirm", Button).has_focus)
            await pilot.press("enter")
            await pilot.pause()

        self.assertEqual(app.stop_calls, [endpoint])

    async def test_duplicate_names_route_by_exact_app_id(self) -> None:
        first = _endpoint("duplicate", "ap-first")
        second = _endpoint("duplicate", "ap-second")
        app = _TestApp()
        app.instances = [first, second]

        async with app.run_test() as pilot:
            app.push_screen(ManageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ManageScreen)
            await pilot.press("down")
            await pilot.pause()
            screen.action_logs_selected()

        self.assertEqual(app.logs_calls, [(second, True)])

    async def test_manage_refreshes_after_returning_from_child_screen(self) -> None:
        app = _TestApp()
        app.instances = [_endpoint("alpha", "ap-alpha")]

        async with app.run_test() as pilot:
            app.push_screen(ManageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ManageScreen)
            screen.action_status_selected()
            await pilot.pause()
            app.pop_screen()
            await pilot.pause()

        self.assertEqual(app.refresh_forces, [False, True])


if __name__ == "__main__":
    unittest.main()
