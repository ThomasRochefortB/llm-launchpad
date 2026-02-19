from __future__ import annotations

import unittest
from types import SimpleNamespace

from textual.app import App
from textual.widgets import OptionList

from llm_launchpad.protocol.enums import BackendType
from llm_launchpad.protocol.models import EndpointInfo
from llm_launchpad.tui.screens.manage import (
    LogsParamsScreen,
    ManageScreen,
    StatusParamsScreen,
    StopParamsScreen,
)


class _TestApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.list_called = 0
        self.pushed: list[object] = []
        self.instances_by_backend: dict[BackendType, list[EndpointInfo]] = {
            BackendType.LLAMACPP: [],
            BackendType.VLLM: [],
        }

    def begin_list(self) -> None:
        self.list_called += 1

    def list_instances(self, _backend=None):  # type: ignore[no-untyped-def]
        if _backend is None:
            return self.instances_by_backend[BackendType.LLAMACPP] + self.instances_by_backend[BackendType.VLLM]
        return self.instances_by_backend.get(_backend, [])

    def push_screen(self, screen, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.pushed.append(screen)
        return super().push_screen(screen, *args, **kwargs)


class ManageScreenRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_manage_screen_focuses_action_menu(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(ManageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ManageScreen)
            action_list = screen.query_one("#manage-action-list", OptionList)
            self.assertTrue(action_list.has_focus)
            highlighted = action_list.highlighted_option
            self.assertIsNotNone(highlighted)
            assert highlighted is not None
            self.assertEqual(highlighted.id, "list")

    async def test_manage_screen_action_routing(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(ManageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ManageScreen)

            for option_id in ["list", "status", "logs", "stop"]:
                screen.on_option_list_option_selected(
                    SimpleNamespace(option=SimpleNamespace(id=option_id))
                )
                await pilot.pause()

        self.assertEqual(app.list_called, 1)
        self.assertTrue(any(isinstance(s, StatusParamsScreen) for s in app.pushed))
        self.assertTrue(any(isinstance(s, LogsParamsScreen) for s in app.pushed))
        self.assertTrue(any(isinstance(s, StopParamsScreen) for s in app.pushed))

    async def test_status_params_menu_is_arrow_navigable(self) -> None:
        app = _TestApp()
        app.instances_by_backend[BackendType.LLAMACPP] = [
            EndpointInfo(name="llamacpp-alpha", app_id="ap-llama", state="running", backend=BackendType.LLAMACPP),
        ]
        app.instances_by_backend[BackendType.VLLM] = [
            EndpointInfo(name="vllm-beta", app_id="ap-vllm", state="deployed", backend=BackendType.VLLM),
        ]
        async with app.run_test() as pilot:
            app.push_screen(StatusParamsScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StatusParamsScreen)
            instance_list = screen.query_one("#status-instance-list", OptionList)
            self.assertTrue(instance_list.has_focus)
            first = instance_list.highlighted_option
            self.assertIsNotNone(first)
            await pilot.press("down")
            await pilot.pause()
            second = instance_list.highlighted_option
            self.assertIsNotNone(second)
            assert first is not None
            assert second is not None
            self.assertNotEqual(first.id, second.id)

    async def test_logs_params_menu_is_arrow_navigable(self) -> None:
        app = _TestApp()
        app.instances_by_backend[BackendType.LLAMACPP] = [
            EndpointInfo(name="llamacpp-alpha", app_id="ap-llama", state="running", backend=BackendType.LLAMACPP),
        ]
        app.instances_by_backend[BackendType.VLLM] = [
            EndpointInfo(name="vllm-beta", app_id="ap-vllm", state="deployed", backend=BackendType.VLLM),
        ]
        async with app.run_test() as pilot:
            app.push_screen(LogsParamsScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, LogsParamsScreen)
            instance_list = screen.query_one("#logs-instance-list", OptionList)
            self.assertTrue(instance_list.has_focus)
            first = instance_list.highlighted_option
            self.assertIsNotNone(first)
            await pilot.press("down")
            await pilot.pause()
            second = instance_list.highlighted_option
            self.assertIsNotNone(second)
            assert first is not None
            assert second is not None
            self.assertNotEqual(first.id, second.id)


if __name__ == "__main__":
    unittest.main()
