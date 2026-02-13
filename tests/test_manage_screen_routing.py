from __future__ import annotations

import unittest
from types import SimpleNamespace

from textual.app import App

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

    def begin_list(self) -> None:
        self.list_called += 1

    def list_instances(self, _backend):  # type: ignore[no-untyped-def]
        return []

    def push_screen(self, screen, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.pushed.append(screen)
        return super().push_screen(screen, *args, **kwargs)


class ManageScreenRoutingTests(unittest.IsolatedAsyncioTestCase):
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


if __name__ == "__main__":
    unittest.main()
