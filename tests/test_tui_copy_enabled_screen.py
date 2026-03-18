from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from textual.app import App, ComposeResult
from textual.selection import SELECT_ALL
from textual.widgets import DataTable, Input, OptionList, Static
from textual.widgets.option_list import Option

from llm_launchpad.tui.app import TuiApp, _osc_52_sequence, _screen_passthrough_sequence, _tmux_passthrough_sequence
from llm_launchpad.tui.screens.copy_enabled import CopyEnabledScreen
from llm_launchpad.tui.screens.deploy import (
    BackendSelectScreen,
    LlamaCppDeployScreen,
    VllmDeployScreen,
)
from llm_launchpad.tui.screens.main_menu import MainMenuScreen
from llm_launchpad.tui.screens.manage import (
    LogsParamsScreen,
    ManageScreen,
    StatusParamsScreen,
    StopParamsScreen,
)
from llm_launchpad.tui.screens.monitor import MonitorScreen
from llm_launchpad.tui.screens.settings import SettingsScreen
from llm_launchpad.tui.screens.storage import StorageScreen


class _StaticSelectionScreen(CopyEnabledScreen):
    def compose(self) -> ComposeResult:
        yield Static("[bold]Hello[/bold] world", id="copy-static")


class _OptionListFallbackScreen(CopyEnabledScreen):
    def compose(self) -> ComposeResult:
        yield OptionList(
            Option("  First option", id="first"),
            Option("  Second option", id="second"),
            id="copy-options",
        )

    def on_mount(self) -> None:
        options = self.query_one("#copy-options", OptionList)
        options.highlighted = 1
        options.focus()


class _DataTableFallbackScreen(CopyEnabledScreen):
    def compose(self) -> ComposeResult:
        yield DataTable(id="copy-table")

    def on_mount(self) -> None:
        table = self.query_one("#copy-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("backend", "model", "revision")
        table.add_row("llamacpp", "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", "main")
        table.add_row("vllm", "Qwen/Qwen3-4B-Thinking-2507-FP8", "-")
        table.move_cursor(row=1, column=0, animate=False)
        table.focus()


class _InputFallbackScreen(CopyEnabledScreen):
    def compose(self) -> ComposeResult:
        yield Input(value="Qwen/Qwen3-8B", id="copy-input")

    def on_mount(self) -> None:
        self.query_one("#copy-input", Input).focus()


class _NoCopyFallbackScreen(CopyEnabledScreen):
    def compose(self) -> ComposeResult:
        yield Static("Nothing focusable/copyable here")


class _CopyTestApp(App[None]):
    pass


class _CopyKeyApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.help_quit_called = False

    def action_help_quit(self) -> None:
        self.help_quit_called = True


class _QuitCaptureApp(TuiApp):
    def __init__(self) -> None:
        super().__init__()
        self.notifications: list[tuple[object, object, object, object]] = []
        self.quit_calls = 0

    def notify(
        self,
        message: object,
        *,
        title: object | None = None,
        severity: str = "information",
        timeout: float | None = None,
        **kwargs: object,
    ) -> None:
        self.notifications.append((message, title, severity, timeout))

    async def action_quit(self) -> None:
        self.quit_calls += 1


class _MouseDriverStub:
    def __init__(self, mouse: bool) -> None:
        self._mouse = mouse
        self.enable_calls = 0
        self.disable_calls = 0

    def _enable_mouse_support(self) -> None:
        self.enable_calls += 1

    def _disable_mouse_support(self) -> None:
        self.disable_calls += 1


class CopyEnabledScreenTests(unittest.IsolatedAsyncioTestCase):
    async def test_selection_copy_from_static_writes_to_clipboard(self) -> None:
        app = _CopyTestApp()
        async with app.run_test() as pilot:
            app.push_screen(_StaticSelectionScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, _StaticSelectionScreen)
            target = screen.query_one("#copy-static", Static)
            screen.selections = {target: SELECT_ALL}
            await pilot.pause()

            screen.action_copy_text()
            self.assertEqual(app.clipboard, "Hello world")

    async def test_option_list_fallback_copies_highlighted_option(self) -> None:
        app = _CopyTestApp()
        async with app.run_test() as pilot:
            app.push_screen(_OptionListFallbackScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, _OptionListFallbackScreen)
            screen.action_copy_text()
            self.assertIn("Second option", app.clipboard)

    async def test_data_table_fallback_copies_highlighted_row_as_tsv(self) -> None:
        app = _CopyTestApp()
        async with app.run_test() as pilot:
            app.push_screen(_DataTableFallbackScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, _DataTableFallbackScreen)
            screen.action_copy_text()
            self.assertEqual(
                app.clipboard,
                "vllm\tQwen/Qwen3-4B-Thinking-2507-FP8\t-",
            )

    async def test_input_fallback_copies_full_value(self) -> None:
        app = _CopyTestApp()
        async with app.run_test() as pilot:
            app.push_screen(_InputFallbackScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, _InputFallbackScreen)
            screen.action_copy_text()
            self.assertEqual(app.clipboard, "Qwen/Qwen3-8B")

    async def test_input_ctrl_c_uses_input_native_copy_behavior(self) -> None:
        app = _CopyKeyApp()
        async with app.run_test() as pilot:
            app.push_screen(_InputFallbackScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, _InputFallbackScreen)
            input_widget = screen.query_one("#copy-input", Input)
            input_widget.action_select_all()
            await pilot.pause()

            await pilot.press("ctrl+c")
            await pilot.pause()

            self.assertFalse(app.help_quit_called)
            self.assertEqual(app.clipboard, "Qwen/Qwen3-8B")

    async def test_y_alias_triggers_copy_action(self) -> None:
        app = _CopyTestApp()
        async with app.run_test() as pilot:
            app.push_screen(_StaticSelectionScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, _StaticSelectionScreen)
            target = screen.query_one("#copy-static", Static)
            screen.selections = {target: SELECT_ALL}
            await pilot.pause()

            await pilot.press("y")
            await pilot.pause()

            self.assertEqual(app.clipboard, "Hello world")

    async def test_selection_change_auto_copies_selection(self) -> None:
        app = _CopyTestApp()
        async with app.run_test() as pilot:
            app.push_screen(_StaticSelectionScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, _StaticSelectionScreen)
            target = screen.query_one("#copy-static", Static)
            target.text_select_all()
            await pilot.pause()
            await pilot.pause()

            self.assertEqual(app.clipboard, "Hello world")

    async def test_ctrl_shift_c_binding_copies_selected_text_to_clipboard(self) -> None:
        app = _CopyTestApp()
        async with app.run_test() as pilot:
            app.push_screen(_StaticSelectionScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, _StaticSelectionScreen)
            target = screen.query_one("#copy-static", Static)
            screen.selections = {target: SELECT_ALL}
            await pilot.pause()

            await pilot.press("ctrl+shift+c")
            await pilot.pause()

            self.assertEqual(app.clipboard, "Hello world")

    async def test_super_c_binding_copies_selected_text_to_clipboard(self) -> None:
        app = _CopyTestApp()
        async with app.run_test() as pilot:
            app.push_screen(_StaticSelectionScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, _StaticSelectionScreen)
            target = screen.query_one("#copy-static", Static)
            screen.selections = {target: SELECT_ALL}
            await pilot.pause()

            await pilot.press("super+c")
            await pilot.pause()

            self.assertEqual(app.clipboard, "Hello world")

    async def test_direct_clipboard_action_copies_to_clipboard(self) -> None:
        app = _CopyTestApp()
        async with app.run_test() as pilot:
            app.push_screen(_StaticSelectionScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, _StaticSelectionScreen)
            target = screen.query_one("#copy-static", Static)
            screen.selections = {target: SELECT_ALL}
            await pilot.pause()

            screen.action_copy_text_to_clipboard()

            self.assertEqual(app.clipboard, "Hello world")

    async def test_selected_text_hook_does_not_fall_back_without_selection(self) -> None:
        app = _CopyTestApp()
        async with app.run_test() as pilot:
            app.push_screen(_OptionListFallbackScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, _OptionListFallbackScreen)
            self.assertIsNone(screen._selected_text_for_copy())
            self.assertEqual(app.clipboard, "")


class TuiAppClipboardTests(unittest.TestCase):
    def test_copy_to_clipboard_uses_pbcopy_on_darwin(self) -> None:
        app = TuiApp()
        with (
            patch("llm_launchpad.tui.app.sys.platform", "darwin"),
            patch("llm_launchpad.tui.app.subprocess.run") as run_mock,
        ):
            app.copy_to_clipboard("copied text")
        self.assertEqual(app.clipboard, "copied text")
        run_mock.assert_called_once_with(
            ["pbcopy"],
            input=b"copied text",
            check=True,
            timeout=2,
        )

    def test_copy_to_clipboard_writes_tmux_passthrough_sequence(self) -> None:
        writes: list[str] = []
        app = TuiApp()
        app._driver = SimpleNamespace(write=writes.append)

        with patch.dict("llm_launchpad.tui.app.os.environ", {"TMUX": "/tmp/tmux-1000/default,123,0"}):
            app.copy_to_clipboard("copied text")

        self.assertEqual(app.clipboard, "copied text")
        self.assertEqual(
            writes,
            [
                _osc_52_sequence("copied text"),
                _tmux_passthrough_sequence("copied text"),
            ],
        )

    def test_copy_to_clipboard_writes_screen_passthrough_sequence(self) -> None:
        writes: list[str] = []
        app = TuiApp()
        app._driver = SimpleNamespace(write=writes.append)

        with patch.dict("llm_launchpad.tui.app.os.environ", {"TERM": "screen-256color"}, clear=True):
            app.copy_to_clipboard("copied text")

        self.assertEqual(app.clipboard, "copied text")
        self.assertEqual(
            writes,
            [
                _osc_52_sequence("copied text"),
                _screen_passthrough_sequence("copied text"),
            ],
        )

class TuiAppQuitTests(unittest.IsolatedAsyncioTestCase):
    async def test_ctrl_c_first_press_warns_before_quitting(self) -> None:
        app = _QuitCaptureApp()

        with patch("llm_launchpad.tui.app.time.monotonic", return_value=10.0):
            await app.action_request_quit()

        self.assertEqual(app.quit_calls, 0)
        self.assertEqual(
            app.notifications,
            [("Ctrl+C again to exit", "Exit llm-launchpad?", "warning", 2.0)],
        )

    async def test_ctrl_c_second_press_within_window_quits(self) -> None:
        app = _QuitCaptureApp()

        with patch("llm_launchpad.tui.app.time.monotonic", side_effect=[10.0, 11.0]):
            await app.action_request_quit()
            await app.action_request_quit()

        self.assertEqual(app.quit_calls, 1)

    async def test_ctrl_c_after_window_warns_again(self) -> None:
        app = _QuitCaptureApp()

        with patch("llm_launchpad.tui.app.time.monotonic", side_effect=[10.0, 13.5]):
            await app.action_request_quit()
            await app.action_request_quit()

        self.assertEqual(app.quit_calls, 0)
        self.assertEqual(len(app.notifications), 2)


class TuiAppMouseModeTests(unittest.TestCase):
    def test_toggle_mouse_mode_enables_driver_mouse_support(self) -> None:
        app = TuiApp(mouse_enabled=False)
        driver = _MouseDriverStub(mouse=False)
        app._driver = driver

        app.action_toggle_mouse_mode()

        self.assertTrue(app.mouse_enabled)
        self.assertTrue(driver._mouse)
        self.assertEqual(driver.enable_calls, 1)
        self.assertEqual(driver.disable_calls, 0)

    def test_toggle_mouse_mode_disables_driver_mouse_support(self) -> None:
        app = TuiApp(mouse_enabled=True)
        driver = _MouseDriverStub(mouse=True)
        app._driver = driver

        app.action_toggle_mouse_mode()

        self.assertFalse(app.mouse_enabled)
        self.assertFalse(driver._mouse)
        self.assertEqual(driver.enable_calls, 0)
        self.assertEqual(driver.disable_calls, 1)


class CopyEnabledScreenInheritanceTests(unittest.TestCase):
    def test_all_tui_screens_subclass_copy_enabled_screen(self) -> None:
        screen_classes = [
            SettingsScreen,
            MainMenuScreen,
            MonitorScreen,
            BackendSelectScreen,
            LlamaCppDeployScreen,
            VllmDeployScreen,
            StorageScreen,
            ManageScreen,
            StatusParamsScreen,
            LogsParamsScreen,
            StopParamsScreen,
        ]
        for screen_class in screen_classes:
            self.assertTrue(issubclass(screen_class, CopyEnabledScreen), screen_class.__name__)


if __name__ == "__main__":
    unittest.main()
