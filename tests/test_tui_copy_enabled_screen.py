from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from textual.app import App, ComposeResult
from textual.selection import SELECT_ALL
from textual.widgets import DataTable, Input, OptionList, Static
from textual.widgets.option_list import Option

from llm_launchpad.tui.app import (
    WizardApp,
    _osc_52_sequence,
    _screen_passthrough_sequence,
    _tmux_passthrough_sequence,
)
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
    def __init__(self) -> None:
        super().__init__()
        self.presented_text = ""

    def present_text_for_copy(self, text: str) -> None:
        self.presented_text = text


class _CopyKeyApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.help_quit_called = False
        self.presented_text = ""

    def action_help_quit(self) -> None:
        self.help_quit_called = True

    def present_text_for_copy(self, text: str) -> None:
        self.presented_text = text


class CopyEnabledScreenTests(unittest.IsolatedAsyncioTestCase):
    async def test_selection_copy_from_static_uses_present_text_fallback(self) -> None:
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
            self.assertEqual(app.presented_text, "Hello world")

    async def test_option_list_fallback_copies_highlighted_option(self) -> None:
        app = _CopyTestApp()
        async with app.run_test() as pilot:
            app.push_screen(_OptionListFallbackScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, _OptionListFallbackScreen)
            screen.action_copy_text()
            self.assertIn("Second option", app.presented_text)

    async def test_data_table_fallback_copies_highlighted_row_as_tsv(self) -> None:
        app = _CopyTestApp()
        async with app.run_test() as pilot:
            app.push_screen(_DataTableFallbackScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, _DataTableFallbackScreen)
            screen.action_copy_text()
            self.assertEqual(
                app.presented_text,
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
            self.assertEqual(app.presented_text, "Qwen/Qwen3-8B")

    async def test_ctrl_c_without_copyable_text_does_not_call_help_quit(self) -> None:
        app = _CopyKeyApp()
        async with app.run_test() as pilot:
            app.push_screen(_NoCopyFallbackScreen())
            await pilot.pause()

            await pilot.press("ctrl+c")
            await pilot.pause()

            self.assertFalse(app.help_quit_called)
            self.assertEqual(app.clipboard, "")

    async def test_command_c_alias_triggers_copy_action(self) -> None:
        app = _CopyTestApp()
        async with app.run_test() as pilot:
            app.push_screen(_StaticSelectionScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, _StaticSelectionScreen)
            target = screen.query_one("#copy-static", Static)
            screen.selections = {target: SELECT_ALL}
            await pilot.pause()

            event = SimpleNamespace(
                key="command+c",
                name="command+c",
                aliases=["command+c"],
                stop=lambda: None,
                prevent_default=lambda: None,
            )
            screen.on_key(event)
            self.assertEqual(app.clipboard, "Hello world")

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

            self.assertEqual(app.presented_text, "Hello world")

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

    async def test_selection_only_helper_does_not_copy_focused_fallback_without_selection(self) -> None:
        app = _CopyTestApp()
        async with app.run_test() as pilot:
            app.push_screen(_OptionListFallbackScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, _OptionListFallbackScreen)
            copied = screen._copy_selected_text_if_any()
            self.assertFalse(copied)
            self.assertEqual(app.clipboard, "")


class WizardAppClipboardTests(unittest.TestCase):
    def test_copy_to_clipboard_uses_pbcopy_on_darwin(self) -> None:
        app = WizardApp()
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
        app = WizardApp()
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
        app = WizardApp()
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

    def test_present_text_for_copy_falls_back_to_clipboard_without_suspend_support(self) -> None:
        app = WizardApp()

        copied: list[str] = []

        def _copy_to_clipboard(text: str) -> None:
            copied.append(text)

        app.copy_to_clipboard = _copy_to_clipboard  # type: ignore[method-assign]

        result = app.present_text_for_copy("copied text")

        self.assertFalse(result)
        self.assertEqual(copied, ["copied text"])


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
