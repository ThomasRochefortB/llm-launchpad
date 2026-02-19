from __future__ import annotations

import unittest

from textual.app import App
from textual.screen import Screen
from textual.widgets import DataTable, Input, OptionList

from llm_launchpad.protocol.enums import BackendType
from llm_launchpad.protocol.models import StorageSnapshot, StoredModelInfo
from llm_launchpad.tui.screens.storage import StorageScreen, _model_label
from llm_launchpad.tui.workers import StorageLoaded


class _TestApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.refresh_calls = 0
        self.refresh_force_flags: list[bool] = []
        self.predownload_calls: list[tuple[BackendType, str, str | None, str | None]] = []
        self.delete_calls: list[str] = []

    def begin_storage_refresh(self, receiver: object, force: bool = False) -> None:
        self.refresh_calls += 1
        self.refresh_force_flags.append(force)
        poster = getattr(receiver, "post_message")
        snapshot = StorageSnapshot(
            llamacpp_models=[
                StoredModelInfo(
                    backend=BackendType.LLAMACPP,
                    model_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
                    revision="main",
                    quant="Q4_K_M",
                    size_bytes=1024,
                    file_count=1,
                    source_volume="huggingface-cache",
                    incomplete=True,
                )
            ],
            vllm_models=[
                StoredModelInfo(
                    backend=BackendType.VLLM,
                    model_id="Qwen/Qwen3-4B-Thinking-2507-FP8",
                    revision=None,
                    quant=None,
                    size_bytes=2048,
                    file_count=2,
                    source_volume="huggingface-cache",
                )
            ],
        )
        poster(StorageLoaded(snapshot=snapshot))

    def begin_storage_predownload(  # type: ignore[override]
        self,
        backend: BackendType,
        model_id: str,
        quant: str | None = None,
        revision: str | None = None,
    ) -> None:
        self.predownload_calls.append((backend, model_id, quant, revision))

    def begin_storage_delete(self, model) -> None:  # type: ignore[no-untyped-def]
        self.delete_calls.append(model.model_id)


class StorageScreenTests(unittest.IsolatedAsyncioTestCase):
    def test_model_label_appends_incomplete_suffix(self) -> None:
        row = StoredModelInfo(
            backend=BackendType.LLAMACPP,
            model_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            source_volume="huggingface-cache",
            incomplete=True,
        )
        self.assertEqual(
            _model_label(row),
            "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF (INCOMPLETE)",
        )

    async def test_mount_loads_snapshot_into_table(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(StorageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StorageScreen)
            table = screen.query_one("#storage-table", DataTable)
            self.assertEqual(table.row_count, 2)

    async def test_mount_focuses_backend_filter_menu_for_arrow_navigation(self) -> None:
        app = _TestApp()
        async with app.run_test(size=(100, 35)) as pilot:
            app.push_screen(StorageScreen(initial_backend=BackendType.VLLM))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StorageScreen)
            backend_filter = screen.query_one("#storage-backend-filter", OptionList)
            self.assertTrue(backend_filter.has_focus)

            highlighted = backend_filter.highlighted_option
            self.assertIsNotNone(highlighted)
            assert highlighted is not None
            self.assertEqual(highlighted.id, "filter-vllm")

            await pilot.press("up")
            await pilot.pause()
            highlighted = backend_filter.highlighted_option
            self.assertIsNotNone(highlighted)
            assert highlighted is not None
            self.assertEqual(highlighted.id, "filter-llamacpp")

    async def test_down_from_backend_filter_moves_focus_to_action_menu(self) -> None:
        app = _TestApp()
        async with app.run_test(size=(100, 35)) as pilot:
            app.push_screen(StorageScreen(initial_backend=BackendType.VLLM))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StorageScreen)
            backend_filter = screen.query_one("#storage-backend-filter", OptionList)
            action_list = screen.query_one("#storage-action-list", OptionList)

            backend_filter.focus()
            backend_filter.highlighted = 2
            await pilot.pause()

            await pilot.press("down")
            await pilot.pause()

            self.assertTrue(action_list.has_focus)
            highlighted = action_list.highlighted_option
            self.assertIsNotNone(highlighted)
            assert highlighted is not None
            self.assertEqual(highlighted.id, "refresh")

    async def test_down_from_action_list_last_option_moves_focus_to_table(self) -> None:
        app = _TestApp()
        async with app.run_test(size=(100, 35)) as pilot:
            app.push_screen(StorageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StorageScreen)
            action_list = screen.query_one("#storage-action-list", OptionList)
            table = screen.query_one("#storage-table", DataTable)

            action_list.focus()
            action_list.highlighted = 2
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()

            self.assertTrue(table.has_focus)

    async def test_down_inside_table_moves_row_until_last_then_focuses_model_id(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(StorageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StorageScreen)
            table = screen.query_one("#storage-table", DataTable)
            model_id = screen.query_one("#storage-model-id", Input)

            table.focus()
            table.move_cursor(row=0)
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()
            self.assertTrue(table.has_focus)
            self.assertEqual(table.cursor_row, 1)

            await pilot.press("down")
            await pilot.pause()
            self.assertTrue(model_id.has_focus)

    async def test_up_from_table_first_row_moves_focus_to_action_list(self) -> None:
        app = _TestApp()
        async with app.run_test(size=(100, 35)) as pilot:
            app.push_screen(StorageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StorageScreen)
            table = screen.query_one("#storage-table", DataTable)
            action_list = screen.query_one("#storage-action-list", OptionList)

            table.focus()
            table.move_cursor(row=0)
            await pilot.pause()
            await pilot.press("up")
            await pilot.pause()

            self.assertTrue(action_list.has_focus)

    async def test_small_viewport_skips_collapsed_option_lists(self) -> None:
        app = _TestApp()
        async with app.run_test(size=(100, 24)) as pilot:
            app.push_screen(StorageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StorageScreen)
            backend_filter = screen.query_one("#storage-backend-filter", OptionList)
            action_list = screen.query_one("#storage-action-list", OptionList)
            table = screen.query_one("#storage-table", DataTable)

            self.assertEqual(backend_filter.size.height, 0)
            self.assertEqual(action_list.size.height, 0)
            self.assertTrue(table.has_focus)

    async def test_small_viewport_table_boundary_moves_to_model_id(self) -> None:
        app = _TestApp()
        async with app.run_test(size=(100, 24)) as pilot:
            app.push_screen(StorageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StorageScreen)
            table = screen.query_one("#storage-table", DataTable)
            model_id = screen.query_one("#storage-model-id", Input)

            self.assertTrue(table.has_focus)
            table.move_cursor(row=0)
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()
            self.assertEqual(table.cursor_row, 1)

            await pilot.press("down")
            await pilot.pause()
            self.assertTrue(model_id.has_focus)

    async def test_predownload_uses_form_values(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(StorageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StorageScreen)
            screen.query_one("#storage-model-id", Input).value = "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
            screen.query_one("#storage-model-backend", Input).value = "llamacpp"
            screen.query_one("#storage-model-quant", Input).value = "Q4_K_M"
            screen.query_one("#storage-model-revision", Input).value = "main"
            screen.action_predownload_selected()
            self.assertEqual(len(app.predownload_calls), 1)
            backend, model_id, quant, revision = app.predownload_calls[0]
            self.assertEqual(backend, BackendType.LLAMACPP)
            self.assertEqual(model_id, "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF")
            self.assertEqual(quant, "Q4_K_M")
            self.assertEqual(revision, "main")

    async def test_delete_selected_model_uses_selected_row(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(StorageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StorageScreen)
            first_model = screen._snapshot.llamacpp_models[0]
            screen._selected_model = first_model
            screen.action_delete_selected_model()
            self.assertEqual(app.delete_calls, [first_model.model_id])

    async def test_keyboard_navigation_prefills_model_fields(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(StorageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StorageScreen)
            table = screen.query_one("#storage-table", DataTable)
            table.focus()
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()
            self.assertEqual(
                screen.query_one("#storage-model-id", Input).value,
                "Qwen/Qwen3-4B-Thinking-2507-FP8",
            )
            self.assertEqual(screen.query_one("#storage-model-backend", Input).value, "vllm")

    async def test_resume_triggers_storage_refresh(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(StorageScreen())
            await pilot.pause()
            self.assertEqual(app.refresh_calls, 1)
            app.push_screen(Screen())
            await pilot.pause()
            app.pop_screen()
            await pilot.pause()
            self.assertEqual(app.refresh_calls, 2)

    async def test_manual_refresh_uses_force_refresh(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(StorageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StorageScreen)
            screen.action_refresh_storage()
            await pilot.pause()
            self.assertEqual(app.refresh_force_flags, [False, True])


if __name__ == "__main__":
    unittest.main()
