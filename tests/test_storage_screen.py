from __future__ import annotations

import unittest

from textual.content import Content
from textual.app import App
from textual.coordinate import Coordinate
from textual.screen import Screen
from textual.widgets import Button, DataTable, Input, Select, Static

from llm_launchpad.protocol.enums import BackendType
from llm_launchpad.protocol.models import StorageSnapshot, StoredModelInfo
from llm_launchpad.tui.screens.storage import (
    StorageDeleteConfirmScreen,
    StorageScreen,
    _model_label,
)
from llm_launchpad.tui.widgets.adaptive_table import AdaptiveDataTable
from llm_launchpad.tui.workers import StorageLoaded

_GIB = 1024**3


def _sample_snapshot() -> StorageSnapshot:
    return StorageSnapshot(
        llamacpp_models=[
            StoredModelInfo(
                backend=BackendType.LLAMACPP,
                model_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
                revision="main",
                quant="Q4_K_M",
                size_bytes=2 * _GIB,
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
                size_bytes=3 * _GIB,
                file_count=2,
                source_volume="huggingface-cache",
            )
        ],
    )


class StorageRowPresentationTests(unittest.TestCase):
    def test_commit_revisions_are_shortened_and_branches_kept(self) -> None:
        from llm_launchpad.tui.screens.storage import _short_revision

        def row(revision: str | None) -> StoredModelInfo:
            return StoredModelInfo(
                backend=BackendType.LLAMACPP, model_id="org/m", revision=revision,
                quant=None, size_bytes=1, file_count=1, source_volume="v",
            )

        self.assertEqual(_short_revision(row("f71df18cc2ddd1ba50ca74e86d944b2f756a56b0")), "f71df18")
        self.assertEqual(_short_revision(row("main")), "main")
        self.assertEqual(_short_revision(row(None)), "-")


class StorageGroupingTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_model_names_are_written_once(self) -> None:
        def quant_row(quant: str) -> StoredModelInfo:
            return StoredModelInfo(
                backend=BackendType.LLAMACPP, model_id="org/model-GGUF", revision="main",
                quant=quant, size_bytes=_GIB, file_count=1, source_volume="v",
            )

        snapshot = StorageSnapshot(
            llamacpp_models=[quant_row("Q2_K"), quant_row("Q4_K_M")], vllm_models=[],
        )

        class _GroupApp(_TestApp):
            def begin_storage_refresh(self, receiver: object, force: bool = False) -> None:
                receiver.post_message(StorageLoaded(snapshot=snapshot))  # type: ignore[attr-defined]

        app = _GroupApp()
        async with app.run_test(size=(140, 40)) as pilot:
            app.push_screen(StorageScreen())
            await pilot.pause()
            table = app.screen.query_one("#storage-table", AdaptiveDataTable)
            column = table.visible_column_keys.index("model")
            first = str(table.get_cell_at(Coordinate(0, column)))
            second = str(table.get_cell_at(Coordinate(1, column)))
            self.assertEqual(first, "org/model-GGUF")
            self.assertNotIn("org/model-GGUF", second)
            # Each quant stays its own row, so each can be deleted on its own.
            self.assertEqual(table.row_count, 2)


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
        receiver.post_message(StorageLoaded(snapshot=_sample_snapshot()))

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


class _DeferredRefreshApp(_TestApp):
    def __init__(self) -> None:
        super().__init__()
        self._refresh_receiver: object | None = None

    def begin_storage_refresh(self, receiver: object, force: bool = False) -> None:
        self.refresh_calls += 1
        self.refresh_force_flags.append(force)
        self._refresh_receiver = receiver

    def deliver_snapshot(self) -> None:
        assert self._refresh_receiver is not None
        self._refresh_receiver.post_message(StorageLoaded(snapshot=_sample_snapshot()))


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
            table = screen.query_one("#storage-table", AdaptiveDataTable)
            self.assertEqual(table.row_count, 2)
            # 5.0 GiB total stays inside Modal's 1 TiB free tier, so row costs are $0.00/mo.
            self.assertEqual(str(table.get_cell_at(Coordinate(0, 4))), "$0.00/mo")
            self.assertEqual(str(table.get_cell_at(Coordinate(1, 4))), "$0.00/mo")

            status = screen.query_one("#storage-status", Static)
            rendered_status = str(status.renderable)
            self.assertIn("5.0 GiB cached", rendered_status)
            # Nothing is billable inside the free tier, so the arithmetic that
            # would say "0.00 GiB billable past 1 TiB free" is left out.
            self.assertNotIn("billable", rendered_status)
            self.assertIn("$0.00/mo", rendered_status)

    async def test_modal_row_costs_reflect_billable_excess_above_free_tier(self) -> None:
        over_tier = StorageSnapshot(
            llamacpp_models=[
                StoredModelInfo(
                    backend=BackendType.LLAMACPP,
                    model_id="org/large-a",
                    size_bytes=800 * _GIB,
                    file_count=1,
                    source_volume="llamacpp-models",
                ),
            ],
            vllm_models=[
                StoredModelInfo(
                    backend=BackendType.VLLM,
                    model_id="org/large-b",
                    size_bytes=400 * _GIB,
                    file_count=2,
                    source_volume="vllm-models",
                ),
            ],
        )

        class _OverTierApp(_TestApp):
            def begin_storage_refresh(self, receiver: object, force: bool = False) -> None:
                receiver.post_message(StorageLoaded(snapshot=over_tier))  # type: ignore[attr-defined]

        app = _OverTierApp()
        async with app.run_test(size=(140, 30)) as pilot:
            app.push_screen(StorageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StorageScreen)
            table = screen.query_one("#storage-table", AdaptiveDataTable)
            cost_col = table.visible_column_keys.index("cost")
            # 1200 GiB total -> 176 GiB billable ($15.84/mo); 2/3 ($10.56) and 1/3 ($5.28).
            self.assertEqual(str(table.get_cell_at(Coordinate(0, cost_col))), "$10.56/mo")
            self.assertEqual(str(table.get_cell_at(Coordinate(1, cost_col))), "$5.28/mo")
            detail = str(screen.query_one("#storage-detail", Static).renderable)
            self.assertIn("deleting saves $15.84/mo", detail)

    async def test_stale_storage_loaded_renders_rows_immediately_with_refreshing_indicator(self) -> None:
        from llm_launchpad.core.deploy_log_summary import SUMMARY_SPINNER_FRAMES

        snapshot = _sample_snapshot()

        class _StaleApp(_TestApp):
            def begin_storage_refresh(self, receiver: object, force: bool = False) -> None:
                _ = force
                poster = getattr(receiver, "post_message", None)
                if callable(poster):
                    poster(StorageLoaded(snapshot=snapshot, is_stale=True))

        app = _StaleApp()
        async with app.run_test(size=(140, 30)) as pilot:
            app.push_screen(StorageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StorageScreen)
            table = screen.query_one("#storage-table", AdaptiveDataTable)
            self.assertEqual(table.row_count, 2)
            status = str(screen.query_one("#storage-status", Static).render())
            self.assertIn("Refreshing…", status)
            self.assertIn(SUMMARY_SPINNER_FRAMES[0], status)
            self.assertIsNotNone(screen._refresh_spinner_timer)

            screen._tick_refresh_spinner()
            next_status = str(screen.query_one("#storage-status", Static).render())
            self.assertIn(SUMMARY_SPINNER_FRAMES[1], next_status)

            screen.post_message(StorageLoaded(snapshot=snapshot, is_stale=False))
            await pilot.pause()
            updated_status = str(screen.query_one("#storage-status", Static).render())
            self.assertNotIn("Refreshing…", updated_status)
            self.assertIsNone(screen._refresh_spinner_timer)

    async def test_table_adapts_columns_and_preserves_highlight_on_resize(self) -> None:
        app = _TestApp()
        async with app.run_test(size=(140, 35)) as pilot:
            app.push_screen(StorageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StorageScreen)
            table = screen.query_one("#storage-table", AdaptiveDataTable)
            self.assertEqual(
                table.visible_column_keys,
                ("backend", "model", "revision", "quant", "files", "size", "cost"),
            )
            table.move_cursor(row=1, column=0, animate=False)
            selected = table.coordinate_to_cell_key(Coordinate(1, 0)).row_key

            await pilot.resize_terminal(50, 35)
            await pilot.pause()

            self.assertEqual(table.visible_column_keys, ("model", "summary"))
            self.assertEqual(
                table.coordinate_to_cell_key(Coordinate(table.cursor_row, 0)).row_key,
                selected,
            )

    async def test_initial_backend_prefills_the_filter(self) -> None:
        app = _TestApp()
        async with app.run_test(size=(100, 35)) as pilot:
            app.push_screen(StorageScreen(initial_backend=BackendType.VLLM))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StorageScreen)
            provider_filter = screen.query_one("#storage-provider-filter", Select)
            filter_input = screen.query_one("#storage-filter", Input)
            # Provider scope comes first; the text filter carries the backend.
            self.assertTrue(provider_filter.has_focus)
            self.assertEqual(filter_input.value, BackendType.VLLM.value)
            table = screen.query_one("#storage-table", DataTable)
            self.assertTrue(table.row_count > 0)

    async def test_down_from_filter_moves_focus_to_table(self) -> None:
        app = _TestApp()
        async with app.run_test(size=(100, 35)) as pilot:
            app.push_screen(StorageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StorageScreen)
            filter_input = screen.query_one("#storage-filter", Input)
            table = screen.query_one("#storage-table", DataTable)

            filter_input.focus()
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()

            self.assertTrue(table.has_focus)

    async def test_slash_focuses_model_filter_input(self) -> None:
        app = _TestApp()
        async with app.run_test(size=(100, 35)) as pilot:
            app.push_screen(StorageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StorageScreen)
            filter_input = screen.query_one("#storage-filter", Input)
            table = screen.query_one("#storage-table", DataTable)

            table.focus()
            await pilot.pause()
            await pilot.press("slash")
            await pilot.pause()

            self.assertTrue(filter_input.has_focus)

    async def test_down_inside_table_moves_row_until_last_then_focuses_filter(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(StorageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StorageScreen)
            table = screen.query_one("#storage-table", DataTable)
            search = screen.query_one("#storage-filter", Input)

            table.focus()
            table.move_cursor(row=0)
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()
            self.assertTrue(table.has_focus)
            self.assertEqual(table.cursor_row, 1)

            # The filter sits above the table now; the collapsed form below
            # takes no focus, so the last row keeps it.
            await pilot.press("down")
            await pilot.pause()
            self.assertTrue(table.has_focus)
            self.assertFalse(search.has_focus)

    async def test_up_from_table_first_row_moves_focus_to_filter(self) -> None:
        app = _TestApp()
        async with app.run_test(size=(100, 35)) as pilot:
            app.push_screen(StorageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StorageScreen)
            table = screen.query_one("#storage-table", DataTable)
            filter_input = screen.query_one("#storage-filter", Input)

            table.focus()
            table.move_cursor(row=0)
            await pilot.pause()
            await pilot.press("up")
            await pilot.pause()

            self.assertTrue(filter_input.has_focus)

    async def test_small_viewport_keeps_filter_visible_and_focusable(self) -> None:
        app = _TestApp()
        async with app.run_test(size=(100, 24)) as pilot:
            app.push_screen(StorageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StorageScreen)
            provider_filter = screen.query_one("#storage-provider-filter", Select)
            filter_input = screen.query_one("#storage-filter", Input)

            self.assertGreater(filter_input.size.height, 0)
            self.assertTrue(provider_filter.has_focus)
            filter_input.focus()
            await pilot.pause()
            self.assertTrue(filter_input.has_focus)

    async def test_small_viewport_table_boundary_moves_to_filter(self) -> None:
        app = _TestApp()
        async with app.run_test(size=(100, 24)) as pilot:
            app.push_screen(StorageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StorageScreen)
            table = screen.query_one("#storage-table", DataTable)
            search = screen.query_one("#storage-filter", Input)

            table.focus()
            await pilot.pause()
            self.assertTrue(table.has_focus)
            table.move_cursor(row=0)
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()
            self.assertEqual(table.cursor_row, 1)

            await pilot.press("up")
            await pilot.press("up")
            await pilot.pause()
            self.assertTrue(search.has_focus)

    async def test_initial_storage_load_refocuses_first_navigation_target(self) -> None:
        app = _DeferredRefreshApp()
        async with app.run_test(size=(100, 24)) as pilot:
            app.push_screen(StorageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StorageScreen)
            table = screen.query_one("#storage-table", DataTable)
            provider_filter = screen.query_one("#storage-provider-filter", Select)

            app.deliver_snapshot()
            await pilot.pause()
            await pilot.pause()

            self.assertEqual(table.row_count, 2)
            self.assertTrue(provider_filter.has_focus)

    async def test_predownload_uses_form_values(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(StorageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StorageScreen)
            screen.query_one("#storage-model-id", Input).value = "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
            screen.query_one("#storage-model-backend", Select).value = "llamacpp"
            screen.query_one("#storage-model-quant", Input).value = "Q4_K_M"
            screen.query_one("#storage-model-revision", Input).value = "main"
            screen.action_toggle_predownload_form()
            screen.action_predownload_selected()
            self.assertEqual(len(app.predownload_calls), 1)
            backend, model_id, quant, revision = app.predownload_calls[0]
            self.assertEqual(backend, BackendType.LLAMACPP)
            self.assertEqual(model_id, "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF")
            self.assertEqual(quant, "Q4_K_M")
            self.assertEqual(revision, "main")

    async def test_predownload_button_uses_form_values(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(StorageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StorageScreen)
            screen.action_toggle_predownload_form()
            screen.query_one("#storage-model-id", Input).value = "acme/model"
            screen.query_one("#storage-predownload-btn", Button).press()
            await pilot.pause()

        self.assertEqual(app.predownload_calls[0][1], "acme/model")

    async def test_predownload_key_acts_on_the_selected_row(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(StorageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StorageScreen)
            screen.action_predownload_selected()
            self.assertEqual(len(app.predownload_calls), 1)

    async def test_delete_requires_confirmation_before_calling_backend(self) -> None:
        app = _TestApp()
        async with app.run_test() as pilot:
            app.push_screen(StorageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, StorageScreen)
            first_model = StoredModelInfo(
                backend=BackendType.LLAMACPP,
                model_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
                source_volume="huggingface-cache",
            )
            screen._selected_model = first_model
            screen.action_delete_selected_model()
            await pilot.pause()
            self.assertEqual(app.delete_calls, [])
            confirm = app.screen
            assert isinstance(confirm, StorageDeleteConfirmScreen)
            self.assertIs(confirm.model, first_model)
            confirm.action_confirm_delete()
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
            self.assertEqual(screen.query_one("#storage-model-backend", Select).value, "vllm")

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


class StorageDeleteScopeTests(unittest.TestCase):
    def test_confirmation_names_the_other_quants_it_will_remove(self) -> None:
        """The dialog named one quant while deletion removed the repository.

        Rows are listed per quant, but _storage_delete_targets removes
        /hub/models--<repo> recursively, so confirming a 2.4 GiB row silently
        took every other cached quant of that model with it.
        """
        q4 = StoredModelInfo(
            backend=BackendType.LLAMACPP, model_id="unsloth/Qwen3-4B-GGUF",
            revision="abc", quant="Q4_K_M", size_bytes=2_600_000_000,
        )
        q8 = StoredModelInfo(
            backend=BackendType.LLAMACPP, model_id="unsloth/Qwen3-4B-GGUF",
            revision="abc", quant="Q8_0", size_bytes=4_300_000_000,
        )
        warning = Content.from_markup(
            StorageDeleteConfirmScreen(q4, also_removed=(q8,))._warning()
        ).plain
        self.assertIn("whole cached repository", warning)
        self.assertIn("Q8_0", warning)
        self.assertIn("6.4 GiB total", warning)

        only = Content.from_markup(StorageDeleteConfirmScreen(q4)._warning()).plain
        self.assertNotIn("whole cached repository", only)

    def test_sibling_rows_are_collected_from_the_loaded_snapshot(self) -> None:
        screen = StorageScreen()
        q4 = StoredModelInfo(
            backend=BackendType.LLAMACPP, model_id="unsloth/Qwen3-4B-GGUF",
            revision="abc", quant="Q4_K_M", size_bytes=1,
        )
        q8 = StoredModelInfo(
            backend=BackendType.LLAMACPP, model_id="unsloth/Qwen3-4B-GGUF",
            revision="abc", quant="Q8_0", size_bytes=2,
        )
        other = StoredModelInfo(
            backend=BackendType.LLAMACPP, model_id="unsloth/gemma-3-4b-it-GGUF",
            revision="def", quant="Q4_K_M", size_bytes=3,
        )
        screen._snapshot = StorageSnapshot(llamacpp_models=[q4, q8, other], vllm_models=[])
        self.assertEqual(screen._rows_removed_with(q4), (q8,))
        self.assertEqual(screen._rows_removed_with(other), ())


if __name__ == "__main__":
    unittest.main()
