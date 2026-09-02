"""Storage management screen for cached models across backends."""

from __future__ import annotations

from rich.markup import escape
from textual import events
from textual.actions import SkipAction
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import Button, DataTable, Footer, Input, OptionList, Select, Static
from textual.widgets.option_list import Option

from ...core.storage_costs import (
    MODAL_VOLUME_FREE_TIER_GIB_MONTH,
    estimate_monthly_storage_cost,
    gross_monthly_storage_cost_usd,
)
from ...protocol.enums import BackendType
from ...protocol.models import StorageSnapshot, StoredModelInfo
from ..navigation import (
    first_enabled_option_index,
    is_focusable_for_navigation,
    move_focus_across_option_lists,
    move_focus_across_widgets,
)
from ..workers import StorageFailed, StorageLoaded
from ..responsive import ViewportProfile, WidthMode
from ..widgets.adaptive_table import AdaptiveColumn, AdaptiveDataTable
from .copy_enabled import CopyEnabledScreen


def _human_bytes(size_bytes: int) -> str:
    size = float(size_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size_bytes} B"


def _format_money(value: float) -> str:
    return f"${value:,.2f}"


def _format_gib(value: float) -> str:
    if value >= 100:
        return f"{value:,.0f} GiB"
    if value >= 10:
        return f"{value:,.1f} GiB"
    return f"{value:,.2f} GiB"


def _format_free_tier(value_gib: float) -> str:
    if value_gib > 0 and value_gib % 1024 == 0:
        return f"{value_gib / 1024:,.0f} TiB"
    return _format_gib(value_gib)


def _model_label(row: StoredModelInfo) -> str:
    if row.incomplete:
        return f"{row.model_id} (INCOMPLETE)"
    return row.model_id


def _storage_row_key(row: StoredModelInfo) -> str:
    """Build a stable identity that survives table presentation changes."""
    return ":".join(
        (
            row.backend.value,
            row.model_id,
            row.revision or "",
            row.quant or "",
        )
    )


_WIDE = WidthMode.WIDE
_STANDARD = WidthMode.STANDARD
_COMPACT = WidthMode.COMPACT
_MINIMAL = WidthMode.MINIMAL

_STORAGE_COLUMNS = (
    AdaptiveColumn.visible(
        "backend",
        "backend",
        lambda row: row.backend.value,
        _WIDE,
        _STANDARD,
    ),
    AdaptiveColumn.visible(
        "model",
        "model",
        _model_label,
        _WIDE,
        _STANDARD,
        _COMPACT,
        _MINIMAL,
    ),
    AdaptiveColumn.visible(
        "revision",
        "revision",
        lambda row: row.revision or "-",
        _WIDE,
    ),
    AdaptiveColumn.visible(
        "quant",
        "quant",
        lambda row: row.quant or "-",
        _WIDE,
        _STANDARD,
    ),
    AdaptiveColumn.visible(
        "files",
        "files",
        lambda row: str(row.file_count),
        _WIDE,
    ),
    AdaptiveColumn.visible(
        "size",
        "size",
        lambda row: _human_bytes(row.size_bytes),
        _WIDE,
        _STANDARD,
        _COMPACT,
    ),
    AdaptiveColumn.visible(
        "cost",
        "list $/mo",
        lambda row: f"{_format_money(gross_monthly_storage_cost_usd(row.size_bytes))}/mo",
        _WIDE,
        _STANDARD,
        _COMPACT,
    ),
    AdaptiveColumn.visible(
        "summary",
        "details",
        lambda row: (
            f"{row.backend.value} · {_human_bytes(row.size_bytes)} · "
            f"{_format_money(gross_monthly_storage_cost_usd(row.size_bytes))}/mo"
        ),
        _MINIMAL,
    ),
)


def _is_focusable_for_arrow_navigation(widget: Widget) -> bool:
    return is_focusable_for_navigation(widget, check_size=True)


class StorageScreen(CopyEnabledScreen):
    """View and pre-download backend model caches."""

    BINDINGS = [
        Binding("up", "navigate_option_list_up", show=False, priority=True),
        Binding("down", "navigate_option_list_down", show=False, priority=True),
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("r", "refresh_storage", "Refresh", show=True),
        Binding("p", "predownload_selected", "Pre-download", show=True),
        Binding("x", "delete_selected_model", "Delete", show=True),
        Binding("/", "focus_model_filter", "Filter", show=True),
    ]
    NAVIGATION_ORDER = (
        "storage-backend-filter",
        "storage-table",
        "storage-model-id",
        "storage-model-backend",
        "storage-model-quant",
        "storage-model-revision",
    )

    def __init__(self, initial_backend: BackendType | None = None) -> None:
        super().__init__()
        self._initial_backend = initial_backend
        self._table_filter = ""

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="screen-scroll"):
            yield Static("[bold #7bf168]Storage[/]  [dim]Cached models, pre-download, delete[/dim]")
            yield Static("[dim]Storage status appears here.[/dim]", id="storage-status")
            yield Static("")
            yield Static("[bold]Backend filter[/bold]")
            yield OptionList(
                Option("  All backends", id="filter-all"),
                Option("  llama.cpp", id="filter-llamacpp"),
                Option("  vLLM", id="filter-vllm"),
                id="storage-backend-filter",
            )
            yield Static("")
            yield Static("[bold]Model inventory[/bold]")
            yield Input(
                placeholder="Filter models (type to filter)",
                id="storage-filter",
            )
            yield AdaptiveDataTable(id="storage-table")
            yield Static(
                "[dim]Tip: select a row to prefill the pre-download form. "
                "p pre-downloads; x deletes the selected model after confirmation.[/dim]",
                id="storage-hint",
            )
            yield Static("")
            yield Static("[bold]Pre-download a model[/bold]")
            yield Input(
                placeholder="Model id (e.g. Qwen/Qwen3-4B-Thinking-2507-FP8)",
                id="storage-model-id",
            )
            yield Static("Backend for pre-download", classes="form-label")
            yield Select(
                options=[("llama.cpp", "llamacpp"), ("vLLM", "vllm")],
                value=(self._initial_backend.value if self._initial_backend else "llamacpp"),
                allow_blank=False,
                id="storage-model-backend",
            )
            yield Input(
                placeholder="Quant pattern (llama.cpp only, optional)",
                id="storage-model-quant",
            )
            yield Input(
                placeholder="Revision (optional)",
                id="storage-model-revision",
            )
            yield Static("[dim]Press p to pre-download using these values.[/dim]")
        yield Footer()

    def on_mount(self) -> None:
        self._snapshot = StorageSnapshot(llamacpp_models=[], vllm_models=[])
        self._selected_filter: BackendType | None = self._initial_backend
        self._rows_by_key: dict[str, StoredModelInfo] = {}
        self._selected_model: StoredModelInfo | None = None
        self._was_suspended = False
        self._initial_focus_pending = True
        table = self.query_one("#storage-table", AdaptiveDataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.configure(
            _STORAGE_COLUMNS,
            row_key=_storage_row_key,
            profile=self.viewport_profile,
        )
        if self._initial_backend is not None:
            self.query_one("#storage-model-backend", Select).value = self._initial_backend.value
        backend_filter = self.query_one("#storage-backend-filter", OptionList)
        if self._selected_filter == BackendType.LLAMACPP:
            backend_filter.highlighted = 1
        elif self._selected_filter == BackendType.VLLM:
            backend_filter.highlighted = 2
        elif backend_filter.option_count > 0:
            backend_filter.highlighted = 0
        self.call_after_refresh(self._focus_first_visible_navigation_target)
        self._refresh_storage_snapshot()

    def on_screen_suspend(self, _: events.ScreenSuspend) -> None:
        self._was_suspended = True

    def on_screen_resume(self, _: events.ScreenResume) -> None:
        """Refresh storage when returning to this screen."""
        if self._was_suspended:
            self._was_suspended = False
            self._refresh_storage_snapshot()

    def viewport_profile_changed(
        self,
        profile: ViewportProfile,
        previous: ViewportProfile | None,
    ) -> None:
        """Change table columns while keeping the highlighted model."""
        _ = previous
        try:
            table = self.query_one("#storage-table", AdaptiveDataTable)
        except NoMatches:
            return
        table.set_viewport_profile(profile)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "storage-backend-filter":
            if event.option.id == "filter-llamacpp":
                self._selected_filter = BackendType.LLAMACPP
            elif event.option.id == "filter-vllm":
                self._selected_filter = BackendType.VLLM
            else:
                self._selected_filter = None
            self._render_table()
            return

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "storage-filter":
            self._table_filter = event.value
            self._render_table()

    def action_focus_model_filter(self) -> None:
        self.query_one("#storage-filter", Input).focus()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        row_key = str(event.row_key.value) if getattr(event, "row_key", None) is not None else ""
        self._apply_row_selection(row_key)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        row_key = str(event.row_key.value) if getattr(event, "row_key", None) is not None else ""
        self._apply_row_selection(row_key)

    def on_storage_loaded(self, message: StorageLoaded) -> None:
        self._snapshot = message.snapshot
        self._render_table()
        estimate = estimate_monthly_storage_cost(self._snapshot)
        self.query_one("#storage-status", Static).update(
            "[green]Storage refreshed.[/green] "
            f"{_human_bytes(estimate.total_size_bytes)} cached; "
            f"{_format_gib(estimate.billable_gib_month)} billable after "
            f"{_format_free_tier(MODAL_VOLUME_FREE_TIER_GIB_MONTH)} free; "
            f"est. {_format_money(estimate.estimated_monthly_cost_usd)}/mo. "
            "Use selected row or type a model to pre-download."
        )
        should_refocus = self._initial_focus_pending
        self._initial_focus_pending = False
        focused = self.focused
        if should_refocus or not isinstance(focused, Widget) or not _is_focusable_for_arrow_navigation(focused):
            self.call_after_refresh(self._focus_first_visible_navigation_target)

    def on_storage_failed(self, message: StorageFailed) -> None:
        self.query_one("#storage-status", Static).update(
            f"[yellow]Storage refresh failed:[/yellow] {message.error}"
        )

    def _render_table(self) -> None:
        table = self.query_one("#storage-table", AdaptiveDataTable)
        rows = self._snapshot.llamacpp_models + self._snapshot.vllm_models
        if self._selected_filter is not None:
            rows = [row for row in rows if row.backend == self._selected_filter]
        query = self._table_filter.strip().casefold()
        if query:
            rows = [row for row in rows if query in row.model_id.casefold()]
        self._rows_by_key = {_storage_row_key(row): row for row in rows}
        table.set_rows(rows)

    def _apply_row_selection(self, row_key: str) -> None:
        selected = self._rows_by_key.get(row_key)
        if selected is None:
            return
        self._selected_model = selected
        self.query_one("#storage-model-id", Input).value = selected.model_id
        self.query_one("#storage-model-backend", Select).value = selected.backend.value
        self.query_one("#storage-model-quant", Input).value = selected.quant or ""
        self.query_one("#storage-model-revision", Input).value = selected.revision or ""

    def action_refresh_storage(self) -> None:
        self._refresh_storage_snapshot(force=True)

    def _refresh_storage_snapshot(self, force: bool = False) -> None:
        self.query_one("#storage-status", Static).update("[dim]Refreshing storage snapshot...[/dim]")
        self.app.begin_storage_refresh(self, force=force)  # type: ignore[attr-defined]

    def action_predownload_selected(self) -> None:
        model_id = self.query_one("#storage-model-id", Input).value.strip()
        backend_raw = str(self.query_one("#storage-model-backend", Select).value).strip().lower()
        quant = self.query_one("#storage-model-quant", Input).value.strip() or None
        revision = self.query_one("#storage-model-revision", Input).value.strip() or None
        if not model_id:
            self.app.notify("Model id is required for pre-download.", severity="error", timeout=5)
            return
        if backend_raw not in {"llamacpp", "vllm"}:
            self.app.notify("Choose a backend for pre-download.", severity="error", timeout=5)
            return
        backend = BackendType(backend_raw)
        if backend == BackendType.VLLM:
            quant = None
        self.app.begin_storage_predownload(  # type: ignore[attr-defined]
            backend=backend,
            model_id=model_id,
            quant=quant,
            revision=revision,
        )

    def action_delete_selected_model(self) -> None:
        if self._selected_model is None:
            self.app.notify("Select a model row first to delete.", severity="warning", timeout=5)
            return
        self.app.push_screen(StorageDeleteConfirmScreen(self._selected_model))

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def action_navigate_option_list_down(self) -> None:
        if move_focus_across_option_lists(
            self,
            ("storage-backend-filter",),
            direction=1,
            is_focusable=_is_focusable_for_arrow_navigation,
        ):
            return
        focused = self.focused
        if isinstance(focused, DataTable) and focused.row_count > 0 and focused.cursor_row < focused.row_count - 1:
            raise SkipAction()
        if move_focus_across_widgets(
            self,
            self.NAVIGATION_ORDER,
            direction=1,
            is_focusable=_is_focusable_for_arrow_navigation,
            fallback_to_edge_if_focus_missing=True,
        ):
            return
        raise SkipAction()

    def action_navigate_option_list_up(self) -> None:
        if move_focus_across_option_lists(
            self,
            ("storage-backend-filter",),
            direction=-1,
            is_focusable=_is_focusable_for_arrow_navigation,
        ):
            return
        focused = self.focused
        if isinstance(focused, DataTable) and focused.row_count > 0 and focused.cursor_row > 0:
            raise SkipAction()
        if move_focus_across_widgets(
            self,
            self.NAVIGATION_ORDER,
            direction=-1,
            is_focusable=_is_focusable_for_arrow_navigation,
            fallback_to_edge_if_focus_missing=True,
        ):
            return
        raise SkipAction()

    def _focus_first_visible_navigation_target(self) -> None:
        for widget_id in self.NAVIGATION_ORDER:
            widget = self.query_one(f"#{widget_id}", Widget)
            if not _is_focusable_for_arrow_navigation(widget):
                continue
            widget.focus()
            if isinstance(widget, OptionList) and widget.highlighted is None:
                widget.highlighted = first_enabled_option_index(widget)
            return


class StorageDeleteConfirmScreen(CopyEnabledScreen):
    """Require explicit confirmation before deleting a cached model."""

    BINDINGS = [
        Binding("left,up", "previous_action", show=False, priority=True),
        Binding("right,down", "next_action", show=False, priority=True),
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("x", "confirm_delete", "Confirm delete", show=True),
    ]
    ACTION_IDS = ("delete-cancel", "delete-confirm")

    def __init__(self, model: StoredModelInfo) -> None:
        super().__init__()
        self.model = model

    def compose(self) -> ComposeResult:
        detail = [escape(self.model.backend.value)]
        if (self.model.quant or "").strip():
            detail.append(escape(self.model.quant.strip()))
        detail.append(_human_bytes(self.model.size_bytes))
        with VerticalScroll(classes="screen-scroll"):
            yield Static("[bold #7bf168]Delete Cached Model[/]")
            yield Static("")
            yield Static(
                f"Delete [bold]{escape(self.model.model_id)}[/bold]?\n"
                f"[dim]{' · '.join(detail)}[/dim]"
            )
            yield Static(
                "[yellow]This will remove the cached model files from provider storage.[/yellow]",
                id="delete-warning",
            )
            with Horizontal(id="delete-confirm-actions"):
                yield Button("Cancel", id="delete-cancel")
                yield Button("Delete model", id="delete-confirm", variant="error")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#delete-cancel", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "delete-cancel":
            self.action_cancel()
        elif event.button.id == "delete-confirm":
            self.action_confirm_delete()

    def action_cancel(self) -> None:
        self.app.pop_screen()

    def action_previous_action(self) -> None:
        move_focus_across_widgets(self, self.ACTION_IDS, -1)

    def action_next_action(self) -> None:
        move_focus_across_widgets(self, self.ACTION_IDS, 1)

    def action_confirm_delete(self) -> None:
        self.app.pop_screen()
        self.app.begin_storage_delete(self.model)  # type: ignore[attr-defined]
