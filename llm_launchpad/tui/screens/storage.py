"""Storage management screen for cached models across backends."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Input, OptionList, Static
from textual.widgets.option_list import Option

from ...protocol.enums import BackendType
from ...protocol.models import StorageSnapshot, StoredModelInfo
from ..workers import StorageFailed, StorageLoaded


def _human_bytes(size_bytes: int) -> str:
    size = float(size_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size_bytes} B"


class StorageScreen(Screen):
    """View and pre-download backend model caches."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("r", "refresh_storage", "Refresh", show=True),
        Binding("p", "predownload_selected", "Pre-download", show=True),
        Binding("x", "delete_selected_model", "Delete", show=True),
    ]

    def __init__(self, initial_backend: BackendType | None = None) -> None:
        super().__init__()
        self._initial_backend = initial_backend

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="menu-container"):
            yield Static("[bold cyan]Storage[/bold cyan]  [dim]Cached models and pre-download[/dim]")
            yield Static("")
            with Vertical(id="storage-controls"):
                yield Static("[bold]Backend filter[/bold]")
                yield OptionList(
                    Option("  All backends", id="filter-all"),
                    Option("  llama.cpp", id="filter-llamacpp"),
                    Option("  vLLM", id="filter-vllm"),
                    id="storage-backend-filter",
                )
                yield Static("")
                yield Static("[bold]Actions[/bold]")
                yield OptionList(
                    Option("  Refresh storage snapshot", id="refresh"),
                    Option("  Pre-download model", id="predownload"),
                    Option("  Delete selected model", id="delete"),
                    id="storage-action-list",
                )
            yield Static("")
            yield Static("[bold]Model inventory[/bold]")
            yield DataTable(id="storage-table")
            yield Static("[dim]Tip: select a row to prefill model fields.[/dim]", id="storage-hint")
            yield Static("")
            yield Input(placeholder="Model id (e.g. Qwen/Qwen3-4B-Thinking-2507-FP8)", id="storage-model-id")
            yield Input(
                placeholder="Backend for pre-download: llamacpp or vllm",
                id="storage-model-backend",
            )
            yield Input(placeholder="Quant pattern (llama.cpp only, optional)", id="storage-model-quant")
            yield Input(placeholder="Revision (optional)", id="storage-model-revision")
            yield Static("[dim]Press p to pre-download using these values.[/dim]", id="storage-status")
        yield Footer()

    def on_mount(self) -> None:
        self._snapshot = StorageSnapshot(llamacpp_models=[], vllm_models=[])
        self._selected_filter: BackendType | None = self._initial_backend
        self._rows_by_key: dict[str, StoredModelInfo] = {}
        self._selected_model: StoredModelInfo | None = None
        table = self.query_one("#storage-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns("backend", "model", "revision", "quant", "files", "size")
        if self._initial_backend is not None:
            self.query_one("#storage-model-backend", Input).value = self._initial_backend.value
        self.app.begin_storage_refresh(self)  # type: ignore[attr-defined]

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

        if event.option_list.id == "storage-action-list":
            if event.option.id == "refresh":
                self.action_refresh_storage()
            elif event.option.id == "predownload":
                self.action_predownload_selected()
            elif event.option.id == "delete":
                self.action_delete_selected_model()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        row_key = str(event.row_key.value) if getattr(event, "row_key", None) is not None else ""
        selected = self._rows_by_key.get(row_key)
        if selected is None:
            return
        self._selected_model = selected
        self.query_one("#storage-model-id", Input).value = selected.model_id
        self.query_one("#storage-model-backend", Input).value = selected.backend.value
        self.query_one("#storage-model-quant", Input).value = selected.quant or ""
        self.query_one("#storage-model-revision", Input).value = selected.revision or ""

    def on_storage_loaded(self, message: StorageLoaded) -> None:
        self._snapshot = message.snapshot
        self._render_table()
        self.query_one("#storage-status", Static).update(
            "[green]Storage refreshed.[/green] Use selected row or type a model to pre-download."
        )

    def on_storage_failed(self, message: StorageFailed) -> None:
        self.query_one("#storage-status", Static).update(
            f"[yellow]Storage refresh failed:[/yellow] {message.error}"
        )

    def _render_table(self) -> None:
        table = self.query_one("#storage-table", DataTable)
        table.clear(columns=False)
        self._rows_by_key = {}

        rows = self._snapshot.llamacpp_models + self._snapshot.vllm_models
        if self._selected_filter is not None:
            rows = [row for row in rows if row.backend == self._selected_filter]

        for index, row in enumerate(rows):
            row_key = f"{row.backend.value}:{index}:{row.model_id}"
            self._rows_by_key[row_key] = row
            table.add_row(
                row.backend.value,
                row.model_id,
                row.revision or "-",
                row.quant or "-",
                str(row.file_count),
                _human_bytes(row.size_bytes),
                key=row_key,
            )

    def action_refresh_storage(self) -> None:
        self.query_one("#storage-status", Static).update("[dim]Refreshing storage snapshot...[/dim]")
        self.app.begin_storage_refresh(self)  # type: ignore[attr-defined]

    def action_predownload_selected(self) -> None:
        model_id = self.query_one("#storage-model-id", Input).value.strip()
        backend_raw = self.query_one("#storage-model-backend", Input).value.strip().lower()
        quant = self.query_one("#storage-model-quant", Input).value.strip() or None
        revision = self.query_one("#storage-model-revision", Input).value.strip() or None
        if not model_id:
            self.app.notify("Model id is required for pre-download.", severity="error", timeout=5)
            return
        if backend_raw not in {"llamacpp", "vllm"}:
            self.app.notify("Backend must be either 'llamacpp' or 'vllm'.", severity="error", timeout=5)
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
        self.app.begin_storage_delete(self._selected_model)  # type: ignore[attr-defined]

    def action_pop_screen(self) -> None:
        self.app.pop_screen()
