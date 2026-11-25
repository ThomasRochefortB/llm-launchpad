"""Storage management screen for cached models and provider disks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from rich.markup import escape

from ..format import clip, format_free_tier, format_gib, format_money
from textual import events
from textual.actions import SkipAction
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.css.query import NoMatches
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Button, DataTable, Input, OptionList, Select, Static

from ...core.deploy_log_summary import SUMMARY_SPINNER_FRAMES
from ...core.storage_costs import (
    MODAL_VOLUME_FREE_TIER_GIB_MONTH,
    allocated_monthly_storage_cost_usd,
    estimate_monthly_storage_cost,
    gross_monthly_storage_cost_usd,
    marginal_monthly_storage_savings_usd,
)
from ...core.storage_resources import storage_scope_label
from ...protocol.enums import BackendType, ComputeProvider
from ...protocol.models import StorageSnapshot, StoredModelInfo
from ..navigation import (
    first_enabled_option_index,
    is_focusable_for_navigation,
    move_focus_across_widgets,
)
from ..workers import PrimeDisksFailed, PrimeDisksLoaded, StorageFailed, StorageLoaded
from ..responsive import ViewportProfile, WidthMode
from ..widgets.adaptive_table import AdaptiveColumn, AdaptiveDataTable
from ..widgets.fitted_footer import FittedFooter
from .copy_enabled import CopyEnabledScreen
from ..visual import labelled_rows_markup, screen_title


def _human_bytes(size_bytes: int) -> str:
    size = float(size_bytes)
    # Binary divisor, binary labels: the same screen reports GiB through
    # format_gib, and calling 1024-based units "GB" understates a model by 7%.
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size_bytes} B"


def _model_label(row: StoredModelInfo) -> str:
    if row.incomplete:
        return f"{row.model_id} (INCOMPLETE)"
    return row.model_id


# A model cached at several quants is one row per quant, so each can be deleted
# on its own; the name is written once and the rows under it only say what
# differs. Keyed by row identity for the table currently shown.
_CONTINUATION_MARK = "  └"


def _backend_display_cell(row: StoredModelInfo) -> Content:
    if row.model_id.startswith("prime:"):
        return Content.from_markup("[dim]disk[/dim]")
    if row.backend == BackendType.VLLM:
        return Content.from_markup("[$primary]vLLM[/]")
    return Content.from_markup("[$secondary]llama.cpp[/]")


def _row_monthly_cost_usd(row: StoredModelInfo, *, modal_total_bytes: int | None = None) -> float:
    """Compute effective monthly cost: Prime uses hourly rate; Modal applies the 1 TiB free tier."""
    if row.model_id.startswith("prime:") and row.source_volume.startswith("$") and row.source_volume.endswith("/hr"):
        try:
            return float(row.source_volume[1:-3]) * 730.0
        except ValueError:
            pass
    total_bytes = row.size_bytes if modal_total_bytes is None else modal_total_bytes
    return allocated_monthly_storage_cost_usd(row.size_bytes, total_bytes)


def _model_max_width(profile: ViewportProfile | None) -> int:
    """Keep long Hugging Face repo IDs from pushing cost off the right edge."""
    if profile is None:
        return 32
    if profile.width >= 130:
        return 54
    if profile.width >= 100:
        return 30
    if profile.width >= 80:
        return 25
    return 22


def _grouped_model_label(
    continuations: set[str],
    *,
    profile_getter: Callable[[], ViewportProfile | None] | None = None,
) -> Callable[[StoredModelInfo], Content]:
    def label(row: StoredModelInfo) -> Content:
        if _storage_row_key(row) in continuations:
            suffix = " [$warning](INCOMPLETE)[/]" if row.incomplete else ""
            return Content.from_markup(f"[dim]{_CONTINUATION_MARK}[/dim]{suffix}")
        max_len = _model_max_width(profile_getter() if profile_getter else None)
        if row.incomplete:
            budget = max(12, max_len - len(" (INCOMPLETE)"))
            name = escape(clip(row.model_id, budget))
            return Content.from_markup(f"[bold]{name}[/bold] [$warning](INCOMPLETE)[/]")
        return Content.from_markup(f"[bold]{escape(clip(row.model_id, max_len))}[/bold]")

    return label


def _dim_or_value(text: str) -> Content:
    if text == "-":
        return Content.from_markup("[dim]-[/dim]")
    return Content.from_markup(escape(text))


def _short_revision(row: StoredModelInfo) -> str:
    """A commit hash is only ever compared, and seven characters do that."""
    revision = row.revision or ""
    if len(revision) == 40 and all(char in "0123456789abcdef" for char in revision):
        return revision[:7]
    return revision or "-"


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
        _backend_display_cell,
        _WIDE,
        _STANDARD,
    ),
    AdaptiveColumn.visible(
        "model",
        "model",
        lambda row: Content.from_markup(f"[bold]{escape(_model_label(row))}[/bold]"),
        _WIDE,
        _STANDARD,
        _COMPACT,
        _MINIMAL,
    ),
    AdaptiveColumn.visible(
        "revision",
        "revision",
        lambda row: _dim_or_value(_short_revision(row)),
        _WIDE,
    ),
    AdaptiveColumn.visible(
        "quant",
        "quant",
        lambda row: _dim_or_value(row.quant or "-"),
        _WIDE,
        _STANDARD,
    ),
    AdaptiveColumn.visible(
        "files",
        "files",
        lambda row: str(row.file_count),
        _WIDE,
        justify="right",
    ),
    AdaptiveColumn.visible(
        "size",
        "size",
        lambda row: _human_bytes(row.size_bytes),
        _WIDE,
        _STANDARD,
        _COMPACT,
        justify="right",
    ),
    AdaptiveColumn.visible(
        "cost",
        "$/mo",
        lambda row: f"{format_money(_row_monthly_cost_usd(row))}/mo",
        _WIDE,
        _STANDARD,
        _COMPACT,
        justify="right",
    ),
    AdaptiveColumn.visible(
        "summary",
        "details",
        lambda row: (
            f"{'disk' if row.model_id.startswith('prime:') else ('vLLM' if row.backend == BackendType.VLLM else 'llama.cpp')} · "
            f"{_human_bytes(row.size_bytes)} · "
            f"{format_money(_row_monthly_cost_usd(row))}/mo"
        ),
        _MINIMAL,
    ),
)


def _storage_columns(
    continuations: set[str],
    *,
    profile_getter: Callable[[], ViewportProfile | None] | None = None,
    total_bytes_getter: Callable[[], int] | None = None,
) -> tuple[AdaptiveColumn, ...]:
    """The inventory columns, with repeated model names grouped and free tier applied."""
    grouped = _grouped_model_label(continuations, profile_getter=profile_getter)

    def cost_cell(row: StoredModelInfo) -> str:
        total = total_bytes_getter() if total_bytes_getter is not None else row.size_bytes
        return f"{format_money(_row_monthly_cost_usd(row, modal_total_bytes=total))}/mo"

    def summary_cell(row: StoredModelInfo) -> str:
        total = total_bytes_getter() if total_bytes_getter is not None else row.size_bytes
        kind = (
            "disk"
            if row.model_id.startswith("prime:")
            else ("vLLM" if row.backend == BackendType.VLLM else "llama.cpp")
        )
        return (
            f"{kind} · {_human_bytes(row.size_bytes)} · "
            f"{format_money(_row_monthly_cost_usd(row, modal_total_bytes=total))}/mo"
        )

    columns: list[AdaptiveColumn] = []
    for column in _STORAGE_COLUMNS:
        if column.key == "model":
            columns.append(replace(column, value=grouped))
        elif column.key == "cost":
            columns.append(replace(column, value=cost_cell))
        elif column.key == "summary":
            columns.append(replace(column, value=summary_cell))
        else:
            columns.append(column)
    return tuple(columns)


def _current_row_key(table: AdaptiveDataTable) -> str | None:
    """The key under the table's cursor, or None when there are no rows."""
    if table.row_count == 0 or not table.columns:
        return None
    try:
        from textual.coordinate import Coordinate

        return str(
            table.coordinate_to_cell_key(Coordinate(table.cursor_row, 0)).row_key.value
        )
    except Exception:
        return None


def _is_focusable_for_arrow_navigation(widget: Widget) -> bool:
    return is_focusable_for_navigation(widget, check_size=True)


class StorageScreen(CopyEnabledScreen):
    """View provider storage: Modal cache, Prime disks, and Vast rentals."""

    BINDINGS = [
        Binding("up", "navigate_option_list_up", show=False, priority=True),
        Binding("down", "navigate_option_list_down", show=False, priority=True),
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("r", "refresh_storage", "Refresh", show=True),
        Binding("p", "predownload_selected", "Pre-download", show=True),
        Binding("n", "toggle_predownload_form", "New download", show=True),
        Binding("x", "delete_selected_model", "Delete", show=True),
        Binding("/", "focus_model_filter", "Filter", show=False),
    ]
    NAVIGATION_ORDER = (
        "storage-provider-filter",
        "storage-filter",
        "storage-table",
        "storage-model-id",
        "storage-model-backend",
        "storage-model-quant",
        "storage-model-revision",
        "storage-predownload-btn",
    )

    def __init__(self, initial_backend: BackendType | None = None) -> None:
        super().__init__()
        self._initial_backend = initial_backend
        self._selected_provider = ComputeProvider.MODAL
        self._prime_disks: list[object] = []
        self._prime_error: str | None = None
        self._selected_prime_disk_id: str | None = None
        self._predownload_expanded = False
        self._continuations: set[str] = set()
        self._table_filter = initial_backend.value if initial_backend else ""
        self._storage_refreshing = False
        self._storage_has_snapshot = False
        self._prime_has_snapshot = False
        self._refresh_spinner_index = 0
        self._refresh_spinner_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="storage-scroll", classes="screen-scroll"):
            yield Static(screen_title("Storage"), id="storage-title")
            yield Static("[dim]Storage status appears here.[/dim]", id="storage-status")
            # Provider and backend ride side by side: each extra vertical filter
            # row is a table row lost on an 80x24 terminal, where the inventory
            # must stay visible.
            with Horizontal(id="storage-filter-row"):
                with Vertical(id="storage-provider-group"):
                    yield Static("", id="storage-provider-label")
                    yield Select(
                        options=[
                            ("Modal · shared volume cache", "modal"),
                            ("Prime · persistent cache disks", "prime"),
                            ("Vast · rental-local disks", "vast"),
                        ],
                        value="modal",
                        allow_blank=False,
                        id="storage-provider-filter",
                    )
                # One text filter covers the model and backend columns; a
                # separate backend dropdown restated the table's first column.
                with Vertical(id="storage-backend-group"):
                    yield Static("[bold]Filter[/bold]")
                    yield Input(
                        value=self._initial_backend.value if self._initial_backend else "",
                        placeholder="model or backend",
                        id="storage-filter",
                    )
            table = AdaptiveDataTable(id="storage-table", fit_to_rows=True, min_fit_rows=5)
            table.border_title = "Cached Models"
            yield table
            detail = Static("", id="storage-detail", classes="hidden")
            detail.border_title = "Selected Model"
            yield detail
            # A table showing only its header cannot say whether nothing is
            # cached, the filter excluded everything, or the refresh failed.
            yield Static("", id="storage-empty", classes="hidden")
            # Only Prime's and Vast's inventory limits are news; the footer
            # carries the keys.
            yield Static("", id="storage-hint", classes="hidden")
            predownload = Vertical(id="storage-predownload-form")
            predownload.border_title = "Pre-download to Modal Cache"
            with predownload:
                yield Static("[bold]Model ID[/bold]", classes="form-label")
                yield Input(
                    placeholder="e.g. Qwen/Qwen3-4B-Thinking-2507-FP8 or unsloth/GLM-4.7-Flash-GGUF",
                    id="storage-model-id",
                )
                with Horizontal(id="storage-predownload-options"):
                    with Vertical(classes="storage-predownload-col"):
                        yield Static("[bold]Backend[/bold]", classes="form-label")
                        yield Select(
                            options=[("llama.cpp", "llamacpp"), ("vLLM", "vllm")],
                            value=(self._initial_backend.value if self._initial_backend else "llamacpp"),
                            allow_blank=False,
                            id="storage-model-backend",
                        )
                    with Vertical(classes="storage-predownload-col"):
                        yield Static("[bold]Quant[/bold] [dim](llama.cpp)[/dim]", classes="form-label")
                        yield Input(
                            placeholder="e.g. Q4_K_M (optional)",
                            id="storage-model-quant",
                        )
                    with Vertical(classes="storage-predownload-col"):
                        yield Static("[bold]Revision[/bold]", classes="form-label")
                        yield Input(
                            placeholder="branch or commit (optional)",
                            id="storage-model-revision",
                        )
                with Horizontal(id="storage-predownload-actions"):
                    yield Button("Pre-download", id="storage-predownload-btn", variant="primary")
                    yield Button("Hide", id="storage-predownload-cancel")
                    yield Static("[dim]Downloads into the shared Modal cache volume.[/dim]")
        yield FittedFooter()

    def on_mount(self) -> None:
        self._snapshot = StorageSnapshot(llamacpp_models=[], vllm_models=[])
        self._rows_by_key: dict[str, StoredModelInfo] = {}
        self._selected_model: StoredModelInfo | None = None
        self._was_suspended = False
        self._initial_focus_pending = True
        self._restored_highlight_key: str | None = None
        self._prefilled_model_id: str | None = None
        table = self.query_one("#storage-table", AdaptiveDataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.configure(
            _storage_columns(
                self._continuations,
                profile_getter=lambda: self.viewport_profile,
                total_bytes_getter=lambda: self._snapshot.total_size_bytes,
            ),
            row_key=_storage_row_key,
            profile=self.viewport_profile,
        )
        if self._initial_backend is not None:
            self.query_one("#storage-model-backend", Select).value = self._initial_backend.value
        self._sync_predownload_visibility()
        self._render_scope()
        self.call_after_refresh(self._focus_first_visible_navigation_target)
        self._refresh_storage_snapshot()

    def _render_scope(self) -> None:
        # The provider dropdown already shows "Modal · shared volume cache";
        # repeating it in the label cut the part only the label carried --
        # what the inventory can show -- off at the column edge. That part
        # now leads the hint under the table it describes.
        # Modal's note ("file inventory available") describes the table the
        # reader is looking at; only Prime's and Vast's limits are news.
        scope = storage_scope_label(self._selected_provider)
        _name, _sep, note = scope.partition(" (")
        note = note.rstrip(")") if self._selected_provider != ComputeProvider.MODAL else ""
        lead = f"{escape(note[:1].upper() + note[1:])}. " if note else ""
        try:
            self.query_one("#storage-provider-label", Static).update("[bold]Provider[/bold]")
            hint = self.query_one("#storage-hint", Static)
            hint.update(f"[dim]{lead.strip()}[/dim]" if lead else "")
            hint.set_class(not lead, "hidden")
        except Exception:
            pass

    def on_screen_suspend(self, _: events.ScreenSuspend) -> None:
        self._was_suspended = True
        if self._refresh_spinner_timer is not None:
            self._refresh_spinner_timer.pause()

    def on_screen_resume(self, _: events.ScreenResume) -> None:
        """Refresh storage when returning to this screen."""
        if self._refresh_spinner_timer is not None and self._storage_refreshing:
            self._refresh_spinner_timer.resume()
        if self._was_suspended:
            self._was_suspended = False
            self._refresh_storage_snapshot()

    def _spinner_frame(self) -> str:
        return SUMMARY_SPINNER_FRAMES[self._refresh_spinner_index % len(SUMMARY_SPINNER_FRAMES)]

    def _set_storage_refreshing(self, active: bool) -> None:
        self._storage_refreshing = active
        if active:
            if self._refresh_spinner_timer is None:
                self._refresh_spinner_timer = self.set_interval(
                    0.15,
                    self._tick_refresh_spinner,
                    name="storage-refresh-spinner",
                )
            else:
                self._refresh_spinner_timer.resume()
        elif self._refresh_spinner_timer is not None:
            self._refresh_spinner_timer.stop()
            self._refresh_spinner_timer = None

    def _tick_refresh_spinner(self) -> None:
        if not self._storage_refreshing:
            self._set_storage_refreshing(False)
            return
        if not self.is_current:
            return
        self._refresh_spinner_index = (
            self._refresh_spinner_index + 1
        ) % len(SUMMARY_SPINNER_FRAMES)
        self._render_storage_status()

    def _render_storage_status(self) -> None:
        try:
            status = self.query_one("#storage-status", Static)
        except NoMatches:
            return
        frame = self._spinner_frame()
        if self._selected_provider == ComputeProvider.VAST:
            status.update(
                "[dim]Vast has no separate storage inventory; rental disks are destroyed with the rental.[/dim]"
            )
            return
        if self._selected_provider == ComputeProvider.PRIME:
            if self._storage_refreshing and not self._prime_has_snapshot:
                status.update(
                    f"[bold $primary]{frame}[/] [bold]Refreshing Prime disks…[/bold]"
                )
                return
            from ...core.prime_disks import RetainedPrimeDisk

            total = sum(
                (disk.size_gb if isinstance(disk, RetainedPrimeDisk) else int(getattr(disk, "size_gb", 0) or 0))
                for disk in self._prime_disks
            )
            hourly = sum(float(getattr(disk, "price_per_hour_usd", 0.0) or 0.0) for disk in self._prime_disks)
            rate_part = f" · ~{format_money(hourly * 730.0)}/mo" if hourly > 0 else ""
            color = "$warning" if self._prime_disks else "$success"
            if self._storage_refreshing:
                status.update(
                    f"[{color}]{len(self._prime_disks)} Prime disk(s)[/{color}] "
                    f"[dim]· {total} GB retained{rate_part}[/dim]  "
                    f"[bold $primary]{frame} Refreshing…[/]"
                )
            else:
                status.update(
                    f"[{color}]{len(self._prime_disks)} Prime disk(s)[/{color}] "
                    f"[dim]· {total} GB retained{rate_part} · stop keeps disks billing; "
                    f"delete unneeded ones[/dim]"
                )
            return

        if self._storage_refreshing and not self._storage_has_snapshot:
            status.update(
                f"[bold $primary]{frame}[/] [bold]Refreshing storage…[/bold] "
                "[dim]scanning Modal volume[/dim]"
            )
            return

        estimate = estimate_monthly_storage_cost(self._snapshot)
        # One line, and only the numbers: the instruction that used to trail it
        # wrapped the status onto a second row and repeated the hint below.
        # The free-tier arithmetic only matters once something is billable.
        free_tier = format_free_tier(MODAL_VOLUME_FREE_TIER_GIB_MONTH)
        total_models = self._snapshot.total_models
        llama_count = len(self._snapshot.llamacpp_models)
        vllm_count = len(self._snapshot.vllm_models)
        breakdown_bits: list[str] = []
        if llama_count:
            breakdown_bits.append(f"{llama_count} llama.cpp")
        if vllm_count:
            breakdown_bits.append(f"{vllm_count} vLLM")
        model_noun = "model" if total_models == 1 else "models"
        count_part = (
            f" ({total_models} {model_noun} · {', '.join(breakdown_bits)})"
            if total_models
            else ""
        )
        if estimate.billable_gib_month > 0:
            cost_detail = (
                f"· {format_gib(estimate.billable_gib_month)} billable past "
                f"{free_tier} free · est. {format_money(estimate.estimated_monthly_cost_usd)}/mo"
            )
        elif estimate.total_size_bytes > 0:
            used_gib = estimate.total_size_bytes / (1024.0**3)
            pct = min(100, int(round((used_gib / MODAL_VOLUME_FREE_TIER_GIB_MONTH) * 100)))
            pct_note = f"{pct}% of {free_tier} free" if pct >= 1 else f"<1% of {free_tier} free"
            cost_detail = f"· {pct_note} · est. {format_money(estimate.estimated_monthly_cost_usd)}/mo"
        else:
            cost_detail = f"· est. {format_money(estimate.estimated_monthly_cost_usd)}/mo"
        refresh_badge = f"  [bold $primary]{frame} Refreshing…[/]" if self._storage_refreshing else ""
        status.update(
            f"[$success]{_human_bytes(estimate.total_size_bytes)} cached[/$success]"
            f"[dim]{count_part} {cost_detail}[/dim]{refresh_badge}"
        )

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
        if getattr(self, "_rows_by_key", None) is not None:
            table._rebuild()
            self._update_selected_detail()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "storage-provider-filter":
            return
        value = str(event.value or "").strip().lower()
        if value == "prime":
            provider = ComputeProvider.PRIME
        elif value == "vast":
            provider = ComputeProvider.VAST
        else:
            provider = ComputeProvider.MODAL
        if provider == self._selected_provider:
            return
        self._selected_provider = provider
        self._selected_model = None
        self._selected_prime_disk_id = None
        self._render_scope()
        self._refresh_storage_snapshot()

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
        # Repopulating the table moves its cursor, and Textual reports that as
        # a highlight exactly like a keypress would -- asynchronously, so a
        # flag held across set_rows is already clear by the time this arrives.
        # Prefilling on it meant a background snapshot refresh, which lands on
        # a timer, overwrote whatever the user was part-way through typing into
        # the pre-download form: "Qwen" became the highlighted row's full id
        # mid-word. The row the rebuild restored is named instead, and that one
        # replay is skipped; moving to any other row still prefills.
        row_key = str(event.row_key.value) if getattr(event, "row_key", None) is not None else ""
        if row_key and row_key == self._restored_highlight_key:
            self._restored_highlight_key = None
            if self._form_is_user_edited():
                return
        self._apply_row_selection(row_key)

    def _form_is_user_edited(self) -> bool:
        """Whether the model id differs from whatever was last prefilled there.

        The first population should prefill -- that is what establishes the
        default target -- so this is about protecting typing, not about
        refusing every repeat.
        """
        try:
            current = self.query_one("#storage-model-id", Input).value
        except Exception:
            return False
        return current != (self._prefilled_model_id or "")

    def on_storage_loaded(self, message: StorageLoaded) -> None:
        self._snapshot = message.snapshot
        self._storage_has_snapshot = True
        if self._selected_provider != ComputeProvider.MODAL:
            self._render_table()
            return
        self._set_storage_refreshing(message.is_stale)
        self._render_table()
        self._render_storage_status()
        should_refocus = self._initial_focus_pending
        self._initial_focus_pending = False
        focused = self.focused
        if should_refocus or not isinstance(focused, Widget) or not _is_focusable_for_arrow_navigation(focused):
            self.call_after_refresh(self._focus_first_visible_navigation_target)

    def on_storage_failed(self, message: StorageFailed) -> None:
        if self._selected_provider != ComputeProvider.MODAL:
            return
        self._set_storage_refreshing(False)
        self.query_one("#storage-status", Static).update(
            f"[$warning]Storage refresh failed:[/$warning] {escape(message.error)}"
        )

    def on_prime_disks_loaded(self, message: PrimeDisksLoaded) -> None:
        self._prime_disks = list(message.disks)
        self._prime_error = None
        self._prime_has_snapshot = True
        if self._selected_provider != ComputeProvider.PRIME:
            return
        self._set_storage_refreshing(False)
        self._render_storage_status()
        self._render_table()

    def on_prime_disks_failed(self, message: PrimeDisksFailed) -> None:
        self._prime_disks = []
        self._prime_error = message.error
        if self._selected_provider != ComputeProvider.PRIME:
            return
        self._set_storage_refreshing(False)
        self.query_one("#storage-status", Static).update(
            f"[$warning]Prime disk refresh failed:[/$warning] {escape(message.error)}"
        )
        self._render_table()

    def _prime_rows(self) -> list[StoredModelInfo]:
        rows: list[StoredModelInfo] = []
        for disk in self._prime_disks:
            disk_id = str(getattr(disk, "id", "") or "").strip()
            if not disk_id:
                continue
            size_gb = getattr(disk, "size_gb", 0) or 0
            try:
                size_bytes = int(float(size_gb) * (1024.0**3))
            except (TypeError, ValueError):
                size_bytes = 0
            location = str(getattr(disk, "location", "") or "")
            status = str(getattr(disk, "status", "") or "")
            price = getattr(disk, "price_per_hour_usd", 0.0) or 0.0
            managed = bool(getattr(disk, "managed", True))
            quant = f"{size_gb} GB" + ("" if managed else " · other")
            rows.append(
                StoredModelInfo(
                    backend=BackendType.LLAMACPP,
                    model_id=f"prime:{disk_id}",
                    revision=location or status or "-",
                    quant=quant,
                    size_bytes=size_bytes,
                    file_count=0,
                    source_volume=f"${float(price):.4f}/hr" if price else "prime-disk",
                )
            )
        query = self._table_filter.strip().casefold()
        if query:
            rows = [row for row in rows if query in row.model_id.casefold()]
        return rows

    def _render_table(self) -> None:
        table = self.query_one("#storage-table", AdaptiveDataTable)
        if self._selected_provider == ComputeProvider.VAST:
            self._rows_by_key = {}
            self._selected_model = None
            table.set_rows([])
            self._render_empty_state([])
            return
        if self._selected_provider == ComputeProvider.PRIME:
            rows = self._prime_rows()
            self._rows_by_key = {_storage_row_key(row): row for row in rows}
            if (
                self._selected_model is not None
                and _storage_row_key(self._selected_model) not in self._rows_by_key
            ):
                self._selected_model = None
                self._selected_prime_disk_id = None
            table.set_rows(rows)
            self._restored_highlight_key = _current_row_key(table)
            self._render_empty_state(rows)
            return
        rows = self._snapshot.llamacpp_models + self._snapshot.vllm_models
        query = self._table_filter.strip().casefold()
        if query:
            rows = [
                row for row in rows
                if query in row.model_id.casefold() or query in row.backend.value
            ]
        self._continuations.clear()
        self._continuations.update(
            _storage_row_key(row)
            for previous, row in zip(rows, rows[1:], strict=False)
            if (previous.backend, previous.model_id) == (row.backend, row.model_id)
        )
        self._rows_by_key = {_storage_row_key(row): row for row in rows}
        if (
            self._selected_model is not None
            and _storage_row_key(self._selected_model) not in self._rows_by_key
        ):
            self._selected_model = None
        table.set_rows(rows)
        self._restored_highlight_key = _current_row_key(table)
        self._render_empty_state(rows)

    def _render_empty_state(self, rows: list[StoredModelInfo]) -> None:
        """Say why the inventory is empty, distinguishing cause from failure."""
        empty = self.query_one("#storage-empty", Static)
        table = self.query_one("#storage-table", AdaptiveDataTable)
        table.border_title = {
            ComputeProvider.VAST: "Vast Rental Disks",
            ComputeProvider.PRIME: "Prime Disks",
        }.get(self._selected_provider, "Cached Models")
        if rows:
            empty.add_class("hidden")
            table.remove_class("hidden")
            if self._selected_model is None:
                key = _current_row_key(table)
                if key and key in self._rows_by_key:
                    self._selected_model = self._rows_by_key[key]
                    if self._selected_provider == ComputeProvider.PRIME:
                        self._selected_prime_disk_id = self._selected_model.model_id.removeprefix("prime:")
            self._update_selected_detail()
            return
        self._update_selected_detail()
        if self._selected_provider == ComputeProvider.VAST:
            empty.update(
                "[dim]Vast rentals use rental-local disks that are destroyed with "
                "the rental. There is no separate cache to inventory or delete; "
                "`stop` deletes the disk along with cached models.[/dim]"
            )
            empty.remove_class("hidden")
            table.add_class("hidden")
            return
        if self._selected_provider == ComputeProvider.PRIME:
            if self._prime_error:
                empty.update(
                    f"[$warning]Prime disk inventory unavailable:[/$warning] {escape(self._prime_error)}"
                )
            else:
                empty.update(
                    "[dim]No Prime disks billing. Disks remain billable after a pod "
                    "stops; delete unneeded ones with prime-disks delete.[/dim]"
                )
            empty.remove_class("hidden")
            table.add_class("hidden")
            return
        cached_anything = bool(
            self._snapshot.llamacpp_models or self._snapshot.vllm_models
        )
        if cached_anything:
            message = "[dim]No cached model matches the current filter.[/dim]"
        else:
            message = (
                "[dim]No models cached yet · n to pre-download one, or deploy "
                "and it is cached for you[/dim]"
            )
        empty.update(message)
        empty.remove_class("hidden")
        table.add_class("hidden")

    def _apply_row_selection(self, row_key: str) -> None:
        selected = self._rows_by_key.get(row_key)
        if selected is None:
            return
        self._selected_model = selected
        if self._selected_provider == ComputeProvider.PRIME:
            disk_id = selected.model_id.removeprefix("prime:")
            self._selected_prime_disk_id = disk_id
            self._update_selected_detail()
            return
        self._selected_prime_disk_id = None
        self._prefilled_model_id = selected.model_id
        self.query_one("#storage-model-id", Input).value = selected.model_id
        self.query_one("#storage-model-backend", Select).value = selected.backend.value
        self.query_one("#storage-model-quant", Input).value = selected.quant or ""
        self.query_one("#storage-model-revision", Input).value = selected.revision or ""
        self._update_selected_detail()

    def _update_selected_detail(self) -> None:
        """Render the compact selected-model or selected-disk inspector card."""
        try:
            detail = self.query_one("#storage-detail", Static)
        except NoMatches:
            return
        selected = self._selected_model
        if selected is None or not self._rows_by_key or self._predownload_expanded:
            detail.add_class("hidden")
            return
        profile = self.viewport_profile
        compact_height = profile.short or profile.constrained
        detail.remove_class("hidden")

        if self._selected_provider == ComputeProvider.PRIME:
            disk_id = selected.model_id.removeprefix("prime:")
            detail.border_title = f"{disk_id} · Prime Disk"
            location = selected.revision or "-"
            rate = selected.source_volume or "-"
            if compact_height:
                detail.update(
                    f"[bold]{escape(disk_id)}[/bold]  [dim]· {escape(location)} · "
                    f"{_human_bytes(selected.size_bytes)} · {escape(rate)} · x delete[/dim]"
                )
                return
            rows_markup = [
                ("Disk", f"[bold]{escape(disk_id)}[/bold]  [dim]· {escape(location)}[/dim]"),
                ("Billing", f"{escape(selected.quant or '-')}  [dim]· {escape(rate)} · bills until deleted[/dim]"),
            ]
            footer = "[dim][bold $error]x[/] delete disk  ·  [bold $primary]/[/] filter  ·  [bold $primary]r[/] refresh[/dim]"
            detail.update(f"{labelled_rows_markup(rows_markup)}\n{footer}")
            return

        runtime_label = "vLLM" if selected.backend == BackendType.VLLM else "llama.cpp"
        short_name = selected.model_id.rsplit("/", 1)[-1]
        detail.border_title = f"{clip(short_name, 42)} · {runtime_label}"
        siblings = self._rows_removed_with(selected)
        total_modal_bytes = self._snapshot.total_size_bytes
        estimate = estimate_monthly_storage_cost(self._snapshot)
        free_tier = format_free_tier(MODAL_VOLUME_FREE_TIER_GIB_MONTH)
        effective_usd = allocated_monthly_storage_cost_usd(selected.size_bytes, total_modal_bytes)
        list_usd = gross_monthly_storage_cost_usd(selected.size_bytes)
        if compact_height:
            parts = [f"[bold]{escape(clip(selected.model_id, 44))}[/bold]"]
            if selected.quant:
                parts.append(escape(selected.quant))
            parts.append(_human_bytes(selected.size_bytes))
            parts.append(f"{format_money(effective_usd)}/mo")
            if selected.source_volume:
                parts.append(f"[dim]{escape(selected.source_volume)}[/dim]")
            if selected.incomplete:
                parts.append("[$warning]incomplete · p resume[/]")
            detail.update(" [dim]·[/dim] ".join(parts))
            return

        cache_bits = [_human_bytes(selected.size_bytes)]
        file_noun = "file" if selected.file_count == 1 else "files"
        cache_bits.append(f"{selected.file_count} {file_noun}")
        if selected.quant:
            cache_bits.append(f"quant {escape(selected.quant)}")
        if selected.revision:
            cache_bits.append(f"rev {escape(_short_revision(selected))}")
        if selected.source_volume:
            cache_bits.append(f"vol {escape(selected.source_volume)}")

        model_value = f"[bold]{escape(selected.model_id)}[/bold]"
        if selected.incomplete:
            model_value += "  [$warning]incomplete download — press p to resume[/]"

        if estimate.billable_gib_month <= 0:
            remaining_gib = max(0.0, MODAL_VOLUME_FREE_TIER_GIB_MONTH - estimate.total_gib_month)
            cost_value = (
                f"[$success]{format_money(0.0)}/mo[/] "
                f"[dim](within {free_tier} free tier · {format_gib(remaining_gib)} left · "
                f"list {format_money(list_usd)}/mo)[/dim]"
            )
        else:
            repo_bytes = selected.size_bytes + sum(item.size_bytes for item in siblings)
            savings_usd = marginal_monthly_storage_savings_usd(repo_bytes, total_modal_bytes)
            cost_value = (
                f"{format_money(effective_usd)}/mo share "
                f"[dim]· deleting saves {format_money(savings_usd)}/mo · "
                f"list {format_money(list_usd)}/mo[/dim]"
            )

        rows_markup = [
            ("Model", model_value),
            ("Cache", "  [dim]·[/dim]  ".join(cache_bits)),
            ("Cost", cost_value),
        ]
        lines = [labelled_rows_markup(rows_markup)]
        if siblings:
            total_bytes = selected.size_bytes + sum(item.size_bytes for item in siblings)
            quants = ", ".join(escape((item.quant or "unquantized").strip()) for item in siblings)
            lines.append(
                f"[$warning]⚠ Deleting removes the whole repo including {len(siblings)} "
                f"sibling quant{'s' if len(siblings) != 1 else ''} ({quants}) · "
                f"{_human_bytes(total_bytes)} total[/]"
            )
        lines.append(
            "[dim][bold $primary]p[/] pre-download  ·  [bold $primary]n[/] new download  ·  "
            "[bold $primary]/[/] filter  ·  [bold $error]x[/] delete  ·  [bold $primary]r[/] refresh[/dim]"
        )
        detail.update("\n".join(lines))

    def action_refresh_storage(self) -> None:
        self._refresh_storage_snapshot(force=True)

    def _refresh_storage_snapshot(self, force: bool = False) -> None:
        if self._selected_provider == ComputeProvider.VAST:
            self._set_storage_refreshing(False)
            self._render_table()
            self._render_storage_status()
            return
        if self._selected_provider == ComputeProvider.PRIME:
            refresher = getattr(self.app, "begin_prime_disks_refresh", None)
            if callable(refresher):
                self._set_storage_refreshing(True)
                self._render_storage_status()
                refresher(self)
                return
            self._set_storage_refreshing(False)
            self.query_one("#storage-status", Static).update(
                "[$warning]Prime disk refresh is unavailable in this session.[/$warning]"
            )
            return
        self._set_storage_refreshing(True)
        self._render_storage_status()
        self.app.begin_storage_refresh(self, force=force)

    def action_predownload_selected(self) -> None:
        if self._selected_provider != ComputeProvider.MODAL:
            self.app.notify(
                "Pre-download targets the Modal shared cache; switch provider to Modal.",
                severity="warning",
                timeout=5,
            )
            return
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
        self.app.begin_storage_predownload(
            backend=backend,
            model_id=model_id,
            quant=quant,
            revision=revision,
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "storage-predownload-btn":
            self.action_predownload_selected()
        elif event.button.id == "storage-predownload-cancel":
            self._predownload_expanded = False
            self._sync_predownload_visibility()
            self._update_selected_detail()
            self.query_one("#storage-table", AdaptiveDataTable).focus()

    def _sync_predownload_visibility(self) -> None:
        """Collapse the pre-download form until it is needed.

        The form is four fields plus a button; showing it always leaves a
        single inventory row on an 80x24 terminal. The toggle and the "p" key
        both expand it; collapsing never clears the values.
        """
        try:
            form = self.query_one("#storage-predownload-form")
        except NoMatches:
            return
        form.display = self._predownload_expanded

    def action_toggle_predownload_form(self) -> None:
        """Toggle the pre-download form; opening focuses the model ID field."""
        self._predownload_expanded = not self._predownload_expanded
        self._sync_predownload_visibility()
        self._update_selected_detail()
        if self._predownload_expanded:
            self.query_one("#storage-model-id", Input).focus()
        else:
            self.query_one("#storage-table", AdaptiveDataTable).focus()

    def action_delete_selected_model(self) -> None:
        if self._selected_provider == ComputeProvider.VAST:
            self.app.notify(
                "Vast rental disks are destroyed with the rental; there is no separate delete.",
                severity="warning",
                timeout=5,
            )
            return
        if self._selected_provider == ComputeProvider.PRIME:
            disk_id = (self._selected_prime_disk_id or "").strip()
            if not disk_id or self._selected_model is None:
                self.app.notify("Select a Prime disk row first to delete.", severity="warning", timeout=5)
                return
            self.app.push_screen(PrimeDiskDeleteConfirmScreen(disk_id))
            return
        if self._selected_model is None:
            self.app.notify("Select a model row first to delete.", severity="warning", timeout=5)
            return
        self.app.push_screen(
            StorageDeleteConfirmScreen(
                self._selected_model,
                also_removed=self._rows_removed_with(self._selected_model),
            )
        )

    def _rows_removed_with(self, model: StoredModelInfo) -> tuple[StoredModelInfo, ...]:
        """Return the other cached rows this delete would take with it.

        Deletion is scoped to the repository directory, while the table lists a
        row per quant, so confirming one row can remove several.
        """
        return tuple(
            row
            for row in (self._snapshot.llamacpp_models + self._snapshot.vllm_models)
            if row.backend == model.backend
            and row.model_id == model.model_id
            and _storage_row_key(row) != _storage_row_key(model)
        )

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def action_navigate_option_list_down(self) -> None:
        focused = self.focused
        if isinstance(focused, OptionList):
            # An open dropdown keeps its arrow keys.
            raise SkipAction()
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
        focused = self.focused
        if isinstance(focused, OptionList):
            # An open dropdown keeps its arrow keys.
            raise SkipAction()
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

    def __init__(
        self,
        model: StoredModelInfo,
        also_removed: tuple[StoredModelInfo, ...] = (),
    ) -> None:
        super().__init__()
        self.model = model
        self.also_removed = also_removed

    def _warning(self) -> str:
        """State the real blast radius, which is the whole repository cache.

        The row names one quant and its size, but deletion removes the
        repository directory, so confirming a 2.4 GiB row could silently take
        every other cached quant of that model with it.
        """
        if not self.also_removed:
            return "[$warning]This will remove the cached model files from provider storage.[/$warning]"
        quants = ", ".join(
            escape((row.quant or "unquantized").strip()) for row in self.also_removed
        )
        total = self.model.size_bytes + sum(row.size_bytes for row in self.also_removed)
        noun = "quant" if len(self.also_removed) == 1 else "quants"
        return (
            f"[$warning]This removes the whole cached repository, including "
            f"{len(self.also_removed)} other {noun} ({quants}). "
            f"{_human_bytes(total)} total.[/$warning]"
        )

    def compose(self) -> ComposeResult:
        detail = [escape(self.model.backend.value)]
        quant = (self.model.quant or "").strip()
        if quant:
            detail.append(escape(quant))
        detail.append(_human_bytes(self.model.size_bytes))
        with VerticalScroll(classes="screen-scroll dialog-scroll"):
            with Vertical(id="delete-confirm-dialog", classes="dialog-panel"):
                yield Static("[bold $primary]Delete Cached Model[/]", classes="dialog-title")
                yield Static(
                    f"Delete [bold]{escape(self.model.model_id)}[/bold]?\n"
                    f"[dim]{' · '.join(detail)}[/dim]"
                )
                yield Static(self._warning(), id="delete-warning")
                with Horizontal(id="delete-confirm-actions", classes="dialog-actions"):
                    yield Button("Cancel", id="delete-cancel")
                    yield Button("Delete model", id="delete-confirm", variant="error")
        yield FittedFooter()

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
        self.app.begin_storage_delete(self.model)


class PrimeDiskDeleteConfirmScreen(CopyEnabledScreen):
    """Confirm terminating a billable Prime persistent disk."""

    BINDINGS = [
        Binding("left,up", "previous_action", show=False, priority=True),
        Binding("right,down", "next_action", show=False, priority=True),
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("x", "confirm_delete", "Confirm delete", show=True),
    ]
    ACTION_IDS = ("delete-cancel", "delete-confirm")

    def __init__(self, disk_id: str) -> None:
        super().__init__()
        self.disk_id = disk_id

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="screen-scroll dialog-scroll"):
            with Vertical(id="prime-delete-confirm-dialog", classes="dialog-panel"):
                yield Static("[bold $primary]Delete Prime Disk[/]", classes="dialog-title")
                yield Static(
                    f"Permanently delete Prime disk [bold]{escape(self.disk_id)}[/bold] "
                    "and its cached weights?"
                )
                yield Static(
                    "[$warning]The disk keeps billing until deleted. Deleting it means "
                    "the next deploy re-downloads the weights.[/$warning]"
                )
                with Horizontal(id="delete-confirm-actions", classes="dialog-actions"):
                    yield Button("Cancel", id="delete-cancel")
                    yield Button("Delete disk", id="delete-confirm", variant="error")
        yield FittedFooter()

    def on_mount(self) -> None:
        self.query_one("#delete-cancel", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "delete-cancel":
            self.app.pop_screen()
        elif event.button.id == "delete-confirm":
            self.action_confirm_delete()

    def action_cancel(self) -> None:
        self.app.pop_screen()

    def action_previous_action(self) -> None:
        move_focus_across_widgets(self, self.ACTION_IDS, -1)

    def action_next_action(self) -> None:
        move_focus_across_widgets(self, self.ACTION_IDS, 1)

    def action_confirm_delete(self) -> None:
        disk_id = self.disk_id
        self.app.pop_screen()
        deleter = getattr(self.app, "begin_prime_disk_delete", None)
        if callable(deleter):
            try:
                screens = list(getattr(self.app, "screen_stack", []))
                receiver = next(
                    (screen for screen in reversed(screens) if type(screen).__name__ == "StorageScreen"),
                    None,
                )
                if receiver is not None:
                    deleter(disk_id, receiver)
                    return
            except Exception:
                pass
        self.app.notify("Prime disk deletion is unavailable in this session.", severity="error")
