"""Settings screen: GPU config and scaledown window persistence."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button, Footer, Input, Select, Static

from ...core.config import ConfigStore
from ...core.modal_gpu import fetch_modal_gpu_types
from ...protocol.models import LaunchpadSettings
from ..widgets.input_form import FormField


def _parse_gpu_config(value: str) -> tuple[str, int]:
    """Split GPU_CONFIG into (gpu_type, gpu_count)."""
    raw = value.strip()
    if not raw:
        return "", 1
    if ":" not in raw:
        return raw.upper(), 1

    gpu_type_raw, count_raw = raw.rsplit(":", 1)
    gpu_type = gpu_type_raw.strip().upper()
    try:
        count = int(count_raw.strip())
    except ValueError:
        count = 1
    if count <= 0:
        count = 1
    return gpu_type, count


def _build_gpu_type_options(gpu_types: list[str]) -> list[str]:
    """Build GPU type dropdown options from raw values."""
    options: list[str] = []
    seen: set[str] = set()
    for value in gpu_types:
        token = value.strip().upper()
        if not token or token in seen:
            continue
        seen.add(token)
        options.append(token)
    return options


class GpuTypesLoaded(Message):
    """GPU type options were fetched successfully."""

    def __init__(self, gpu_types: list[str]) -> None:
        super().__init__()
        self.gpu_types = gpu_types


class GpuTypesFailed(Message):
    """GPU type option fetch failed."""

    def __init__(self, error: str) -> None:
        super().__init__()
        self.error = error


class SettingsScreen(Screen):
    """Edit and persist GPU_CONFIG and SCALEDOWN_WINDOW."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("ctrl+s", "save", "Save", show=True),
        Binding("ctrl+r", "refresh_gpu_types", "Refresh GPU types", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._store = ConfigStore()
        self._selected_gpu_type = ""

    def compose(self) -> ComposeResult:
        settings = self._store.load()
        gpu_type, gpu_count = _parse_gpu_config(settings.gpu_config)
        self._selected_gpu_type = gpu_type
        initial_options = [(gpu_type, gpu_type)] if gpu_type else []

        with Center():
            with Vertical(id="settings-form"):
                yield Static("[bold cyan]Settings[/bold cyan]")
                yield Static("")
                yield Static("GPU configuration", classes="form-label")
                with Horizontal(id="gpu-config-row"):
                    with Vertical(id="gpu-type-group"):
                        yield Static("GPU type", classes="form-label")
                        yield Select(
                            options=initial_options,
                            prompt="Select GPU type",
                            value=gpu_type if gpu_type else Select.BLANK,
                            id="gpu-type-dropdown",
                        )
                    with Vertical(id="gpu-count-group"):
                        yield Static("GPU count", classes="form-label")
                        yield Input(
                            value=str(gpu_count),
                            placeholder="1",
                            id="gpu-count",
                            type="integer",
                        )
                yield Static(
                    "[dim]Loading GPU types from Modal docs...[/dim]",
                    id="gpu-types-status",
                )
                yield FormField(
                    "Scaledown window (seconds)",
                    "scaledown-window",
                    default=str(settings.scaledown_window),
                    hint="Seconds before idle containers scale down",
                )
                yield Static("")
                yield Button("Save", id="save-btn", variant="primary")
                yield Static("", id="save-feedback")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_gpu_types()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            self._save()

    def action_save(self) -> None:
        self._save()

    def action_refresh_gpu_types(self) -> None:
        self._refresh_gpu_types()

    def _refresh_gpu_types(self) -> None:
        self.query_one("#gpu-types-status", Static).update(
            "[dim]Loading GPU types from Modal docs...[/dim]"
        )
        self.run_worker(
            lambda: self._run_fetch_gpu_types(),
            name="settings-fetch-gpu-types",
            thread=True,
        )

    def _run_fetch_gpu_types(self) -> None:
        poster = getattr(self, "post_message", None)
        if poster is None:
            return
        try:
            gpu_types = fetch_modal_gpu_types()
        except Exception as exc:
            poster(GpuTypesFailed(error=str(exc)))
            return
        poster(GpuTypesLoaded(gpu_types=gpu_types))

    def on_gpu_types_loaded(self, message: GpuTypesLoaded) -> None:
        dropdown = self.query_one("#gpu-type-dropdown", Select)
        options = _build_gpu_type_options(message.gpu_types)
        if self._selected_gpu_type and self._selected_gpu_type not in options:
            options.insert(0, self._selected_gpu_type)

        if options:
            dropdown.set_options([(value, value) for value in options])
            selected = self._selected_gpu_type if self._selected_gpu_type in options else options[0]
            dropdown.value = selected
            self._selected_gpu_type = selected
            self.query_one("#gpu-types-status", Static).update(
                f"[dim]GPU type list ready ({len(message.gpu_types)} types).[/dim]"
            )
            return

        dropdown.set_options([])
        dropdown.value = Select.BLANK
        self._selected_gpu_type = ""
        self.query_one("#gpu-types-status", Static).update(
            "[yellow]Modal docs fetch returned no GPU types.[/yellow]"
        )

    def on_gpu_types_failed(self, message: GpuTypesFailed) -> None:
        self.query_one("#gpu-types-status", Static).update(
            f"[yellow]Could not load GPU types:[/yellow] {message.error}"
        )

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "gpu-type-dropdown":
            return
        if isinstance(event.value, str) and event.value.strip():
            self._selected_gpu_type = event.value.strip().upper()

    def _save(self) -> None:
        gpu_type = self._selected_gpu_type.strip().upper()
        if not gpu_type:
            self.query_one("#save-feedback", Static).update("[red]GPU type is required.[/red]")
            return

        gpu_count_str = self.query_one("#gpu-count", Input).value.strip()
        try:
            gpu_count = int(gpu_count_str)
        except ValueError:
            self.query_one("#save-feedback", Static).update("[red]GPU count must be an integer.[/red]")
            return
        if gpu_count <= 0:
            self.query_one("#save-feedback", Static).update("[red]GPU count must be >= 1.[/red]")
            return

        scaledown_str = self.query_one("#scaledown-window", Input).value.strip()
        try:
            scaledown = int(scaledown_str)
        except ValueError:
            self.query_one("#save-feedback", Static).update(
                "[red]Scaledown must be an integer.[/red]"
            )
            return

        settings = LaunchpadSettings(
            gpu_config=f"{gpu_type}:{gpu_count}",
            scaledown_window=scaledown,
        )
        self._store.save(settings)
        self.query_one("#save-feedback", Static).update(
            "[green]Settings saved.[/green]"
        )

    def action_pop_screen(self) -> None:
        self.app.pop_screen()
