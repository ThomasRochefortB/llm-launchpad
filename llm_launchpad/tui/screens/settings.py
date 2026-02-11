"""Settings screen: GPU config and scaledown window persistence."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Input, Static

from ...core.config import ConfigStore
from ...protocol.models import LaunchpadSettings
from ..widgets.input_form import FormField


class SettingsScreen(Screen):
    """Edit and persist GPU_CONFIG and SCALEDOWN_WINDOW."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("ctrl+s", "save", "Save", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._store = ConfigStore()

    def compose(self) -> ComposeResult:
        settings = self._store.load()
        with Center():
            with Vertical(id="settings-form"):
                yield Static("[bold cyan]Settings[/bold cyan]")
                yield Static("")
                yield FormField(
                    "GPU_CONFIG",
                    "gpu-config",
                    default=settings.gpu_config,
                    hint="e.g., A100-80GB:1, H100:2",
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

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            self._save()

    def action_save(self) -> None:
        self._save()

    def _save(self) -> None:
        gpu = self.query_one("#gpu-config", Input).value.strip()
        scaledown_str = self.query_one("#scaledown-window", Input).value.strip()
        try:
            scaledown = int(scaledown_str)
        except ValueError:
            self.query_one("#save-feedback", Static).update(
                "[red]Scaledown must be an integer.[/red]"
            )
            return

        settings = LaunchpadSettings(gpu_config=gpu, scaledown_window=scaledown)
        self._store.save(settings)
        self.query_one("#save-feedback", Static).update(
            "[green]Settings saved.[/green]"
        )

    def action_pop_screen(self) -> None:
        self.app.pop_screen()
