"""Settings screen: scaledown window persistence."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Vertical
from textual.widgets import Button, Footer, Input, Static

from ...core.config import ConfigStore
from ...protocol.models import LaunchpadSettings
from ..widgets.input_form import FormField
from .copy_enabled import CopyEnabledScreen


class SettingsScreen(CopyEnabledScreen):
    """Edit and persist SCALEDOWN_WINDOW."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("ctrl+s", "save", "Save", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._store = ConfigStore()
        self._load_error: str | None = None

    def compose(self) -> ComposeResult:
        loaded = self._store.load_result()
        settings = loaded.settings
        self._load_error = loaded.error

        with Center():
            with Vertical(id="settings-form"):
                yield Static("[bold #7bf168]Settings[/]")
                yield Static("")
                yield FormField(
                    "Scaledown window (seconds)",
                    "scaledown-window",
                    default=str(settings.scaledown_window),
                    hint="Seconds before idle containers scale down",
                )
                yield Static("")
                yield Button("Save", id="save-btn", variant="primary")
                yield Static(
                    f"[yellow]{self._load_error} Using defaults.[/yellow]"
                    if self._load_error
                    else "",
                    id="save-feedback",
                )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            self._save()

    def action_save(self) -> None:
        self._save()


    def _save(self) -> None:
        scaledown_str = self.query_one("#scaledown-window", Input).value.strip()
        try:
            scaledown = int(scaledown_str)
        except ValueError:
            self.query_one("#save-feedback", Static).update(
                "[red]Scaledown must be an integer.[/red]"
            )
            return

        settings = LaunchpadSettings(
            scaledown_window=scaledown,
        )
        result = self._store.save_result(settings)
        if result.success:
            self.query_one("#save-feedback", Static).update(
                "[green]Settings saved.[/green]"
            )
        else:
            self.query_one("#save-feedback", Static).update(
                f"[red]{result.error or 'Settings could not be saved.'}[/red]"
            )

    def action_pop_screen(self) -> None:
        self.app.pop_screen()
