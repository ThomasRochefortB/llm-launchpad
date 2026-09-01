"""Settings screen: scaledown window persistence."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Button, Footer, Input, Select, Static

from ...core.config import ConfigStore
from ...protocol.models import LaunchpadSettings
from ..visual import (
    TUI_DENSITY_OPTIONS,
    TUI_THEME_OPTIONS,
    normalize_tui_density,
    normalize_tui_theme,
)
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

        with VerticalScroll(id="settings-scroll", classes="screen-scroll"):
            with Vertical(id="settings-form"):
                yield Static("[bold #7bf168]Settings[/]")
                yield Static("")
                yield FormField(
                    "Scaledown window (seconds)",
                    "scaledown-window",
                    default=str(settings.scaledown_window),
                    hint="Seconds before idle containers scale down",
                )
                yield Static("Theme", classes="form-label")
                yield Select(
                    TUI_THEME_OPTIONS,
                    value=normalize_tui_theme(settings.tui_theme),
                    allow_blank=False,
                    id="tui-theme",
                )
                yield Static(
                    "[dim]Dark, high-contrast, and low-color terminal palettes.[/dim]",
                    classes="form-hint",
                )
                yield Static("Density", classes="form-label")
                yield Select(
                    TUI_DENSITY_OPTIONS,
                    value=normalize_tui_density(settings.tui_density),
                    allow_blank=False,
                    id="tui-density",
                )
                yield Static(
                    "[dim]Compact density reduces spacing without hiding content.[/dim]",
                    classes="form-hint",
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
            tui_theme=normalize_tui_theme(
                self.query_one("#tui-theme", Select).value
            ),
            tui_density=normalize_tui_density(
                self.query_one("#tui-density", Select).value
            ),
        )
        result = self._store.save_result(settings)
        if result.success:
            self.query_one("#save-feedback", Static).update(
                "[green]Settings saved.[/green]"
            )
            apply_preferences = getattr(self.app, "apply_visual_preferences", None)
            if callable(apply_preferences):
                apply_preferences(settings.tui_theme, settings.tui_density)
        else:
            self.query_one("#save-feedback", Static).update(
                f"[red]{result.error or 'Settings could not be saved.'}[/red]"
            )

    def action_pop_screen(self) -> None:
        self.app.pop_screen()
