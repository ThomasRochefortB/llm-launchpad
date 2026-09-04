"""Settings screen: appearance, behavior defaults, and scaledown window."""

from __future__ import annotations

import time

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Button, Footer, Input, Select, Static, Switch

from ...core.config import ConfigStore
from ...protocol.models import LaunchpadSettings
from ..visual import (
    TUI_DENSITY_OPTIONS,
    TUI_THEME_OPTIONS,
    normalize_tui_density,
    normalize_tui_theme,
)
from ..widgets.input_form import FormField, ToggleField
from .copy_enabled import CopyEnabledScreen


class SettingsScreen(CopyEnabledScreen):
    """Edit and persist scaledown, appearance, and TUI behavior settings."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("ctrl+s", "save", "Save", show=True),
    ]
    _UNSAVED_DISCARD_WINDOW_SECONDS = 6.0

    def __init__(self) -> None:
        super().__init__()
        self._store = ConfigStore()
        self._load_error: str | None = None
        self._dirty = False
        self._unsaved_warning_at = 0.0
        # Textual emits Changed events while widgets take their initial values
        # during mount. Treating those as edits told anyone who merely opened
        # this screen that they had unsaved changes, and then demanded a second
        # esc to "discard" them.
        self._accepting_edits = False

    def compose(self) -> ComposeResult:
        loaded = self._store.load_result()
        settings = loaded.settings
        self._load_error = loaded.error

        with VerticalScroll(id="settings-scroll", classes="screen-scroll"):
            with Vertical(id="settings-form"):
                yield Static("[bold #7bf168]Settings[/]")
                yield Static("")
                yield Static("[bold]Deployment[/bold]")
                yield FormField(
                    "Scaledown window (seconds)",
                    "scaledown-window",
                    default=str(settings.scaledown_window),
                    hint="Seconds before idle containers scale down",
                )
                yield Static("[bold]Appearance[/bold]")
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
                yield Static("[bold]Behavior[/bold]")
                yield ToggleField(
                    "Enable mouse support",
                    "tui-mouse",
                    default=True if settings.tui_mouse is None else settings.tui_mouse,
                )
                yield Static(
                    "[dim]Off enables native terminal text selection and copy shortcuts; "
                    "ctrl+t toggles this at runtime.[/dim]",
                    classes="form-hint",
                )
                yield ToggleField(
                    "Require a second Ctrl+C to quit",
                    "confirm-quit",
                    default=settings.confirm_quit,
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

    def on_mount(self) -> None:
        # Only edits that arrive after the initial values have settled count.
        self.call_after_refresh(self._start_accepting_edits)

    def _start_accepting_edits(self) -> None:
        self._accepting_edits = True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            self._save()

    def on_input_changed(self, event: Input.Changed) -> None:
        self._mark_dirty()

    def on_select_changed(self, event: Select.Changed) -> None:
        self._mark_dirty()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        self._mark_dirty()

    def _mark_dirty(self) -> None:
        if not self._accepting_edits:
            return
        self._dirty = True
        try:
            self.query_one("#save-feedback", Static).update(
                "[yellow]Unsaved changes — ctrl+s to save.[/yellow]"
            )
        except Exception:
            return

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
            tui_mouse=self.query_one("#tui-mouse", Switch).value,
            confirm_quit=self.query_one("#confirm-quit", Switch).value,
        )
        result = self._store.save_result(settings)
        if result.success:
            self._dirty = False
            self._unsaved_warning_at = 0.0
            self.query_one("#save-feedback", Static).update(
                "[green]Settings saved.[/green]"
            )
            apply_preferences = getattr(self.app, "apply_visual_preferences", None)
            if callable(apply_preferences):
                apply_preferences(settings.tui_theme, settings.tui_density)
            self._apply_behavior_settings(settings)
        else:
            self.query_one("#save-feedback", Static).update(
                f"[red]{result.error or 'Settings could not be saved.'}[/red]"
            )

    def _apply_behavior_settings(self, settings: LaunchpadSettings) -> None:
        """Apply mouse-mode and quit-confirmation preferences to the running app."""
        set_mouse_mode = getattr(self.app, "_set_mouse_mode", None)
        if (
            callable(set_mouse_mode)
            and settings.tui_mouse is not None
            and bool(getattr(self.app, "mouse_enabled", True)) != settings.tui_mouse
        ):
            set_mouse_mode(settings.tui_mouse)
        self.app._confirm_quit = settings.confirm_quit

    def action_pop_screen(self) -> None:
        if self._dirty and not self._discard_confirmed():
            return
        self.app.pop_screen()

    def _discard_confirmed(self) -> bool:
        """Require a second esc to discard unsaved edits."""
        now = time.monotonic()
        if now - self._unsaved_warning_at <= self._UNSAVED_DISCARD_WINDOW_SECONDS:
            return True
        self._unsaved_warning_at = now
        self.query_one("#save-feedback", Static).update(
            "[yellow]Unsaved changes. Press esc again to discard, or ctrl+s to save.[/yellow]"
        )
        return False
