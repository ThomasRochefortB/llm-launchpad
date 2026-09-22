"""Settings screen: appearance, behavior defaults, and scaledown window."""

from __future__ import annotations

import re
import time

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Select, Static, Switch

from ...core.config import ConfigStore
from ...protocol.models import LaunchpadSettings
from ..visual import (
    TUI_DENSITY_OPTIONS,
    TUI_THEME_OPTIONS,
    normalize_tui_density,
    normalize_tui_theme,
)
from ..widgets.input_form import FormField, ToggleField
from ..widgets.fitted_footer import FittedFooter
from ..navigation import move_focus_with_arrows
from .copy_enabled import CopyEnabledScreen


_DURATION_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*$", re.IGNORECASE)


def parse_scaledown_window(value: str) -> int:
    """Parse a human duration into whole seconds for idle scale-down.

    Accepts plain seconds (``1800``), or suffixed ``90s`` / ``30m`` / ``2h``.
    Fractional values round to the nearest second.
    """
    match = _DURATION_PATTERN.match(value or "")
    if match is None:
        raise ValueError(f"Cannot parse duration: {value!r}")
    amount = float(match.group(1))
    unit = (match.group(2) or "s").lower()
    multiplier = {"s": 1, "m": 60, "h": 3600}[unit]
    return max(0, int(round(amount * multiplier)))


def format_scaledown_window(seconds: int) -> str:
    """Render seconds in the largest unit ``parse_scaledown_window`` reads back."""
    if seconds and seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds and seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


class SettingsScreen(CopyEnabledScreen):
    """Edit and persist scaledown, appearance, and TUI behavior settings."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("ctrl+s", "save", "Save", show=True),
        # Arrows supplement Tab navigation, including on phone keyboards.
        Binding("up", "focus_previous_control", "Previous", show=False, priority=True),
        Binding("down", "focus_next_control", "Next", show=False, priority=True),
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

        with Vertical(id="settings-layout"):
            with VerticalScroll(id="settings-scroll", classes="screen-scroll"):
                with Vertical(id="settings-form"):
                    yield Static("[bold primary]Settings[/]", id="settings-title")
                    yield Static("[bold]Deployment[/bold]", classes="settings-section")
                    yield FormField(
                        "Idle timeout before scale-down",
                        "scaledown-window",
                        default=format_scaledown_window(settings.scaledown_window),
                        hint="e.g. 30m, 90s, or 1800 · idle containers scale to zero after this",
                    )
                    yield Static("[bold]Appearance[/bold]", classes="settings-section")
                    with Horizontal(id="settings-appearance-row"):
                        with Vertical(classes="settings-appearance-control"):
                            yield Static("Theme", classes="form-label")
                            yield Select(
                                TUI_THEME_OPTIONS,
                                value=normalize_tui_theme(settings.tui_theme),
                                allow_blank=False,
                                id="tui-theme",
                            )
                            yield Static("[dim]Color palette[/dim]", classes="form-hint")
                        with Vertical(classes="settings-appearance-control"):
                            yield Static("Density", classes="form-label")
                            yield Select(
                                TUI_DENSITY_OPTIONS,
                                value=normalize_tui_density(settings.tui_density),
                                allow_blank=False,
                                id="tui-density",
                            )
                            yield Static("[dim]Spacing[/dim]", classes="form-hint")
                    yield Static("[bold]Behavior[/bold]", classes="settings-section")
                    yield Static(
                        "[dim]Navigate with Tab, Shift+Tab, arrow keys, and Enter. "
                        "Use your terminal's selection and copy/paste shortcuts.[/dim]",
                        classes="form-hint",
                    )
                    yield ToggleField(
                        "Require a second Ctrl+C to quit",
                        "confirm-quit",
                        default=settings.confirm_quit,
                    )
                    yield Static("[bold]Providers[/bold]", classes="settings-section")
                    yield Static(
                        "[dim]Vast.ai marketplace rentals and saved keys.[/dim]",
                        classes="form-hint",
                    )
                    yield Button("Vast.ai rentals", id="vast-preview-btn")
            with Horizontal(id="settings-actions"):
                yield Button("Save", id="save-btn", variant="primary")
                yield Static(
                    f"[yellow]{escape(self._load_error)} Using defaults.[/yellow]"
                    if self._load_error
                    else "",
                    id="save-feedback",
                )
        yield FittedFooter()

    def on_mount(self) -> None:
        # Land on the first field. Textual's default auto-focus took the
        # scrolling container instead, so the screen opened with nothing
        # visibly focused and the first keystroke went nowhere.
        self.query_one("#scaledown-window", Input).focus()
        # Only edits that arrive after the initial values have settled count.
        self.call_after_refresh(self._start_accepting_edits)

    def _start_accepting_edits(self) -> None:
        self._accepting_edits = True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            self._save()
        elif event.button.id == "vast-preview-btn":
            from .vast import VastPreviewScreen

            self.app.push_screen(VastPreviewScreen())

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "scaledown-window":
            try:
                event.input.remove_class("-invalid")
            except Exception:
                pass
        self._mark_dirty()

    def on_select_changed(self, event: Select.Changed) -> None:
        self._mark_dirty()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        self._mark_dirty()

    def _mark_dirty(self) -> None:
        if not self._accepting_edits:
            return
        self._dirty = True
        self._announce("[yellow]Unsaved changes — ctrl+s to save.[/yellow]")

    def _announce(self, markup: str) -> None:
        """Update the feedback line in the persistent action bar.

        Save lives outside the scrolling form now, so feedback is always
        visible: no scrolling to the bottom, and the viewport never jumps away
        from the field being edited.
        """
        try:
            feedback = self.query_one("#save-feedback", Static)
        except Exception:
            return
        feedback.update(markup)

    def action_save(self) -> None:
        self._save()

    def _reject_scaledown(self) -> None:
        """Flag the timeout field in place so the error cannot scroll away."""
        try:
            field = self.query_one("#scaledown-window", Input)
        except Exception:
            self._announce("[red]Idle timeout must be seconds or a duration like 30m.[/red]")
            return
        field.add_class("-invalid")
        field.focus()
        self._announce("[red]Idle timeout must be seconds or a duration like 30m.[/red]")

    def _save(self) -> None:
        scaledown_str = self.query_one("#scaledown-window", Input).value.strip()
        try:
            scaledown = parse_scaledown_window(scaledown_str)
        except ValueError:
            self._reject_scaledown()
            return
        try:
            self.query_one("#scaledown-window", Input).remove_class("-invalid")
        except Exception:
            pass

        settings = LaunchpadSettings(
            scaledown_window=scaledown,
            tui_theme=normalize_tui_theme(
                self.query_one("#tui-theme", Select).value
            ),
            tui_density=normalize_tui_density(
                self.query_one("#tui-density", Select).value
            ),
            confirm_quit=self.query_one("#confirm-quit", Switch).value,
        )
        result = self._store.save_result(settings)
        if result.success:
            self._dirty = False
            self._unsaved_warning_at = 0.0
            self._announce("[green]Settings saved.[/green]")
            apply_preferences = getattr(self.app, "apply_visual_preferences", None)
            if callable(apply_preferences):
                apply_preferences(settings.tui_theme, settings.tui_density)
            self._apply_behavior_settings(settings)
        else:
            self._announce(
                f"[red]{escape(result.error or 'Settings could not be saved.')}[/red]"
            )

    def _apply_behavior_settings(self, settings: LaunchpadSettings) -> None:
        """Apply quit-confirmation preferences to the running app."""
        self.app._confirm_quit = settings.confirm_quit

    def action_focus_next_control(self) -> None:
        move_focus_with_arrows(self, 1)

    def action_focus_previous_control(self) -> None:
        move_focus_with_arrows(self, -1)

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
        self._announce(
            "[yellow]Unsaved changes. Press esc again to discard, or ctrl+s to save.[/yellow]"
        )
        return False
