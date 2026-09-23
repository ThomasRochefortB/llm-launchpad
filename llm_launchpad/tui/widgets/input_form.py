"""Reusable form input helpers for deploy/settings screens."""

from __future__ import annotations

from typing import Any, Literal

from rich.markup import escape

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, Static, Switch
from textual.validation import Length


class FormField(Vertical):
    """A labelled input field with optional hint text."""

    DEFAULT_CSS = """
    FormField {
        height: auto;
        padding: 0 0 1 0;
    }
    FormField .form-label {
        color: $text;
        height: auto;
        text-wrap: wrap;
    }
    FormField .form-hint {
        color: $text-muted;
        height: auto;
        text-wrap: wrap;
    }
    FormField .form-error {
        color: $error;
        height: auto;
        text-wrap: wrap;
    }
    FormField .form-error.hidden {
        display: none;
    }
    """

    def __init__(
        self,
        label: str,
        field_id: str,
        default: str = "",
        hint: str = "",
        password: bool = False,
        required: bool = False,
        input_type: Literal["integer", "number", "text"] = "text",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._field_id = field_id
        self._default = default
        self._hint = hint
        self._password = password
        self._required = required
        self._input_type = input_type

    def compose(self) -> ComposeResult:
        yield Static(self._label, classes="form-label")
        validators = [Length(minimum=1)] if self._required else []
        placeholder = "" if self._hint else self._label
        yield Input(
            value=self._default,
            placeholder=placeholder,
            password=self._password,
            id=self._field_id,
            type=self._input_type,
            validators=validators,
        )
        if self._hint:
            yield Static(self._hint, classes="form-hint")
        # Validation that refuses a deploy says why here, under the field, and
        # keeps saying it until the value changes: a toast was gone before the
        # reader had scrolled back to find which field it meant.
        yield Static("", classes="form-error hidden")

    @property
    def value(self) -> str:
        return self.query_one(f"#{self._field_id}", Input).value

    def show_error(self, message: str) -> None:
        """Mark the field invalid and say why beneath it."""
        self.query_one(f"#{self._field_id}", Input).add_class("-rejected")
        error = self.query_one(".form-error", Static)
        error.update(f"✗ {escape(message)}")
        error.remove_class("hidden")

    def clear_error(self) -> None:
        """Drop an error shown by :meth:`show_error`."""
        self.query_one(f"#{self._field_id}", Input).remove_class("-rejected")
        error = self.query_one(".form-error", Static)
        error.update("")
        error.add_class("hidden")


class ToggleField(Horizontal):
    """A labelled boolean toggle: label, switch, and the state in words.

    The switch alone distinguishes on from off by colour, which the monochrome
    theme and the low-colour terminals it exists for both flatten. The state
    word carries the same information without relying on hue, and laying the
    row out horizontally keeps a settings form two rows shorter per toggle.
    """

    DEFAULT_CSS = """
    ToggleField {
        height: auto;
        padding: 0 0 1 0;
    }
    ToggleField .toggle-label {
        color: $text;
        width: 1fr;
        height: 3;
        content-align: left middle;
        text-wrap: wrap;
    }
    ToggleField .toggle-state {
        width: 5;
        height: 3;
        padding-right: 1;
        content-align: right middle;
        color: $text-muted;
    }
    ToggleField.-on .toggle-state {
        color: $text;
        text-style: bold;
    }
    """

    def __init__(
        self,
        label: str,
        field_id: str,
        default: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._field_id = field_id
        self._default = default

    def compose(self) -> ComposeResult:
        yield Static(self._label, classes="toggle-label")
        yield Static(_toggle_state_label(self._default), classes="toggle-state")
        yield Switch(value=self._default, id=self._field_id)

    def on_mount(self) -> None:
        self.set_class(self._default, "-on")

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Keep the state word in step. The event still reaches the screen."""
        if event.switch.id != self._field_id:
            return
        self.set_class(event.value, "-on")
        self.query_one(".toggle-state", Static).update(
            _toggle_state_label(event.value)
        )

    @property
    def value(self) -> bool:
        return self.query_one(f"#{self._field_id}", Switch).value


def _toggle_state_label(value: bool) -> str:
    return "On" if value else "Off"
