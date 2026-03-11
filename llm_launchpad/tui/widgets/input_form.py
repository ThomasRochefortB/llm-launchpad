"""Reusable form input helpers for deploy/settings screens."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
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
        color: #c8d6c9;
        height: 1;
    }
    FormField .form-hint {
        color: #7f9082;
        height: 1;
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
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._field_id = field_id
        self._default = default
        self._hint = hint
        self._password = password
        self._required = required

    def compose(self) -> ComposeResult:
        yield Static(self._label, classes="form-label")
        validators = [Length(minimum=1)] if self._required else []
        placeholder = "" if self._hint else self._label
        yield Input(
            value=self._default,
            placeholder=placeholder,
            password=self._password,
            id=self._field_id,
            validators=validators,
        )
        if self._hint:
            yield Static(self._hint, classes="form-hint")

    @property
    def value(self) -> str:
        return self.query_one(f"#{self._field_id}", Input).value


class ToggleField(Vertical):
    """A labelled boolean toggle (switch)."""

    DEFAULT_CSS = """
    ToggleField {
        height: auto;
        padding: 0 0 1 0;
    }
    ToggleField .form-label {
        color: #c8d6c9;
        height: 1;
    }
    """

    def __init__(
        self,
        label: str,
        field_id: str,
        default: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._field_id = field_id
        self._default = default

    def compose(self) -> ComposeResult:
        yield Static(self._label, classes="form-label")
        yield Switch(value=self._default, id=self._field_id)

    @property
    def value(self) -> bool:
        return self.query_one(f"#{self._field_id}", Switch).value
