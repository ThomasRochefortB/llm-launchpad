"""Keybinding help overlay for the active screen."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Footer, Static


def _iter_screen_bindings(obj: object) -> list[tuple[str, str]]:
    """Collect (key, label) pairs from a widget's class BINDINGS across the MRO."""
    bindings: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for base in reversed(type(obj).__mro__):
        class_bindings = base.__dict__.get("BINDINGS")
        if not class_bindings:
            continue
        for entry in class_bindings:
            if not isinstance(entry, Binding) or entry.system:
                continue
            key_token = entry.key.split(",", 1)[0].strip()
            if not key_token:
                continue
            label = (entry.description or entry.action or "").strip()
            identity = (key_token, label)
            if identity in seen:
                continue
            seen.add(identity)
            bindings.append(identity)
    return bindings


class HelpOverlayScreen(ModalScreen):
    """List this screen's keybindings without leaving the flow."""

    BINDINGS = [
        Binding("escape,q,?", "dismiss_help", "Close", show=True, priority=True),
    ]

    DEFAULT_CSS = """
    HelpOverlayScreen {
        align: center middle;
    }
    #help-overlay-card {
        width: 64;
        max-width: 92%;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: round #7bf168;
        padding: 1 2;
    }
    #help-overlay-body {
        height: auto;
    }
    """

    def __init__(self, sections: list[tuple[str, list[tuple[str, str]]]]) -> None:
        super().__init__()
        self._sections = sections

    @classmethod
    def from_screen(cls, screen: object) -> HelpOverlayScreen:
        sections: list[tuple[str, list[tuple[str, str]]]] = []
        app = getattr(screen, "app", None)
        screen_bindings = _iter_screen_bindings(screen)
        if screen_bindings:
            sections.append(("This screen", screen_bindings))
        if app is not None:
            app_bindings = _iter_screen_bindings(app)
            if app_bindings:
                sections.append(("Global", app_bindings))
        return cls(sections)

    def compose(self) -> ComposeResult:
        with Vertical(id="help-overlay-card"):
            yield Static("[bold #7bf168]Keyboard shortcuts[/]", id="help-overlay-title")
            with VerticalScroll(id="help-overlay-body"):
                for section_name, bindings in self._sections:
                    yield Static(f"[bold]{section_name}[/bold]")
                    for key, label in bindings:
                        yield Static(f"  [bold]{key:<12}[/] {label}")
                    yield Static("")
                yield Static(
                    "[dim]Mouse mode, copy, and paste work across every screen. "
                    "Press ctrl+t to toggle mouse support.[/dim]"
                )
        yield Footer()

    def action_dismiss_help(self) -> None:
        self.app.pop_screen()
