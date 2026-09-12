"""Keybinding help overlay for the active screen."""

from __future__ import annotations

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from .fitted_footer import FittedFooter


def _class_bindings(obj: object) -> list[Binding]:
    """Every ``Binding`` declared across a widget's MRO, base classes first."""
    found: list[Binding] = []
    for base in reversed(type(obj).__mro__):
        for entry in base.__dict__.get("BINDINGS") or ():
            if isinstance(entry, Binding):
                found.append(entry)
    return found


def _iter_screen_bindings(
    obj: object,
    *,
    shadowed_keys: frozenset[str] = frozenset(),
) -> list[tuple[str, str]]:
    """Collect (keys, label) rows from a widget's class BINDINGS across the MRO.

    A binding with no description is internal plumbing -- arrow keys that hop
    between option lists, the focus shuffle inside a confirm dialog -- and is
    skipped. Falling back to the action name printed ``navigate_option_list_up``
    and ``close_details`` at the reader, which names nothing they can use.

    Keys that do the same named thing are listed together rather than once each,
    because three separate rows reading "Copy" describe one capability, not
    three. ``shadowed_keys`` drops rows the app overrides with a priority
    binding: Textual's own ``ctrl+c`` claims to copy the selection, but this app
    binds the same key at priority and quits with it.
    """
    by_label: dict[str, list[str]] = {}
    order: list[str] = []
    for entry in _class_bindings(obj):
        label = (entry.description or "").strip()
        if not label:
            continue
        key_token = entry.key.split(",", 1)[0].strip()
        if not key_token or key_token in shadowed_keys:
            continue
        if label not in by_label:
            by_label[label] = []
            order.append(label)
        if key_token not in by_label[label]:
            by_label[label].append(key_token)
    return [(", ".join(by_label[label]), label) for label in order]


def _priority_keys(app: object) -> frozenset[str]:
    """Keys the app claims at priority, which outrank any screen binding."""
    return frozenset(
        entry.key.split(",", 1)[0].strip()
        for entry in _class_bindings(app)
        if entry.priority
    )


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
        shadowed = _priority_keys(app) if app is not None else frozenset()
        screen_bindings = _iter_screen_bindings(screen, shadowed_keys=shadowed)
        if screen_bindings:
            sections.append(("This screen", screen_bindings))
        if app is not None:
            app_bindings = _iter_screen_bindings(app)
            if app_bindings:
                sections.append(("Global", app_bindings))
        return cls(sections)

    def compose(self) -> ComposeResult:
        # One column width for every section, measured rather than guessed:
        # a fixed 12 left `ctrl+shift+c` touching its own description.
        keys = [key for _section, bindings in self._sections for key, _label in bindings]
        key_width = max((len(key) for key in keys), default=0) + 2

        with Vertical(id="help-overlay-card"):
            yield Static("[bold #7bf168]Keyboard shortcuts[/]", id="help-overlay-title")
            with VerticalScroll(id="help-overlay-body"):
                for section_name, bindings in self._sections:
                    yield Static(f"[bold]{section_name}[/bold]")
                    for key, label in bindings:
                        padding = " " * (key_width - len(key))
                        yield Static(f"  [bold]{escape(key)}[/]{padding}{escape(label)}")
                    yield Static("")
                yield Static(
                    "[dim]Mouse mode, copy, and paste work across every screen. "
                    "Press ctrl+t to toggle mouse support.[/dim]"
                )
        yield FittedFooter()

    def action_dismiss_help(self) -> None:
        self.app.pop_screen()
