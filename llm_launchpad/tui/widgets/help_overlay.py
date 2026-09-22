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


def _iter_effective_bindings(
    screen: object,
) -> list[tuple[str, str]] | None:
    """Collect (keys, label) rows from the screen's *effective* bindings.

    Unlike :func:`_iter_screen_bindings` (class-level ``BINDINGS``), this reads
    ``screen.active_bindings``, which already applies ``check_action`` phase /
    tab filtering and priority shadowing. Returns ``None`` when the screen is
    not mounted (no app) so callers can fall back to class bindings.
    """
    active: object = getattr(screen, "active_bindings", None)
    app = getattr(screen, "app", None)
    if not isinstance(active, dict):
        return None
    # Scroll and focus plumbing is always active, but it is not screen help:
    # listing every arrow/scroll binding buries the screen's own actions and
    # breaks the measured key column. Keep copy/paste (real actions here)
    # and drop pure navigation chrome.
    _NAVIGATION_LABELS = frozenset(
        {
            "Scroll Up",
            "Scroll Down",
            "Scroll Left",
            "Scroll Right",
            "Scroll Home",
            "Scroll End",
            "Scroll To Top",
            "Scroll To Bottom",
            "Page Up",
            "Page Down",
            "Page Left",
            "Page Right",
            "Half Page Up",
            "Half Page Down",
            "Focus Next",
            "Focus Previous",
        }
    )
    if app is not None:
        get_key_display = getattr(app, "get_key_display", None)
    by_label: dict[str, list[str]] = {}
    order: list[str] = []
    for _key, entry in active.items():
        try:
            _namespace, binding, _enabled, _tooltip = entry
        except Exception:
            continue
        label = (getattr(binding, "description", "") or "").strip()
        if not label or label in _NAVIGATION_LABELS:
            continue
        try:
            raw_key = str(binding.key or "").strip()
        except Exception:
            raw_key = ""
        # The footer and the tests name the full key chord ("ctrl+shift+c");
        # the display form ("shift+^c") abbreviates it. Help must use the
        # full chord so readers can match it to documentation and tests.
        key_token = raw_key.split(",", 1)[0].strip() if raw_key else ""
        if not key_token:
            if callable(get_key_display):
                try:
                    key_token = str(get_key_display(binding))
                except Exception:
                    key_token = str(_key)
            else:
                key_token = str(_key)
            key_token = key_token.strip()
        if not key_token:
            continue
        if label not in by_label:
            by_label[label] = []
            order.append(label)
        if key_token not in by_label[label]:
            by_label[label].append(key_token)
    return [(", ".join(by_label[label]), label) for label in order]


def _focused_control_hints(screen: object) -> list[tuple[str, str]]:
    """Describe keys for the currently focused control, if any."""
    from textual.widgets import DataTable, Input, OptionList, Select

    try:
        focused = getattr(screen, "focused", None)
    except Exception:
        return []
    if isinstance(focused, Input):
        return [
            ("type", "Filter; Enter confirms"),
            ("esc", "Leave search"),
        ]
    if isinstance(focused, Select):
        expanded = bool(getattr(focused, "expanded", False))
        if expanded:
            return [("↑/↓, enter", "Choose option")]
        return [("enter/space", "Open choices"), ("↑/↓", "Move between controls")]
    if isinstance(focused, (OptionList, DataTable)):
        return [("↑/↓, enter", "Highlight and choose")]
    # LogViewer and generic focusables fall back to the global hint in compose.
    return []


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
        border: round $primary;
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
        try:
            effective = _iter_effective_bindings(screen)
        except Exception:
            effective = None
        if effective is not None:
            # active_bindings already merges app + screen with priority, so a
            # single contextual section stays truthful about what works now.
            if effective:
                sections.append(("Available now", effective))
            focused_hints = _focused_control_hints(screen)
            if focused_hints:
                sections.append(("Focused control", focused_hints))
            if not sections:
                # Mounted but no described bindings (e.g. minimum-size gate):
                # fall back to class bindings so help is never empty.
                fallback = _iter_screen_bindings(screen)
                if fallback:
                    sections.append(("This screen", fallback))
            return cls(sections)
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
            yield Static("[bold]Keyboard shortcuts[/]", id="help-overlay-title")
            with VerticalScroll(id="help-overlay-body"):
                for section_name, bindings in self._sections:
                    yield Static(f"[bold]{section_name}[/bold]")
                    for key, label in bindings:
                        padding = " " * (key_width - len(key))
                        yield Static(f"  [bold]{escape(key)}[/]{padding}{escape(label)}")
                    yield Static("")
                yield Static(
                    "[dim]Navigate with Tab, Shift+Tab, arrow keys, and Enter. "
                    "Select text and copy/paste with your terminal's shortcuts. "
                    "In tmux, use the terminal's selection override if needed "
                    "(usually Shift+drag).[/dim]"
                )
        yield FittedFooter()

    def action_dismiss_help(self) -> None:
        app = self.app
        # Restore focus to the control that opened help, preserving row and
        # search state that would otherwise be lost to a default focus target.
        previous_screen = None
        try:
            stack = list(app.screen_stack)
            if len(stack) >= 2:
                previous_screen = stack[-2]
        except Exception:
            previous_screen = None
        restore = getattr(previous_screen, "_focus_before_help", None) if previous_screen is not None else None
        app.pop_screen()
        if restore is not None:
            try:
                if restore.is_mounted and restore.can_focus:
                    app.set_timer(0.05, restore.focus)
            except Exception:
                pass
