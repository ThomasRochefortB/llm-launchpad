"""Launchpad themes and persisted visual preference helpers."""

from __future__ import annotations

from rich.markup import escape
from textual.theme import Theme


DEFAULT_TUI_THEME = "launchpad-dark"
DEFAULT_TUI_DENSITY = "comfortable"

TUI_THEME_OPTIONS = (
    ("Launchpad Dark", "launchpad-dark"),
    ("High Contrast", "launchpad-high-contrast"),
    ("Monochrome", "launchpad-monochrome"),
)
TUI_DENSITY_OPTIONS = (
    ("Comfortable", "comfortable"),
    ("Compact", "compact"),
)

_THEME_NAMES = frozenset(value for _, value in TUI_THEME_OPTIONS)
_DENSITY_NAMES = frozenset(value for _, value in TUI_DENSITY_OPTIONS)


LAUNCHPAD_THEMES = (
    Theme(
        name="launchpad-dark",
        primary="#7bf168",
        secondary="#4dc879",
        accent="#95ff85",
        foreground="#eef7ef",
        background="#050806",
        surface="#070b08",
        panel="#0f1410",
        boost="#17321e",
        success="#7bf168",
        warning="#ffd166",
        error="#ff6b6b",
        dark=True,
        luminosity_spread=0.1,
    ),
    Theme(
        name="launchpad-high-contrast",
        primary="#00ff66",
        secondary="#00e5ff",
        accent="#ffff00",
        foreground="#ffffff",
        background="#000000",
        surface="#000000",
        panel="#080808",
        boost="#ffffff",
        success="#00ff66",
        warning="#ffff00",
        error="#ff4d4d",
        dark=True,
        luminosity_spread=0.2,
        text_alpha=1.0,
    ),
    Theme(
        name="launchpad-monochrome",
        primary="#ffffff",
        secondary="#d0d0d0",
        accent="#ffffff",
        foreground="#ffffff",
        background="#000000",
        surface="#080808",
        panel="#111111",
        boost="#808080",
        success="#ffffff",
        warning="#ffffff",
        error="#ffffff",
        dark=True,
        luminosity_spread=0.12,
        text_alpha=1.0,
    ),
)


def normalize_tui_theme(value: object) -> str:
    """Return a supported theme name, falling back safely."""
    normalized = str(value or "").strip().lower()
    return normalized if normalized in _THEME_NAMES else DEFAULT_TUI_THEME


def normalize_tui_density(value: object) -> str:
    """Return a supported density name, falling back safely."""
    normalized = str(value or "").strip().lower()
    return normalized if normalized in _DENSITY_NAMES else DEFAULT_TUI_DENSITY


# ---------------------------------------------------------------------------
# Semantic styling helpers (item 6: visual consistency)
#
# Hard-coded accent hex values in Rich markup bypass the active Textual theme,
# so High Contrast and Monochrome kept rendering Launchpad-Dark green. These
# helpers emit theme-variable styles (``primary``/``success``/``warning`` /
# ``error``) which Textual resolves against the active theme.
# ---------------------------------------------------------------------------

#: Theme-variable style names, not hex values. Use these in Rich markup so the
#: active theme decides the actual color.
ACCENT_STYLE = "primary"
SUCCESS_STYLE = "success"
WARNING_STYLE = "warning"
ERROR_STYLE = "error"
MUTED_STYLE = "dim"

#: ASCII status markers paired with a semantic style. Color is never the only
#: signal: every marker has distinct text so monochrome terminals stay usable.
STATUS_MARKERS: dict[str, tuple[str, str]] = {
    "done": ("OK", SUCCESS_STYLE),
    "active": (">>", ACCENT_STYLE),
    "pending": ("..", MUTED_STYLE),
    "failed": ("XX", ERROR_STYLE),
    "paused": ("||", WARNING_STYLE),
    "skipped": ("--", MUTED_STYLE),
}


def accent_title(text: str) -> str:
    """Render a bold accent title that follows the active theme."""
    return f"[bold {ACCENT_STYLE}]{escape(text)}[/]"


def accent_markup(text: str) -> str:
    """Render inline accent text that follows the active theme."""
    return f"[{ACCENT_STYLE}]{escape(text)}[/]"


def status_markup(status: str, text: str) -> str:
    """Render text with a semantic status style plus an ASCII marker."""
    marker, style = STATUS_MARKERS.get(status, ("..", MUTED_STYLE))
    return f"[{style}]{marker}[/] {escape(text)}"
