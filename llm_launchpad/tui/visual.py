"""Launchpad themes and persisted visual preference helpers."""

from __future__ import annotations

from rich.markup import escape
from textual.theme import Theme


DEFAULT_TUI_THEME = "launchpad-dark"
DEFAULT_TUI_DENSITY = "comfortable"

TUI_THEME_OPTIONS = (
    ("Launchpad Dark", "launchpad-dark"),
    ("Launchpad Light", "launchpad-light"),
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
        # Three distinct steps, so a bordered panel reads as raised off the
        # screen rather than as the same black with a line around it.
        background="#050806",
        surface="#0b120d",
        panel="#131c15",
        boost="#17321e",
        success="#7bf168",
        warning="#ffd166",
        error="#ff6b6b",
        dark=True,
        luminosity_spread=0.1,
    ),
    Theme(
        name="launchpad-light",
        # The dark theme's greens drop below 3:1 on white, so every accent is
        # a deeper shade of the same hue.
        primary="#1a7f37",
        secondary="#2f7d4f",
        accent="#0f6b3a",
        foreground="#17221a",
        background="#f6f8f6",
        surface="#ffffff",
        panel="#e9efea",
        boost="#d7e6da",
        success="#1a7f37",
        warning="#9a6700",
        error="#cf222e",
        dark=False,
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
ACCENT_STYLE = "$primary"
SUCCESS_STYLE = "$success"
WARNING_STYLE = "$warning"
ERROR_STYLE = "$error"
MUTED_STYLE = "dim"

#: Status markers paired with a semantic style. Color is never the only
#: signal: every marker is a distinct shape so monochrome terminals stay usable.
#: Each glyph is East Asian Width "N" (one cell everywhere, never emoji), for
#: the same reason as the provider markers in ``markers.py``.
STATUS_MARKERS: dict[str, tuple[str, str]] = {
    "done": ("✓", SUCCESS_STYLE),
    "active": ("▸", ACCENT_STYLE),
    "pending": ("◦", MUTED_STYLE),
    "failed": ("✗", ERROR_STYLE),
    "paused": ("⋯", WARNING_STYLE),
    "skipped": ("⊘", MUTED_STYLE),
}


def accent_title(text: str) -> str:
    """Render a bold accent title that follows the active theme."""
    return f"[bold {ACCENT_STYLE}]{escape(text)}[/]"


def status_markup(status: str, text: str) -> str:
    """Render text with a semantic status style plus an ASCII marker."""
    marker, style = STATUS_MARKERS.get(status, STATUS_MARKERS["pending"])
    return f"[{style}]{marker}[/] {escape(text)}"


# ---------------------------------------------------------------------------
# Screen headings
#
# Every screen opens with the same shape: an accent title, then either a step
# trail (inside a multi-step flow) or a muted one-line context. Screens used to
# compose this by hand, so the step wording, separators and emphasis drifted
# from one screen to the next.
# ---------------------------------------------------------------------------

#: The quick deploy flow: model picker, placement picker, confirmation.
DEPLOY_STEPS = ("Model", "Placement", "Confirm")
#: The advanced flow: engine choice, then the engine's form.
ADVANCED_DEPLOY_STEPS = ("Engine", "Configure")

# "›" rather than "→": the arrow is East Asian Width "A" and draws two cells in
# a CJK-configured terminal.
STEP_SEPARATOR = " [dim]›[/] "


def step_trail(steps: tuple[str, ...], current: int) -> str:
    """Render ``✓ Done › ▸ Current › ◦ Next`` for a multi-step flow."""
    chips: list[str] = []
    for index, step in enumerate(steps):
        if index < current:
            chips.append(f"[{SUCCESS_STYLE}]✓[/] [dim]{escape(step)}[/dim]")
        elif index == current:
            chips.append(f"[bold]▸ {escape(step)}[/bold]")
        else:
            chips.append(f"[dim]◦ {escape(step)}[/dim]")
    return STEP_SEPARATOR.join(chips)


def screen_title(
    title: str,
    context: str = "",
    *,
    steps: tuple[str, ...] = (),
    current: int = 0,
) -> str:
    """Render a screen heading: accent title, step trail, muted context.

    ``context`` is plain text and is escaped here.
    """
    parts = [accent_title(title)]
    if steps:
        parts.append(step_trail(steps, current))
    if context:
        parts.append(f"[dim]{escape(context)}[/dim]")
    return "   ".join(parts)


def tag(text: str, style: str) -> str:
    """Render a short label as a tinted tag, e.g. ``FOLLOWING`` or ``healthy``.

    ``style`` must be a theme color variable (``$success`` and friends): the
    tag is that color on a 20% tint of itself, so it follows the theme and
    stays legible in Monochrome, where every variable is white.
    """
    return f"[bold {style} on {style} 20%] {escape(text)} [/]"


def labelled_rows_markup(rows: list[tuple[str, str]]) -> str:
    """Render label/value rows with the values in one column.

    Hand-counted padding kept drifting: the connection card lined its values up
    and the result card did not, so `Status  Healthy` and `Test command  curl`
    began at different columns in the same style of panel. Measuring the widest
    label keeps every card aligned and every future row aligned with it.

    Values are markup, so a row can carry colour; callers escape plain text.
    Escaping here printed the Tools row's own ``[/]`` as text.
    """
    if not rows:
        return ""
    width = max(len(label) for label, _ in rows) + 2
    return "\n".join(
        f"[dim]{escape(label)}[/dim]{' ' * (width - len(label))}{value}"
        for label, value in rows
    )
