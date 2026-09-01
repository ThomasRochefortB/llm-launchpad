"""Viewport classification shared by responsive TUI screens and widgets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from textual.geometry import Size


MIN_TERMINAL_WIDTH = 40
MIN_TERMINAL_HEIGHT = 12
NARROW_TERMINAL_WIDTH = 120
COMPACT_TERMINAL_WIDTH = 80
MINIMAL_TERMINAL_WIDTH = 60
ULTRA_WIDE_TERMINAL_WIDTH = 180
SHORT_TERMINAL_HEIGHT = 20
SHALLOW_TERMINAL_HEIGHT = 15


class WidthMode(StrEnum):
    """Mutually exclusive horizontal presentation modes."""

    MINIMAL = "minimal"
    COMPACT = "compact"
    STANDARD = "standard"
    WIDE = "wide"


class HeightMode(StrEnum):
    """Mutually exclusive vertical presentation modes."""

    SHALLOW = "shallow"
    SHORT = "short"
    TALL = "tall"


@dataclass(frozen=True)
class ViewportProfile:
    """Terminal capabilities used to choose content, not just dimensions."""

    width: int
    height: int
    width_mode: WidthMode
    height_mode: HeightMode
    too_small: bool
    ultra_wide: bool

    @classmethod
    def from_size(cls, size: Size) -> ViewportProfile:
        """Classify a terminal size on independent width and height axes."""
        if size.width < MINIMAL_TERMINAL_WIDTH:
            width_mode = WidthMode.MINIMAL
        elif size.width < COMPACT_TERMINAL_WIDTH:
            width_mode = WidthMode.COMPACT
        elif size.width < NARROW_TERMINAL_WIDTH:
            width_mode = WidthMode.STANDARD
        else:
            width_mode = WidthMode.WIDE

        if size.height <= SHALLOW_TERMINAL_HEIGHT:
            height_mode = HeightMode.SHALLOW
        elif size.height <= SHORT_TERMINAL_HEIGHT:
            height_mode = HeightMode.SHORT
        else:
            height_mode = HeightMode.TALL

        return cls(
            width=size.width,
            height=size.height,
            width_mode=width_mode,
            height_mode=height_mode,
            too_small=(
                size.width < MIN_TERMINAL_WIDTH
                or size.height < MIN_TERMINAL_HEIGHT
            ),
            ultra_wide=size.width >= ULTRA_WIDE_TERMINAL_WIDTH,
        )

    @property
    def narrow(self) -> bool:
        return self.width_mode != WidthMode.WIDE

    @property
    def compact(self) -> bool:
        return self.width_mode in {WidthMode.COMPACT, WidthMode.MINIMAL}

    @property
    def minimal(self) -> bool:
        return self.width_mode == WidthMode.MINIMAL

    @property
    def short(self) -> bool:
        return self.height_mode != HeightMode.TALL

    @property
    def shallow(self) -> bool:
        return self.height_mode == HeightMode.SHALLOW

    @property
    def class_names(self) -> tuple[str, ...]:
        """Return all responsive classes that should be active together."""
        classes = [f"viewport-{self.width_mode.value}"]
        if self.narrow:
            classes.append("viewport-narrow")
        if self.compact:
            classes.append("viewport-compact")
        if self.minimal:
            classes.append("viewport-minimal")
        if self.short:
            classes.append("viewport-short")
        if self.shallow:
            classes.append("viewport-shallow")
        if self.ultra_wide:
            classes.append("viewport-ultra-wide")
        if self.too_small:
            classes.append("viewport-too-small")
        return tuple(classes)


RESPONSIVE_CLASS_NAMES = frozenset(
    {
        "viewport-wide",
        "viewport-standard",
        "viewport-compact",
        "viewport-minimal",
        "viewport-narrow",
        "viewport-short",
        "viewport-shallow",
        "viewport-ultra-wide",
        "viewport-too-small",
    }
)
