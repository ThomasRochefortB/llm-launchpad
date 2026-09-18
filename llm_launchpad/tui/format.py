"""Shared text and number formatting helpers for Textual screens."""

from __future__ import annotations

from rich.cells import cell_len, set_cell_size


def clip(value: str, width: int) -> str:
    """Trim ``value`` to ``width`` columns, ending with an ellipsis when cut."""
    text = (value or "").strip()
    if width <= 0:
        return ""
    if cell_len(text) <= width:
        return text
    if width <= 3:
        return set_cell_size(text, width).rstrip()
    return f"{set_cell_size(text, width - 3).rstrip()}..."


def format_money(value: float, *, decimals: int = 2) -> str:
    """Render a USD amount, allowing extra precision for per-unit rates."""
    return f"${value:,.{decimals}f}"


def format_gib(value: float) -> str:
    """Render a GiB amount, shedding decimals as the magnitude grows."""
    if value >= 100:
        return f"{value:,.0f} GiB"
    if value >= 10:
        return f"{value:,.1f} GiB"
    return f"{value:,.2f} GiB"


def format_free_tier(value_gib: float) -> str:
    """Render a free-tier allowance, promoting whole TiB values to TiB."""
    if value_gib > 0 and value_gib % 1024 == 0:
        return f"{value_gib / 1024:,.0f} TiB"
    return format_gib(value_gib)


def format_age(seconds: float) -> str:
    """Render how old a reading is, in the coarsest unit that stays honest."""
    if seconds < 60:
        return "just now"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m ago"
    hours = int(seconds // 3600)
    if hours < 24:
        return f"{hours}h ago"
    days = int(seconds // 86400)
    return f"{days}d ago"


def format_token_count(value: float) -> str:
    """Render a token total, shedding digits as the magnitude grows."""
    count = max(float(value), 0.0)
    # The promotion rule below has to be reached before the plain branch, or
    # the lowest boundary is the one place it never applies: 999.6 rounded up
    # to "1,000" -- five columns in a field sized for four, and the only
    # decade that failed to shed its digits.
    if count < 1e3 * 0.9995:
        return f"{count:,.0f}"
    for divisor, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        # Promote a count that only rounds up into the next unit, so 999,999
        # reads as 1.0M rather than 1,000K.
        if count >= divisor * 0.9995:
            scaled = count / divisor
            return f"{scaled:,.1f}{suffix}" if scaled < 10 else f"{scaled:,.0f}{suffix}"
    return f"{count:,.0f}"


def format_token_rate(value: float) -> str:
    """Render generation throughput at a precision the number can support."""
    rate = max(float(value), 0.0)
    if rate == 0:
        # An idle endpoint is exactly idle; a decimal place only implies a
        # precision the reading does not have.
        return "0 tok/s"
    # Hand over at the same boundary the count formatter promotes on, so a
    # rate that rounds up into the next unit is never printed as "1,000".
    if rate >= 1e3 * 0.9995:
        return f"{format_token_count(rate)} tok/s"
    if rate >= 10:
        return f"{rate:,.0f} tok/s"
    return f"{rate:,.1f} tok/s"
