"""Shared text and number formatting helpers for Textual screens."""

from __future__ import annotations


def clip(value: str, width: int) -> str:
    """Trim ``value`` to ``width`` columns, ending with an ellipsis when cut."""
    text = (value or "").strip()
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return f"{text[:width - 3]}..."


def format_money(value: float) -> str:
    """Render a USD amount with two decimals."""
    return f"${value:,.2f}"


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
