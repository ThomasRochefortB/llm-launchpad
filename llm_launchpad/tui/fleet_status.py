"""Shared wording for a fleet whose providers did not all answer.

The fleet screens each render their own layout, but they must not describe the
same outage in two different ways: an unreachable provider is reported with
what it last returned and how old that is, never as a shorter list.
"""

from __future__ import annotations

import time

from rich.markup import escape

from ..protocol.models import FleetDiscovery
from .format import clip, format_age

_ERROR_WIDTH = 72


def provider_outage_lines(
    discovery: FleetDiscovery | None,
    *,
    now: float | None = None,
) -> list[str]:
    """Return one styled line per provider that could not be reached."""
    if discovery is None:
        return []
    moment = time.time() if now is None else now
    lines: list[str] = []
    for listing in discovery.unavailable:
        name = listing.provider.display_name
        reason = escape(clip(listing.error or "unknown error", _ERROR_WIDTH))
        age = listing.age_seconds(moment)
        if listing.rows and age is not None:
            count = len(listing.rows)
            noun = "endpoint" if count == 1 else "endpoints"
            detail = f"showing {count} {noun} from {format_age(age)}"
        else:
            detail = "its deployments are not listed"
        lines.append(f"[yellow]{escape(name)} unavailable:[/yellow] {reason} [dim]({detail})[/dim]")
    return lines


def retained_providers(discovery: FleetDiscovery | None) -> set[str]:
    """Provider values whose rows came from a failed pass and may be stale."""
    if discovery is None:
        return set()
    return {
        listing.provider.value for listing in discovery.unavailable if listing.is_retained
    }
