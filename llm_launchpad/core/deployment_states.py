"""Provider lifecycle states shared by fleet listing and connection display.

Modal, Prime and Vast each spell their terminal states differently, but every
caller that asks "is this deployment over?" needs the same answer: a stopped
app must not be counted as live, deduped as live, or handed out as a URL.
"""

from __future__ import annotations


# Terminal or winding-down states. A row in one of these will not start serving
# again on its own, so nothing should be derived from it as though it might.
TERMINAL_DEPLOYMENT_STATES = frozenset(
    {"stopped", "stopping", "terminated", "archived"}
)


def is_terminal_deployment_state(state: str) -> bool:
    """Return whether *state* names a deployment that is over."""
    return (state or "").strip().lower() in TERMINAL_DEPLOYMENT_STATES
