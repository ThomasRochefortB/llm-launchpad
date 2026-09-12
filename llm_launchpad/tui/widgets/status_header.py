"""Status header widget: shows backend, state, and operation context."""

from __future__ import annotations

from rich.markup import escape
from textual.reactive import reactive
from textual.widgets import Static

from ...protocol.enums import BackendType, DeploymentState, OperationType


class StatusHeader(Static):
    """Top-of-screen context bar showing current operation state."""

    DEFAULT_CSS = """
    StatusHeader {
        /* One content row plus the bottom border; render() is a single line. */
        height: 2;
        padding: 0 2;
        background: $surface;
        border-bottom: solid $border;
        /* The operation detail is a full provider command line. Letting it
           wrap turns the context bar into three lines of truncated JSON; the
           log pane below already carries the command in full. */
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }
    """

    backend: reactive[str] = reactive("--")
    state: reactive[str] = reactive("idle")
    operation: reactive[str] = reactive("--")
    detail: reactive[str] = reactive("")

    def render(self) -> str:
        state_icon = _state_icon(self.state)
        compact = self.screen.has_class("viewport-compact")
        short = self.screen.has_class("viewport-short")
        parts = [
            f"[bold]backend:[/] {escape(self.backend)}",
            f"  {state_icon} [bold]state:[/] {escape(self.state)}",
        ]
        if not compact and self.operation and self.operation != "--":
            parts.append(f"  [bold]op:[/] {escape(self.operation)}")
        if not compact and not short and self.detail:
            parts.append(f"  [dim]{escape(self.detail)}[/]")
        return " ".join(parts)

    def watch_state(self, state: str) -> None:
        """Expose semantic state to CSS without relying on color alone."""
        for name in DeploymentState:
            self.remove_class(f"state-{name.value.replace('_', '-')}")
        self.add_class(f"state-{state.replace('_', '-')}")

    def update_from_event(
        self,
        state: DeploymentState | None = None,
        backend: BackendType | None = None,
        operation: OperationType | None = None,
        detail: str = "",
    ) -> None:
        if backend is not None:
            self.backend = backend.value
        if state is not None:
            self.state = state.value
        if operation is not None:
            self.operation = operation.value
        if detail:
            self.detail = detail


# Deliberately ASCII: these have to stay readable with colour stripped, which
# is what the monochrome theme and low-colour terminals do. Every marker is
# padded to the same width so `state:` does not shift sideways as the state
# changes underneath it.
_STATE_MARKER_WIDTH = 2

# No marker may begin with "[": these are interpolated into Rich markup.
_STATE_MARKERS: dict[str, tuple[str, str]] = {
    "idle": ("..", "dim"),
    "queued": ("~~", "yellow"),
    "running": (">>", "green"),
    "deploying": ("^^", "green"),
    "warming_up": ("**", "yellow"),
    "healthy": ("OK", "green"),
    "unhealthy": ("!!", "red"),
    "stopped": ("--", "dim"),
    "error": ("XX", "red"),
    "cancelled": ("//", "dim"),
}


def _state_icon(state: str) -> str:
    marker, style = _STATE_MARKERS.get(state, ("??", "dim"))
    return f"[{style}]{marker:<{_STATE_MARKER_WIDTH}}[/]"
