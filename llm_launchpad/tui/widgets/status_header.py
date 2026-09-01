"""Status header widget: shows backend, state, and operation context."""

from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import Static

from ...protocol.enums import BackendType, DeploymentState, OperationType


class StatusHeader(Static):
    """Top-of-screen context bar showing current operation state."""

    DEFAULT_CSS = """
    StatusHeader {
        height: 3;
        padding: 0 2;
        background: $surface;
        border-bottom: solid $border;
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
            f"[bold]backend:[/] {self.backend}",
            f"  {state_icon} [bold]state:[/] {self.state}",
        ]
        if not compact and self.operation and self.operation != "--":
            parts.append(f"  [bold]op:[/] {self.operation}")
        if not compact and not short and self.detail:
            parts.append(f"  [dim]{self.detail}[/]")
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


def _state_icon(state: str) -> str:
    icons = {
        "idle": "[dim]o[/]",
        "queued": "[yellow]~[/]",
        "running": "[green]>[/]",
        "deploying": "[green]>>[/]",
        "warming_up": "[yellow]*[/]",
        "healthy": "[green]OK[/]",
        "unhealthy": "[red]X[/]",
        "stopped": "[dim].[/]",
        "error": "[red]![/]",
        "cancelled": "[dim]-[/]",
    }
    return icons.get(state, "[dim]?[/]")
