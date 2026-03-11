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
        background: #070b08;
        border-bottom: solid #17321e;
    }
    """

    backend: reactive[str] = reactive("--")
    state: reactive[str] = reactive("idle")
    operation: reactive[str] = reactive("--")
    detail: reactive[str] = reactive("")

    def render(self) -> str:
        state_icon = _state_icon(self.state)
        parts = [
            f"[bold #7bf168]backend:[/] {self.backend}",
            f"  {state_icon} [bold #7bf168]state:[/] {self.state}",
        ]
        if self.operation and self.operation != "--":
            parts.append(f"  [bold #7bf168]op:[/] {self.operation}")
        if self.detail:
            parts.append(f"  [dim]{self.detail}[/]")
        return " ".join(parts)

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
        "running": "[#7bf168]>[/]",
        "deploying": "[#7bf168]>>[/]",
        "warming_up": "[yellow]*[/]",
        "healthy": "[green]OK[/]",
        "unhealthy": "[red]X[/]",
        "stopped": "[dim].[/]",
        "error": "[red]![/]",
        "cancelled": "[dim]-[/]",
    }
    return icons.get(state, "[dim]?[/]")
