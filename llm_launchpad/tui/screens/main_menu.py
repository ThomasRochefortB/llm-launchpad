"""Main menu screen: deploy, manage endpoints, settings.

Inspired by the Codex TUI main screen with a prominent banner,
auth status line, and keyboard-navigable option list.
"""

from __future__ import annotations

from collections import Counter

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Footer, OptionList, Static
from textual.widgets.option_list import Option

from ...core.backend import ModalBackend
from ...protocol.models import EndpointInfo

BANNER = r"""[bold cyan]
_     _     __  __
| |   | |   |  \/  |
| |   | |   | |\/| |
| |___| |___| |  | |
|_____|_____|_|  |_|
    LAUNCHPAD
[/bold cyan]"""


class DeploymentsLoaded(Message):
    """Main-menu deployment status fetch completed."""

    def __init__(self, rows: list[EndpointInfo]) -> None:
        super().__init__()
        self.rows = rows


class DeploymentsLoadFailed(Message):
    """Main-menu deployment status fetch failed."""

    def __init__(self, error: str) -> None:
        super().__init__()
        self.error = error


def _clip(value: str, width: int) -> str:
    text = (value or "").strip()
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return f"{text[:width - 3]}..."


def _state_bucket(state: str) -> str:
    normalized = (state or "").strip().lower()
    if normalized in {"running", "deployed"}:
        return "healthy"
    if normalized in {"deploying", "starting", "initializing", "building"}:
        return "deploying"
    if normalized in {"queued", "pending"}:
        return "queued"
    if normalized in {"failed", "error", "crashed"}:
        return "error"
    if normalized in {"stopped", "stopping"}:
        return "stopped"
    return "other"


def _style_state(state: str) -> str:
    normalized = (state or "").strip().lower() or "unknown"
    bucket = _state_bucket(normalized)
    if bucket == "healthy":
        return f"[green]{normalized}[/green]"
    if bucket == "deploying":
        return f"[cyan]{normalized}[/cyan]"
    if bucket == "queued":
        return f"[yellow]{normalized}[/yellow]"
    if bucket == "error":
        return f"[red]{normalized}[/red]"
    if bucket == "stopped":
        return f"[magenta]{normalized}[/magenta]"
    return f"[dim]{normalized}[/dim]"


def _should_show_in_panel(state: str) -> bool:
    bucket = _state_bucket(state)
    return bucket in {"healthy", "deploying", "queued", "error"}


def _render_deployment_status(rows: list[EndpointInfo]) -> str:
    if not rows:
        return (
            "[bold]Fleet Pulse[/bold]\n"
            "[dim]No active launchpad deployments.[/dim]\n"
            "\n"
            "[dim]Use Manage -> List deployments for full details.[/dim]"
        )

    state_counts = Counter(_state_bucket(row.state) for row in rows)
    backend_counts = Counter(
        row.backend.value if row.backend is not None else "unknown"
        for row in rows
    )
    healthy = state_counts.get("healthy", 0)
    pending = state_counts.get("deploying", 0) + state_counts.get("queued", 0)
    issues = state_counts.get("error", 0)

    header_lines = [
        "[bold]Fleet Pulse[/bold]",
        f"[dim]deployments={len(rows)}  healthy={healthy}  pending={pending}  issues={issues}[/dim]",
        f"[dim]vllm={backend_counts.get('vllm', 0)}  llamacpp={backend_counts.get('llamacpp', 0)}[/dim]",
        "",
    ]

    display_rows = sorted(
        rows,
        key=lambda row: (
            {"healthy": 0, "deploying": 1, "queued": 2, "error": 3, "stopped": 4, "other": 5}.get(
                _state_bucket(row.state),
                6,
            ),
            row.name.casefold(),
        ),
    )[:8]

    app_lines = []
    for row in display_rows:
        backend = row.backend.value if row.backend is not None else "unknown"
        instance = row.instance_name or "-"
        app_lines.append(
            "  "
            f"[bold]{_clip(backend, 8):<8}[/bold] "
            f"{_clip(instance, 12):<12} "
            f"{_clip(row.name, 20):<20} "
            f"{_style_state(row.state)}"
        )
    if len(rows) > len(display_rows):
        app_lines.append(f"[dim]  ... and {len(rows) - len(display_rows)} more[/dim]")

    return "\n".join(header_lines + app_lines)


class MainMenuScreen(Screen):
    """Top-level menu: deploy, manage, settings."""

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("d", "select_deploy", "Deploy", show=True),
        Binding("m", "select_manage", "Manage", show=True),
        Binding("t", "select_storage", "Storage", show=True),
        Binding("s", "select_settings", "Settings", show=True),
    ]

    def __init__(self, username: str = "", version: str = "") -> None:
        super().__init__()
        self.username = username
        self.version = version
        self._status_refresh_inflight = False

    def compose(self) -> ComposeResult:
        with Center():
            with Horizontal(id="main-menu-layout"):
                with Vertical(id="menu-container"):
                    yield Static(BANNER, id="banner-text")
                    version_text = f"v{self.version}  " if self.version else ""
                    yield Static(
                        f"[bold]{version_text}[/bold][dim]Modal LLM backends[/dim]",
                        classes="centered",
                    )
                    if self.username:
                        yield Static(
                            f"[green]  Authenticated as {self.username}[/green]",
                            id="auth-status",
                        )
                    yield Static("")  # spacer
                    yield OptionList(
                        Option("  Deploy            Launch a new LLM backend", id="deploy"),
                        Option("  Manage            List, status, logs, stop", id="manage"),
                        Option("  Storage           Cached models and pre-download", id="storage"),
                        Option("  Settings          Scaledown defaults", id="settings"),
                        id="action-list",
                    )
                with Vertical(id="deployment-status-panel"):
                    yield Static("[bold cyan]Deployment Status[/bold cyan]", id="deployment-status-title")
                    yield Static("[dim]Refreshing deployment status...[/dim]", id="deployment-status-body")
        yield Footer()

    def on_mount(self) -> None:
        """Focus the option list so arrow-key navigation works immediately."""
        self.query_one("#action-list", OptionList).focus()
        self._refresh_deployment_status()
        self.set_interval(20.0, self._refresh_deployment_status)

    def _refresh_deployment_status(self) -> None:
        if self._status_refresh_inflight:
            return
        self._status_refresh_inflight = True
        self.query_one("#deployment-status-body", Static).update("[dim]Refreshing deployment status...[/dim]")
        self.run_worker(
            self._run_load_deployments,
            name="main-menu-status-worker",
            thread=True,
        )

    def _run_load_deployments(self) -> None:
        poster = getattr(self, "post_message", None)
        if poster is None:
            return
        try:
            rows = ModalBackend.list_apps()
        except Exception as exc:
            poster(DeploymentsLoadFailed(error=str(exc)))
            return
        if rows is None:
            poster(DeploymentsLoadFailed(error="Could not read Modal app list."))
            return
        poster(DeploymentsLoaded(rows=[row for row in rows if row.backend is not None]))

    def on_deployments_loaded(self, message: DeploymentsLoaded) -> None:
        self._status_refresh_inflight = False
        visible_rows = [row for row in message.rows if _should_show_in_panel(row.state)]
        self.query_one("#deployment-status-body", Static).update(_render_deployment_status(visible_rows))

    def on_deployments_load_failed(self, message: DeploymentsLoadFailed) -> None:
        self._status_refresh_inflight = False
        self.query_one("#deployment-status-body", Static).update(
            "[bold]Fleet Pulse[/bold]\n"
            "[yellow]Status unavailable.[/yellow]\n"
            f"[dim]{_clip(message.error, 80)}[/dim]"
        )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        option_id = event.option.id
        if option_id == "deploy":
            self.app.action_push_deploy()  # type: ignore[attr-defined]
        elif option_id == "manage":
            self.app.action_push_manage()  # type: ignore[attr-defined]
        elif option_id == "storage":
            self.app.action_push_storage()  # type: ignore[attr-defined]
        elif option_id == "settings":
            self.app.action_push_settings()  # type: ignore[attr-defined]

    def action_select_deploy(self) -> None:
        self.app.action_push_deploy()  # type: ignore[attr-defined]

    def action_select_manage(self) -> None:
        self.app.action_push_manage()  # type: ignore[attr-defined]

    def action_select_storage(self) -> None:
        self.app.action_push_storage()  # type: ignore[attr-defined]

    def action_select_settings(self) -> None:
        self.app.action_push_settings()  # type: ignore[attr-defined]

    async def action_quit(self) -> None:
        await self.app.action_quit()  # type: ignore[attr-defined]
