"""Setup-required screen shown when no compute provider has credentials."""

from __future__ import annotations

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Static

from ..widgets.fitted_footer import FittedFooter
from .copy_enabled import CopyEnabledScreen

_MODAL_COMMAND = "modal setup"
_PRIME_COMMAND = "prime login"
_VAST_COMMAND = "llm-launchpad vast-auth login"


class SetupRequiredScreen(CopyEnabledScreen):
    """Explain how to authenticate instead of flashing an unreadable toast."""

    BINDINGS = [
        Binding("escape,q", "quit_app", "Quit", show=True),
        Binding("r", "recheck", "Re-check", show=True),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="setup-required-root"):
            with VerticalScroll(id="setup-required-scroll", classes="screen-scroll"):
                yield Static("[bold #7bf168]Compute provider required[/]")
                yield Static(
                    "llm-launchpad deploys inference endpoints through at least one "
                    "compute provider. None currently has credentials.\n",
                    id="setup-required-intro",
                )
                yield Static("[dim]Checking provider credentials…[/dim]", id="setup-readiness-status")
                yield Static(
                    "[bold]Option 1 · Modal[/bold]\n"
                    "[dim]Install the CLI, then authenticate:[/dim]\n"
                    f"  {_MODAL_COMMAND}",
                    classes="setup-option",
                )
                yield Static(
                    "[bold]Option 2 · Prime Intellect[/bold]\n"
                    "[dim]Authenticate from your terminal:[/dim]\n"
                    f"  {_PRIME_COMMAND}",
                    classes="setup-option",
                )
                yield Static(
                    "[bold]Option 3 · Vast.ai[/bold]\n"
                    "[dim]Save a marketplace API key, here or in your terminal:[/dim]\n"
                    f"  {_VAST_COMMAND}",
                    classes="setup-option",
                )
                yield Static(
                    "[dim]Authenticate in another terminal, then press "
                    "[bold]r[/bold] to re-check. Quit with esc.[/dim]",
                    id="setup-required-hint",
                )
                yield Static("", id="setup-required-feedback")
                yield Button("Set up Vast.ai key", id="setup-vast-preview-btn")
                with Horizontal(id="setup-required-actions"):
                    yield Button("Re-check", id="setup-recheck-btn", variant="primary")
                    yield Button("Copy Modal", id="setup-copy-modal-btn")
                    yield Button("Copy Prime", id="setup-copy-prime-btn")
                    yield Button("Copy Vast", id="setup-copy-vast-btn")
                    yield Button("Quit", id="setup-quit-btn", variant="error")
        yield FittedFooter()

    def on_mount(self) -> None:
        # Focus without scrolling. This screen exists to explain how to
        # authenticate, and letting Textual scroll the button into view opened
        # it below its own title and all three options on a short terminal --
        # the reader landed mid-document with no sign of what came above.
        self.query_one("#setup-recheck-btn", Button).focus(scroll_visible=False)
        self.query_one("#setup-required-scroll", VerticalScroll).scroll_home(
            animate=False,
        )
        self._render_readiness(self._local_readiness())

    def _local_readiness(self) -> tuple[object, ...]:
        snapshot = getattr(self.app, "provider_readiness_snapshot", None)
        if callable(snapshot):
            try:
                return tuple(snapshot())
            except Exception:
                return ()
        return ()

    def on_provider_readiness(self, readiness: tuple[object, ...]) -> None:
        """Update installed/checking/authenticated/failed without re-mounting."""
        self._render_readiness(readiness)

    def _render_readiness(self, readiness: tuple[object, ...]) -> None:
        try:
            status = self.query_one("#setup-readiness-status", Static)
        except Exception:
            return
        if not readiness:
            status.update("[dim]Checking provider credentials…[/dim]")
            return
        lines = []
        for row in readiness:
            provider = getattr(row, "provider", None)
            name = getattr(provider, "display_name", str(provider))
            stage = getattr(getattr(row, "stage", None), "display_name", str(getattr(row, "stage", "")))
            detail = str(getattr(row, "detail", "") or "")
            color = {
                "Authenticated": "green",
                "Checking": "yellow",
                "Not configured": "dim",
                "Not installed": "dim",
                "Authentication failed": "red",
                "Verification unavailable": "yellow",
            }.get(stage, "")
            label = f"[{color}]{escape(stage)}[/]" if color else escape(stage)
            lines.append(f"• {escape(str(name))}: {label}" + (f" — {escape(detail)}" if detail else ""))
        status.update("\n".join(lines))

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "setup-recheck-btn":
            self.action_recheck()
        elif event.button.id == "setup-copy-modal-btn":
            self._copy_command(_MODAL_COMMAND)
        elif event.button.id == "setup-copy-prime-btn":
            self._copy_command(_PRIME_COMMAND)
        elif event.button.id == "setup-copy-vast-btn":
            self._copy_command(_VAST_COMMAND)
        elif event.button.id == "setup-quit-btn":
            await self.action_quit_app()
        elif event.button.id == "setup-vast-preview-btn":
            from .vast import VastPreviewScreen

            self.app.push_screen(VastPreviewScreen())

    def _copy_command(self, command: str) -> None:
        self.app.copy_to_clipboard(command)
        self.query_one("#setup-required-feedback", Static).update(
            f"[green]Copied:[/green] [bold]{command}[/bold]"
        )

    def action_recheck(self) -> None:
        """Re-run provider detection and enter the TUI when configured."""
        rechecker = getattr(self.app, "recheck_provider_setup", None)
        entered = bool(callable(rechecker) and rechecker())
        if entered:
            return
        refresher = getattr(self.app, "refresh_provider_readiness", None)
        if callable(refresher):
            try:
                refresher()
            except Exception:
                pass
        self._render_readiness(self._local_readiness())
        self.query_one("#setup-required-feedback", Static).update(
            "[yellow]Still no provider with credentials.[/yellow] "
            f"Run {_MODAL_COMMAND}, {_PRIME_COMMAND}, or {_VAST_COMMAND}, "
            "then re-check."
        )

    async def action_quit_app(self) -> None:
        quit_action = getattr(self.app, "action_request_quit", None)
        if callable(quit_action):
            await quit_action()
            return
        self.app.exit()
