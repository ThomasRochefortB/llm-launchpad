"""Setup-required screen shown when no compute provider is authenticated."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Footer, Static

from .copy_enabled import CopyEnabledScreen

_MODAL_COMMAND = "modal setup"
_PRIME_COMMAND = "prime login"


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
                    "compute provider. None is currently authenticated.\n",
                    id="setup-required-intro",
                )
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
                    "[dim]Authenticate in another terminal, then press "
                    "[bold]r[/bold] to re-check. Quit with esc.[/dim]",
                    id="setup-required-hint",
                )
                yield Static("", id="setup-required-feedback")
                with Horizontal(id="setup-required-actions"):
                    yield Button("Re-check", id="setup-recheck-btn", variant="primary")
                    yield Button("Copy modal setup", id="setup-copy-modal-btn")
                    yield Button("Copy prime login", id="setup-copy-prime-btn")
                    yield Button("Quit", id="setup-quit-btn", variant="error")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#setup-recheck-btn", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "setup-recheck-btn":
            self.action_recheck()
        elif event.button.id == "setup-copy-modal-btn":
            self._copy_command(_MODAL_COMMAND)
        elif event.button.id == "setup-copy-prime-btn":
            self._copy_command(_PRIME_COMMAND)
        elif event.button.id == "setup-quit-btn":
            self.action_quit_app()

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
        self.query_one("#setup-required-feedback", Static).update(
            "[yellow]Still no authenticated provider.[/yellow] "
            f"Run {_MODAL_COMMAND} or {_PRIME_COMMAND}, then re-check."
        )

    def action_quit_app(self) -> None:
        quit_action = getattr(self.app, "action_request_quit", None)
        if callable(quit_action):
            quit_action()
            return
        self.app.exit()
