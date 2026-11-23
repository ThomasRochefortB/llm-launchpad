"""Setup-required screen shown when no compute provider has credentials."""

from __future__ import annotations

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Static

from ...core.provider_readiness import ProviderReadiness, ProviderReadinessStage
from ...protocol.enums import ComputeProvider
from ..widgets.fitted_footer import FittedFooter
from .copy_enabled import CopyEnabledScreen
from ..visual import screen_title

_SETUP_COMMANDS = {
    ComputeProvider.MODAL: "modal setup",
    ComputeProvider.PRIME: "prime login",
    ComputeProvider.VAST: "llm-launchpad vast-auth login",
}


class SetupRequiredScreen(CopyEnabledScreen):
    """Explain how to authenticate instead of flashing an unreadable toast."""

    BINDINGS = [
        Binding("escape,q", "quit_app", "Quit", show=True),
        Binding("r", "recheck", "Re-check", show=True),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="setup-required-root"):
            with VerticalScroll(id="setup-required-scroll", classes="screen-scroll"):
                yield Static(screen_title("Compute provider required"))
                yield Static("Connect a provider, then press r to re-check.", id="setup-required-intro")
                for provider, command in _SETUP_COMMANDS.items():
                    with Horizontal(classes="setup-option"):
                        yield Static(
                            self._provider_setup_text(provider, command),
                            id=f"setup-status-{provider.value}",
                            classes="setup-provider-status",
                        )
                        yield Button(
                            "Copy",
                            id=f"setup-copy-{provider.value}-btn",
                            classes="setup-copy-command",
                        )
                with Horizontal(id="setup-required-actions"):
                    yield Button("Re-check", id="setup-recheck-btn", variant="primary")
                    yield Button("Vast.ai key", id="setup-vast-preview-btn")
                    yield Button("Quit", id="setup-quit-btn")
                yield Static("", id="setup-required-feedback")
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

    def on_screen_resume(self) -> None:
        # A check can finish while Help or the Vast setup form covers Setup.
        resume = getattr(self.app, "continue_from_setup", None)
        if callable(resume):
            self.call_after_refresh(resume)

    def _local_readiness(self) -> tuple[ProviderReadiness, ...]:
        snapshot = getattr(self.app, "provider_readiness_snapshot", None)
        if callable(snapshot):
            try:
                return tuple(snapshot())
            except Exception:
                return ()
        return ()

    def on_provider_readiness(self, readiness: tuple[ProviderReadiness, ...]) -> None:
        """Update installed/checking/authenticated/failed without re-mounting."""
        self._render_readiness(readiness)

    @staticmethod
    def _provider_setup_text(
        provider: ComputeProvider, command: str, row: ProviderReadiness | None = None,
    ) -> str:
        stage = row.stage if row is not None else ProviderReadinessStage.CREDENTIALS_PRESENT
        style = {
            ProviderReadinessStage.READY: "$success",
            ProviderReadinessStage.AUTH_FAILED: "$error",
            ProviderReadinessStage.UNREACHABLE: "$warning",
        }.get(stage, "dim")
        detail = f" · {escape(row.detail)}" if row is not None and row.detail else ""
        return (
            f"[bold]{escape(provider.display_name)}[/] · [{style}]{escape(stage.display_name)}[/]"
            f"{detail}\n{escape(command)}"
        )

    def _render_readiness(self, readiness: tuple[ProviderReadiness, ...]) -> None:
        by_provider = {row.provider: row for row in readiness}
        for provider, command in _SETUP_COMMANDS.items():
            self.query_one(f"#setup-status-{provider.value}", Static).update(
                self._provider_setup_text(provider, command, by_provider.get(provider))
            )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "setup-recheck-btn":
            self.action_recheck()
        elif button_id == "setup-quit-btn":
            await self.action_quit_app()
        elif button_id == "setup-vast-preview-btn":
            from .vast import VastPreviewScreen

            self.app.push_screen(VastPreviewScreen())
        else:
            for provider, command in _SETUP_COMMANDS.items():
                if button_id == f"setup-copy-{provider.value}-btn":
                    self._copy_command(command)
                    break

    def _copy_command(self, command: str) -> None:
        self.app.copy_to_clipboard(command)
        self.query_one("#setup-required-feedback", Static).update(
            f"Copy requested: [bold]{command}[/bold]"
        )

    def action_recheck(self) -> None:
        """Request another background pass and display its latest results."""
        refresher = getattr(self.app, "refresh_provider_readiness", None)
        if callable(refresher):
            refresher()
        self._render_readiness(self._local_readiness())
        self.query_one("#setup-required-feedback", Static).update("")

    async def action_quit_app(self) -> None:
        quit_action = getattr(self.app, "action_request_quit", None)
        if callable(quit_action):
            await quit_action()
            return
        self.app.exit()
