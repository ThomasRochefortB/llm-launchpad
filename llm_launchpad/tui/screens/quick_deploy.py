"""Quick-deploy screen for curated llama.cpp coding profiles."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Button, Footer, Input, Static, Switch

from ...core.quick_deploy import (
    QuickDeployProfile,
    build_quick_deploy_config,
    format_hourly_cost,
    get_quick_deploy_profile,
)
from ..widgets.input_form import FormField, ToggleField
from .copy_enabled import CopyEnabledScreen


def _escape_markup(value: str) -> str:
    return (value or "").replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def _render_profile_summary(profile: QuickDeployProfile) -> str:
    return "\n".join(
        [
            f"[bold #7bf168]{_escape_markup(profile.display_name)}[/]",
            f"[dim]{_escape_markup(profile.summary)}[/dim]",
            "",
            f"[bold]Profile[/bold]  {_escape_markup(profile.profile_label)}",
            f"[bold]Quant[/bold]    {_escape_markup(profile.quant)}",
            f"[bold]GPU[/bold]      {_escape_markup(profile.gpu_type)} x{profile.gpu_count}",
            f"[bold]Cost[/bold]     {_escape_markup(format_hourly_cost(profile.approx_cost_per_hour_usd))}",
            f"[bold]Repo[/bold]     {_escape_markup(profile.repo_id)}",
            f"[bold]Default slug[/bold]  {_escape_markup(profile.instance_slug_hint)}",
        ]
    )


class QuickDeployScreen(CopyEnabledScreen):
    """Deploy one curated llama.cpp profile with minimal overrides."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("ctrl+d", "deploy", "Deploy", show=True),
    ]

    def __init__(self, profile_id: str) -> None:
        super().__init__()
        self.profile = get_quick_deploy_profile(profile_id)

    def compose(self) -> ComposeResult:
        yield Static(
            f"[bold #7bf168]Quick Deploy[/]  [dim]Curated llama.cpp profile for {_escape_markup(self.profile.display_name)}[/dim]",
            id="quick-deploy-title",
        )
        yield Static(
            "Review the curated profile, add optional naming overrides, then deploy.",
            id="quick-deploy-subtitle",
        )
        yield Static("")
        with Vertical(id="quick-deploy-layout"):
            with Vertical(id="quick-deploy-profile-card"):
                yield Static("[bold]Profile Summary[/bold]", id="quick-deploy-profile-title")
                yield Static(_render_profile_summary(self.profile), id="quick-deploy-profile-body")
            with Vertical(id="quick-deploy-form"):
                yield Button("Advanced options...", id="toggle-advanced-quick", variant="default")
                yield FormField(
                    "Instance name (optional)",
                    "quick-instance-name",
                    hint="Leave blank to use the curated default slug hint.",
                    classes="quick-advanced",
                )
                yield FormField(
                    "App name override (optional)",
                    "quick-app-name",
                    hint="Leave blank to keep the standard llamacpp-<instance> naming.",
                    classes="quick-advanced",
                )
                yield ToggleField(
                    "Warm up after deploy",
                    "quick-warmup",
                    default=True,
                    classes="quick-advanced",
                )
                yield ToggleField(
                    "Show debug logs",
                    "quick-debug-logs",
                    default=False,
                    classes="quick-advanced",
                )
                yield Static("", id="quick-deploy-feedback")
                yield Button("Deploy", id="quick-deploy-btn", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        for widget in self.query(".quick-advanced"):
            widget.add_class("hidden")
        self.query_one("#quick-deploy-btn", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "toggle-advanced-quick":
            for widget in self.query(".quick-advanced"):
                widget.toggle_class("hidden")
        elif event.button.id == "quick-deploy-btn":
            self._deploy()

    def action_deploy(self) -> None:
        self._deploy()

    def _deploy(self) -> None:
        config = build_quick_deploy_config(
            self.profile,
            instance_name=self.query_one("#quick-instance-name", Input).value,
            app_name=self.query_one("#quick-app-name", Input).value,
            do_warmup=self.query_one("#quick-warmup", Switch).value,
            show_debug_logs=self.query_one("#quick-debug-logs", Switch).value,
        )
        self.app.begin_deploy(config)  # type: ignore[attr-defined]

    def action_pop_screen(self) -> None:
        self.app.pop_screen()
