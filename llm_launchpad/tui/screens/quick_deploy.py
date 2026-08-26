"""Quick-deploy confirmation for provider-neutral inference plans."""

from __future__ import annotations

from dataclasses import replace

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Button, Footer, Input, Static, Switch

from ...core.quick_deploy import (
    QuickDeployProfile,
    build_quick_deploy_config,
    format_context_length,
    get_quick_deploy_plan,
    get_quick_deploy_profile,
    quick_deploy_profile_for_plan,
    quick_deploy_model_label_parts,
    resolve_quick_deploy_plans,
)
from ...core.provider_options import prime_provider_options
from ...protocol.enums import BillingModel, ComputeProvider
from ...protocol.models import InferencePlan
from ..widgets.input_form import FormField, ToggleField
from .copy_enabled import CopyEnabledScreen


def _escape_markup(value: str) -> str:
    return (value or "").replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def _render_profile_label(profile: QuickDeployProfile, *, accent: str = "") -> str:
    label, quant_suffix = quick_deploy_model_label_parts(profile)
    label_markup = _escape_markup(label)
    if accent:
        label_markup = f"[{accent}]{label_markup}[/]"
    if not quant_suffix:
        return label_markup
    return f"{label_markup} [dim]{_escape_markup(quant_suffix)}[/dim]"


def _render_profile_summary(profile: QuickDeployProfile, plan: InferencePlan) -> str:
    lines = [
        _render_profile_label(profile, accent="bold #7bf168"),
        f"[dim]{_escape_markup(profile.summary)}[/dim]",
        "",
    ]
    if profile.resource_tier_label:
        tier_detail = profile.resource_tier_label
        if profile.profile_label and profile.profile_label != profile.resource_tier_label:
            tier_detail = f"{tier_detail} {profile.profile_label}"
        lines.append(f"[bold]Tier[/bold]     {_escape_markup(tier_detail)}")
    lines.extend(
        [
            f"[bold]Provider[/bold] {_escape_markup(plan.quote.provider.display_name)}",
            f"[bold]Billing[/bold]  {_escape_markup(_billing_label(plan.quote.billing_model))}",
            f"[bold]Backend[/bold]  {_escape_markup(plan.recipe.backend.display_name)}",
            f"[bold]GPU[/bold]      {_escape_markup(plan.quote.gpu_type)} x{plan.quote.gpu_count}",
        ]
    )
    if profile.quant:
        lines.insert(-1, f"[bold]Quant[/bold]    {_escape_markup(profile.quant)}")
    if profile.required_vram_gb:
        lines.append(f"[bold]VRAM[/bold]     {profile.required_vram_gb:.0f} GB required")
    lines.extend(
        [
            f"[bold]Max ctx[/bold]  {_escape_markup(format_context_length(profile.max_context_tokens))}",
            f"[bold]Hourly[/bold]   {_escape_markup(_plan_hourly_cost(plan))}",
            f"[bold]Monthly[/bold]  {_escape_markup(_plan_monthly_cost(plan))}",
            f"[bold]Model[/bold]    {_escape_markup(plan.recipe.model_id)}",
            f"[bold]Default slug[/bold]  {_escape_markup(profile.instance_slug_hint)}",
            "",
            (
                "[dim]Monthly estimate assumes an 8-hour daily serving window at "
                "25% utilization. Provisioned resources bill the full window.[/dim]"
            ),
        ]
    )
    return "\n".join(lines)


def _billing_label(value: BillingModel) -> str:
    if value == BillingModel.SCALE_TO_ZERO:
        return "Scale to zero"
    return "Provisioned resource"


def _plan_hourly_cost(plan: InferencePlan) -> str:
    value = plan.quote.price_per_hour_usd
    if value is None:
        return "Unavailable"
    prefix = "~" if plan.quote.is_estimate else ""
    return f"{prefix}${value:.2f}/hr"


def _plan_monthly_cost(plan: InferencePlan) -> str:
    value = plan.estimated_monthly_cost_usd
    if value is None:
        return "Unavailable"
    return f"~${value:,.2f}/mo"


class QuickDeployScreen(CopyEnabledScreen):
    """Deploy one curated inference plan with minimal overrides."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("ctrl+d", "deploy", "Deploy", show=True),
    ]

    def __init__(self, profile_id: str | QuickDeployProfile | InferencePlan) -> None:
        super().__init__()
        if isinstance(profile_id, InferencePlan):
            self.plan = profile_id
            self.profile = quick_deploy_profile_for_plan(profile_id)
        elif isinstance(profile_id, QuickDeployProfile):
            self.profile = profile_id
            self.plan = resolve_quick_deploy_plans((profile_id,))[0]
        else:
            try:
                self.plan = get_quick_deploy_plan(profile_id)
                self.profile = quick_deploy_profile_for_plan(self.plan)
            except KeyError:
                self.profile = get_quick_deploy_profile(profile_id)
                self.plan = get_quick_deploy_plan(self.profile.id)

    def compose(self) -> ComposeResult:
        yield Static(
            "[bold #7bf168]Quick Deploy[/]  "
            f"[dim]Curated inference plan for[/dim] {_render_profile_label(self.profile)}",
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
                yield Static(
                    _render_profile_summary(self.profile, self.plan),
                    id="quick-deploy-profile-body",
                )
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
                    hint="Leave blank to use the selected provider's standard naming.",
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
                if self.plan.quote.provider == ComputeProvider.PRIME:
                    yield Static(
                        "[dim]Prime Tunnel provides the endpoint over HTTPS by default.[/dim]"
                    )
                    yield FormField(
                        "Existing Prime disk ID (optional)",
                        "quick-prime-disk-id",
                        classes="quick-advanced",
                    )
                    yield ToggleField(
                        "Use direct HTTP fallback (insecure)",
                        "quick-prime-insecure-http",
                        default=False,
                        classes="quick-advanced",
                    )
                    yield ToggleField(
                        "Keep failed Prime pod (billing may continue)",
                        "quick-prime-keep-failed",
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
            plan=self.plan,
            instance_name=self.query_one("#quick-instance-name", Input).value,
            app_name=self.query_one("#quick-app-name", Input).value,
            do_warmup=self.query_one("#quick-warmup", Switch).value,
            show_debug_logs=self.query_one("#quick-debug-logs", Switch).value,
        )
        if config.provider == ComputeProvider.PRIME:
            options = prime_provider_options(config)
            allow_insecure_http = self.query_one(
                "#quick-prime-insecure-http", Switch
            ).value
            config.provider_options = replace(
                options,
                disk_id=(
                    self.query_one("#quick-prime-disk-id", Input).value.strip()
                    or options.disk_id
                ),
                allow_insecure_http=allow_insecure_http,
                keep_failed_resource=self.query_one(
                    "#quick-prime-keep-failed", Switch
                ).value,
            )
        self.app.begin_deploy(config)  # type: ignore[attr-defined]

    def action_pop_screen(self) -> None:
        self.app.pop_screen()
