"""Quick-deploy confirmation for provider-neutral inference plans."""

from __future__ import annotations

from dataclasses import replace

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Button, Footer, Input, Select, Static, Switch

from ...core.compute_availability import display_gpu_type
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
from ...protocol.enums import BillingModel, ComputeProvider, QuoteAvailability
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
            f"[bold]GPU[/bold]      {_escape_markup(display_gpu_type(plan.quote.gpu_type))} x{plan.quote.gpu_count}",
        ]
    )
    if plan.quote.region:
        lines.append(f"[bold]Region[/bold]   {_escape_markup(plan.quote.region)}")
    lines.append(
        f"[bold]Availability[/bold] {_escape_markup(_availability_label(plan))}"
    )
    reference = (plan.quote.provider_reference or "").strip()
    if reference and _show_placement_reference(reference, plan.quote.gpu_type):
        lines.append(
            f"[bold]Placement[/bold] {_escape_markup(reference)}"
        )
    if profile.quant:
        lines.insert(-1, f"[bold]Quant[/bold]    {_escape_markup(profile.quant)}")
    if profile.required_vram_gb:
        lines.append(f"[bold]VRAM[/bold]     {profile.required_vram_gb:.0f} GB required")
    if profile.speculative_decoding is not None:
        lines.append(
            "[bold]Spec decode[/bold] Native MTP · up to "
            f"{profile.speculative_decoding.num_speculative_tokens} draft tokens"
        )
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
            "[dim]Availability is revalidated when deployment starts.[/dim]",
        ]
    )
    return "\n".join(lines)


def _show_placement_reference(reference: str, gpu_type: str) -> bool:
    """Hide provider-internal IDs and GPU names already shown on the GPU line."""

    cleaned = reference.strip()
    if not cleaned:
        return False
    lowered = cleaned.casefold()
    if lowered in {gpu_type.strip().casefold(), display_gpu_type(gpu_type).casefold()}:
        return False
    if all(character in "0123456789abcdef" for character in lowered):
        return False
    return len(cleaned) > 8


def _billing_label(value: BillingModel) -> str:
    if value == BillingModel.SCALE_TO_ZERO:
        return "Scale to zero"
    return "Provisioned resource"


def _availability_label(plan: InferencePlan) -> str:
    if plan.quote.availability == QuoteAvailability.AVAILABLE:
        return "Live now"
    if plan.quote.availability == QuoteAvailability.UNAVAILABLE:
        return "Unavailable"
    if plan.quote.billing_model == BillingModel.SCALE_TO_ZERO:
        return "On demand"
    return "Provider reported"


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


def _fulfillment_option(plan: InferencePlan, *, recommended: bool = False) -> str:
    location = plan.quote.region or "provider-managed region"
    price = _plan_hourly_cost(plan)
    prefix = "Best available · " if recommended else ""
    return (
        f"{prefix}{plan.quote.provider.display_name} · "
        f"{plan.quote.gpu_count}x {display_gpu_type(plan.quote.gpu_type)} · {location} · {price}"
    )


class QuickDeployScreen(CopyEnabledScreen):
    """Deploy one curated inference plan with minimal overrides."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("ctrl+d", "deploy", "Deploy", show=True),
    ]

    def __init__(
        self,
        profile_id: str | QuickDeployProfile | InferencePlan,
        *,
        alternative_plans: tuple[InferencePlan, ...] | None = None,
    ) -> None:
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
        alternatives = alternative_plans or (self.plan,)
        matching = tuple(
            plan
            for plan in alternatives
            if plan.recipe.id == self.plan.recipe.id
        )
        self._alternative_plans = matching or (self.plan,)
        self._plan_by_id = {
            plan.quote.id: plan
            for plan in self._alternative_plans
        }

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
        with VerticalScroll(id="quick-deploy-layout"):
            with Vertical(id="quick-deploy-profile-card"):
                yield Static("[bold]Profile Summary[/bold]", id="quick-deploy-profile-title")
                yield Static(
                    _render_profile_summary(self.profile, self.plan),
                    id="quick-deploy-profile-body",
                )
            with Vertical(id="quick-deploy-form"):
                if len(self._alternative_plans) > 1:
                    yield Static("Fulfillment", classes="form-label")
                    yield Select(
                        options=[
                            (
                                _fulfillment_option(plan, recommended=index == 0),
                                plan.quote.id,
                            )
                            for index, plan in enumerate(self._alternative_plans)
                        ],
                        value=self.plan.quote.id,
                        allow_blank=False,
                        id="quick-fulfillment",
                    )
                    yield Static(
                        "[dim]The provider is revealed here because it determines billing, region, and credentials.[/dim]",
                        id="quick-fulfillment-note",
                    )
                if self.profile.speculative_decoding is not None:
                    yield ToggleField(
                        "Use MTP speculative decoding",
                        "quick-speculative-decoding",
                        default=True,
                    )
                    yield Static(
                        "[dim]Native MTP · drafts up to 3 tokens[/dim]",
                        id="quick-speculative-note",
                    )
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
                if any(
                    plan.quote.provider == ComputeProvider.PRIME
                    for plan in self._alternative_plans
                ):
                    yield Static(
                        "[dim]Prime Tunnel provides the endpoint over HTTPS by default.[/dim]",
                        classes="quick-advanced quick-prime-only",
                    )
                    yield ToggleField(
                        "Attach persistent cache disk",
                        "quick-prime-auto-disk",
                        default=True,
                        classes="quick-advanced quick-prime-only",
                    )
                    yield FormField(
                        "Prime disk ID (optional)",
                        "quick-prime-disk-id",
                        hint="Leave blank to auto-attach a persistent cache disk",
                        classes="quick-advanced quick-prime-only",
                    )
                    yield ToggleField(
                        "Use direct HTTP fallback (insecure)",
                        "quick-prime-insecure-http",
                        default=False,
                        classes="quick-advanced quick-prime-only",
                    )
                    yield ToggleField(
                        "Keep failed Prime pod (billing may continue)",
                        "quick-prime-keep-failed",
                        default=False,
                        classes="quick-advanced quick-prime-only",
                    )
        with Vertical(id="quick-deploy-actions"):
            yield Static("", id="quick-deploy-feedback")
            yield Button("Deploy", id="quick-deploy-btn", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        for widget in self.query(".quick-advanced"):
            widget.add_class("hidden")
        self._sync_prime_option_visibility()
        self.query_one("#quick-deploy-btn", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "toggle-advanced-quick":
            for widget in self.query(".quick-advanced"):
                widget.toggle_class("hidden")
            self._sync_prime_option_visibility()
        elif event.button.id == "quick-deploy-btn":
            self._deploy()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "quick-fulfillment" or not isinstance(event.value, str):
            return
        plan = self._plan_by_id.get(event.value)
        if plan is None:
            return
        self.plan = plan
        self.query_one("#quick-deploy-profile-body", Static).update(
            _render_profile_summary(self.profile, self.plan)
        )
        self._sync_prime_option_visibility()

    def _sync_prime_option_visibility(self) -> None:
        advanced_visible = any(
            not widget.has_class("hidden")
            and not widget.has_class("quick-prime-only")
            for widget in self.query(".quick-advanced")
        )
        for widget in self.query(".quick-prime-only"):
            widget.set_class(
                self.plan.quote.provider != ComputeProvider.PRIME
                or not advanced_visible,
                "hidden",
            )

    def action_deploy(self) -> None:
        self._deploy()

    def _deploy(self) -> None:
        enable_speculative_decoding = (
            self.query_one("#quick-speculative-decoding", Switch).value
            if self.profile.speculative_decoding is not None
            else False
        )
        config = build_quick_deploy_config(
            self.profile,
            plan=self.plan,
            instance_name=self.query_one("#quick-instance-name", Input).value,
            app_name=self.query_one("#quick-app-name", Input).value,
            do_warmup=self.query_one("#quick-warmup", Switch).value,
            show_debug_logs=self.query_one("#quick-debug-logs", Switch).value,
            enable_speculative_decoding=enable_speculative_decoding,
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
                auto_disk=self.query_one("#quick-prime-auto-disk", Switch).value,
            )
        self.app.begin_deploy(config)  # type: ignore[attr-defined]

    def action_pop_screen(self) -> None:
        self.app.pop_screen()
