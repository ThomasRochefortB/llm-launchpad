"""Quick-deploy confirmation for provider-neutral inference plans."""

from __future__ import annotations

from ..widgets.vision_options import VisionOptions

from dataclasses import replace

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Button, Input, Select, Static, Switch

from ...core.compute_availability import display_gpu_type
from ...core.inference_options import (
    COST_SCENARIO_WORKDAY,
    continuous_monthly_compute_cost,
    evaluate_quote_cost,
)
from ...core.quick_deploy import (
    QuickDeployProfile,
    build_quick_deploy_config,
    format_context_length,
    get_quick_deploy_plan,
    get_quick_deploy_profile,
    instance_slug_for_plan,
    placement_matches_profile,
    quick_deploy_profile_for_plan,
    quick_deploy_model_label_parts,
    retune_quick_deploy_plan,
    resolve_quick_deploy_plans,
)
from ...core.diagnostics import log_exception
from ...core.provider_options import prime_provider_options
from ...protocol.enums import (
    BillingModel,
    CertificationState,
    ComputeProvider,
    QuoteAvailability,
    ServingObjective,
)
from ...protocol.models import DeploymentConfig, InferencePlan
from ..navigation import move_focus_with_arrows
from ..widgets.input_form import FormField, ToggleField
from ..widgets.fitted_footer import FittedFooter
from .copy_enabled import CopyEnabledScreen


def _render_profile_label(profile: QuickDeployProfile, *, accent: str = "") -> str:
    label, quant_suffix = quick_deploy_model_label_parts(profile)
    label_markup = escape(label)
    if accent:
        label_markup = f"[{accent}]{label_markup}[/]"
    if not quant_suffix:
        return label_markup
    return f"{label_markup} [dim]{escape(quant_suffix)}[/dim]"


# The widest label in the summary, so every value starts in the same column.
# These rows used to pad themselves: most landed on column 9, but Availability,
# Placement, Default slug, If left up and Spec decode were all longer than the
# padding assumed and pushed their values out of line.
_SUMMARY_LABEL_WIDTH = 13


def _summary_row(label: str, value: str) -> str:
    """Render one aligned `label  value` row of the profile summary."""
    return f"[bold]{label}[/bold]{' ' * (_SUMMARY_LABEL_WIDTH - len(label))}{value}"


def _render_decision_facts(profile: QuickDeployProfile, plan: InferencePlan) -> str:
    """Five facts for the deploy decision: model, compute, context, cost, availability.

    Rendered outside the scroll container so it stays on screen; the card
    below carries the technical evidence and monthly assumptions.
    """
    return (
        f"[bold]{escape(profile.display_name)}[/bold]"
        f"{f' ({escape(profile.quant)})' if profile.quant else ''} · "
        f"{escape(plan.quote.provider.display_name)} · "
        f"{escape(display_gpu_type(plan.quote.gpu_type))} x{plan.quote.gpu_count}\n"
        f"[dim]Context[/dim] {escape(format_context_length(profile.max_context_tokens))} · "
        f"[bold]{escape(_plan_hourly_cost(plan))}[/bold] · "
        f"{escape(_billing_label(plan.quote.billing_model))}\n"
        f"[dim]Availability[/dim] {escape(_availability_label(plan))}"
        f"{_idle_shutdown_fact(plan)}"
    )


def _idle_shutdown_fact(plan: InferencePlan) -> str:
    """What stops a rental that bills until deleted; nothing for scale-to-zero."""
    if plan.quote.billing_model == BillingModel.SCALE_TO_ZERO:
        return ""
    if plan.quote.provider not in (ComputeProvider.VAST, ComputeProvider.PRIME):
        return ""
    from ...core.config import ConfigStore
    from ...core.idle_watchdog import describe_idle_shutdown

    try:
        settings = ConfigStore().load()
    except Exception:
        return ""
    text = describe_idle_shutdown(settings.rental_idle_shutdown)
    if (
        settings.rental_idle_shutdown > 0
        and plan.quote.provider == ComputeProvider.PRIME
        and not settings.prime_self_terminate
    ):
        text += " while this computer is awake"
    style = "yellow" if settings.rental_idle_shutdown <= 0 else "dim"
    return f"\n[dim]Idle stop[/dim] [{style}]{escape(text)}[/{style}]"


def _render_profile_details(profile: QuickDeployProfile, plan: InferencePlan) -> str:
    """Technical evidence and cost assumptions behind the decision facts."""
    lines = [f"[dim]{escape(profile.summary)}[/dim]", ""]
    if profile.resource_tier_label and placement_matches_profile(profile, plan):
        tier_detail = profile.resource_tier_label
        if profile.profile_label and profile.profile_label != profile.resource_tier_label:
            tier_detail = f"{tier_detail} {profile.profile_label}"
        lines.append(_summary_row("Tier", escape(tier_detail)))
    lines.extend(
        [
            _summary_row("Backend", escape(plan.recipe.backend.display_name)),
        ]
    )
    if plan.quote.region:
        lines.append(_summary_row("Region", escape(plan.quote.region)))
    reference = (plan.quote.provider_reference or "").strip()
    if reference and _show_placement_reference(reference, plan.quote.gpu_type):
        lines.append(_summary_row("Placement", escape(reference)))
    required_memory = (
        plan.assessment.memory.total_gb
        if plan.assessment is not None
        else profile.required_vram_gb
    )
    if required_memory:
        lines.append(_summary_row("VRAM", f"{required_memory:.0f} GB required"))
    requirements = plan.recipe.serving_requirements
    if requirements is not None:
        lines.append(
            _summary_row("Optimize", escape(requirements.objective.display_name))
        )
    if plan.assessment is not None:
        single = max(
            (
                point.output_tokens_per_second or 0.0
                for point in plan.assessment.performance
                if point.concurrency == 1
            ),
            default=0.0,
        )
        aggregate = max(
            (
                point.aggregate_output_tokens_per_second or 0.0
                for point in plan.assessment.performance
            ),
            default=0.0,
        )
        evidence = (
            "certified"
            if plan.assessment.certification == CertificationState.CERTIFIED
            else "estimated; verified during deploy"
        )
        if single > 0:
            lines.append(_summary_row("Single", f"~{single:.0f} output tok/s"))
            lines.append(_summary_row("Batch", f"~{aggregate:.0f} aggregate tok/s"))
        lines.append(_summary_row("Evidence", escape(evidence)))
    if profile.speculative_decoding is not None:
        lines.append(
            _summary_row(
                "Spec decode",
                "Native MTP · up to "
                f"{profile.speculative_decoding.num_speculative_tokens} draft tokens",
            )
        )
    lines.extend(
        [
            _summary_row("If left up", escape(_plan_continuous_monthly_cost(plan))),
            _summary_row("Scenario", escape(_plan_scenario_cost(plan))),
            _summary_row("Model", escape(plan.recipe.model_id)),
            _summary_row("Default slug", escape(instance_slug_for_plan(profile, plan))),
            "",
            f"[dim]{escape(_plan_cost_basis_label(plan))}[/dim]",
            f"[dim]{escape(_plan_storage_note(plan))}[/dim]",
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
    """Capacity wording, kept separate from billing behavior.

    "Live now" means the provider verified capacity for this placement;
    "On demand" describes scale-to-zero billing, not verified capacity. An
    unverified scale-to-zero placement says both, so it never reads as
    guaranteed capacity.
    """
    if plan.quote.availability == QuoteAvailability.AVAILABLE:
        return "Live now"
    if plan.quote.availability == QuoteAvailability.UNAVAILABLE:
        return "Unavailable"
    if plan.quote.billing_model == BillingModel.SCALE_TO_ZERO:
        return "Starts on first request · scales to zero"
    return "Checked when deployment starts"


def _plan_hourly_cost(plan: InferencePlan) -> str:
    value = plan.quote.price_per_hour_usd
    if value is None:
        return "Unavailable"
    prefix = "~" if plan.quote.is_estimate else ""
    return f"{prefix}${value:.2f}/hr while billed"


def _plan_evaluation(plan: InferencePlan):
    """Canonical cost assumptions attached to the plan, recomputed on demand."""
    if plan.cost is not None:
        return plan.cost
    return evaluate_quote_cost(plan.quote, COST_SCENARIO_WORKDAY)


def _plan_scenario_cost(plan: InferencePlan) -> str:
    value = _plan_evaluation(plan).estimated_monthly_cost_usd
    if value is None:
        return "Unavailable"
    scenario = _plan_evaluation(plan).scenario
    return f"~${value:,.2f}/mo '{scenario.display_name}'"


def _plan_cost_basis_label(plan: InferencePlan) -> str:
    return _plan_evaluation(plan).basis_label()


def _plan_storage_note(plan: InferencePlan) -> str:
    if _plan_evaluation(plan).includes_storage:
        return "Quoted hourly total already includes disk rent."
    return "Storage is billed separately; unknown storage cost is not $0."


def _plan_continuous_monthly_cost(plan: InferencePlan) -> str:
    value = continuous_monthly_compute_cost(plan.quote)
    if value is None:
        return "Unavailable"
    return f"~${value:,.2f}/mo at 24/7"


def _deployment_decision_summary(profile: QuickDeployProfile, plan: InferencePlan) -> str:
    """Return the few facts needed immediately before committing a deploy."""
    return (
        f"[bold]{escape(profile.display_name)}[/bold] · "
        f"{escape(display_gpu_type(plan.quote.gpu_type))} x{plan.quote.gpu_count} · "
        f"[bold]{escape(_plan_hourly_cost(plan))}[/bold] · "
        f"{escape(_billing_label(plan.quote.billing_model))}"
    )


# Enter does not deploy, on purpose. That leaves the keyboard route stated only
# in the footer -- which the fulfillment dropdown covers while it is open, and
# which a phone terminal hides behind its on-screen keyboard entirely. So it is
# said here, and kept short enough to survive a narrow viewport: this line is
# one row high, so anything that does not fit is simply cut off.


def deploy_key_hint(*, narrow: bool) -> str:
    """Describe keyboard deployment for the current viewport."""
    if narrow:
        return "[dim]ctrl+d deploys · or tab to Deploy[/dim]"
    return (
        "[dim]Press ctrl+d to deploy, or tab to Deploy and press enter.[/dim]"
    )


def _fulfillment_option(plan: InferencePlan, *, recommended: bool = False) -> str:
    location = plan.quote.region or "provider-managed region"
    price = _plan_hourly_cost(plan)
    prefix = "Best available · " if recommended else ""
    return (
        f"{prefix}{plan.quote.provider.display_name} · "
        f"{plan.quote.gpu_count}x {display_gpu_type(plan.quote.gpu_type)} · {location} · {price}"
    )


def _dedupe_fulfillment_options(
    plans: tuple[InferencePlan, ...],
    keep: InferencePlan,
) -> tuple[InferencePlan, ...]:
    """Drop placements this list cannot tell apart.

    A provider can return several quotes for one shape -- Modal lists two B200
    placements -- and the line above renders provider, GPU count, region and
    price only, so they arrived as the same row twice with nothing to choose
    between them. The selected plan is never dropped.
    """

    seen: set[str] = set()
    unique: list[InferencePlan] = []
    for plan in plans:
        label = _fulfillment_option(plan)
        if label in seen and plan.quote.id != keep.quote.id:
            continue
        seen.add(label)
        unique.append(plan)
    return tuple(unique)


def _fulfillment_caution(plan: InferencePlan) -> str:
    """Warn about the plan the user is about to deploy, not the menu's contents.

    The fulfillment list mixes providers, so keying this off "any alternative is
    Vast" told someone deploying to Modal or Prime that their endpoint would be
    local and bill continuously.
    """
    if plan.quote.provider != ComputeProvider.VAST:
        return ""
    return (
        "[yellow]Vast.ai serves through an SSH endpoint on this computer only. "
        "The rental bills continuously; Stop destroys the instance and its disk. "
        "Hourly prices include disk; traffic costs extra.[/yellow]"
    )


class QuickDeployScreen(CopyEnabledScreen):
    """Deploy one curated inference plan with minimal overrides."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("ctrl+d", "deploy", "Deploy", show=True, priority=True),
        # Up and down move between the controls. Textual moves focus with tab,
        # but this form is mostly buttons and selects, and a closed Select
        # binds "down" to opening itself -- so arrowing down from the
        # fulfillment select opened the dropdown and then never left it, with
        # every further press moving inside the overlay. On a phone keyboard,
        # where tab is a modifier-bar item and the arrows are the obvious way
        # to move, that made the Deploy button unreachable.
        Binding("up", "focus_previous_control", "Previous", show=False, priority=True),
        Binding("down", "focus_next_control", "Next", show=False, priority=True),
    ]

    def __init__(
        self,
        profile_id: str | QuickDeployProfile | InferencePlan,
        *,
        alternative_plans: tuple[InferencePlan, ...] | None = None,
        profile: QuickDeployProfile | None = None,
    ) -> None:
        super().__init__()
        if isinstance(profile_id, InferencePlan):
            self.plan = profile_id
            # Prefer the profile the caller already resolved. Re-deriving it by
            # searching the global catalog fails whenever that catalog and the
            # plan came from different refreshes.
            self.profile = profile if profile is not None else quick_deploy_profile_for_plan(profile_id)
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
        self._alternative_plans = _dedupe_fulfillment_options(
            matching or (self.plan,), self.plan
        )
        self._plan_by_id = {
            plan.quote.id: plan
            for plan in self._alternative_plans
        }
        requirements = self.plan.recipe.serving_requirements
        self._objective = (
            requirements.objective
            if requirements is not None
            else ServingObjective.GENERAL_PURPOSE
        )

    def compose(self) -> ComposeResult:
        # Steps 1 and 2 are the model and placement pickers this screen is
        # reached from; naming the last one keeps the flow's count complete.
        yield Static(
            "[bold primary]Deploy[/]  "
            f"{_render_profile_label(self.profile)} "
            "[dim]· Step 3: Confirm and deploy[/dim]",
            id="quick-deploy-title",
        )
        yield Static(
            _render_decision_facts(self.profile, self.plan),
            id="quick-deploy-facts",
        )
        with VerticalScroll(id="quick-deploy-layout"):
            with Vertical(id="quick-deploy-profile-card"):
                yield Button(
                    "Technical details...",
                    id="toggle-quick-details",
                    variant="default",
                )
                yield Static(
                    _render_profile_details(self.profile, self.plan),
                    id="quick-deploy-profile-body",
                )
            with Vertical(id="quick-deploy-form"):
                yield Static("Runs on", classes="form-label")
                caution = Static(_fulfillment_caution(self.plan), id="quick-vast-note")
                caution.display = bool(_fulfillment_caution(self.plan))
                yield caution
                if len(self._alternative_plans) > 1:
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
                else:
                    yield Static(
                        _fulfillment_option(self.plan, recommended=True),
                        id="quick-fulfillment-single",
                    )
                yield Button("Advanced options...", id="toggle-advanced-quick", variant="default")
                if self.profile.speculative_decoding is not None:
                    # On by default and faster; a tuning knob, not a decision.
                    yield ToggleField(
                        "Use MTP speculative decoding",
                        "quick-speculative-decoding",
                        default=True,
                        classes="quick-advanced",
                    )
                    yield Static(
                        "[dim]Native MTP · drafts up to "
                        f"{self.profile.speculative_decoding.num_speculative_tokens} "
                        "tokens[/dim]",
                        id="quick-speculative-note",
                        classes="quick-advanced",
                    )
                if self.plan.recipe.serving_requirements is not None:
                    yield Static("Optimize for", classes="form-label quick-advanced")
                    yield Select(
                        options=[
                            ("General purpose (recommended)", ServingObjective.GENERAL_PURPOSE.value),
                            ("Interactive", ServingObjective.INTERACTIVE.value),
                            ("Throughput", ServingObjective.THROUGHPUT.value),
                            ("Benchmark", ServingObjective.BENCHMARK.value),
                        ],
                        value=self._objective.value,
                        allow_blank=False,
                        id="quick-objective",
                        classes="quick-advanced",
                    )
                # A guaranteed-fit plan cannot serve images, because image
                # working memory is not part of the placement it certifies.
                if self.plan.recipe.serving_requirements is None:
                    yield VisionOptions(classes="quick-advanced")
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
                if self.plan.recipe.serving_requirements is None:
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
            yield Static(
                _deployment_decision_summary(self.profile, self.plan),
                id="quick-deploy-decision-summary",
            )
            # Enter is deliberately not bound to Deploy (see on_mount), which
            # leaves the keyboard route stated only in the footer -- and the
            # fulfillment dropdown paints over the footer while it is open,
            # exactly when someone is looking for how to continue. Say it
            # here instead, next to the button it refers to.
            yield Static(self._deploy_key_hint(), id="quick-deploy-feedback")
            yield Button("Deploy", id="quick-deploy-btn", variant="primary")
        yield FittedFooter()

    def on_mount(self) -> None:
        for widget in self.query(".quick-advanced"):
            widget.add_class("hidden")
        self.query_one("#quick-deploy-profile-body", Static).display = False
        self._sync_prime_option_visibility()
        # This screen is reached by pressing enter, and Deploy spends money the
        # moment it fires, so a second enter must not be able to rent a GPU.
        # Start on the first interactive control; ctrl+d and tab still deploy.
        # Deferred past mount: focusing during on_mount races the footer and
        # surrounding layout, which can steal it back before first paint.
        self.call_after_refresh(self._focus_initial_control)

    def _focus_initial_control(self) -> None:
        """Focus the first control after mount settles (see on_mount)."""
        if self.focused is not None and getattr(self.focused, "id", None) not in {
            None, "quick-deploy-layout",
        }:
            return
        target = next(iter(self.query("#toggle-quick-details")), None) or next(
            iter(self.query("#quick-fulfillment")), None
        ) or next(iter(self.query("#toggle-advanced-quick")), None)
        if target is None:
            target = self.query_one("#quick-deploy-btn", Button)
        # Not scroll_visible: on a short terminal, scrolling the form control
        # into view pushes the profile summary the user came here to read off
        # the top of the screen.
        target.focus(scroll_visible=False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "toggle-quick-details":
            body = self.query_one("#quick-deploy-profile-body", Static)
            body.display = not body.display
            toggle = self.query_one("#toggle-quick-details", Button)
            toggle.label = "Hide technical details" if body.display else "Technical details..."
            return
        if event.button.id == "toggle-advanced-quick":
            for widget in self.query(".quick-advanced"):
                widget.toggle_class("hidden")
            self._sync_prime_option_visibility()
        elif event.button.id == "quick-deploy-btn":
            self._deploy()

    def _refresh_plan_widgets(self) -> None:
        """Keep facts, details, sticky summary and caution on the same plan."""
        self.query_one("#quick-deploy-facts", Static).update(
            _render_decision_facts(self.profile, self.plan)
        )
        self.query_one("#quick-deploy-profile-body", Static).update(
            _render_profile_details(self.profile, self.plan)
        )
        self.query_one("#quick-deploy-decision-summary", Static).update(
            _deployment_decision_summary(self.profile, self.plan)
        )

    def on_select_changed(self, event: Select.Changed) -> None:
        if not isinstance(event.value, str):
            return
        if event.select.id == "quick-objective":
            try:
                objective = ServingObjective(event.value)
            except ValueError:
                return
            self._objective = objective
            selected_quote_id = self.plan.quote.id
            self._alternative_plans = tuple(
                retune_quick_deploy_plan(self.profile, plan, objective)
                for plan in self._alternative_plans
            )
            self._plan_by_id = {
                plan.quote.id: plan for plan in self._alternative_plans
            }
            self.plan = self._plan_by_id.get(
                selected_quote_id,
                self._alternative_plans[0],
            )
            self._refresh_plan_widgets()
            try:
                fulfillment = self.query_one("#quick-fulfillment", Select)
            except Exception:
                fulfillment = None
            if fulfillment is not None:
                fulfillment.set_options(
                    [
                        (
                            _fulfillment_option(plan, recommended=index == 0),
                            plan.quote.id,
                        )
                        for index, plan in enumerate(self._alternative_plans)
                    ]
                )
                fulfillment.value = self.plan.quote.id
            self._sync_fulfillment_caution()
            return
        if event.select.id != "quick-fulfillment":
            return
        plan = self._plan_by_id.get(event.value)
        if plan is None:
            return
        self.plan = plan
        self._refresh_plan_widgets()
        self._sync_fulfillment_caution()
        self._sync_prime_option_visibility()

    def _sync_fulfillment_caution(self) -> None:
        """Keep the billing warning matched to the selected provider."""
        caution = _fulfillment_caution(self.plan)
        note = self.query_one("#quick-vast-note", Static)
        note.update(caution)
        note.display = bool(caution)

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

    def _deploy_key_hint(self) -> str:
        return deploy_key_hint(
            narrow=self.viewport_profile.narrow,
        )

    def refresh_keyboard_hints(self) -> None:
        """Re-render the keyboard hint for the current viewport."""

        for hint in self.query("#quick-deploy-feedback"):
            hint.update(self._deploy_key_hint())

    def viewport_profile_changed(self, profile, previous) -> None:  # type: ignore[no-untyped-def]
        """Re-fit the hint when the terminal is resized."""

        super().viewport_profile_changed(profile, previous)
        self.refresh_keyboard_hints()

    def action_focus_next_control(self) -> None:
        move_focus_with_arrows(self, 1)

    def action_focus_previous_control(self) -> None:
        move_focus_with_arrows(self, -1)

    def action_deploy(self) -> None:
        self._deploy()

    def _deploy(self) -> None:
        # No toggle means the catalog found nothing to offer, which is not the
        # same as the reader declining. Only an unticked switch is a decline;
        # anything else leaves preflight free to resolve MTP from the model.
        enable_speculative_decoding = (
            self.query_one("#quick-speculative-decoding", Switch).value
            if self.profile.speculative_decoding is not None
            else True
        )
        instance_name = self.query_one("#quick-instance-name", Input).value
        app_name = self.query_one("#quick-app-name", Input).value
        show_debug_logs = self.query_one("#quick-debug-logs", Switch).value

        def _config_for_plan(plan: InferencePlan) -> DeploymentConfig:
            candidate = build_quick_deploy_config(
                self.profile,
                plan=plan,
                instance_name=instance_name,
                app_name=app_name,
                do_warmup=(
                    True
                    if plan.recipe.serving_requirements is not None
                    else self.query_one("#quick-warmup", Switch).value
                ),
                show_debug_logs=show_debug_logs,
                enable_speculative_decoding=enable_speculative_decoding,
            )
            if candidate.provider == ComputeProvider.PRIME:
                options = prime_provider_options(candidate)
                candidate.provider_options = replace(
                    options,
                    disk_id=(
                        self.query_one("#quick-prime-disk-id", Input).value.strip()
                        or options.disk_id
                    ),
                    allow_insecure_http=self.query_one(
                        "#quick-prime-insecure-http", Switch
                    ).value,
                    keep_failed_resource=self.query_one(
                        "#quick-prime-keep-failed", Switch
                    ).value,
                    auto_disk=self.query_one("#quick-prime-auto-disk", Switch).value,
                )
            for options in self.query(VisionOptions):
                options.apply(candidate)
            return candidate

        config = _config_for_plan(self.plan)
        approved_price = self.plan.quote.price_per_hour_usd
        fallback_plans = [
            plan
            for plan in self._alternative_plans
            if plan.quote.id != self.plan.quote.id
            and plan.quote.price_per_hour_usd is not None
            and (
                approved_price is None
                or plan.quote.price_per_hour_usd <= approved_price
            )
            and (plan.assessment is None or plan.assessment.fits)
        ]
        config.fallback_configs = tuple(_config_for_plan(plan) for plan in fallback_plans)
        self._remember_launch()
        self.app.begin_deploy(config)  # type: ignore[attr-defined]

    def _remember_launch(self) -> None:
        """Offer this choice on the home screen next time; never blocks a deploy."""
        from ...core.last_launch import LastLaunch, save_last_launch
        from ...core.quick_deploy import quick_deploy_model_key

        quote = self.plan.quote
        try:
            save_last_launch(
                LastLaunch(
                    model_id=quick_deploy_model_key(self.profile),
                    display_name=self.profile.display_name,
                    provider=quote.provider.value,
                    gpu_type=quote.gpu_type,
                    gpu_count=quote.gpu_count,
                    price_per_hour_usd=quote.price_per_hour_usd,
                )
            )
        except Exception:
            log_exception("Failed to remember the last launch")

    def action_pop_screen(self) -> None:
        self.app.pop_screen()
