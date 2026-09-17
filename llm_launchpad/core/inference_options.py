"""Provider-agnostic inference recipe resolution and pricing quotes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol
from collections.abc import Iterable, Sequence

from ..protocol.enums import BackendType, BillingModel, ComputeProvider, QuoteAvailability
from ..protocol.models import (
    ComputeOffer,
    CostScenario,
    InferencePlan,
    InferenceRecipe,
    ModalProviderOptions,
    PrimeProviderOptions,
    ProviderCapabilities,
    ProviderQuote,
    WorkloadProfile,
)
from .prime_backend import (
    PrimeBackend,
    prime_stock_availability,
    is_compatible_prime_offer,
    preferred_prime_offer_image,
    prime_offer_gpu_memory_gb,
    prime_offer_matches_location,
)


def recommended_vllm_tool_call_parser(model_name: str | None) -> str | None:
    """Return a conservative tool parser recommendation for known Qwen models.

    Standard Qwen 2.5/QwQ/Qwen3 chat templates emit Hermes-style JSON inside
    ``<tool_call>`` tags. Qwen3-Coder uses vLLM's distinct XML parser. Models
    with separate embedding, reranking, or vision runtimes are intentionally
    left alone instead of guessing.
    """

    model_id = (model_name or "").strip().rsplit("/", 1)[-1].casefold()
    if not model_id:
        return None
    if model_id.startswith("qwen3-coder-"):
        return "qwen3_xml"
    if any(marker in model_id for marker in ("embedding", "reranker", "-vl")):
        return None
    if model_id.startswith(("qwen3-", "qwen2.5-", "qwq-")):
        return "hermes"
    return None


class InferenceProviderAdapter(Protocol):
    """Translate provider-neutral recipes into provider quotes."""

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return provider features without performing network I/O."""

    def quote(
        self,
        recipe: InferenceRecipe,
        workload: WorkloadProfile,
    ) -> list[ProviderQuote]:
        """Return compatible quotes for one recipe."""


@dataclass(frozen=True)
class ModalCatalogOption:
    """One bundled Modal fulfillment choice for a neutral recipe."""

    id: str
    recipe_id: str
    gpu_type: str
    gpu_count: int
    price_per_hour_usd: float | None
    gpu_memory_gb: float | None = None
    estimated_output_tokens_per_second: float | None = None


class ModalInferenceAdapter:
    """Expose bundled or live Modal GPU shapes through the quote contract."""

    def __init__(self, options: Iterable[ModalCatalogOption]) -> None:
        self._options = tuple(options)

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider=ComputeProvider.MODAL,
            supported_backends=frozenset({BackendType.LLAMACPP, BackendType.VLLM}),
            billing_model=BillingModel.SCALE_TO_ZERO,
        )

    def quote(
        self,
        recipe: InferenceRecipe,
        workload: WorkloadProfile,
    ) -> list[ProviderQuote]:
        del workload
        if not self.capabilities.supports_backend(recipe.backend):
            return []
        return [
            ProviderQuote(
                id=option.id,
                recipe_id=recipe.id,
                provider=ComputeProvider.MODAL,
                provider_reference=option.id,
                gpu_type=option.gpu_type,
                gpu_count=option.gpu_count,
                price_per_hour_usd=option.price_per_hour_usd,
                billing_model=self.capabilities.billing_model,
                gpu_memory_gb=option.gpu_memory_gb,
                availability=QuoteAvailability.UNKNOWN,
                is_estimate=True,
                estimated_output_tokens_per_second=(
                    option.estimated_output_tokens_per_second
                ),
                configuration_id=option.id,
                provider_options=ModalProviderOptions(),
            )
            for option in self._options
            if option.recipe_id == recipe.id
        ]


class PrimeInferenceAdapter:
    """Resolve live Prime marketplace offers for compatible inference recipes."""

    def __init__(
        self,
        backend: PrimeBackend | None = None,
        provider_options: PrimeProviderOptions | None = None,
    ) -> None:
        self.backend = backend or PrimeBackend()
        self.provider_options = provider_options or PrimeProviderOptions()
        self._cached_offers: tuple[ComputeOffer, ...] | None = None

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider=ComputeProvider.PRIME,
            supported_backends=frozenset({BackendType.LLAMACPP, BackendType.VLLM}),
            billing_model=BillingModel.PROVISIONED,
            live_availability=True,
            supports_regions=True,
            supports_spot=True,
            supports_secure_cloud=True,
        )

    def quote(
        self,
        recipe: InferenceRecipe,
        workload: WorkloadProfile,
    ) -> list[ProviderQuote]:
        del workload
        if not self.capabilities.supports_backend(recipe.backend):
            return []

        quotes: list[ProviderQuote] = []
        if self._cached_offers is None:
            self._cached_offers = tuple(self.backend.list_offers())
        required_image = preferred_prime_offer_image(recipe.backend)
        for offer in self._cached_offers:
            if not prime_offer_matches_location(offer, self.provider_options.region):
                continue
            if not is_compatible_prime_offer(
                offer,
                recipe.required_vram_gb,
                required_image=required_image,
            ):
                continue
            availability = prime_stock_availability(offer.stock_status)
            region = offer.country or offer.region or offer.data_center
            # Prime sometimes reports aggregate node memory. Planner placement
            # math is per-device, so quotes always carry the per-GPU size.
            per_gpu_memory_gb = prime_offer_gpu_memory_gb(offer)
            quotes.append(
                ProviderQuote(
                    id=f"prime:{recipe.id}:{offer.id}",
                    recipe_id=recipe.id,
                    provider=ComputeProvider.PRIME,
                    provider_reference=offer.id,
                    gpu_type=offer.gpu_type,
                    gpu_count=offer.gpu_count,
                    price_per_hour_usd=offer.price_per_hour,
                    billing_model=self.capabilities.billing_model,
                    gpu_memory_gb=per_gpu_memory_gb,
                    availability=availability,
                    region=region,
                    security=offer.security,
                    is_estimate=False,
                    provider_options=replace(
                        self.provider_options,
                        offer_id=offer.id,
                    ),
                )
            )
        return quotes


def resolve_inference_plans(
    recipes: Iterable[InferenceRecipe],
    adapters: Sequence[InferenceProviderAdapter],
    workload: WorkloadProfile | None = None,
) -> list[InferencePlan]:
    """Resolve and rank provider quotes for a collection of recipes."""

    workload = workload or WorkloadProfile()
    plans: list[InferencePlan] = []
    for recipe in recipes:
        for adapter in adapters:
            capabilities = adapter.capabilities
            if not capabilities.supports_backend(recipe.backend):
                continue
            for quote in adapter.quote(recipe, workload):
                if quote.recipe_id != recipe.id:
                    raise ValueError(
                        f"Provider quote {quote.id!r} targets {quote.recipe_id!r}, "
                        f"expected {recipe.id!r}."
                    )
                plans.append(_plan_from_quote(recipe, quote, workload))

    plans.sort(key=_plan_monthly_cost_sort_key)
    recommended_recipes: set[str] = set()
    for index, plan in enumerate(plans):
        if plan.recipe.id in recommended_recipes:
            continue
        if (
            plan.estimated_monthly_cost_usd is None
            or plan.quote.availability == QuoteAvailability.UNAVAILABLE
        ):
            continue
        plans[index] = replace(
            plan,
            recommendation_reason="Lowest estimated cost for this workload",
        )
        recommended_recipes.add(plan.recipe.id)
    return plans


HOURS_PER_MONTH_CONTINUOUS: float = 24.0 * 30.0

COST_SCENARIO_CONTINUOUS = CostScenario(
    id="continuous",
    display_name="Always on",
    provisioned_hours_per_day=24.0,
    active_hours_per_day=24.0,
    sessions_per_day=1.0,
    idle_timeout_seconds=1800.0,
    description="Running 24/7 with no shutdown.",
)

COST_SCENARIO_WORKDAY = CostScenario(
    id="workday-8h",
    display_name="Workday 8h",
    provisioned_hours_per_day=8.0,
    active_hours_per_day=2.0,
    sessions_per_day=4.0,
    idle_timeout_seconds=1800.0,
    description="Up 8h/day, 2h active across 4 sessions, 30min idle timeout.",
)

COST_SCENARIO_SPARSE = CostScenario(
    id="sparse",
    display_name="Sparse",
    provisioned_hours_per_day=8.0,
    active_hours_per_day=0.5,
    sessions_per_day=10.0,
    idle_timeout_seconds=1800.0,
    description="Up 8h/day, 0.5h active spread over 10 sessions.",
)

COST_SCENARIO_CLUSTERED = CostScenario(
    id="clustered",
    display_name="Clustered",
    provisioned_hours_per_day=8.0,
    active_hours_per_day=0.5,
    sessions_per_day=2.0,
    idle_timeout_seconds=1800.0,
    description="Up 8h/day, 0.5h active clustered into 2 sessions.",
)

COST_SCENARIOS: tuple[CostScenario, ...] = (
    COST_SCENARIO_CONTINUOUS,
    COST_SCENARIO_WORKDAY,
    COST_SCENARIO_SPARSE,
    COST_SCENARIO_CLUSTERED,
)


def get_cost_scenario(scenario_id: str | None) -> CostScenario:
    """Return a named scenario, defaulting to the workday schedule."""
    if not scenario_id:
        return COST_SCENARIO_WORKDAY
    for scenario in COST_SCENARIOS:
        if scenario.id == scenario_id:
            return scenario
    raise ValueError(f"Unknown cost scenario: {scenario_id}")


def workload_basis_label(workload: WorkloadProfile | None = None) -> str:
    """State the assumptions a monthly estimate rests on, in one sentence.

    Monthly figures are normalized through a workload profile, not wall-clock,
    so a provisioned rental's "/mo" is a fraction of what leaving it running
    costs. Printed unqualified next to "the rental bills continuously" the two
    contradict each other, so every surface that shows the figure states this.
    """

    profile = workload or WorkloadProfile()
    hours = profile.paid_hours_per_day
    noun = "hour" if hours == 1 else "hours"
    return (
        f"Monthly estimates assume {hours:g} paid {noun} a day, scaled by "
        f"{profile.utilization:.0%} utilization on scale-to-zero providers. "
        f"Hourly and 24/7 figures are exact; monthly usage is a scenario."
    )


def scenario_basis_label(scenario: CostScenario | None = None) -> str:
    """State an explicit cost scenario's assumptions in one sentence."""
    resolved = scenario or COST_SCENARIO_WORKDAY
    return (
        f"Cost scenario '{resolved.display_name}': {resolved.describe()} "
        f"Provisioned bills {resolved.provisioned_hours_per_day:g}h/day; "
        f"scale-to-zero bills active plus idle timeout."
    )


def cost_sort_basis_label(scenario: CostScenario | None = None) -> str:
    """Name the cost basis placement ordering uses, so it matches the display."""
    resolved = scenario or COST_SCENARIO_WORKDAY
    return f"Sorted by '{resolved.display_name}' scenario cost ({resolved.describe()})"


def continuous_monthly_compute_cost(quote: ProviderQuote) -> float | None:
    """Return the monthly cost of never stopping a provisioned resource."""

    if quote.price_per_hour_usd is None:
        return None
    return quote.price_per_hour_usd * HOURS_PER_MONTH_CONTINUOUS


def estimate_modal_billed_hours_per_day(
    active_hours_per_day: float,
    sessions_per_day: float,
    idle_timeout_seconds: float,
    *,
    window_hours_per_day: float = 24.0,
) -> float:
    """Billed hours for a scale-to-zero container, including warm idle time.

    Each session keeps its container warm for the idle timeout after its last
    request. Sparse sessions each pay the timeout; clustered sessions share
    it, so identical active time bills differently. The result never exceeds
    the usage window and never drops below active time.
    """
    active = min(max(0.0, active_hours_per_day), 24.0)
    sessions = max(0.0, sessions_per_day)
    idle_hours = max(0.0, idle_timeout_seconds) / 3600.0 * sessions
    window = min(max(0.0, window_hours_per_day), 24.0)
    return max(0.0, min(window, active + idle_hours))


def estimate_cost_for_scenario(
    quote: ProviderQuote,
    scenario: CostScenario | None = None,
    *,
    idle_timeout_seconds: float | None = None,
) -> float | None:
    """Monthly cost under an explicit scenario; unknown storage is excluded.

    Provisioned rentals bill scheduled uptime. Scale-to-zero bills active
    compute plus one idle timeout per session, capped by the usage window.
    Returns None when the hourly price is unknown; storage is reported
    separately and never treated as zero.
    """
    if quote.price_per_hour_usd is None:
        return None
    resolved = scenario or COST_SCENARIO_WORKDAY
    idle_timeout = idle_timeout_seconds if idle_timeout_seconds is not None else resolved.idle_timeout_seconds
    if quote.billing_model == BillingModel.SCALE_TO_ZERO:
        billed_per_day = estimate_modal_billed_hours_per_day(
            resolved.active_hours_per_day,
            resolved.sessions_per_day,
            idle_timeout,
            window_hours_per_day=resolved.provisioned_hours_per_day,
        )
    else:
        billed_per_day = min(24.0, max(0.0, resolved.provisioned_hours_per_day))
    return quote.price_per_hour_usd * billed_per_day * 30.0


def format_cost_summary(
    quote: ProviderQuote,
    scenario: CostScenario | None = None,
    *,
    idle_timeout_seconds: float | None = None,
) -> str:
    """Hourly rate, 24/7 ceiling, and scenario cost in one comparable line."""
    resolved = scenario or COST_SCENARIO_WORKDAY
    if quote.price_per_hour_usd is None:
        return "Hourly unavailable · monthly unavailable (storage excluded)"
    hourly = f"${quote.price_per_hour_usd:.2f}/hr while billed"
    continuous = continuous_monthly_compute_cost(quote)
    continuous_text = f"${continuous:,.2f}/mo if left running 24/7" if continuous is not None else "24/7 unavailable"
    scenario_cost = estimate_cost_for_scenario(quote, resolved, idle_timeout_seconds=idle_timeout_seconds)
    scenario_text = (
        f"${scenario_cost:,.2f}/mo '{resolved.display_name}' ({resolved.describe()})"
        if scenario_cost is not None
        else f"'{resolved.display_name}' unavailable"
    )
    return f"{hourly} · {continuous_text} · {scenario_text} · storage separate"


def estimate_monthly_compute_cost(
    quote: ProviderQuote,
    workload: WorkloadProfile,
) -> float | None:
    """Normalize active-compute and provisioned billing into monthly spend."""

    if quote.price_per_hour_usd is None:
        return None
    paid_hours = min(24.0, max(0.0, workload.paid_hours_per_day))
    utilization = min(1.0, max(0.0, workload.utilization))
    if quote.billing_model == BillingModel.SCALE_TO_ZERO:
        paid_hours *= utilization
    return quote.price_per_hour_usd * paid_hours * 30.0


def estimate_cost_per_million_output_tokens(
    quote: ProviderQuote,
    workload: WorkloadProfile,
    monthly_cost_usd: float | None = None,
) -> float | None:
    """Estimate token cost from workload demand or measured throughput."""

    monthly_cost = (
        monthly_cost_usd
        if monthly_cost_usd is not None
        else estimate_monthly_compute_cost(quote, workload)
    )
    if monthly_cost is None:
        return None
    if workload.output_tokens_per_month and workload.output_tokens_per_month > 0:
        return monthly_cost * 1_000_000 / workload.output_tokens_per_month
    throughput = quote.estimated_output_tokens_per_second
    if throughput is None or throughput <= 0 or quote.price_per_hour_usd is None:
        return None
    cost = quote.price_per_hour_usd * 1_000_000 / (throughput * 3600.0)
    if quote.billing_model == BillingModel.PROVISIONED:
        utilization = min(1.0, max(0.0, workload.utilization))
        if utilization <= 0:
            return None
        cost /= utilization
    return cost


def _plan_from_quote(
    recipe: InferenceRecipe,
    quote: ProviderQuote,
    workload: WorkloadProfile,
) -> InferencePlan:
    monthly_cost = estimate_monthly_compute_cost(quote, workload)
    return InferencePlan(
        recipe=recipe,
        quote=quote,
        estimated_monthly_cost_usd=monthly_cost,
        estimated_cost_per_million_output_tokens_usd=(
            estimate_cost_per_million_output_tokens(quote, workload, monthly_cost)
        ),
    )


def _plan_monthly_cost_sort_key(plan: InferencePlan) -> tuple[int, int, float, str]:
    monthly_cost = plan.estimated_monthly_cost_usd
    return (
        plan.quote.availability.sort_rank,
        1 if monthly_cost is None else 0,
        monthly_cost if monthly_cost is not None else float("inf"),
        plan.quote.id,
    )

