"""Provider-agnostic inference recipe resolution and pricing quotes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Protocol, Sequence

from ..protocol.enums import BackendType, BillingModel, ComputeProvider, QuoteAvailability
from ..protocol.models import (
    ComputeOffer,
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
    is_compatible_prime_offer,
    preferred_prime_offer_image,
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
            if not is_compatible_prime_offer(
                offer,
                recipe.required_vram_gb,
                required_image=required_image,
            ):
                continue
            availability = _prime_availability(offer.stock_status)
            region = offer.country or offer.region or offer.data_center
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

    plans.sort(key=_plan_sort_key)
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


def _plan_sort_key(plan: InferencePlan) -> tuple[int, int, float, str]:
    availability_order = {
        QuoteAvailability.AVAILABLE: 0,
        QuoteAvailability.UNKNOWN: 1,
        QuoteAvailability.UNAVAILABLE: 2,
    }
    monthly_cost = plan.estimated_monthly_cost_usd
    return (
        availability_order[plan.quote.availability],
        1 if monthly_cost is None else 0,
        monthly_cost if monthly_cost is not None else float("inf"),
        plan.quote.id,
    )


def _prime_availability(stock_status: str | None) -> QuoteAvailability:
    value = (stock_status or "").strip().casefold()
    if value in {"available", "in_stock", "instock"}:
        return QuoteAvailability.AVAILABLE
    if value in {"unavailable", "out_of_stock", "outofstock"}:
        return QuoteAvailability.UNAVAILABLE
    return QuoteAvailability.UNKNOWN
