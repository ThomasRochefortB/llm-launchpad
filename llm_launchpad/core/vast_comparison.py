"""Model-aware Vast rental comparisons for Fast Deploy."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import math

from ..protocol.enums import BackendType, BillingModel, ComputeProvider, QuoteAvailability
from ..protocol.models import InferencePlan, OfferCostBreakdown, ProviderQuote, VastModelOffer, VastOffer, VastProviderOptions, WorkloadProfile
from .compute_availability import canonical_gpu_identity
from .llamacpp_planner import assess_memory_placement
from .quick_deploy import QuickDeployModel, QuickDeployProfile, quick_deploy_recipe
from .inference_options import estimate_monthly_compute_cost
from .runtime_support import load_llamacpp_support_manifest
from .vast_runtime import VAST_MAX_GPU_COUNT, VAST_MIN_COMPUTE_CAPABILITY, VAST_MIN_CUDA_VERSION


def vast_gpu_label(offer: VastOffer) -> str:
    """Use planner memory units so equivalent cards share the GPU filter."""
    return canonical_gpu_identity(offer.gpu_type, offer.gpu_memory_gib)[1]


def vast_plan_for_offer(row: VastModelOffer, profile: QuickDeployProfile) -> InferencePlan | None:
    """Promote a supported runtime to a deployable local endpoint."""
    price = row.costs.total_per_hour_usd
    manifest = load_llamacpp_support_manifest(profile.gguf_architecture)
    if not 1 <= row.offer.gpu_count <= VAST_MAX_GPU_COUNT or price is None or price <= 0 or manifest.build_recipe:
        return None
    if row.offer.cuda_max_good is None or row.offer.cuda_max_good < VAST_MIN_CUDA_VERSION:
        return None
    if row.offer.compute_capability is None or row.offer.compute_capability < VAST_MIN_COMPUTE_CAPABILITY:
        return None
    quote = ProviderQuote(
        id=row.id.replace(":comparison:", ":deploy:"), recipe_id=row.recipe.id,
        provider=ComputeProvider.VAST, provider_reference=row.offer.id,
        gpu_type=row.offer.gpu_type, gpu_count=row.offer.gpu_count, gpu_memory_gb=row.offer.gpu_memory_gib,
        price_per_hour_usd=price, billing_model=BillingModel.PROVISIONED,
        availability=QuoteAvailability.AVAILABLE, region=row.offer.location,
        security="datacenter" if row.offer.datacenter else "verified marketplace",
        provider_options=VastProviderOptions(
            row.offer.id, row.disk_gb, price, row.offer.machine_id, row.offer.gpu_count
        ),
        is_estimate=True,
    )
    return InferencePlan(
        recipe=row.recipe, quote=quote, assessment=row.assessment,
        estimated_monthly_cost_usd=estimate_monthly_compute_cost(quote, WorkloadProfile()),
    )


def _costs_for_disk(offer: VastOffer, disk_gb: int) -> OfferCostBreakdown:
    """Estimate resized disk at the quoted per-GB rate; retain unknown prices."""
    costs = offer.costs
    if disk_gb == offer.disk_gb:
        return costs
    if costs.disk_per_hour_usd is None or offer.disk_gb <= 0:
        return replace(costs, disk_per_hour_usd=None, total_per_hour_usd=None)
    disk = costs.disk_per_hour_usd * disk_gb / offer.disk_gb
    total = (
        costs.total_per_hour_usd - costs.disk_per_hour_usd + disk
        if costs.total_per_hour_usd is not None else None
    )
    return replace(costs, disk_per_hour_usd=disk, total_per_hour_usd=total)


@lru_cache(maxsize=128)
def vast_offers_for_model(
    model: QuickDeployModel,
    offers: tuple[VastOffer, ...],
) -> tuple[VastModelOffer, ...]:
    """Return the cheapest full-context fit for each quant and GPU topology.

    These comparisons do not certify the host, runtime, or endpoint transport.
    Incomplete model metadata cannot establish a fit and is excluded.
    """
    result: list[VastModelOffer] = []
    seen_recipes: set[str] = set()
    for profile in model.profiles:
        memory = profile.memory_estimate
        recipe = quick_deploy_recipe(profile)
        if (
            recipe.id in seen_recipes or recipe.backend != BackendType.LLAMACPP
            or memory is None or recipe.serving_requirements is None
            or recipe.runtime_tuning is None
            or recipe.serving_requirements.context_tokens < model.max_context_tokens
            or not math.isfinite(memory.weights_gb) or memory.weights_gb <= 0
        ):
            continue
        seen_recipes.add(recipe.id)
        # Model weights plus download/metadata headroom and room for the image.
        # Keep at least the discovery allocation so small models are not shown
        # at an artificially discounted disk price.
        required_disk = max(100, math.ceil(memory.weights_gb * 1.1 + 10))
        grouped: dict[tuple[str, int, float], tuple[VastOffer, int, OfferCostBreakdown]] = {}
        for offer in offers:
            disk = max(required_disk, offer.disk_gb)
            capacity = offer.disk_capacity_gb if offer.disk_capacity_gb is not None else offer.disk_gb
            if capacity < disk:
                continue
            costs = _costs_for_disk(offer, disk)
            shape = (offer.gpu_type, offer.gpu_count, offer.gpu_memory_gib)
            existing = grouped.get(shape)
            price = costs.total_per_hour_usd
            old_price = existing[2].total_per_hour_usd if existing else None
            if existing is None or (price is not None and (old_price is None or price < old_price)):
                grouped[shape] = (offer, disk, costs)
        for offer, disk, costs in grouped.values():
            assessment = assess_memory_placement(
                memory, model_id=recipe.model_id, revision=None, quant=recipe.quant,
                runtime_id=profile.llamacpp_runtime_id,
                requirements=recipe.serving_requirements, tuning=recipe.runtime_tuning,
                gpu_type=offer.gpu_type, gpu_count=offer.gpu_count,
                gpu_memory_gb=offer.gpu_memory_gib,
                price_per_hour_usd=costs.total_per_hour_usd,
            )
            if not assessment.fits or not assessment.gpu_resident:
                continue
            result.append(VastModelOffer(
                id=f"vast:comparison:{recipe.id}:{offer.id}", recipe=recipe,
                offer=offer, gpu_label=vast_gpu_label(offer), disk_gb=disk,
                costs=costs, assessment=assessment,
            ))
    return tuple(sorted(result, key=lambda row: (
        row.costs.total_per_hour_usd is None,
        row.costs.total_per_hour_usd if row.costs.total_per_hour_usd is not None else math.inf,
        row.id,
    )))
