"""Model-aware Vast rental comparisons for Fast Deploy."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import math
from typing import Any

from ..protocol.enums import BackendType, BillingModel, ComputeProvider, QuoteAvailability
from ..protocol.models import InferencePlan, OfferCostBreakdown, ProviderQuote, VastModelOffer, VastOffer, VastProviderOptions
from .compute_availability import canonical_gpu_identity, recipe_for_placement
from .llamacpp_planner import assess_memory_placement, tuning_for_gpu_memory
from .quick_deploy import QuickDeployModel, QuickDeployProfile, quick_deploy_recipe
from .inference_options import COMPARISON_RENT_HOURS, COST_SCENARIO_WORKDAY, evaluate_quote_cost
from .runtime_support import load_llamacpp_support_manifest
from .vast_runtime import VAST_MAX_GPU_COUNT, VAST_MIN_COMPUTE_CAPABILITY, VAST_MIN_CUDA_VERSION
from .vast_startup_history import (
    load_vast_startup_history,
    machine_startup_stats,
)


def vast_first_hour_cost_usd(costs: OfferCostBreakdown, download_gb: float) -> float | None:
    """An hour of rent plus downloading the weights once; ``None`` if either price is unknown.

    Hourly price alone ranked a host charging $0.039/GB first, whose ~$0.47
    of transfer outweighed the rent it saved several times over.
    """
    hourly, per_gb = costs.total_per_hour_usd, costs.download_per_gb_usd
    if hourly is None or per_gb is None:
        return None
    return hourly * COMPARISON_RENT_HOURS + per_gb * download_gb


def _cost_rank(costs: OfferCostBreakdown, download_gb: float) -> tuple[bool, float, bool, float]:
    """Known first-hour cost first, then hourly price; unknown prices sort last."""
    total = vast_first_hour_cost_usd(costs, download_gb)
    hourly = costs.total_per_hour_usd
    return (
        total is None, total if total is not None else math.inf,
        hourly is None, hourly if hourly is not None else math.inf,
    )


def vast_gpu_label(offer: VastOffer) -> str:
    """Use planner memory units so equivalent cards share the GPU filter."""
    return canonical_gpu_identity(offer.gpu_type, offer.gpu_memory_gib)[1]


def vast_offer_is_rentable(offer: VastOffer) -> bool:
    """Whether a rental could serve *something*, independently of the model.

    The GPU filter spans every model, so it can only apply the checks that do
    not depend on one: bundle size, a known price, and the driver/architecture
    floors the pinned runtime was built against.
    """
    price = offer.costs.total_per_hour_usd
    return (
        _vast_offer_meets_runtime_floors(offer)
        and price is not None and price > 0
    )


def _vast_offer_meets_runtime_floors(offer: VastOffer) -> bool:
    """Whether a host clears the model-independent runtime floors.

    This is the eligibility subset of ``vast_offer_is_rentable`` that does not
    depend on pricing. It must run *before* cheapest-offer selection: picking
    the cheapest host per topology first and checking rentability afterward
    lets a cheaper incompatible host shadow a slightly more expensive host
    that could actually serve the model.
    """
    return (
        1 <= offer.gpu_count <= VAST_MAX_GPU_COUNT
        and offer.cuda_max_good is not None and offer.cuda_max_good >= VAST_MIN_CUDA_VERSION
        and offer.compute_capability is not None
        and offer.compute_capability >= VAST_MIN_COMPUTE_CAPABILITY
    )


def vast_plan_for_offer(row: VastModelOffer, profile: QuickDeployProfile) -> InferencePlan | None:
    """Promote a supported runtime to a deployable local endpoint."""
    price = row.costs.total_per_hour_usd
    manifest = load_llamacpp_support_manifest(profile.gguf_architecture)
    # An architecture Launchpad can only build has no pinned image to rent, so
    # no offer can serve it however capable the hardware is.
    if manifest.build_recipe or not vast_offer_is_rentable(row.offer):
        return None
    assert price is not None
    quote = ProviderQuote(
        id=row.id.replace(":comparison:", ":deploy:"), recipe_id=row.recipe.id,
        # Modal and Prime quotes carry this, and the deploy screen resolves a
        # plan's catalog profile through it. Without it a Vast plan fell back to
        # matching recipe ids across a separately cached catalog, which failed.
        configuration_id=profile.id,
        provider=ComputeProvider.VAST, provider_reference=row.offer.id,
        gpu_type=row.offer.gpu_type, gpu_count=row.offer.gpu_count, gpu_memory_gb=row.offer.gpu_memory_gib,
        price_per_hour_usd=price, billing_model=BillingModel.PROVISIONED,
        availability=QuoteAvailability.AVAILABLE, region=row.offer.location,
        security="datacenter" if row.offer.datacenter else "verified marketplace",
        provider_options=VastProviderOptions(
            row.offer.id, row.disk_gb, price, row.offer.machine_id, row.offer.gpu_count
        ),
        is_estimate=True,
        one_time_cost_usd=(
            row.costs.download_per_gb_usd * row.download_gb
            if row.costs.download_per_gb_usd is not None else None
        ),
    )
    evaluation = evaluate_quote_cost(
        quote, COST_SCENARIO_WORKDAY, includes_storage=True
    )
    return InferencePlan(
        recipe=row.recipe, quote=quote, assessment=row.assessment,
        estimated_monthly_cost_usd=evaluation.estimated_monthly_cost_usd,
        cost=evaluation,
    )


@lru_cache(maxsize=128)
def deployable_vast_offers(
    model: QuickDeployModel,
    offers: tuple[VastOffer, ...],
) -> tuple[VastModelOffer, ...]:
    """Return only the rows a user can actually rent for this model.

    Fast Deploy showed every memory-fitting offer and labelled the unrentable
    ones, which advertised prices and GPU types that led nowhere.
    """
    profiles = {quick_deploy_recipe(profile).id: profile for profile in model.profiles}
    return tuple(
        row for row in vast_offers_for_model(model, offers)
        if row.recipe.id in profiles
        and vast_plan_for_offer(row, profiles[row.recipe.id]) is not None
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
            if not _vast_offer_meets_runtime_floors(offer):
                # Skip hosts that can never run the pinned runtime before
                # price selection: an incompatible host must not shadow the
                # cheapest eligible host of the same topology.
                continue
            disk = max(required_disk, offer.disk_gb)
            capacity = offer.disk_capacity_gb if offer.disk_capacity_gb is not None else offer.disk_gb
            if capacity < disk:
                continue
            costs = _costs_for_disk(offer, disk)
            shape = (offer.gpu_type, offer.gpu_count, offer.gpu_memory_gib)
            existing = grouped.get(shape)
            if existing is None or (
                _cost_rank(costs, memory.weights_gb) < _cost_rank(existing[2], memory.weights_gb)
            ):
                grouped[shape] = (offer, disk, costs)
        for offer, disk, costs in grouped.values():
            # Same as the Modal and Prime path: the runtime margin the memory
            # model promises is a share of the device, and the device is only
            # known here. A rental left on the catalog's 2 GiB floor packs the
            # card and then has nothing left for llama.cpp's compute graphs.
            # Both derive from tuning_for_gpu_memory, so the recipe the plan
            # carries and the tuning it was assessed under cannot disagree.
            offer_tuning = tuning_for_gpu_memory(recipe.runtime_tuning, offer.gpu_memory_gib)
            offer_recipe = recipe_for_placement(recipe, offer.gpu_memory_gib)
            assessment = assess_memory_placement(
                memory, model_id=offer_recipe.model_id, revision=None,
                quant=offer_recipe.quant,
                runtime_id=profile.llamacpp_runtime_id,
                requirements=recipe.serving_requirements,
                tuning=offer_tuning,
                gpu_type=offer.gpu_type, gpu_count=offer.gpu_count,
                gpu_memory_gb=offer.gpu_memory_gib,
                price_per_hour_usd=costs.total_per_hour_usd,
            )
            if not assessment.fits or not assessment.gpu_resident:
                continue
            result.append(VastModelOffer(
                id=f"vast:comparison:{recipe.id}:{offer.id}", recipe=offer_recipe,
                offer=offer, gpu_label=vast_gpu_label(offer), disk_gb=disk,
                costs=costs, assessment=assessment, download_gb=memory.weights_gb,
            ))
    return tuple(
        sorted(result, key=lambda row: _vast_row_sort_key(row, _cached_startup_history()))
    )


def _cached_startup_history() -> dict[str, Any] | None:
    """Best-effort measured history; a corrupt file means no evidence."""
    try:
        return load_vast_startup_history()
    except Exception:
        return None


def _vast_row_sort_key(
    row: VastModelOffer, history: dict[str, Any] | None
) -> tuple[bool, float, bool, float, int, float, str]:
    """Cost first, measured startup second: keep the cheapest-host default.

    Cost is the first hour including the weight download, not the hourly
    price: transfer is billed per GB and varies more than rent between hosts.
    A machine with a recent successful observation sorts ahead of an
    unmeasured one at the *same* cost, so evidence moves the needle without
    ever spending more than the cheapest tier. ``inet_down`` stays out of the
    ranking entirely: advertised bandwidth predicts startup poorly.
    """
    stats = machine_startup_stats(history, row.offer.machine_id or "")
    return (
        *_cost_rank(row.costs, row.download_gb),
        0 if stats is not None else 1,
        stats["healthy_seconds"] if stats is not None else 0.0,
        row.id,
    )
