"""Provider-neutral aggregation and fulfillment for deployable GPU compute."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
import math
import re
from typing import Iterable, Sequence

from ..protocol.enums import (
    BackendType,
    BillingModel,
    ComputeProvider,
    QuoteAvailability,
)
from ..protocol.models import (
    ComputeAvailabilitySnapshot,
    ComputeConfiguration,
    ComputeOffer,
    ComputePlacement,
    InferencePlan,
    ModalProviderOptions,
    PrimeProviderOptions,
    ProviderQuote,
    WorkloadProfile,
)
from .inference_options import (
    estimate_cost_per_million_output_tokens,
    estimate_monthly_compute_cost,
)
from .modal_cli import resolve_modal_cli_path
from .modal_gpu import ModalGpuSpec, fetch_modal_gpu_catalog
from .prime_auth import get_prime_auth_status
from .prime_backend import (
    PrimeBackend,
    is_prime_gpu_offer,
    preferred_prime_offer_image,
    prime_offer_gpu_memory_gb,
    supports_prime_image,
)
from .quick_deploy import QuickDeployProfile, quick_deploy_recipe

_MODAL_GPU_COUNT_MAX = 8
_GPU_MEMORY_GB: dict[str, float] = {
    "T4": 16.0,
    "L4": 24.0,
    "A10": 24.0,
    "A100": 40.0,
    "A100-40GB": 40.0,
    "A100-80GB": 80.0,
    "L40S": 48.0,
    "RTX-PRO-6000": 96.0,
    "H100": 80.0,
    "H100!": 80.0,
    "H200": 141.0,
    "B200": 180.0,
    "B200+": 180.0,
}
_UNAVAILABLE_STOCK = {
    "unavailable",
    "out_of_stock",
    "outofstock",
    "sold_out",
    "soldout",
    "not_available",
    "notavailable",
}
_AVAILABLE_STOCK = {"available", "in_stock", "instock"}


def load_compute_availability() -> ComputeAvailabilitySnapshot:
    """Fetch connected providers concurrently and return one aggregated view."""

    errors: list[str] = []
    include_modal = False
    include_prime = False
    # GPU types and prices come from public Modal pages. Avoid a live token
    # subprocess here; deployment preflight remains responsible for auth.
    include_modal = resolve_modal_cli_path() is not None
    try:
        include_prime = get_prime_auth_status().authenticated
    except Exception as exc:
        errors.append(f"Prime authentication check failed: {exc}")

    modal_catalog: Sequence[ModalGpuSpec] = ()
    prime_offers: Sequence[ComputeOffer] = ()
    modal_future: Future[list[ModalGpuSpec]] | None = None
    prime_future: Future[list[ComputeOffer]] | None = None
    with ThreadPoolExecutor(max_workers=2) as executor:
        if include_modal:
            modal_future = executor.submit(fetch_modal_gpu_catalog)
        if include_prime:
            prime_future = executor.submit(PrimeBackend().list_offers)
        if modal_future is not None:
            try:
                modal_catalog = modal_future.result()
            except Exception as exc:
                errors.append(f"Modal catalog unavailable: {exc}")
        if prime_future is not None:
            try:
                prime_offers = prime_future.result()
            except Exception as exc:
                errors.append(f"Prime availability unavailable: {exc}")

    if not include_modal and not include_prime and not errors:
        errors.append("Connect a compute provider to load availability.")
    providers = tuple(
        provider
        for provider, included in (
            (ComputeProvider.MODAL, include_modal),
            (ComputeProvider.PRIME, include_prime),
        )
        if included
    )
    snapshot = aggregate_compute_availability(
        modal_catalog=modal_catalog,
        prime_offers=prime_offers,
    )
    return replace(snapshot, errors=tuple(errors), providers=providers)


def aggregate_compute_availability(
    *,
    modal_catalog: Sequence[ModalGpuSpec] = (),
    prime_offers: Sequence[ComputeOffer] = (),
) -> ComputeAvailabilitySnapshot:
    """Normalize provider catalogs and group equivalent GPU types."""

    placements = [*_modal_placements(modal_catalog), *_prime_placements(prime_offers)]
    grouped: dict[str, list[ComputePlacement]] = {}
    labels: dict[str, tuple[str, float]] = {}
    for placement in placements:
        config_id, display_name = canonical_gpu_identity(
            placement.gpu_type,
            placement.gpu_memory_gb,
        )
        grouped.setdefault(config_id, []).append(placement)
        labels[config_id] = (display_name, placement.gpu_memory_gb)

    configurations = []
    for config_id, rows in grouped.items():
        display_name, memory_gb = labels[config_id]
        rows.sort(key=_placement_sort_key)
        configurations.append(
            ComputeConfiguration(
                id=config_id,
                gpu_type=display_name,
                gpu_memory_gb=memory_gb,
                placements=tuple(rows),
            )
        )
    configurations.sort(
        key=lambda row: (
            row.minimum_price_per_hour_usd is None,
            row.minimum_price_per_hour_usd
            if row.minimum_price_per_hour_usd is not None
            else float("inf"),
            -row.live_placement_count,
            -row.gpu_memory_gb,
            row.gpu_type.casefold(),
        )
    )
    return ComputeAvailabilitySnapshot(configurations=tuple(configurations))


def canonical_gpu_identity(value: str, memory_gb: float) -> tuple[str, str]:
    """Return a stable cross-provider ID and concise display label."""

    normalized = re.sub(r"[^A-Z0-9]+", "-", value.strip().upper()).strip("-")
    memory_suffix = re.compile(rf"-{int(memory_gb)}GB$")
    family = memory_suffix.sub("", normalized)
    aliases = {
        "A100-40GB": "A100",
        "A100-80GB": "A100",
        "B200": "B200",
        "B200-PLUS": "B200",
        "H100": "H100",
        "H100-80GB": "H100",
        "H200": "H200",
        "H200-141GB": "H200",
        "RTX-PRO-6000": "RTX PRO 6000",
        "RTXPRO6000": "RTX PRO 6000",
    }
    family = aliases.get(normalized, aliases.get(family, family.replace("-", " ")))
    memory_label = f"{memory_gb:g}GB"
    display_name = f"{family} {memory_label}"
    config_id = re.sub(r"[^a-z0-9]+", "-", display_name.casefold()).strip("-")
    return config_id, display_name


def display_gpu_type(gpu_type: str, memory_gb: float | None = None) -> str:
    """Return the canonical GPU label used in availability and deploy lists."""

    resolved = memory_gb if memory_gb is not None and memory_gb > 0 else _modal_gpu_memory_gb(gpu_type)
    if resolved is None:
        match = re.search(r"(\d{1,3})\s*GB", gpu_type.upper().replace("_", " "))
        resolved = float(match.group(1)) if match is not None else None
    if resolved is None:
        cleaned = re.sub(r"[!+]+", "", gpu_type.strip())
        return re.sub(r"[_-]+", " ", cleaned).strip() or gpu_type
    return canonical_gpu_identity(gpu_type, resolved)[1]


def plans_for_compute_profile(
    configuration: ComputeConfiguration,
    profile: QuickDeployProfile,
    workload: WorkloadProfile | None = None,
) -> tuple[InferencePlan, ...]:
    """Build ranked fulfillment plans for one model on a selected GPU type."""

    workload = workload or WorkloadProfile()
    recipe = quick_deploy_recipe(profile)
    required_vram_gb = profile_required_vram_gb(profile)
    if recipe.required_vram_gb is None and required_vram_gb > 0:
        recipe = replace(recipe, required_vram_gb=required_vram_gb)
    plans: list[InferencePlan] = []
    for placement in configuration.placements:
        if placement.is_spot or recipe.backend not in placement.supported_backends:
            continue
        gpu_count = _placement_gpu_count(placement, required_vram_gb)
        if gpu_count is None:
            continue
        price = placement.price_per_hour_usd
        if price is not None and placement.price_is_per_gpu:
            price *= gpu_count
        quote = ProviderQuote(
            id=f"{placement.provider.value}:compute:{recipe.id}:{placement.id}:{gpu_count}",
            recipe_id=recipe.id,
            provider=placement.provider,
            provider_reference=placement.provider_reference,
            gpu_type=placement.gpu_type,
            gpu_count=gpu_count,
            price_per_hour_usd=price,
            billing_model=placement.billing_model,
            availability=placement.availability,
            region=placement.region,
            security=placement.security,
            is_estimate=placement.is_estimate,
            configuration_id=configuration.id,
            provider_options=placement.provider_options,
        )
        monthly_cost = estimate_monthly_compute_cost(quote, workload)
        plans.append(
            InferencePlan(
                recipe=recipe,
                quote=quote,
                estimated_monthly_cost_usd=monthly_cost,
                estimated_cost_per_million_output_tokens_usd=(
                    estimate_cost_per_million_output_tokens(
                        quote,
                        workload,
                        monthly_cost,
                    )
                ),
            )
        )
    plans.sort(key=_plan_sort_key)
    if plans:
        plans[0] = replace(
            plans[0],
            recommendation_reason="Best available placement for this GPU type",
        )
    return tuple(plans)


def profile_required_vram_gb(profile: QuickDeployProfile) -> float:
    """Return measured VRAM or infer it from the profile's known GPU shape."""

    if profile.required_vram_gb is not None and profile.required_vram_gb > 0:
        return float(profile.required_vram_gb)
    per_gpu_memory = _modal_gpu_memory_gb(profile.gpu_type.strip().upper())
    if per_gpu_memory is None:
        return 0.0
    # The selected shape already includes the catalog's five-percent headroom.
    return per_gpu_memory * max(1, profile.gpu_count) / 1.05


def compatible_compute_profiles(
    configuration: ComputeConfiguration,
    profiles: Iterable[QuickDeployProfile],
) -> tuple[tuple[QuickDeployProfile, tuple[InferencePlan, ...]], ...]:
    """Return catalog profiles that have at least one valid fulfillment plan."""

    compatible = []
    for profile in profiles:
        plans = plans_for_compute_profile(configuration, profile)
        if plans:
            compatible.append((profile, plans))
    compatible.sort(
        key=lambda item: (
            item[0].aa_rank is None,
            item[0].aa_rank if item[0].aa_rank is not None else 10**9,
            item[0].display_name.casefold(),
            item[0].id,
        )
    )
    return tuple(compatible)


def _modal_placements(catalog: Sequence[ModalGpuSpec]) -> list[ComputePlacement]:
    placements = []
    for spec in catalog:
        gpu_type = spec.value.strip().upper()
        memory_gb = _modal_gpu_memory_gb(gpu_type)
        if not gpu_type or memory_gb is None:
            continue
        placements.append(
            ComputePlacement(
                id=f"modal:{gpu_type.casefold()}",
                provider=ComputeProvider.MODAL,
                provider_reference=gpu_type,
                gpu_type=gpu_type,
                gpu_memory_gb=memory_gb,
                gpu_count_min=1,
                gpu_count_max=_MODAL_GPU_COUNT_MAX,
                price_per_hour_usd=spec.price_per_hour_usd,
                billing_model=BillingModel.SCALE_TO_ZERO,
                availability=QuoteAvailability.UNKNOWN,
                supported_backends=frozenset(
                    {BackendType.LLAMACPP, BackendType.VLLM}
                ),
                is_estimate=True,
                price_is_per_gpu=True,
                provider_options=ModalProviderOptions(),
            )
        )
    return placements


def _prime_placements(offers: Sequence[ComputeOffer]) -> list[ComputePlacement]:
    placements = []
    for offer in offers:
        memory_gb = prime_offer_gpu_memory_gb(offer)
        availability = _prime_availability(offer.stock_status)
        backends = frozenset(
            backend
            for backend in BackendType
            if supports_prime_image(offer, preferred_prime_offer_image(backend))
        )
        if (
            not is_prime_gpu_offer(offer)
            or memory_gb is None
            or availability == QuoteAvailability.UNAVAILABLE
            or offer.is_variable_price
            or (offer.security or "").casefold() != "secure_cloud"
            or not backends
        ):
            continue
        placements.append(
            ComputePlacement(
                id=f"prime:{offer.id}",
                provider=ComputeProvider.PRIME,
                provider_reference=offer.id,
                gpu_type=offer.gpu_type,
                gpu_memory_gb=memory_gb,
                gpu_count_min=offer.gpu_count,
                gpu_count_max=offer.gpu_count,
                price_per_hour_usd=offer.price_per_hour,
                billing_model=BillingModel.PROVISIONED,
                availability=availability,
                supported_backends=backends,
                region=offer.country or offer.region or offer.data_center,
                security=offer.security,
                is_spot=offer.is_spot,
                provider_options=PrimeProviderOptions(offer_id=offer.id),
            )
        )
    return placements


def _modal_gpu_memory_gb(gpu_type: str) -> float | None:
    if gpu_type in _GPU_MEMORY_GB:
        return _GPU_MEMORY_GB[gpu_type]
    normalized = re.sub(r"[^A-Z0-9]+", "-", gpu_type.upper()).strip("-")
    match = re.search(r"(?:^|-)(\d{1,3})GB(?:-|$)", normalized)
    return float(match.group(1)) if match is not None else None


def _prime_availability(stock_status: str | None) -> QuoteAvailability:
    normalized = re.sub(
        r"[^a-z0-9]+",
        "_",
        (stock_status or "").casefold(),
    ).strip("_")
    if normalized in _UNAVAILABLE_STOCK:
        return QuoteAvailability.UNAVAILABLE
    if normalized in _AVAILABLE_STOCK:
        return QuoteAvailability.AVAILABLE
    return QuoteAvailability.UNKNOWN


def _placement_gpu_count(
    placement: ComputePlacement,
    required_vram_gb: float,
) -> int | None:
    count = placement.gpu_count_min
    if required_vram_gb > 0:
        count = max(
            count,
            math.ceil(required_vram_gb * 1.05 / placement.gpu_memory_gb),
        )
    if count > placement.gpu_count_max:
        return None
    return count


def _placement_sort_key(row: ComputePlacement) -> tuple[int, int, float, str]:
    availability_order = {
        QuoteAvailability.AVAILABLE: 0,
        QuoteAvailability.UNKNOWN: 1,
        QuoteAvailability.UNAVAILABLE: 2,
    }
    return (
        availability_order[row.availability],
        row.price_per_hour_usd is None,
        row.price_per_hour_usd
        if row.price_per_hour_usd is not None
        else float("inf"),
        row.id,
    )


def _plan_sort_key(plan: InferencePlan) -> tuple[int, int, float, str]:
    availability_order = {
        QuoteAvailability.AVAILABLE: 0,
        QuoteAvailability.UNKNOWN: 1,
        QuoteAvailability.UNAVAILABLE: 2,
    }
    price = plan.quote.price_per_hour_usd
    return (
        availability_order[plan.quote.availability],
        price is None,
        price if price is not None else float("inf"),
        plan.quote.id,
    )
