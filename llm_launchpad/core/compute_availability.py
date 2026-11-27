"""Provider-neutral aggregation and fulfillment for deployable GPU compute."""

from __future__ import annotations

from dataclasses import replace
import math
import threading
import time
import re
from collections.abc import Callable, Sequence
from queue import Empty, SimpleQueue
from typing import Any

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
    InferenceRecipe,
    MemoryEstimate,
    ModalProviderOptions,
    PlacementAssessment,
    PrimeProviderOptions,
    ProviderQuote,
    WorkloadProfile,
    VastOfferQuery,
)
from .inference_options import (
    COST_SCENARIO_WORKDAY,
    evaluate_quote_cost,
)
from .llamacpp_planner import (
    assess_memory_placement,
    assessment_score,
    compile_server_args,
    memory_fits_device,
    memory_on_topology,
    tuning_for_gpu_memory,
)
from .modal_cli import resolve_modal_cli_path
from .modal_gpu import (
    MODAL_GPU_MEMORY_GB,
    ModalGpuSpec,
    modal_gpu_capacity_gib,
    fetch_modal_gpu_catalog,
    modal_gpu_count_limit,
    modal_gpu_count_options,
    modal_gpu_supports_fast_deploy,
)
from .provider_readiness import check_modal_readiness
from .prime_auth import get_prime_auth_status
from .prime_backend import (
    PrimeBackend,
    prime_stock_availability,
    is_prime_gpu_offer,
    preferred_prime_offer_image,
    prime_offer_gpu_memory_gb,
    supports_prime_image,
)
from .prime_offers import prime_gpu_capacity_gib
from .quick_deploy import QuickDeployProfile, quick_deploy_recipe
from .vast_auth import VastCredentials, resolve_vast_credentials
from .vast_backend import VastBackend

# Shared deadline for provider lookups that have yet to report back.
COMPUTE_AVAILABILITY_TIMEOUT_SECONDS = 45.0


def load_compute_availability(
    *,
    force_refresh: bool = False,
    on_partial: Callable[[ComputeAvailabilitySnapshot, tuple[ComputeProvider, ...]], None] | None = None,
) -> ComputeAvailabilitySnapshot:
    """Return the final snapshot and optionally publish completed providers as they arrive."""
    errors: list[str] = []
    include_modal = resolve_modal_cli_path() is not None
    include_prime = False
    try:
        include_prime = get_prime_auth_status().authenticated
    except Exception as exc:
        errors.append(f"Prime authentication check failed: {exc}")
    vast_credentials = VastCredentials()
    try:
        vast_credentials = resolve_vast_credentials()
    except ValueError as exc:
        errors.append(f"Vast key unavailable: {exc}")
    include_vast = bool(vast_credentials.api_key)

    requests: dict[str, tuple[ComputeProvider, Callable[[], Any]]] = {}
    if include_modal:
        requests["Modal catalog"] = (
            ComputeProvider.MODAL, lambda: fetch_modal_gpu_catalog(force_refresh=force_refresh)
        )
        requests["Modal authentication"] = (
            ComputeProvider.MODAL, lambda: check_modal_readiness(refresh=force_refresh)
        )
    if include_prime:
        requests["Prime availability"] = (ComputeProvider.PRIME, PrimeBackend().list_offers)
    if include_vast:
        requests["Vast availability"] = (
            ComputeProvider.VAST,
            lambda: VastBackend(vast_credentials).list_offers(VastOfferQuery(gpu_count=None, limit=500)),
        )

    completed: SimpleQueue[tuple[str, Any, str | None]] = SimpleQueue()
    results: dict[str, Any] = {}
    remaining = {label: provider for label, (provider, _) in requests.items()}

    def fetch(label: str, call: Callable[[], Any]) -> None:
        try:
            completed.put((label, call(), None))
        except BaseException as exc:
            completed.put((label, None, f"{label} unavailable: {exc}"))

    def snapshot(*, final: bool = False) -> ComputeAvailabilitySnapshot:
        readiness = results.get("Modal authentication")
        modal_ready = "Modal catalog" not in remaining and bool(readiness and readiness.verified)
        modal_catalog = (results.get("Modal catalog") or ()) if modal_ready else ()
        notes = errors.copy()
        if readiness and not readiness.verified:
            notes.append(f"Modal {readiness.display}")
        if modal_catalog and not any(spec.price_per_hour_usd is not None for spec in modal_catalog):
            notes.append("Modal GPU rates could not be read.")
        elif any(spec.stale_since is not None for spec in modal_catalog):
            oldest = min(spec.stale_since for spec in modal_catalog if spec.stale_since is not None)
            age_minutes = max(0, int(time.monotonic() - oldest)) // 60
            notes.append(f"Modal pricing refresh failed. Cached rates are {age_minutes} min old.")
        if final and not modal_ready and not include_prime and not include_vast and not notes:
            notes.append("Connect a compute provider to load availability.")
        providers = tuple(
            provider
            for provider, included in (
                (ComputeProvider.MODAL, modal_ready),
                (ComputeProvider.PRIME, include_prime and "Prime availability" not in remaining),
                (ComputeProvider.VAST, include_vast and "Vast availability" not in remaining),
            )
            if included
        )
        return replace(
            aggregate_compute_availability(
                modal_catalog=modal_catalog,
                prime_offers=results.get("Prime availability") or (),
            ),
            errors=tuple(notes),
            providers=providers,
            vast_offers=tuple(results.get("Vast availability") or ()),
            vast_configured=include_vast,
        )

    # Daemon fetches keep an unresponsive provider from holding the process
    # open. Modal's pricing and authentication must both settle before its
    # offers can be shown.
    deadline = time.monotonic() + COMPUTE_AVAILABILITY_TIMEOUT_SECONDS
    for label, (_, call) in requests.items():
        threading.Thread(target=fetch, args=(label, call), daemon=True).start()

    while remaining:
        try:
            label, value, error = completed.get(timeout=max(0.0, deadline - time.monotonic()))
        except Empty:
            for label in remaining:
                errors.append(f"{label} did not answer within {int(COMPUTE_AVAILABILITY_TIMEOUT_SECONDS)}s")
            remaining.clear()
            break
        provider = remaining.pop(label)
        results[label] = value
        if error:
            errors.append(error)
        pending = tuple(dict.fromkeys(remaining.values()))
        if on_partial is not None and pending and provider not in pending:
            on_partial(snapshot(), pending)
    return snapshot(final=True)


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


def gpu_memory_label(memory_gb: float) -> str:
    """Render VRAM the way vendors quote it: whole gigabytes.

    Providers derive capacity from the device byte count, so Vast reports
    5.99414 for a 6 GB card and 79.6475 for an 80 GB one. Printing that raw
    filled the GPU filter with numbers like "RTX A2000 5.99414GB", and because
    the identity below is built from this label it also split one card into
    several entries whenever two hosts rounded differently.
    """

    if memory_gb <= 0:
        return "0GB"
    if memory_gb < 1:
        return f"{memory_gb:.2g}GB"
    return f"{round(memory_gb)}GB"


def canonical_gpu_identity(value: str, memory_gb: float) -> tuple[str, str]:
    """Return a stable cross-provider ID and concise display label."""

    normalized = re.sub(r"[^A-Z0-9]+", "-", value.strip().upper()).strip("-")
    # Keep the product label stable when ECC lowers the reported framebuffer.
    nominal = MODAL_GPU_MEMORY_GB.get(normalized)
    memory_label = gpu_memory_label(nominal if nominal and 0.93 * nominal <= memory_gb <= nominal else memory_gb)
    # Strip the capacity the name already carries, using the same rounded label
    # that is about to be appended; deriving it with int() truncation left
    # "RTX A2000 6GB" reading "RTX A2000 6GB 6GB".
    memory_suffix = re.compile(rf"-{re.escape(memory_label)}$")
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
    *,
    rejected: list[str] | None = None,
) -> tuple[InferencePlan, ...]:
    """Build ranked fulfillment plans for one recipe on a selected GPU type.

    Placements that cannot hold the full context on GPU are excluded. Pass
    ``rejected`` to collect the reasons: a silently shorter list tells the
    reader nothing about why their hardware is missing.
    """
    plans, _ = assessed_plans_for_compute_profile(
        configuration, profile, workload, rejected=rejected
    )
    return plans


def recipe_for_placement(
    recipe: InferenceRecipe,
    gpu_memory_gb: float | None,
) -> InferenceRecipe:
    """Set the runtime's free-memory target for the selected device."""

    tuning = recipe.runtime_tuning
    requirements = recipe.serving_requirements
    if tuning is None or requirements is None:
        return recipe
    retuned = tuning_for_gpu_memory(tuning, gpu_memory_gb)
    if retuned == tuning:
        return recipe
    return replace(
        recipe,
        runtime_tuning=retuned,
        server_args=compile_server_args(requirements, retuned),
    )


def assessed_plans_for_compute_profile(
    configuration: ComputeConfiguration,
    profile: QuickDeployProfile,
    workload: WorkloadProfile | None = None,
    *,
    rejected: list[str] | None = None,
) -> tuple[tuple[InferencePlan, ...], tuple[PlacementAssessment, ...]]:
    """Return deployable plans plus the assessments behind every placement.

    The returned assessments cover placements that were evaluated but not
    offered, so callers can distinguish "does not fit this hardware" from "no
    current offer is deployable". Policy exclusions (spot, backend, price) are
    reported through ``rejected``; memory and runtime outcomes are reported
    through the assessments.

    ``workload`` is a compatibility shim: stored cost always uses the canonical
    workday scenario, with explicit output-token demand when supplied.
    """

    del workload
    recipe = quick_deploy_recipe(profile)
    required_vram_gb = profile_required_vram_gb(profile)
    if not math.isfinite(required_vram_gb) or required_vram_gb <= 0:
        if rejected is not None:
            rejected.append("Model memory requirements are unknown.")
        return (), ()
    if recipe.required_vram_gb is None:
        recipe = replace(recipe, required_vram_gb=required_vram_gb)
    memory = profile.memory_estimate
    plans: list[InferencePlan] = []
    evaluated: list[PlacementAssessment] = []
    for placement in configuration.placements:
        if placement.provider == ComputeProvider.MODAL and not modal_gpu_supports_fast_deploy(placement.gpu_type):
            if rejected is not None:
                rejected.append(f"{placement.gpu_type} can allocate a B300, which requires a CUDA 13.1-compatible runtime.")
            continue
        if placement.is_spot or recipe.backend not in placement.supported_backends:
            if rejected is not None:
                if placement.is_spot:
                    rejected.append(
                        f"{placement.gpu_type} is spot-only and excluded by policy."
                    )
                else:
                    rejected.append(
                        f"{placement.gpu_type} does not support the "
                        f"{recipe.backend.value} backend."
                    )
            continue
        capacity = placement.gpu_memory_gb
        if placement.provider == ComputeProvider.PRIME and recipe.gguf_artifact is not None:
            capacity = prime_gpu_capacity_gib(placement.gpu_type, capacity)
        placement_recipe = recipe_for_placement(recipe, capacity)
        assessment = None
        if (
            memory is not None
            and placement_recipe.serving_requirements is not None
            and placement_recipe.runtime_tuning is not None
        ):
            gpu_count = None
            for candidate in _placement_gpu_counts(placement):
                assessment = assess_memory_placement(
                    memory, model_id=placement_recipe.model_id, revision=None,
                    quant=placement_recipe.quant, runtime_id=profile.llamacpp_runtime_id,
                    requirements=placement_recipe.serving_requirements, tuning=placement_recipe.runtime_tuning,
                    gpu_type=placement.gpu_type, gpu_count=candidate, gpu_memory_gb=capacity,
                    price_per_hour_usd=_placement_price(placement, candidate),
                    artifact=recipe.gguf_artifact, provider=placement.provider,
                )
                evaluated.append(assessment)
                if assessment.fits and assessment.gpu_resident:
                    gpu_count = candidate
                    break
        else:
            gpu_count = _placement_gpu_count(placement, required_vram_gb, memory_estimate=memory)
        if gpu_count is None:
            if rejected is not None:
                rejected.append(
                    (assessment.rejection_reason if assessment is not None else None)
                    or f"{placement.gpu_type} has no topology large enough for the full-context plan."
                )
            continue
        price = _placement_price(placement, gpu_count)
        single_tps = None
        if assessment is not None:
            single_tps = max(
                (
                    point.output_tokens_per_second or 0.0
                    for point in assessment.performance
                    if point.concurrency == 1
                ),
                default=0.0,
            ) or None
        quote = ProviderQuote(
            id=f"{placement.provider.value}:compute:{recipe.id}:{placement.id}:{gpu_count}",
            recipe_id=recipe.id,
            provider=placement.provider,
            provider_reference=placement.provider_reference,
            gpu_type=placement.gpu_type,
            gpu_count=gpu_count,
            price_per_hour_usd=price,
            billing_model=placement.billing_model,
            gpu_memory_gb=capacity,
            availability=placement.availability,
            region=placement.region,
            security=placement.security,
            is_estimate=placement.is_estimate,
            configuration_id=configuration.id,
            provider_options=placement.provider_options,
            estimated_output_tokens_per_second=single_tps,
        )
        monthly_cost = None
        per_million = None
        evaluation = evaluate_quote_cost(quote, COST_SCENARIO_WORKDAY)
        monthly_cost = evaluation.estimated_monthly_cost_usd
        per_million = evaluation.estimated_cost_per_million_output_tokens_usd
        plans.append(
            InferencePlan(
                recipe=placement_recipe,
                quote=quote,
                estimated_monthly_cost_usd=monthly_cost,
                estimated_cost_per_million_output_tokens_usd=per_million,
                assessment=assessment,
                cost=evaluation,
            )
        )
    plans.sort(
        key=lambda plan: (
            -assessment_score(
                plan.assessment,
                recipe.serving_requirements.objective,
            )
            if plan.assessment is not None and recipe.serving_requirements is not None
            else 0.0,
            *_plan_hourly_price_sort_key(plan),
        )
    )
    if plans:
        plans[0] = replace(
            plans[0],
            recommendation_reason=(
                "Best full-context throughput for this GPU type"
                if plans[0].assessment is not None
                else "Best available placement for this GPU type"
            ),
        )
    return tuple(plans), tuple(evaluated)


def profile_required_vram_gb(profile: QuickDeployProfile) -> float:
    """Return measured VRAM or infer it from the profile's known GPU shape."""

    if profile.required_vram_gb is not None and profile.required_vram_gb > 0:
        return float(profile.required_vram_gb)
    per_gpu_memory = _modal_gpu_memory_gb(profile.gpu_type.strip().upper())
    if per_gpu_memory is None:
        return 0.0
    # The selected shape already includes the catalog's five-percent headroom.
    return per_gpu_memory * max(1, profile.gpu_count) / 1.05


def _modal_placements(catalog: Sequence[ModalGpuSpec]) -> list[ComputePlacement]:
    placements = []
    for spec in catalog:
        gpu_type = spec.value.strip().upper()
        memory_gb = modal_gpu_capacity_gib(gpu_type) or _modal_gpu_memory_gb(gpu_type)
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
                gpu_count_max=modal_gpu_count_limit(gpu_type),
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
        availability = prime_stock_availability(offer.stock_status)
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
    if gpu_type in MODAL_GPU_MEMORY_GB:
        return MODAL_GPU_MEMORY_GB[gpu_type]
    normalized = re.sub(r"[^A-Z0-9]+", "-", gpu_type.upper()).strip("-")
    match = re.search(r"(?:^|-)(\d{1,3})GB(?:-|$)", normalized)
    return float(match.group(1)) if match is not None else None


def _placement_price(placement: ComputePlacement, gpu_count: int) -> float | None:
    price = placement.price_per_hour_usd
    return price * gpu_count if price is not None and placement.price_is_per_gpu else price


def _placement_gpu_counts(placement: ComputePlacement) -> tuple[int, ...]:
    if placement.provider == ComputeProvider.MODAL:
        return tuple(
            candidate
            for candidate in modal_gpu_count_options(placement.gpu_type)
            if placement.gpu_count_min <= candidate <= placement.gpu_count_max
        )
    return tuple(range(placement.gpu_count_min, placement.gpu_count_max + 1))


def _placement_gpu_count(
    placement: ComputePlacement,
    required_vram_gb: float,
    *,
    memory_estimate: MemoryEstimate | None = None,
) -> int | None:
    candidates = _placement_gpu_counts(placement)
    if not candidates:
        return None
    if memory_estimate is not None:
        capacity = placement.gpu_memory_gb
        return next((
            candidate for candidate in candidates
            if memory_fits_device(memory_on_topology(memory_estimate, candidate, capacity), capacity)
        ), None)
    if required_vram_gb > 0:
        needed = max(
            placement.gpu_count_min,
            math.ceil(required_vram_gb * 1.05 / placement.gpu_memory_gb),
        )
        return next((candidate for candidate in candidates if candidate >= needed), None)
    return candidates[0]


def _placement_sort_key(row: ComputePlacement) -> tuple[int, int, float, str]:
    return (
        row.availability.sort_rank,
        row.price_per_hour_usd is None,
        row.price_per_hour_usd
        if row.price_per_hour_usd is not None
        else float("inf"),
        row.id,
    )


def _plan_hourly_price_sort_key(plan: InferencePlan) -> tuple[int, int, float, str]:
    price = plan.quote.price_per_hour_usd
    return (
        plan.quote.availability.sort_rank,
        price is None,
        price if price is not None else float("inf"),
        plan.quote.id,
    )
