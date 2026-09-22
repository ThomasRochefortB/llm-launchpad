"""Provider-neutral aggregation and fulfillment for deployable GPU compute."""

from __future__ import annotations

from dataclasses import replace
import math
import threading
import time
import re
from collections.abc import Sequence
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
    ModalProviderOptions,
    PlacementAssessment,
    PrimeProviderOptions,
    ProviderQuote,
    WorkloadProfile,
    VastOffer,
    VastOfferQuery,
)
from .inference_options import (
    COST_SCENARIO_WORKDAY,
    evaluate_quote_cost,
)
from .fit_calibration import MemoryCalibration, calibration_key, load_memory_calibration
from .llamacpp_planner import (
    assess_memory_placement,
    assessment_score,
    compile_server_args,
    device_capacity_bytes,
    gib_to_bytes,
    per_device_requirements,
    tuning_for_gpu_memory,
)
from .modal_cli import resolve_modal_cli_path
from .modal_gpu import ModalGpuSpec, fetch_modal_gpu_catalog
from .prime_auth import get_prime_auth_status
from .prime_backend import (
    PrimeBackend,
    prime_stock_availability,
    is_prime_gpu_offer,
    preferred_prime_offer_image,
    prime_offer_gpu_memory_gb,
    supports_prime_image,
)
from .quick_deploy import QuickDeployProfile, quick_deploy_recipe
from .vast_auth import VastCredentials, resolve_vast_credentials
from .vast_backend import VastBackend

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
# How long every provider fetch together may hold the screen. Each of these
# calls has its own request timeout, but they were awaited with none: one
# catalog that never answered left Fast Deploy on "loading" with no deadline
# and no way back, which is indistinguishable from a hung client. Availability
# is a comparison across providers, so a slow one should cost its own row, not
# the whole screen -- whatever has not arrived is reported unavailable and the
# rest is still shown.
COMPUTE_AVAILABILITY_TIMEOUT_SECONDS = 45.0


class _ProviderFetch:
    """One provider lookup on a daemon thread, collected against a deadline.

    The thread is never joined past the deadline: an unanswered provider costs
    its own row and nothing else, and it cannot outlive the interpreter.
    """

    def __init__(self, call: Any, *args: Any) -> None:
        self._call = call
        self._args = args
        self._value: Any = None
        self._error: BaseException | None = None
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            self._value = self._call(*self._args)
        except BaseException as exc:  # reported as this provider's own error
            self._error = exc
        finally:
            self._done.set()

    def start(self) -> None:
        self._thread.start()

    def collect(self, deadline: float) -> tuple[Any, str | None]:
        """Return (value, error phrase); the phrase is None on success."""
        if not self._done.wait(timeout=max(0.0, deadline - time.monotonic())):
            return None, (
                "did not answer within "
                f"{int(COMPUTE_AVAILABILITY_TIMEOUT_SECONDS)}s"
            )
        if self._error is not None:
            return None, f"unavailable: {self._error}"
        return self._value, None


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
    vast_credentials = VastCredentials()
    try:
        vast_credentials = resolve_vast_credentials()
    except ValueError as exc:
        errors.append(f"Vast key unavailable: {exc}")

    modal_catalog: Sequence[ModalGpuSpec] = ()
    prime_offers: Sequence[ComputeOffer] = ()
    vast_offers: Sequence[VastOffer] = ()
    deadline = time.monotonic() + COMPUTE_AVAILABILITY_TIMEOUT_SECONDS

    # Daemon threads, not a pool. A ThreadPoolExecutor's workers are joined by
    # the interpreter at exit whatever shutdown() was told, so a provider that
    # never answers was not abandoned at all -- it was deferred to quitting
    # time, where it holds the process open with no interface left to explain
    # why. A daemon thread is genuinely dropped, which is what this budget
    # promises. Each fetch still has its own request timeout underneath.
    fetches: list[tuple[str, _ProviderFetch]] = []
    if include_modal:
        fetches.append(("Modal catalog", _ProviderFetch(fetch_modal_gpu_catalog)))
    if include_prime:
        fetches.append(("Prime availability", _ProviderFetch(PrimeBackend().list_offers)))
    if vast_credentials.api_key:
        fetches.append((
            "Vast availability",
            _ProviderFetch(
                VastBackend(vast_credentials).list_offers,
                VastOfferQuery(gpu_count=None, limit=500),
            ),
        ))
    for _, fetch in fetches:
        fetch.start()

    results: dict[str, Any] = {}
    for label, fetch in fetches:
        value, error = fetch.collect(deadline)
        if error is not None:
            errors.append(f"{label} {error}")
        results[label] = value
    modal_catalog = results.get("Modal catalog") or ()
    prime_offers = results.get("Prime availability") or ()
    vast_offers = results.get("Vast availability") or ()

    if not include_modal and not include_prime and not vast_credentials.api_key and not errors:
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
    return replace(
        snapshot, errors=tuple(errors), providers=providers,
        vast_offers=tuple(vast_offers), vast_configured=bool(vast_credentials.api_key),
    )


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
    memory_label = gpu_memory_label(memory_gb)
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
    """Re-cut one recipe for the device it will actually run on.

    Only the runtime margin depends on the device, and it is deliberately not
    part of the calibration key, so a re-cut recipe still matches measurements
    taken on any other topology of the same plan.
    """

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
    if recipe.required_vram_gb is None and required_vram_gb > 0:
        recipe = replace(recipe, required_vram_gb=required_vram_gb)
    # The topology picker and the assessment below have to agree on how much
    # memory the plan needs. If the picker sized on the formula while the
    # assessment used a runtime measurement, it would propose shapes the
    # assessment then rejected and the model would end up with no plan at all.
    measured = None
    if recipe.serving_requirements is not None and recipe.runtime_tuning is not None:
        measured = load_memory_calibration(
            calibration_key(
                model_id=recipe.model_id,
                revision=None,
                quant=recipe.quant,
                runtime_id=profile.llamacpp_runtime_id,
                requirements=recipe.serving_requirements,
                tuning=recipe.runtime_tuning,
            )
        )
    plans: list[InferencePlan] = []
    evaluated: list[PlacementAssessment] = []
    for placement in configuration.placements:
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
        gpu_count = _placement_gpu_count(
            placement,
            required_vram_gb,
            memory_estimate=profile.memory_estimate,
            calibration=measured,
        )
        if gpu_count is None:
            if rejected is not None:
                rejected.append(
                    f"{placement.gpu_type} has no topology large enough for the "
                    "full-context plan."
                )
            continue
        price = placement.price_per_hour_usd
        if price is not None and placement.price_is_per_gpu:
            price *= gpu_count
        # The catalog is built before any GPU is chosen, so its runtime margin
        # is the 2 GiB floor. The margin the memory model actually promises is
        # 5% of the device, and that only becomes computable here. Leaving the
        # floor in place shipped a plan that packed a 180 GB B200 to within
        # 2 GiB and then had nothing left for llama.cpp's compute graphs: five
        # GLM-5.3-Flash deploys died in graph_reserve, and the one that
        # survived differed from them in this argument alone (9216 vs 2048).
        placement_recipe = recipe_for_placement(recipe, placement.gpu_memory_gb)
        assessment = None
        if (
            profile.memory_estimate is not None
            and placement_recipe.serving_requirements is not None
            and placement_recipe.runtime_tuning is not None
        ):
            assessment = assess_memory_placement(
                profile.memory_estimate,
                model_id=placement_recipe.model_id,
                revision=None,
                quant=placement_recipe.quant,
                runtime_id=profile.llamacpp_runtime_id,
                requirements=placement_recipe.serving_requirements,
                tuning=placement_recipe.runtime_tuning,
                gpu_type=placement.gpu_type,
                gpu_count=gpu_count,
                gpu_memory_gb=placement.gpu_memory_gb,
                price_per_hour_usd=price,
            )
            evaluated.append(assessment)
            if not assessment.fits or not assessment.gpu_resident:
                if rejected is not None:
                    rejected.append(
                        assessment.rejection_reason
                        or f"{placement.gpu_type} cannot hold the full context on GPU."
                    )
                continue
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
            gpu_memory_gb=placement.gpu_memory_gb,
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
    if gpu_type in _GPU_MEMORY_GB:
        return _GPU_MEMORY_GB[gpu_type]
    normalized = re.sub(r"[^A-Z0-9]+", "-", gpu_type.upper()).strip("-")
    match = re.search(r"(?:^|-)(\d{1,3})GB(?:-|$)", normalized)
    return float(match.group(1)) if match is not None else None


def _placement_gpu_count(
    placement: ComputePlacement,
    required_vram_gb: float,
    *,
    memory_estimate: object | None = None,
    calibration: MemoryCalibration | None = None,
) -> int | None:
    count = placement.gpu_count_min
    if memory_estimate is not None:
        total_gb = float(getattr(memory_estimate, "total_gb", 0.0)) - float(
            getattr(memory_estimate, "reserve_gb", 0.0)
        )
        # Graph memory is replicated on every device, so it stays out of the
        # division. Sharding it here proposed topologies that the placement
        # assessment then rejected, leaving the model with no plan at all
        # rather than the larger topology that does hold it.
        base_count = max(1, len(getattr(memory_estimate, "per_device_required_gb", ()) or ()))
        per_device_graph = float(getattr(memory_estimate, "compute_gb", 0.0)) + float(
            getattr(memory_estimate, "attention_scratch_gb", 0.0)
        )
        shardable_gb = max(0.0, total_gb - per_device_graph * base_count)
        layer_count = getattr(memory_estimate, "total_layer_count", None)
        capacity = device_capacity_bytes(placement.gpu_memory_gb)
        for candidate in range(count, placement.gpu_count_max + 1):
            reserve_per_gpu = max(2.0, placement.gpu_memory_gb * 0.05)
            # Layers are indivisible: size on the device that ends up with the
            # remainder, not on the average.
            if calibration is not None:
                busiest = max(
                    calibration.per_device_gb(
                        shardable_gb=shardable_gb,
                        gpu_count=candidate,
                        layer_count=layer_count,
                    )
                ) + reserve_per_gpu
            else:
                busiest = max(
                    per_device_requirements(
                        shardable_gb=shardable_gb,
                        per_device_gb=per_device_graph + reserve_per_gpu,
                        gpu_count=candidate,
                        layer_count=layer_count,
                    )
                )
            if gib_to_bytes(busiest) <= capacity:
                return candidate
        return None
    if required_vram_gb > 0:
        count = max(
            count,
            math.ceil(required_vram_gb * 1.05 / placement.gpu_memory_gb),
        )
    if count > placement.gpu_count_max:
        return None
    return count


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
