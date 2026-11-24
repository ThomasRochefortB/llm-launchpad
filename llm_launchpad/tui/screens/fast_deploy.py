"""Model-first Deploy flow over live compute availability."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from collections.abc import Sequence
from time import monotonic

from rich.cells import set_cell_size
from rich.markup import escape

from ..format import clip, format_money
from textual.actions import SkipAction
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Input, OptionList, Select, Static
from textual.widgets.option_list import Option

from ...core.compute_availability import (
    assessed_plans_for_compute_profile,
    display_gpu_type,
    load_compute_availability,
    plans_for_compute_profile,
)
from ...core.inference_options import (
    COST_SCENARIO_WORKDAY,
    evaluate_quote_cost,
    first_hour_cost_usd,
)
from ...core.llamacpp_planner import assessment_score
from ...core.quant_quality import is_reduced_quality, quant_quality_label
from ...core.quick_deploy_refresh import (
    DEFAULT_OVERALL_MODEL_LIMIT,
)
from ...core.serving_tiers import (
    ServingTier,
    economy_plan,
    economy_sort_key,
    reduced_quality_note,
    reduced_quality_plan_note,
    serving_tiers,
)
from ...core.runtime_support import load_llamacpp_support_manifest
from ...core.vast_comparison import (
    deployable_vast_offers, vast_gpu_label, vast_offer_is_rentable,
    vast_offers_for_model, vast_plan_for_offer,
)
from ...protocol.enums import ServingObjective
from ...core.deploy_log_summary import SUMMARY_SPINNER_FRAMES
from ...core.quick_deploy import (
    QuickDeployCatalogInfo,
    QuickDeployModel,
    QuickDeployProfile,
    format_context_length,
    get_quick_deploy_catalog_info,
    list_quick_deploy_models,
    model_for_context,
    quick_deploy_recipe,
)
from ...protocol.enums import ComputeProvider
from ...protocol.models import (
    CatalogExclusion,
    ComputeAvailabilitySnapshot,
    ComputeConfiguration,
    ComputePlacement,
    InferencePlan,
    ProviderQuote,
)
from ..responsive import ViewportProfile, WidthMode
from ..widgets.fitted_footer import FittedFooter
from .copy_enabled import CopyEnabledScreen
from ..compat import highlighted_option, replace_options
from ..visual import DEPLOY_STEPS, money, screen_title
from ..widgets.fitted_option_list import FittedOptionList


class FastDeployAvailabilityLoaded(Message):
    """Live compute availability loaded for a fast deploy model."""

    def __init__(
        self,
        snapshot: ComputeAvailabilitySnapshot,
        *,
        request_id: int | None = None,
        purpose: str = "infra",
        pending_providers: tuple[ComputeProvider, ...] = (),
    ) -> None:
        super().__init__()
        self.pending_providers = pending_providers
        self.snapshot = snapshot
        self.request_id = request_id
        self.purpose = purpose


class FastDeployAvailabilityFailed(Message):
    """Live compute availability could not be loaded."""

    def __init__(
        self,
        error: str,
        *,
        request_id: int | None = None,
        purpose: str = "infra",
    ) -> None:
        super().__init__()
        self.error = error
        self.request_id = request_id
        self.purpose = purpose


_CUSTOM_DEPLOY_OPTION_ID = "fast-deploy-open-custom-deploy"
_SEARCH_ALL_GPUS_ID = "fast-deploy-search-all-gpus"


@dataclass(frozen=True)
class _InfraRow:
    """One deployable (profile, plan) pairing behind a fast-deploy row."""

    profile: QuickDeployProfile
    plan: InferencePlan
    configuration: ComputeConfiguration
    alternative_plans: tuple[InferencePlan, ...]

    def with_selected_quote(self, quote_id: str) -> _InfraRow:
        if self.plan.quote.id != quote_id:
            for alternative in self.alternative_plans:
                if alternative.quote.id == quote_id:
                    return replace(self, plan=alternative)
        return self


def _format_price(value: float | None, *, estimate: bool = False) -> str:
    if value is None:
        return "price n/a"
    prefix = "~" if estimate else ""
    return f"{prefix}${value:.2f}/hr"


def _one_time_label(quote: ProviderQuote) -> str:
    """Name a per-deployment charge, since ranking counts it and the rate does not show it."""
    once = quote.one_time_cost_usd
    if once is None:
        return "download price n/a"
    return f"+{format_money(once)} download" if once >= 0.005 else ""


def _plan_cost_evaluation(plan: object):
    """Canonical scenario cost; legacy plans without one recompute it."""
    evaluation = getattr(plan, "cost", None)
    if evaluation is not None:
        return evaluation
    quote = getattr(plan, "quote", None)
    if quote is None:
        raise ValueError("Plan has no quote for cost evaluation.")
    return evaluate_quote_cost(quote, COST_SCENARIO_WORKDAY)


def _scenario_monthly_label(plan: object) -> str:
    """Explicit-scenario cost from the plan's canonical evaluation."""
    try:
        evaluation = _plan_cost_evaluation(plan)
    except Exception:
        return "monthly est. n/a"
    value = evaluation.estimated_monthly_cost_usd
    if value is None:
        return "monthly est. n/a"
    return f"~${value:,.0f}/mo {evaluation.scenario.display_name}"


def _compact_quant_label(quant: str) -> str:
    cleaned = quant.strip().strip("()")
    if not cleaned:
        return ""
    return (
        cleaned.replace("UD-", "")
        .replace("_K_XL", "XL")
        .replace("_K_M", "M")
        .replace("_K_S", "S")
        .replace("_", "")
        .replace("-", "")
    )


def _quant_markup(profile: QuickDeployProfile, *, width: int = 0) -> str:
    quant = _compact_quant_label(profile.quant)
    if not quant:
        return " " * width if width else ""
    padded = f"{quant:<{width}}" if width else quant
    if is_reduced_quality(profile.quant):
        # Green here used to mark the smallest quantization, which read as a
        # recommendation for the row that quietly costs the most quality.
        return f"[$warning]{escape(padded)}[/$warning]"
    return f"[dim]{escape(padded)}[/dim]"


def _short_context(tokens: int) -> str:
    """Compact context label (e.g. 256K ctx instead of 262,144 ctx)."""
    if tokens >= 1024 and tokens % 1024 == 0:
        return f"{tokens // 1024}K ctx"
    if tokens >= 1000:
        return f"{round(tokens / 1024)}K ctx"
    return format_context_length(tokens)


# Marks a model whose GGUF carries native MTP heads the pinned runtime can
# use. A one-cell glyph plus a space fills exactly the two-column gutter each
# model row already reserves. The bolt emoji (U+26A1) used to fill it alone,
# but it is emoji-presented: terminals disagree about how to draw it, and
# Termius on iOS drew it over the list's edge. "↯" is East Asian Width "N" --
# one cell everywhere, never an emoji -- and still reads as a lightning bolt.
MTP_GLYPH = "↯"
MTP_MARKER = f"{MTP_GLYPH} "
MODEL_ROW_GUTTER = "  "


def _model_supports_mtp(model: QuickDeployModel) -> bool:
    """Whether any recipe for this model resolved native MTP.

    False also covers "not probed yet": the catalog attaches recommendations
    on a trailing pass, so an unmarked row is never a claim that the model
    cannot speculate, only that nothing has established that it can.
    """

    return any(profile.speculative_decoding is not None for profile in model.profiles)


def _model_row_gutter(model: QuickDeployModel) -> str:
    return MTP_MARKER if _model_supports_mtp(model) else MODEL_ROW_GUTTER


def _format_aa_index(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _capacity_label(count: int, adjective: str, color: str) -> str:
    noun = "placement" if count == 1 else "placements"
    return f"[{color}]{count} {adjective} {noun}[/]"


def _availability_label(configuration: ComputeConfiguration) -> str:
    parts = []
    if configuration.live_placement_count:
        parts.append(_capacity_label(configuration.live_placement_count, "live", "$success"))
    if configuration.has_on_demand_capacity:
        parts.append("[$warning]on demand[/$warning]")
    if configuration.spot_placement_count:
        parts.append(_capacity_label(configuration.spot_placement_count, "spot", "$accent"))
    return " + ".join(parts) or "[dim]availability unknown[/dim]"


def _subtitle(info: QuickDeployCatalogInfo) -> str:
    """Name the catalog source only when it is not the live one.

    The live source label ran to a full sentence of provenance on every visit;
    it only changes what the reader should trust when the catalog is loading,
    cached, or a fallback.
    """
    if info.is_live and not info.is_fallback:
        return ""
    return escape(info.source_label)


def _quote_price_label(quote: ProviderQuote) -> str:
    hourly = _format_price(quote.price_per_hour_usd, estimate=quote.is_estimate)
    first = first_hour_cost_usd(quote)
    if quote.one_time_cost_usd is None:
        return f"{hourly} · first hour unknown"
    if first is not None and quote.one_time_cost_usd:
        return f"{hourly} · ${first:.2f} first hour"
    return hourly


def _model_cost_label(model: QuickDeployModel, snapshot: ComputeAvailabilitySnapshot | None = None,
                      gpu_type: str = "any", pricing_pending: bool = False) -> str:
    if snapshot is None:
        return "pricing…" if pricing_pending else "price n/a"
    rows = _filter_infra_rows(infra_rows_for_model(model, snapshot), gpu_type)
    plan = economy_plan([row.plan for row in rows])
    return _quote_price_label(plan.quote) if plan else "no priced 4-bit+ fit"


def _cost_markup(cost: str) -> str:
    """Colour a price as money; leave "pricing…" and "price n/a" dim."""
    return money(cost) if "$" in cost else f"[dim]{cost}[/dim]"


def _model_option(
    model: QuickDeployModel,
    width_mode: WidthMode = WidthMode.WIDE,
    snapshot: ComputeAvailabilitySnapshot | None = None,
    gpu_type: str = "any",
    pricing_pending: bool = False,
    cost_label: str | None = None,
) -> str:
    cost = cost_label or _model_cost_label(model, snapshot, gpu_type, pricing_pending)
    # Without an Artificial Analysis key nothing is ranked, and the word
    # "unranked" then repeats down every row of the list saying nothing. An
    # absent score reads as absent; the detail panel names it for the one
    # model the reader has actually selected.
    gutter = _model_row_gutter(model)
    name_width = {
        WidthMode.MINIMAL: 20, WidthMode.COMPACT: 28,
        WidthMode.STANDARD: 34, WidthMode.WIDE: 50,
    }[width_mode]
    if "first hour" in cost and width_mode != WidthMode.WIDE:
        name_width = max(12, name_width - 15)
    name = escape(set_cell_size(clip(model.display_name, name_width), name_width))
    price = _cost_markup(cost)
    if width_mode == WidthMode.MINIMAL:
        return f"{gutter}{name}  {price}"
    score = _model_score_segment(model)
    prefix = "from " if width_mode != WidthMode.COMPACT and "$" in cost else ""
    return f"{gutter}{name}  [dim]{score:>14}{prefix}[/dim]{price}"


def _model_score_segment(model: QuickDeployModel) -> str:
    """`"score 42 · "`, or nothing at all when the model carries no score."""
    if model.quality_score is None:
        return ""
    return f"score {_format_aa_index(model.quality_score)} · "


def _unique_quant_labels(model: QuickDeployModel) -> tuple[str, ...]:
    labels: list[str] = []
    seen: set[str] = set()
    for recipe in model.recipes:
        quant = (recipe.quant or "").strip()
        key = quant.casefold()
        if quant and key not in seen:
            seen.add(key)
            labels.append(quant)
    if labels:
        return tuple(labels)
    for profile in model.profiles:
        quant = (profile.quant or "").strip()
        key = quant.casefold()
        if quant and key not in seen:
            seen.add(key)
            labels.append(quant)
    return tuple(labels)


def _quant_options_label(model: QuickDeployModel) -> str:
    labels = _unique_quant_labels(model)
    count = len(labels) or len(model.recipes) or len(model.profiles)
    if labels:
        return f"{count} quant option{'s' if count != 1 else ''}"
    return f"{count} option{'s' if count != 1 else ''}"


def _model_detail(model: QuickDeployModel) -> str:
    runtime_note = (
        "\n[dim]First launch builds a model-compatible runtime and may take longer.[/]"
        if load_llamacpp_support_manifest(model.profiles[0].gguf_architecture).build_recipe else ""
    )
    # Names the bolt in the list for the one model the reader has selected,
    # so the marker does not have to be guessed at from the rows alone.
    mtp_note = (
        f" · {MTP_GLYPH} native MTP"
        if _model_supports_mtp(model)
        else ""
    )
    profile = model.profiles[0]
    sizes = [f"{n:g}B {label}" for n, label in ((profile.parameter_count_b, "total"),
             (profile.active_parameter_count_b, "active")) if n is not None]
    meta_parts: list[str] = []
    if model.quality_score is None:
        meta_parts.append("unranked")
    if sizes:
        meta_parts.append(" / ".join(sizes))
    meta_parts.append(_quant_options_label(model))
    repositories = ", ".join(dict.fromkeys(p.repo_id for p in model.profiles if p.repo_id))
    source_line = f"[dim]{escape(repositories)}[/dim]\n" if repositories else ""
    return (
        f"[bold]{escape(model.display_name)}[/bold]  "
        f"[dim]{escape(_short_context(model.max_context_tokens))}[/dim]\n"
        f"{source_line}"
        f"[dim]{' · '.join(meta_parts)}{mtp_note}[/dim]{runtime_note}"
    )


def representative_profiles_for_model(
    model: QuickDeployModel,
) -> tuple[QuickDeployProfile, ...]:
    """Return one catalog profile per unique inference recipe."""

    unique: list[QuickDeployProfile] = []
    seen: set[str] = set()
    for profile in model.profiles:
        recipe_id = quick_deploy_recipe(profile).id
        if recipe_id in seen:
            continue
        seen.add(recipe_id)
        unique.append(profile)
    return tuple(unique)


NO_PRICE_EXCLUSION = "publishes no hourly price"


def infra_rows_for_model(
    model: QuickDeployModel,
    snapshot: ComputeAvailabilitySnapshot,
    *,
    rejected: list[str] | None = None,
    selected_quote_id: str | None = None,
) -> tuple[_InfraRow, ...]:
    """Build unique live placements for a model, cheapest first.

    Rows are one GPU type × quant × provider. Extra regions for the same
    shape stay attached as fulfillment alternatives instead of duplicate
    list entries.
    """

    rows: list[_InfraRow] = []
    seen: set[tuple[str, str, str, int]] = set()
    profiles_for_context = [
        p for p in representative_profiles_for_model(model) if p.max_context_tokens == model.max_context_tokens
    ]
    for configuration in snapshot.configurations:
        for profile in profiles_for_context:
            grouped: dict[tuple[str, int], list[InferencePlan]] = {}
            plans, _ = assessed_plans_for_compute_profile(
                configuration, profile, rejected=rejected
            )
            for plan in plans:
                # An unpublished hourly rate cannot be approved or revalidated.
                if plan.quote.price_per_hour_usd is None:
                    if rejected is not None:
                        rejected.append(
                            f"{configuration.gpu_type}: "
                            f"{plan.quote.provider.display_name} {NO_PRICE_EXCLUSION}"
                        )
                    continue
                grouped.setdefault((plan.quote.provider.value, plan.quote.gpu_count), []).append(plan)
            for provider_plans in grouped.values():
                best = min(provider_plans, key=economy_sort_key)
                key = (configuration.id, best.recipe.id, best.quote.provider.value, best.quote.gpu_count)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    _InfraRow(
                        profile=profile,
                        plan=best,
                        configuration=configuration,
                        alternative_plans=tuple(provider_plans),
                    )
                )
    profiles = {quick_deploy_recipe(profile).id: profile for profile in model.profiles}
    for comparison in vast_offers_for_model(model, snapshot.vast_offers):
        profile = profiles[comparison.recipe.id]
        plan = vast_plan_for_offer(comparison, profile)
        if plan is None:
            continue
        quote = plan.quote
        placement = ComputePlacement(
            id=quote.id, provider=quote.provider, provider_reference=quote.provider_reference,
            gpu_type=quote.gpu_type, gpu_memory_gb=comparison.offer.gpu_memory_gib,
            gpu_count_min=comparison.offer.gpu_count, gpu_count_max=comparison.offer.gpu_count,
            price_per_hour_usd=quote.price_per_hour_usd,
            billing_model=quote.billing_model, availability=quote.availability,
            supported_backends=frozenset({profile.backend}), region=quote.region,
            provider_options=quote.provider_options,
        )
        configuration = ComputeConfiguration(
            id="vast:" + comparison.gpu_label, gpu_type=comparison.gpu_label,
            gpu_memory_gb=comparison.offer.gpu_memory_gib, placements=(placement,),
        )
        rows.append(_InfraRow(profile, plan, configuration, (plan,)))
    if selected_quote_id is not None:
        rows = [row.with_selected_quote(selected_quote_id) for row in rows]
    rows.sort(key=_infra_sort_key)
    return tuple(rows)


@dataclass(frozen=True)
class _ModelPrices:
    model: QuickDeployModel
    rows: tuple[_InfraRow, ...]
    rejected: tuple[str, ...]
    best_by_gpu: dict[str, InferencePlan | None]


def _price_model(model: QuickDeployModel, snapshot: ComputeAvailabilitySnapshot) -> _ModelPrices:
    rejected: list[str] = []
    rows = infra_rows_for_model(model, snapshot, rejected=rejected)
    by_gpu: dict[str, list[InferencePlan]] = {}
    for row in rows:
        by_gpu.setdefault(_infra_gpu_label(row), []).append(row.plan)
    best = {gpu: economy_plan(plans) for gpu, plans in by_gpu.items()}
    best["any"] = min((plan for plan in best.values() if plan is not None), key=economy_sort_key, default=None)
    return _ModelPrices(model, rows, tuple(rejected), best)


def _price_shorter_contexts(model: QuickDeployModel, snapshot: ComputeAvailabilitySnapshot) -> tuple[_ModelPrices, ...]:
    return tuple(
        _price_model(variant, snapshot)
        for tokens in (32768, 8192)
        if tokens < model.max_context_tokens and (variant := model_for_context(model, tokens)) is not None
    )


def _infra_gpu_label(row: _InfraRow) -> str:
    return row.configuration.gpu_type or display_gpu_type(row.plan.quote.gpu_type)


def _tier_quality_width(tiers: Sequence[ServingTier]) -> int:
    """Width of the bit-width column, or zero when no tier needs one.

    Reserving the column unconditionally would indent every row of every
    model to make room for a disclosure most of them do not carry.
    """

    return max(
        (
            len(quant_quality_label(tier.plan.recipe.quant))
            for tier in tiers
            if is_reduced_quality(tier.plan.recipe.quant)
        ),
        default=0,
    )


def _tier_option(
    tier: ServingTier,
    width_mode: WidthMode = WidthMode.WIDE,
    quality_width: int = 0,
) -> str:
    """One tier line: what it costs, how fast it is, what it trades away.

    Hardware is deliberately absent. The user picked a model, not a data
    centre; the GPU belongs in the detail pane.
    """

    marker = "[$primary]*[/]" if tier.is_recommended else " "
    quote = tier.plan.quote
    price = _format_price(quote.price_per_hour_usd, estimate=quote.is_estimate)
    speed = tier.output_tokens_per_second
    speed_text = f"~{speed:.0f} tok/s" if speed > 0 else "speed n/a"
    # A merged role now occupies the tradeoff column on the recommended row,
    # where the bare word "recommended" used to sit alone. Both belong there.
    note = " · ".join(
        text for text in (_one_time_label(tier.plan.quote), tier.tradeoff or "") if text
    )
    if tier.is_recommended:
        note = f"{note} · recommended" if note else "recommended"
    # The tradeoff clause is the first thing a narrow terminal drops, and it is
    # what carries the quality disclosure. The bit width rides beside the label
    # instead, in a column every row shares so the prices stay aligned.
    quality = ""
    if quality_width:
        label = (
            quant_quality_label(tier.plan.recipe.quant)
            if is_reduced_quality(tier.plan.recipe.quant)
            else ""
        )
        padded = f"{escape(label):<{quality_width}}"
        quality = f" [$warning]{padded}[/$warning]" if label else f" {padded}"
    if width_mode == WidthMode.MINIMAL:
        return f" {marker} {escape(tier.label):<9}{quality} {money(price)}"
    if width_mode in (WidthMode.COMPACT, WidthMode.STANDARD):
        return (
            f" {marker} {escape(tier.label):<9}{quality} "
            f"{money(price)} [dim]· {speed_text}[/dim]"
        )
    return (
        f" {marker} {escape(tier.label):<10}{quality} "
        f"{money(f'{price:<12}')} {speed_text:<14} "
        f"[dim]{escape(note)}[/dim]"
    )


def _infra_option(row: _InfraRow, width_mode: WidthMode = WidthMode.WIDE) -> str:
    quote = row.plan.quote
    gpu = _infra_gpu_label(row)
    price = _format_price(quote.price_per_hour_usd, estimate=quote.is_estimate)
    provider = escape(quote.provider.display_name)
    if width_mode == WidthMode.MINIMAL:
        quant = _quant_markup(row.profile)
        return (
            f"  {escape(clip(gpu, 14))} x{quote.gpu_count} "
            f"{quant} [dim]{provider}[/dim] {money(price)}"
        )
    quant = _quant_markup(row.profile, width=5)
    if width_mode == WidthMode.COMPACT:
        return (
            f"  {escape(clip(gpu, 16)):<16} x{quote.gpu_count:<2} "
            f"{quant} {provider:<15} {money(f'{price:>10}')}"
        )
    if width_mode == WidthMode.STANDARD:
        return (
            f"  {escape(clip(gpu, 22)):<22} x{quote.gpu_count:<2} "
            f"{quant} {provider:<15} {money(f'{price:>10}')}"
        )
    return (
        f"  {escape(clip(gpu, 22)):<22} x{quote.gpu_count:<2} "
        f"{quant} {provider:<15} {money(f'{price:>10}')}  "
        f"{_availability_label(row.configuration)}"
    )


def _infra_detail(row: _InfraRow) -> str:
    quote = row.plan.quote
    monthly = _scenario_monthly_label(row.plan)
    region = escape((quote.region or "").strip() or "provider-managed regions")
    basis = escape(_plan_cost_evaluation(row.plan).basis_label().rstrip("."))
    performance = ""
    if row.plan.assessment is not None:
        single = max(
            (
                point.output_tokens_per_second or 0.0
                for point in row.plan.assessment.performance
                if point.concurrency == 1
            ),
            default=0.0,
        )
        aggregate = max(
            (
                point.aggregate_output_tokens_per_second or 0.0
                for point in row.plan.assessment.performance
            ),
            default=0.0,
        )
        if single > 0:
            evidence = (
                "measured"
                if any(point.measured for point in row.plan.assessment.performance)
                else "estimated"
            )
            performance = (
                f"~{single:.0f} tok/s single · ~{aggregate:.0f} aggregate ({evidence}) · "
            )
    once = _one_time_label(quote)
    return (
        f"[bold]{escape(monthly)}[/bold]"
        f"{f' · {escape(once)}' if once else ''}  "
        f"[dim]{performance}{region} · {_availability_label(row.configuration)}[/dim]\n"
        f"[dim]{basis} · storage separate[/dim]"
    )


def _infra_sort_key(row: _InfraRow) -> tuple[int, float, str, int]:
    price = first_hour_cost_usd(row.plan.quote)
    requirements = row.plan.recipe.serving_requirements
    if row.plan.assessment is not None and requirements is not None:
        score = assessment_score(row.plan.assessment, requirements.objective)
        return (
            0,
            -score,
            row.plan.quote.gpu_type.casefold(),
            row.plan.quote.gpu_count,
        )
    return (
        2 if price is None else 1,
        price if price is not None else float("inf"),
        row.plan.quote.gpu_type.casefold(),
        row.plan.quote.gpu_count,
    )


def _gpu_filter_options(snapshot: ComputeAvailabilitySnapshot) -> list[tuple[str, str]]:
    """Filter by hardware family. A model and GPU count are needed to quote a price."""
    names = {row.gpu_type for row in snapshot.configurations if row.gpu_type}
    names.update(vast_gpu_label(offer) for offer in snapshot.vast_offers if vast_offer_is_rentable(offer))
    return [("Any GPU", "any"), *((name, name) for name in sorted(names, key=str.casefold))]


def _model_fits_gpu_type(
    model: QuickDeployModel,
    snapshot: ComputeAvailabilitySnapshot,
    gpu_type: str,
) -> bool:
    """Return True when the model has at least one fulfillment on ``gpu_type``."""
    configurations = [row for row in snapshot.configurations if row.gpu_type == gpu_type]
    for profile in representative_profiles_for_model(model):
        if any(plans_for_compute_profile(configuration, profile) for configuration in configurations):
            return True
    return any(row.gpu_label == gpu_type for row in deployable_vast_offers(model, snapshot.vast_offers))


def _model_matches_query(model: QuickDeployModel, terms: Sequence[str]) -> bool:
    """Match every search term across the model's names and repositories."""
    names = {model.display_name, model.id}
    for profile in model.profiles:
        names.update(name for name in (profile.repo_id, profile.model_name, profile.aa_model_name) if name)
    return _names_match(names, terms)


def _names_match(names: Sequence[str] | set[str], terms: Sequence[str]) -> bool:
    def normalize(value: str) -> str:
        return "".join(c for c in value.casefold() if c.isalnum())

    keys = [normalize(name) for name in names]
    return all(normalize(term) and any(normalize(term) in key for key in keys) for term in terms)


def _filter_infra_rows(rows: tuple[_InfraRow, ...], gpu_type: str) -> tuple[_InfraRow, ...]:
    if not gpu_type or gpu_type == "any":
        return rows
    return tuple(
        row
        for row in rows
        if row.configuration.gpu_type == gpu_type or row.plan.quote.gpu_type == gpu_type
    )


class CatalogExclusionsScreen(CopyEnabledScreen):
    """Explain why discovered models were omitted from the shortlist."""

    BINDINGS = [Binding("escape", "pop_screen", "Back")]

    def __init__(self, exclusions: tuple[CatalogExclusion, ...]) -> None:
        super().__init__()
        self.exclusions = exclusions

    def compose(self) -> ComposeResult:
        with Vertical(id="fast-deploy-container"):
            yield Static(screen_title("Excluded models"), id="fast-deploy-title")
            yield Static(
                "Model availability from the latest catalog check.",
                id="fast-deploy-subtitle",
            )
            yield FittedOptionList(
                *(Option(escape(row.display_name), id=str(i)) for i, row in enumerate(self.exclusions)),
                id="fast-deploy-list",
            )
            yield Static("", id="fast-deploy-detail")
        yield FittedFooter()

    def on_mount(self) -> None:
        self.query_one(OptionList).focus()

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        row = self.exclusions[int(event.option.id or "0")]
        self.query_one("#fast-deploy-detail", Static).update(
            f"[bold]{escape(row.display_name)}[/]\n{escape(row.reason)}\n[dim]{escape(row.repo_id)}[/]"
        )

    def action_pop_screen(self) -> None:
        self.app.pop_screen()


class FastDeployScreen(CopyEnabledScreen):
    """Pick a catalog model, filter by GPU, then confirm a live placement."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        # Footer order is by FittedFooter's presentation priority: Choose,
        # Search and Back must survive narrow terminals ahead of view and
        # refresh keys, which remain reachable from help and their shortcuts.
        Binding("enter", "choose_selected", "Choose", show=True, priority=True),
        Binding("/", "focus_model_search", "Search", show=True, priority=True),
        Binding("down", "focus_model_results", "Browse results", show=True, priority=True),
        # The GPU filter and the exclusions count both show their shortcuts.
        Binding("g", "focus_gpu_filter", "GPU filter", show=False),
        Binding("v", "toggle_model_view", "All models", show=True),
        Binding("c", "cycle_context", "Context", show=True),
        Binding("a", "toggle_all_placements", "Compare all", show=True),
        Binding("x", "show_excluded_models", "Excluded models", show=False),
        Binding("r", "refresh_availability", "Refresh offers", show=False),
    ]
    # Half of those keys belong to one step only. Leaving them live on the
    # other one advertised a search box that step two does not have, and an
    # "a" that quietly did nothing on step one.
    _PHASE_ACTIONS = {
        "focus_model_search": {"models"},
        "toggle_model_view": {"models"},
        "show_excluded_models": {"models"},
        "cycle_context": {"infra"},
        "toggle_all_placements": {"infra"},
    }

    @property
    def _phase(self) -> str:
        return self._phase_name

    @_phase.setter
    def _phase(self, value: str) -> None:
        if getattr(self, "_phase_name", None) == value:
            return
        self._phase_name = value
        if self.is_mounted:
            self.set_class(value == "infra", "placement-step")
            self.set_class(value == "loading", "placement-loading")
            self.refresh_bindings()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Hide and disable the keys that do not apply to the current step."""
        if action == "choose_selected" and (
            self._phase == "loading" or (self._phase == "infra" and not self._infra_rows)
        ):
            return False
        editing_search = getattr(self.focused, "id", None) == "fast-deploy-model-search"
        if action == "cycle_context" and (self._snapshot is None or (len(self._contexts) < 2 and self._context_tokens in (None, *self._contexts))):
            return False
        if action == "toggle_all_placements" and not self._show_all_placements and not self._has_tiers:
            return False
        if action == "focus_model_results":
            return self._phase == "models" and editing_search
        if action == "focus_model_search" and editing_search:
            return False  # A slash in a repository name is ordinary input.
        phases = self._PHASE_ACTIONS.get(action)
        if phases is not None and self._phase not in phases:
            return False
        return super().check_action(action, parameters)

    def __init__(
        self,
        initial_model_id: str | None = None,
        preferred_placement: tuple[str, str, int] | None = None,
        preferred_context: int | None = None,
    ) -> None:
        """``initial_model_id`` opens straight on step two for that model.

        ``preferred_placement`` is a (provider, GPU type, GPU count) shape to
        highlight there, so a relaunch lands on what was used last time --
        re-priced live, never replayed from a stale quote.
        """
        super().__init__()
        self._initial_model_id = initial_model_id
        self._context_tokens = preferred_context
        self._contexts: list[int] = []
        self._blocked: dict[str, CatalogExclusion] = {}
        self._prices: dict[str, _ModelPrices] = {}
        self._shorter_contexts: dict[str, tuple[_ModelPrices, ...]] = {}
        self._fetch_pending = False
        self._pending_providers: tuple[ComputeProvider, ...] = ()
        self._pricing_error: str | None = None
        self._preferred_placement = preferred_placement
        self._placement_id: str | None = None
        # Step two offers named tiers by default. The full placement list is a
        # detail view: picking between "two L40S" and "one A100" is a question
        # almost nobody can answer, while cheap-or-fast is one everybody can.
        self._show_all_placements = False
        self._has_tiers = False
        self._catalog_info = QuickDeployCatalogInfo("Loading catalog", ready=False)
        self._models: tuple[QuickDeployModel, ...] = ()
        self._model_by_id: dict[str, QuickDeployModel] = {}
        self._selected_model: QuickDeployModel | None = None
        self._infra_rows: dict[str, _InfraRow] = {}
        self._availability_inflight = False
        self._availability_request_id = 0
        self._phase = "models"
        self._snapshot: ComputeAvailabilitySnapshot | None = None
        # First paint prices models from the cached catalog, which holds a
        # Modal list rate for the shape the planner picked when the catalog
        # was built. Live placements routinely come in several times cheaper,
        # so the provisional number has to say that it is one.
        self._pricing_pending = False
        self._gpu_filter = "any"
        self._gpu_choices = [("Any GPU", "any")]
        self._model_search = ""
        self._show_top_models = True
        self._updating_gpu_filter = False
        self._catalog_building = False
        self._catalog_spinner_index = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="fast-deploy-container"):
            yield Static(
                screen_title("Deploy", steps=DEPLOY_STEPS, current=0),
                id="fast-deploy-title",
            )
            subtitle = _subtitle(self._catalog_info)
            yield Static(
                subtitle,
                id="fast-deploy-subtitle",
                classes="" if subtitle else "hidden",
            )
            # Search stays visible at every supported size: it is the primary
            # way to reach a model, while the GPU filter narrows what is
            # already listed. Both ride one row where they fit; the filter
            # drops to its own line on constrained terminals.
            with Horizontal(id="fast-deploy-filter-row"):
                with Vertical(id="fast-deploy-search-group"):
                    yield Static("[dim]Search  /[/dim]", id="fast-deploy-search-label")
                    yield Input(
                        placeholder="Model name or repository",
                        id="fast-deploy-model-search",
                    )
                    yield Select([("Full context", 0)], value=0, allow_blank=False, classes="hidden", id="fast-deploy-context")
                with Vertical(id="fast-deploy-gpu-group"):
                    yield Static("[dim]GPU filter  g[/dim]", id="fast-deploy-gpu-label")
                    yield Select(
                        options=[("Any GPU", "any")],
                        value="any",
                        allow_blank=False,
                        id="fast-deploy-gpu-filter",
                    )
            yield Static("[dim]Loading models...[/dim]", id="fast-deploy-status")
            yield FittedOptionList(id="fast-deploy-list")
            yield Static("", id="fast-deploy-detail")
        yield FittedFooter()

    def _set_subtitle(self, text: str) -> None:
        subtitle = self.query_one("#fast-deploy-subtitle", Static)
        subtitle.update(text)
        # The class, not `display`: an inline display would override the
        # stylesheet rule that hides the subtitle on short terminals.
        subtitle.set_class(not text, "hidden")

    def on_mount(self) -> None:
        self._availability_inflight = False
        self._apply_model_catalog(force=True)
        self.query_one("#fast-deploy-list", OptionList).focus()
        self._load_availability(purpose="filter")
        self.set_interval(
            0.25,
            self._tick_catalog_spinner,
            name="fast-deploy-catalog-spinner",
        )

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_list.id != "fast-deploy-list":
            return
        option_id = str(event.option.id)
        if highlighted_option(event.option_list) is not event.option:
            return
        if self._phase == "models":
            if option_id in self._blocked:
                self._show_blocker(option_id)
                return
            model = self._model_by_id.get(option_id)
            if model is not None:
                self._update_model_detail(model)
        elif self._phase == "infra":
            row = self._infra_rows.get(option_id)
            if row is not None:
                self._update_infra_detail(row)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "fast-deploy-list":
            self._choose(str(event.option.id))

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "fast-deploy-model-search":
            return
        self._model_search = event.value
        if self._phase == "models":
            self._render_model_list(preferred_id=self._highlighted_model_id())

    def _goto_custom_deploy(self) -> None:
        app = self.app
        popper = getattr(app, "pop_to_main_menu", None)
        if callable(popper):
            popper()
        else:
            try:
                app.pop_screen()
            except Exception:
                pass
        pusher = getattr(app, "action_push_custom_deploy", None)
        if callable(pusher):
            pusher()

    def _render_catalog_unavailable(self, option_list: OptionList) -> None:
        info = self._catalog_info
        self._set_subtitle(_subtitle(info))
        self.query_one("#fast-deploy-detail", Static).update("")
        if info.error:
            self._catalog_building = False
            replace_options(option_list, 
                [
                    Option(
                        "  Open Advanced deploy to configure a model manually",
                        id=_CUSTOM_DEPLOY_OPTION_ID,
                    )
                ]
            )
            option_list.highlighted = 0
            status = (
                "[$warning]Live model catalog unavailable.[/$warning]\n"
                "[dim]Enter opens Advanced deploy for manual setup; "
                "press Esc to go back.[/dim]\n"
                f"[dim]{escape(clip(info.error, 140))}[/dim]"
            )
        else:
            self._catalog_building = True
            frame = SUMMARY_SPINNER_FRAMES[self._catalog_spinner_index]
            replace_options(option_list, 
                [Option(f"  {frame} Building the live model catalog…", disabled=True)]
            )
            status = (
                f"[bold $primary]{frame}[/][dim] Building the live model catalog "
                "from Artificial Analysis and Hugging Face…[/dim]"
            )
        self.query_one("#fast-deploy-status", Static).update(status)
        option_list.focus()

    def _tick_catalog_spinner(self) -> None:
        """Advance the building-catalog spinner like deploy-log in-progress steps."""
        if not self._catalog_building:
            return
        self._catalog_spinner_index = (
            self._catalog_spinner_index + 1
        ) % len(SUMMARY_SPINNER_FRAMES)
        frame = SUMMARY_SPINNER_FRAMES[self._catalog_spinner_index]
        try:
            option_list = self.query_one("#fast-deploy-list", OptionList)
            replace_options(option_list, 
                [Option(f"  {frame} Building the live model catalog…", disabled=True)]
            )
            self.query_one("#fast-deploy-status", Static).update(
                f"[bold $primary]{frame}[/][dim] Building the live model catalog "
                "from Artificial Analysis and Hugging Face…[/dim]"
            )
        except Exception:
            return

    def _choose(self, option_id: str) -> None:
        if option_id == _CUSTOM_DEPLOY_OPTION_ID:
            self._goto_custom_deploy()
            return
        if self._phase == "loading":
            return
        if self._phase == "models":
            if option_id == _SEARCH_ALL_GPUS_ID:
                self.query_one("#fast-deploy-gpu-filter", Select).value = "any"
                return
            if option_id in self._blocked:
                self.app.push_screen(CatalogExclusionsScreen((self._blocked[option_id],)))
                return
            self._open_model(option_id)
            return
        if self._availability_inflight:
            return
        if self._phase == "infra":
            row = self._infra_rows.get(option_id)
            if row is not None:
                alternatives: list[InferencePlan] = []
                seen_quotes: set[str] = set()
                for candidate in self._infra_rows.values():
                    if candidate.plan.recipe.id != row.plan.recipe.id:
                        continue
                    for plan in (candidate.plan, *candidate.alternative_plans):
                        if plan.quote.id in seen_quotes:
                            continue
                        seen_quotes.add(plan.quote.id)
                        alternatives.append(plan)
                alternatives.sort(
                    key=lambda plan: (
                        plan.quote.id != row.plan.quote.id,
                        first_hour_cost_usd(plan.quote) is None,
                        first_hour_cost_usd(plan.quote) or float("inf"),
                    )
                )
                self.app.push_quick_deploy(
                    row.plan,
                    alternative_plans=tuple(alternatives),
                    catalog_profile=row.profile,
                )
            return
    def _open_model(self, option_id: str) -> None:
        model = self._model_by_id.get(option_id)
        if model is None or self._availability_inflight:
            return
        # Opening a model leaves the building state; the initial-model path
        # never renders the model list that would otherwise stop the spinner.
        self._catalog_building = False
        self._selected_model = model
        self._placement_id = None
        # Hide the whole group: hiding only the input left its "Search" label
        # above an empty gap on the placement step.
        self.query_one("#fast-deploy-search-group").add_class("hidden")
        self.query_one("#fast-deploy-model-search").add_class("hidden")
        self.query_one("#fast-deploy-search-label", Static).update("[dim]Context[/dim]")
        self.query_one("#fast-deploy-title", Static).update(
            screen_title("Deploy", model.display_name, steps=DEPLOY_STEPS, current=1)
        )
        self._show_placement_loading()
        self.query_one("#fast-deploy-list", OptionList).focus()
        if self._snapshot is not None:
            self._prepare_prices(self._snapshot)
        elif not self._fetch_pending:
            self._load_availability(purpose="infra")

    def _offer_wait_label(self) -> str:
        providers = ", ".join(provider.display_name for provider in self._pending_providers)
        return f"Checking offers: {providers}…" if providers else "Checking live offers…"

    def _show_placement_loading(self) -> None:
        self._phase = "loading"
        self._infra_rows = {}
        replace_options(self.query_one("#fast-deploy-list", OptionList), [])
        self.query_one("#fast-deploy-detail", Static).update("")
        status = self._offer_wait_label() if self._fetch_pending or self._snapshot is None else "Pricing placements…"
        self.query_one("#fast-deploy-status", Static).update(f"[dim]{status}[/dim]")

    def _load_availability(self, *, purpose: str, force_refresh: bool = False) -> None:
        self._availability_request_id += 1
        request_id = self._availability_request_id
        self._fetch_pending = True
        self._pending_providers = ()
        self._snapshot = None
        self._pricing_error = None
        self._pricing_pending = True
        self._availability_inflight = purpose == "infra"
        self.workers.cancel_group(self, "fast-deploy-prices")

        def run_load() -> None:
            self._run_load_availability(request_id, purpose=purpose, force_refresh=force_refresh)

        self.run_worker(
            run_load, name="fast-deploy-availability-worker",
            thread=True, group="fast-deploy-fetch", exclusive=True,
        )

    def _run_load_availability(self, request_id: int, *, purpose: str = "infra", force_refresh: bool = False) -> None:
        def publish(
            snapshot: ComputeAvailabilitySnapshot,
            pending: tuple[ComputeProvider, ...] = (),
        ) -> None:
            self.post_message(FastDeployAvailabilityLoaded(
                snapshot,
                request_id=request_id,
                purpose=purpose,
                pending_providers=pending,
            ))

        try:
            snapshot = load_compute_availability(force_refresh=force_refresh, on_partial=publish)
        except Exception as exc:
            self.post_message(
                FastDeployAvailabilityFailed(
                    str(exc),
                    request_id=request_id,
                    purpose=purpose,
                )
            )
            return
        publish(snapshot)

    def on_fast_deploy_availability_loaded(self, message: FastDeployAvailabilityLoaded) -> None:
        if message.request_id is not None and message.request_id != self._availability_request_id:
            return
        self._pending_providers = message.pending_providers
        self._fetch_pending = bool(message.pending_providers)
        self._pricing_error = None
        if self._snapshot is not message.snapshot:
            self._gpu_choices = _gpu_filter_options(message.snapshot)
            self._prices.clear()
            self._shorter_contexts.clear()
        self._snapshot = message.snapshot
        self._populate_gpu_filter()
        if self._phase == "loading" and self._fetch_pending:
            self._show_placement_loading()
        self._prepare_prices(message.snapshot)

    def on_fast_deploy_availability_failed(self, message: FastDeployAvailabilityFailed) -> None:
        if message.request_id is not None and message.request_id != self._availability_request_id:
            return
        self._fetch_pending = False
        self._pending_providers = ()
        self.workers.cancel_group(self, "fast-deploy-prices")
        self._show_pricing_error(message.error)

    def _show_pricing_error(self, error: str) -> None:
        self._pricing_error = error
        self._snapshot = None
        self._gpu_choices = [("Any GPU", "any")]
        self._populate_gpu_filter()
        self._prices.clear()
        self._shorter_contexts.clear()
        self._pricing_pending = False
        if self._selected_model is None:
            self._render_model_list(preferred_id=self._highlighted_model_id())
        else:
            self._contexts = []
            self.query_one("#fast-deploy-context").add_class("hidden")
            self._render_unavailable_placement(error)

    def _prepare_prices(self, snapshot: ComputeAvailabilitySnapshot) -> None:
        """Price one model at a time so navigation can supersede catalog work."""
        selected = self._selected_model
        models = (selected,) if selected else self._models
        self._pricing_pending = True

        async def prepare() -> None:
            try:
                last_paint = monotonic()
                for model in models:
                    if model.id in self._prices:
                        continue
                    self._prices[model.id] = await asyncio.to_thread(_price_model, model, snapshot)
                    if not self.is_attached:
                        return
                    # Surface prices as they arrive without rebuilding the list for every row.
                    if selected is None and monotonic() - last_paint >= 0.25:
                        self._render_model_list(preferred_id=self._highlighted_model_id())
                        last_paint = monotonic()
                if selected and selected.id not in self._shorter_contexts:
                    self._shorter_contexts[selected.id] = await asyncio.to_thread(
                        _price_shorter_contexts, selected, snapshot
                    )
            except Exception as exc:
                if self.is_attached:
                    self._show_pricing_error(str(exc))
                return
            if not self.is_attached:
                return
            self._pricing_pending = False
            if selected is None:
                self._render_model_list(preferred_id=self._highlighted_model_id())
            else:
                self._render_infra_from_snapshot(selected, snapshot)

        self.run_worker(prepare, group="fast-deploy-prices", exclusive=True)

    def _tiers_for(self, rows: Sequence[_InfraRow]) -> tuple[ServingTier, ...]:
        """Collapse the visible placements into named tiers."""

        plans = [row.plan for row in rows]
        requirements = next(
            (
                plan.recipe.serving_requirements
                for plan in plans
                if plan.recipe.serving_requirements is not None
            ),
            None,
        )
        objective = (
            requirements.objective
            if requirements is not None
            else ServingObjective.GENERAL_PURPOSE
        )
        tiers = serving_tiers(plans, objective)
        # A single tier is not a choice; fall back to the full list rather than
        # presenting one option as though it were a considered selection.
        return tiers if len(tiers) > 1 else ()

    def action_toggle_all_placements(self) -> None:
        """Swap between named tiers and the full placement list."""

        if self._phase != "infra" or self._selected_model is None:
            return
        self._show_all_placements = not self._show_all_placements
        if self._snapshot is not None:
            self._render_infra_from_snapshot(self._selected_model, self._snapshot)

    def action_refresh_availability(self) -> None:
        """Reload prices and credentials, invalidating pending results."""
        if self._phase == "models":
            self._load_availability(purpose="filter", force_refresh=True)
        elif self._selected_model is not None:
            self._remember_highlighted_placement()
            self._load_availability(purpose="infra", force_refresh=True)
            self._show_placement_loading()
        self.query_one("#fast-deploy-status", Static).update("[dim]Refreshing live offers…[/dim]")

    def _render_infra_from_snapshot(
        self,
        model: QuickDeployModel,
        snapshot: ComputeAvailabilitySnapshot,
    ) -> None:
        self._remember_highlighted_placement()
        full = model
        prices = self._prices.get(full.id)
        if prices is None or full.id not in self._shorter_contexts:
            return
        full_plan = prices.best_by_gpu.get(self._gpu_filter)
        full_cost = first_hour_cost_usd(full_plan.quote) if full_plan else None
        self._contexts = [full.max_context_tokens]
        for variant in self._shorter_contexts[full.id]:
            plan = variant.best_by_gpu.get(self._gpu_filter)
            cost = first_hour_cost_usd(plan.quote) if plan else None
            tokens = variant.model.max_context_tokens
            if cost is not None and (full_cost is None or cost < full_cost or tokens == self._context_tokens):
                self._contexts.append(tokens)
                if tokens == self._context_tokens:
                    model, prices = variant.model, variant
        all_rows, excluded = prices.rows, prices.rejected
        selector = self.query_one("#fast-deploy-context", Select)
        selected = self._context_tokens or full.max_context_tokens
        choices = [(f"{n:,} tokens" if n != full.max_context_tokens else f"Full · {n:,} tokens", n)
                   for n in self._contexts]
        if selected not in self._contexts:
            choices.append((f"{selected:,} tokens · unavailable", selected))
        with selector.prevent(Select.Changed):
            selector.set_options(choices)
            selector.value = selected
        show_context = len(choices) > 1
        self.query_one("#fast-deploy-search-group").set_class(not show_context, "hidden")
        selector.set_class(not show_context, "hidden")
        self.refresh_bindings()
        if self._context_tokens and self._context_tokens not in self._contexts:
            if self._fetch_pending:
                self._show_placement_loading()
                return
            self._render_unavailable_placement("The previously chosen context has no priced placement. Press c to select full context.")
            return
        if self._placement_id is not None:
            pinned = (row.with_selected_quote(self._placement_id) for row in all_rows)
            all_rows = tuple(sorted(pinned, key=_infra_sort_key))
        rows = _filter_infra_rows(all_rows, self._gpu_filter)
        self._populate_gpu_filter(prices)
        if not rows:
            if self._fetch_pending:
                self._show_placement_loading()
                return
            if all_rows and self._gpu_filter not in {"", "any"}:
                self._render_empty_gpu_filter()
                return
            self._render_unavailable_placement(
                " ".join(("No priced placement fits at this context.", *dict.fromkeys(excluded[:2]), *snapshot.errors)),
            )
            return
        self._infra_rows = {row.plan.quote.id: row for row in rows}
        self._phase = "infra"
        self._availability_inflight = False
        self.refresh_bindings()
        width_mode = self.viewport_profile.width_mode
        option_list = self.query_one("#fast-deploy-list", OptionList)
        tiers = () if self._show_all_placements else self._tiers_for(rows)
        self._has_tiers = bool(tiers)
        self.refresh_bindings()
        if tiers:
            # A placement from Compare all can fall outside every named tier.
            if self._placement_id in self._infra_rows and all(
                tier.plan.quote.id != self._placement_id for tier in tiers
            ):
                selected = self._infra_rows[self._placement_id].plan
                tiers = (*tiers, ServingTier("selected", "Selected", selected))
            quality_width = _tier_quality_width(tiers)
            replace_options(option_list, 
                [
                    Option(
                        _tier_option(tier, width_mode, quality_width),
                        id=tier.plan.quote.id,
                    )
                    for tier in tiers
                ]
            )
            recommended = next(
                (index for index, tier in enumerate(tiers) if tier.is_recommended),
                0,
            )
            plans = [tier.plan for tier in tiers]
        else:
            replace_options(option_list, 
                [
                    Option(_infra_option(row, width_mode), id=row.plan.quote.id)
                    for row in rows
                ]
            )
            plans = [row.plan for row in rows]
            best = economy_plan(plans)
            recommended = plans.index(best) if best else -1
        default = self._take_preferred_index(plans, default=recommended)
        selected_id = self._restore_placement(option_list, default)
        if selected_id is not None:
            self._update_infra_detail(self._infra_rows[selected_id])
        self._set_subtitle(_subtitle(self._catalog_info))
        prefix = "Full " if model is full and any(p.context_known for p in model.profiles) else ""
        context = prefix + escape(format_context_length(model.max_context_tokens))
        hidden = f" · {len(excluded)} hidden" if excluded else ""
        count = f"{len(rows)} placement{'s' if len(rows) != 1 else ''}"
        status = f"[dim]{count} · {context}{hidden}[/dim]"
        # Item four of the quality floor: a model that only fits below it is
        # still worth offering, but never without saying why.
        quality_note = (
            reduced_quality_note(tiers)
            if tiers
            else reduced_quality_plan_note([row.plan for row in rows])
        )
        if len(self._contexts) > 1:
            status += " [dim]· c context[/dim]"
        if quality_note:
            status += f"\n[$warning]{escape(quality_note)}[/$warning]"
        if self._gpu_filter != "any":
            status += f" [dim]· GPU {escape(self._gpu_filter)}[/dim]"
        if self._fetch_pending:
            status += f"\n[dim]{escape(self._offer_wait_label())}[/dim]"
        if (
            snapshot.vast_configured
            and ComputeProvider.VAST not in self._pending_providers
            and not any(row.plan.quote.provider == ComputeProvider.VAST for row in rows)
        ):
            # Unrentable offers are no longer listed, so this line is the only
            # thing distinguishing "Vast has nothing for you" from "Vast is
            # misconfigured".
            status += "\n[dim]No Vast offer can serve this model.[/dim]"
        if snapshot.errors:
            status += "\n[$warning]Partial results:[/$warning] " + escape(
                "; ".join(snapshot.errors)
            )
        self.query_one("#fast-deploy-status", Static).update(status)
        if self.focused is None:
            option_list.focus()

    def _take_preferred_index(self, plans: Sequence[InferencePlan], *, default: int) -> int:
        """Index of the relaunch shape among ``plans``; applies only once."""
        preferred = self._preferred_placement
        self._preferred_placement = None
        if preferred is None:
            return default
        provider, gpu_type, gpu_count = preferred
        for index, plan in enumerate(plans):
            quote = plan.quote
            if (
                quote.provider.value == provider
                and quote.gpu_type.casefold() == gpu_type.casefold()
                and quote.gpu_count == gpu_count
            ):
                return index
        return default

    def _remember_highlighted_placement(self) -> None:
        if self._phase == "infra":
            option = highlighted_option(self.query_one("#fast-deploy-list", OptionList))
            if option is not None and option.id is not None:
                self._placement_id = option.id

    def _restore_placement(self, options: OptionList, default: int) -> str | None:
        index = default
        if self._placement_id is not None:
            for index in range(options.option_count):
                if options.get_option_at_index(index).id == self._placement_id:
                    break
            else:
                options.highlighted = None
                self.query_one("#fast-deploy-detail", Static).update(
                    "[dim]Checking the previous placement…[/]" if self._fetch_pending
                    else "[$warning]Previous placement is no longer listed. Use ↑/↓ to choose another.[/]"
                )
                return None
        if index < 0:
            options.highlighted = None
            self.query_one("#fast-deploy-detail", Static).update("No priced 4-bit+ option. Use ↑/↓ to choose a placement explicitly.")
            return None
        options.highlighted = index
        return options.get_option_at_index(index).id

    def _render_empty_gpu_filter(self) -> None:
        self._phase = "infra"
        self._availability_inflight = False
        self._infra_rows = {}
        self.refresh_bindings()
        gpu = escape(self._gpu_filter)
        option_list = self.query_one("#fast-deploy-list", OptionList)
        replace_options(option_list, 
            [Option(f"  No placements on {gpu}", disabled=True)]
        )
        self.query_one("#fast-deploy-detail", Static).update("")
        self._set_subtitle(_subtitle(self._catalog_info))
        self.query_one("#fast-deploy-status", Static).update(
            f"[$warning]No live placements on {gpu}.[/$warning] "
            "[dim]Press g to change the GPU filter.[/dim]"
        )
        option_list.focus()

    def _render_unavailable_placement(self, reason: str) -> None:
        self._phase = "infra"
        self._infra_rows = {}
        self._has_tiers = False
        self._availability_inflight = False
        self.refresh_bindings()
        options = self.query_one("#fast-deploy-list", OptionList)
        replace_options(options, [Option("  No current placement", disabled=True)])
        self.query_one("#fast-deploy-detail", Static).update(escape(reason or "Live pricing unavailable."))
        context_hint = " · c change context" if len(self._contexts) > 1 or self._context_tokens else ""
        self.query_one("#fast-deploy-status", Static).update(f"[dim]r refresh offers{context_hint}[/dim]")

    def action_cycle_context(self) -> None:
        if self._selected_model is None or self._snapshot is None or not self._contexts:
            return
        current = self._context_tokens or self._contexts[0]
        index = self._contexts.index(current) if current in self._contexts else -1
        self._context_tokens = self._contexts[(index + 1) % len(self._contexts)]
        self._placement_id = None
        self._phase = "loading"
        self._render_infra_from_snapshot(self._selected_model, self._snapshot)

    def _best(self, model: QuickDeployModel) -> InferencePlan | None:
        prices = self._prices.get(model.id)
        if prices is None or not any(p.context_known for p in model.profiles):
            return None
        return prices.best_by_gpu.get(self._gpu_filter)

    def _show_blocker(self, option_id: str) -> None:
        row = self._blocked[option_id]
        self.query_one("#fast-deploy-detail", Static).update(
            f"[bold]{escape(row.display_name)}[/] · {escape(row.repo_id)}\n{escape(row.reason)}"
        )

    def _highlighted_model_id(self) -> str | None:
        try:
            option_list = self.query_one("#fast-deploy-list", OptionList)
        except Exception:
            return None
        highlighted = highlighted_option(option_list)
        if highlighted is None or highlighted.id is None:
            return None
        return str(highlighted.id)

    def refresh_quick_deploy_catalog(self) -> None:
        """Apply a catalog activation broadcast by the app."""
        self._apply_model_catalog(force=False)

    def _apply_model_catalog(self, *, force: bool = False, preferred_id: str | None = None) -> None:
        if self._phase != "models":
            return

        def read_catalog() -> tuple[QuickDeployCatalogInfo, tuple[QuickDeployModel, ...]]:
            return get_quick_deploy_catalog_info(), list_quick_deploy_models()

        async def apply() -> None:
            info, models = await asyncio.to_thread(read_catalog)
            if not self.is_attached or self._phase != "models":
                return
            changed = models != self._models
            if not force and not changed and info == self._catalog_info:
                return
            self._catalog_info = info
            self._models = models
            self._model_by_id = {model.id: model for model in models}
            if changed:
                self._pricing_pending = self._fetch_pending or self._snapshot is not None
                self._prices.clear()
                self._shorter_contexts.clear()
            initial = self._initial_model_id
            if initial and initial in self._model_by_id:
                self._initial_model_id = None
                self._open_model(initial)
                return
            if info.ready or info.error:
                self._initial_model_id = None
                self._preferred_placement = None
            self._render_model_list(preferred_id=preferred_id or self._highlighted_model_id())
            if self._snapshot is not None:
                self._prepare_prices(self._snapshot)

        self.run_worker(apply, group="fast-deploy-catalog", exclusive=True)

    def _visible_models(self) -> tuple[QuickDeployModel, ...]:
        unchecked = {row.model_id for row in self._catalog_info.exclusions if row.unchecked}
        visible = tuple(m for m in self._models if m.id not in unchecked)
        terms = self._model_search.casefold().split()
        if terms:
            visible = tuple(model for model in visible if _model_matches_query(model, terms))
        if self._gpu_filter != "any":
            visible = tuple(model for model in visible if self._best(model) is not None)
        if terms or not self._show_top_models:
            return visible
        scored = any(m.quality_score is not None for m in self._models)
        return tuple(sorted(
            (m for m in visible if (not scored or m.quality_score is not None)
             and (m.id not in self._prices or self._best(m) is not None)),
            key=lambda m: (-(m.quality_score or 0), m.quality_rank or 10**9),
        )[:DEFAULT_OVERALL_MODEL_LIMIT])

    def action_toggle_model_view(self) -> None:
        """Switch between ready recommendations and the complete catalog."""
        if self._phase != "models":
            return
        self._show_top_models = not self._show_top_models
        self._render_model_list(preferred_id=self._highlighted_model_id())

    def viewport_profile_changed(
        self,
        profile: ViewportProfile,
        previous: ViewportProfile | None,
    ) -> None:
        """Reformat list records when their available width class changes."""
        if previous is not None and previous.width_mode == profile.width_mode:
            return
        try:
            if self._phase == "models":
                self._render_model_list(preferred_id=self._highlighted_model_id())
            elif (
                self._phase == "infra"
                and self._selected_model is not None
                and self._snapshot is not None
            ):
                self._render_infra_from_snapshot(self._selected_model, self._snapshot)
        except Exception:
            return

    def _render_model_list(self, *, preferred_id: str | None = None) -> None:
        option_list = self.query_one("#fast-deploy-list", OptionList)
        self.query_one("#fast-deploy-title", Static).update(
            screen_title("Deploy", steps=DEPLOY_STEPS, current=0)
        )
        self.query_one("#fast-deploy-search-group").remove_class("hidden")
        self.query_one("#fast-deploy-context").add_class("hidden")
        self.query_one("#fast-deploy-model-search").remove_class("hidden")
        self.query_one("#fast-deploy-search-label", Static).update("[dim]Search  /[/dim]")
        self._catalog_building = False
        visible = self._visible_models()
        if not self._models and not self._catalog_info.ready and not self._catalog_info.exclusions:
            self._render_catalog_unavailable(option_list)
            return
        terms = self._model_search.split()
        options: list[Option] = []
        model_indices: dict[str, int] = {}
        for model in visible:
            plan = self._best(model)
            cost = _quote_price_label(plan.quote) if plan else (
                "context unknown" if not any(p.context_known for p in model.profiles) else
                "no priced 4-bit+ fit" if model.id in self._prices else
                "pricing…" if self._pricing_pending else "price n/a"
            )
            model_indices[model.id] = len(options)
            options.append(Option(_model_option(model, self.viewport_profile.width_mode, cost_label=cost), id=model.id))
        self._blocked = {}
        if self._gpu_filter == "any" and (terms or not self._show_top_models):
            for row in self._catalog_info.exclusions:
                if not _names_match((row.display_name, row.repo_id, row.model_id), terms):
                    continue
                key = f"blocked:{row.model_id}"
                if key in self._blocked:
                    continue
                self._blocked[key] = row
                model_indices[key] = len(options)
                label = "unchecked" if row.unchecked else "unavailable"
                options.append(Option(f"  {escape(row.display_name)}  [dim]{label}[/dim]", id=key))
        if not options:
            # Name whichever filter is hiding everything. Falling through to
            # "no models in the catalog" told a reader whose search simply
            # missed to go and rebuild a catalog that was never the problem.
            search_query = self._model_search.strip()
            if search_query:
                scope = "catalog models"
                gpu = f" on {escape(self._gpu_filter)}" if self._gpu_filter != "any" else ""
                prompt = f"No {scope} match {escape(search_query)}{gpu}"
                hint = "Try fewer words or edit the repository name."
            elif self._gpu_filter != "any":
                prompt = f"No catalog models fit {escape(self._gpu_filter)}"
                hint = "Search all GPUs or press g to choose another GPU."
            elif self._show_top_models:
                prompt = "No ready models at full context"
                hint = "Press v or / to browse the catalog and check other contexts."
            else:
                prompt = "No models in the quick-deploy catalog"
                hint = ""
            options.append(Option(f"  {prompt}", disabled=True))
            self.query_one("#fast-deploy-detail", Static).update(hint)
        show_all_gpus = self._gpu_filter != "any" and (terms or not model_indices)
        if show_all_gpus:
            options.append(Option("  Search all GPUs", id=_SEARCH_ALL_GPUS_ID))
        replace_options(option_list, options)
        if show_all_gpus and not model_indices:
            option_list.highlighted = len(options) - 1
        if model_indices:
            selected_id = preferred_id if preferred_id in model_indices else next(iter(model_indices))
            option_list.highlighted = model_indices[selected_id]
            if selected_id in self._blocked:
                self._show_blocker(selected_id)
            else:
                self._update_model_detail(self._model_by_id[selected_id])
            # A resize can retain the index while changing row heights; wait
            # for layout before making the selected model visible again.
            self.call_after_refresh(option_list.scroll_to_highlight)
        self._set_subtitle(_subtitle(self._catalog_info))
        if terms or not self._show_top_models:
            view_label = "Search results" if terms else "All catalog models"
        elif any(m.quality_score is not None for m in self._models):
            view_label = "Top 10 by AA score"
        else:
            view_label = "Trending" if self._catalog_info.is_fallback else "Unranked"
        total = len(model_indices)
        count_label = f"{total} model{'s' if total != 1 else ''}"
        filter_note = ""
        if self._gpu_filter != "any":
            filter_note = f" · GPU {escape(self._gpu_filter)}"
        excluded_note = (
            f" · {len(self._catalog_info.exclusions)} excluded (x)"
            if self._catalog_info.exclusions else ""
        )
        # The source label belongs to the subtitle; repeating it here put the
        # same sentence on screen twice, three lines apart.
        self.query_one("#fast-deploy-status", Static).update(
            f"[dim]{count_label} · {view_label}{filter_note}{excluded_note}[/dim]"
            + ("\n[$warning]Partial results: " + escape("; ".join(self._snapshot.errors)) + "[/$warning]"
               if self._snapshot is not None and self._snapshot.errors else "")
            + (f"\n[$warning]{escape(self._pricing_error)}[/$warning]" if self._pricing_error else "")
            + (f"\n[dim]{escape(self._offer_wait_label())}[/dim]" if self._pending_providers
               else "\n[dim]Pricing live placements…[/dim]"
               if self._fetch_pending or self._pricing_pending else "")
            # A build that stopped early still publishes what it found. Saying
            # so is the difference between "nothing better exists" and "we ran
            # out of requests before we could look". Every way of not looking
            # counts here, not just the exhausted request budget: Hugging Face
            # throttling leaves the same gap and used to say nothing at all.
            + (f"\n[dim]{unchecked} more still to check · reopen Deploy to continue[/dim]"
               if (unchecked := sum(
                   1 for exclusion in self._catalog_info.exclusions
                   if exclusion.unchecked)) else "")
        )
        if self.focused is None:
            option_list.focus()

    def _populate_gpu_filter(self, prices: _ModelPrices | None = None) -> None:
        selector = self.query_one("#fast-deploy-gpu-filter", Select)
        options = self._gpu_choices.copy()
        if prices:
            for index, (label, value) in enumerate(options):
                best = prices.best_by_gpu.get(value) if value != "any" else None
                if best:
                    options[index] = (f"{label} ×{best.quote.gpu_count} · {_quote_price_label(best.quote)}", value)
        current = self._gpu_filter
        values = {value for _label, value in options}
        if current not in values:
            if self._selected_model is not None or self._fetch_pending:
                # Partial updates may not include the user's GPU yet.
                options.append((current, current))
            else:
                current = "any"
                self._gpu_filter = "any"
        self._updating_gpu_filter = True
        with selector.prevent(Select.Changed):
            selector.set_options(options)
            selector.value = current
        self._updating_gpu_filter = False

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "fast-deploy-context":
            if (self._phase == "infra" and self._selected_model and self._snapshot
                    and isinstance(event.value, int) and event.value in self._contexts
                    and event.value != (self._context_tokens or self._selected_model.max_context_tokens)):
                self._context_tokens = event.value
                self._placement_id = None
                self._phase = "loading"
                self._render_infra_from_snapshot(self._selected_model, self._snapshot)
            return
        if event.select.id != "fast-deploy-gpu-filter" or self._updating_gpu_filter:
            return
        if not isinstance(event.value, str) or not event.value:
            return
        if event.value == self._gpu_filter:
            return
        self._gpu_filter = event.value
        if self._phase == "infra":
            self._placement_id = None
            replace_options(self.query_one("#fast-deploy-list", OptionList), [])
        if self._phase == "models":
            self._render_model_list(preferred_id=self._highlighted_model_id())
            return
        if self._phase == "infra" and self._selected_model is not None and self._snapshot is not None:
            self._render_infra_from_snapshot(self._selected_model, self._snapshot)

    def _reset_to_models(self) -> None:
        preferred_id = self._selected_model.id if self._selected_model is not None else None
        self._phase = "models"
        self._selected_model = None
        self._context_tokens = None
        self._contexts = []
        if self._snapshot is not None:
            self._populate_gpu_filter()
        self._infra_rows = {}
        self._has_tiers = False
        self._availability_inflight = False
        self.workers.cancel_group(self, "fast-deploy-prices")
        self._pricing_pending = self._fetch_pending or (
            self._snapshot is not None and len(self._prices) < len(self._models)
        )
        self._render_model_list(preferred_id=preferred_id)
        self.query_one("#fast-deploy-list", OptionList).focus()
        self._apply_model_catalog(force=True, preferred_id=preferred_id)

    def _update_model_detail(self, model: QuickDeployModel) -> None:
        # Prices are on the row and provider caveats on Confirm; the detail
        # pane only adds what the row cannot hold.
        detail = _model_detail(model)
        if model.id in self._prices and not self._best(model):
            reasons = self._prices[model.id].rejected
            detail += "\n" + escape(next(iter(reasons), "No priced full-context placement with verified 4-bit or better weights."))
        self.query_one("#fast-deploy-detail", Static).update(detail)

    def _update_infra_detail(self, row: _InfraRow) -> None:
        self.query_one("#fast-deploy-detail", Static).update(_infra_detail(row))

    def action_choose_selected(self) -> None:
        if getattr(self.focused, "id", None) not in {"fast-deploy-list", "fast-deploy-model-search"}:
            raise SkipAction()
        highlighted = highlighted_option(self.query_one("#fast-deploy-list", OptionList))
        if highlighted is not None and highlighted.id is not None:
            self._choose(str(highlighted.id))

    def action_focus_model_results(self) -> None:
        if self._highlighted_model_id() is not None:
            self.query_one("#fast-deploy-list", OptionList).focus()

    def action_focus_gpu_filter(self) -> None:
        self.query_one("#fast-deploy-gpu-filter", Select).focus()

    def action_show_excluded_models(self) -> None:
        if self._catalog_info.exclusions:
            self.app.push_screen(CatalogExclusionsScreen(self._catalog_info.exclusions))
        else:
            self.notify("No model exclusions were recorded for this catalog.", timeout=3)

    def action_focus_model_search(self) -> None:
        search = self.query_one("#fast-deploy-model-search", Input)
        if self._phase != "models":
            self.notify(
                "Model search belongs to step 1; press esc to pick another model.",
                timeout=3,
            )
            return
        search.focus()

    def action_pop_screen(self) -> None:
        if self._phase == "models":
            if isinstance(self.focused, Input):
                self.focused.value = ""
                self.query_one("#fast-deploy-list", OptionList).focus()
                return
            self.app.pop_screen()
            return
        self._reset_to_models()
