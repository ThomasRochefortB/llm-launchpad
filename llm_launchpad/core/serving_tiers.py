"""Reduce a placement frontier to the few choices a person can actually make.

Step two of Fast Deploy used to be a hardware menu: L40S x2, A100-80GB x1,
RTX-PRO-6000 x1. Nobody knows whether they want two L40S. Everybody knows
whether they want cheap or fast, so the frontier is collapsed to at most three
named points and the hardware becomes a detail.

Every tier serves the model's full context on GPU at the same bit width. Tiers
move price and speed; they never quietly reduce what the model can do, which is
what lets a cheap option exist without reopening the silent-degradation
question.

Weights are held to that same promise. Price and speed both improve as the
quantization shrinks, so a frontier spanning two bit widths hands every tier to
the smaller one -- 2-bit weights chosen on the user's behalf, sometimes on the
identical GPU the 4-bit weights would have used. The frontier is therefore
explored at one quality level: the best the connected placements can serve. A
smaller quantization is offered beside it only when it buys a real discount,
and only with its bit width named.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Sequence

from ..protocol.enums import ServingObjective
from ..protocol.models import InferencePlan
from .inference_options import first_hour_cost_usd
from .llamacpp_planner import assessment_score
from .quant_quality import (
    QUALITY_FLOOR_BITS,
    is_reduced_quality,
    quant_quality_label,
    serving_quality_bits,
)
from .quick_deploy import plan_is_eligible

# Below this, two options are the same on that axis and saying "1.0x" would be
# noise. A flank that gains nothing is still shown -- with the gain named as
# absent, so a bad deal reads as a bad deal rather than disappearing.
_MEANINGFUL_MARGIN = 0.05

# A smaller quantization has to pay for the quality it gives up. Five percent
# is the bar for "these two options differ at all"; handing someone 2-bit
# weights to save that little would be a bad trade made on their behalf.
SAVER_MIN_SAVING = 0.25

ECONOMY = "economy"
BALANCED = "balanced"
FASTEST = "fastest"
SAVER = "saver"

_TIER_LABELS = {
    ECONOMY: "Economy",
    BALANCED: "Balanced",
    FASTEST: "Fastest",
    SAVER: "Saver",
}

# How a role reads when some other tier's row has already claimed the placement
# that won it.
_SECONDARY_ROLE_NAMES = {
    ECONOMY: "the cheapest",
    BALANCED: "the best value",
    FASTEST: "the fastest",
}


@dataclass(frozen=True)
class ServingTier:
    """One named point on the certified placement frontier."""

    key: str
    label: str
    plan: InferencePlan
    is_recommended: bool = False
    tradeoff: str | None = None
    #: Roles this same placement also won, which therefore have no row of
    #: their own. Empty when each role went to a different placement.
    also: tuple[str, ...] = ()

    @property
    def price_per_hour_usd(self) -> float | None:
        return self.plan.quote.price_per_hour_usd

    @property
    def output_tokens_per_second(self) -> float:
        """Single-stream decode speed, the number a person feels while typing."""

        return _single_stream_tps(self.plan)

    @property
    def aggregate_output_tokens_per_second(self) -> float:
        """Combined decode speed across concurrent requests."""

        return _aggregate_tps(self.plan)

    @property
    def measured(self) -> bool:
        """Whether these numbers came from a real deployment."""

        assessment = self.plan.assessment
        if assessment is None:
            return False
        return any(point.measured for point in assessment.performance)


def _comparison_price(plan: InferencePlan) -> float | None:
    """What tiers compare: an hour of rent plus one-time charges.

    Equal to the hourly rate except where a provider bills something once per
    deployment, as Vast does for the weight download. Ranking on the rate
    alone let a cheap-rent, costly-transfer host win Economy.
    """

    return first_hour_cost_usd(plan.quote)


def _performance_points(plan: InferencePlan) -> tuple:
    """Return comparable performance points for tier ranking.

    Measured and estimated points carry different workloads and metric
    semantics, so mixing them lets a short-prompt measurement outrank an
    estimate on speed alone. Rank within one evidence class: measured when the
    plan has any, estimated otherwise.
    """

    assessment = plan.assessment
    if assessment is None:
        return ()
    measured = tuple(point for point in assessment.performance if point.measured)
    if measured:
        return measured
    return tuple(point for point in assessment.performance if not point.measured)


def _single_stream_tps(plan: InferencePlan) -> float:
    return max(
        (
            point.output_tokens_per_second or 0.0
            for point in _performance_points(plan)
            if point.concurrency == 1
        ),
        default=0.0,
    )


def _aggregate_tps(plan: InferencePlan) -> float:
    return max(
        (
            point.aggregate_output_tokens_per_second or 0.0
            for point in _performance_points(plan)
        ),
        default=0.0,
    )


def _throughput(plan: InferencePlan, objective: ServingObjective) -> float:
    """The speed a given objective is actually trying to maximize."""

    if objective in {ServingObjective.THROUGHPUT, ServingObjective.BENCHMARK}:
        return _aggregate_tps(plan)
    return _single_stream_tps(plan)


def _efficiency(plan: InferencePlan) -> float:
    """Throughput per dollar: the axis the middle tier is chosen on.

    ``assessment_score`` blends raw speed with efficiency, which is right for
    ranking one recommendation but pulls the middle tier onto the largest
    machine -- leaving nothing between cheapest and fastest. Value is the knee
    of the curve, so it is measured directly.

    Efficiency is always recomputed from the plan's current price: a cached
    tokens-per-dollar figure from an earlier quote must not survive a price
    change.
    """

    assessment = plan.assessment
    if assessment is None:
        return 0.0
    price = _comparison_price(plan)
    if not price:
        return 0.0
    return _aggregate_tps(plan) / price


def _describe_tradeoff(
    tier_plan: InferencePlan,
    baseline: InferencePlan,
    objective: ServingObjective,
) -> str | None:
    """Compare a tier to the recommended one in one clause.

    A ratio is easier to act on than two absolute numbers the reader has to
    divide themselves. Speed ratios are only stated when both sides share an
    evidence class: a measured number and a heuristic estimate have different
    workloads behind them, so their ratio is not a speedup the reader can buy.
    """

    if tier_plan.quote.id == baseline.quote.id:
        return None
    speed = _throughput(tier_plan, objective)
    baseline_speed = _throughput(baseline, objective)
    price = _comparison_price(tier_plan)
    baseline_price = _comparison_price(baseline)
    speed_text = ""
    if _evidence_classes_match(tier_plan, baseline) and speed > 0 and baseline_speed > 0:
        ratio = speed / baseline_speed
        if ratio >= 1.0 + _MEANINGFUL_MARGIN:
            speed_text = f"{ratio:.1f}x faster"
        elif ratio <= 1.0 - _MEANINGFUL_MARGIN:
            speed_text = f"{1 / ratio:.1f}x slower"
        else:
            speed_text = "same speed"
    price_text = ""
    if price and baseline_price:
        ratio = price / baseline_price
        if ratio >= 1.0 + _MEANINGFUL_MARGIN:
            price_text = f"{ratio:.1f}x the price"
        elif ratio <= 1.0 - _MEANINGFUL_MARGIN:
            price_text = f"{1 / ratio:.1f}x cheaper"
        else:
            # Naming the absent benefit is the point: an option that is much
            # slower and no cheaper should read as the bad deal it is, rather
            # than being quietly dropped from the list.
            price_text = "no cheaper"
    parts = [text for text in (speed_text, price_text) if text]
    return ", ".join(parts) if parts else None


def _evidence_classes_match(left: InferencePlan, right: InferencePlan) -> bool:
    """Whether two plans carry the same kind of speed evidence."""

    def _evidence(plan: InferencePlan) -> bool:
        assessment = plan.assessment
        if assessment is None:
            return False
        return any(point.measured for point in assessment.performance)

    return _evidence(left) == _evidence(right)


def _quality_partition(
    plans: Sequence[InferencePlan],
) -> tuple[list[InferencePlan], list[InferencePlan]]:
    """Split a frontier into the best bit width available and everything below.

    Grouping is by width rather than by label: Q4_K_M and UD-Q4_K_XL are one
    quality level with two spellings, and splitting them would strand half the
    frontier.
    """

    best = max(serving_quality_bits(plan.recipe.quant) for plan in plans)
    primary = [
        plan for plan in plans if serving_quality_bits(plan.recipe.quant) == best
    ]
    lower = [plan for plan in plans if serving_quality_bits(plan.recipe.quant) < best]
    return primary, lower


def _frontier_tiers(
    plans: Sequence[InferencePlan],
    objective: ServingObjective,
) -> tuple[tuple[ServingTier, ...], InferencePlan]:
    """Name the cheapest, best-value and fastest points at one quality level."""

    priced = [plan for plan in plans if _comparison_price(plan) is not None]
    cheapest = (
        min(priced, key=lambda plan: _comparison_price(plan) or 0.0)
        if priced
        else plans[0]
    )
    fastest = max(plans, key=lambda plan: _throughput(plan, objective))
    balanced = max(
        plans,
        key=lambda plan: (
            _efficiency(plan),
            # Ties break toward the placement the planner would have ranked
            # first anyway, keeping the recommendation consistent with the
            # detailed comparison behind it.
            assessment_score(plan.assessment, objective)
            if plan.assessment is not None
            else 0.0,
        ),
    )

    ordered: list[tuple[str, InferencePlan]] = [
        (ECONOMY, cheapest),
        (BALANCED, balanced),
        (FASTEST, fastest),
    ]
    # One placement routinely wins more than one role -- a B200 is often both
    # the fastest and the best value. It still gets one row, because inventing
    # a second would defeat the point of showing the flanks, but the roles it
    # swallowed are named rather than dropped: a row reading only "Balanced"
    # sends the reader off to the full list hunting for a faster placement
    # that does not exist.
    roles: dict[str, list[str]] = {}
    plan_by_quote: dict[str, InferencePlan] = {}
    for key, plan in ordered:
        roles.setdefault(plan.quote.id, []).append(key)
        plan_by_quote.setdefault(plan.quote.id, plan)

    tiers: list[ServingTier] = []
    for quote_id, keys in roles.items():
        plan = plan_by_quote[quote_id]
        primary, *extra = keys
        clause = _also_clause(extra)
        detail = _describe_tradeoff(plan, balanced, objective)
        parts = [text for text in (clause, detail) if text]
        tiers.append(
            ServingTier(
                key=primary,
                label=_TIER_LABELS[primary],
                plan=plan,
                is_recommended=quote_id == balanced.quote.id,
                tradeoff=" · ".join(parts) if parts else None,
                also=tuple(extra),
            )
        )
    return tuple(tiers), balanced


def _saver_tier(
    primary: Sequence[InferencePlan],
    lower: Sequence[InferencePlan],
    baseline: InferencePlan,
    objective: ServingObjective,
) -> tuple[ServingTier, ...]:
    """Offer a smaller quantization only when it pays for what it costs.

    On the compact models a 2-bit build lands on the very same GPU as the
    4-bit one, so the trade is pure loss and the option should not exist. It
    earns a place only once the discount is large enough to be a real
    decision, and it goes last: this is an opt-out from the quality floor, not
    the way into the list.
    """

    priced_lower = [
        plan for plan in lower if _comparison_price(plan) is not None
    ]
    priced_primary = [
        plan for plan in primary if _comparison_price(plan) is not None
    ]
    if not priced_lower or not priced_primary:
        return ()
    cheapest_lower = min(
        priced_lower, key=lambda plan: _comparison_price(plan) or 0.0
    )
    floor_price = min(_comparison_price(plan) or 0.0 for plan in priced_primary)
    saver_price = _comparison_price(cheapest_lower) or 0.0
    if not floor_price or saver_price > floor_price * (1.0 - SAVER_MIN_SAVING):
        return ()

    quality = quant_quality_label(cheapest_lower.recipe.quant) or "smaller"
    parts = [f"{quality} weights, lower output quality"]
    detail = _describe_tradeoff(cheapest_lower, baseline, objective)
    if detail:
        parts.append(detail)
    return (
        ServingTier(
            key=SAVER,
            label=_TIER_LABELS[SAVER],
            plan=cheapest_lower,
            is_recommended=False,
            tradeoff=" · ".join(parts),
        ),
    )


def _also_clause(extra: Sequence[str], scope: str = "") -> str:
    """Name the roles a placement also won, so their labels are not just lost.

    BALANCED is deliberately left unsaid. The recommendation marker already
    carries it, and "also the best value · recommended" says one thing twice.

    ``scope`` narrows the claim to the bit width it was decided within, for the
    rows a saver out-runs or undercuts.
    """

    names = [_SECONDARY_ROLE_NAMES[key] for key in extra if key != BALANCED]
    if not names:
        return ""
    suffix = f" at {scope}" if scope else ""
    return "also " + " and ".join(names) + suffix


def _scoped_against_saver(
    tiers: Sequence[ServingTier],
    saver: ServingTier,
    baseline: InferencePlan,
    objective: ServingObjective,
) -> tuple[ServingTier, ...]:
    """Qualify the frontier's superlatives that the saver row disproves.

    The frontier is drawn at one bit width, so "the fastest" means the fastest
    at that width -- but the saver is rendered in the same list, and a reader
    compares rows, not partitions. GLM 5.3 Flash shipped a recommended row
    reading "also the fastest" at 12 tok/s immediately above a saver at 30.

    The claim is narrowed rather than dropped: naming the width keeps the
    reason the clause exists, which is to stop the reader going off to the full
    placement list after something that is not there.
    """

    saver_speed = _throughput(saver.plan, objective)
    saver_price = _comparison_price(saver.plan)
    rescoped: list[ServingTier] = []
    for tier in tiers:
        beaten = [
            key
            for key in tier.also
            if (key == FASTEST and saver_speed > _throughput(tier.plan, objective))
            or (
                key == ECONOMY
                and saver_price is not None
                and (_comparison_price(tier.plan) or 0.0) > saver_price
            )
        ]
        if not beaten:
            rescoped.append(tier)
            continue
        scope = quant_quality_label(tier.plan.recipe.quant) or "this quality"
        detail = _describe_tradeoff(tier.plan, baseline, objective)
        parts = [text for text in (_also_clause(tier.also, scope), detail) if text]
        rescoped.append(
            replace(tier, tradeoff=" · ".join(parts) if parts else None)
        )
    return tuple(rescoped)


def serving_tiers(
    plans: Sequence[InferencePlan],
    objective: ServingObjective = ServingObjective.GENERAL_PURPOSE,
) -> tuple[ServingTier, ...]:
    """Pick the cheapest, best-value and fastest placements from a frontier.

    The frontier is first narrowed to the best bit width the connected
    placements can serve, so the three tiers differ in price and speed alone.
    A smaller quantization follows as a fourth, labelled option when it is
    materially cheaper.

    Returns fewer than three tiers when one placement wins more than one role.
    Manufacturing variety would undermine the point of showing the flanks,
    which is to make the recommended choice checkable rather than trusted.
    """

    eligible = [plan for plan in plans if plan_is_eligible(plan)]
    if not eligible:
        return ()

    primary, lower = _quality_partition(eligible)
    tiers, balanced = _frontier_tiers(primary, objective)
    saver = _saver_tier(primary, lower, balanced, objective)
    if saver:
        tiers = _scoped_against_saver(tiers, saver[0], balanced, objective)
    return tiers + saver


def _quality_floor_note(quants: Sequence[str | None]) -> str | None:
    """Say that every way of serving this model is below the floor."""

    if not quants or not all(is_reduced_quality(quant) for quant in quants):
        return None
    label = quant_quality_label(quants[0]) or "reduced-precision"
    return (
        f"Only {label} weights of this model fit the connected placements. "
        f"Output quality is below the {QUALITY_FLOOR_BITS}-bit baseline."
    )


def reduced_quality_note(tiers: Sequence[ServingTier]) -> str | None:
    """Disclose a quality floor the user did not choose.

    A model that only fits below the floor is still worth offering -- Kimi K3
    is 2-bit or nothing -- but the reason has to be on screen, or the
    degradation is exactly as silent as the kind this module set out to avoid.
    """

    return _quality_floor_note(
        [tier.plan.recipe.quant for tier in tiers if tier.key != SAVER]
    )


def reduced_quality_plan_note(plans: Sequence[InferencePlan]) -> str | None:
    """The same disclosure for a frontier shown as placements, not tiers.

    A model with a single live placement has no tiers to collapse into, so it
    is listed directly -- and that is exactly the case where the floor is most
    likely to bite. Reading the note off tiers alone left the one model that
    needed the sentence without it.
    """

    eligible = [plan for plan in plans if plan_is_eligible(plan)]
    if not eligible:
        return None
    primary, _ = _quality_partition(eligible)
    return _quality_floor_note([plan.recipe.quant for plan in primary])
