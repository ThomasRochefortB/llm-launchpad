"""Reduce a placement frontier to the few choices a person can actually make.

Step two of Fast Deploy used to be a hardware menu: L40S x2, A100-80GB x1,
RTX-PRO-6000 x1. Nobody knows whether they want two L40S. Everybody knows
whether they want cheap or fast, so the frontier is collapsed to at most three
named points and the hardware becomes a detail.

Every tier serves the model's full context on GPU. Tiers move price and speed;
they never quietly reduce what the model can do, which is what lets a cheap
option exist without reopening the silent-degradation question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..protocol.enums import ServingObjective
from ..protocol.models import InferencePlan
from .llamacpp_planner import assessment_score

# Below this, two options are the same on that axis and saying "1.0x" would be
# noise. A flank that gains nothing is still shown -- with the gain named as
# absent, so a bad deal reads as a bad deal rather than disappearing.
_MEANINGFUL_MARGIN = 0.05

ECONOMY = "economy"
BALANCED = "balanced"
FASTEST = "fastest"

_TIER_LABELS = {
    ECONOMY: "Economy",
    BALANCED: "Balanced",
    FASTEST: "Fastest",
}


@dataclass(frozen=True)
class ServingTier:
    """One named point on the certified placement frontier."""

    key: str
    label: str
    plan: InferencePlan
    is_recommended: bool = False
    tradeoff: str | None = None

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


def _single_stream_tps(plan: InferencePlan) -> float:
    assessment = plan.assessment
    if assessment is None:
        return 0.0
    return max(
        (
            point.output_tokens_per_second or 0.0
            for point in assessment.performance
            if point.concurrency == 1
        ),
        default=0.0,
    )


def _aggregate_tps(plan: InferencePlan) -> float:
    assessment = plan.assessment
    if assessment is None:
        return 0.0
    return max(
        (
            point.aggregate_output_tokens_per_second or 0.0
            for point in assessment.performance
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
    """

    assessment = plan.assessment
    if assessment is None:
        return 0.0
    measured = max(
        (point.output_tokens_per_dollar or 0.0 for point in assessment.performance),
        default=0.0,
    )
    if measured > 0:
        return measured
    price = plan.quote.price_per_hour_usd
    if not price:
        return 0.0
    return _aggregate_tps(plan) / price


def _is_eligible(plan: InferencePlan) -> bool:
    assessment = plan.assessment
    if assessment is None:
        # Legacy plans carry no assessment; they are offered but never ranked
        # as though their performance were known.
        return True
    return assessment.fits and assessment.gpu_resident


def _describe_tradeoff(
    tier_plan: InferencePlan,
    baseline: InferencePlan,
    objective: ServingObjective,
) -> str | None:
    """Compare a tier to the recommended one in one clause.

    A ratio is easier to act on than two absolute numbers the reader has to
    divide themselves.
    """

    if tier_plan.quote.id == baseline.quote.id:
        return None
    speed = _throughput(tier_plan, objective)
    baseline_speed = _throughput(baseline, objective)
    price = tier_plan.quote.price_per_hour_usd
    baseline_price = baseline.quote.price_per_hour_usd
    speed_text = ""
    if speed > 0 and baseline_speed > 0:
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


def serving_tiers(
    plans: Sequence[InferencePlan],
    objective: ServingObjective = ServingObjective.GENERAL_PURPOSE,
) -> tuple[ServingTier, ...]:
    """Pick the cheapest, best-value and fastest placements from a frontier.

    Returns fewer than three tiers when one placement wins more than one role.
    Manufacturing variety would undermine the point of showing the flanks,
    which is to make the recommended choice checkable rather than trusted.
    """

    eligible = [plan for plan in plans if _is_eligible(plan)]
    if not eligible:
        return ()

    priced = [plan for plan in eligible if plan.quote.price_per_hour_usd is not None]
    cheapest = (
        min(priced, key=lambda plan: plan.quote.price_per_hour_usd or 0.0)
        if priced
        else eligible[0]
    )
    fastest = max(eligible, key=lambda plan: _throughput(plan, objective))
    balanced = max(
        eligible,
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
    tiers: list[ServingTier] = []
    seen: set[str] = set()
    for key, plan in ordered:
        if plan.quote.id in seen:
            continue
        seen.add(plan.quote.id)
        tiers.append(
            ServingTier(
                key=key,
                label=_TIER_LABELS[key],
                plan=plan,
                is_recommended=plan.quote.id == balanced.quote.id,
                tradeoff=_describe_tradeoff(plan, balanced, objective),
            )
        )
    return tuple(tiers)
