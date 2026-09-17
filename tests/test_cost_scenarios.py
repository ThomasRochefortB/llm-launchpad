from __future__ import annotations

import unittest

from llm_launchpad.core.inference_options import (
    COST_SCENARIO_CLUSTERED,
    COST_SCENARIO_CONTINUOUS,
    COST_SCENARIO_SPARSE,
    COST_SCENARIO_WORKDAY,
    estimate_cost_for_scenario,
    estimate_modal_billed_hours_per_day,
    format_cost_summary,
)
from llm_launchpad.protocol.enums import BillingModel, ComputeProvider
from llm_launchpad.protocol.models import ProviderQuote


def _quote(billing: BillingModel, price: float = 1.0) -> ProviderQuote:
    return ProviderQuote(
        id="q",
        recipe_id="r",
        provider=ComputeProvider.MODAL if billing == BillingModel.SCALE_TO_ZERO else ComputeProvider.PRIME,
        provider_reference="ref",
        gpu_type="H100",
        gpu_count=1,
        price_per_hour_usd=price,
        billing_model=billing,
    )


class CostScenarioTests(unittest.TestCase):
    def test_continuous_rental_is_720_at_one_dollar(self) -> None:
        quote = _quote(BillingModel.PROVISIONED, 1.0)
        self.assertEqual(estimate_cost_for_scenario(quote, COST_SCENARIO_CONTINUOUS), 720.0)

    def test_workday_schedule_is_240_with_shutdown_assumption(self) -> None:
        quote = _quote(BillingModel.PROVISIONED, 1.0)
        self.assertEqual(estimate_cost_for_scenario(quote, COST_SCENARIO_WORKDAY), 240.0)

    def test_modal_sparse_vs_clustered_same_active_different_bills(self) -> None:
        modal = _quote(BillingModel.SCALE_TO_ZERO, 1.0)
        sparse = estimate_cost_for_scenario(modal, COST_SCENARIO_SPARSE)
        clustered = estimate_cost_for_scenario(modal, COST_SCENARIO_CLUSTERED)
        assert sparse is not None and clustered is not None
        # Same 0.5h active; sparse spreads it over 10 sessions (10 idle timeouts),
        # clustered over 2. Idle timeout dominates, so sparse bills more.
        self.assertGreater(sparse, clustered)
        self.assertEqual(clustered, (0.5 + 2 * 0.5) * 30.0)
        self.assertEqual(sparse, (0.5 + 10 * 0.5) * 30.0)

    def test_modal_billed_hours_capped_by_window(self) -> None:
        billed = estimate_modal_billed_hours_per_day(
            active_hours_per_day=0.5,
            sessions_per_day=100.0,
            idle_timeout_seconds=1800.0,
            window_hours_per_day=8.0,
        )
        self.assertEqual(billed, 8.0)

    def test_format_summary_leads_with_hourly_and_continuous(self) -> None:
        quote = _quote(BillingModel.PROVISIONED, 1.0)
        summary = format_cost_summary(quote, COST_SCENARIO_WORKDAY)
        self.assertIn("$1.00/hr while billed", summary)
        self.assertIn("$720.00/mo if left running 24/7", summary)
        self.assertIn("Workday 8h", summary)
        self.assertIn("storage separate", summary)

    def test_unknown_price_returns_none_not_zero(self) -> None:
        quote = _quote(BillingModel.PROVISIONED, 1.0)
        quote = ProviderQuote(
            id=quote.id,
            recipe_id=quote.recipe_id,
            provider=quote.provider,
            provider_reference=quote.provider_reference,
            gpu_type=quote.gpu_type,
            gpu_count=quote.gpu_count,
            price_per_hour_usd=None,
            billing_model=quote.billing_model,
        )
        self.assertIsNone(estimate_cost_for_scenario(quote, COST_SCENARIO_WORKDAY))
