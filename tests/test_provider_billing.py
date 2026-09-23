"""Provider billing: one shape per provider, and one way of saying each thing.

The home panel used to carry three unrelated layouts with three vocabularies
for the same four states. These tests pin the parsing per provider and then
assert the property that was missing before: whatever the provider, the panel
says "checking...", "not configured" and "unavailable" the same way.
"""

from __future__ import annotations

import unittest

from textual.content import Content

from llm_launchpad.core.provider_billing import (
    PROVIDER_BILLING_ORDER,
    PROVIDER_SETUP_COMMANDS,
    BalanceKind,
    BillingStatus,
    ProviderBilling,
    parse_modal_billing,
    parse_prime_billing,
    parse_vast_billing,
)
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import StorageSnapshot, StoredModelInfo
from llm_launchpad.tui.billing_panel import render_provider_billing
from llm_launchpad.tui.markers import PROVIDER_MARKERS

_GIB = 1024**3


def _plain(markup: str) -> str:
    return Content.from_markup(markup).plain


def _snapshot(size_bytes: int) -> StorageSnapshot:
    return StorageSnapshot(
        llamacpp_models=[
            StoredModelInfo(
                backend=BackendType.LLAMACPP, model_id="m", size_bytes=size_bytes
            )
        ],
        vllm_models=[],
    )


class ModalBillingParseTests(unittest.TestCase):
    def test_summary_payload_yields_month_to_date_spend(self) -> None:
        row = parse_modal_billing(
            {"summary": {"total_usd": 12.5, "gpu_cost_usd": 8.1, "period_end": "2026-02-18"}}
        )
        self.assertIs(row.status, BillingStatus.READY)
        self.assertIs(row.kind, BalanceKind.SPEND_MTD)
        self.assertEqual(row.amount_usd, 12.5)
        self.assertEqual([(c.label, c.amount_usd) for c in row.charges], [("gpu", 8.1)])

    def test_row_payload_is_summed_across_apps(self) -> None:
        row = parse_modal_billing(
            [
                {"Description": "app-a", "Cost": "1.25"},
                {"Description": "app-b", "Cost": "2.75"},
            ]
        )
        self.assertEqual(row.amount_usd, 4.0)

    def test_an_empty_report_is_zero_rather_than_unknown(self) -> None:
        """Nothing billed yet is an answer; it must not read as a failure."""
        row = parse_modal_billing([])
        self.assertIs(row.status, BillingStatus.READY)
        self.assertEqual(row.amount_usd, 0.0)

    def test_envelope_keys_are_unwrapped(self) -> None:
        self.assertEqual(
            parse_modal_billing({"report": {"total_usd": 3.0}}).amount_usd, 3.0
        )
        self.assertEqual(
            parse_modal_billing({"data": {"total_usd": 4.0}}).amount_usd, 4.0
        )

    def test_money_strings_carry_currency_and_separators(self) -> None:
        self.assertEqual(parse_modal_billing([{"Cost": "$1,234.50"}]).amount_usd, 1234.5)

    def test_an_unreadable_payload_says_so_without_raising(self) -> None:
        row = parse_modal_billing("not-json")
        self.assertIsNone(row.amount_usd)
        self.assertTrue(row.notes)


class PrimeBillingParseTests(unittest.TestCase):
    def test_balance_and_charges_are_grouped_by_resource(self) -> None:
        row = parse_prime_billing(
            {
                "balance_usd": 41.25,
                "recent_billings": [
                    {"amount_usd": 1.25, "resource_type": "compute"},
                    {"amount_usd": 0.75, "resource_type": "compute"},
                    {"amount_usd": "0.50", "resource_type": "disks"},
                    {"amount_usd": None, "resource_type": "inference"},
                ],
            }
        )
        self.assertIs(row.kind, BalanceKind.BALANCE)
        self.assertEqual(row.amount_usd, 41.25)
        self.assertEqual(
            [(c.label, c.amount_usd) for c in row.charges],
            [("compute", 2.0), ("disks", 0.5)],
        )

    def test_a_row_without_an_amount_contributes_no_resource(self) -> None:
        row = parse_prime_billing(
            {"balance_usd": 5, "recent_billings": [{"amount_usd": None, "resource_type": "x"}]}
        )
        self.assertEqual(row.charges, ())

    def test_an_unreadable_payload_says_so_without_raising(self) -> None:
        row = parse_prime_billing("not-json")
        self.assertIsNone(row.amount_usd)
        self.assertIn("prime wallet", " ".join(row.notes))


class VastBillingParseTests(unittest.TestCase):
    def test_available_credit_is_the_headline(self) -> None:
        row = parse_vast_billing(
            {"credit_usd": 20.0, "balance_usd": 0.0, "available_usd": 18.5}
        )
        self.assertIs(row.kind, BalanceKind.CREDIT)
        self.assertEqual(row.amount_usd, 18.5)
        self.assertIsNone(row.owed_usd)

    def test_an_owed_balance_is_carried_separately(self) -> None:
        """A negative balance is why a healthy credit will not start a rental."""
        row = parse_vast_billing(
            {"credit_usd": 0.0, "balance_usd": -4.25, "available_usd": -4.25}
        )
        self.assertEqual(row.owed_usd, 4.25)

    def test_malformed_payloads_do_not_raise(self) -> None:
        for payload in (None, {}, {"available_usd": None}, "nonsense"):
            with self.subTest(payload=payload):
                row = parse_vast_billing(payload)
                self.assertIs(row.provider, ComputeProvider.VAST)
                self.assertIsNone(row.amount_usd)


class BillingPanelRenderTests(unittest.TestCase):
    def test_every_provider_is_named_with_its_figure_and_meaning(self) -> None:
        rendered = _plain(
            render_provider_billing(
                [
                    parse_modal_billing({"summary": {"total_usd": 15.75}}),
                    parse_prime_billing({"balance_usd": 142.0}),
                    parse_vast_billing({"available_usd": 18.5, "balance_usd": 0.0}),
                ]
            )
        )
        for name in ("Modal", "Prime Intellect", "Vast.ai"):
            self.assertIn(name, rendered)
        for figure in ("$15.75", "$142.00", "$18.50"):
            self.assertIn(figure, rendered)
        # Spend and balance are different things, and the row has to say which.
        self.assertIn("spent this month", rendered)
        self.assertIn("balance", rendered)
        self.assertIn("credit", rendered)

    def test_figures_line_up_in_one_column(self) -> None:
        """Three sections became one table; the amounts share a right edge."""
        rendered = _plain(
            render_provider_billing(
                [
                    parse_modal_billing({"summary": {"total_usd": 5.0}}),
                    parse_prime_billing({"balance_usd": 142.0}),
                    parse_vast_billing({"available_usd": 18.5, "balance_usd": 0.0}),
                ]
            )
        )
        ends = [
            len(line.rstrip())
            for line in rendered.splitlines()
            if any(marker in line for marker in PROVIDER_MARKERS.values())
        ]
        self.assertEqual(len(ends), 3)
        self.assertEqual(len(set(ends)), 1, rendered)

    def test_one_vocabulary_covers_every_provider(self) -> None:
        """The regression this refactor exists to prevent: three ways to say
        'still loading', 'not set up' and 'could not read that'."""
        for status, expected in (
            (BillingStatus.LOADING, "checking..."),
            (BillingStatus.UNCONFIGURED, "not configured"),
            (BillingStatus.FAILED, "unavailable"),
        ):
            rows = [
                {
                    BillingStatus.LOADING: ProviderBilling.loading,
                    BillingStatus.UNCONFIGURED: ProviderBilling.unconfigured,
                }.get(status, lambda p: ProviderBilling.failed(p, "boom"))(provider)
                for provider in PROVIDER_BILLING_ORDER
            ]
            rendered = _plain(render_provider_billing(rows))
            with self.subTest(status=status):
                self.assertEqual(rendered.count(expected), len(PROVIDER_BILLING_ORDER))

    def test_an_unconfigured_provider_names_its_own_command(self) -> None:
        rendered = _plain(
            render_provider_billing(
                [ProviderBilling.unconfigured(p) for p in PROVIDER_BILLING_ORDER]
            )
        )
        for command in PROVIDER_SETUP_COMMANDS.values():
            self.assertIn(command, rendered)

    def test_a_failed_read_shows_its_reason(self) -> None:
        rendered = _plain(
            render_provider_billing(
                [ProviderBilling.failed(ComputeProvider.VAST, "Vast request failed (HTTP 429).")]
            )
        )
        self.assertIn("unavailable", rendered)
        self.assertIn("HTTP 429", rendered)

    def test_markup_in_an_error_is_escaped_rather_than_interpreted(self) -> None:
        rendered = _plain(
            render_provider_billing(
                [ProviderBilling.failed(ComputeProvider.MODAL, "Usage: modal [red]CMD[/red]")]
            )
        )
        self.assertIn("modal [red]CMD[/red]", rendered)

    def test_markup_in_a_provider_payload_is_escaped(self) -> None:
        """Resource names come from Prime's API, not from us."""
        rendered = _plain(
            render_provider_billing(
                [
                    parse_prime_billing(
                        {
                            "balance_usd": 1.0,
                            "recent_billings": [
                                {"amount_usd": 1.0, "resource_type": "[bold]pod[/bold]"}
                            ],
                        }
                    )
                ]
            )
        )
        self.assertIn("[bold]pod[/bold]", rendered)

    def test_storage_estimate_survives_a_failed_modal_billing_call(self) -> None:
        """The estimate is local data and does not depend on Modal billing.

        Rendering only the billing error hid the sole standing warning that
        cached models are still costing money.
        """
        rendered = _plain(
            render_provider_billing(
                [ProviderBilling.failed(ComputeProvider.MODAL, "modal CLI not found")],
                storage_snapshot=_snapshot(1400 * _GIB),
            )
        )
        self.assertIn("unavailable", rendered)
        self.assertIn("modal CLI not found", rendered)
        self.assertIn("storage est. $", rendered)
        self.assertIn("1,400 GiB cached", rendered)

    def test_storage_estimate_reports_the_free_tier_it_is_measured_against(self) -> None:
        rendered = _plain(
            render_provider_billing(
                [parse_modal_billing({"summary": {"total_usd": 12.5}})],
                storage_snapshot=_snapshot(1026 * _GIB),
            )
        )
        self.assertIn("$0.18/mo", rendered)
        self.assertIn("2.00 GiB over 1 TiB free", rendered)

    def test_only_modal_carries_the_volume_estimate(self) -> None:
        """The free tier and rate are Modal Volume's, not every provider's."""
        rendered = _plain(
            render_provider_billing(
                [parse_prime_billing({"balance_usd": 1.0})],
                storage_snapshot=_snapshot(1400 * _GIB),
            )
        )
        self.assertNotIn("cached", rendered)


if __name__ == "__main__":
    unittest.main()
