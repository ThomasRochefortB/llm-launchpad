from __future__ import annotations

import unittest

from llm_launchpad.core.prime_live import (
    BudgetExceeded,
    PrimeBudgetGuard,
    PrimeLiveReport,
    PrimeLiveStage,
    PrimeResourceLedger,
    redact_live_value,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class PrimeBudgetGuardTests(unittest.TestCase):
    def test_tracks_only_elapsed_billable_time(self) -> None:
        clock = _Clock()
        guard = PrimeBudgetGuard(clock=clock)
        guard.register("pod", "pod-1", 3.60)
        clock.now += 500
        self.assertAlmostEqual(guard.estimated_cost_usd(), 0.50)
        guard.close("pod-1")
        clock.now += 500
        self.assertAlmostEqual(guard.estimated_cost_usd(), 0.50)

    def test_reservation_stops_before_cleanup_margin(self) -> None:
        guard = PrimeBudgetGuard(cap_usd=3.0, cleanup_reserve_usd=0.30)
        with self.assertRaisesRegex(BudgetExceeded, "operational cutoff"):
            guard.require_capacity(
                hourly_rate_usd=5.40,
                maximum_runtime_seconds=1801,
                description="expensive stage",
            )

    def test_duplicate_registration_does_not_restart_billing_clock(self) -> None:
        clock = _Clock()
        guard = PrimeBudgetGuard(clock=clock)
        guard.register("pod", "pod-1", 1.0)
        clock.now += 60
        guard.register("pod", "pod-1", 100.0)
        self.assertAlmostEqual(guard.estimated_cost_usd(), 1.0 / 60.0)


class PrimeLiveReportTests(unittest.TestCase):
    def test_report_redacts_known_and_structured_secrets(self) -> None:
        report = PrimeLiveReport(run_id="run", commit="abc", budget_cap_usd=3.0)
        report.stages.append(
            PrimeLiveStage(
                name="auth",
                evidence={
                    "known": "secret-value",
                    "header": "Authorization: Bearer another-secret",
                    "env": "VLLM_API_KEY=third-secret",
                },
            )
        )
        payload = str(report.to_dict(("secret-value",)))
        self.assertNotIn("secret-value", payload)
        self.assertNotIn("another-secret", payload)
        self.assertNotIn("third-secret", payload)
        self.assertIn("[redacted]", payload)

    def test_standalone_redactor_handles_api_key_assignment(self) -> None:
        self.assertEqual(redact_live_value("api_key=abc"), "api_key=[redacted]")

    def test_standalone_redactor_handles_quoted_api_key_list(self) -> None:
        redacted = redact_live_value("{'api_key': ['endpoint-secret']}")

        self.assertNotIn("endpoint-secret", redacted)
        self.assertIn("'api_key': [redacted]", redacted)

    def test_standalone_redactor_preserves_bearer_header_quote(self) -> None:
        redacted = redact_live_value("-H 'Authorization: Bearer endpoint-secret'")

        self.assertEqual(redacted, "-H 'Authorization: Bearer [redacted]'")

    def test_standalone_redactor_removes_tunnel_binding_secret(self) -> None:
        redacted = redact_live_value('binding_secret = "tunnel-secret"')

        self.assertNotIn("tunnel-secret", redacted)


class PrimeResourceLedgerTests(unittest.TestCase):
    def test_ledger_tracks_only_open_resources(self) -> None:
        ledger = PrimeResourceLedger()
        ledger.add_pod("pod-1")
        ledger.add_disk("disk-1")
        ledger.close_pod("pod-1")
        self.assertEqual(ledger.pod_ids, set())
        self.assertEqual(ledger.disk_ids, {"disk-1"})


if __name__ == "__main__":
    unittest.main()
