from __future__ import annotations

import unittest

from llm_launchpad.protocol.enums import BackendType
from llm_launchpad.protocol.models import EndpointInfo
from llm_launchpad.tui.screens.main_menu import (
    _render_billing_load_error,
    _render_billing_report,
    _render_deployment_status,
    _should_show_in_panel,
)


class MainMenuStatusRenderTests(unittest.TestCase):
    def test_render_deployment_status_empty_state(self) -> None:
        rendered = _render_deployment_status([])
        self.assertIn("No active launchpad deployments", rendered)

    def test_render_deployment_status_includes_counts_and_rows(self) -> None:
        rows = [
            EndpointInfo(
                name="vllm-qwen",
                app_id="ap-1",
                state="running",
                backend=BackendType.VLLM,
                instance_name="qwen",
            ),
            EndpointInfo(
                name="llamacpp-phi",
                app_id="ap-2",
                state="deploying",
                backend=BackendType.LLAMACPP,
                instance_name="phi",
            ),
            EndpointInfo(
                name="vllm-broken",
                app_id="ap-3",
                state="failed",
                backend=BackendType.VLLM,
                instance_name="broken",
            ),
        ]

        rendered = _render_deployment_status(rows)
        self.assertIn("deployments=3", rendered)
        self.assertIn("healthy=1", rendered)
        self.assertIn("pending=1", rendered)
        self.assertIn("issues=1", rendered)
        self.assertIn("vllm-qwen", rendered)
        self.assertIn("llamacpp-phi", rendered)
        self.assertIn("vllm-broken", rendered)

    def test_render_deployment_status_truncates_long_names(self) -> None:
        rows = [
            EndpointInfo(
                name="vllm-very-very-very-very-long-application-name",
                app_id="ap-1",
                state="running",
                backend=BackendType.VLLM,
                instance_name="very-long-instance-name",
            )
        ]

        rendered = _render_deployment_status(rows)
        self.assertIn("...", rendered)

    def test_should_show_in_panel_hides_stopped(self) -> None:
        self.assertFalse(_should_show_in_panel("stopped"))
        self.assertFalse(_should_show_in_panel("stopping"))
        self.assertTrue(_should_show_in_panel("running"))
        self.assertTrue(_should_show_in_panel("deploying"))
        self.assertTrue(_should_show_in_panel("failed"))

    def test_render_billing_report_includes_total_and_current_month_label(self) -> None:
        payload = {
            "summary": {
                "total_usd": 12.5,
                "gpu_cost_usd": 8.1,
                "period_start": "2026-02-01",
                "period_end": "2026-02-18",
            }
        }

        rendered = _render_billing_report(payload)
        self.assertIn("Workspace Spend", rendered)
        self.assertIn("Current month spend", rendered)
        self.assertIn("$12.50", rendered)
        self.assertIn("$8.10", rendered)

    def test_render_billing_report_handles_unrecognized_payload(self) -> None:
        rendered = _render_billing_report("not-json")
        self.assertIn("Billing data unavailable", rendered)

    def test_render_billing_report_aggregates_list_payload(self) -> None:
        payload = [
            {"Description": "app-a", "Interval Start": "2026-02-13T00:00:00", "Cost": "1.25"},
            {"Description": "app-b", "Interval Start": "2026-02-18T00:00:00", "Cost": "2.75"},
        ]
        rendered = _render_billing_report(payload)
        self.assertIn("Workspace Spend", rendered)
        self.assertIn("Current month spend", rendered)
        self.assertIn("$4.00", rendered)

    def test_render_billing_report_empty_list_shows_zero_total(self) -> None:
        rendered = _render_billing_report([])
        self.assertIn("Workspace Spend", rendered)
        self.assertIn("Current month spend", rendered)
        self.assertIn("$0.00", rendered)
        self.assertIn("No billed usage", rendered)

    def test_render_billing_load_error_escapes_rich_markup_chars(self) -> None:
        rendered = _render_billing_load_error("Usage: modal [OPTIONS] COMMAND")
        self.assertIn("modal \\[OPTIONS\\] COMMAND", rendered)


if __name__ == "__main__":
    unittest.main()
