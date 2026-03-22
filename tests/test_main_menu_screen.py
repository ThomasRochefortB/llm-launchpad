from __future__ import annotations

import unittest

from llm_launchpad.core.modal_auth import ModalAuthStatus
from llm_launchpad.protocol.enums import BackendType
from llm_launchpad.protocol.models import EndpointInfo
from llm_launchpad.tui.screens.main_menu import (
    _render_auth_status_block,
    _render_billing_load_error,
    _render_billing_report,
    _render_deployment_status,
    _render_hf_auth_status,
    _render_modal_auth_status,
    _should_show_in_panel,
)
from llm_launchpad.core.hf_auth import HuggingFaceAuthStatus


class MainMenuStatusRenderTests(unittest.TestCase):
    def test_render_deployment_status_empty_state(self) -> None:
        rendered = _render_deployment_status([])
        self.assertIn("No active launchpad apps", rendered)

    def test_render_deployment_status_includes_counts_and_rows(self) -> None:
        rows = [
            EndpointInfo(
                name="vllm-qwen",
                app_id="ap-1",
                state="running",
                backend=BackendType.VLLM,
                instance_name="qwen",
                web_url="https://alice--vllm-qwen-serve.modal.run",
                served_model_name="Qwen3-4B",
                model_name="Qwen/Qwen3-4B",
            ),
            EndpointInfo(
                name="llamacpp-phi",
                app_id="ap-2",
                state="deploying",
                backend=BackendType.LLAMACPP,
                instance_name="phi",
                web_url="https://alice--llamacpp-phi-serve-abc123.modal.run",
                repo_id="unsloth/phi-gguf",
                quant="Q4_K_M",
            ),
            EndpointInfo(
                name="vllm-broken",
                app_id="ap-3",
                state="failed",
                backend=BackendType.VLLM,
                instance_name="broken",
            ),
        ]

        rendered = _render_deployment_status(rows, username="alice")
        self.assertIn("3 active launchpad apps", rendered)
        self.assertIn("1 healthy", rendered)
        self.assertIn("1 in progress", rendered)
        self.assertIn("1 error", rendered)
        self.assertIn("qwen", rendered)
        self.assertIn("phi", rendered)
        self.assertIn("broken", rendered)
        self.assertIn("Modal app:", rendered)
        self.assertIn("ap-1", rendered)
        self.assertIn("modal: running", rendered)
        self.assertIn("modal: deploying", rendered)
        self.assertIn("modal: failed", rendered)
        self.assertIn("Base URL:", rendered)
        self.assertIn("Display name:", rendered)
        self.assertIn("Model ID:", rendered)
        self.assertIn("Qwen3-4B", rendered)
        self.assertIn("https://alice--vllm-qwen-serve.modal.run", rendered)

    def test_render_deployment_status_derives_base_url_when_web_url_missing(self) -> None:
        rows = [
            EndpointInfo(
                name="vllm-very-very-very-very-long-application-name",
                app_id="ap-1",
                state="running",
                backend=BackendType.VLLM,
                instance_name="very-long-instance-name",
            )
        ]

        rendered = _render_deployment_status(rows, username="alice")
        self.assertIn("https://alice--vllm-very-very-very-very-", rendered)
        self.assertIn("long-application-name-serve.modal.run", rendered)
        self.assertIn("API key", rendered)

    def test_should_show_in_panel_hides_stopped(self) -> None:
        self.assertFalse(_should_show_in_panel("stopped"))
        self.assertFalse(_should_show_in_panel("stopping"))
        self.assertTrue(_should_show_in_panel("ephemeral"))
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

    def test_render_hf_auth_status_shows_authenticated_username(self) -> None:
        rendered = _render_hf_auth_status(
            HuggingFaceAuthStatus(authenticated=True, username="alice")
        )
        self.assertIn("Hugging Face authenticated as alice", rendered)

    def test_render_hf_auth_status_shows_login_hint_when_unauthenticated(self) -> None:
        rendered = _render_hf_auth_status(HuggingFaceAuthStatus(authenticated=False))
        self.assertIn("Hugging Face not authenticated", rendered)
        self.assertIn("hf auth login", rendered)

    def test_render_hf_auth_status_shows_invalid_token_error(self) -> None:
        rendered = _render_hf_auth_status(
            HuggingFaceAuthStatus(authenticated=False, error="Invalid Hugging Face token")
        )
        self.assertIn("auth check failed", rendered)
        self.assertIn("Invalid Hugging Face token", rendered)

    def test_render_modal_auth_status_shows_authenticated_state(self) -> None:
        rendered = _render_modal_auth_status(ModalAuthStatus(authenticated=True))
        self.assertIn("Modal authenticated", rendered)

    def test_render_modal_auth_status_shows_login_hint_when_unauthenticated(self) -> None:
        rendered = _render_modal_auth_status(ModalAuthStatus(authenticated=False))
        self.assertIn("Modal not authenticated", rendered)
        self.assertIn("modal setup", rendered)

    def test_render_auth_status_block_shows_profile_not_username(self) -> None:
        rendered = _render_auth_status_block(
            username="default",
            modal_status=ModalAuthStatus(authenticated=True, profile="default"),
        )
        self.assertIn("Modal authenticated", rendered)
        self.assertIn("Modal profile: default", rendered)
        self.assertNotIn("authenticated as: default", rendered)

    def test_render_auth_status_block_includes_both_modal_and_hf_lines(self) -> None:
        rendered = _render_auth_status_block(
            username="default",
            modal_status=ModalAuthStatus(authenticated=False, profile="default"),
            hf_status=HuggingFaceAuthStatus(authenticated=True, username="alice"),
        )
        self.assertIn("Modal not authenticated", rendered)
        self.assertIn("Modal profile: default", rendered)
        self.assertIn("Hugging Face authenticated as alice", rendered)


if __name__ == "__main__":
    unittest.main()
