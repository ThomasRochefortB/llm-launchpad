from __future__ import annotations

import unittest

from llm_launchpad.core.modal_auth import ModalAuthStatus
from llm_launchpad.core.quick_deploy_refresh import ArtificialAnalysisAuthStatus
from llm_launchpad.protocol.enums import BackendType
from llm_launchpad.protocol.models import EndpointInfo, StorageSnapshot, StoredModelInfo
from llm_launchpad.tui.screens.main_menu import (
    MainMenuScreen,
    _render_auth_status_block,
    _render_artificial_analysis_auth_status,
    _render_billing_load_error,
    _render_billing_report,
    _render_deployment_status,
    _render_hf_auth_status,
    _render_modal_auth_status,
    _render_prime_billing_load_error,
    _render_prime_billing_report,
    _render_provider_billing_body,
    _should_show_in_panel,
)
from llm_launchpad.core.hf_auth import HuggingFaceAuthStatus

_GIB = 1024**3


class MainMenuStatusRenderTests(unittest.TestCase):
    def test_main_menu_bindings_do_not_include_q_quit(self) -> None:
        self.assertFalse(any(binding.key == "q" for binding in MainMenuScreen.BINDINGS))

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

    def test_render_billing_report_includes_storage_estimate_separately(self) -> None:
        payload = {"summary": {"total_usd": 12.5, "gpu_cost_usd": 8.1}}
        snapshot = StorageSnapshot(
            llamacpp_models=[
                StoredModelInfo(
                    backend=BackendType.LLAMACPP,
                    model_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
                    size_bytes=1024 * _GIB,
                    file_count=1,
                    source_volume="huggingface-cache",
                )
            ],
            vllm_models=[
                StoredModelInfo(
                    backend=BackendType.VLLM,
                    model_id="Qwen/Qwen3-4B",
                    size_bytes=2 * _GIB,
                    file_count=2,
                    source_volume="huggingface-cache",
                )
            ],
        )

        rendered = _render_billing_report(payload, storage_snapshot=snapshot)

        self.assertIn("[dim]total[/dim] [bold]$12.50[/bold]", rendered)
        self.assertIn("[dim]gpu[/dim] $8.10", rendered)
        self.assertIn("Launchpad storage est.", rendered)
        self.assertIn("$0.18/mo", rendered)
        self.assertIn("1,026 GiB cached", rendered)
        self.assertIn("2.00 GiB billable after 1 TiB free", rendered)

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
        self.assertNotIn("No billed usage", rendered)

    def test_render_billing_load_error_escapes_rich_markup_chars(self) -> None:
        rendered = _render_billing_load_error("Usage: modal [OPTIONS] COMMAND")
        self.assertIn("modal \\[OPTIONS\\] COMMAND", rendered)

    def test_render_prime_billing_report_shows_balance_and_resource_totals(self) -> None:
        payload = {
            "wallet_id": "wallet-1",
            "balance_usd": 41.25,
            "currency": "USD",
            "recent_billings": [
                {"amount_usd": 1.25, "resource_type": "compute"},
                {"amount_usd": 0.75, "resource_type": "compute"},
                {"amount_usd": "0.50", "resource_type": "disks"},
                {"amount_usd": None, "resource_type": "inference"},
            ],
        }

        rendered = _render_prime_billing_report(payload)
        self.assertIn("Prime Intellect Wallet", rendered)
        self.assertIn("[dim]balance[/dim] [bold]$41.25[/bold]", rendered)
        self.assertIn("[dim]recent charges[/dim]", rendered)
        self.assertIn("compute $2.00", rendered)
        self.assertIn("disks $0.50", rendered)
        self.assertNotIn("inference", rendered)

    def test_render_prime_billing_report_handles_unrecognized_payload(self) -> None:
        rendered = _render_prime_billing_report("not-json")
        self.assertIn("Prime Intellect Wallet", rendered)
        self.assertIn("Wallet data unavailable", rendered)
        self.assertIn("prime wallet", rendered)

    def test_render_prime_billing_report_without_rows_shows_placeholder(self) -> None:
        rendered = _render_prime_billing_report({"balance_usd": 5, "recent_billings": []})
        self.assertIn("$5.00", rendered)
        self.assertIn("No recent billing rows.", rendered)

    def test_render_prime_billing_load_error_escapes_rich_markup_chars(self) -> None:
        rendered = _render_prime_billing_load_error("denied [401]")
        self.assertIn("denied \\[401\\]", rendered)

    def test_provider_billing_body_combines_modal_and_prime_sections(self) -> None:
        body = _render_provider_billing_body(
            modal_payload={"summary": {"total_usd": 12.5}},
            modal_error=None,
            prime_state="loaded",
            prime_payload={"balance_usd": 3},
            prime_error=None,
        )
        self.assertIn("Workspace Spend", body)
        self.assertIn("$12.50", body)
        self.assertIn("Prime Intellect Wallet", body)
        self.assertIn("[bold]$3.00[/bold]", body)

    def test_provider_billing_body_reports_unauthenticated_prime(self) -> None:
        body = _render_provider_billing_body(
            modal_payload=None,
            modal_error=None,
            prime_state="unavailable",
            prime_payload=None,
            prime_error=None,
        )
        self.assertIn("Not authenticated (run: prime login)", body)

    def test_provider_billing_body_keeps_modal_error_while_prime_loads(self) -> None:
        body = _render_provider_billing_body(
            modal_payload=None,
            modal_error="Modal timed out",
            prime_state="loading",
            prime_payload=None,
            prime_error=None,
        )
        self.assertIn("Modal timed out", body)
        self.assertIn("Refreshing wallet...", body)

    def test_render_hf_auth_status_hides_authenticated_username(self) -> None:
        rendered = _render_hf_auth_status(
            HuggingFaceAuthStatus(authenticated=True, username="alice")
        )
        self.assertIn("Hugging Face authenticated", rendered)
        self.assertNotIn("alice", rendered)

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

    def test_render_aai_auth_status_shows_authenticated_tier(self) -> None:
        rendered = _render_artificial_analysis_auth_status(
            ArtificialAnalysisAuthStatus(authenticated=True, tier="free")
        )
        self.assertIn("Artificial Analysis authenticated", rendered)
        self.assertIn("free tier", rendered)

    def test_render_aai_auth_status_shows_environment_hint_when_missing(self) -> None:
        rendered = _render_artificial_analysis_auth_status(
            ArtificialAnalysisAuthStatus(authenticated=False)
        )
        self.assertIn("Artificial Analysis not authenticated", rendered)
        self.assertIn("ARTIFICIAL_ANALYSIS_API_KEY", rendered)

    def test_render_aai_auth_status_shows_invalid_key_error(self) -> None:
        rendered = _render_artificial_analysis_auth_status(
            ArtificialAnalysisAuthStatus(
                authenticated=False,
                error="Invalid Artificial Analysis API key",
            )
        )
        self.assertIn("auth check failed", rendered)
        self.assertIn("Invalid Artificial Analysis API key", rendered)

    def test_render_auth_status_block_hides_modal_profile_details(self) -> None:
        rendered = _render_auth_status_block(
            username="default",
            modal_status=ModalAuthStatus(authenticated=True, profile="default"),
        )
        self.assertIn("Modal authenticated", rendered)
        self.assertNotIn("Modal profile: default", rendered)
        self.assertNotIn("default", rendered)

    def test_render_auth_status_block_includes_provider_auth_lines(self) -> None:
        rendered = _render_auth_status_block(
            username="default",
            modal_status=ModalAuthStatus(authenticated=False, profile="default"),
            hf_status=HuggingFaceAuthStatus(authenticated=True, username="alice"),
            aai_status=ArtificialAnalysisAuthStatus(authenticated=True, tier="pro"),
        )
        self.assertIn("Modal not authenticated", rendered)
        self.assertNotIn("Modal profile: default", rendered)
        self.assertIn("Hugging Face authenticated", rendered)
        self.assertIn("Artificial Analysis authenticated", rendered)
        self.assertNotIn("alice", rendered)


if __name__ == "__main__":
    unittest.main()
