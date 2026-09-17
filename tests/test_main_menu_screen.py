from __future__ import annotations

import unittest
from unittest.mock import Mock, patch


from llm_launchpad.core.modal_auth import ModalAuthStatus
from llm_launchpad.core.artificial_analysis import ArtificialAnalysisAuthStatus
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import EndpointInfo
from llm_launchpad.core.provider_billing import (
    BalanceKind,
    BillingStatus,
    ProviderBilling,
)
from llm_launchpad.tui.screens.main_menu import (
    MainMenuScreen,
    ProviderBillingLoaded,
    _render_auth_status_block,
    _render_artificial_analysis_auth_status,
    _render_deployment_status,
    _render_hf_auth_status,
    _render_modal_auth_status,
    _should_show_in_panel,
)
from llm_launchpad.core.hf_auth import HuggingFaceAuthStatus

_GIB = 1024**3


class MainMenuStatusRenderTests(unittest.TestCase):
    def test_resume_cancels_deferred_timer_before_starting_secondary_refresh(self) -> None:
        screen = MainMenuScreen(username="alice")
        secondary_timer = Mock()
        screen._secondary_refresh_timer = secondary_timer
        screen._was_suspended = True

        with patch.object(screen, "_resume_refresh_timers"), patch.object(
            screen,
            "_refresh_panels",
        ), patch.object(screen, "_refresh_secondary_panels") as refresh_secondary:
            screen.on_screen_resume(Mock())

        secondary_timer.stop.assert_called_once_with()
        self.assertIsNone(screen._secondary_refresh_timer)
        refresh_secondary.assert_called_once_with()

    def test_each_provider_row_names_its_own_kind_of_resource(self) -> None:
        """A Vast rental was labelled a Prime pod by a two-way branch."""
        rows = [
            EndpointInfo(
                name="llamacpp-modal", app_id="ap-1", backend=BackendType.LLAMACPP,
                instance_name="on-modal", provider=ComputeProvider.MODAL, state="deployed",
            ),
            EndpointInfo(
                name="llp-prime-vllm-pod", app_id="pod-1", backend=BackendType.VLLM,
                instance_name="on-prime", provider=ComputeProvider.PRIME, state="running",
            ),
            EndpointInfo(
                name="llp-vast-llamacpp-rental", app_id="284412", backend=BackendType.LLAMACPP,
                instance_name="on-vast", provider=ComputeProvider.VAST, state="running",
            ),
        ]
        rendered = _render_deployment_status(rows)
        self.assertIn("Modal app:[/dim] llamacpp-modal", rendered)
        self.assertIn("Prime Intellect pod:[/dim] llp-prime-vllm-pod", rendered)
        self.assertIn("Vast.ai rental:[/dim] llp-vast-llamacpp-rental", rendered)

    def test_main_menu_bindings_do_not_include_q_quit(self) -> None:
        self.assertFalse(any(binding.key == "q" for binding in MainMenuScreen.BINDINGS))

    def test_render_deployment_status_empty_state(self) -> None:
        rendered = _render_deployment_status([])
        self.assertIn("No active launchpad apps", rendered)
        # The panel is already titled "Deployment Status"; the second heading
        # appeared only in the empty state, so the panel renamed itself when
        # the fleet emptied.
        self.assertNotIn("Fleet Pulse", rendered)
        self.assertNotIn("Fleet Pulse", _render_deployment_status(
            [
                EndpointInfo(
                    name="vllm-qwen",
                    app_id="ap-1",
                    state="running",
                    backend=BackendType.VLLM,
                    instance_name="qwen",
                )
            ]
        ))

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
        # A deployed Modal app without an explicit check is "not checked",
        # never "healthy": background refreshes must not wake the container
        # to learn whether it is warm.
        self.assertIn("0 healthy", rendered)
        self.assertIn("1 not checked", rendered)
        self.assertIn("health not checked", rendered)
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

    def test_render_aai_auth_status_shows_login_hint_when_missing(self) -> None:
        rendered = _render_artificial_analysis_auth_status(
            ArtificialAnalysisAuthStatus(authenticated=False)
        )
        self.assertIn("Artificial Analysis not authenticated", rendered)
        self.assertIn("llm-launchpad aai-auth login", rendered)

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


class ProviderBillingWiringTests(unittest.TestCase):
    """The panel's three workers and eleven attributes became one of each."""

    def _screen(self) -> MainMenuScreen:
        screen = MainMenuScreen(username="alice")
        screen._secondary_refresh_started = True
        return screen

    def test_every_provider_starts_out_checking(self) -> None:
        screen = self._screen()
        self.assertEqual(
            {row.status for row in screen._provider_billing.values()},
            {BillingStatus.LOADING},
        )

    def test_one_pass_reads_every_provider(self) -> None:
        """Three independent workers with three in-flight flags became one
        fan-out, so a slow provider no longer holds up the two beside it."""
        screen = self._screen()
        posted: list[ProviderBillingLoaded] = []
        rows = {
            ComputeProvider.MODAL: ProviderBilling.ready(
                ComputeProvider.MODAL, BalanceKind.SPEND_MTD, 1.0
            ),
            ComputeProvider.PRIME: ProviderBilling.ready(
                ComputeProvider.PRIME, BalanceKind.BALANCE, 2.0
            ),
            ComputeProvider.VAST: ProviderBilling.ready(
                ComputeProvider.VAST, BalanceKind.CREDIT, 3.0
            ),
        }
        with patch.object(screen, "post_message", posted.append), patch(
            "llm_launchpad.tui.screens.main_menu.load_modal_billing",
            return_value=rows[ComputeProvider.MODAL],
        ), patch(
            "llm_launchpad.tui.screens.main_menu.load_prime_billing",
            return_value=rows[ComputeProvider.PRIME],
        ), patch(
            "llm_launchpad.tui.screens.main_menu.load_vast_billing",
            return_value=rows[ComputeProvider.VAST],
        ):
            screen._run_load_provider_billing(
                modal_authenticated=True, prime_authenticated=True
            )

        loaded = [m.row for m in posted if isinstance(m, ProviderBillingLoaded)]
        self.assertEqual({row.provider for row in loaded}, set(rows))

    def test_a_loader_that_raises_becomes_that_provider_s_failure(self) -> None:
        screen = self._screen()
        posted: list[object] = []
        with patch.object(screen, "post_message", posted.append), patch(
            "llm_launchpad.tui.screens.main_menu.load_modal_billing",
            side_effect=RuntimeError("boom"),
        ), patch(
            "llm_launchpad.tui.screens.main_menu.load_prime_billing",
            return_value=ProviderBilling.loading(ComputeProvider.PRIME),
        ), patch(
            "llm_launchpad.tui.screens.main_menu.load_vast_billing",
            return_value=ProviderBilling.loading(ComputeProvider.VAST),
        ):
            screen._run_load_provider_billing(
                modal_authenticated=True, prime_authenticated=True
            )

        modal = next(
            m.row
            for m in posted
            if isinstance(m, ProviderBillingLoaded)
            and m.row.provider is ComputeProvider.MODAL
        )
        self.assertIs(modal.status, BillingStatus.FAILED)
        self.assertIn("boom", modal.error or "")

    def test_an_unauthenticated_provider_names_its_setup_command(self) -> None:
        """Modal had no unconfigured state and leaked a CLI error instead."""
        screen = self._screen()
        with patch.object(screen, "_update_billing_panel"):
            screen._apply_auth_to_billing(ComputeProvider.MODAL, False)
        row = screen._provider_billing[ComputeProvider.MODAL]
        self.assertIs(row.status, BillingStatus.UNCONFIGURED)
        self.assertEqual(row.setup_command, "modal setup")

    def test_an_auth_wobble_does_not_discard_a_reading_in_hand(self) -> None:
        screen = self._screen()
        screen._provider_billing[ComputeProvider.PRIME] = ProviderBilling.ready(
            ComputeProvider.PRIME, BalanceKind.BALANCE, 12.0
        )
        with patch.object(screen, "_update_billing_panel"):
            screen._apply_auth_to_billing(ComputeProvider.PRIME, False)
        self.assertEqual(screen._provider_billing[ComputeProvider.PRIME].amount_usd, 12.0)

    def test_authenticating_starts_a_read_once_the_deferred_pass_has_begun(self) -> None:
        screen = self._screen()
        with patch.object(screen, "_refresh_provider_billing") as refresh:
            screen._apply_auth_to_billing(ComputeProvider.PRIME, True)
        refresh.assert_called_once_with()

    def test_auth_before_first_paint_does_not_jump_the_deferred_pass(self) -> None:
        """Billing stays deferred so the first paint is not held up by it."""
        screen = MainMenuScreen(username="alice")
        with patch.object(screen, "_refresh_provider_billing") as refresh:
            screen._apply_auth_to_billing(ComputeProvider.PRIME, True)
        refresh.assert_not_called()

    def test_a_settled_pass_lets_the_next_one_start(self) -> None:
        screen = self._screen()
        screen._billing_refresh_inflight = True
        screen.on_provider_billing_finished(Mock())
        self.assertFalse(screen._billing_refresh_inflight)


if __name__ == "__main__":
    unittest.main()
