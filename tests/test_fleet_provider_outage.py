"""A provider that cannot be reached must not read as an empty fleet."""

from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from textual.app import App
from textual.widgets import Static

from llm_launchpad.core.backend import ModalCliError, ModalListAppsResult
from llm_launchpad.core.prime_auth import PrimeAuthStatus
from llm_launchpad.core.vast_auth import VastCredentials
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import EndpointInfo, FleetDiscovery, ProviderListing
from llm_launchpad.tui.app import TuiApp
from llm_launchpad.tui.fleet_status import provider_outage_lines, retained_providers
from llm_launchpad.tui.format import format_age
from llm_launchpad.tui.screens.main_menu import _render_deployment_status
from llm_launchpad.tui.screens.manage import ManageScreen
from llm_launchpad.tui.widgets.adaptive_table import AdaptiveDataTable
from llm_launchpad.tui.workers import EndpointsLoaded


def _endpoint(
    name: str,
    *,
    provider: ComputeProvider = ComputeProvider.MODAL,
    state: str = "running",
    backend: BackendType = BackendType.VLLM,
) -> EndpointInfo:
    return EndpointInfo(
        name=name,
        app_id=f"ap-{name}",
        state=state,
        backend=backend,
        instance_name=name,
        provider=provider,
        web_url=f"https://example.test/{name}",
    )


def _vast_unconfigured() -> VastCredentials:
    return VastCredentials(api_key="")


class FleetDiscoveryRetentionTests(unittest.TestCase):
    """`discover_fleet` reports each provider's own outcome."""

    def _app(self) -> TuiApp:
        app = TuiApp()
        app._merge_deploy_connection_cache = lambda rows: None  # type: ignore[method-assign]
        return app

    def test_failed_provider_keeps_its_last_rows_and_leaves_prune_scope(self) -> None:
        app = self._app()
        rows = [_endpoint("vllm-qwen3")]
        with (
            patch("llm_launchpad.tui.app.ModalBackend.is_cli_available", return_value=True),
            patch(
                "llm_launchpad.tui.app.get_prime_auth_status",
                return_value=PrimeAuthStatus(authenticated=False),
            ),
            patch("llm_launchpad.tui.app.resolve_vast_credentials", side_effect=_vast_unconfigured),
            patch(
                "llm_launchpad.tui.app.ModalBackend.list_apps_result",
                return_value=ModalListAppsResult(rows=rows),
            ),
        ):
            first = app.discover_fleet()

        self.assertEqual([row.name for row in first.rows], ["vllm-qwen3"])
        self.assertEqual(first.prune_providers, (ComputeProvider.MODAL,))
        self.assertEqual(first.unavailable, ())

        with (
            patch("llm_launchpad.tui.app.ModalBackend.is_cli_available", return_value=True),
            patch(
                "llm_launchpad.tui.app.get_prime_auth_status",
                return_value=PrimeAuthStatus(authenticated=False),
            ),
            patch("llm_launchpad.tui.app.resolve_vast_credentials", side_effect=_vast_unconfigured),
            patch(
                "llm_launchpad.tui.app.ModalBackend.list_apps_result",
                return_value=ModalListAppsResult(
                    error=ModalCliError(message="Timed out while querying Modal app list.")
                ),
            ),
        ):
            second = app.discover_fleet()

        # The deployment is still running and still billing, so it is still shown.
        self.assertEqual([row.name for row in second.rows], ["vllm-qwen3"])
        # Nothing may be pruned on the word of a provider that did not answer.
        self.assertEqual(second.prune_providers, ())
        self.assertEqual(len(second.unavailable), 1)
        listing = second.unavailable[0]
        self.assertEqual(listing.provider, ComputeProvider.MODAL)
        self.assertEqual(listing.error, "Timed out while querying Modal app list.")
        self.assertTrue(listing.is_retained)
        self.assertIsNotNone(listing.retrieved_at_epoch)

    def test_first_pass_failure_reports_the_outage_with_no_rows(self) -> None:
        app = self._app()
        with (
            patch("llm_launchpad.tui.app.ModalBackend.is_cli_available", return_value=False),
            patch(
                "llm_launchpad.tui.app.get_prime_auth_status",
                return_value=PrimeAuthStatus(authenticated=True),
            ),
            patch("llm_launchpad.tui.app.resolve_vast_credentials", side_effect=_vast_unconfigured),
            patch(
                "llm_launchpad.tui.app.PrimeBackend.list_deployments",
                autospec=True,
                side_effect=RuntimeError("prime API returned 503"),
            ),
        ):
            discovery = app.discover_fleet()

        self.assertEqual(discovery.rows, [])
        self.assertEqual(discovery.prune_providers, ())
        self.assertEqual(len(discovery.unavailable), 1)
        self.assertEqual(discovery.unavailable[0].error, "prime API returned 503")
        self.assertFalse(discovery.unavailable[0].is_retained)

    def test_unconfigured_provider_is_not_reported_as_unavailable(self) -> None:
        app = self._app()
        with (
            patch("llm_launchpad.tui.app.ModalBackend.is_cli_available", return_value=False),
            patch(
                "llm_launchpad.tui.app.get_prime_auth_status",
                return_value=PrimeAuthStatus(authenticated=False),
            ),
            patch("llm_launchpad.tui.app.resolve_vast_credentials", side_effect=_vast_unconfigured),
        ):
            discovery = app.discover_fleet()

        self.assertEqual(discovery.listings, ())
        self.assertEqual(discovery.unavailable, ())

    def test_signing_out_of_a_provider_drops_its_retained_rows(self) -> None:
        app = self._app()
        vast_rows = [_endpoint("llamacpp-qwen3", provider=ComputeProvider.VAST)]
        with (
            patch("llm_launchpad.tui.app.ModalBackend.is_cli_available", return_value=False),
            patch(
                "llm_launchpad.tui.app.get_prime_auth_status",
                return_value=PrimeAuthStatus(authenticated=False),
            ),
            patch(
                "llm_launchpad.tui.app.resolve_vast_credentials",
                return_value=VastCredentials(api_key="key"),
            ),
            patch(
                "llm_launchpad.tui.app.VastDeploymentBackend.list_deployments",
                autospec=True,
                return_value=vast_rows,
            ),
        ):
            self.assertEqual(len(app.discover_fleet().rows), 1)

        with (
            patch("llm_launchpad.tui.app.ModalBackend.is_cli_available", return_value=False),
            patch(
                "llm_launchpad.tui.app.get_prime_auth_status",
                return_value=PrimeAuthStatus(authenticated=False),
            ),
            patch("llm_launchpad.tui.app.resolve_vast_credentials", side_effect=_vast_unconfigured),
        ):
            discovery = app.discover_fleet()

        self.assertEqual(discovery.rows, [])
        self.assertEqual(discovery.unavailable, ())

    def test_healthy_provider_rows_survive_another_provider_failing(self) -> None:
        app = self._app()
        with (
            patch("llm_launchpad.tui.app.ModalBackend.is_cli_available", return_value=True),
            patch(
                "llm_launchpad.tui.app.get_prime_auth_status",
                return_value=PrimeAuthStatus(authenticated=True),
            ),
            patch("llm_launchpad.tui.app.resolve_vast_credentials", side_effect=_vast_unconfigured),
            patch(
                "llm_launchpad.tui.app.ModalBackend.list_apps_result",
                return_value=ModalListAppsResult(rows=[_endpoint("vllm-modal")]),
            ),
            patch(
                "llm_launchpad.tui.app.PrimeBackend.list_deployments",
                autospec=True,
                side_effect=RuntimeError("prime API returned 503"),
            ),
        ):
            discovery = app.discover_fleet()

        self.assertEqual([row.name for row in discovery.rows], ["vllm-modal"])
        self.assertEqual(discovery.prune_providers, (ComputeProvider.MODAL,))
        self.assertEqual(
            [listing.provider for listing in discovery.unavailable], [ComputeProvider.PRIME]
        )


class ProviderOutageWordingTests(unittest.TestCase):
    def test_retained_rows_are_reported_with_their_age(self) -> None:
        now = time.time()
        discovery = FleetDiscovery(
            listings=(
                ProviderListing(
                    provider=ComputeProvider.VAST,
                    rows=(_endpoint("llamacpp-qwen3", provider=ComputeProvider.VAST),),
                    error="Vast authentication failed.",
                    retrieved_at_epoch=now - 240,
                ),
            )
        )
        (line,) = provider_outage_lines(discovery, now=now)
        self.assertIn("Vast.ai unavailable", line)
        self.assertIn("Vast authentication failed.", line)
        self.assertIn("showing 1 endpoint from 4m ago", line)
        self.assertEqual(retained_providers(discovery), {ComputeProvider.VAST.value})

    def test_provider_with_no_retained_rows_says_so(self) -> None:
        discovery = FleetDiscovery(
            listings=(
                ProviderListing(
                    provider=ComputeProvider.PRIME,
                    error="prime API returned 503",
                ),
            )
        )
        (line,) = provider_outage_lines(discovery)
        self.assertIn("Prime Intellect unavailable", line)
        self.assertIn("its deployments are not listed", line)
        self.assertEqual(retained_providers(discovery), set())

    def test_markup_in_a_provider_error_cannot_break_rendering(self) -> None:
        discovery = FleetDiscovery(
            listings=(
                ProviderListing(provider=ComputeProvider.MODAL, error="failed [bold]now[/]"),
            )
        )
        (line,) = provider_outage_lines(discovery)
        self.assertIn(r"\[bold]", line)

    def test_format_age_uses_the_coarsest_honest_unit(self) -> None:
        self.assertEqual(format_age(12), "just now")
        self.assertEqual(format_age(180), "3m ago")
        self.assertEqual(format_age(7200), "2h ago")
        self.assertEqual(format_age(200000), "2d ago")


class DeploymentPanelOutageTests(unittest.TestCase):
    def test_panel_never_calls_an_unreachable_provider_an_empty_fleet(self) -> None:
        discovery = FleetDiscovery(
            listings=(
                ProviderListing(provider=ComputeProvider.MODAL, error="modal: command not found"),
            )
        )
        body = _render_deployment_status([], discovery=discovery)
        self.assertNotIn("No active launchpad apps", body)
        self.assertIn("Modal unavailable", body)

    def test_panel_reports_the_outage_above_the_rows_it_still_has(self) -> None:
        discovery = FleetDiscovery(
            listings=(
                ProviderListing(
                    provider=ComputeProvider.MODAL,
                    rows=(_endpoint("vllm-qwen3"),),
                    error="modal: command not found",
                    retrieved_at_epoch=time.time() - 90,
                ),
            )
        )
        body = _render_deployment_status([_endpoint("vllm-qwen3")], discovery=discovery)
        lines = body.splitlines()
        self.assertIn("Modal unavailable", lines[0])
        self.assertIn("qwen3", body)

    def test_panel_without_a_discovery_keeps_the_empty_wording(self) -> None:
        self.assertIn("No active launchpad apps", _render_deployment_status([]))


class _OutageApp(App[None]):
    """Serves one discovery result to the manage screen."""

    def __init__(self, rows: list[EndpointInfo], discovery: FleetDiscovery) -> None:
        super().__init__()
        self.rows = rows
        self.discovery = discovery

    def begin_endpoint_refresh(self, receiver: object, force: bool = False) -> None:
        receiver.post_message(  # type: ignore[attr-defined]
            EndpointsLoaded(rows=list(self.rows), discovery=self.discovery)
        )


class ManageScreenOutageTests(unittest.IsolatedAsyncioTestCase):
    async def test_retained_rows_stay_actionable_and_are_marked_stale(self) -> None:
        row = _endpoint("vllm-qwen3")
        discovery = FleetDiscovery(
            listings=(
                ProviderListing(
                    provider=ComputeProvider.MODAL,
                    rows=(row,),
                    error="Timed out while querying Modal app list.",
                    retrieved_at_epoch=time.time() - 120,
                ),
            )
        )
        app = _OutageApp([row], discovery)
        async with app.run_test() as pilot:
            app.push_screen(ManageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ManageScreen)
            table = screen.query_one("#manage-endpoint-table", AdaptiveDataTable)
            status = str(screen.query_one("#manage-status", Static).content)
            detail = str(screen.query_one("#manage-selection-detail", Static).content)

            self.assertEqual(table.row_count, 1)
            self.assertIn("Fleet partly refreshed", status)
            self.assertIn("Modal unavailable", status)
            self.assertIn("showing 1 endpoint from 2m ago", status)
            self.assertIn("may be out of date", detail)
            # Actions stay offered: stopping a rental that is still billing must
            # not require the provider listing to recover first.
            self.assertIn("stop", detail)

    async def test_empty_fleet_with_an_outage_does_not_claim_there_is_nothing(self) -> None:
        discovery = FleetDiscovery(
            listings=(
                ProviderListing(
                    provider=ComputeProvider.VAST,
                    error="Vast authentication failed.",
                ),
            )
        )
        app = _OutageApp([], discovery)
        async with app.run_test() as pilot:
            app.push_screen(ManageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ManageScreen)
            status = str(screen.query_one("#manage-status", Static).content)

            self.assertNotIn("No managed endpoints found", status)
            self.assertIn("No endpoints could be listed", status)
            self.assertIn("Vast.ai unavailable", status)

    async def test_a_healthy_fleet_still_reports_a_plain_refresh(self) -> None:
        row = _endpoint("vllm-qwen3")
        discovery = FleetDiscovery(
            listings=(
                ProviderListing(
                    provider=ComputeProvider.MODAL,
                    rows=(row,),
                    retrieved_at_epoch=time.time(),
                ),
            )
        )
        app = _OutageApp([row], discovery)
        async with app.run_test() as pilot:
            app.push_screen(ManageScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ManageScreen)
            status = str(screen.query_one("#manage-status", Static).content)
            detail = str(screen.query_one("#manage-selection-detail", Static).content)

            self.assertIn("Fleet refreshed.", status)
            self.assertNotIn("unavailable", status)
            self.assertNotIn("may be out of date", detail)


if __name__ == "__main__":
    unittest.main()
