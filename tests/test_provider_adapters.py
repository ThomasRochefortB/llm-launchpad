"""Normalized provider adapters: listings, stop outcomes, fleet policy."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from llm_launchpad.core.provider_adapters import (
    ProviderResource,
    discover_fleet,
    is_live_row,
    listing_adapter,
    stop_with_orchestrator,
)
from llm_launchpad.protocol.enums import BackendType, ComputeProvider, OperationType
from llm_launchpad.protocol.events import LogEvent, OperationCompleteEvent
from llm_launchpad.protocol.models import EndpointInfo, FleetDiscovery


class ListingAdapterTests(unittest.TestCase):
    def test_none_listing_becomes_unavailable_not_empty(self) -> None:
        with patch(
            "llm_launchpad.core.providers.deployment_lister",
            return_value=lambda: None,
        ):
            listing = listing_adapter(ComputeProvider.MODAL).list()
        self.assertFalse(listing.available)
        self.assertEqual(listing.rows, ())

    def test_raising_lister_becomes_error_listing(self) -> None:
        def _boom():  # type: ignore[no-untyped-def]
            raise RuntimeError("marketplace down")

        with patch(
            "llm_launchpad.core.providers.deployment_lister",
            return_value=_boom,
        ):
            listing = listing_adapter(ComputeProvider.PRIME).list()
        self.assertFalse(listing.available)
        self.assertIn("marketplace down", listing.error or "")

    def test_successful_empty_listing_is_available(self) -> None:
        with patch(
            "llm_launchpad.core.providers.deployment_lister",
            return_value=lambda: [],
        ):
            listing = listing_adapter(ComputeProvider.MODAL).list()
        self.assertTrue(listing.available)
        self.assertEqual(listing.rows, ())


class StopAdapterTests(unittest.TestCase):
    def test_successful_stop_names_storage_consequence(self) -> None:
        orch = unittest.mock.MagicMock()
        orch.stop_app.return_value = [
            OperationCompleteEvent(operation=OperationType.STOP, success=True),
        ]
        cleanup = stop_with_orchestrator(
            orch,
            ProviderResource(
                provider=ComputeProvider.PRIME, app_name="x", resource_id="pod-1"
            ),
        )
        self.assertEqual(cleanup.disposition.value, "confirmed")
        self.assertTrue(cleanup.remains_billable)
        orch.stop_app.assert_called_once_with(
            BackendType.LLAMACPP, app_name="x", app_id="pod-1",
            provider=ComputeProvider.PRIME,
        )

    def test_stop_without_completion_is_failed(self) -> None:
        orch = unittest.mock.MagicMock()
        orch.stop_app.return_value = [LogEvent(line="stopping")]
        cleanup = stop_with_orchestrator(
            orch,
            ProviderResource(provider=ComputeProvider.MODAL, app_name="x"),
        )
        self.assertEqual(cleanup.disposition.value, "failed")


class FleetPolicyTests(unittest.TestCase):
    def test_only_successful_listings_authorize_pruning(self) -> None:
        listings = discover_fleet((ComputeProvider.MODAL,))
        discovery = FleetDiscovery(listings=listings)
        self.assertEqual(
            set(discovery.prune_providers),
            {listing.provider for listing in listings if listing.available},
        )

    def test_terminal_rows_are_not_live(self) -> None:
        self.assertFalse(is_live_row(EndpointInfo(name="x", state="stopped")))
        self.assertTrue(is_live_row(EndpointInfo(name="x", state="running")))


if __name__ == "__main__":
    unittest.main()
