from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import Mock, patch

import requests

from llm_launchpad.core.vast_auth import VastCredentials
from llm_launchpad.core.vast_backend import (
    VastApiError, VastBackend, parse_vast_offer, vast_offer_search_payload,
)
from llm_launchpad.protocol.models import VastOfferQuery


def offer_payload(**changes: object) -> dict[str, object]:
    """Synthetic fixture following Vast's raw API units; not a live price."""
    row: dict[str, object] = {
        "id": 1001, "machine_id": 42, "gpu_name": "RTX 4090", "num_gpus": 1,
        "gpu_ram": 24576, "cpu_ram": 64000, "disk_space": 200,
        "gpu_arch": "nvidia", "cpu_arch": "amd64",
        "cuda_max_good": 12.8, "compute_cap": 890,
        "verification": "verified", "rentable": True, "rented": False,
        "is_bid": False, "reliability": 0.999, "datacenter": False,
        "duration": 86400, "geolocation": "US",
        "dph_base": 0.4, "dph_total": 0.42,
        "inet_down_cost": 0.002, "inet_up_cost": 0.003,
    }
    return {**row, **changes}


class VastOfferTests(unittest.TestCase):
    def test_payload_filters_and_prices_requested_disk(self) -> None:
        query = VastOfferQuery(gpu_type="RTX_4090", country="us", disk_gb=120, datacenter_only=True)
        payload = vast_offer_search_payload(query)
        self.assertEqual(payload["type"], "on-demand")
        self.assertEqual(payload["allocated_storage"], 120)
        self.assertEqual(payload["disk_space"], {"gte": 120})
        self.assertEqual(payload["gpu_name"], {"eq": "RTX 4090"})
        self.assertEqual(payload["geolocation"], {"eq": "US"})
        self.assertEqual(payload["verified"], {"eq": True})
        self.assertEqual(payload["datacenter"], {"eq": True})

    def test_invalid_filters_fail_before_http(self) -> None:
        for changes in (
            {"gpu_count": 0}, {"gpu_count": True}, {"disk_gb": -1},
            {"country": "us-west"}, {"limit": 501}, {"min_reliability": float("nan")},
            {"min_reliability": float("inf")}, {"min_reliability": -0.1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                vast_offer_search_payload(replace(VastOfferQuery(), **changes))

    def test_normalizes_units_without_double_counting_gpu_or_disk(self) -> None:
        row = parse_vast_offer(offer_payload(num_gpus=2), VastOfferQuery(gpu_count=2))
        assert row is not None
        self.assertEqual(row.gpu_memory_gb, 24.576)
        self.assertEqual(row.cpu_memory_gb, 64)
        self.assertEqual(row.max_duration_hours, 24)
        self.assertEqual(row.costs.compute_per_hour_usd, 0.4)
        self.assertAlmostEqual(row.costs.disk_per_hour_usd, 0.02)
        self.assertEqual(row.costs.total_per_hour_usd, 0.42)
        self.assertEqual(row.costs.download_per_gb_usd, 0.002)

    def test_rejects_ineligible_or_malformed_rows(self) -> None:
        changes_list = (
            {"id": True}, {"machine_id": None}, {"num_gpus": 2}, {"gpu_ram": -1},
            {"gpu_ram": float("inf")}, {"gpu_name": {}}, {"disk_space": 10},
            {"reliability": 0.5}, {"reliability": 2}, {"verified": False},
            {"verification": "unverified"}, {"rentable": False}, {"rented": True},
            {"is_bid": True}, {"gpu_arch": "amd"}, {"cpu_arch": "arm64"},
            {"duration": 0},
        )
        for changes in changes_list:
            with self.subTest(changes=changes):
                self.assertIsNone(parse_vast_offer(offer_payload(**changes), VastOfferQuery()))
        self.assertIsNone(parse_vast_offer([], VastOfferQuery()))

    def test_nullable_rented_field_is_available(self) -> None:
        self.assertIsNotNone(parse_vast_offer(offer_payload(rented=None), VastOfferQuery()))

    def test_verified_is_not_datacenter_and_unknown_is_not_free(self) -> None:
        self.assertIsNone(parse_vast_offer(offer_payload(), VastOfferQuery(datacenter_only=True)))
        row = parse_vast_offer(offer_payload(dph_base=None, dph_total=None, inet_up_cost=None), VastOfferQuery())
        assert row is not None
        self.assertIsNone(row.costs.compute_per_hour_usd)
        self.assertIsNone(row.costs.disk_per_hour_usd)
        self.assertIsNone(row.costs.total_per_hour_usd)
        self.assertIsNone(row.costs.upload_per_gb_usd)

    def test_nonfinite_prices_become_unknown(self) -> None:
        for value in (float("nan"), float("inf"), -1, True, "bad"):
            row = parse_vast_offer(offer_payload(dph_base=value, dph_total=value, inet_down_cost=value), VastOfferQuery())
            assert row is not None
            self.assertIsNone(row.costs.total_per_hour_usd)
            self.assertIsNone(row.costs.download_per_gb_usd)

    def test_structured_price_breakdown_and_zero_are_preserved(self) -> None:
        row = parse_vast_offer(offer_payload(
            search={"gpuCostPerHour": 0.3, "diskHour": 0}, dph_total=None, inet_down_cost=0,
        ), VastOfferQuery())
        assert row is not None
        self.assertEqual(row.costs.total_per_hour_usd, 0.3)
        self.assertEqual(row.costs.disk_per_hour_usd, 0)
        self.assertEqual(row.costs.download_per_gb_usd, 0)


class VastBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = VastBackend(VastCredentials("private-key", "environment"))
        self.response = Mock(status_code=200)
        patcher = patch("llm_launchpad.core.vast_backend.requests.request", return_value=self.response)
        self.request = patcher.start()
        self.addCleanup(patcher.stop)

    def test_validates_account_without_returning_account_secrets(self) -> None:
        self.response.json.return_value = {"id": 12, "ssh_key": "secret", "api_key": "private-key"}
        result = self.backend.auth_status()
        self.assertTrue(result.authenticated)
        self.assertEqual(result.account_id, "12")
        self.assertNotIn("private-key", repr(result))
        self.assertEqual(self.request.call_args.args[0], "GET")
        self.assertFalse(self.request.call_args.kwargs["allow_redirects"])
        self.response.close.assert_called_once()

    def test_no_credentials_does_not_contact_api(self) -> None:
        result = VastBackend(VastCredentials()).auth_status()
        self.assertFalse(result.authenticated)
        self.request.assert_not_called()

    def test_errors_are_useful_and_never_echo_response_or_token(self) -> None:
        for code, expected in ((401, "key"), (403, "permissions"), (429, "rate limit"), (500, "HTTP 500"), (302, "HTTP 302")):
            with self.subTest(code=code):
                self.response.status_code = code
                self.response.text = "private-key"
                result = self.backend.auth_status()
                self.assertFalse(result.authenticated)
                self.assertIn(expected, result.error or "")
                self.assertNotIn("private-key", result.error or "")

    def test_network_failure_is_sanitized(self) -> None:
        self.request.side_effect = requests.ConnectionError("private-key")
        result = self.backend.auth_status()
        self.assertIn("Could not reach Vast", result.error or "")
        self.assertNotIn("private-key", repr(result))

    def test_malformed_account_cannot_authenticate(self) -> None:
        for payload in ([], {}, {"id": True}, {"success": False, "id": 12}):
            self.response.json.return_value = payload
            self.assertFalse(self.backend.auth_status().authenticated)
        self.response.json.side_effect = ValueError("private-key")
        result = self.backend.auth_status()
        self.assertIn("invalid JSON", result.error or "")

    def test_offer_search_is_read_only_and_deduplicates(self) -> None:
        self.response.json.return_value = {"offers": [
            offer_payload(id=1002, dph_total=None, dph_base=None), offer_payload(), offer_payload(),
        ]}
        rows = self.backend.list_offers()
        self.assertEqual([row.id for row in rows], ["1001", "1002"])
        self.assertEqual(self.request.call_args.args, ("POST", "https://console.vast.ai/api/v0/bundles/"))
        self.assertEqual(self.request.call_args.kwargs["json"]["type"], "on-demand")
        self.assertEqual(self.request.call_count, 1)

    def test_missing_offers_is_an_error_not_an_empty_marketplace(self) -> None:
        self.response.json.return_value = {}
        with self.assertRaisesRegex(VastApiError, "offers list"):
            self.backend.list_offers()

    def test_selected_offer_revalidates_disk_and_exact_identity(self) -> None:
        self.response.json.return_value = {"offers": [offer_payload()]}
        self.assertEqual(self.backend.get_offer("1001", VastOfferQuery(disk_gb=120)).id, "1001")
        payload = self.request.call_args.kwargs["json"]
        self.assertEqual(payload["ask_contract_id"], {"eq": 1001})
        self.assertNotIn("id", payload)
        self.assertEqual(payload["allocated_storage"], 120)
        with self.assertRaisesRegex(VastApiError, "no longer available"):
            self.backend.get_offer("1002", VastOfferQuery())

    def test_create_uses_ssh_mode_and_never_retries_uncertain_mutation(self) -> None:
        self.response.json.return_value = {"success": True, "new_contract": 900}
        self.assertEqual(self.backend.create_instance("1001", image="image@sha256:abc", disk_gb=120, label="owned"), "900")
        self.assertEqual(self.request.call_args.args, ("PUT", "https://console.vast.ai/api/v0/asks/1001/"))
        self.assertEqual(self.request.call_args.kwargs["json"], {
            "client_id": "me", "image": "image@sha256:abc", "disk": 120,
            "label": "owned", "runtype": "ssh", "target_state": "running", "cancel_unavail": True,
        })
        self.request.reset_mock()
        self.request.side_effect = requests.Timeout("private-key")
        with self.assertRaises(VastApiError):
            self.backend.create_instance("1001", image="image", disk_gb=100, label="owned")
        self.request.assert_called_once()

    def test_instance_absence_requires_404_and_expected_identity(self) -> None:
        self.response.json.return_value = {"instances": {"id": 900, "machine_id": 42, "label": "owned", "actual_status": "running"}}
        self.assertEqual(self.backend.get_instance("900").label, "owned")
        with self.assertRaisesRegex(VastApiError, "identity"):
            self.backend.get_instance("901")
        self.response.json.return_value = {"instances": []}
        with self.assertRaises(VastApiError):
            self.backend.get_instance("900")
        self.response.json.return_value = {"instances": None}
        self.assertIsNone(self.backend.get_instance("900"))
        self.response.json.return_value = {}
        with self.assertRaises(VastApiError):
            self.backend.get_instance("900")
        self.response.status_code = 500
        with self.assertRaises(VastApiError):
            self.backend.get_instance("900")
        self.response.status_code = 404
        self.assertIsNone(self.backend.get_instance("900"))

    def test_create_includes_the_public_key_startup_hook(self) -> None:
        self.response.json.return_value = {"success": True, "new_contract": 900}
        self.backend.create_instance(
            "1001", image="image", disk_gb=100, label="owned", onstart="key setup",
        )
        self.assertEqual(self.request.call_args.kwargs["json"]["onstart"], "key setup")

    def test_account_lookup_preserves_rate_limit_for_cleanup_retries(self) -> None:
        self.response.status_code = 429
        with self.assertRaises(VastApiError) as caught:
            self.backend.account_id()
        self.assertEqual(caught.exception.status_code, 429)
        self.assertFalse(self.backend.auth_status().authenticated)

    def test_rejected_key_attachment_is_never_assumed_to_be_a_duplicate(self) -> None:
        for status in (400, 409):
            with self.subTest(status=status):
                self.response.status_code = status
                with self.assertRaises(VastApiError) as caught:
                    self.backend.attach_key("900", "ssh-ed25519 PUBLIC")
                self.assertEqual(caught.exception.status_code, status)
        self.response.status_code = 200
        for payload in ({}, {"success": False}):
            self.response.json.return_value = payload
            with self.assertRaises(VastApiError):
                self.backend.attach_key("900", "ssh-ed25519 PUBLIC")

    def test_reconcile_paginates_and_checks_exact_label(self) -> None:
        self.response.json.side_effect = [
            {"instances": [{"id": 800, "label": "other"}], "next_token": "page2"},
            {"instances": [{"id": 900, "label": "owned"}], "next_token": None},
        ]
        self.assertEqual([row.id for row in self.backend.find_instances("owned")], ["900"])
        self.assertEqual(self.request.call_args.args, ("GET", "https://console.vast.ai/api/v1/instances/"))
        self.assertEqual(self.request.call_args.kwargs["params"]["after_token"], "page2")
        self.response.json.side_effect = None
        self.response.json.return_value = {"instances": [], "next_token": "repeat"}
        with self.assertRaisesRegex(VastApiError, "pagination"):
            self.backend.find_instances("owned")

    def test_key_is_attached_to_one_instance_and_destroy_requires_confirmation(self) -> None:
        self.response.json.return_value = {"success": True}
        self.backend.attach_key("900", "ssh-ed25519 PUBLIC")
        self.assertEqual(self.request.call_args.args, ("POST", "https://console.vast.ai/api/v0/instances/900/ssh/"))
        self.assertEqual(self.request.call_args.kwargs["json"], {"ssh_key": "ssh-ed25519 PUBLIC"})
        self.backend.destroy_instance("900")
        self.assertEqual(self.request.call_args.args, ("DELETE", "https://console.vast.ai/api/v0/instances/900/"))
        self.response.json.return_value = {}
        with self.assertRaises(VastApiError):
            self.backend.destroy_instance("900")
        self.response.status_code = 404
        self.backend.destroy_instance("900")

    def test_resource_ids_cannot_escape_api_paths(self) -> None:
        for identifier in ("../users", "0", "-1", "１２", "900?key=secret"):
            with self.assertRaises(ValueError):
                self.backend.destroy_instance(identifier)
        self.request.assert_not_called()
