from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from llm_launchpad.cli.main import app
from llm_launchpad.core.vast_auth import VastCredentials, save_vast_api_key
from llm_launchpad.core.vast_backend import VastApiError, parse_vast_offer
from llm_launchpad.protocol.enums import BackendType, ComputeProvider, OperationType
from llm_launchpad.protocol.events import OperationCompleteEvent
from llm_launchpad.protocol.models import ComputeOffer, EndpointInfo, VastAuthStatus, VastOfferQuery, VastProviderOptions
from tests.test_vast_backend import offer_payload


class VastCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.key_path = Path(temp.name) / "auth.json"
        patches = [
            patch("llm_launchpad.core.vast_auth.VAST_AUTH_PATH", self.key_path),
            patch.dict("os.environ", {"XDG_CONFIG_HOME": temp.name}, clear=True),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_login_validates_then_stores_without_echoing_key(self) -> None:
        with patch("llm_launchpad.cli.vast.VastBackend.auth_status", return_value=VastAuthStatus(True, account_id="12")):
            result = self.runner.invoke(app, ["vast-auth", "login"], input="private-key\n")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertNotIn("private-key", result.output)
        self.assertIn("12", result.output)
        self.assertEqual(json.loads(self.key_path.read_text())["api_key"], "private-key")

    def test_invalid_login_preserves_previous_credential(self) -> None:
        save_vast_api_key("old-key", self.key_path)
        with patch("llm_launchpad.cli.vast.VastBackend.auth_status", return_value=VastAuthStatus(False, error="Denied")):
            result = self.runner.invoke(app, ["vast-auth", "login", "--key-stdin"], input="private-key\n")
        self.assertEqual(result.exit_code, 1)
        self.assertNotIn("private-key", result.output)
        self.assertEqual(json.loads(self.key_path.read_text())["api_key"], "old-key")

    def test_login_reports_saved_key_even_when_invalid_environment_overrides_it(self) -> None:
        with patch.dict("os.environ", {"VAST_API_KEY": "invalid key"}), patch(
            "llm_launchpad.cli.vast.VastBackend.auth_status", return_value=VastAuthStatus(True, account_id="12")
        ):
            result = self.runner.invoke(app, ["vast-auth", "login", "--key-stdin"], input="private-key\n")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("takes precedence", result.output)
        self.assertNotIn("private-key", result.output)
        self.assertEqual(json.loads(self.key_path.read_text())["api_key"], "private-key")

    def test_local_status_does_not_verify_and_logout_reports_environment(self) -> None:
        with patch.dict("os.environ", {"VAST_API_KEY": "env-key"}), patch("llm_launchpad.cli.vast.VastBackend") as backend:
            status = self.runner.invoke(app, ["vast-auth", "status", "--local"])
            logout = self.runner.invoke(app, ["vast-auth", "logout"])
        backend.assert_not_called()
        self.assertEqual(status.exit_code, 0)
        self.assertIn("not verified", status.output)
        self.assertIn("environment", logout.output)
        self.assertNotIn("env-key", status.output + logout.output)

    def test_verified_status_reports_auth_failure(self) -> None:
        with patch("llm_launchpad.cli.vast.resolve_vast_credentials", return_value=VastCredentials("key", "stored")), patch(
            "llm_launchpad.cli.vast.VastBackend.auth_status", return_value=VastAuthStatus(False, error="Denied")
        ):
            result = self.runner.invoke(app, ["vast-auth", "status"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Denied", result.output)

    def test_offer_json_uses_vast_without_modal_or_prime_auth(self) -> None:
        row = parse_vast_offer(offer_payload(), VastOfferQuery())
        with patch("llm_launchpad.cli.vast.VastBackend.list_offers", return_value=[row]) as search, patch(
            "llm_launchpad.cli.main._preflight", side_effect=AssertionError("must not authenticate Prime")
        ):
            result = self.runner.invoke(app, ["offers", "--provider", "vast", "--region", "US", "--disk-gb", "120", "--json"])
        self.assertEqual(result.exit_code, 0, result.output)
        data = json.loads(result.output)
        self.assertEqual(data[0]["id"], "1001")
        self.assertNotIn("raw", data[0])
        query = search.call_args.args[0]
        self.assertEqual(query.disk_gb, 120)
        self.assertEqual(query.country, "US")
        self.assertFalse(query.datacenter_only)

    def test_offer_text_marks_missing_prices_and_deployment_scope(self) -> None:
        row = parse_vast_offer(offer_payload(inet_up_cost=None), VastOfferQuery())
        with patch("llm_launchpad.cli.vast.VastBackend.list_offers", return_value=[row]):
            result = self.runner.invoke(app, ["offers", "--provider", "vast"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("unknown", result.output)
        self.assertIn("--provider vast", result.output)
        self.assertIn("0.0020", result.output)

    def test_api_error_and_incompatible_flags_have_nonzero_exit(self) -> None:
        with patch("llm_launchpad.cli.vast.VastBackend.list_offers", side_effect=VastApiError("Rate limit")):
            result = self.runner.invoke(app, ["offers", "--provider", "vast"])
        self.assertEqual(result.exit_code, 1)
        for args in (
            ["offers", "--provider", "vast", "--disk-id", "prime-disk"],
            ["offers", "--provider", "vast", "--no-on-demand-only"],
            ["offers", "--disk-gb", "120"],
        ):
            result = self.runner.invoke(app, args)
            self.assertEqual(result.exit_code, 2, result.output)

    def test_prime_remains_the_default_and_keeps_secure_filter(self) -> None:
        prime = Mock()
        prime.list_offers.return_value = [ComputeOffer(
            id="prime-offer", cloud_id="cloud", provider_name="prime", gpu_type="H100", gpu_count=1,
            security="secure_cloud", price_per_hour=2.5,
        ), ComputeOffer(
            id="community", cloud_id="cloud", provider_name="prime", gpu_type="H100", gpu_count=1,
            security="community", price_per_hour=1,
        )]
        with patch("llm_launchpad.cli.main._preflight", return_value=(Mock(prime_backend=prime), "user")):
            default = self.runner.invoke(app, ["offers", "--json"])
            unfiltered = self.runner.invoke(app, ["offers", "--no-secure-only", "--json"])
        self.assertEqual(default.exit_code, 0, default.output)
        self.assertEqual(len(json.loads(default.output)), 1)
        self.assertEqual(len(json.loads(unfiltered.output)), 2)

    def test_vast_requires_its_own_credentials_and_explicit_rental_parameters(self) -> None:
        for command in ("deploy", "stop", "list", "logs", "status", "switch", "warmup"):
            with self.subTest(command=command):
                result = self.runner.invoke(app, [command, "--provider", "vast"])
                self.assertEqual(result.exit_code, 2 if command in {"deploy", "switch"} else 1, result.output)
                if command not in {"deploy", "switch"}:
                    self.assertIn("Vast", result.output)

    def test_manual_deploy_keeps_selected_offer_and_explicit_price_cap(self) -> None:
        with patch("llm_launchpad.cli.main._preflight", return_value=(Mock(), "12")), patch(
            "llm_launchpad.cli.main._deploy_and_maybe_warmup"
        ) as deploy:
            result = self.runner.invoke(app, [
                "deploy", "--provider", "vast", "--repo-id", "acme/model-GGUF", "--quant", "Q4_K_M",
                "--vast-offer-id", "1001", "--vast-disk-gb", "120", "--max-hourly-cost", "0.5",
            ])
        self.assertEqual(result.exit_code, 0, result.output)
        config = deploy.call_args.kwargs["config"]
        self.assertEqual(config.provider, ComputeProvider.VAST)
        self.assertEqual(config.gpu_count, 1)
        self.assertEqual(config.provider_options, VastProviderOptions("1001", 120, 0.5))
        self.assertTrue(config.app_name.startswith("llp-vast-"))

    def test_stop_confirmation_describes_disk_deletion_and_routes_to_vast(self) -> None:
        endpoint = EndpointInfo(name="llp-vast-llamacpp-test", app_id="900", backend=BackendType.LLAMACPP, provider=ComputeProvider.VAST)
        orchestrator = Mock()
        orchestrator.stop_app.return_value = iter([OperationCompleteEvent(operation=OperationType.STOP, success=True)])
        with patch("llm_launchpad.cli.main._preflight", return_value=(orchestrator, "12")), patch(
            "llm_launchpad.cli.main._resolve_manage_target", return_value=endpoint
        ), patch("llm_launchpad.cli.main.remove_connection"), patch("llm_launchpad.cli.main._sync_opencode_cli"):
            result = self.runner.invoke(app, ["stop", "--provider", "vast"], input="y\n")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("permanently delete its disk", result.output)
        self.assertEqual(orchestrator.stop_app.call_args.kwargs["provider"], ComputeProvider.VAST)
        self.assertEqual(orchestrator.stop_app.call_args.kwargs["app_id"], "900")

    def test_reconnect_uses_existing_instance_without_printing_credentials(self) -> None:
        with patch("llm_launchpad.core.vast_deployment.VastDeploymentBackend.connect", return_value=EndpointInfo(
            web_url="http://127.0.0.1:48123", endpoint_api_key="secret",
        )) as connect:
            result = self.runner.invoke(app, ["vast", "connect", "900"])
        self.assertEqual(result.exit_code, 0, result.output)
        connect.assert_called_once_with("900")
        self.assertIn("http://127.0.0.1:48123", result.output)
        self.assertNotIn("secret", result.output)
