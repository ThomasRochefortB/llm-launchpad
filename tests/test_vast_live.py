"""Certification harness regressions, without renting or contacting providers."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, Mock, patch

from llm_launchpad.core.vast_backend import VastApiError, VastBackend
from llm_launchpad.core.vast_state import VastState
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import DeploymentConfig, EndpointInfo, VastInstance, VisionCapabilities
from scripts.validate_vast_custom_live import probe_vision
from scripts.validate_vast_live import budget_estimate, probe_image
from tests.test_vast_fast_deploy import vast_offer


def arguments(report: Path) -> argparse.Namespace:
    return argparse.Namespace(
        offer_id="1001", disk_gb=100, max_hourly_cost=0.5, budget_usd=1.0,
        max_minutes=10, transfer_gb=20, image="image@sha256:pinned", report=report,
    )


class VastLiveBudgetTests(unittest.TestCase):
    def test_estimate_includes_disk_transfer_and_cleanup_time(self) -> None:
        offer = vast_offer()
        args = arguments(Path("unused"))
        estimate = budget_estimate(offer, args, 5)
        self.assertAlmostEqual(estimate, 0.42 * 15 / 60 + 0.002 * 20 + 0.003)

    def test_unknown_negative_or_excessive_transfer_prices_refuse_rental(self) -> None:
        offer = vast_offer()
        for download in (None, -1, float("nan"), float("inf"), 1):
            with self.subTest(download=download), self.assertRaises(ValueError):
                budget_estimate(
                    replace(offer, costs=replace(offer.costs, download_per_gb_usd=download)),
                    arguments(Path("unused")), 5,
                )

    def test_invalid_budget_or_insufficient_credit_is_refused(self) -> None:
        for field in ("budget_usd", "max_minutes", "transfer_gb", "max_hourly_cost"):
            for value in (0, -1, float("nan"), float("inf")):
                args = arguments(Path("unused"))
                setattr(args, field, value)
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    budget_estimate(vast_offer(), args, 5)
        with self.assertRaises(ValueError):
            budget_estimate(vast_offer(), arguments(Path("unused")), 0.5)


class VastImageProbeTests(unittest.TestCase):
    def test_production_startup_and_ambiguous_create_both_confirm_cleanup(self) -> None:
        self._check_probe(ambiguous=False)
        self._check_probe(ambiguous=True)

    def _check_probe(self, *, ambiguous: bool) -> None:
        with TemporaryDirectory() as directory:
            state = VastState(Path(directory) / "state")
            api = Mock(spec=VastBackend)
            api.get_offer.return_value = vast_offer()
            api.account_id.return_value = "12"
            remote = None

            def create(_offer_id: str, **kwargs: object) -> str:
                nonlocal remote
                records = state.records()
                self.assertEqual(len(records), 1)
                self.assertIsNone(records[0].instance_id)
                self.assertNotIn("onstart", kwargs)
                remote = VastInstance(
                    id="900", label=records[0].label, machine_id="42", state="running",
                    ssh_host="ssh1.vast.ai", ssh_port=2200,
                )
                if ambiguous:
                    raise VastApiError("Create response lost")
                return "900"

            def destroy(instance_id: str) -> None:
                nonlocal remote
                self.assertEqual(instance_id, "900")
                remote = None

            api.create_instance.side_effect = create
            api.destroy_instance.side_effect = destroy
            api.get_instance.side_effect = lambda _: remote
            api.find_instances.side_effect = lambda _: [remote] if remote else []
            ssh = Mock()
            ssh.public_key.return_value = "ssh-ed25519 PUBLICKEY"
            ssh.run.return_value = "probe output"
            report_path = Path(directory) / "report.json"
            with (
                patch("scripts.validate_vast_live.VastBackend", return_value=api),
                patch("scripts.validate_vast_live.VastState", return_value=state),
                patch("scripts.validate_vast_live.VastSsh", return_value=ssh),
                patch("scripts.validate_vast_live.account_credit", return_value=5),
            ):
                result = probe_image(arguments(report_path))
            self.assertEqual(result, 1 if ambiguous else 0)
            self.assertTrue(json.loads(report_path.read_text())["cleanup_confirmed"])
            self.assertEqual(state.records(), [])
            api.create_instance.assert_called_once()
            api.destroy_instance.assert_called_once_with("900")

    def test_excessive_total_cost_never_creates_a_rental(self) -> None:
        api = Mock(spec=VastBackend)
        offer = vast_offer()
        api.get_offer.return_value = replace(offer, costs=replace(offer.costs, download_per_gb_usd=1))
        with (
            patch("scripts.validate_vast_live.VastBackend", return_value=api),
            patch("scripts.validate_vast_live.account_credit", return_value=5),
            self.assertRaises(ValueError),
        ):
            probe_image(arguments(Path("unused")))
        api.create_instance.assert_not_called()


class VastLiveVisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = MagicMock()
        self.response = self.session.post.return_value.__enter__.return_value
        self.response.json.return_value = {"choices": [{"message": {"content": "A red square."}}]}
        self.endpoint = EndpointInfo(
            name="test", app_id="900", backend=BackendType.VLLM, provider=ComputeProvider.VAST,
            state="running", web_url="http://127.0.0.1:1234", endpoint_api_key="private",
            served_model_name="vision-model",
        )

    def test_vllm_vision_does_not_require_a_gguf_projector(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.VLLM, vision=VisionCapabilities(supported=True, enabled=True),
        )
        checks = probe_vision(self.session, self.endpoint, config)
        self.assertEqual(checks["image_answer"], "A red square.")
        self.assertNotIn("projector", checks)
        self.session.post.assert_called_once()

    def test_llamacpp_missing_projector_is_refused_before_request(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP, vision=VisionCapabilities(supported=True, enabled=True),
        )
        with self.assertRaisesRegex(ValueError, "projector"):
            probe_vision(self.session, self.endpoint, config)
        self.session.post.assert_not_called()

    def test_empty_image_response_does_not_certify_vision(self) -> None:
        self.response.json.return_value = {"choices": [{"message": {"content": ""}}]}
        config = DeploymentConfig(
            backend=BackendType.VLLM, vision=VisionCapabilities(supported=True, enabled=True),
        )
        with self.assertRaisesRegex(RuntimeError, "no text"):
            probe_vision(self.session, self.endpoint, config)
