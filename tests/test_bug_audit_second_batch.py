"""Behavioral reproductions for the second batch of ten bugs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from rich.cells import cell_len

from llm_launchpad.core import artificial_analysis_auth, deploy_journal, gguf_metadata
from llm_launchpad.core import prime_disks, vision, vision_probe
from llm_launchpad.core.inference_options import PrimeInferenceAdapter
from llm_launchpad.protocol.enums import BackendType, ComputeProvider, VisionMode, VisionVerification
from llm_launchpad.protocol.models import (
    ComputeOffer, DeploymentConfig, InferenceRecipe, PrimeProviderOptions,
    VisionCapabilities, WorkloadProfile,
)
from llm_launchpad.tui.app import TuiApp
from llm_launchpad.tui.format import clip


def test_cancelled_image_probe_preserves_previous_verification() -> None:
    capability = VisionCapabilities(verification=VisionVerification.PASSED)
    with patch.object(vision_probe, "is_shutting_down", return_value=True):
        with pytest.raises(vision_probe.ImageProbeCancelled):
            vision_probe.verify_image_request("https://example.test", "model", None, capability)
    assert capability.verification == VisionVerification.PASSED


@pytest.mark.parametrize("text", [None, False, 0, {}, []])
def test_image_probe_does_not_accept_nontext_content_parts(text: object) -> None:
    capability = VisionCapabilities()
    response = Mock()
    response.json.return_value = {"choices": [{"message": {"content": [{"text": text}]}}]}
    with patch.object(vision_probe, "is_shutting_down", return_value=False):
        with patch.object(vision_probe.requests, "post", return_value=response):
            with pytest.raises(ValueError, match="no assistant text"):
                vision_probe.verify_image_request("https://example.test", "model", None, capability)
    assert capability.verification == VisionVerification.FAILED


def test_projector_fallback_uses_selected_model_revision() -> None:
    config = DeploymentConfig(
        backend=BackendType.LLAMACPP, repo_id="owner/model", revision="release-branch",
        vision_mode=VisionMode.ON, projector_file="mmproj.gguf",
    )
    api = Mock()
    api.model_info.return_value = SimpleNamespace(
        sha="resolved-sha", siblings=[SimpleNamespace(rfilename="mmproj.gguf", size=100)],
    )
    with patch.object(vision, "inspect_model_vision", side_effect=RuntimeError("temporary failure")):
        with patch("huggingface_hub.HfApi", return_value=api):
            vision.prepare_vision(config)
    assert api.model_info.call_args.kwargs["revision"] == "release-branch"


@pytest.mark.parametrize("field", ["gpu_count", "price_per_hour_usd", "started_at_epoch"])
def test_malformed_journal_numbers_do_not_hide_other_deployments(tmp_path: Path, field: str) -> None:
    path = tmp_path / "journal.json"
    base = {"app_name": "good", "provider": "modal", "backend": "vllm"}
    path.write_text(json.dumps({"schema_version": 1, "entries": [
        {**base, "app_name": "damaged", field: "not-a-number"}, base,
    ]}))
    entries = deploy_journal.load_in_flight(path)
    assert {entry.app_name for entry in entries} == {"good", "damaged"}


def _offer(identifier: str, region: str = "canada") -> ComputeOffer:
    return ComputeOffer(
        id=identifier, cloud_id=identifier, provider_name="provider", gpu_type="H100_80GB",
        gpu_count=1, gpu_memory_gb=80, region=region, security="secure_cloud",
        stock_status="Available", price_per_hour=2, images=("ubuntu_22_cuda_12",),
    )


def test_optional_cached_disk_lookup_failure_falls_back_to_gpu(tmp_path: Path) -> None:
    path = tmp_path / "disks.json"
    prime_disks.remember_prime_disk(prime_disks.StoredPrimeDisk(id="old"), path)
    offer = _offer("available")
    backend = Mock()
    backend.get_disk.return_value = {"status": "UNATTACHED"}
    backend.list_offers.side_effect = [RuntimeError("disk no longer attachable"), [offer]]
    backend.list_disk_offers.return_value = []
    config = DeploymentConfig(
        backend=BackendType.LLAMACPP, provider=ComputeProvider.PRIME,
        gpu_type="H100_80GB", provider_options=PrimeProviderOptions(auto_disk=True),
    )
    selected, disk_id, _ = prime_disks.resolve_prime_offer_and_disk(
        backend, config, required_image="ubuntu_22_cuda_12", path=path,
    )
    assert selected == offer
    assert disk_id is None


def test_prime_quotes_respect_requested_region() -> None:
    backend = Mock()
    backend.list_offers.return_value = [_offer("ca", "canada"), _offer("us", "usa")]
    adapter = PrimeInferenceAdapter(backend, PrimeProviderOptions(region="canada"))
    recipe = InferenceRecipe("recipe", "model", "Model", BackendType.LLAMACPP, "owner/model")
    assert [quote.provider_reference for quote in adapter.quote(recipe, WorkloadProfile())] == ["ca"]


@pytest.mark.parametrize("header,body", [("bytes 0-9/100", b"short"), ("bytes 0-2/100", b"too long"), ("bytes 5-2/100", b"")])
def test_gguf_range_body_must_match_advertised_bounds(header: str, body: bytes) -> None:
    response = Mock(status_code=206, headers={"Content-Range": header})
    response.iter_content.return_value = [body]
    start = 5 if "5-" in header else 0
    with patch("requests.get", return_value=response), patch("huggingface_hub.get_token", return_value=None):
        with pytest.raises(RuntimeError):
            gguf_metadata._fetch_hf_file_range("owner/model", "model.gguf", revision=None, start=start, end=9)
    response.close.assert_called_once()


def test_unconfirmed_stop_keeps_deployment_recovery_journal() -> None:
    app = TuiApp()
    config = DeploymentConfig(backend=BackendType.VLLM, app_name="vllm-pending")
    app._begin_in_flight(config)
    pending = app._in_flight_deploys[app._deployment_key(config)]
    assert pending is not None
    with patch.object(app._orchestrator, "stop_app", return_value=iter(())):
        assert app._stop_in_flight(pending) is False
    assert [entry.app_name for entry in deploy_journal.load_in_flight()] == ["vllm-pending"]


@pytest.mark.parametrize("text,width", [("模型模型模型", 5), ("模型", 2), ("abcdef", -1)])
def test_clipping_respects_terminal_cell_width(text: str, width: int) -> None:
    assert cell_len(clip(text, width)) <= max(0, width)


@pytest.mark.parametrize("value", [None, False, 123, [], {}])
def test_invalid_saved_api_key_is_not_used_as_a_credential(tmp_path: Path, value: object) -> None:
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"api_key": value}))
    assert artificial_analysis_auth.load_saved_artificial_analysis_api_key(path) == ""
