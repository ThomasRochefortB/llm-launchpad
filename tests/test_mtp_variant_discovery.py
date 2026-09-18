"""MTP artifacts must be selected before catalog placement is calculated."""

from dataclasses import replace
import unittest
from unittest.mock import patch

from llm_launchpad.core.gguf_metadata import GgufMtpCapability, GgufMtpStatus
from llm_launchpad.core.hf_budget import HubRequestBudget
from llm_launchpad.core.hf_models import GgufQuantMetadata, ModelCandidate
from llm_launchpad.core.modal_gpu import ModalGpuSpec
from llm_launchpad.core.quick_deploy_refresh import (
    _profiles_for_model,
    _resolve_aa_model,
    _verified_mtp_variant,
)
from tests.test_quick_deploy_refresh import _aa_candidate


REPO = "unsloth/Qwen3.6-27B-GGUF"
VARIANT = "unsloth/Qwen3.6-27B-MTP-GGUF"
FETCH = "llm_launchpad.core.quick_deploy_refresh.fetch_gguf_quant_metadata"


def _metadata(*, mtp: bool = False) -> GgufQuantMetadata:
    return GgufQuantMetadata(
        quantizations=["UD-Q4_K_XL", "UD-Q2_K_XL"],
        vram_gb_by_quant={"UD-Q4_K_XL": 18.0, "UD-Q2_K_XL": 10.0},
        architecture="qwen35",
        context_length=8192,
        mtp=GgufMtpCapability(
            status=GgufMtpStatus.SUPPORTED if mtp else GgufMtpStatus.UNSUPPORTED,
            nextn_predict_layers=1 if mtp else 0,
        ),
    )


class MtpVariantDiscoveryTests(unittest.TestCase):
    def test_verifies_each_offered_quant(self) -> None:
        with patch(FETCH, return_value=_metadata(mtp=True)) as fetch:
            result = _verified_mtp_variant(REPO, _metadata(), None)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], VARIANT)
        self.assertEqual(
            [call.kwargs.get("mtp_quant") for call in fetch.call_args_list],
            [None, "UD-Q4_K_XL", "UD-Q2_K_XL"],
        )

    def test_rejects_unverified_quant_and_different_architecture(self) -> None:
        for responses in (
            [_metadata(mtp=True), _metadata(mtp=True), _metadata()],
            [replace(_metadata(mtp=True), architecture="deepseek2")],
            [_metadata()],
        ):
            with self.subTest(responses=responses), patch(FETCH, side_effect=responses):
                self.assertIsNone(_verified_mtp_variant(REPO, _metadata(), None))

    def test_failure_or_insufficient_budget_preserves_base_profiles(self) -> None:
        with patch(FETCH, side_effect=RuntimeError("Hub unavailable")):
            profiles = _profiles_for_model(
                ModelCandidate(repo_id=REPO), [ModalGpuSpec("L4", 0.5)],
                metadata=_metadata(),
            )
        self.assertTrue(profiles)
        self.assertTrue(all(p.repo_id == REPO for p in profiles))
        with patch(FETCH) as fetch:
            self.assertIsNone(_verified_mtp_variant(REPO, _metadata(), HubRequestBudget(0)))
        fetch.assert_not_called()

    def test_unsupported_runtime_and_existing_variants_do_not_probe(self) -> None:
        with patch(FETCH) as fetch:
            self.assertIsNone(_verified_mtp_variant(VARIANT, _metadata(), None))
            self.assertIsNone(_verified_mtp_variant(
                REPO, replace(_metadata(), architecture="llama"), None,
            ))
        fetch.assert_not_called()

    def test_cached_base_match_gets_mtp_before_memory_planning(self) -> None:
        candidate = _aa_candidate("Qwen3.6 27B", 27, 90)
        with patch(
            "llm_launchpad.core.quick_deploy_refresh._fetch_serving_metadata",
            return_value=_metadata(),
        ), patch(FETCH, return_value=_metadata(mtp=True)):
            resolved = _resolve_aa_model(
                candidate, [ModalGpuSpec("L4", 0.5)],
                repo_by_model_key={"qwen3627b": REPO},
            )
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.repo_id, VARIANT)
        self.assertTrue(resolved.profiles)
        for profile in resolved.profiles:
            self.assertEqual(profile.repo_id, VARIANT)
            self.assertIsNotNone(profile.speculative_decoding)
            self.assertGreater(profile.memory_estimate.speculative_gb, 0)
            self.assertAlmostEqual(
                profile.required_vram_gb, profile.memory_estimate.total_gb, places=1,
            )

    def test_variant_that_cannot_fit_falls_back_to_base(self) -> None:
        huge = replace(
            _metadata(mtp=True),
            vram_gb_by_quant={"UD-Q4_K_XL": 10000.0, "UD-Q2_K_XL": 9000.0},
        )
        with patch(FETCH, return_value=huge):
            profiles = _profiles_for_model(
                ModelCandidate(repo_id=REPO), [ModalGpuSpec("L4", 0.5)],
                metadata=_metadata(),
            )
        self.assertTrue(profiles)
        self.assertTrue(all(p.repo_id == REPO for p in profiles))
