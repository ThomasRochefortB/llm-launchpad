from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from llm_launchpad.core.artificial_analysis import AAModelCandidate
from llm_launchpad.core.backend import ModalBackend
from llm_launchpad.core.compute_availability import aggregate_compute_availability, plans_for_compute_profile
from llm_launchpad.core.gguf_metadata import GgufServingMetadata
from llm_launchpad.core.hf_models import GgufQuantMetadata, ModelCandidate
from llm_launchpad.core.llamacpp_planner import estimate_memory, serving_requirements, tuning_for_objective
from llm_launchpad.core.orchestrator import Orchestrator
from llm_launchpad.core.prime_backend import PrimeBackend, resolve_prime_launch_spec
from llm_launchpad.core.quick_deploy import QuickDeployCatalogInfo, QuickDeployProfile, build_quick_deploy_config, retune_quick_deploy_plan
from llm_launchpad.core.quick_deploy_refresh import (
    _build_resolved_aa_model, _profiles_for_model, _profiles_from_aa_rankings,
    _read_quick_deploy_catalog_cache, _write_quick_deploy_catalog_cache,
)
from llm_launchpad.core.modal_gpu import ModalGpuSpec
from llm_launchpad.core.runtime_support import evaluate_llamacpp_architecture, load_llamacpp_support_manifest
from llm_launchpad.protocol.enums import ComputeProvider, ServingObjective
from llm_launchpad.protocol.models import CatalogExclusion


def metadata_fixture(name: str) -> GgufQuantMetadata:
    """Public GGUF headers inspected on 2026-09-06; no network or weights."""
    path = Path(__file__).parent / "fixtures" / "hybrid_gguf_metadata.json"
    payload = json.loads(path.read_text())[name]
    serving = payload.pop("serving_metadata")
    serving["attention_head_count_kv_by_layer"] = tuple(serving["attention_head_count_kv_by_layer"])
    return GgufQuantMetadata(**payload, serving_metadata=GgufServingMetadata(**serving))


def candidate(name: str, rank: int) -> AAModelCandidate:
    return AAModelCandidate(
        aa_model_id=name, name=name, slug=name.lower(), creator_name="",
        coding_score=70.0, intelligence_score=50.0, rank=rank,
        parameter_count_b=None, max_context_tokens=None,
    )


class HybridModelSupportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gpus = [ModalGpuSpec("B200", price_per_hour_usd=6.0)]
        self.tuning = tuning_for_objective(ServingObjective.GENERAL_PURPOSE)

    def profiles(self, name: str) -> list[QuickDeployProfile]:
        return _profiles_for_model(
            ModelCandidate(repo_id=f"unsloth/{name}-GGUF"), self.gpus,
            metadata=metadata_fixture(name), aa_candidate=candidate(name, 1), size_bucket="large",
        )

    def test_kimi_cache_uses_24_compressed_layers_and_69_recurrent_layers(self) -> None:
        metadata = metadata_fixture("Kimi-K3")
        memory = estimate_memory(metadata, weights_gb=861.0, requirements=serving_requirements(1048576), tuning=self.tuning)
        token_cache = 24 * 576 * 1048576 * 2 / 1e9
        recurrent = 69 * (3 * 3 * 96 * 128 + 96 * 128 * 128) * 4 * 4 / 1e9
        self.assertAlmostEqual(memory.kv_cache_gb, token_cache + recurrent, places=3)
        self.assertEqual(memory.source, "gguf-hybrid-metadata")
        self.assertLess(memory.total_gb, 1000)
        self.assertEqual(memory.total_layer_count, 93)
        profiles = self.profiles("Kimi-K3")
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0].quant, "UD-Q2_K_XL")
        self.assertEqual(profiles[0].gpu_count, 6)

    def test_glm_cache_excludes_nextn_and_includes_pooling_indexer(self) -> None:
        metadata = metadata_fixture("GLM-5.3-Flash")
        memory = estimate_memory(metadata, weights_gb=109.0, requirements=serving_requirements(1048576), tuning=self.tuning)
        token_cache = 11 * (512 + 3 * 128) * 1048576 * 2 / 1e9
        recurrent = 34 * (3 * 3 * 64 * 128 + 64 * 128 * 128) * 4 * 4 / 1e9
        self.assertAlmostEqual(memory.kv_cache_gb, token_cache + recurrent, places=3)
        self.assertEqual(memory.total_layer_count, 45)
        profiles = self.profiles("GLM-5.3-Flash")
        self.assertEqual([p.gpu_count for p in profiles], [1, 2])
        for profile in profiles:
            self.assertFalse(profile.runtime_tuning.flash_attention)
            self.assertIn("off", profile.server_args)

    def test_missing_hybrid_layout_is_unverified_and_explained(self) -> None:
        metadata = replace(metadata_fixture("Kimi-K3"), serving_metadata=None)
        exclusions: list[CatalogExclusion] = []
        with patch("llm_launchpad.core.quick_deploy_refresh._fetch_serving_metadata", return_value=metadata):
            result = _build_resolved_aa_model(candidate("Kimi-K3", 1), self.gpus, "unsloth/Kimi-K3-GGUF", exclusions=exclusions)
        self.assertIsNone(result)
        self.assertIn("cannot be verified", exclusions[0].reason)

    def test_both_models_survive_real_shortlist_and_placement_pipeline(self) -> None:
        names = ("Kimi-K3", "GLM-5.3-Flash")
        with (
            patch("llm_launchpad.core.quick_deploy_refresh._find_unsloth_gguf_match", side_effect=lambda c, api: f"unsloth/{c.name}-GGUF"),
            patch("llm_launchpad.core.quick_deploy_refresh._fetch_serving_metadata", side_effect=lambda repo: metadata_fixture(repo.split('/')[1].removesuffix('-GGUF'))),
        ):
            profiles = _profiles_from_aa_rankings(tuple(candidate(name, i + 1) for i, name in enumerate(names)), self.gpus, model_limit=3, candidate_limit=80)
        self.assertEqual({p.display_name for p in profiles}, set(names))
        snapshot = aggregate_compute_availability(modal_catalog=self.gpus)
        for profile in profiles:
            with self.subTest(model=profile.display_name, quant=profile.quant):
                plans = plans_for_compute_profile(snapshot.configurations[0], profile)
                self.assertTrue(plans)
                self.assertTrue(plans[0].assessment.fits)
                config = build_quick_deploy_config(profile, plan=plans[0])
                self.assertEqual(config.max_context_tokens, 1048576)
                retuned = retune_quick_deploy_plan(profile, plans[0], ServingObjective.THROUGHPUT)
                self.assertAlmostEqual(retuned.assessment.memory.recurrent_gb, profile.memory_estimate.recurrent_gb * 2, places=5)
                if profile.gguf_architecture == "glm5next":
                    self.assertFalse(retuned.recipe.runtime_tuning.flash_attention)

    def test_cache_roundtrip_keeps_recurrent_memory_and_exclusion_reasons(self) -> None:
        profiles = self.profiles("Kimi-K3")
        info = QuickDeployCatalogInfo(source_label="test", exclusions=(CatalogExclusion("missing", "Missing", "org/model", "Unsupported architecture"),))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            _write_quick_deploy_catalog_cache(info, profiles, cache_path=path)
            self.assertEqual(_read_quick_deploy_catalog_cache(path), (info, tuple(profiles)))

    def test_glm_runtime_is_selected_consistently_by_both_providers(self) -> None:
        config = build_quick_deploy_config(self.profiles("GLM-5.3-Flash")[0])
        runtime = load_llamacpp_support_manifest("glm5next")
        self.assertTrue(evaluate_llamacpp_architecture("glm5next").is_supported)
        with patch.dict(os.environ, {"LLAMA_CPP_IMAGE_REF": "", "LLM_LAUNCHPAD_PRIME_LLAMACPP_CONTAINER_IMAGE": ""}):
            self.assertEqual(ModalBackend.env_for_backend(config)["LLAMA_CPP_BUILD_RECIPE"], runtime.build_recipe)
            self.assertEqual(ModalBackend.env_for_backend(config)["LLAMA_CPP_CUDA_ARCHITECTURES"], "100")
            for provider in (ComputeProvider.MODAL, ComputeProvider.PRIME):
                config.provider = provider
                decision, error = Orchestrator()._llamacpp_compatibility(config)
                self.assertIsNone(error)
                self.assertTrue(decision.is_supported)
                self.assertEqual(decision.runtime_id, runtime.runtime_id)
            launch = resolve_prime_launch_spec(config)
            self.assertEqual(launch.container_image, runtime.image_ref)
            self.assertIn(runtime.source_revision, launch.build_recipe)
            self.assertIn("NVIDIA_TF32_OVERRIDE=0", launch.build_recipe)
            script = PrimeBackend._bootstrap_script(config, launch)
            self.assertIn("docker build", script)
            self.assertIn("--build-arg CUDA_ARCHITECTURES=100", script)
            self.assertNotIn("docker pull", script)

    def test_custom_images_are_not_replaced_with_the_source_runtime(self) -> None:
        config = build_quick_deploy_config(self.profiles("GLM-5.3-Flash")[0])
        with patch.dict(os.environ, {"LLAMA_CPP_IMAGE_REF": "custom/image:tag", "LLM_LAUNCHPAD_PRIME_LLAMACPP_CONTAINER_IMAGE": "custom/image:tag"}):
            self.assertNotIn("LLAMA_CPP_BUILD_RECIPE", ModalBackend.env_for_backend(config))
            launch = resolve_prime_launch_spec(config)
            self.assertIsNone(launch.build_recipe)
            self.assertEqual(launch.container_image, "custom/image:tag")
