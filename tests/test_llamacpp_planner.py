from __future__ import annotations

import shlex
import unittest
from unittest.mock import patch
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from llm_launchpad.core.hf_models import GgufQuantMetadata
from llm_launchpad.core.llamacpp_planner import (
    assess_placement,
    compile_server_args,
    estimate_memory,
    load_runtime_attestation,
    save_runtime_attestation,
    QUANTIZED_KV_ARCHITECTURES,
    default_cache_type,
    memory_for_cache_type,
    with_cache_type,
    serving_fingerprint,
    serving_requirements,
    tuning_for_gpu_memory,
    tuning_for_objective,
)
from llm_launchpad.protocol.enums import CertificationState, ServingObjective
from llm_launchpad.protocol.models import RuntimeAttestation, RuntimeTuning


class LlamaCppPlannerTests(unittest.TestCase):
    def test_canonical_args_preserve_full_context_with_parallel_slots(self) -> None:
        requirements = serving_requirements(131_072)
        tuning = tuning_for_objective(ServingObjective.GENERAL_PURPOSE)

        args = compile_server_args(
            requirements,
            tuning,
            extra_args=shlex.split(
                "--ctx-size 4096 --parallel 1 --no-kv-unified "
                "--n-gpu-layers 12 --temp 0.7"
            ),
        )

        self.assertEqual(args.count("--ctx-size"), 1)
        self.assertEqual(args[args.index("--ctx-size") + 1], "131072")
        self.assertEqual(args[args.index("--parallel") + 1], "4")
        self.assertIn("--kv-unified", args)
        self.assertNotIn("--no-kv-unified", args)
        self.assertEqual(
            args[args.index("--kv-unified-per-slot") + 1],
            "131072",
        )
        self.assertEqual(args[args.index("--n-gpu-layers") + 1], "all")
        self.assertEqual(args[args.index("--fit") + 1], "on")
        self.assertEqual(args[args.index("--fit-target") + 1], "2048")
        self.assertEqual(args[-2:], ("--temp", "0.7"))

    def test_exact_kv_estimate_uses_architecture_shape_and_full_context(self) -> None:
        metadata = GgufQuantMetadata(
            quantizations=["Q4_K_M"],
            vram_gb_by_quant={"Q4_K_M": 8.0},
            context_length=32_768,
            block_count=32,
            embedding_length=4096,
            attention_head_count=32,
            attention_head_count_kv=8,
            attention_key_length=128,
            attention_value_length=128,
        )
        requirements = serving_requirements(32_768)
        tuning = tuning_for_objective(ServingObjective.GENERAL_PURPOSE)

        memory = estimate_memory(
            metadata,
            weights_gb=8.0,
            requirements=requirements,
            tuning=tuning,
        )

        self.assertAlmostEqual(memory.kv_cache_gb, 4.295, places=3)
        self.assertEqual(memory.total_layer_count, 32)
        self.assertEqual(memory.source, "gguf-metadata")
        self.assertGreater(memory.total_gb, memory.weights_gb + memory.kv_cache_gb)

    def test_gpu_margin_is_maximum_of_two_gib_and_five_percent(self) -> None:
        tuning = tuning_for_objective(ServingObjective.THROUGHPUT)

        l4 = tuning_for_gpu_memory(tuning, 24.0)
        b200 = tuning_for_gpu_memory(tuning, 180.0)

        self.assertEqual(l4.fit_target_mib, 2048)
        self.assertEqual(b200.fit_target_mib, 9216)

    def test_assessment_rejects_weight_plus_kv_plan_that_does_not_fit(self) -> None:
        metadata = GgufQuantMetadata(
            quantizations=["Q4_K_M"],
            vram_gb_by_quant={"Q4_K_M": 20.0},
            block_count=32,
            embedding_length=4096,
            attention_head_count=32,
            attention_head_count_kv=8,
        )
        requirements = serving_requirements(131_072)
        tuning = tuning_for_objective(ServingObjective.GENERAL_PURPOSE)

        assessment = assess_placement(
            metadata,
            model_id="org/model",
            revision="abc",
            quant="Q4_K_M",
            runtime_id="llama.cpp-b10689-cuda12",
            weights_gb=20.0,
            requirements=requirements,
            tuning=tuning,
            gpu_type="L4",
            gpu_count=1,
            gpu_memory_gb=24.0,
        )

        self.assertFalse(assessment.fits)
        self.assertFalse(assessment.gpu_resident)
        self.assertEqual(assessment.certification, CertificationState.ESTIMATED)
        self.assertIn("per GPU", assessment.rejection_reason or "")

    def test_certificate_is_reused_only_for_the_exact_fingerprint(self) -> None:
        requirements = serving_requirements(32_768)
        tuning = tuning_for_objective(ServingObjective.INTERACTIVE)
        fingerprint = serving_fingerprint(
            model_id="org/model",
            revision="abc",
            quant="Q4_K_M",
            runtime_id="llama.cpp-b10689-cuda12",
            requirements=requirements,
            tuning=tuning,
            gpu_type="L4",
            gpu_count=1,
        )
        changed = serving_fingerprint(
            model_id="org/model",
            revision="abc",
            quant="Q4_K_M",
            runtime_id="llama.cpp-b10689-cuda12",
            requirements=replace(requirements, context_tokens=65_536),
            tuning=tuning,
            gpu_type="L4",
            gpu_count=1,
        )
        attestation = RuntimeAttestation(
            fingerprint=fingerprint,
            requested_context_tokens=32_768,
            effective_context_tokens=32_768,
            gpu_layers=32,
            total_layers=32,
            gpu_resident=True,
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "certificates.json"
            save_runtime_attestation(attestation, path)

            self.assertEqual(load_runtime_attestation(fingerprint, path), attestation)
            self.assertIsNone(load_runtime_attestation(changed, path))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)



class KvCacheTypeTests(unittest.TestCase):
    """The planner may vary KV precision, but never silently."""

    def _tuning(self) -> RuntimeTuning:
        return tuning_for_objective(ServingObjective.GENERAL_PURPOSE)

    def _metadata(self) -> GgufQuantMetadata:
        return GgufQuantMetadata(
            quantizations=[],
            vram_gb_by_quant={},
            block_count=65,
            embedding_length=4096,
            attention_head_count=32,
            attention_head_count_kv=8,
            attention_key_length=128,
            attention_value_length=128,
        )

    def test_quantized_cache_halves_the_full_context_footprint(self) -> None:
        requirements = serving_requirements(262144)
        f16 = self._tuning()
        q8 = with_cache_type(f16, "q8_0")
        metadata = self._metadata()

        big = estimate_memory(
            metadata,
            weights_gb=9.83,
            requirements=requirements,
            tuning=f16,
            gpu_count=1,
        )
        small = estimate_memory(
            metadata,
            weights_gb=9.83,
            requirements=requirements,
            tuning=q8,
            gpu_count=1,
        )

        self.assertEqual((q8.cache_type_k, q8.cache_type_v), ("q8_0", "q8_0"))
        # q8_0 stores 1.0625 bytes per element against f16's 2.0.
        self.assertAlmostEqual(
            small.kv_cache_gb / big.kv_cache_gb,
            1.0625 / 2.0,
            places=4,
        )
        self.assertLess(small.total_gb, big.total_gb)
        # Weights are untouched by cache precision.
        self.assertEqual(small.weights_gb, big.weights_gb)

    def test_cache_precision_changes_the_serving_fingerprint(self) -> None:
        requirements = serving_requirements(262144)
        common = {
            "model_id": "org/model",
            "revision": None,
            "quant": "Q2_K_XL",
            "runtime_id": "runtime-1",
            "requirements": requirements,
            "gpu_type": "RTX-PRO-6000",
            "gpu_count": 1,
        }
        f16 = serving_fingerprint(tuning=self._tuning(), **common)
        q8 = serving_fingerprint(
            tuning=with_cache_type(self._tuning(), "q8_0"), **common
        )

        # A cached f16 certificate must never certify a q8_0 deployment.
        self.assertNotEqual(f16, q8)

    def test_quantized_cache_requires_flash_attention(self) -> None:
        without_fa = replace(self._tuning(), flash_attention=False)

        with self.assertRaisesRegex(ValueError, "flash attention"):
            with_cache_type(without_fa, "q8_0")

        # f16 has no such coupling.
        self.assertEqual(with_cache_type(without_fa, "f16").cache_type_k, "f16")

    def test_unknown_cache_type_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported KV cache type"):
            with_cache_type(self._tuning(), "q3_k_m")

    def test_rescaling_matches_a_full_recomputation(self) -> None:
        requirements = serving_requirements(262144)
        metadata = self._metadata()
        f16 = estimate_memory(
            metadata,
            weights_gb=9.83,
            requirements=requirements,
            tuning=self._tuning(),
            gpu_count=1,
        )
        recomputed = estimate_memory(
            metadata,
            weights_gb=9.83,
            requirements=requirements,
            tuning=with_cache_type(self._tuning(), "q8_0"),
            gpu_count=1,
        )

        rescaled = memory_for_cache_type(f16, from_cache_type="f16", to_cache_type="q8_0")

        # Rescaling must agree with re-reading the GGUF geometry.
        self.assertAlmostEqual(rescaled.kv_cache_gb, recomputed.kv_cache_gb, places=2)
        self.assertAlmostEqual(rescaled.total_gb, recomputed.total_gb, places=2)
        self.assertEqual(rescaled.weights_gb, recomputed.weights_gb)

    def test_rescaling_refuses_a_heuristic_estimate(self) -> None:
        bare = GgufQuantMetadata(quantizations=[], vram_gb_by_quant={})
        fallback = estimate_memory(
            bare,
            weights_gb=9.83,
            requirements=serving_requirements(262144),
            tuning=self._tuning(),
        )
        self.assertEqual(fallback.source, "conservative-fallback")

        # The fallback ignores cache precision, so rescaling it would invent a
        # saving that the runtime will not deliver.
        with self.assertRaisesRegex(ValueError, "fallback heuristic"):
            memory_for_cache_type(fallback, from_cache_type="f16", to_cache_type="q8_0")

    def test_an_unmeasured_architecture_plans_with_f16(self) -> None:
        # The safe default must win for anything not in the allowlist, including
        # an architecture string we have never seen.
        self.assertEqual(default_cache_type("glm-dsa"), "f16")
        self.assertEqual(default_cache_type("deepseek4"), "f16")
        self.assertEqual(default_cache_type("some-future-arch"), "f16")
        self.assertEqual(default_cache_type(None), "f16")
        self.assertEqual(default_cache_type(""), "f16")

    def test_mla_architectures_are_never_allowlisted(self) -> None:
        # These compress KV in the attention design itself; quantizing on top
        # is unmeasured, so they must not be added without a bake-off.
        for architecture in ("deepseek2", "deepseek32", "deepseek4", "glm-dsa"):
            self.assertNotIn(architecture, QUANTIZED_KV_ARCHITECTURES)
            self.assertEqual(default_cache_type(architecture), "f16")

    def test_an_allowlisted_architecture_plans_with_the_measured_type(self) -> None:
        with patch(
            "llm_launchpad.core.llamacpp_planner.QUANTIZED_KV_ARCHITECTURES",
            frozenset({"qwen35"}),
        ):
            self.assertEqual(default_cache_type("qwen35"), "q8_0")
            self.assertEqual(default_cache_type("QWEN35"), "q8_0")
            self.assertEqual(default_cache_type("qwen3next"), "f16")

    def test_compiled_args_carry_the_selected_cache_type(self) -> None:
        args = compile_server_args(
            serving_requirements(262144),
            with_cache_type(self._tuning(), "q8_0"),
        )

        self.assertIn("--cache-type-k", args)
        self.assertEqual(args[args.index("--cache-type-k") + 1], "q8_0")
        self.assertEqual(args[args.index("--cache-type-v") + 1], "q8_0")


if __name__ == "__main__":
    unittest.main()
