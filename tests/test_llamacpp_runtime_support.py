from __future__ import annotations

import unittest
from pathlib import Path

from llm_launchpad.core.prime_backend import default_prime_container_image
from llm_launchpad.core.runtime_support import (
    DEFAULT_LLAMACPP_IMAGE_REF,
    LlamaCppSupportManifest,
    RuntimeCompatibility,
    evaluate_llamacpp_architecture,
    extract_llamacpp_architectures,
    load_llamacpp_support_manifest,
)
from llm_launchpad.protocol.enums import BackendType


class LlamaCppRuntimeSupportTests(unittest.TestCase):
    def test_bundled_manifest_matches_pinned_runtime_and_known_architectures(self) -> None:
        manifest = load_llamacpp_support_manifest()

        self.assertEqual(manifest.image_ref, DEFAULT_LLAMACPP_IMAGE_REF)
        self.assertEqual(manifest.runtime_build, "b10689")
        self.assertIn("qwen35", manifest.architectures)
        self.assertIn("glm-dsa", manifest.architectures)
        self.assertNotIn("glm5next", manifest.architectures)

    def test_modal_and_prime_defaults_match_support_manifest(self) -> None:
        manifest = load_llamacpp_support_manifest()
        backend_source = (
            Path(__file__).resolve().parents[1]
            / "llm_launchpad"
            / "backends"
            / "modal_llamacpp_app.py"
        ).read_text(encoding="utf-8")

        self.assertEqual(manifest.image_ref, DEFAULT_LLAMACPP_IMAGE_REF)
        self.assertEqual(
            default_prime_container_image(BackendType.LLAMACPP),
            manifest.image_ref,
        )
        self.assertIn(manifest.image_ref, backend_source)

    def test_evaluator_blocks_architecture_absent_from_exact_runtime(self) -> None:
        decision = evaluate_llamacpp_architecture("glm5next")

        self.assertEqual(decision.status, RuntimeCompatibility.UNSUPPORTED)
        self.assertFalse(decision.is_supported)
        self.assertIn("b10689", decision.message)

    def test_evaluator_treats_custom_image_as_unknown(self) -> None:
        decision = evaluate_llamacpp_architecture(
            "glm5next",
            image_ref="example.invalid/custom-llamacpp:latest",
        )

        self.assertEqual(decision.status, RuntimeCompatibility.UNKNOWN)

    def test_extract_architectures_ignores_dummy_and_unknown_rows(self) -> None:
        source = """
        { LLM_ARCH_CLIP, "clip" },
        { LLM_ARCH_LLAMA, "llama" },
        { LLM_ARCH_QWEN35, "qwen35" },
        { LLM_ARCH_UNKNOWN, "(unknown)" },
        """

        self.assertEqual(extract_llamacpp_architectures(source), ["llama", "qwen35"])

    def test_evaluator_accepts_manifest_compatible_image_alias(self) -> None:
        manifest = LlamaCppSupportManifest(
            runtime_id="test-runtime",
            runtime_build="b1",
            image_ref="example/runtime:server-cuda-b1",
            image_digest="sha256:test",
            source_revision="abc",
            source_url="https://example.invalid/source",
            generated_at="2026-08-30T00:00:00Z",
            architectures=frozenset({"llama"}),
            compatible_image_refs=frozenset({"example/runtime:server-cuda12-b1"}),
        )

        decision = evaluate_llamacpp_architecture(
            "llama",
            image_ref="example/runtime:server-cuda12-b1",
            manifest=manifest,
        )

        self.assertEqual(decision.status, RuntimeCompatibility.SUPPORTED)


if __name__ == "__main__":
    unittest.main()
