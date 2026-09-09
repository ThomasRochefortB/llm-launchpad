"""Vast refusals, which must land before anything billable happens."""

from dataclasses import replace
import unittest

from llm_launchpad.core.providers import capabilities, refuse
from llm_launchpad.core.vast_runtime import vast_refusal, vast_runtime, vast_runtime_image
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import (
    DeploymentConfig,
    VastProviderOptions,
    VisionCapabilities,
)


def config(**overrides: object) -> DeploymentConfig:
    """A supported Vast llama.cpp deployment, before any refusal is provoked."""
    base = DeploymentConfig(
        backend=BackendType.LLAMACPP,
        provider=ComputeProvider.VAST,
        app_name="llp-vast-llamacpp-test",
        repo_id="acme/model-GGUF",
        quant="Q4_K_M",
        gpu_type="RTX 4090",
        gpu_count=1,
        do_deploy=True,
        provider_options=VastProviderOptions("1001", 100, 0.42, "42"),
        gguf_architecture="llama",
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


class VastRefusalTests(unittest.TestCase):
    def test_supported_config_resolves_a_digest_pinned_image(self) -> None:
        runtime = vast_runtime(config())
        self.assertIsNone(refuse(config()))
        self.assertIn("@sha256:", runtime.image)
        self.assertEqual(runtime.image, vast_runtime_image(config()))
        self.assertGreaterEqual(runtime.min_cuda_version, 12.8)

    def test_vllm_backend_is_refused_before_rental(self) -> None:
        reason = refuse(config(backend=BackendType.VLLM))
        self.assertIsNotNone(reason)
        self.assertIn("vLLM", reason or "")
        with self.assertRaises(ValueError):
            vast_runtime(config(backend=BackendType.VLLM))

    def test_vision_enabled_is_refused_before_rental(self) -> None:
        vision = VisionCapabilities(supported=True, enabled=True)
        reason = refuse(config(vision=vision))
        self.assertIsNotNone(reason)
        self.assertIn("text models only", reason or "")

    def test_pinned_revision_is_refused_before_rental(self) -> None:
        reason = refuse(config(revision="refs/pr/1"))
        self.assertIsNotNone(reason)
        self.assertIn("default HF revision", reason or "")

    def test_multi_gpu_is_refused_while_the_beta_is_single_gpu(self) -> None:
        reason = refuse(config(gpu_count=2))
        self.assertIsNotNone(reason)
        self.assertIn("single GPU", reason or "")

    def test_preload_only_is_refused_because_a_rental_must_serve(self) -> None:
        reason = refuse(config(do_deploy=False))
        self.assertIsNotNone(reason)
        self.assertIn("preload-only", reason or "")

    def test_smoke_test_only_is_refused(self) -> None:
        reason = refuse(config(run_smoke=True))
        self.assertIsNotNone(reason)
        self.assertIn("smoke-test-only", reason or "")

    def test_missing_repo_or_quant_is_refused_by_the_runtime(self) -> None:
        self.assertIsNotNone(vast_refusal(config(repo_id="")))
        self.assertIsNotNone(vast_refusal(config(quant="")))

    def test_architecture_without_a_pinned_image_is_refused(self) -> None:
        # glm5next ships a build recipe rather than a published digest.
        reason = refuse(config(gguf_architecture="glm5next"))
        self.assertIsNotNone(reason)
        self.assertIn("digest-pinned", reason or "")


class ProviderCapabilityTests(unittest.TestCase):
    def test_modal_keeps_every_capability_it_had(self) -> None:
        caps = capabilities(ComputeProvider.MODAL)
        self.assertEqual(caps.backends, frozenset({BackendType.LLAMACPP, BackendType.VLLM}))
        self.assertTrue(caps.supports_vision)
        self.assertTrue(caps.supports_smoke_test_only)
        self.assertFalse(caps.gpu_shape_from_offer)

    def test_prime_refuses_smoke_tests_and_llamacpp_revisions(self) -> None:
        prime = config(provider=ComputeProvider.PRIME, provider_options=None)
        self.assertIn("smoke-test-only", refuse(replace(prime, run_smoke=True)) or "")
        self.assertIn("default HF revision", refuse(replace(prime, revision="abc")) or "")
        # vLLM can pin a revision; llama.cpp cannot.
        vllm = replace(prime, backend=BackendType.VLLM, revision="abc", model_name="acme/model")
        self.assertIsNone(refuse(vllm))

    def test_unknown_provider_is_an_error_not_a_silent_pass(self) -> None:
        with self.assertRaises(ValueError):
            capabilities("nope")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
