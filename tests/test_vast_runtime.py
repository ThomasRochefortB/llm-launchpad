"""Vast refusals, which must land before anything billable happens."""

from dataclasses import replace
import unittest

from llm_launchpad.core.providers import capabilities, refuse
from llm_launchpad.core.vast_runtime import (
    GpuDevice,
    parse_gpu_inventory,
    vast_refusal,
    vast_runtime,
    vast_runtime_image,
    vast_runtime_script,
    verify_gpu_topology,
)
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import (
    DeploymentConfig,
    ProjectorArtifact,
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

    def test_vision_enabled_stages_the_projector_and_passes_mmproj(self) -> None:
        artifact = ProjectorArtifact(
            repo_id="acme/model-GGUF", revision="main", filename="mmproj.gguf", size_bytes=1024
        )
        vision = VisionCapabilities(supported=True, enabled=True, projector=artifact)
        candidate = config(vision=vision, endpoint_api_key="private-key")
        self.assertIsNone(refuse(candidate))
        script = vast_runtime_script(candidate)
        self.assertIn("--mmproj", script)
        self.assertNotIn("--no-mmproj", script)
        self.assertIn("/root/.llm-launchpad/projectors/", script)
        self.assertIn("curl --fail --location", script)

    def test_vision_disabled_keeps_no_mmproj(self) -> None:
        candidate = config(endpoint_api_key="private-key")
        self.assertIn("--no-mmproj", vast_runtime_script(candidate))

    def test_vision_enabled_without_a_projector_is_an_error(self) -> None:
        vision = VisionCapabilities(supported=True, enabled=True)
        with self.assertRaises(ValueError):
            vast_runtime_script(config(vision=vision, endpoint_api_key="k"))

    def test_pinned_revision_is_refused_before_rental(self) -> None:
        reason = refuse(config(revision="refs/pr/1"))
        self.assertIsNotNone(reason)
        self.assertIn("default HF revision", reason or "")

    def test_multi_gpu_is_allowed_up_to_the_bundle_ceiling(self) -> None:
        for count in (1, 2, 4, 8):
            self.assertIsNone(refuse(config(gpu_count=count)), count)
        reason = refuse(config(gpu_count=9))
        self.assertIsNotNone(reason)
        self.assertIn("at most 8 GPUs", reason or "")

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


class GpuInventoryTests(unittest.TestCase):
    def test_parses_nvidia_smi_rows_in_device_order(self) -> None:
        devices = parse_gpu_inventory(
            "1, NVIDIA A100-PCIE-80GB, 81920, 512\n0, NVIDIA A100-PCIE-80GB, 81920, 0"
        )
        self.assertEqual([device.index for device in devices], [0, 1])
        self.assertAlmostEqual(devices[0].memory_free_gib, 80.0)

    def test_malformed_output_yields_no_devices_rather_than_a_guess(self) -> None:
        for output in ("", "no devices found", "0, GPU, notanumber, 0", "0, GPU, 100"):
            self.assertEqual(parse_gpu_inventory(output), ())


def device(index: int, name: str = "NVIDIA RTX 4090", total: int = 24564, used: int = 0) -> GpuDevice:
    return GpuDevice(index=index, name=name, memory_total_mib=total, memory_used_mib=used)


class TopologyVerificationTests(unittest.TestCase):
    def test_matching_homogeneous_devices_pass(self) -> None:
        verify_gpu_topology(
            [device(0), device(1)], gpu_count=2, per_device_required_gb=20.0
        )

    def test_a_host_reporting_no_gpus_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "did not report any GPUs"):
            verify_gpu_topology([], gpu_count=1, per_device_required_gb=0.0)

    def test_a_device_count_that_differs_from_the_rental_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "rented 2 GPUs but the host exposes 1"):
            verify_gpu_topology([device(0)], gpu_count=2, per_device_required_gb=0.0)

    def test_mixed_gpu_models_are_refused_because_the_plan_divides_evenly(self) -> None:
        with self.assertRaisesRegex(ValueError, "mixes GPU models"):
            verify_gpu_topology(
                [device(0), device(1, name="NVIDIA RTX 3090")],
                gpu_count=2,
                per_device_required_gb=0.0,
            )

    def test_a_device_without_enough_free_memory_is_refused(self) -> None:
        # A leftover process on one card is invisible in the offer listing.
        with self.assertRaisesRegex(ValueError, "GPU 1 has"):
            verify_gpu_topology(
                [device(0), device(1, used=20000)],
                gpu_count=2,
                per_device_required_gb=20.0,
            )

