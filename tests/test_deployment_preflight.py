"""Authoritative preflight: one gate for every execution path."""

from __future__ import annotations

import unittest

from llm_launchpad.core.deployment_preflight import (
    preflight_config,
    preflight_request,
    validate_request,
)
from llm_launchpad.core.quick_deploy import (
    config_from_request,
    intent_from_legacy_flags,
    request_from_config,
)
from llm_launchpad.protocol.enums import (
    BackendType,
    ComputeProvider,
    OperationIntent,
)
from llm_launchpad.protocol.models import DeploymentConfig, VastProviderOptions


def _vast_config(**overrides: object) -> DeploymentConfig:
    base = DeploymentConfig(
        backend=BackendType.LLAMACPP,
        provider=ComputeProvider.VAST,
        app_name="llp-vast-llamacpp-test",
        repo_id="acme/model-GGUF",
        quant="Q4_K_M",
        provider_options=VastProviderOptions("1001", 100, 0.42, "42"),
        gguf_architecture="llama",
    )
    kwargs = {**base.__dict__, **overrides}
    return DeploymentConfig(**kwargs)  # type: ignore[arg-type]


class IntentMappingTests(unittest.TestCase):
    def test_deploy_wins_over_smoke_like_vllm_always_did(self) -> None:
        self.assertEqual(
            intent_from_legacy_flags(do_deploy=True, run_smoke=True),
            OperationIntent.SERVE,
        )

    def test_smoke_requires_run_smoke_without_deploy(self) -> None:
        self.assertEqual(
            intent_from_legacy_flags(do_deploy=False, run_smoke=True),
            OperationIntent.SMOKE,
        )

    def test_neither_flag_means_preload(self) -> None:
        self.assertEqual(
            intent_from_legacy_flags(do_deploy=False, run_smoke=False),
            OperationIntent.PRELOAD,
        )

    def test_nothing_requested_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            intent_from_legacy_flags(do_deploy=False, run_smoke=False, preload=False)


class RequestRoundTripTests(unittest.TestCase):
    def test_request_survives_config_round_trip(self) -> None:
        config = _vast_config()
        request = request_from_config(config)
        rebuilt = config_from_request(request)
        self.assertEqual(rebuilt.backend, config.backend)
        self.assertEqual(rebuilt.provider, config.provider)
        self.assertEqual(rebuilt.repo_id, config.repo_id)
        self.assertEqual(rebuilt.quant, config.quant)
        self.assertEqual(rebuilt.do_deploy, config.do_deploy)
        self.assertEqual(rebuilt.run_smoke, config.run_smoke)

    def test_resolution_does_not_mutate_the_request(self) -> None:
        request = request_from_config(_vast_config())
        before = request
        result = preflight_request(request)
        self.assertTrue(result.ok)
        self.assertEqual(request, before)
        assert result.plan is not None
        # Identity defaults resolve onto the returned request; the caller's
        # object is untouched.
        self.assertEqual(result.plan.request.instance_name, "test")


class AuthoritativeGateTests(unittest.TestCase):
    def test_prime_preload_only_is_refused_before_any_pod(self) -> None:
        config = _vast_config(
            provider=ComputeProvider.PRIME,
            provider_options=None,
            do_deploy=False,
        )
        result = preflight_config(config)
        self.assertFalse(result.ok)
        self.assertIn("preload-only", result.findings[0].message)

    def test_prime_smoke_only_is_refused_before_any_pod(self) -> None:
        config = _vast_config(
            provider=ComputeProvider.PRIME,
            provider_options=None,
            do_deploy=False,
            run_smoke=True,
        )
        result = preflight_config(config)
        self.assertFalse(result.ok)
        self.assertTrue(
            any("smoke-test-only" in finding.message for finding in result.findings)
        )

    def test_vast_smoke_only_is_refused(self) -> None:
        result = preflight_config(
            _vast_config(do_deploy=False, run_smoke=True)
        )
        self.assertFalse(result.ok)
        self.assertTrue(
            any("smoke-test-only" in finding.message for finding in result.findings)
        )

    def test_tensor_parallelism_beyond_allocation_is_refused(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.VLLM,
            provider=ComputeProvider.MODAL,
            model_name="acme/model",
            gpu_count=1,
            n_gpu=2,
        )
        result = preflight_config(config)
        self.assertFalse(result.ok)
        self.assertEqual(
            result.findings[0].code, "vllm-tensor-parallel-exceeds-gpus"
        )

    def test_stable_codes_and_fields_for_forms(self) -> None:
        findings = validate_request(request_from_config(DeploymentConfig(
            backend=BackendType.VLLM,
            provider=ComputeProvider.MODAL,
            gpu_count=1,
            n_gpu=2,
        )))
        codes = {finding.code for finding in findings}
        self.assertIn("vllm-model-required", codes)
        self.assertIn("vllm-tensor-parallel-exceeds-gpus", codes)


if __name__ == "__main__":
    unittest.main()
