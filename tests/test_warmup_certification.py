from __future__ import annotations

import json
import types
import unittest
from unittest.mock import patch

from llm_launchpad.core.orchestrator import Orchestrator
from llm_launchpad.core.warmup import (
    _calibration_is_acceptable,
    extract_effective_context,
)
from llm_launchpad.protocol.enums import (
    BackendType,
    CertificationState,
    DeploymentState,
    OperationType,
    ServingObjective,
)
from llm_launchpad.protocol.events import OperationCompleteEvent, StateChangeEvent
from llm_launchpad.protocol.models import (
    MemoryEstimate,
    PerformancePoint,
    PlacementAssessment,
    RuntimeTuning,
    ServingRequirements,
)


class _Response:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self) -> object:
        return self._payload

    def raise_for_status(self) -> None:
        if not 200 <= self.status_code < 300:
            raise RuntimeError(f"HTTP {self.status_code}")


def _assessment() -> PlacementAssessment:
    tuning = RuntimeTuning(parallel_slots=4)
    memory = MemoryEstimate(
        weights_gb=8.0,
        kv_cache_gb=4.0,
        compute_gb=2.0,
        speculative_gb=0.0,
        reserve_gb=2.0,
        total_gb=16.0,
        per_device_required_gb=(16.0,),
        total_layer_count=32,
    )
    return PlacementAssessment(
        fingerprint="serving-fingerprint",
        memory=memory,
        tuning=tuning,
        certification=CertificationState.ESTIMATED,
        fits=True,
        gpu_resident=True,
    )


def _performance(output_tps: float = 20.0) -> tuple[PerformancePoint, ...]:
    return (
        PerformancePoint(
            prompt_tokens=512,
            output_tokens=128,
            concurrency=1,
            prompt_tokens_per_second=300.0,
            output_tokens_per_second=output_tps,
            aggregate_output_tokens_per_second=output_tps,
            error_rate=0.0,
            measured=True,
        ),
    )


class WarmupCertificationTests(unittest.TestCase):
    def test_extract_effective_context_uses_runtime_slot_context(self) -> None:
        payload = {
            "default_generation_settings": {"n_ctx": 131_072},
            "total_slots": 4,
            "model": {"n_ctx_train": 262_144},
        }

        self.assertEqual(extract_effective_context(payload), 131_072)

    def test_full_context_endpoint_is_calibrated_before_success(self) -> None:
        requirements = ServingRequirements(context_tokens=131_072)
        assessment = _assessment()
        fake_requests = types.SimpleNamespace(
            post=lambda *_args, **_kwargs: _Response(
                200,
                {"choices": [{"text": "ok"}]},
            ),
            get=lambda *_args, **_kwargs: _Response(
                200,
                {"default_generation_settings": {"n_ctx": 131_072}},
            ),
        )

        with (
            patch.dict("sys.modules", {"requests": fake_requests}),
            patch(
                "llm_launchpad.core.warmup.shutdown_event",
                return_value=types.SimpleNamespace(wait=lambda **_kwargs: False),
            ),
            patch(
                "llm_launchpad.core.warmup._calibrate_endpoint",
                return_value=_performance(),
            ) as calibrate,
            patch("llm_launchpad.core.warmup.save_runtime_attestation") as save,
            patch(
                "llm_launchpad.core.warmup.ModalBackend.test_curl_command",
                return_value="curl ok",
            ),
        ):
            events = list(
                Orchestrator().warmup(
                    backend=BackendType.LLAMACPP,
                    server_url="https://example.modal.run",
                    timeout=10,
                    tail_logs=False,
                    serving_requirements=requirements,
                    placement_assessment=assessment,
                    runtime_id="llama.cpp-b10689-cuda12",
                )
            )

        states = [
            event.current
            for event in events
            if isinstance(event, StateChangeEvent)
        ]
        self.assertLess(
            states.index(DeploymentState.VERIFYING),
            states.index(DeploymentState.CALIBRATING),
        )
        self.assertLess(
            states.index(DeploymentState.CALIBRATING),
            states.index(DeploymentState.HEALTHY),
        )
        calibrate.assert_called_once()
        save.assert_called_once()
        completion = next(
            event
            for event in events
            if isinstance(event, OperationCompleteEvent)
            and event.operation == OperationType.WARMUP
        )
        self.assertTrue(completion.success)
        attestation = completion.data["attestation"]
        self.assertEqual(attestation.effective_context_tokens, 131_072)
        self.assertEqual(attestation.gpu_layers, 32)
        self.assertTrue(attestation.gpu_resident)

    def test_runtime_with_reduced_context_is_never_published(self) -> None:
        requirements = ServingRequirements(context_tokens=131_072)
        fake_requests = types.SimpleNamespace(
            post=lambda *_args, **_kwargs: _Response(
                200,
                {"choices": [{"text": "ok"}]},
            ),
            get=lambda *_args, **_kwargs: _Response(
                200,
                {"default_generation_settings": {"n_ctx": 32_768}},
            ),
        )

        with (
            patch.dict("sys.modules", {"requests": fake_requests}),
            patch(
                "llm_launchpad.core.warmup.shutdown_event",
                return_value=types.SimpleNamespace(wait=lambda **_kwargs: False),
            ),
            patch("llm_launchpad.core.warmup._calibrate_endpoint") as calibrate,
        ):
            events = list(
                Orchestrator().warmup(
                    backend=BackendType.LLAMACPP,
                    server_url="https://example.modal.run",
                    timeout=10,
                    tail_logs=False,
                    serving_requirements=requirements,
                    placement_assessment=_assessment(),
                )
            )

        calibrate.assert_not_called()
        completion = next(
            event
            for event in events
            if isinstance(event, OperationCompleteEvent)
            and event.operation == OperationType.WARMUP
        )
        self.assertFalse(completion.success)
        self.assertIn("131,072", completion.detail)
        self.assertIn("32,768", completion.detail)

    def test_benchmark_objective_still_requires_stable_calibration(self) -> None:
        unstable = (
            PerformancePoint(
                prompt_tokens=512,
                output_tokens=128,
                concurrency=1,
                error_rate=1.0,
                measured=True,
            ),
        )

        accepted, detail = _calibration_is_acceptable(
            unstable,
            ServingObjective.BENCHMARK,
        )

        self.assertFalse(accepted)
        self.assertIn("no stable requests", detail)


if __name__ == "__main__":
    unittest.main()
