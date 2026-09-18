from __future__ import annotations

import json
import types
import unittest
from unittest.mock import patch

from llm_launchpad.core.orchestrator import Orchestrator
from llm_launchpad.core.warmup import (
    StartupPhaseTimer,
    _calibration_is_acceptable,
    extract_effective_context,
    parse_startup_phase_line,
    startup_phase_summary_line,
)
from llm_launchpad.core.runtime_evidence import LlamacppRuntimeEvidence
from llm_launchpad.protocol.enums import (
    BackendType,
    CertificationState,
    DeploymentState,
    EvidenceLevel,
    OperationType,
    ServingObjective,
)
from llm_launchpad.protocol.events import (
    LogEvent,
    OperationCompleteEvent,
    StateChangeEvent,
)
from llm_launchpad.protocol.models import (
    MemoryEstimate,
    PerformancePoint,
    PlacementAssessment,
    RuntimeAttestation,
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


_OFFLOAD_LOG = "load_tensors: offloaded 32/32 layers to GPU\n"


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
            actual_prompt_tokens=512,
            actual_output_tokens=128,
            sample_count=1,
            completion_reason="calibration-curve",
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
            patch(
                "llm_launchpad.core.warmup._ModalLogTail",
                return_value=types.SimpleNamespace(
                    seen_lines=set(_OFFLOAD_LOG.splitlines()),
                    attach=lambda: iter(()),
                    drain=lambda: iter(()),
                    stop=lambda _reason: None,
                    may_attach=lambda: False,
                ),
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

    def _warmup_without_tail_evidence(self, **patches: object) -> list[object]:
        """Run a GPU-only warmup whose live log tail captured no offload report."""

        requirements = ServingRequirements(context_tokens=131_072)
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
        recovered = patches.get("refetched", LlamacppRuntimeEvidence())
        remembered = patches.get("remembered")

        with (
            patch.dict("sys.modules", {"requests": fake_requests}),
            patch(
                "llm_launchpad.core.warmup.shutdown_event",
                return_value=types.SimpleNamespace(wait=lambda **_kwargs: False),
            ),
            patch(
                "llm_launchpad.core.warmup._calibrate_endpoint",
                return_value=_performance(),
            ),
            patch("llm_launchpad.core.warmup.save_runtime_attestation"),
            patch(
                "llm_launchpad.core.warmup.ModalBackend.test_curl_command",
                return_value="curl ok",
            ),
            patch(
                "llm_launchpad.core.warmup._refetched_log_evidence",
                return_value=recovered,
            ),
            patch(
                "llm_launchpad.core.warmup.load_runtime_attestation",
                return_value=remembered,
            ),
            patch(
                "llm_launchpad.core.warmup._ModalLogTail",
                return_value=types.SimpleNamespace(
                    seen_lines=set(),
                    attach=lambda: iter(()),
                    drain=lambda: iter(()),
                    stop=lambda _reason: None,
                    may_attach=lambda: False,
                ),
            ),
        ):
            return list(
                Orchestrator().warmup(
                    backend=BackendType.LLAMACPP,
                    server_url="https://example.modal.run",
                    timeout=10,
                    tail_logs=False,
                    serving_requirements=requirements,
                    placement_assessment=_assessment(),
                    runtime_id="llama.cpp-b10689-cuda12",
                )
            )

    @staticmethod
    def _warmup_completion(events: list[object]) -> OperationCompleteEvent:
        return next(
            event
            for event in events
            if isinstance(event, OperationCompleteEvent)
            and event.operation == OperationType.WARMUP
        )

    def test_a_lost_offload_report_is_recovered_from_the_retained_logs(self) -> None:
        """The live tail can lose a burst; the retained logs still have it.

        Modal coalesced the 70-second window containing the offload report on
        a deploy that had already loaded and served, so the stream's silence
        was about the stream, not about the model.
        """

        events = self._warmup_without_tail_evidence(
            refetched=LlamacppRuntimeEvidence(
                gpu_layers=32,
                total_layers=32,
                offload_evidence=EvidenceLevel.OBSERVED,
                detail="runtime reported 32/32 layers on GPU",
            )
        )

        completion = self._warmup_completion(events)
        self.assertTrue(completion.success)
        attestation = completion.data["attestation"]
        self.assertTrue(attestation.gpu_resident)
        self.assertEqual(attestation.residency_evidence, EvidenceLevel.OBSERVED)

    def test_a_lost_offload_report_falls_back_to_this_placements_certificate(self) -> None:
        """An earlier run of this exact fingerprint already observed residency."""

        events = self._warmup_without_tail_evidence(
            remembered=RuntimeAttestation(
                fingerprint="serving-fingerprint",
                requested_context_tokens=131_072,
                effective_context_tokens=131_072,
                gpu_layers=32,
                total_layers=32,
                gpu_resident=True,
                verified_at="2026-09-15T16:15:16+00:00",
            )
        )

        completion = self._warmup_completion(events)
        self.assertTrue(completion.success)
        self.assertEqual(
            completion.data["attestation"].residency_evidence,
            EvidenceLevel.OBSERVED,
        )
        self.assertTrue(
            any(
                isinstance(event, LogEvent) and "earlier run" in event.line
                for event in events
            )
        )

    def test_unverifiable_residency_keeps_the_endpoint_and_grades_it(self) -> None:
        """Missing telemetry is unknown: it must not destroy a serving endpoint.

        The endpoint has loaded, answered /props and served a completion. The
        certificate says residency is predicted rather than observed, which is
        the honest reading, and the run continues.
        """

        events = self._warmup_without_tail_evidence()

        completion = self._warmup_completion(events)
        self.assertTrue(completion.success)
        attestation = completion.data["attestation"]
        self.assertEqual(attestation.residency_evidence, EvidenceLevel.PREDICTED)
        self.assertTrue(
            any(
                isinstance(event, LogEvent)
                and "residency could not be verified" in event.line.casefold()
                for event in events
            )
        )

    def test_a_partially_offloaded_runtime_is_rejected(self) -> None:
        """An observed 30/32 offload report fails a GPU-only request."""

        requirements = ServingRequirements(context_tokens=131_072)
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
            patch("llm_launchpad.core.warmup._calibrate_endpoint") as calibrate,
            patch(
                "llm_launchpad.core.warmup._ModalLogTail",
                return_value=types.SimpleNamespace(
                    seen_lines={"load_tensors: offloaded 30/32 layers to GPU"},
                    attach=lambda: iter(()),
                    drain=lambda: iter(()),
                    stop=lambda _reason: None,
                    may_attach=lambda: False,
                ),
            ),
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
            patch(
                "llm_launchpad.core.warmup._ModalLogTail",
                return_value=types.SimpleNamespace(
                    seen_lines=set(_OFFLOAD_LOG.splitlines()),
                    attach=lambda: iter(()),
                    drain=lambda: iter(()),
                    stop=lambda _reason: None,
                    may_attach=lambda: False,
                ),
            ),
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


class StartupPhaseLineTests(unittest.TestCase):
    def test_phase_line_round_trips(self) -> None:
        line = startup_phase_summary_line("deploy", 12.34)
        self.assertEqual(line, "startup-phase deploy 12.3s")
        self.assertEqual(parse_startup_phase_line(line), ("deploy", 12.3))

    def test_phase_line_rejects_unknown_names_and_noise(self) -> None:
        self.assertIsNone(parse_startup_phase_line("Server is ready!"))
        self.assertIsNone(parse_startup_phase_line("startup-phase frobnicate 1.0s"))
        self.assertIsNone(parse_startup_phase_line("startup-phase deploy nope"))


class StartupPhaseTimerTests(unittest.TestCase):
    def _events(self, **kwargs: object) -> list:
        requirements = ServingRequirements(context_tokens=131_072)
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
            ),
            patch("llm_launchpad.core.warmup.save_runtime_attestation"),
            patch(
                "llm_launchpad.core.warmup.ModalBackend.test_curl_command",
                return_value="curl ok",
            ),
            patch(
                "llm_launchpad.core.warmup._ModalLogTail",
                return_value=types.SimpleNamespace(
                    seen_lines=set(_OFFLOAD_LOG.splitlines()),
                    attach=lambda: iter(()),
                    drain=lambda: iter(()),
                    stop=lambda _reason: None,
                    may_attach=lambda: False,
                ),
            ),
        ):
            return list(
                Orchestrator().warmup(
                    backend=BackendType.LLAMACPP,
                    server_url="https://example.modal.run",
                    timeout=10,
                    tail_logs=False,
                    serving_requirements=requirements,
                    placement_assessment=_assessment(),
                    runtime_id="llama.cpp-b10689-cuda12",
                    **kwargs,  # type: ignore[arg-type]
                )
            )

    def test_warmup_emits_phase_lines_with_timer(self) -> None:
        timer = StartupPhaseTimer()
        timer.deploy_started()
        timer.warmup_started()
        events = self._events(phase_timer=timer)
        completion = next(
            event
            for event in events
            if isinstance(event, OperationCompleteEvent)
            and event.operation == OperationType.WARMUP
        )
        self.assertTrue(completion.success)
        parsed = [
            parse_startup_phase_line(event.line)
            for event in events
            if isinstance(event, LogEvent)
        ]
        names = {row[0] for row in parsed if row is not None}
        self.assertEqual(names, {"warmup-wait", "calibration", "total"})

    def test_warmup_without_timer_emits_no_phase_lines(self) -> None:
        events = self._events()
        lines = [
            event.line for event in events if isinstance(event, LogEvent)
        ]
        self.assertFalse(any("startup-phase" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
