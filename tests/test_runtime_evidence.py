"""Runtime evidence is read off llama.cpp, never off the plan."""

from __future__ import annotations

import unittest

from llm_launchpad.core.runtime_evidence import (
    combine_runtime_evidence,
    parse_fit_projection_mib,
    parse_offload_report,
    runtime_log_evidence,
    runtime_props_evidence,
)
from llm_launchpad.protocol.enums import EvidenceLevel


class OffloadReportTests(unittest.TestCase):
    def test_bare_offloading_line_names_only_the_offloaded_count(self) -> None:
        done, total = parse_offload_report("load_tensors: offloading 63 layers\n")

        self.assertEqual(done, 63)
        self.assertIsNone(total)

    def test_final_report_wins_over_earlier_attempts(self) -> None:
        text = (
            "load_tensors: offloading 30 layers\n"
            "load_tensors: offloaded 32/32 layers to GPU\n"
        )

        done, total = parse_offload_report(text)

        self.assertEqual((done, total), (32, 32))

    def test_noise_yields_no_evidence(self) -> None:
        self.assertEqual(parse_offload_report("Server is ready!\n"), (None, None))

    def test_log_evidence_grades_a_full_offload_as_observed(self) -> None:
        evidence = runtime_log_evidence("load_tensors: offloaded 32/32 layers to GPU\n")

        self.assertEqual(evidence.gpu_layers, 32)
        self.assertEqual(evidence.total_layers, 32)
        self.assertEqual(evidence.offload_evidence, EvidenceLevel.OBSERVED)
        self.assertTrue(evidence.gpu_resident)

    def test_log_evidence_grades_a_partial_offload_as_not_resident(self) -> None:
        evidence = runtime_log_evidence("load_tensors: offloaded 30/32 layers to GPU\n")

        self.assertFalse(evidence.gpu_resident)

    def test_a_bare_count_leaves_residency_unknown(self) -> None:
        evidence = runtime_log_evidence("load_tensors: offloading 63 layers\n")

        self.assertIsNone(evidence.gpu_resident)


class PropsEvidenceTests(unittest.TestCase):
    def test_props_observes_context_but_not_offload(self) -> None:
        evidence = runtime_props_evidence(
            {"default_generation_settings": {"n_ctx": 131_072}}
        )

        self.assertEqual(evidence.effective_context_tokens, 131_072)
        self.assertEqual(evidence.context_evidence, EvidenceLevel.OBSERVED)
        self.assertIsNone(evidence.offload_evidence)
        self.assertIsNone(evidence.gpu_resident)

    def test_props_without_context_is_not_evidence(self) -> None:
        evidence = runtime_props_evidence({"total_slots": 4})

        self.assertFalse(evidence.context_verified)


class FitProjectionTests(unittest.TestCase):
    def test_projection_reads_per_device_mib_without_interpreting_it(self) -> None:
        text = (
            "common_params_fit_impl:   - CUDA0 (NVIDIA A100-SXM4-80GB):  81152 total,"
            "  78810 used,   1839 free vs. target of   4096\n"
            "common_params_fit_impl:   - CUDA1 (NVIDIA A100-SXM4-80GB):  81152 total,"
            "  74738 used,   5918 free vs. target of   4096\n"
        )

        self.assertEqual(parse_fit_projection_mib(text), (78810, 74738))

    def test_combined_evidence_keeps_each_axis(self) -> None:
        combined = combine_runtime_evidence(
            runtime_props_evidence(
                {"default_generation_settings": {"n_ctx": 131_072}}
            ),
            runtime_log_evidence("load_tensors: offloaded 32/32 layers to GPU\n"),
        )

        self.assertEqual(combined.effective_context_tokens, 131_072)
        self.assertTrue(combined.gpu_resident)


if __name__ == "__main__":
    unittest.main()
