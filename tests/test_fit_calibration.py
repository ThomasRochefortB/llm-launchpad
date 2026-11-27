"""Scoped allocation projections from the native llama.cpp server."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from llm_launchpad.core import fit_calibration as fc
from llm_launchpad.core.llamacpp_planner import assess_memory_placement, serving_requirements
from llm_launchpad.protocol.enums import CertificationState, ComputeProvider
from llm_launchpad.protocol.models import (
    LLAMACPP_FIT_START_MARKER,
    GgufArtifact,
    MemoryEstimate,
    PlacementAssessment,
    RuntimeTuning,
)

_MIB = 1024**2
_ARTIFACT = GgufArtifact("a" * 40, (("model-UD-Q2_K_XL.gguf", 117_000_000_000),))
_MEMORY = MemoryEstimate(109.0, 21.28, 0.5, 0, 0, 130.78, total_layer_count=45)
_BUILD = "0.00.096.528 I cmn common_param: common_params_print_info: build 12464 (d3c3e242c) with GNU 13.3.0"
_DEVICE_ROWS = (
    "0.01.823.408 I common_params_fit_impl:   - CUDA0 (NVIDIA A100-SXM4-80GB): "
    "81152 total, 78810 used, 1839 free vs. target of 4096",
    "0.01.823.410 I common_params_fit_impl:   - CUDA1 (NVIDIA A100-SXM4-80GB): "
    "81152 total, 74738 used, 5918 free vs. target of 4096",
)
_PROJECTION = (
    "0.01.823.411 I common_params_fit_impl: "
    "projected to use 153549 MiB of device memory vs. 161307 MiB of free device memory"
)


def _placement(**overrides) -> PlacementAssessment:
    params = {
        "model_id": "unsloth/GLM-5.3-Flash-GGUF",
        "revision": None,
        "artifact": _ARTIFACT,
        "quant": "UD-Q2_K_XL",
        "runtime_id": "llama.cpp-b12464-cuda12",
        "provider": ComputeProvider.MODAL,
        "requirements": serving_requirements(1_048_576),
        "tuning": RuntimeTuning(),
        "gpu_type": "A100-80GB",
        "gpu_count": 2,
        "gpu_memory_gb": 80.0,
    }
    return assess_memory_placement(_MEMORY, **(params | overrides))


class FitCalibrationTests(unittest.TestCase):
    def test_refused_projection_is_saved_in_device_order_and_adds_headroom_once(self) -> None:
        with TemporaryDirectory() as directory, patch.object(fc, "CALIBRATION_CACHE_PATH", Path(directory)):
            assessment = _placement()
            self.assertTrue(assessment.fits)
            recorder = fc.FitCalibrationRecorder(assessment)
            recorder.observe(f"{LLAMACPP_FIT_START_MARKER} {assessment.fingerprint}")
            recorder.observe(_BUILD)
            for row in reversed(_DEVICE_ROWS):
                self.assertIsNone(recorder.observe(row))
            record = recorder.observe(_PROJECTION)
            assert record is not None
            self.assertEqual(record.per_device_bytes, (78_811 * _MIB, 74_739 * _MIB))
            self.assertEqual(recorder.headroom_shortfall_mib, 4096 - 1839)
            self.assertEqual(fc.load_memory_calibration(assessment.calibration_key), record)
            projected = _placement()
            self.assertFalse(projected.fits)
            self.assertEqual(projected.certification, CertificationState.ESTIMATED)
            self.assertEqual(projected.memory.source, "runtime-projection")
            self.assertEqual(projected.memory.per_device_required_gb[0], (78_811 + 4096) / 1024)

            # Same artifact, different topology: use the independent estimate.
            other = _placement(gpu_count=4)
            self.assertTrue(other.fits)
            self.assertNotEqual(other.memory.source, "runtime-projection")

    def test_projection_key_covers_revision_file_runtime_settings_and_hardware(self) -> None:
        with TemporaryDirectory() as directory, patch.object(fc, "CALIBRATION_CACHE_PATH", Path(directory)):
            key = _placement().calibration_key
            self.assertTrue(key)
            for change in (
                {"artifact": replace(_ARTIFACT, revision="b" * 40)},
                {"artifact": replace(_ARTIFACT, files=(("other.gguf", _ARTIFACT.size_bytes),))},
                {"runtime_id": "another-build"},
                {"gpu_type": "H100"},
                {"gpu_count": 4},
                {"gpu_memory_gb": 79.0},
                {"provider": ComputeProvider.PRIME},
                {"tuning": RuntimeTuning(parallel_slots=8)},
                {"requirements": serving_requirements(131_072)},
            ):
                with self.subTest(change=change):
                    self.assertNotEqual(key, _placement(**change).calibration_key)
            self.assertEqual(_placement(artifact=None).calibration_key, "")

    def test_legacy_standalone_log_and_incomplete_vectors_are_ignored(self) -> None:
        assessment = _placement()
        marker = f"{LLAMACPP_FIT_START_MARKER} {assessment.fingerprint}"
        with TemporaryDirectory() as directory:
            path = Path(directory)
            for lines in (
                (_BUILD, *_DEVICE_ROWS, _PROJECTION),
                (f"{LLAMACPP_FIT_START_MARKER} different-run", _BUILD, *_DEVICE_ROWS, _PROJECTION),
                (marker, _BUILD, _DEVICE_ROWS[0], _PROJECTION),
                (marker, _BUILD, _DEVICE_ROWS[0], *_DEVICE_ROWS, _PROJECTION),
                (marker, _BUILD, *_DEVICE_ROWS, _PROJECTION.replace("153549", "155000")),
                (marker, _BUILD, *_DEVICE_ROWS, _PROJECTION.replace("of device memory", "of host memory")),
                (marker, _BUILD, _DEVICE_ROWS[0].replace("4096", "2048"), _DEVICE_ROWS[1], _PROJECTION),
                (marker, _BUILD, *[line.replace("A100-SXM4", "H100") for line in _DEVICE_ROWS], _PROJECTION),
                (marker, *_DEVICE_ROWS, _PROJECTION),
                (marker, _BUILD.replace("d3c3e242c", "ccccccccc"), *_DEVICE_ROWS, _PROJECTION),
                (marker, _BUILD, "failed to measure the memory of the extra model, fitting without it",
                 *_DEVICE_ROWS, _PROJECTION),
            ):
                recorder = fc.FitCalibrationRecorder(assessment, path=path)
                for line in lines:
                    self.assertIsNone(recorder.observe(line))
            self.assertIsNone(fc.load_memory_calibration(assessment.calibration_key, path))

    def test_accepts_single_gpu_projection_from_a_real_server_log(self) -> None:
        assessment = _placement(gpu_type="RTX 3090", gpu_count=1, gpu_memory_gb=24.0)
        with TemporaryDirectory() as directory:
            recorder = fc.FitCalibrationRecorder(assessment, path=Path(directory))
            recorder.observe(f"{LLAMACPP_FIT_START_MARKER} {assessment.fingerprint}")
            recorder.observe(_BUILD)
            recorder.observe("0.00.270.736 I cmn  common_param:   - CUDA0   : NVIDIA GeForce RTX 3090 (24123 MiB, 4138 MiB free)")
            record = recorder.observe(
                "0.00.474.602 I common_params_fit_impl: "
                "projected to use 1082 MiB of device memory vs. 4138 MiB of free device memory"
            )
            assert record is not None
            self.assertEqual(record.per_device_bytes, (1083 * _MIB,))
            self.assertEqual(recorder.headroom_shortfall_mib, 0)
            self.assertTrue(recorder.topology_verified)
