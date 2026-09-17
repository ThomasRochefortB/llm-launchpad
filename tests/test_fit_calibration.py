"""llama.cpp's own fit measurement, captured and reused as planning evidence.

The numbers in these tests are a real rejected fit: GLM-5.3-Flash Q2_K_XL at
1,048,576 tokens on 2x A100-80GB, refused for a 433 MiB shortfall.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from llm_launchpad.core import fit_calibration as fc
from llm_launchpad.core.llamacpp_planner import (
    assess_memory_placement,
    serving_requirements,
    tuning_for_objective,
)
from llm_launchpad.protocol.enums import ServingObjective
from llm_launchpad.protocol.models import (
    MemoryEstimate,
    PlacementAssessment,
    RuntimeTuning,
)

MEASURED_LOG = """
0.01.823.408 I common_params_fit_impl:   - CUDA0 (NVIDIA A100-SXM4-80GB):  81152 total,  78810 used,   1839 free vs. target of   4096
0.01.823.410 I common_params_fit_impl:   - CUDA1 (NVIDIA A100-SXM4-80GB):  81152 total,  74738 used,   5918 free vs. target of   4096
0.01.823.411 I common_params_fit_impl: projected to use 153549 MiB of device memory vs. 161307 MiB of free device memory
0.01.823.412 I common_params_fit_impl: cannot meet free memory targets on all devices, need to use 433 MiB less in total
"""

# Weights and KV cache, which GGUF headers give exactly.
SHARDABLE_GB = 109.0 + 21.28
LAYERS = 45


def _assessment(key: str, *, compute_gb: float) -> PlacementAssessment:
    memory = MemoryEstimate(
        weights_gb=109.0,
        kv_cache_gb=21.28,
        compute_gb=compute_gb,
        speculative_gb=0.0,
        reserve_gb=0.0,
        total_gb=109.0 + 21.28 + compute_gb,
        per_device_required_gb=(109.0 + 21.28 + compute_gb,),
        total_layer_count=LAYERS,
    )
    return PlacementAssessment(
        fingerprint="fp",
        calibration_key=key,
        runtime_id="glm5next",
        memory=memory,
        tuning=RuntimeTuning(),
    )


class FitMeasurementTests(unittest.TestCase):
    def test_the_runtime_arithmetic_is_read_out_of_the_log(self) -> None:
        measurement = fc.parse_fit_measurement(MEASURED_LOG)

        self.assertEqual(measurement.per_device_used_mib, (78810, 74738))
        self.assertEqual(measurement.per_device_total_mib, (81152, 81152))
        self.assertEqual(measurement.margin_mib, 4096)
        self.assertEqual(measurement.projected_used_mib, 153549)
        self.assertTrue(measurement.is_usable)

    def test_two_devices_separate_the_graph_from_the_fixed_tensors(self) -> None:
        """One measurement, two equations, both unknowns.

        A formula can derive weights and KV cache from headers. It cannot
        derive the graph memory every device pays, nor the output and
        embedding tensors that sit on one device only -- those are what the
        measurement supplies.
        """
        calibration = fc.solve_calibration(
            fc.parse_fit_measurement(MEASURED_LOG),
            key="k",
            shardable_gb=SHARDABLE_GB,
            layer_count=LAYERS,
            gpu_type="A100-80GB",
            gpu_count=2,
        )

        assert calibration is not None
        self.assertAlmostEqual(calibration.graph_gb, 14.68, places=1)
        self.assertAlmostEqual(calibration.fixed_extra_gb, 1.37, places=1)
        # Replaying the measured topology must return the measured figures.
        busiest, quietest = calibration.per_device_gb(
            shardable_gb=SHARDABLE_GB, gpu_count=2, layer_count=LAYERS
        )
        self.assertAlmostEqual(busiest * 1000**3 / 1024**2, 78810, delta=1)
        self.assertAlmostEqual(quietest * 1000**3 / 1024**2, 74738, delta=1)

    def test_one_measurement_sizes_every_other_topology(self) -> None:
        calibration = fc.solve_calibration(
            fc.parse_fit_measurement(MEASURED_LOG),
            key="k",
            shardable_gb=SHARDABLE_GB,
            layer_count=LAYERS,
            gpu_type="A100-80GB",
            gpu_count=2,
        )
        assert calibration is not None

        def busiest(count: int) -> float:
            return max(
                calibration.per_device_gb(
                    shardable_gb=SHARDABLE_GB, gpu_count=count, layer_count=LAYERS
                )
            )

        # The measured topology is the one that failed; a third card is not a
        # guess but the same arithmetic with a different denominator.
        self.assertGreater(busiest(2) + 4.0, 80.0)
        self.assertLess(busiest(3) + 4.0, 80.0)
        self.assertLess(busiest(2) + 4.8, 96.0)

    def test_a_measurement_is_not_keyed_to_the_card_it_ran_on(self) -> None:
        """Compute buffers follow tensor shapes, so the key omits hardware.

        Keying on the GPU would make every measurement apply only to the
        topology that already failed, which is the one case it is not needed
        for.
        """
        common = {
            "model_id": "unsloth/GLM-5.3-Flash-GGUF",
            "revision": None,
            "quant": "UD-Q2_K_XL",
            "runtime_id": "glm5next",
            "requirements": serving_requirements(1_048_576),
        }
        tuning = tuning_for_objective(ServingObjective.GENERAL_PURPOSE)

        self.assertEqual(
            fc.calibration_key(tuning=tuning, **common),
            fc.calibration_key(tuning=tuning, **common),
        )
        # Anything that changes the graph must change the key.
        from dataclasses import replace

        self.assertNotEqual(
            fc.calibration_key(tuning=tuning, **common),
            fc.calibration_key(tuning=replace(tuning, ubatch_size=128), **common),
        )
        self.assertNotEqual(
            fc.calibration_key(tuning=tuning, **common),
            fc.calibration_key(
                tuning=tuning, **{**common, "requirements": serving_requirements(131_072)}
            ),
        )


class FitCalibrationRecorderTests(unittest.TestCase):
    def test_a_rejected_fit_is_recorded_rather_than_discarded(self) -> None:
        """The failures are the valuable measurements.

        A plan that fits teaches little; a plan the runtime refuses is
        precisely where the formula was wrong, and that is the run whose
        numbers were being thrown away.
        """
        with TemporaryDirectory() as directory:
            path = Path(directory) / "calibrations.json"
            recorder = fc.FitCalibrationRecorder(
                _assessment("glm-key", compute_gb=9.22), path=path
            )

            saved = [recorder.observe(line) for line in MEASURED_LOG.splitlines()]

            recorded = [item for item in saved if item is not None]
            self.assertEqual(len(recorded), 1)
            self.assertEqual(recorded[0].gpu_count, 2)
            self.assertEqual(recorded[0].gpu_type, "A100-SXM4-80GB")
            reloaded = fc.load_memory_calibration("glm-key", path)
            assert reloaded is not None
            self.assertAlmostEqual(reloaded.graph_gb, recorded[0].graph_gb, places=3)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_an_incomplete_run_records_nothing(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "calibrations.json"
            recorder = fc.FitCalibrationRecorder(
                _assessment("glm-key", compute_gb=9.22), path=path
            )

            recorder.observe(MEASURED_LOG.splitlines()[1])

            self.assertIsNone(recorder.saved)
            self.assertIsNone(fc.load_memory_calibration("glm-key", path))

    def test_a_placement_without_an_assessment_is_ignored(self) -> None:
        recorder = fc.FitCalibrationRecorder(None)

        for line in MEASURED_LOG.splitlines():
            self.assertIsNone(recorder.observe(line))


class CalibratedPlacementTests(unittest.TestCase):
    def test_a_measurement_overrides_the_formula_that_predicted_it(self) -> None:
        """This is the whole point: evidence outranks the estimate.

        The formula here is deliberately wrong -- half a gigabyte of compute
        buffers for a plan the runtime measured at 14.7 GiB per device. With
        the measurement on file the placement is refused anyway.
        """
        common = {
            "model_id": "unsloth/GLM-5.3-Flash-GGUF",
            "revision": None,
            "quant": "UD-Q2_K_XL",
            "runtime_id": "glm5next",
            "requirements": serving_requirements(1_048_576),
            "tuning": tuning_for_objective(ServingObjective.GENERAL_PURPOSE),
        }
        key = fc.calibration_key(**common)
        optimistic = _assessment(key, compute_gb=0.5).memory

        with TemporaryDirectory() as directory:
            path = Path(directory) / "calibrations.json"
            uncalibrated = assess_memory_placement(
                optimistic, gpu_type="A100-80GB", gpu_count=2, gpu_memory_gb=80.0, **common
            )
            self.assertTrue(uncalibrated.fits, "the wrong formula accepts the plan")

            recorder = fc.FitCalibrationRecorder(
                _assessment(key, compute_gb=0.5), path=path
            )
            for line in MEASURED_LOG.splitlines():
                recorder.observe(line)

            with patch.object(fc, "CALIBRATION_CACHE_PATH", path):
                calibrated = assess_memory_placement(
                    optimistic, gpu_type="A100-80GB", gpu_count=2, gpu_memory_gb=80.0, **common
                )
                roomier = assess_memory_placement(
                    optimistic, gpu_type="A100-80GB", gpu_count=3, gpu_memory_gb=80.0, **common
                )

        self.assertFalse(calibrated.fits, "the measurement refuses it")
        self.assertGreater(
            max(calibrated.memory.per_device_required_gb),
            max(uncalibrated.memory.per_device_required_gb),
        )
        # And the same measurement says which topology does work.
        self.assertTrue(roomier.fits)
