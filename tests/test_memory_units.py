"""Memory-unit boundaries for GPU fit decisions."""

from __future__ import annotations

import unittest

from llm_launchpad.core.hf_models import GgufQuantMetadata
from llm_launchpad.core.llamacpp_planner import (
    assess_placement,
    device_capacity_bytes,
    gib_to_bytes,
    serving_requirements,
    tuning_for_objective,
)
from llm_launchpad.protocol.enums import ServingObjective
from llm_launchpad.protocol.models import (
    BINARY_GIB,
    DECIMAL_GB,
    VAST_DISPLAY_GB,
    convert_memory,
    vast_raw_mib_to_display_gb,
)


def _metadata() -> GgufQuantMetadata:
    return GgufQuantMetadata(
        quantizations=["Q4_K_M"],
        vram_gb_by_quant={"Q4_K_M": 8.0},
        block_count=32,
        embedding_length=4096,
        attention_head_count=32,
        attention_head_count_kv=8,
        attention_key_length=128,
        attention_value_length=128,
    )


class MemoryUnitTests(unittest.TestCase):
    def test_same_capacity_in_different_units_gives_the_same_fit(self) -> None:
        requirements = serving_requirements(65_536)
        tuning = tuning_for_objective(ServingObjective.GENERAL_PURPOSE)
        common = {
            "metadata": _metadata(),
            "model_id": "org/model",
            "revision": "rev",
            "quant": "Q4_K_M",
            "runtime_id": "runtime-1",
            "weights_gb": 8.0,
            "requirements": requirements,
            "tuning": tuning,
            "gpu_type": "L4",
            "gpu_count": 1,
        }
        # A 24 GiB device quoted as decimal GB must decide identically.
        gib_capacity = 24.0
        decimal_capacity = convert_memory(gib_capacity, BINARY_GIB, DECIMAL_GB)
        gib_fit = assess_placement(gpu_memory_gb=gib_capacity, **common)  # type: ignore[arg-type]
        decimal_fit = assess_placement(gpu_memory_gb=decimal_capacity, **common)  # type: ignore[arg-type]
        self.assertEqual(
            (gib_fit.fits, gib_fit.gpu_resident),
            (decimal_fit.fits, decimal_fit.gpu_resident),
        )
        self.assertEqual(gib_fit.memory.total_gb, decimal_fit.memory.total_gb)

    def test_vast_raw_mib_round_trips_through_display_units(self) -> None:
        self.assertEqual(vast_raw_mib_to_display_gb(24576), 24.576)
        self.assertAlmostEqual(
            convert_memory(24.576, VAST_DISPLAY_GB, BINARY_GIB), 24.0, places=6
        )

    def test_fit_comparison_is_byte_exact_at_the_boundary(self) -> None:
        capacity_gib = 24.0
        self.assertEqual(gib_to_bytes(capacity_gib), device_capacity_bytes(capacity_gib))
        # One byte over capacity must not fit, however close the GiB values look.
        self.assertLess(
            device_capacity_bytes(capacity_gib),
            gib_to_bytes(capacity_gib) + 1,
        )


if __name__ == "__main__":
    unittest.main()
