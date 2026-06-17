from __future__ import annotations

import unittest

from llm_launchpad.core.storage_costs import (
    MODAL_VOLUME_FREE_TIER_GIB_MONTH,
    MODAL_VOLUME_PRICE_PER_GIB_MONTH_USD,
    billable_storage_gib_month,
    estimate_monthly_storage_cost,
    estimated_monthly_storage_cost_usd,
    gross_monthly_storage_cost_usd,
    storage_gib_month,
)
from llm_launchpad.protocol.enums import BackendType
from llm_launchpad.protocol.models import StorageSnapshot, StoredModelInfo


_GIB = 1024**3


class StorageCostTests(unittest.TestCase):
    def test_storage_gib_month_uses_binary_gib(self) -> None:
        self.assertEqual(storage_gib_month(_GIB), 1.0)
        self.assertEqual(storage_gib_month(5 * _GIB), 5.0)

    def test_negative_bytes_are_treated_as_zero(self) -> None:
        self.assertEqual(storage_gib_month(-1), 0.0)
        self.assertEqual(gross_monthly_storage_cost_usd(-1), 0.0)
        self.assertEqual(billable_storage_gib_month(-1), 0.0)
        self.assertEqual(estimated_monthly_storage_cost_usd(-1), 0.0)

    def test_gross_monthly_cost_uses_modal_list_rate(self) -> None:
        self.assertEqual(MODAL_VOLUME_PRICE_PER_GIB_MONTH_USD, 0.09)
        self.assertAlmostEqual(gross_monthly_storage_cost_usd(10 * _GIB), 0.90)

    def test_estimated_cost_is_zero_under_free_tier(self) -> None:
        self.assertEqual(MODAL_VOLUME_FREE_TIER_GIB_MONTH, 1024.0)
        self.assertEqual(billable_storage_gib_month(1024 * _GIB), 0.0)
        self.assertEqual(estimated_monthly_storage_cost_usd(1024 * _GIB), 0.0)

    def test_estimated_cost_charges_only_above_free_tier(self) -> None:
        self.assertEqual(billable_storage_gib_month(1025 * _GIB), 1.0)
        self.assertAlmostEqual(estimated_monthly_storage_cost_usd(1025 * _GIB), 0.09)

    def test_estimate_monthly_storage_cost_uses_snapshot_total(self) -> None:
        snapshot = StorageSnapshot(
            llamacpp_models=[
                StoredModelInfo(
                    backend=BackendType.LLAMACPP,
                    model_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
                    size_bytes=1024 * _GIB,
                    file_count=1,
                    source_volume="huggingface-cache",
                )
            ],
            vllm_models=[
                StoredModelInfo(
                    backend=BackendType.VLLM,
                    model_id="Qwen/Qwen3-4B",
                    size_bytes=2 * _GIB,
                    file_count=2,
                    source_volume="huggingface-cache",
                )
            ],
        )

        estimate = estimate_monthly_storage_cost(snapshot)

        self.assertEqual(estimate.total_size_bytes, 1026 * _GIB)
        self.assertEqual(estimate.total_gib_month, 1026.0)
        self.assertAlmostEqual(estimate.gross_monthly_cost_usd, 92.34)
        self.assertEqual(estimate.billable_gib_month, 2.0)
        self.assertAlmostEqual(estimate.estimated_monthly_cost_usd, 0.18)


if __name__ == "__main__":
    unittest.main()
