"""Category labels say billions of parameters, so that is what must be measured."""

from __future__ import annotations

import unittest

from llm_launchpad.core.hf_models import GgufQuantMetadata
from llm_launchpad.core.quick_deploy_refresh import (
    _size_bucket_from_gguf_metadata,
    _total_parameters_b,
)


def _metadata(sizes: dict[str, float]) -> GgufQuantMetadata:
    return GgufQuantMetadata(quantizations=list(sizes), vram_gb_by_quant=dict(sizes))


# Real published size tables. Each is the whole list, junk rows included.
FLASH_NEXT = {
    "UD-IQ1_S": 72.5, "UD-IQ1_M": 74.5, "UD-Q2_K_XL": 78.9, "UD-IQ3_XXS": 82.0,
    "UD-Q3_K_XL": 90.0, "UD-IQ4_XS": 93.7, "Q4_K_M": 2.79, "UD-Q4_K_XL": 111.0,
    "UD-Q5_K_XL": 158.0, "UD-Q6_K_XL": 169.0, "Q8_0": 192.0, "BF16": 354.0,
}
QWEN_122B = {
    "BF16": 244.0, "UD-Q8_K_XL": 171.0, "Q8_0": 130.0, "UD-Q6_K_XL": 112.0,
    "Q6_K": 101.0, "UD-Q5_K_XL": 91.9, "Q5_K_S": 86.4, "UD-Q4_K_XL": 77.0,
    "Q4_K_S": 71.7, "UD-Q3_K_XL": 57.0, "UD-IQ3_XXS": 44.7, "UD-Q2_K_XL": 41.8,
    "UD-IQ1_M": 34.2,
}
QWEN_27B = {
    "BF16": 54.7, "UD-Q8_K_XL": 31.5, "Q8_0": 29.0, "UD-Q6_K_XL": 25.3,
    "UD-Q5_K_XL": 20.9, "UD-Q4_K_XL": 17.6, "UD-Q4_K_S": 15.4, "UD-Q3_K_XL": 13.1,
    "UD-Q2_K_XL": 9.83, "UD-IQ1_S": 6.19, "Q4_0": 1.37, "UD-Q8_K_L": 0.0,
}
DEEPSEEK_FLASH = {
    "Q8_0": 10.9, "BF16": 11.3, "UD-IQ1_S": 82.5, "UD-IQ2_XXS": 90.9,
    "UD-Q2_K_XL": 96.8, "UD-IQ3_XXS": 104.0, "UD-Q3_K_XL": 128.0,
    "UD-IQ4_XS": 137.0, "UD-Q4_K_XL": 155.0, "UD-Q8_K_XL": 162.0,
}


class TotalParameterEstimateTests(unittest.TestCase):
    def test_published_weights_give_the_parameter_count(self) -> None:
        for name, sizes, expected in (
            ("Qwen3.8-27B", QWEN_27B, 27.4),
            ("Qwen3.5-122B-A10B", QWEN_122B, 122.0),
        ):
            with self.subTest(model=name):
                got = _total_parameters_b(_metadata(sizes))
                assert got is not None
                self.assertLess(abs(got - expected) / expected, 0.10)

    def test_a_mixture_of_experts_is_counted_whole(self) -> None:
        # 122B total, 10B active. Weights cover every expert and every expert
        # has to be resident, so the count that matters is the total.
        got = _total_parameters_b(_metadata(QWEN_122B))
        assert got is not None
        self.assertGreater(got, 100.0)
        self.assertEqual(_size_bucket_from_gguf_metadata(_metadata(QWEN_122B)), "medium")

    def test_a_stray_small_file_does_not_shrink_the_model(self) -> None:
        # Unsloth's Qwen3.8-Flash-Next page lists a 2.79 GB Q4_K_M beside a
        # 111 GB one. Reading the first match put this 177B model in compact.
        self.assertEqual(_size_bucket_from_gguf_metadata(_metadata(FLASH_NEXT)), "large")

    def test_a_stray_wide_file_does_not_shrink_the_model(self) -> None:
        # DeepSeek-V4-Flash lists an 11 GB BF16 beside a 155 GB Q4. A 16-bit
        # copy cannot be smaller than a 4-bit copy, so that row is a projector
        # or a shard; preferring the widest width read it as a 5.7B model.
        self.assertEqual(_size_bucket_from_gguf_metadata(_metadata(DEEPSEEK_FLASH)), "large")

    def test_gigabytes_are_not_treated_as_billions_of_parameters(self) -> None:
        # The old thresholds compared GB of Q4 weights against a bucket named
        # in billions of parameters, so "Compact <=40B" admitted ~71B models.
        sixty_billion = {"BF16": 120.0, "UD-Q4_K_XL": 36.0, "UD-Q2_K_XL": 19.2}

        self.assertEqual(_size_bucket_from_gguf_metadata(_metadata(sixty_billion)), "medium")

    def test_one_width_cannot_outvote_the_table_by_publishing_more_files(self) -> None:
        # Thirteen Q4 variants and one of everything else must not let the Q4
        # row's own error dominate the estimate.
        crowded = dict(QWEN_27B)
        crowded.update({f"Q4_K_S_{index}": 15.4 for index in range(13)})

        got = _total_parameters_b(_metadata(crowded))
        assert got is not None
        self.assertLess(abs(got - 27.4) / 27.4, 0.10)

    def test_an_empty_or_unreadable_table_yields_no_opinion(self) -> None:
        self.assertIsNone(_total_parameters_b(_metadata({})))
        self.assertIsNone(_total_parameters_b(_metadata({"mystery": 40.0})))
        self.assertIsNone(_size_bucket_from_gguf_metadata(_metadata({})))


class SizeBucketBoundaryTests(unittest.TestCase):
    def test_the_buckets_match_the_labels_they_are_shown_under(self) -> None:
        for params_b, expected in ((7.0, "compact"), (40.0, "compact"),
                                   (41.0, "medium"), (150.0, "medium"), (151.0, "large")):
            with self.subTest(params_b=params_b):
                sizes = {"BF16": params_b * 2.0, "UD-Q4_K_XL": params_b * 0.60}
                self.assertEqual(_size_bucket_from_gguf_metadata(_metadata(sizes)), expected)
