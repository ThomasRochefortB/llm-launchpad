"""Bit width is a quality fact, and has to survive the labels it hides in."""

from __future__ import annotations

import unittest

from llm_launchpad.core.quant_quality import (
    QUALITY_FLOOR_BITS,
    is_reduced_quality,
    quant_bits,
    quant_quality_label,
    serving_quality_bits,
)


class QuantBitsTests(unittest.TestCase):
    def test_reads_the_width_out_of_every_label_shape_in_the_catalog(self) -> None:
        cases = {
            "UD-Q2_K_XL": 2,
            "UD_Q2_K_XL": 2,
            "Q4_K_M": 4,
            "UD-Q4_K_XL": 4,
            "Q4_K_S": 4,
            "IQ2_XXS": 2,
            "IQ1_S": 1,
            "Q5_K_M": 5,
            "Q6_K": 6,
            "Q8_0": 8,
        }
        for label, bits in cases.items():
            with self.subTest(label=label):
                self.assertEqual(quant_bits(label), bits)

    def test_reads_widths_their_spelling_does_not_give_away(self) -> None:
        self.assertEqual(quant_bits("MXFP4_MOE"), 4)
        self.assertEqual(quant_bits("BF16"), 16)
        self.assertEqual(quant_bits("F16"), 16)

    def test_an_unreadable_label_is_unknown_rather_than_small(self) -> None:
        for label in (None, "", "   ", "custom-build"):
            with self.subTest(label=label):
                self.assertIsNone(quant_bits(label))
                # Unknown is not evidence of degradation. Scoring it zero would
                # rank it below 2-bit and quietly bury whatever it is.
                self.assertFalse(is_reduced_quality(label))
                self.assertEqual(serving_quality_bits(label), QUALITY_FLOOR_BITS)


class QualityFloorTests(unittest.TestCase):
    def test_four_bits_is_the_floor_and_is_not_called_lossless(self) -> None:
        self.assertEqual(QUALITY_FLOOR_BITS, 4)
        for label in ("UD-Q2_K_XL", "IQ2_XXS", "Q3_K_M", "IQ1_S"):
            with self.subTest(label=label):
                self.assertTrue(is_reduced_quality(label))
        for label in ("Q4_K_M", "UD-Q4_K_XL", "Q6_K", "Q8_0", "BF16"):
            with self.subTest(label=label):
                self.assertFalse(is_reduced_quality(label))

    def test_names_the_width_for_a_caller_that_must_disclose_it(self) -> None:
        self.assertEqual(quant_quality_label("UD-Q2_K_XL"), "2-bit")
        self.assertEqual(quant_quality_label("Q8_0"), "8-bit")
        self.assertEqual(quant_quality_label("custom-build"), "")
