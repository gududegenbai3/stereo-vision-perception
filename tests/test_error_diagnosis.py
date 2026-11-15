"""Verify diagnosis domains and percentages independently of image matching."""

import unittest

import numpy as np

from src.error_diagnosis import DiagnosisConfig, classify_regions, jump_band, measure, pool, texture_std
from src.model import DisparityResult, SgbmConfig


class ErrorDiagnosisContracts(unittest.TestCase):
    def test_unknown_gt_does_not_create_a_jump(self) -> None:
        truth: np.ndarray = np.array([[10, 10, np.inf, 40, 40]], np.float32)
        self.assertFalse(jump_band(truth, 2, 1).any())
        truth[0, 1] = 20
        np.testing.assert_array_equal(jump_band(truth, 2, 0), [[True, True, False, False, False]])

    def test_region_priority_partition_and_reference_exclusion(self) -> None:
        truth: np.ndarray = np.array([[5, 5, 20, 200, 200, 200]], np.float32)
        reference: np.ndarray = np.array([[True, True, True, True, True, False]])
        std: np.ndarray = np.array([[0, 10, 0, 0, 0, 0]], np.float32)
        settings: DiagnosisConfig = DiagnosisConfig(2, 2, 0, 5, 5)
        labels: np.ndarray = classify_regions(truth, reference, std, SgbmConfig(0, 128, 5), settings)
        np.testing.assert_array_equal(labels, [[3, 2, 2, 1, 1, 0]])
        self.assertEqual(int((labels > 0).sum()), int(reference.sum()))

    def test_bad_threshold_missing_and_pool_denominators(self) -> None:
        truth: np.ndarray = np.array([[10, 10, 10, 10]], np.float32)
        values: np.ndarray = np.array([[10, 12, 13, np.nan]], np.float32)
        prediction: DisparityResult = DisparityResult(values, np.isfinite(values), (0, 0, 4, 1))
        stats: dict = measure(truth, prediction, np.ones((1, 4), bool), 2)
        self.assertEqual(stats["bad_pixels"], 1)
        self.assertEqual(stats["missing_pixels"], 1)
        self.assertEqual(stats["coverage"], .75)
        self.assertAlmostEqual(stats["bad_fraction"], 1 / 3)
        combined: dict = pool([stats, stats])
        self.assertEqual(combined["evaluated_pixels"], 6)
        self.assertEqual(combined["bad_pixels"], 2)
        self.assertAlmostEqual(combined["mae_px"], 5 / 3)
        self.assertAlmostEqual(combined["rmse_px"], np.sqrt(13 / 3))

    def test_empty_reference_is_undefined_not_zero_error(self) -> None:
        values: np.ndarray = np.full((1, 3), 10, np.float32)
        prediction: DisparityResult = DisparityResult(values, np.ones((1, 3), bool), (0, 0, 3, 1))
        stats: dict = measure(values, prediction, np.zeros((1, 3), bool), 2)
        self.assertIsNone(stats["mae_px"])
        self.assertIsNone(pool([stats])["bad_fraction"])

    def test_texture_and_invalid_inputs(self) -> None:
        image: np.ndarray = np.full((7, 7, 3), 50, np.uint8)
        np.testing.assert_array_equal(texture_std(image, 5), np.zeros((7, 7), np.float32))
        with self.assertRaises(ValueError):
            texture_std(image, 4)
        with self.assertRaises(ValueError):
            jump_band(np.ones((2, 2), np.float32), 0, 1)
        invalid: DisparityResult = DisparityResult(np.full((1, 1), np.nan, np.float32), np.ones((1, 1), bool), (0, 0, 1, 1))
        with self.assertRaises(ValueError):
            measure(np.ones((1, 1), np.float32), invalid, np.ones((1, 1), bool), 2)


if __name__ == "__main__":
    unittest.main()
