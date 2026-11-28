"""Check the displayed target geometry and reject incomplete stereo detections."""

import unittest

import cv2
import numpy as np

from src.calibration_pattern import detect_paired_corners, make_chessboard


class CalibrationPatternTests(unittest.TestCase):
    def test_exact_target_detects_54_corners_in_both_halves(self) -> None:
        image: np.ndarray = make_chessboard(9, 6, 48)
        self.assertEqual(image.shape, (432, 576))
        self.assertTrue(np.all(image[:48] == 240))
        np.testing.assert_array_equal(np.unique(image), [20, 240])
        bgr: np.ndarray = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        left, right = detect_paired_corners(np.concatenate([bgr, bgr], axis=1), 9, 6)
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        self.assertEqual(left.shape, (54, 1, 2))
        np.testing.assert_allclose(left, right)
        self.assertAlmostEqual(abs(float(np.median(np.diff(left.reshape(6, 9, 2), axis=1)[:, :, 0]))), 48, delta=.1)

    def test_missing_right_board_does_not_count_as_a_complete_pair(self) -> None:
        bgr: np.ndarray = cv2.cvtColor(make_chessboard(9, 6, 32), cv2.COLOR_GRAY2BGR)
        left, right = detect_paired_corners(np.concatenate([bgr, np.full_like(bgr, 240)], axis=1), 9, 6)
        self.assertIsNotNone(left)
        self.assertIsNone(right)


if __name__ == "__main__":
    unittest.main()
