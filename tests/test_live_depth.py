"""Protect resized reprojection and rejection of unsupported matches."""

from pathlib import Path
from threading import Event
import unittest

import numpy as np

from src.camera_io import calibrated_inputs
from src.live_depth import LiveGeometry, LiveProcessor, make_depth, scaled_q


class LiveDepthTests(unittest.TestCase):
    def test_scaled_grid_has_the_same_xyz(self) -> None:
        q: np.ndarray = np.array([[1, 0, 0, -320], [0, 1, 0, -210],
                                  [0, 0, 0, 1000], [0, 0, 10, -2.5]], dtype=np.float64)
        native: np.ndarray = q @ np.array([120.5, 300.5, 40.0, 1])
        half: np.ndarray = scaled_q(q, 0.5) @ np.array([60.0, 150.0, 20.0, 1])
        np.testing.assert_allclose(half[:3] / half[3], native[:3] / native[3])

    def test_depth_preserves_invalid_matches_in_both_camera_masks(self) -> None:
        q: np.ndarray = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                                  [0, 0, 0, 20], [0, 0, 10, 0]], dtype=np.float64)
        disparity: np.ndarray = np.array([[2, 0.5, 0.5, np.nan, 1, 1, 193]], dtype=np.float32)
        inputs: dict[str, np.ndarray] = {
            "left_support": np.array([[255, 255, 255, 255, 0, 255, 255]], dtype=np.uint8),
            "right_support": np.array([[255, 0, 255, 255, 255, 255, 255]], dtype=np.uint8),
        }
        values, depth = make_depth(disparity, inputs, q, 192)
        np.testing.assert_array_equal(np.isfinite(values), [[False, False, False, False, False, True, False]])
        np.testing.assert_array_equal(np.isfinite(values), np.isfinite(depth))
        self.assertEqual(depth[0, 5], 2.0)


    def test_live_support_masks_reused_with_identical_rectification(self) -> None:
        width, height = 80, 48
        y, x = np.indices((height, width), dtype=np.float32)
        maps: dict[str, np.ndarray] = {
            "image_size": np.array([width, height]),
            "map1_x": x - 1.25, "map1_y": y,
            "map2_x": x + 1.25, "map2_y": y + 0.375,
        }
        geometry = LiveGeometry(maps, (width, height), np.eye(4), 0.5)
        processor = LiveProcessor(geometry, {}, Path("."), Path("."), 8, Event())
        rng = np.random.default_rng(1)
        first, second = rng.integers(0, 256, (2, height, 2 * width, 3), dtype=np.uint8)
        initial = processor.prepare(first)
        expected = calibrated_inputs(second[:, :width], second[:, width:], maps, 0.5)
        updated = processor.prepare(second)
        for name, value in expected.items():
            np.testing.assert_array_equal(updated[name], value)
        for name in ("left_support", "right_support"):
            self.assertIs(updated[name], initial[name])
            self.assertTrue(np.any(updated[name] == 0))
            self.assertTrue(np.any(updated[name] == 255))


if __name__ == "__main__":
    unittest.main()
