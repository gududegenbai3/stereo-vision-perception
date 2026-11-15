"""Fair comparison contracts: units, holes, shared domains, random scenes and hashes."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.banet_sgbm_comparison import compare_predictions
from src.model import DisparityResult, RectificationGeometry, RectifiedPair
from src.stereo_comparison_data import choose_random_views, mobile_group, resize_pair, restore_native
from src.stereo_leakage_audit import image_fingerprint


class ComparisonContracts(unittest.TestCase):
    def test_scaling_and_q_preserve_original_camera_coordinates(self) -> None:
        image: np.ndarray = np.zeros((8, 12, 3), np.uint8)
        q: np.ndarray = np.array([[1, 0, 0, -6], [0, 1, 0, -4], [0, 0, 0, 10], [0, 0, 2, 0]], np.float64)
        geometry: RectificationGeometry = RectificationGeometry((12, 8), q, (0, 0, 12, 8), (0, 0, 12, 8))
        pair, scaled = resize_pair(RectifiedPair(image, image.copy()), geometry, .5)
        self.assertEqual(pair.left_bgr.shape, (4, 6, 3))
        vector: np.ndarray = np.array([2, 1, 3, 1], np.float64)
        original: np.ndarray = np.array([(2 + .5) / .5 - .5, (1 + .5) / .5 - .5, 3 / .5, 1])
        transformed: list[float] = [sum(float(scaled.q[row, column] * vector[column]) for column in range(4)) for row in range(4)]
        expected: list[float] = [sum(float(geometry.q[row, column] * original[column]) for column in range(4)) for row in range(4)]
        np.testing.assert_allclose(transformed, expected)

    def test_restore_changes_disparity_units_and_never_fills_invalid_holes(self) -> None:
        values: np.ndarray = np.full((4, 6), 3.5, np.float32)
        valid: np.ndarray = np.ones_like(values, bool)
        restored: DisparityResult = restore_native(values, valid, (8, 12), .5)
        np.testing.assert_array_equal(restored.disparity_px, np.full((8, 12), 7, np.float32))
        valid[1, 2] = False
        values[1, 2] = np.nan
        holed: DisparityResult = restore_native(values, valid, (8, 12), .5)
        self.assertFalse(holed.valid_mask[2:4, 4:6].any())
        self.assertTrue(np.isnan(holed.disparity_px[~holed.valid_mask]).all())
        np.testing.assert_allclose(holed.disparity_px[holed.valid_mask], 7)

    def test_same_pixel_errors_and_missing_penalty_prevent_apparent_improvement(self) -> None:
        truth: np.ndarray = np.zeros((2, 4), np.float32)
        reference: np.ndarray = np.ones((2, 4), bool)
        sparse: np.ndarray = np.array([[0, np.nan, 2, np.nan], [0, np.nan, 0, np.nan]], np.float32)
        sgbm: DisparityResult = DisparityResult(sparse, np.isfinite(sparse), (0, 0, 4, 2))
        dense: DisparityResult = DisparityResult(np.full((2, 4), 3, np.float32), np.ones((2, 4), bool), (0, 0, 4, 2))
        results: dict = compare_predictions(truth, reference, {"sgbm128": sgbm, "sgbm192": sgbm, "banet": dense}, np.full((2, 4), 3, np.uint8), 2)
        self.assertEqual(results["methods"]["sgbm192"]["bad_or_missing_fraction"], .5)
        self.assertEqual(results["methods"]["banet"]["bad_or_missing_fraction"], 1)
        self.assertEqual(results["pairs"]["sgbm192"]["sgbm192"]["evaluated_pixels"], 4)
        self.assertEqual(results["pairs"]["sgbm192"]["banet"]["evaluated_pixels"], 4)
        self.assertEqual(results["pairs"]["sgbm192"]["banet"]["mae_px"], 3)

    def test_random_display_samples_are_distinct_physical_groups(self) -> None:
        names: list[str] = [f"{name}{index}" for name in ("artroom", "bandsaw", "chess", "curule", "ladder", "skates", "terrace") for index in (1, 2)]
        chosen: list[str] = choose_random_views(names, 20261115, 6)
        self.assertEqual(chosen, choose_random_views(list(reversed(names)), 20261115, 6))
        self.assertEqual(len({mobile_group(name) for name in chosen}), 6)
        with self.assertRaises(ValueError):
            choose_random_views(names, 1, 4)

    def test_decoded_hash_catches_recompressed_copies(self) -> None:
        rng: np.random.Generator = np.random.default_rng(7)
        image: np.ndarray = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            first: Path = Path(directory) / "a.png"
            second: Path = Path(directory) / "b.png"
            cv2.imwrite(str(first), image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite(str(second), image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            a: dict = image_fingerprint(first)
            b: dict = image_fingerprint(second)
            self.assertNotEqual(a["encoded_sha256"], b["encoded_sha256"])
            self.assertEqual(a["decoded_sha256"], b["decoded_sha256"])
            self.assertEqual(a["phash63"], b["phash63"])


if __name__ == "__main__":
    unittest.main()
