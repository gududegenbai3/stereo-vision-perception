"""Frozen research contracts: geometry, confidence and GT-free selection."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from src.confidence import combine_confidence, dps_confidence, left_right_confidence, validate_disparity
from src.confidence_experiment import aggregate, predict_with_confidence, shifted_right_pair, validate_artifacts
from src.model import ConfidenceResult, DisparityResult, RectificationGeometry, RectifiedPair, SgbmConfig
from src.disparity import compute_disparity
from src.reestimation import select_candidate
from src.reprojection import reproject_disparity


def disparity(values: np.ndarray, mask: np.ndarray | None = None) -> DisparityResult:
    values = np.array(values, dtype=np.float32)
    return DisparityResult(values, np.isfinite(values) if mask is None else mask, (0, 0, values.shape[1], values.shape[0]))


def confidence(scores: list[list[float]]) -> ConfidenceResult:
    score = np.array(scores, dtype=np.float32)
    with np.errstate(divide="ignore"):
        residual = -np.log2(score)
    return ConfidenceResult(score, residual, np.isfinite(score))


class ConfidenceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cv2.setNumThreads(1)
        (Path("artifacts/confidence_extension") / "test_temp").mkdir(parents=True, exist_ok=True)

    def test_required_runtime(self) -> None:
        import sys
        self.assertEqual(sys.version_info[:2], (3, 11))
        self.assertTrue((4, 10) <= tuple(map(int, cv2.__version__.split(".")[:2])) < (5, 0))

    def test_signed_round_trip_integer(self) -> None:
        result = left_right_confidence(disparity([[3] * 8]), disparity([[-3] * 8]))
        self.assertFalse(result.known_mask[0, :3].any())
        np.testing.assert_array_equal(result.score[0, 3:], 1)

    def test_fractional_right_sampling(self) -> None:
        result = left_right_confidence(disparity([[1.5] * 8]), disparity([[-1.0, -2.0] * 4]))
        self.assertEqual(float(result.residual_px[0, 3]), 0)
        right = disparity([[-1.0, np.nan] * 4])
        self.assertFalse(left_right_confidence(disparity([[1.5] * 8]), right).known_mask.any())

    def test_wrong_right_sign_is_detected(self) -> None:
        result = left_right_confidence(disparity([[3] * 8]), disparity([[3] * 8]))
        self.assertEqual(float(result.residual_px[0, 4]), 6)
        self.assertAlmostEqual(float(result.score[0, 4]), 1 / 64)

    def test_integer_sampling_does_not_require_next_neighbour(self) -> None:
        result = left_right_confidence(disparity([[2] * 6]), disparity([[-2, np.nan, -2, np.nan, -2, np.nan]]))
        self.assertTrue(result.known_mask[0, 2])
        self.assertFalse(result.known_mask[0, 3])

    def test_empty_support_is_unknown(self) -> None:
        result = left_right_confidence(disparity([[100] * 8]), disparity([[-100] * 8]))
        self.assertFalse(result.known_mask.any())
        self.assertTrue(np.isnan(result.score).all())

    def test_shape_and_nonfinite_valid_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_disparity(disparity([[np.nan]], np.ones((1, 1), bool)))
        with self.assertRaises(ValueError):
            left_right_confidence(disparity([[1, 1]]), disparity([[1]]))

    def test_dps_ideal_ramp_and_warp_sign(self) -> None:
        anchor = disparity([[8, 8, 8]])
        result = dps_confidence(anchor, {k: disparity([[8 + k] * 3]) for k in [-2, -1, 1, 2]})
        np.testing.assert_array_equal(result.score, 1)
        wrong = dps_confidence(anchor, {k: disparity([[8 - k] * 3]) for k in [-2, -1, 1, 2]})
        np.testing.assert_allclose(wrong.residual_px, 3)

    def test_dps_invalid_shift_intersection_and_rank(self) -> None:
        result = dps_confidence(disparity([[8, 8, 8]]), {-1: disparity([[7, 8, np.nan]]), 1: disparity([[9, 11, 9]])})
        self.assertGreater(float(result.score[0, 0]), float(result.score[0, 1]))
        self.assertTrue(np.isnan(result.score[0, 2]))
        with self.assertRaises(ValueError):
            dps_confidence(disparity([[1]]), {1: disparity([[2]])})

    def test_shift_has_no_wrap_and_updates_roi(self) -> None:
        image = np.tile(np.arange(8, dtype=np.uint8)[None, :, None], (3, 1, 3))
        pair = RectifiedPair(image.copy(), image.copy())
        geometry = RectificationGeometry((8, 3), np.eye(4), (0, 0, 8, 3), (0, 0, 8, 3))
        shifted, new_geometry = shifted_right_pair(pair, geometry, 2)
        np.testing.assert_array_equal(shifted.right_bgr[0, :, 0], [2, 3, 4, 5, 6, 7, 0, 0])
        self.assertEqual(new_geometry.roi_right, (0, 0, 6, 3))
        shifted, new_geometry = shifted_right_pair(pair, geometry, -2)
        np.testing.assert_array_equal(shifted.right_bgr[0, :, 0], [0, 0, 0, 1, 2, 3, 4, 5])
        self.assertEqual(new_geometry.roi_right, (2, 0, 6, 3))

    def test_combination_requires_both_known(self) -> None:
        result = combine_confidence(confidence([[1, np.nan]]), confidence([[0.25, 1]]))
        self.assertEqual(float(result.score[0, 0]), 0.25)
        self.assertTrue(np.isnan(result.score[0, 1]))

    def test_candidate_changes_only_known_low_confidence(self) -> None:
        base = disparity([[8, 8, 8, np.nan]])
        result = select_candidate(base, disparity([[10, 10, 10, 10]]), confidence([[1, 0.1, np.nan, np.nan]]), confidence([[1, 0.9, 1, 1]]))
        np.testing.assert_array_equal(result.replaced_mask, [[False, True, False, False]])
        np.testing.assert_array_equal(result.disparity.disparity_px, [[8, 10, 8, np.nan]])
        np.testing.assert_array_equal(base.disparity_px, [[8, 8, 8, np.nan]])

    def test_candidate_margin_and_score_validation(self) -> None:
        result = select_candidate(disparity([[8]]), disparity([[9]]), confidence([[0.45]]), confidence([[0.55]]))
        self.assertFalse(result.replaced_mask.any())
        with self.assertRaises(ValueError):
            select_candidate(disparity([[8]]), disparity([[9]]), confidence([[2]]), confidence([[1]]))

    def test_q_scale_and_nan_after_selection(self) -> None:
        image = np.zeros((1, 2, 3), np.uint8)
        q = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 100], [0, 0, 10, 20]], np.float64)
        geometry = RectificationGeometry((2, 1), q, (0, 0, 2, 1), (0, 0, 2, 1))
        result = reproject_disparity(RectifiedPair(image, image), geometry, disparity([[8, np.nan]]))
        self.assertAlmostEqual(float(result.depth_m[0, 0]), 1)
        self.assertTrue(np.isnan(result.xyz_m[0, 1]).all())

    def test_actual_inference_is_gt_independent_and_baseline_unchanged(self) -> None:
        rng = np.random.default_rng(23)
        left = rng.integers(0, 256, (32, 160, 3), np.uint8)
        right = np.zeros_like(left)
        right[:, :-6] = left[:, 6:]
        pair = RectifiedPair(left, right)
        geometry = RectificationGeometry((160, 32), np.eye(4), (0, 0, 160, 32), (0, 0, 160, 32))
        config = SgbmConfig(0, 16, 3)
        truth = np.full((32, 160), 6, np.float32)
        first = predict_with_confidence(pair, geometry, config, replace(config, block_size=5))
        truth[:] = rng.uniform(-50, 50, truth.shape)
        second = predict_with_confidence(pair, geometry, config, replace(config, block_size=5))
        raw = compute_disparity(pair, geometry, config)
        np.testing.assert_array_equal(first.baseline.disparity_px, raw.disparity_px)
        np.testing.assert_array_equal(first.baseline_dps.score, second.baseline_dps.score)
        np.testing.assert_array_equal(first.refined.disparity.disparity_px, second.refined.disparity.disparity_px)

    def test_artifact_verifier_rejects_missing_and_nonfinite_valid(self) -> None:
        with tempfile.TemporaryDirectory(dir="artifacts/confidence_extension/test_temp") as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                validate_artifacts(root, ["missing"])
            path = root / "scene"
            path.mkdir()
            (path / "report.json").write_text(json.dumps({"scene": "scene", "status": "success"}))
            values = np.array([[np.nan]], np.float32)
            mask = np.array([[True]])
            np.savez(path / "arrays.npz", baseline=values, baseline_valid=mask, candidate=values, candidate_valid=mask, final=values, final_valid=mask, replaced=np.array([[False]]))
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                validate_artifacts(root, ["scene"])

    def test_failed_scene_cannot_disappear_from_summary(self) -> None:
        method = {"mae_px": 1.0, "rmse_px": 2.0, "bad2_fraction": 0.1, "coverage": 0.9}
        successful = {"status": "success", "metrics": {name: method for name in ["baseline", "lrc_filter", "dps_filter", "refined"]}}
        result = aggregate([successful, {"status": "failed", "reason": "missing input"}])
        self.assertEqual(result["total_scenes"], 2)
        self.assertEqual(result["failed_scenes"], 1)
        self.assertIsNone(result["baseline"]["mae_px"])

    def test_invalid_dps_shift_type_is_explicit_error(self) -> None:
        with self.assertRaises(ValueError):
            dps_confidence(disparity([[8]]), {"1": disparity([[9]])})
