"""Regression checks for geometry, masks, PFM bytes and reusable pure stages."""
from __future__ import annotations

import sys
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stereo_vision.config import load_config
from stereo_vision.datasets import load_middlebury_scene, read_pfm
from stereo_vision.evaluation import evaluate_disparity
from stereo_vision.models import DisparityResult, GroundTruth, RectifiedGeometry
from stereo_vision.outputs import save_point_cloud
from stereo_vision.pipeline import reconstruct_scene
from stereo_vision.reprojection import reconstruct

ROOT = Path(__file__).resolve().parents[1]


class OfflineContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        (ROOT / "artifacts").mkdir(exist_ok=True)

    def test_config_rejects_unknown_fields_and_boolean_parameters(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            path = Path(directory) / "config.json"
            settings = json.loads((ROOT / "config/middlebury.json").read_text())
            settings["sgbm"]["block_size"] = True
            path.write_text(json.dumps(settings))
            with self.assertRaisesRegex(ValueError, "integer"):
                load_config(path)
            settings["sgbm"]["block_size"] = 5
            settings["unknown"] = 1
            path.write_text(json.dumps(settings))
            with self.assertRaisesRegex(ValueError, "expected fields"):
                load_config(path)

    def test_reprojection_keeps_doffs_and_metre_scale(self) -> None:
        q = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 100], [0, 0, 10, 20]], dtype=np.float64)
        mask = np.array([[True, False]])
        disparity = DisparityResult(np.array([[8, np.nan]], dtype=np.float32), mask, np.ones_like(mask), 16)
        result = reconstruct(disparity, RectifiedGeometry(q, (0, 0, 2, 1), (0, 0, 2, 1)))
        self.assertAlmostEqual(float(result.depth_m[0, 0]), 1.0)
        self.assertTrue(np.isnan(result.xyz_m[0, 1]).all())

    def test_evaluation_excludes_occlusions_and_counts_missing_predictions(self) -> None:
        predicted = np.array([[5, 80, np.nan]], dtype=np.float32)
        prediction = DisparityResult(predicted, np.isfinite(predicted), np.ones((1, 3), bool), 16)
        truth = GroundTruth(np.array([[4, 4, 4]], dtype=np.float32), np.array([[True, False, True]]))
        metrics = evaluate_disparity(prediction, truth)
        self.assertEqual(metrics.reference_pixels, 2)
        self.assertEqual(metrics.coverage, 0.5)
        self.assertEqual(metrics.epe_px, 1)

    def test_pfm_endianness_row_order_and_sample_units(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            path = Path(directory) / "disparity.pfm"
            for scale, dtype in (("-0.003922", "<f4"), ("1", ">f4")):
                path.write_bytes(f"Pf\n2 2\n{scale}\n".encode() + np.array([[3, 4], [1, 2]], dtype=dtype).tobytes())
                np.testing.assert_array_equal(read_pfm(path), [[1, 2], [3, 4]])
            path.write_bytes(b"Pf\n2 2\n-1\n")
            with self.assertRaisesRegex(ValueError, "payload"):
                read_pfm(path)

    def test_cloud_preserves_rgb_and_point_count(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            path = Path(directory) / "cloud.ply"
            save_point_cloud(path, np.array([[1, 2, 3]], np.float32), np.array([[255, 0, 1]], np.uint8))
            text = path.read_text()
            self.assertIn("element vertex 1", text)
            self.assertEqual(text.splitlines()[-1], "1 2 3 255 0 1")

    def test_scene_truth_cannot_change_reconstruction(self) -> None:
        config = load_config(ROOT / "config/middlebury.json")
        scene = load_middlebury_scene(config.data_root, config.scene, config.split)
        with_truth = reconstruct_scene(scene, config.sgbm)
        without_truth = reconstruct_scene(replace(scene, ground_truth=None), config.sgbm)
        np.testing.assert_array_equal(with_truth.disparity.disparity_px, without_truth.disparity.disparity_px)
        np.testing.assert_array_equal(with_truth.reconstruction.xyz_m, without_truth.reconstruction.xyz_m)
        self.assertIsNone(without_truth.metrics)
        self.assertEqual(int(scene.ground_truth.nonoccluded_mask.sum()), 325114)

    def test_missing_scene_and_mismatched_truth_report_errors(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_middlebury_scene(ROOT, "missing_scene", "trainingQ")
        prediction = DisparityResult(np.ones((1, 2), np.float32), np.ones((1, 2), bool), np.ones((1, 2), bool), 16)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            evaluate_disparity(prediction, GroundTruth(np.ones((2, 2), np.float32), np.ones((2, 2), bool)))


if __name__ == "__main__":
    unittest.main()
