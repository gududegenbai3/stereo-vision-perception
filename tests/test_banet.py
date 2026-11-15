"""Geometry, preprocessing and evaluation boundaries of optional BANet adapter."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.banet import padding_for_shape, prepare_vendor, valid_disparity
from src.banet_experiment import evaluate_predictions


class BanetContracts(unittest.TestCase):
    def test_padding_preserves_original_pixel_grid(self) -> None:
        for height, width in ((496, 718), (32, 32), (1, 65), (497, 739)):
            left, right, top, bottom = padding_for_shape(height, width)
            self.assertEqual((width + left + right) % 32, 0)
            self.assertEqual((height + top + bottom) % 32, 0)
            self.assertLessEqual(abs(left - right), 1)
            self.assertLessEqual(abs(top - bottom), 1)
            original: np.ndarray = np.arange(height * width).reshape(height, width)
            padded: np.ndarray = np.pad(original, ((top, bottom), (left, right)), mode="edge")
            np.testing.assert_array_equal(padded[top:top + height, left:left + width], original)
        with self.assertRaises(ValueError):
            padding_for_shape(0, 32)

    def test_pixel_disparity_uses_right_projection_without_wrap_or_rescale(self) -> None:
        values: np.ndarray = np.array([[0, 2, 2, -1, np.nan, np.inf, 192, 3.5]], np.float32)
        result = valid_disparity(values, 192)
        np.testing.assert_array_equal(result.valid_mask, [[True, False, True, False, False, False, False, True]])
        self.assertEqual(result.disparity_px[0, 7], 3.5)
        self.assertTrue(np.isnan(result.disparity_px[~result.valid_mask]).all())
        with self.assertRaises(ValueError):
            valid_disparity(values.astype(np.float64), 192)

    def test_comparison_counts_missing_and_uses_same_pixels(self) -> None:
        truth: np.ndarray = np.array([[0, 0, 0, 0, 0]], np.float32)
        baseline = valid_disparity(np.array([[0, np.nan, 2, 0, np.nan]], np.float32), 192)
        banet = valid_disparity(np.array([[0, 0, 0, 3, 0]], np.float32), 192)
        rows: dict = evaluate_predictions({"baseline": baseline, "banet": banet}, truth,
            np.ones((1, 5), bool), np.zeros((1, 5), np.uint8))
        self.assertEqual(rows["baseline"]["missing_pixels"], 2)
        self.assertEqual(rows["baseline"]["bad_pixels"], 0)  # Exactly 2px is not Bad2.
        self.assertEqual(rows["baseline"]["coverage"], .6)
        paired: dict = rows["baseline"]["paired_with_banet"]
        self.assertEqual(paired["method"]["evaluated_pixels"], 3)
        self.assertEqual(paired["banet"]["evaluated_pixels"], 3)
        self.assertEqual(paired["banet"]["bad_pixels"], 1)

    def test_vendor_patch_preserves_source_and_rejects_unknown_interface(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root: Path = Path(directory)
            source: Path = root / "source"
            source.mkdir()
            for name in ("__init__.py", "BANet.py", "aggregation.py", "submodule.py"):
                (source / name).write_bytes(b"# original\r\n")
            encoder: bytes = b"pretrained=True\nself.act1 = model.act1\n"
            (source / "fnet.py").write_bytes(encoder)
            manifest: dict = prepare_vendor(source, root / "runtime")
            self.assertEqual((source / "fnet.py").read_bytes(), encoder)
            self.assertEqual((root / "runtime" / "BANet.py").read_bytes(), b"# original\r\n")
            self.assertNotEqual(manifest["files"]["fnet.py"]["source_sha256"], manifest["files"]["fnet.py"]["runtime_sha256"])
            (source / "fnet.py").write_text("changed API", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unexpected upstream"):
                prepare_vendor(source, root / "runtime")


if __name__ == "__main__":
    unittest.main()
