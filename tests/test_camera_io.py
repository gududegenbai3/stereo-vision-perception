"""Capture integrity and known-disparity checks; no camera/GPU required."""

import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

import cv2
import numpy as np

from src.camera_io import calibrated_inputs, capture_inventory, sgbm_predict
from src.capture_ui import save_raw_pair


class CameraIoTests(unittest.TestCase):
    def test_cpu_entry_points_import_when_optional_network_packages_are_blocked(self) -> None:
        code: str = """
import builtins
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torchvision', 'timm'}:
        raise ImportError('Optional package deliberately unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
import src.capture_ui, src.calibration, src.rectification
import src.display_calibration, src.display_calibration_report
import src.camera_comparison, src.camera_report, src.pretrained_stereo
"""
        result: subprocess.CompletedProcess[str] = subprocess.run([sys.executable, "-s", "-c", code], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_saved_pair_roundtrip_and_tampering_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root: Path = Path(temporary)
            image: np.ndarray = np.random.default_rng(1).integers(0, 256, (40, 120, 3), dtype=np.uint8)
            save_raw_pair(root, image, "pair_1", 123)
            (root / "DONE.json").write_text(json.dumps({"saved_pairs": 1}))
            rows = capture_inventory(root)
            self.assertEqual(rows[0]["wh"], [60, 40])
            (root / "pair_1/left.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash"):
                capture_inventory(root)

    def test_incomplete_or_repeated_inventory_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root: Path = Path(temporary)
            save_raw_pair(root, np.zeros((20, 80, 3), np.uint8), "pair_1", 123)
            (root / "DONE.json").write_text(json.dumps({"saved_pairs": 2}))
            with self.assertRaises(ValueError):
                capture_inventory(root)
            csv: Path = root / "pairs.csv"
            csv.write_text(csv.read_text() + csv.read_text().splitlines()[1] + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                capture_inventory(root)

    def test_fixed_identity_maps_preserve_pixels_and_half_size(self) -> None:
        height, width = 48, 80
        y, x = np.indices((height, width), dtype=np.float32)
        maps = {"image_size": np.array([width, height]), "map1_x": x, "map1_y": y, "map2_x": x, "map2_y": y}
        image: np.ndarray = np.random.default_rng(2).integers(0, 256, (height, width, 3), dtype=np.uint8)
        native = calibrated_inputs(image, image, maps, 1.0)
        np.testing.assert_array_equal(native["left"], image)
        self.assertTrue(np.all(native["right_support"] == 255))
        half = calibrated_inputs(image, image, maps, .5)
        np.testing.assert_array_equal(half["left"], cv2.resize(image, (40, 24), interpolation=cv2.INTER_AREA))
        with self.assertRaises(ValueError):
            calibrated_inputs(image[:, :-1], image, maps, 1.0)

    def test_sgbm_recovers_known_textured_shift_with_search_border_invalid(self) -> None:
        left: np.ndarray = np.random.default_rng(3).integers(0, 256, (64, 256, 3), dtype=np.uint8)
        right: np.ndarray = np.zeros_like(left)
        right[:, :-8] = left[:, 8:]
        parameters = {"block":5, "input":"gray", "mode":"SGBM", "uniqueness_ratio":0, "speckle_window_size":0, "speckle_range":0, "disp12_max_diff":0}
        prediction = sgbm_predict({"left":left, "right":right}, parameters, 32)
        self.assertTrue(np.isnan(prediction[:, :32]).all())
        self.assertLess(float(np.nanmean(np.abs(prediction[8:-8, 40:-16] - 8))), .1)


if __name__ == "__main__":
    unittest.main()
