"""Known synthetic camera geometry checks; no real validation data used for tests."""

import unittest

import cv2
import numpy as np

from src.calibration import BoardGeometry, CornerPair, StereoCalibration, calibrate_stereo, canonical_corner_order, object_points
from src.rectification import build_rectification, rectify_corner_pair


class StereoCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cv2.setNumThreads(1)
        cls.board: BoardGeometry = BoardGeometry(9, 6, .01794)
        cls.k1: np.ndarray = np.array([[800., 0, 320], [0, 810, 240], [0, 0, 1]])
        cls.k2: np.ndarray = np.array([[790., 0, 318], [0, 800, 242], [0, 0, 1]])
        cls.d: np.ndarray = np.array([-.04, .01, .001, -.001, -.002])
        cls.rotation: np.ndarray = cv2.Rodrigues(np.array([.005, -.012, .004]))[0]
        cls.translation: np.ndarray = np.array([[-.12], [.001], [.002]])
        rng: np.random.Generator = np.random.default_rng(20261122)
        points: np.ndarray = object_points(cls.board)
        cls.pairs: list[CornerPair] = []
        for index in range(20):
            rvec: np.ndarray = rng.uniform(-.4, .4, 3)
            rotation: np.ndarray = cv2.Rodrigues(rvec)[0]
            translation: np.ndarray = np.array([rng.uniform(-.13, -.01), rng.uniform(-.09, .04), rng.uniform(.55, 1.05)]).reshape(3, 1)
            left: np.ndarray = cv2.projectPoints(points, rvec, translation, cls.k1, cls.d)[0].astype(np.float32)
            right: np.ndarray = cv2.projectPoints(points, cv2.Rodrigues(cls.rotation @ rotation)[0], cls.rotation @ translation + cls.translation, cls.k2, cls.d)[0].astype(np.float32)
            cls.pairs.append(CornerPair(str(index), left, right))
        cls.fitted: StereoCalibration = calibrate_stereo(cls.pairs, cls.board, (640, 480))

    def test_recovers_known_intrinsics_rotation_and_metre_baseline(self) -> None:
        np.testing.assert_allclose(self.fitted.k1, self.k1, atol=.05)
        np.testing.assert_allclose(self.fitted.k2, self.k2, atol=.05)
        np.testing.assert_allclose(self.fitted.rotation, self.rotation, atol=1e-4)
        np.testing.assert_allclose(self.fitted.translation_m, self.translation, atol=1e-5)
        self.assertLess(self.fitted.rms_stereo_px, .001)

    def test_target_scale_changes_translation_not_camera_pixel_intrinsics(self) -> None:
        doubled: StereoCalibration = calibrate_stereo(self.pairs, BoardGeometry(9, 6, .03588), (640, 480))
        np.testing.assert_allclose(doubled.k1, self.fitted.k1, atol=.05)
        np.testing.assert_allclose(doubled.translation_m, self.fitted.translation_m * 2, atol=1e-5)

    def test_rectified_correspondences_are_horizontal_at_original_size(self) -> None:
        maps = build_rectification(self.fitted)
        left, right = rectify_corner_pair(self.pairs[-1], self.fitted, maps)
        self.assertLess(float(np.max(np.abs(left[:, 1] - right[:, 1]))), .002)
        self.assertEqual(maps.map1_x.shape, (480, 640))
        self.assertGreater(float(np.median(left[:, 0] - right[:, 0])), 0)

    def test_resolves_180_degree_detector_order_but_rejects_mirror(self) -> None:
        original: np.ndarray = self.pairs[0].left_xy
        np.testing.assert_array_equal(canonical_corner_order(original[::-1], self.board), original)
        mirrored: np.ndarray = original.reshape(6, 9, 1, 2)[:, ::-1].reshape(54, 1, 2)
        with self.assertRaises(ValueError):
            canonical_corner_order(mirrored, self.board)


if __name__ == "__main__":
    unittest.main()
