"""Paper recurrence against a scalar reference, plus actual synthetic stereo."""

import unittest

import numpy as np

from src.model import RectificationGeometry, RectifiedPair
from src.sgm import DIRECTIONS, SgmSettings, aggregate_paths, bt_cost_volume, compute_sgm_disparity, lr_mask, penalty_values


class SgmContracts(unittest.TestCase):
    def test_eight_path_recurrence_against_scalar_reference(self) -> None:
        rng: np.random.Generator = np.random.default_rng(7)
        cost: np.ndarray = rng.integers(0, 10, (3, 4, 5)).astype(np.float32)
        gray: np.ndarray = rng.integers(0, 30, (3, 4), dtype=np.uint8)
        settings: SgmSettings = SgmSettings(0, 5, 2, 10, 1)
        for mode in ("fixed", "adaptive"):
            expected: np.ndarray = np.zeros_like(cost)
            for dx, dy in DIRECTIONS:
                path: np.ndarray = np.zeros_like(cost)
                for y in (range(3) if dy >= 0 else range(2, -1, -1)):
                    for x in (range(4) if dx >= 0 else range(3, -1, -1)):
                        px, py = x-dx, y-dy
                        if not (0 <= px < 4 and 0 <= py < 3):
                            path[y, x] = cost[y, x]
                            continue
                        previous: np.ndarray = path[py, px]
                        p2: float = 10 if mode == "fixed" else max(2, 10/max(1, abs(int(gray[y,x])-int(gray[py,px]))))
                        for d in range(5):
                            candidates: list[float] = [float(previous[d]), float(previous.min()) + p2]
                            if d > 0:
                                candidates.append(float(previous[d-1]) + 2)
                            if d < 4:
                                candidates.append(float(previous[d+1]) + 2)
                            path[y,x,d] = cost[y,x,d] + min(candidates) - previous.min()
                expected += path
            actual, _ = aggregate_paths(cost, gray, settings, mode)
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-5)

    def test_flat_intensity_makes_three_penalty_modes_identical(self) -> None:
        cost: np.ndarray = np.random.default_rng(4).random((3,4,5)).astype(np.float32)
        gray: np.ndarray = np.full((3,4), 50, np.uint8)
        settings: SgmSettings = SgmSettings(0,5,2,10,1)
        fixed, _ = aggregate_paths(cost, gray, settings, "fixed")
        for mode in ("adaptive", "matched_mean"):
            other, mean = aggregate_paths(cost, gray, settings, mode)
            np.testing.assert_array_equal(other, fixed)
            self.assertEqual(mean, 10)

    def test_penalty_is_finite_and_uint8_subtraction_does_not_wrap(self) -> None:
        settings: SgmSettings = SgmSettings(0,5,2,32,1)
        values: np.ndarray = penalty_values(np.array([0,255,5],np.uint8),np.array([255,0,5],np.uint8),settings)
        np.testing.assert_array_equal(values,[2,2,32])

    def test_pixel_bt_and_invalid_projection(self) -> None:
        left: np.ndarray = np.random.default_rng(3).integers(0,256,(6,24),dtype=np.uint8)
        right: np.ndarray = np.zeros_like(left)
        right[:,:-3] = left[:,3:]
        cost: np.ndarray = bt_cost_volume(left,right,SgmSettings(0,8,2,10,1))
        np.testing.assert_array_equal(cost[:,4:-2,3],np.zeros((6,18),np.float32))
        self.assertTrue((cost[:,:3,3] >= 60000).all())

    def test_right_positive_convention_and_zero_disparity(self) -> None:
        left: np.ndarray = np.array([[0,0,0,3,3,3]],np.float32)
        right: np.ndarray = np.full_like(left,3)
        valid: np.ndarray = np.ones_like(left,bool)
        mask: np.ndarray = lr_mask(left,right,valid,valid,1)
        self.assertTrue(mask[0,3])
        self.assertFalse(lr_mask(left,-right,valid,valid,1)[0,3])
        self.assertTrue(lr_mask(left,np.zeros_like(left),valid,valid,1)[0,0])

    def test_complete_synthetic_stereo_and_input_errors(self) -> None:
        rng: np.random.Generator = np.random.default_rng(12)
        left: np.ndarray = rng.integers(0,256,(24,80,3),dtype=np.uint8)
        right: np.ndarray = np.zeros_like(left)
        right[:,:-6] = left[:,6:]
        full: tuple[int,int,int,int] = (0,0,80,24)
        geometry: RectificationGeometry = RectificationGeometry((80,24),np.eye(4,dtype=np.float64),full,full)
        result = compute_sgm_disparity(RectifiedPair(left,right),geometry,SgmSettings(0,16,8,32,1),"adaptive")
        middle: np.ndarray = result.final.disparity_px[3:-3,20:-12]
        self.assertGreater(np.isfinite(middle).mean(),.9)
        self.assertLess(float(np.nanmedian(np.abs(middle-6))),.1)
        with self.assertRaises(ValueError):
            bt_cost_volume(left[:,:,0],right[:,:,0],SgmSettings(0,100,8,32,1))


if __name__ == "__main__":
    unittest.main()
