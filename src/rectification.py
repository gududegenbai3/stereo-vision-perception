"""Calibrated horizontal stereo rectification; map/input size is preserved."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from .calibration import CornerPair, StereoCalibration
from .model import RawPair, RectifiedPair


@dataclass(frozen=True)
class StereoRectification:
    """Float32 H,W maps; R dimensionless, P in pixel/metre convention, Q uses m."""
    r1: NDArray[np.float64]
    r2: NDArray[np.float64]
    p1: NDArray[np.float64]
    p2: NDArray[np.float64]
    q: NDArray[np.float64]
    map1_x: NDArray[np.float32]
    map1_y: NDArray[np.float32]
    map2_x: NDArray[np.float32]
    map2_y: NDArray[np.float32]
    roi1: tuple[int, int, int, int]
    roi2: tuple[int, int, int, int]
    image_size_wh: tuple[int, int]


def build_rectification(calibration: StereoCalibration, alpha: float = 0.0) -> StereoRectification:
    """CALIB_ZERO_DISPARITY, original size; reject vertical-baseline solutions."""
    if not 0 <= alpha <= 1:
        raise ValueError("Rectification alpha must be within [0,1]")
    size: tuple[int, int] = calibration.image_size_wh
    r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(calibration.k1, calibration.d1, calibration.k2,
        calibration.d2, size, calibration.rotation, calibration.translation_m,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha, newImageSize=size)
    if abs(p2[0, 3]) < 1e-9 or abs(p2[1, 3]) > 1e-6:
        raise ValueError("This pipeline requires a horizontal stereo baseline")
    map1_x, map1_y = cv2.initUndistortRectifyMap(calibration.k1, calibration.d1, r1, p1, size, cv2.CV_32FC1)
    map2_x, map2_y = cv2.initUndistortRectifyMap(calibration.k2, calibration.d2, r2, p2, size, cv2.CV_32FC1)
    return StereoRectification(r1, r2, p1, p2, q, map1_x, map1_y, map2_x, map2_y, tuple(roi1), tuple(roi2), size)


def rectify_pair(pair: RawPair, rectification: StereoRectification) -> RectifiedPair:
    """Remap BGR uint8 H,W,3 without annotation, extra crop or resizing."""
    expected: tuple[int, int, int] = (rectification.image_size_wh[1], rectification.image_size_wh[0], 3)
    if pair.left_bgr.shape != expected or pair.right_bgr.shape != expected or pair.left_bgr.dtype != np.uint8 or pair.right_bgr.dtype != np.uint8:
        raise ValueError("Calibration and input image sizes/types do not match")
    return RectifiedPair(cv2.remap(pair.left_bgr, rectification.map1_x, rectification.map1_y, cv2.INTER_LINEAR),
        cv2.remap(pair.right_bgr, rectification.map2_x, rectification.map2_y, cv2.INTER_LINEAR))


def rectify_corner_pair(pair: CornerPair, calibration: StereoCalibration, rectification: StereoRectification) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return N,2 calibrated rectified pixel coordinates, independently of images."""
    left: NDArray[np.float64] = cv2.undistortPoints(pair.left_xy, calibration.k1, calibration.d1, R=rectification.r1, P=rectification.p1).reshape(-1, 2).astype(np.float64)
    right: NDArray[np.float64] = cv2.undistortPoints(pair.right_xy, calibration.k2, calibration.d2, R=rectification.r2, P=rectification.p2).reshape(-1, 2).astype(np.float64)
    return left, right
