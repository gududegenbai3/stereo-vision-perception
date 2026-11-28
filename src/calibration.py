"""Typed pinhole stereo calibration; input points in pixels, target dimensions in m."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BoardGeometry:
    cols: int
    rows: int
    square_size_m: float


@dataclass(frozen=True)
class CornerPair:
    """Matching identities: float32 (cols*rows,1,2) XY pixels per raw camera."""
    pair_id: str
    left_xy: NDArray[np.float32]
    right_xy: NDArray[np.float32]


@dataclass(frozen=True)
class StereoCalibration:
    """K in pixels; D dimensionless; X_right = rotation @ X_left + translation_m."""
    k1: NDArray[np.float64]
    d1: NDArray[np.float64]
    k2: NDArray[np.float64]
    d2: NDArray[np.float64]
    rotation: NDArray[np.float64]
    translation_m: NDArray[np.float64]
    essential: NDArray[np.float64]
    fundamental: NDArray[np.float64]
    rms_left_px: float
    rms_right_px: float
    rms_stereo_px: float
    image_size_wh: tuple[int, int]
    board: BoardGeometry


def object_points(board: BoardGeometry) -> NDArray[np.float32]:
    """Return N,3 XYZ metre target points, row-major with columns varying fastest."""
    if type(board.cols) is not int or type(board.rows) is not int or min(board.cols, board.rows) < 2 or not np.isfinite(board.square_size_m) or board.square_size_m <= 0:
        raise ValueError("Require positive measured square size and >=2x2 inner corners")
    values: NDArray[np.float32] = np.zeros((board.cols * board.rows, 3), np.float32)
    values[:, :2] = np.mgrid[:board.cols, :board.rows].T.reshape(-1, 2) * board.square_size_m
    return values


def canonical_corner_order(points: NDArray[np.float32], board: BoardGeometry) -> NDArray[np.float32]:
    """Resolve the detector's 180-degree ambiguity for upright targets (<90deg roll).

    Never reorder arbitrary individual points. Reject mirrored/unsupported ordering
    rather than guessing a different identity for one camera.
    """
    if points.shape != (board.cols * board.rows, 1, 2) or not np.isfinite(points).all():
        raise ValueError("Invalid full corner array")
    result: NDArray[np.float32] = np.asarray(points, np.float32).copy()
    grid: NDArray[np.float32] = result.reshape(board.rows, board.cols, 2)
    row_direction: float = float(np.mean(np.diff(grid, axis=0)[:, :, 1]))
    if row_direction < 0:
        result = result[::-1].copy()
        grid = result.reshape(board.rows, board.cols, 2)
    if float(np.mean(np.diff(grid, axis=0)[:, :, 1])) <= 0 or float(np.mean(np.diff(grid, axis=1)[:, :, 0])) <= 0:
        raise ValueError("Target ordering is mirrored or roll exceeds the supported upright convention")
    return result


def calibrate_stereo(
    pairs: Sequence[CornerPair], board: BoardGeometry, image_size_wh: tuple[int, int],
    maximum_iterations: int = 100, epsilon: float = 1e-7,
) -> StereoCalibration:
    """Fit mono intrinsics then FIX_INTRINSIC stereo using only supplied fit pairs.

    Validation pairs must be kept outside this function. No configuration/image
    files are read. OpenCV/value errors propagate, no automatic outlier deletion.
    """
    if len(pairs) < 3 or min(image_size_wh) <= 0 or maximum_iterations < 1 or epsilon <= 0:
        raise ValueError("Insufficient fit views or invalid dimensions/termination")
    objects: list[NDArray[np.float32]] = [object_points(board) for _ in pairs]
    for pair in pairs:
        for points in (pair.left_xy, pair.right_xy):
            if points.shape != (board.cols * board.rows, 1, 2) or not np.isfinite(points).all():
                raise ValueError("Incomplete/nonfinite image points")
    left: list[NDArray[np.float32]] = [pair.left_xy for pair in pairs]
    right: list[NDArray[np.float32]] = [pair.right_xy for pair in pairs]
    criteria: tuple[int, int, float] = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, maximum_iterations, epsilon)
    rms1, k1, d1, _, _ = cv2.calibrateCamera(objects, left, image_size_wh, None, None, flags=0, criteria=criteria)
    rms2, k2, d2, _, _ = cv2.calibrateCamera(objects, right, image_size_wh, None, None, flags=0, criteria=criteria)
    rms_stereo, k1, d1, k2, d2, rotation, translation, essential, fundamental = cv2.stereoCalibrate(
        objects, left, right, k1, d1, k2, d2, image_size_wh, flags=cv2.CALIB_FIX_INTRINSIC, criteria=criteria)
    for values in (k1, d1, k2, d2, rotation, translation):
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite calibration solution")
    if min(k1[0, 0], k1[1, 1], k2[0, 0], k2[1, 1]) <= 0 or np.linalg.norm(translation) <= 0 or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("Invalid focal length, baseline or rotation")
    return StereoCalibration(k1, d1, k2, d2, rotation, translation, essential, fundamental, float(rms1), float(rms2), float(rms_stereo), image_size_wh, board)
