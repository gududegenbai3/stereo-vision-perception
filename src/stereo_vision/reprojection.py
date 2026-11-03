"""Reproject dataset-independent Q geometry into the rectified left camera."""
from __future__ import annotations

import cv2
import numpy as np

from stereo_vision.models import DisparityResult, FloatMap, Mask, ReconstructionResult, RectifiedGeometry


def reconstruct(disparity: DisparityResult, geometry: RectifiedGeometry) -> ReconstructionResult:
    """Return float32 XYZ (H,W,3), axial Z (H,W) in m and a bool mask.

    X right, Y down, Z forward. All invalid coordinates/depth are NaN.
    Raises ValueError for invalid Q, inconsistent masks, or no finite positive Z.
    """
    if geometry.q.shape != (4, 4) or not np.isfinite(geometry.q).all():
        raise ValueError("Q must be a finite 4x4 reprojection matrix")
    if disparity.valid_mask.shape != disparity.disparity_px.shape:
        raise ValueError("Disparity and validity mask dimensions differ")
    xyz_m: FloatMap = cv2.reprojectImageTo3D(disparity.disparity_px, geometry.q)
    valid_mask: Mask = disparity.valid_mask & np.isfinite(xyz_m).all(axis=2) & (xyz_m[:, :, 2] > 0)
    if not valid_mask.any():
        raise ValueError("Reprojection produced no finite positive-depth points")
    xyz_m[~valid_mask] = np.nan
    return ReconstructionResult(xyz_m[:, :, 2].copy(), xyz_m, valid_mask)
