"""SGBM returns matching validity and the fixed geometric search domain."""
from __future__ import annotations

import cv2
import numpy as np

from stereo_vision.models import DisparityResult, FloatMap, Mask, RectifiedGeometry, RectifiedPair, Roi, SgbmConfig

DISPARITY_SCALE: float = 16.0


def compute_disparity(
    pair: RectifiedPair,
    geometry: RectifiedGeometry,
    config: SgbmConfig,
    disparity_levels: int,
) -> DisparityResult:
    """Match uint8 BGR (H,W,3) images; return float32 px and bool masks.

    Uses minDisparity=0 and rounds the dataset bound up to 16. Empty geometric
    ROI or invalid image shape/search range raises ValueError.
    """
    if pair.left_bgr.shape != pair.right_bgr.shape or pair.left_bgr.ndim != 3 or pair.left_bgr.shape[2] != 3:
        raise ValueError("Left/right images must have matching (H,W,3) BGR shapes")
    if pair.left_bgr.dtype != np.uint8 or pair.right_bgr.dtype != np.uint8:
        raise ValueError("SGBM requires uint8 images")
    height, width = pair.left_bgr.shape[:2]
    num_disparities: int = ((disparity_levels + 15) // 16) * 16
    if disparity_levels <= 0 or num_disparities >= width:
        raise ValueError(f"Search range {num_disparities} does not fit image width {width}")
    roi: Roi = cv2.getValidDisparityROI(geometry.roi_left, geometry.roi_right, 0, num_disparities, config.block_size)
    x, y, roi_width, roi_height = roi
    if roi_width <= 0 or roi_height <= 0:
        raise ValueError("Empty geometric search ROI; check image size and matching parameters")
    search_mask: Mask = np.zeros((height, width), dtype=np.bool_)
    search_mask[y:y + roi_height, x:x + roi_width] = True
    matcher: cv2.StereoSGBM = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=num_disparities, blockSize=config.block_size,
        P1=config.p1_scale * config.block_size**2,
        P2=config.p2_scale * config.block_size**2,
        uniquenessRatio=config.uniqueness_ratio,
        speckleWindowSize=config.speckle_window_size, speckleRange=config.speckle_range,
        disp12MaxDiff=config.disp12_max_diff, mode=cv2.STEREO_SGBM_MODE_SGBM,
    )
    raw: np.ndarray = matcher.compute(
        cv2.cvtColor(pair.left_bgr, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(pair.right_bgr, cv2.COLOR_BGR2GRAY),
    )
    disparity_px: FloatMap = raw.astype(np.float32) / DISPARITY_SCALE
    valid_mask: Mask = search_mask & (raw > -DISPARITY_SCALE) & (disparity_px > 0)
    disparity_px[~valid_mask] = np.nan
    return DisparityResult(disparity_px, valid_mask, search_mask, num_disparities)
