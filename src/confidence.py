"""Pure confidence measures: signed left-right consistency and DPS stability.

Poggi et al. 2021, section 3.3 (LRC); Lee et al. 2024 equations 1-5 (DPS).
Computing disparities remains the caller's job. Ground truth is never an input.
"""

from collections.abc import Mapping

import numpy as np

from .model import ConfidenceResult, DisparityMap, DisparityResult, ValidMask


def validate_disparity(result: DisparityResult) -> None:
    """Reject inconsistent float32 px arrays and bool masks, including valid NaN."""
    values: DisparityMap = result.disparity_px
    if values.ndim != 2 or values.dtype != np.float32:
        raise ValueError("Disparity must be float32 (H,W)")
    if result.valid_mask.shape != values.shape or result.valid_mask.dtype != np.bool_:
        raise ValueError("Invalid disparity mask")
    if not np.isfinite(values[result.valid_mask]).all():
        raise ValueError("Nonfinite disparity marked valid")
    height: int
    width: int
    height, width = values.shape
    x, y, roi_width, roi_height = result.valid_roi
    if not (0 <= x < x + roi_width <= width and 0 <= y < y + roi_height <= height):
        raise ValueError("Invalid disparity search ROI")
    domain: ValidMask = np.zeros(values.shape, bool)
    domain[y:y + roi_height, x:x + roi_width] = True
    if np.any(result.valid_mask & ~domain):
        raise ValueError("Valid disparity outside search ROI")


def _confidence(residual: DisparityMap, known: ValidMask) -> ConfidenceResult:
    residual_px: DisparityMap = np.where(known, residual, np.nan).astype(np.float32)
    score: DisparityMap = np.exp2(-residual_px).astype(np.float32)
    return ConfidenceResult(score, residual_px, known.copy())


def left_right_confidence(left: DisparityResult, right: DisparityResult) -> ConfidenceResult:
    """Return abs(dL(x)+dR(x-dL)) residual, px, and 2**(-residual).

    dR is signed x_right-x_left, normally negative. Bilinear sampling on the same
    row requires finite valid neighbours; exact integer coordinates need one.
    Out-of-image, missing correspondence or empty support stays unknown (NaN).
    """
    validate_disparity(left)
    validate_disparity(right)
    if left.disparity_px.shape != right.disparity_px.shape:
        raise ValueError("Left/right disparity shapes differ")
    height, width = left.disparity_px.shape
    rows, columns = np.indices((height, width))
    coordinates: DisparityMap = columns.astype(np.float32) - left.disparity_px
    inside: ValidMask = left.valid_mask & np.isfinite(coordinates) & (coordinates >= 0) & (coordinates <= width - 1)
    safe: DisparityMap = np.where(inside, coordinates, 0)
    x0: np.ndarray = np.floor(safe).astype(np.int32)
    x1: np.ndarray = np.minimum(x0 + 1, width - 1)
    fraction: DisparityMap = safe - x0
    integer: ValidMask = fraction <= 1e-6
    known: ValidMask = inside & right.valid_mask[rows, x0] & (integer | right.valid_mask[rows, x1])
    first: DisparityMap = right.disparity_px[rows, x0]
    second: DisparityMap = right.disparity_px[rows, x1]
    sampled: DisparityMap = np.where(integer, first, first + fraction * (second - first))
    residual: DisparityMap = np.abs(left.disparity_px + sampled).astype(np.float32)
    known &= np.isfinite(residual)
    return _confidence(residual, known)


def dps_confidence(anchor: DisparityResult, shifted: Mapping[int, DisparityResult]) -> ConfidenceResult:
    """Compare d_k with d_0+k, with positive k moving right-image content left.

    Input predictions must already exclude padding and use shifted search ranges.
    Evaluate on the intersection of anchor and every nonzero shift's valid mask.
    At least one positive and one negative shift are required. Mean absolute
    residual excludes zero shift (Lee eq.4); 2**(-U) follows eq.5 normalization.
    """
    validate_disparity(anchor)
    if any(type(shift) is not int or shift == 0 for shift in shifted):
        raise ValueError("DPS shifts must be nonzero integers")
    if not shifted or not any(k > 0 for k in shifted) or not any(k < 0 for k in shifted):
        raise ValueError("DPS requires both positive and negative shifts")
    known: ValidMask = anchor.valid_mask.copy()
    residual: DisparityMap = np.zeros_like(anchor.disparity_px)
    for shift, result in shifted.items():
        if type(shift) is not int or shift == 0:
            raise ValueError("DPS shifts must be nonzero integers")
        validate_disparity(result)
        if result.disparity_px.shape != anchor.disparity_px.shape:
            raise ValueError("DPS disparity shapes differ")
        known &= result.valid_mask
        residual += np.abs(result.disparity_px - (anchor.disparity_px + shift))
    residual /= len(shifted)
    known &= np.isfinite(residual)
    return _confidence(residual, known)


def combine_confidence(first: ConfidenceResult, second: ConfidenceResult) -> ConfidenceResult:
    """Conservative joint score=min(scores), known only if both methods are known."""
    if first.score.shape != second.score.shape:
        raise ValueError("Confidence shapes differ")
    known: ValidMask = first.known_mask & second.known_mask
    residual: DisparityMap = np.maximum(first.residual_px, second.residual_px)
    return _confidence(residual, known)
