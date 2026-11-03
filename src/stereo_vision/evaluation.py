"""Evaluate prediction reliability without replacing missing predictions by zero."""
from __future__ import annotations

import numpy as np

from stereo_vision.models import DisparityMetrics, DisparityResult, FloatMap, GroundTruth, Mask


def evaluate_disparity(prediction: DisparityResult, truth: GroundTruth) -> DisparityMetrics:
    """Measure px errors on finite predictions inside nonoccluded GT/search ROI.

    Coverage uses this fixed reference domain as denominator. Empty domains or
    predictions yield null errors, preserving counts. These are local Q-resolution
    diagnostics, not the official full-resolution Middlebury leaderboard metrics.
    """
    if (
        prediction.disparity_px.shape != truth.disparity_px.shape
        or truth.nonoccluded_mask.shape != truth.disparity_px.shape
    ):
        raise ValueError("Prediction, ground truth and mask dimensions differ")
    reference: Mask = (
        prediction.search_mask
        & truth.nonoccluded_mask
        & np.isfinite(truth.disparity_px)
        & (truth.disparity_px > 0)
    )
    evaluated: Mask = reference & prediction.valid_mask & np.isfinite(prediction.disparity_px)
    reference_count: int = int(reference.sum())
    evaluated_count: int = int(evaluated.sum())
    errors_px: FloatMap = np.abs(prediction.disparity_px[evaluated] - truth.disparity_px[evaluated])
    return DisparityMetrics(
        reference_count, evaluated_count,
        evaluated_count / reference_count if reference_count else None,
        float(errors_px.mean()) if evaluated_count else None,
        float(np.percentile(errors_px, 95)) if evaluated_count else None,
        float(np.mean(errors_px > 2)) if evaluated_count else None,
    )
