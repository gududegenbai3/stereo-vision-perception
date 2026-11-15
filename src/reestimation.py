"""Our candidate-selection strategy; not a method claimed by either paper."""

import numpy as np

from .model import ConfidenceResult, DisparityMap, DisparityResult, ReestimationResult, ValidMask


def select_candidate(
    baseline: DisparityResult,
    candidate: DisparityResult,
    baseline_confidence: ConfidenceResult,
    candidate_confidence: ConfidenceResult,
    *,
    threshold: float = 0.5,
    margin: float = 0.15,
) -> ReestimationResult:
    """Replace only valid low-confidence baseline pixels with better candidates.

    All arrays have the same left-reference grid. Require known candidate score
    >=threshold and >=baseline+margin. Unknown or high-confidence baseline remains
    unchanged. No filling or deletion: output coverage equals baseline coverage.
    Input scores must be float32 in [0,1] where known, NaN otherwise.
    """
    if not 0 <= threshold <= 1 or not 0 <= margin <= 1:
        raise ValueError("Invalid confidence threshold/margin")
    shape: tuple[int, ...] = baseline.disparity_px.shape
    for disparity in (baseline, candidate):
        if disparity.disparity_px.shape != shape or disparity.disparity_px.dtype != np.float32:
            raise ValueError("Disparity shapes/types differ")
        if disparity.valid_mask.shape != shape or disparity.valid_mask.dtype != np.bool_:
            raise ValueError("Invalid disparity mask")
        if not np.isfinite(disparity.disparity_px[disparity.valid_mask]).all():
            raise ValueError("Nonfinite disparity marked valid")
    for confidence in (baseline_confidence, candidate_confidence):
        if confidence.score.shape != shape or confidence.known_mask.shape != shape or confidence.known_mask.dtype != np.bool_:
            raise ValueError("Confidence shapes/masks differ")
        known_scores: DisparityMap = confidence.score[confidence.known_mask]
        if not np.isfinite(known_scores).all() or np.any((known_scores < 0) | (known_scores > 1)):
            raise ValueError("Known confidence scores must be finite in [0,1]")
        if np.any(np.isfinite(confidence.score[~confidence.known_mask])):
            raise ValueError("Unknown confidence must be NaN")
    replaced: ValidMask = (
        baseline.valid_mask & candidate.valid_mask
        & baseline_confidence.known_mask & candidate_confidence.known_mask
        & (baseline_confidence.score < threshold)
        & (candidate_confidence.score >= threshold)
        & (candidate_confidence.score >= baseline_confidence.score + margin)
    )
    final_px: DisparityMap = baseline.disparity_px.copy()
    final_px[replaced] = candidate.disparity_px[replaced]
    return ReestimationResult(
        DisparityResult(final_px, baseline.valid_mask.copy(), baseline.valid_roi), replaced,
    )
