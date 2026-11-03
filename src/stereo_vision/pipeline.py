"""Compose pure stages; ground truth never enters matching or reconstruction."""
from __future__ import annotations

from time import perf_counter

from stereo_vision.disparity import compute_disparity
from stereo_vision.evaluation import evaluate_disparity
from stereo_vision.models import (
    DisparityMetrics,
    DisparityResult,
    PipelineResult,
    ReconstructionResult,
    SgbmConfig,
    StereoScene,
)
from stereo_vision.reprojection import reconstruct


def reconstruct_scene(scene: StereoScene, config: SgbmConfig) -> PipelineResult:
    """Run an already rectified scene; time stages in ms excluding file I/O.

    Reuses data interfaces in CLI and Notebook. Raw-camera calibration and
    rectification are separate future stages, not implicit operations here.
    """
    start: float = perf_counter()
    disparity: DisparityResult = compute_disparity(scene.pair, scene.geometry, config, scene.disparity_levels)
    matched: float = perf_counter()
    reconstruction: ReconstructionResult = reconstruct(disparity, scene.geometry)
    reprojected: float = perf_counter()
    metrics: DisparityMetrics | None = None
    if scene.ground_truth is not None:
        metrics = evaluate_disparity(disparity, scene.ground_truth)
    return PipelineResult(disparity, reconstruction, metrics, {
        "sgbm_ms": (matched - start) * 1000,
        "reprojection_ms": (reprojected - matched) * 1000,
        "reconstruction_total_ms": (reprojected - start) * 1000,
    })
