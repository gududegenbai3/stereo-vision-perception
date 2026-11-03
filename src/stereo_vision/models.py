"""Shared contracts: arrays retain their original image dimensions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]
FloatMap = NDArray[np.float32]
Mask = NDArray[np.bool_]
Matrix = NDArray[np.float64]
Roi = tuple[int, int, int, int]


@dataclass(frozen=True)
class SgbmConfig:
    """Search range is supplied by the dataset; remaining settings are explicit."""

    block_size: int
    p1_scale: int
    p2_scale: int
    uniqueness_ratio: int
    speckle_window_size: int
    speckle_range: int
    disp12_max_diff: int


@dataclass(frozen=True)
class RunConfig:
    data_root: Path
    output_root: Path
    scene: str
    split: str
    threads: int
    sgbm: SgbmConfig


@dataclass(frozen=True)
class RectifiedPair:
    """uint8 BGR left/right images, both (H, W, 3)."""

    left_bgr: Image
    right_bgr: Image


@dataclass(frozen=True)
class RectifiedGeometry:
    """Q maps (u,v,d,1) to the rectified left camera, in metres.

    X points right, Y down and Z forward. ROIs use (x,y,width,height).
    """

    q: Matrix
    roi_left: Roi
    roi_right: Roi


@dataclass(frozen=True)
class GroundTruth:
    """Left-view disparity float32 (H,W), px; bool nonoccluded mask (H,W)."""

    disparity_px: FloatMap
    nonoccluded_mask: Mask


@dataclass(frozen=True)
class StereoScene:
    name: str
    split: str
    pair: RectifiedPair
    geometry: RectifiedGeometry
    disparity_levels: int
    ground_truth: GroundTruth | None
    source_dir: Path


@dataclass(frozen=True)
class DisparityResult:
    """float32 (H,W) px, NaN outside matching/search validity; bool masks."""

    disparity_px: FloatMap
    valid_mask: Mask
    search_mask: Mask
    num_disparities: int


@dataclass(frozen=True)
class ReconstructionResult:
    """float32 depth (H,W) and XYZ (H,W,3), metres; invalid values are NaN."""

    depth_m: FloatMap
    xyz_m: FloatMap
    valid_mask: Mask


@dataclass(frozen=True)
class DisparityMetrics:
    reference_pixels: int
    evaluated_pixels: int
    coverage: float | None
    epe_px: float | None
    p95_px: float | None
    bad2_fraction: float | None


@dataclass(frozen=True)
class PipelineResult:
    disparity: DisparityResult
    reconstruction: ReconstructionResult
    metrics: DisparityMetrics | None
    stage_ms: dict[str, float]
