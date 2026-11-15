"""Shared pixel grids and native-resolution evaluation for stereo comparisons."""

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .datasets import _read_calibration, load_middlebury_scene
from .model import DisparityResult, RectificationGeometry, RectifiedPair
from .stereo_training_data import load_training_sample, read_training_pfm


@dataclass(frozen=True)
class ComparisonScene:
    dataset: str
    name: str
    group: str
    pair: RectifiedPair  # uint8 native-size BGR (H,W,3)
    geometry: RectificationGeometry
    truth: np.ndarray  # float32 (H,W), native-resolution px
    reference: np.ndarray  # bool (H,W), method-independent GT validity
    scale: float
    reference_policy: str
    image_paths: tuple[Path, Path]


def mobile_group(name: str) -> str:
    """Group mobile views by physical scene, not numbered viewpoint."""
    if re.fullmatch(r"[a-z]+[123]", name) is None:
        raise ValueError(f"Unexpected mobile scene: {name}")
    return name[:-1]


def load_mobile(root: Path, name: str, scale: float, image_size_override: tuple[int, int] | None = None) -> ComparisonScene:
    """Load the 2021 default pair and disp0; this dataset has no ETH3D nocc mask."""
    folder: Path = root / name
    geometry, _, _, _ = _read_calibration(folder / "calib.txt")
    if image_size_override is not None:
        # Official ladder1/2 PNG and PFM headers agree on portrait dimensions,
        # while calib width/height are landscape. Preserve K/baseline/Q and image
        # orientation; explicitly correct only the declared image size and ROIs.
        if min(image_size_override) <= 0:
            raise ValueError("Invalid explicit image size correction")
        roi: tuple[int, int, int, int] = (0, 0, *image_size_override)
        geometry = RectificationGeometry(image_size_override, geometry.q.copy(), roi, roi)
    width, height = geometry.image_size
    images: list[np.ndarray] = []
    paths: tuple[Path, Path] = (folder / "im0.png", folder / "im1.png")
    for path in paths:
        image: np.ndarray | None = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.dtype != np.uint8 or image.shape != (height, width, 3):
            raise ValueError(f"Invalid mobile image: {path}")
        images.append(image)
    truth: np.ndarray = read_training_pfm(folder / "disp0.pfm", (height, width))
    reference: np.ndarray = np.isfinite(truth) & (truth > 0)
    return ComparisonScene("mobile2021_new", name, mobile_group(name), RectifiedPair(*images), geometry, truth,
        reference, scale, "all finite positive left GT; no provided non-occlusion mask", paths)


def load_eth(root: Path, name: str) -> ComparisonScene:
    sample = load_training_sample(root, name)
    geometry, _, _, _ = _read_calibration(root / name / "calib.txt")
    pair: RectifiedPair = RectifiedPair(sample.left_rgb[..., ::-1].copy(), sample.right_rgb[..., ::-1].copy())
    return ComparisonScene("eth3d_previous_holdout", name, sample.group, pair, geometry, sample.disparity_px,
        sample.non_occluded_mask & (sample.disparity_px > 0), 1., "finite positive non-occluded GT (mask=255)",
        (root / name / "im0.png", root / name / "im1.png"))


def load_previous(image_root: Path, gt_root: Path, name: str) -> ComparisonScene:
    scene = load_middlebury_scene(image_root, "trainingQ", name, ground_truth_root=gt_root)
    if scene.ground_truth is None:
        raise ValueError("Expected GT for previous development scene")
    group: str = {"MotorcycleE": "Motorcycle", "PianoL": "Piano", "PlaytableP": "Playtable"}.get(name, name)
    return ComparisonScene("middlebury_previous_development", name, group, scene.rectified_pair, scene.geometry,
        scene.ground_truth.disparity_px, scene.ground_truth.non_occluded_mask & np.isfinite(scene.ground_truth.disparity_px)
        & (scene.ground_truth.disparity_px > 0), 1., "finite positive non-occluded GT (mask=255)",
        (image_root / "trainingQ" / name / "im0.png", image_root / "trainingQ" / name / "im1.png"))


def resize_pair(pair: RectifiedPair, geometry: RectificationGeometry, scale: float) -> tuple[RectifiedPair, RectificationGeometry]:
    """Resize both views equally and transform Q with OpenCV pixel-center mapping."""
    if scale <= 0 or scale > 1:
        raise ValueError("Expected 0<scale<=1")
    width, height = geometry.image_size
    target: tuple[int, int] = (round(width * scale), round(height * scale))
    if not np.allclose([target[0] / width, target[1] / height], scale, atol=1e-12, rtol=0):
        raise ValueError("Require exact isotropic integer-sized scaling")
    if scale == 1:
        return pair, geometry
    images: list[np.ndarray] = [cv2.resize(image, target, interpolation=cv2.INTER_AREA) for image in (pair.left_bgr, pair.right_bgr)]
    # Explicit four-column affine composition avoids initializing a second BLAS
    # OpenMP runtime beside PyTorch in this Windows Conda process.
    scaled_q: np.ndarray = geometry.q.copy()
    scaled_q[:, :3] /= scale
    offset: float = (1 / scale - 1) / 2
    scaled_q[:, 3] += offset * (geometry.q[:, 0] + geometry.q[:, 1])
    roi: tuple[int, int, int, int] = (0, 0, *target)
    scaled_geometry: RectificationGeometry = RectificationGeometry(target, scaled_q, roi, roi)
    return RectifiedPair(*images), scaled_geometry


def restore_native(values: np.ndarray, valid: np.ndarray, native_shape: tuple[int, int], scale: float) -> DisparityResult:
    """Upsample px disparity /scale; require all interpolation support valid.

    No averaging with invalid sentinels, no hole filling. Returned float32 (H,W)
    values are in native GT px; invalid output is NaN with a bool validity mask.
    """
    if values.shape != valid.shape or valid.dtype != np.bool_ or not np.isfinite(values[valid]).all():
        raise ValueError("Invalid disparity/mask input")
    if scale == 1:
        restored: np.ndarray = values.copy()
        mask: np.ndarray = valid.copy()
    else:
        height, width = native_shape
        if values.shape != (round(height * scale), round(width * scale)):
            raise ValueError("Disparity grid does not match image scale")
        support: np.ndarray = cv2.resize(valid.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        numerator: np.ndarray = cv2.resize(np.where(valid, values, 0).astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        mask = support >= 1 - 1e-6
        restored = np.divide(numerator, np.maximum(support, 1e-12)) / scale
    restored[~mask] = np.nan
    return DisparityResult(restored.astype(np.float32), mask, (0, 0, native_shape[1], native_shape[0]))


def choose_random_views(names: list[str], seed: int, count: int) -> list[str]:
    """Choose distinct physical scenes, then one viewpoint per scene, before metrics."""
    groups: list[str] = sorted({mobile_group(name) for name in names})
    if count < 5 or count > len(groups):
        raise ValueError("Need at least five distinct physical scenes")
    rng: np.random.Generator = np.random.default_rng(seed)
    selected: list[str] = sorted(str(group) for group in rng.choice(groups, count, replace=False))
    return [str(rng.choice(sorted(name for name in names if mobile_group(name) == group))) for group in selected]
