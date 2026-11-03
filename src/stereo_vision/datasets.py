"""Adapt Middlebury v3 files into dataset-independent stereo contracts."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from stereo_vision.models import (
    FloatMap, GroundTruth, Image, Matrix, RectifiedGeometry, RectifiedPair, StereoScene,
)


def read_image(path: Path) -> Image:
    """Read uint8 BGR (H,W,3); raise for missing or undecodable images."""
    if not path.is_file():
        raise FileNotFoundError(path)
    image: Image | None = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def read_pfm(path: Path) -> FloatMap:
    """Read Middlebury grayscale PFM samples as float32 (H,W), pixels.

    Row order is bottom-up; the scale sign specifies byte order. The stored
    Middlebury disparity samples already use pixels: do not rescale their values.
    """
    try:
        with path.open("rb") as stream:
            if stream.readline().strip() != b"Pf":
                raise ValueError("expected grayscale Pf header")
            width, height = map(int, stream.readline().split())
            scale: float = float(stream.readline())
            if width <= 0 or height <= 0 or not np.isfinite(scale) or scale == 0:
                raise ValueError("invalid dimensions or byte-order scale")
            data: np.ndarray = np.frombuffer(stream.read(), dtype="<f4" if scale < 0 else ">f4")
        if data.size != width * height:
            raise ValueError("payload size does not match header")
    except ValueError as exc:
        raise ValueError(f"Invalid PFM {path}: {exc}") from exc
    return np.flipud(data.reshape(height, width)).astype(np.float32, copy=True)


def read_calibration(path: Path) -> tuple[RectifiedGeometry, tuple[int, int], int]:
    """Return Q in metres, image (width,height), and dataset disparity bound.

    Both rectified intrinsics are validated; doffs=cx_right-cx_left and the
    source baseline is converted from mm to m. Raises ValueError for bad data.
    """
    fields: dict[str, str] = dict(
        line.strip().split("=", 1)
        for line in path.read_text(encoding="utf-8").splitlines() if "=" in line
    )
    try:
        cam_left: Matrix = np.array([
            [float(v) for v in row.split()]
            for row in fields["cam0"].strip("[]").split(";")
        ], dtype=np.float64)
        cam_right: Matrix = np.array([
            [float(v) for v in row.split()]
            for row in fields["cam1"].strip("[]").split(";")
        ], dtype=np.float64)
        width: int = int(fields["width"])
        height: int = int(fields["height"])
        disparity_levels: int = int(fields["ndisp"])
        baseline_m: float = float(fields["baseline"]) / 1000.0
        doffs_px: float = float(fields["doffs"])
        if cam_left.shape != (3, 3) or cam_right.shape != (3, 3):
            raise ValueError("camera matrices must be 3x3")
        fx_px: float = float(cam_left[0, 0])
        cx_px: float = float(cam_left[0, 2])
        cy_px: float = float(cam_left[1, 2])
        expected_left: Matrix = np.array([[fx_px, 0, cx_px], [0, fx_px, cy_px], [0, 0, 1]])
        expected_right: Matrix = expected_left.copy()
        expected_right[0, 2] += doffs_px
        if not (
            np.allclose(cam_left, expected_left, atol=0.01)
            and np.allclose(cam_right, expected_right, atol=0.01)
        ):
            raise ValueError("unsupported rectified intrinsics or inconsistent doffs")
        if not np.isfinite([fx_px, baseline_m, doffs_px]).all() or min(
            fx_px, baseline_m, width, height, disparity_levels
        ) <= 0:
            raise ValueError("geometry and dimensions must be finite and positive")
    except (KeyError, ValueError, IndexError) as exc:
        raise ValueError(f"Invalid calibration {path}: {exc}") from exc
    q: Matrix = np.array([
        [1, 0, 0, -cx_px], [0, 1, 0, -cy_px],
        [0, 0, 0, fx_px], [0, 0, 1 / baseline_m, doffs_px / baseline_m],
    ], dtype=np.float64)
    geometry: RectifiedGeometry = RectifiedGeometry(q, (0, 0, width, height), (0, 0, width, height))
    return geometry, (width, height), disparity_levels


def load_middlebury_scene(root: Path, scene: str, split: str) -> StereoScene:
    """Load matching rectified images, geometry and optional training truth.

    Missing files raise FileNotFoundError; malformed or inconsistent data raise
    ValueError with the scene/path. testQ has no public GT and is not evaluated.
    """
    if split not in {"trainingQ", "testQ"} or not scene or Path(scene).name != scene or scene in {".", ".."}:
        raise ValueError(f"Invalid Middlebury scene: {split}/{scene}")
    scene_dir: Path = root / "MiddEval3-Q" / split / scene
    geometry, image_size, disparity_levels = read_calibration(scene_dir / "calib.txt")
    pair: RectifiedPair = RectifiedPair(read_image(scene_dir / "im0.png"), read_image(scene_dir / "im1.png"))
    width, height = image_size
    if pair.left_bgr.shape != (height, width, 3) or pair.right_bgr.shape != pair.left_bgr.shape:
        raise ValueError(f"Image sizes disagree with calibration: {scene_dir}")
    ground_truth: GroundTruth | None = None
    if split == "trainingQ":
        truth_dir: Path = root / "MiddEval3-GT0" / split / scene
        disparity_px: FloatMap = read_pfm(truth_dir / "disp0GT.pfm")
        mask_image: Image = read_image(truth_dir / "mask0nocc.png")
        if disparity_px.shape != (height, width) or mask_image.shape[:2] != (height, width):
            raise ValueError(f"Truth/mask sizes disagree with images: {truth_dir}")
        # Middlebury: 255 nonoccluded, 128 occluded, 0 no ground truth.
        ground_truth = GroundTruth(disparity_px, mask_image[:, :, 0] == 255)
    return StereoScene(scene, split, pair, geometry, disparity_levels, ground_truth, scene_dir.resolve())
