"""Typed ETH3D samples and scene-disjoint training crops, no model dependencies."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class StereoTrainingSample:
    name: str
    group: str
    left_rgb: np.ndarray  # uint8 (H,W,3), replicated gray for ETH3D
    right_rgb: np.ndarray  # uint8 (H,W,3), rectified
    disparity_px: np.ndarray  # float32 (H,W), left-reference px, +inf unknown
    known_mask: np.ndarray  # bool (H,W), includes known occluded GT
    non_occluded_mask: np.ndarray  # bool (H,W), mask value 255 only


def scene_group(name: str) -> str:
    """Keep every viewpoint and small/large-baseline variant of one site together."""
    match: re.Match | None = re.fullmatch(r"(.+)_\d+[ls]", name)
    if match is None:
        raise ValueError(f"Unrecognized ETH3D sample name: {name}")
    return match.group(1)


def freeze_split(names: list[str], groups: dict[str, list[str]]) -> dict[str, list[str]]:
    """Require a complete, exclusive physical-scene partition; reject leakage."""
    if set(groups) != {"train", "validation", "test"}:
        raise ValueError("Require train/validation/test groups")
    flattened: list[str] = [group for values in groups.values() for group in values]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Physical scene group leakage across splits")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate samples")
    unknown: set[str] = {scene_group(name) for name in names} - set(flattened)
    if unknown:
        raise ValueError(f"Unassigned physical scenes: {unknown}")
    splits: dict[str, list[str]] = {split: sorted(name for name in names if scene_group(name) in sites)
                                    for split, sites in groups.items()}
    if any(not values for values in splits.values()):
        raise ValueError("Every split must contain samples")
    return splits


def read_training_pfm(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Read grayscale PFM float32 px, bottom-up, +inf invalid; no value rescaling."""
    with path.open("rb") as stream:
        if stream.readline().strip() != b"Pf":
            raise ValueError(f"Expected grayscale PFM: {path}")
        width, height = map(int, stream.readline().split())
        scale: float = float(stream.readline())
        if (height, width) != shape or not np.isfinite(scale) or scale == 0:
            raise ValueError(f"Invalid PFM shape or byte order: {path}")
        payload: bytes = stream.read()
    if len(payload) != height * width * 4:
        raise ValueError(f"Invalid PFM payload length: {path}")
    values: np.ndarray = np.frombuffer(payload, dtype="<f4" if scale < 0 else ">f4").reshape(shape)
    result: np.ndarray = values[::-1].astype(np.float32).copy()
    if np.isnan(result).any() or np.isneginf(result).any():
        raise ValueError(f"Unsupported PFM invalid values: {path}")
    return result


def load_training_sample(root: Path, name: str) -> StereoTrainingSample:
    """Load official ETH3D pair and masks; missing files/format errors are explicit."""
    group: str = scene_group(name)
    folder: Path = root / name
    images: list[np.ndarray] = []
    for filename in ("im0.png", "im1.png"):
        path: Path = folder / filename
        image: np.ndarray | None = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.dtype != np.uint8:
            raise ValueError(f"Cannot read uint8 image: {path}")
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if images[0].shape != images[1].shape:
        raise ValueError(f"Unpaired image sizes: {name}")
    shape: tuple[int, int] = images[0].shape[:2]
    disparity: np.ndarray = read_training_pfm(folder / "disp0GT.pfm", shape)
    mask: np.ndarray | None = cv2.imdecode(np.frombuffer((folder / "mask0nocc.png").read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != shape or not np.isin(mask, [0, 128, 255]).all():
        raise ValueError(f"Invalid official occlusion mask: {name}")
    known: np.ndarray = (mask > 0) & np.isfinite(disparity)
    if not np.array_equal(mask > 0, np.isfinite(disparity)):
        raise ValueError(f"Mask/PFM validity mismatch: {name}")
    return StereoTrainingSample(name, group, images[0], images[1], disparity, known, known & (mask == 255))


def training_crop(sample: StereoTrainingSample, crop_hw: tuple[int, int], max_disparity: int,
                  rng: np.random.Generator, brightness: tuple[float, float], gamma: tuple[float, float]) -> tuple[np.ndarray, ...]:
    """Shared crop and photometric transform; disparities retain their px units.

    Returns RGB float32 (3,h,w) twice, float32 GT (1,h,w), bool validity (1,h,w).
    Crop validity excludes correspondences outside the cropped right image.
    """
    crop_height, crop_width = crop_hw
    height, width = sample.disparity_px.shape
    if crop_height > height or crop_width > width or min(crop_hw) <= 0:
        raise ValueError(f"Crop exceeds sample {sample.name}")
    y: int = int(rng.integers(height - crop_height + 1))
    x: int = int(rng.integers(width - crop_width + 1))
    disparity: np.ndarray = sample.disparity_px[y:y + crop_height, x:x + crop_width].copy()
    valid: np.ndarray = sample.known_mask[y:y + crop_height, x:x + crop_width].copy()
    valid &= (disparity > 0) & (disparity < max_disparity) & (np.arange(crop_width)[None, :] - disparity >= 0)
    if valid.sum() < 100:
        raise ValueError(f"Too little crop supervision: {sample.name}")
    gain: float = float(rng.uniform(*brightness))
    exponent: float = float(rng.uniform(*gamma))
    transformed: list[np.ndarray] = []
    for image in (sample.left_rgb, sample.right_rgb):
        crop: np.ndarray = image[y:y + crop_height, x:x + crop_width].astype(np.float32) / 255
        changed: np.ndarray = np.clip(gain * crop**exponent, 0, 1) * 255
        transformed.append(np.ascontiguousarray(changed.transpose(2, 0, 1)))
    target: np.ndarray = np.where(valid, disparity, 0).astype(np.float32)[None]
    return transformed[0], transformed[1], target, valid[None]


def data_fingerprint(root: Path, names: list[str]) -> dict[str, str]:
    """Hash original pair/GT/mask files, independent of augmentation or split."""
    return {f"{name}/{filename}": hashlib.sha256((root / name / filename).read_bytes()).hexdigest()
            for name in names for filename in ("im0.png", "im1.png", "disp0GT.pfm", "mask0nocc.png")}
