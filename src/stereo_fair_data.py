"""Acquire the frozen public stereo pairs and normalize RGB, disparity and masks."""

import argparse
import hashlib
from io import BytesIO
from functools import lru_cache
import json
import shutil
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np

from .camera_io import digest, write_json
from .config import load_offline_config
from .datasets import load_middlebury_scene
from .provenance import download_checked
from .stereo_fair_archives import booster_index, download_member


@lru_cache(maxsize=24)
def load_arrays(path: str) -> dict[str, np.ndarray]:
    """Read normalized RGB uint8 HW3, float32 HW pixel GT and bool HW masks."""
    with np.load(path, allow_pickle=False) as archive:
        arrays: dict[str, np.ndarray] = {key: archive[key] for key in archive.files}
    if arrays["left"].dtype != np.uint8 or arrays["disparity"].dtype != np.float32 or arrays["known"].dtype != np.bool_:
        raise ValueError(f"Unexpected normalized sample types: {path}")
    return arrays


def decoded_hash(image: np.ndarray) -> str:
    """Hash shape and RGB pixel bytes, independent of lossless file encoding."""
    return hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()


def perceptual_hash(image: np.ndarray) -> int:
    """63-bit DCT screening hash; proximity is a review signal, not independence proof."""
    gray: np.ndarray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    values: np.ndarray = cv2.dct(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8].ravel()[1:]
    return sum(int(value) << index for index, value in enumerate(values > np.median(values)))


def prepare_scene(scene: str, root: Path, archive: Path, divisor: float, tolerance: float) -> dict[str, object]:
    """RGB uint8 HW3, float32 HW pixel GT, bool known and GT-derived nonocc masks.

    View1 is left, view5 right. Zero labels are unknown. dmin is used only for
    3D depth reconstruction, never added to matching pixel disparity.
    """
    with zipfile.ZipFile(archive) as source:
        names: dict[str, str] = {Path(name).name: name for name in source.namelist() if not name.endswith("/")}
        def decode(name: str, flag: int) -> np.ndarray:
            value: np.ndarray | None = cv2.imdecode(np.frombuffer(source.read(names[name]), np.uint8), flag)
            if value is None:
                raise ValueError(f"Cannot decode {scene}/{name}")
            return value
        left: np.ndarray = cv2.cvtColor(decode("view1.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        right: np.ndarray = cv2.cvtColor(decode("view5.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        raw: np.ndarray = decode("disp1.png", cv2.IMREAD_UNCHANGED)
        raw_right: np.ndarray = decode("disp5.png", cv2.IMREAD_UNCHANGED)
        if raw.ndim != 2 or left.shape != right.shape or left.shape[:2] != raw.shape:
            raise ValueError("Unexpected official dataset dimensions")
        truth: np.ndarray = raw.astype(np.float32)/divisor
        truth_right: np.ndarray = raw_right.astype(np.float32)/divisor
        known: np.ndarray = raw > 0
        height, width = truth.shape
        xr: np.ndarray = np.rint(np.arange(width)[None, :]-truth).astype(np.int32)
        inside: np.ndarray = (xr >= 0) & (xr < width)
        clipped: np.ndarray = np.clip(xr, 0, width-1)
        counterpart: np.ndarray = truth_right[np.arange(height)[:, None], clipped]
        nonocc: np.ndarray = known & inside & (counterpart > 0) & (np.abs(truth-counterpart) <= tolerance)
        # Sign/unit sanity uses GT warping only, not any evaluated matcher.
        y: np.ndarray = np.arange(height)[:, None]
        wrong: np.ndarray = np.clip(np.rint(np.arange(width)[None, :]+truth).astype(np.int32), 0, width-1)
        correct_residual: float = float(np.abs(left.astype(np.float32)-right[y, clipped].astype(np.float32))[nonocc].mean())
        wrong_residual: float = float(np.abs(left.astype(np.float32)-right[y, wrong].astype(np.float32))[nonocc].mean())
        arrays: dict[str, np.ndarray] = {"left": left, "right": right, "disparity": truth.astype(np.float32), "known": known, "non_occluded": nonocc}
        target: Path = root/f"{scene}.npz"
        if target.exists():
            raise ValueError("Normalized scene exists; preserve original preparation")
        np.savez(target, **arrays)
        if not correct_residual < wrong_residual:
            raise ValueError(f"Middlebury GT direction check failed: {scene}")
        return {"gt_warp_rgb_residual": correct_residual, "wrong_sign_rgb_residual": wrong_residual}


def normalize_pair(left: np.ndarray, right: np.ndarray, disparity: np.ndarray, known: np.ndarray,
                   non_occluded: np.ndarray, maximum_width: int) -> dict[str, np.ndarray]:
    """RGB uint8 HW3, disparity float32 HW px, masks bool HW at shared input scale.

    Resize labels/masks with nearest-neighbor to avoid synthesizing depth-edge
    labels. Multiply disparity by actual horizontal scale; preserve unknowns.
    """
    if left.shape != right.shape or left.shape[:2] != disparity.shape or known.shape != disparity.shape or non_occluded.shape != disparity.shape:
        raise ValueError("Unpaired stereo source dimensions")
    height, width = disparity.shape
    target_width: int = min(width, maximum_width)
    target_height: int = int(round(height*target_width/width))
    if target_width != width:
        left = cv2.resize(left, (target_width, target_height), interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, (target_width, target_height), interpolation=cv2.INTER_AREA)
        disparity = cv2.resize(disparity, (target_width, target_height), interpolation=cv2.INTER_NEAREST)*np.float32(target_width/width)
        known = cv2.resize(known.astype(np.uint8), (target_width, target_height), interpolation=cv2.INTER_NEAREST).astype(bool)
        non_occluded = cv2.resize(non_occluded.astype(np.uint8), (target_width, target_height), interpolation=cv2.INTER_NEAREST).astype(bool)
    return {"left": left, "right": right, "disparity": np.where(known, disparity, 0).astype(np.float32),
        "known": known, "non_occluded": non_occluded & known}


def read_pfm(stream: BinaryIO, shape: tuple[int, int]) -> np.ndarray:
    """Read grayscale PFM float32 px, bottom-up, +inf invalid; no value rescaling."""
    if stream.readline().strip() != b"Pf":
        raise ValueError("Expected grayscale PFM")
    width, height = map(int, stream.readline().split())
    scale: float = float(stream.readline())
    if (height, width) != shape or not np.isfinite(scale) or scale == 0:
        raise ValueError("Invalid PFM shape or byte order")
    payload: bytes = stream.read()
    if len(payload) != height * width * 4:
        raise ValueError("Invalid PFM payload length")
    values: np.ndarray = np.frombuffer(payload, dtype="<f4" if scale < 0 else ">f4").reshape(shape)
    result: np.ndarray = values[::-1].astype(np.float32).copy()
    if np.isnan(result).any() or np.isneginf(result).any():
        raise ValueError("Unsupported PFM invalid values")
    return result


def decode(payload: bytes, *, grayscale: bool = False) -> np.ndarray:
    flag: int = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image: np.ndarray | None = cv2.imdecode(np.frombuffer(payload, np.uint8), flag)
    if image is None:
        raise ValueError("Cannot decode stereo input")
    return image if grayscale else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def fetch_booster(raw: dict, inputs: Path, rows: list[dict], offline: bool) -> Path:
    """Fetch just the 109 selected pairs and their shared scene labels."""
    folder: Path = inputs / "booster"
    paths: set[str] = set()
    for row in rows:
        prefix: str = f"train/balanced/{row['scene']}"
        paths.update((f"{prefix}/disp_00.npy", f"{prefix}/mask_00.png",
                      f"{prefix}/camera_00/{row['view']}", f"{prefix}/camera_02/{row['view']}"))
    if all((folder / name).is_file() for name in paths):
        return folder
    if offline:
        raise FileNotFoundError(f"Booster source files missing from {folder}")
    url: str = raw["booster_archive"]
    length: int = raw["booster_archive_bytes"]
    index: dict[str, dict] = {
        row["name"]: row for row in booster_index(url, length) if row["name"] in paths
    }
    if set(index) != paths:
        raise ValueError("The official Booster archive differs from the frozen sample selection")
    with ThreadPoolExecutor(max_workers=4) as pool:
        for number, _ in enumerate(pool.map(lambda name: download_member(url, length, index[name], folder),
                                             sorted(paths)), start=1):
            if number % 25 == 0:
                print(f"Booster source files: {number}/{len(paths)}", flush=True)
    return folder


def normalize_public(config: Path, offline: bool) -> int:
    raw: dict = json.loads(config.read_text(encoding="utf-8"))
    samples: list[dict] = json.loads((config.parent / raw["samples"]).read_text(encoding="utf-8"))
    target: Path = (config.parent / raw["normalized_root"]).resolve()
    inputs: Path = (config.parent / raw["input_root"]).resolve()
    target.mkdir(parents=True, exist_ok=True)
    inputs.mkdir(parents=True, exist_ok=True)
    baseline = load_offline_config(config.parent / raw["baseline_config"])
    if baseline.ground_truth_root is None:
        raise ValueError("Middlebury development requires left ground truth")

    def archive(url: str, path: Path, expected: str) -> Path:
        if offline and not path.is_file():
            raise FileNotFoundError(path)
        return download_checked(url, path, expected)

    pending: list[dict] = []
    for row in samples:
        path: Path = target / row["path"]
        if path.is_file():
            if digest(path) != row["sha256"]:
                raise ValueError(f"Prepared file differs from the frozen sample: {path}")
        else:
            pending.append(row)

    datasets: set[str] = {row["dataset"] for row in pending}
    if "eth3d_development" in datasets:
        eth_root: Path = inputs / "eth3d"
        eth_root.mkdir(exist_ok=True)
        tar: str | None = shutil.which("tar")
        if tar is None:
            raise FileNotFoundError("Install bsdtar (Windows tar), or unpack the two ETH3D archives into the same eth3d folder.")
        for source in raw["eth3d_archives"]:
            path = archive(source["url"], inputs / source["filename"], source["sha256"])
            subprocess.run([tar, "-xf", str(path), "-C", str(eth_root)], check=True)

    mobile: Path | None = None
    if "mobile_development" in datasets:
        mobile = archive(raw["mobile_archive"], inputs / "mobile-all.zip", raw["mobile_archive_sha256"])
    booster: Path | None = None
    if "booster" in datasets:
        booster = fetch_booster(raw, inputs, [row for row in pending if row["dataset"] == "booster"], offline)

    for number, row in enumerate(pending, start=1):
        path = target / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset: str = row["dataset"]
        scene: str = row["scene"]
        if dataset == "middlebury2006_new":
            name: str = f"{scene}-2views.zip"
            compressed: Path = archive(raw["new_data_url"] + name, inputs / name, row["archive_sha256"])
            prepare_scene(scene, path.parent, compressed, raw["new_gt_encoding_divisor"], raw["new_gt_lr_tolerance_px"])
        else:
            if dataset == "middlebury_development":
                pair = load_middlebury_scene(baseline.image_root, "trainingQ", scene, ground_truth_root=baseline.ground_truth_root)
                left: np.ndarray = cv2.cvtColor(pair.rectified_pair.left_bgr, cv2.COLOR_BGR2RGB)
                right: np.ndarray = cv2.cvtColor(pair.rectified_pair.right_bgr, cv2.COLOR_BGR2RGB)
                truth: np.ndarray = pair.ground_truth.disparity_px
                known: np.ndarray = np.isfinite(truth) & (truth > 0)
                nonocc: np.ndarray = pair.ground_truth.non_occluded_mask
            elif dataset == "eth3d_development":
                folder: Path = inputs / "eth3d" / scene
                left = decode((folder / "im0.png").read_bytes())
                right = decode((folder / "im1.png").read_bytes())
                with (folder / "disp0GT.pfm").open("rb") as stream:
                    truth = read_pfm(stream, left.shape[:2])
                mask: np.ndarray = decode((folder / "mask0nocc.png").read_bytes(), grayscale=True)
                known = (mask > 0) & np.isfinite(truth)
                nonocc = mask == 255
            elif dataset == "mobile_development":
                assert mobile is not None
                with zipfile.ZipFile(mobile) as bundle:
                    left = decode(bundle.read(f"data/{scene}/im0.png"))
                    right = decode(bundle.read(f"data/{scene}/im1.png"))
                    truth = read_pfm(BytesIO(bundle.read(f"data/{scene}/disp0.pfm")), left.shape[:2])
                known = np.isfinite(truth) & (truth > 0)
                nonocc = known  # This source provides no official non-occlusion mask.
            elif dataset == "booster":
                assert booster is not None
                folder = booster / "train" / "balanced" / scene
                left = decode((folder / "camera_00" / row["view"]).read_bytes())
                right = decode((folder / "camera_02" / row["view"]).read_bytes())
                truth = np.load(folder / "disp_00.npy", allow_pickle=False).astype(np.float32)
                known = np.isfinite(truth) & (truth > 0)
                nonocc = decode((folder / "mask_00.png").read_bytes(), grayscale=True) == 255
            else:
                raise ValueError(f"Unknown dataset: {dataset}")
            arrays = normalize_pair(left, right, truth, known, nonocc, raw["canonical_max_width"])
            np.savez(path, **arrays)
        if digest(path) != row["sha256"]:
            path.unlink()
            raise ValueError(f"Prepared sample differs from the recorded bytes: {row['path']}")
        if number % 25 == 0:
            print(f"Normalized pairs: {number}/{len(pending)}", flush=True)
    write_json(target / "inventory.json", {"pairs": len(samples), "sample_manifest_sha256": digest(config.parent / raw["samples"])})
    return len(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/stereo_fair_evaluation.json"))
    parser.add_argument("--offline", action="store_true", help="Use only already downloaded official source archives")
    args = parser.parse_args()
    print(f"Verified {normalize_public(args.config.resolve(), args.offline)} stereo pairs.")


if __name__ == "__main__":
    main()
