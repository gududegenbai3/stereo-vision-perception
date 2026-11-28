"""Camera record IO; no model or experiment-entry-point dependencies."""

from __future__ import annotations

from collections.abc import Mapping
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


def digest(path: Path) -> str:
    """SHA256 of exact file bytes; IO failures propagate."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write UTF-8 metadata; caller owns output directory and overwrite policy."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def capture_inventory(session: Path) -> list[dict[str, Any]]:
    """Read a completed side-by-side capture; verify hashes, split and dimensions.

    Image arrays are native BGR uint8 HxWx3. Reject escaping/duplicate pair IDs,
    incomplete records and modified PNGs instead of silently skipping samples.
    """
    done: dict[str, Any] = json.loads((session / "DONE.json").read_text(encoding="utf-8"))
    with (session / "pairs.csv").open(encoding="utf-8") as stream:
        rows: list[dict[str, str]] = list(csv.DictReader(stream))
    if not rows or len(rows) != done["saved_pairs"]:
        raise ValueError("Capture is empty or incomplete")
    seen: set[str] = set()
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        pair_id: str = row["pair_id"]
        if pair_id in seen or Path(pair_id).name != pair_id or pair_id in (".", ".."):
            raise ValueError("Invalid/duplicate pair identity")
        seen.add(pair_id)
        folder: Path = session / pair_id
        if not folder.resolve().is_relative_to(session.resolve()):
            raise ValueError("Capture path escapes session")
        detail: dict[str, Any] = json.loads((folder / "pair.json").read_text(encoding="utf-8"))
        images: dict[str, NDArray[np.uint8]] = {}
        hashes: dict[str, str] = {}
        for name in ("raw", "left", "right"):
            file: Path = folder / f"{name}.png"
            hashes[name] = digest(file)
            if hashes[name] != detail["sha256"][name]:
                raise ValueError("Capture PNG hash mismatch")
            images[name] = cv2.imread(str(file))
            if images[name] is None:
                raise ValueError("Unreadable capture PNG")
        height, width = images["left"].shape[:2]
        if images["raw"].shape != (height, 2 * width, 3) or images["right"].shape != images["left"].shape or not np.array_equal(images["left"], images["raw"][:, :width]) or not np.array_equal(images["right"], images["raw"][:, width:]):
            raise ValueError("Native stereo split mismatch")
        if (width, height) != (int(row["width"]), int(row["height"])):
            raise ValueError("Capture CSV dimensions mismatch")
        samples.append({"index": index, "pair_id": pair_id, "sha256": hashes, "wh": [width, height]})
    return samples


def calibrated_inputs(
    left: NDArray[np.uint8], right: NDArray[np.uint8],
    maps: Mapping[str, NDArray[Any]], scale: float, *,
    support_masks: Mapping[str, NDArray[np.uint8]] | None = None,
) -> dict[str, NDArray[np.uint8]]:
    """Fixed maps -> BGR uint8 images and 0/255 support masks, no scene fitting.

    Input HxWx3 must match map HxW. Output uses the same scale for both cameras;
    disparity pixels refer to this output grid. Q is NOT resized by this function.
    Cached support masks must come from the same maps and scale.
    """
    if scale not in (.5, 1.0):
        raise ValueError("Only native or uniform half size is supported")
    width, height = map(int, maps["image_size"])
    size: tuple[int, int] = (round(width * scale), round(height * scale))
    result: dict[str, NDArray[np.uint8]] = {}
    for number, (name, image) in enumerate((("left", left), ("right", right)), 1):
        if image is None or image.shape != (height, width, 3) or image.dtype != np.uint8:
            raise ValueError("Native camera input shape/dtype mismatch")
        sources: tuple[tuple[str, NDArray[np.uint8]], ...] = (("", image),)
        if support_masks is None:
            sources += (("_support", np.full((height, width), 255, np.uint8)),)
        else:
            result[name + "_support"] = support_masks[name + "_support"]
        for suffix, source in sources:
            corrected: NDArray[np.uint8] = cv2.remap(source, maps[f"map{number}_x"], maps[f"map{number}_y"], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            value: NDArray[np.uint8] = cv2.resize(corrected, size, interpolation=cv2.INTER_AREA) if scale != 1 else corrected
            result[name + suffix] = ((value == 255) * 255).astype(np.uint8) if suffix else value
    return result


def sgbm_predict(arrays: dict[str, NDArray[np.uint8]], parameters: dict[str, Any], maximum: int) -> NDArray[np.float32]:
    """RGB uint8 HxWx3 -> float32 HxW pixels; OpenCV invalid/ROI -> NaN."""
    block: int = parameters["block"]
    if maximum < 16 or maximum % 16 or type(block) is not int or block < 1 or block % 2 != 1:
        raise ValueError("Invalid disparity range or block size")
    if parameters["input"] != "gray" or parameters["mode"] != "SGBM":
        raise ValueError("This comparison fixes gray SGBM")
    matcher: cv2.StereoSGBM = cv2.StereoSGBM_create(minDisparity=0, numDisparities=maximum, blockSize=block,
        P1=8 * block**2, P2=32 * block**2, mode=cv2.STEREO_SGBM_MODE_SGBM,
        uniquenessRatio=parameters["uniqueness_ratio"], speckleWindowSize=parameters["speckle_window_size"],
        speckleRange=parameters["speckle_range"], disp12MaxDiff=parameters["disp12_max_diff"])
    raw: NDArray[np.int16] = matcher.compute(cv2.cvtColor(arrays["left"], cv2.COLOR_RGB2GRAY), cv2.cvtColor(arrays["right"], cv2.COLOR_RGB2GRAY))
    height, width = raw.shape
    x, y, w, h = cv2.getValidDisparityROI((0, 0, width, height), (0, 0, width, height), 0, maximum, block)
    valid: NDArray[np.bool_] = np.zeros(raw.shape, bool)
    valid[y:y+h, x:x+w] = True
    valid &= (raw > -16) & (raw < maximum * 16)
    return np.where(valid, raw.astype(np.float32) / 16, np.nan).astype(np.float32)
