"""Post-hoc analysis of the unchanged V0 SGBM; GT never enters matching.

Region labels describe measured spatial strata, not proven causes of error.
All disparities/errors are float32 (H,W), px; image intensities use 0..255.
"""

import argparse
import csv
import hashlib
import json
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import OfflineConfig, load_offline_config
from .datasets import MiddleburyScene, load_middlebury_scene
from .disparity import compute_disparity
from .evaluation import evaluate_disparity
from .model import BgrImage, DisparityMap, DisparityResult, SgbmConfig, ValidMask

REGIONS: dict[int, str] = {1: "outside_disparity_range", 2: "disparity_jump_band",
                          3: "low_texture_interior", 4: "other_interior"}


@dataclass(frozen=True)
class DiagnosisConfig:
    """Validated analysis settings; lengths px, intensity standard deviation 0..255."""

    bad_threshold_px: float
    boundary_jump_px: float
    boundary_radius_px: int
    texture_window_px: int
    texture_std_threshold: float


def jump_band(truth_px: DisparityMap, threshold_px: float, radius_px: int) -> ValidMask:
    """Dilate finite-neighbour GT jumps; unknown GT never creates jump seeds."""
    if truth_px.ndim != 2 or truth_px.dtype != np.float32:
        raise ValueError("GT must be float32 (H,W), px")
    if not np.isfinite(threshold_px) or threshold_px <= 0 or type(radius_px) is not int or radius_px < 0:
        raise ValueError("Invalid jump threshold/radius")
    finite: ValidMask = np.isfinite(truth_px)
    safe: DisparityMap = np.where(finite, truth_px, 0)
    horizontal: ValidMask = finite[:, 1:] & finite[:, :-1] & (np.abs(safe[:, 1:] - safe[:, :-1]) > threshold_px)
    vertical: ValidMask = finite[1:, :] & finite[:-1, :] & (np.abs(safe[1:, :] - safe[:-1, :]) > threshold_px)
    seeds: ValidMask = np.zeros_like(finite)
    seeds[:, 1:] |= horizontal
    seeds[:, :-1] |= horizontal
    seeds[1:, :] |= vertical
    seeds[:-1, :] |= vertical
    kernel: NDArray[np.uint8] = np.ones((2 * radius_px + 1, 2 * radius_px + 1), np.uint8)
    return cv2.dilate(seeds.astype(np.uint8), kernel).astype(bool)


def texture_std(left_bgr: BgrImage, window_px: int) -> DisparityMap:
    """Population std of grayscale intensity in a reflected-border local window."""
    if left_bgr.ndim != 3 or left_bgr.shape[2] != 3 or left_bgr.dtype != np.uint8:
        raise ValueError("Expected uint8 BGR (H,W,3)")
    if type(window_px) is not int or window_px < 1 or window_px % 2 != 1:
        raise ValueError("Texture window must be positive odd integer")
    gray: DisparityMap = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean: DisparityMap = cv2.boxFilter(gray, -1, (window_px, window_px), borderType=cv2.BORDER_REFLECT_101)
    square_mean: DisparityMap = cv2.boxFilter(gray * gray, -1, (window_px, window_px), borderType=cv2.BORDER_REFLECT_101)
    return np.sqrt(np.maximum(square_mean - mean * mean, 0)).astype(np.float32)


def classify_regions(
    truth_px: DisparityMap, reference: ValidMask, local_std: DisparityMap,
    sgbm: SgbmConfig, settings: DiagnosisConfig,
) -> NDArray[np.uint8]:
    """Mutually exclusive strata on finite reference GT; 0 means out of domain."""
    if reference.shape != truth_px.shape or local_std.shape != truth_px.shape or reference.dtype != np.bool_:
        raise ValueError("Reference/texture shapes or types differ")
    if not np.isfinite(truth_px[reference]).all() or not np.isfinite(local_std).all():
        raise ValueError("Nonfinite data in classification domain")
    if not np.isfinite(settings.texture_std_threshold) or settings.texture_std_threshold < 0:
        raise ValueError("Invalid texture threshold")
    boundary: ValidMask = jump_band(truth_px, settings.boundary_jump_px, settings.boundary_radius_px)
    outside: ValidMask = (truth_px < sgbm.min_disparity) | (truth_px >= sgbm.min_disparity + sgbm.num_disparities)
    labels: NDArray[np.uint8] = np.zeros(truth_px.shape, np.uint8)
    labels[reference] = 4
    labels[reference & (local_std < settings.texture_std_threshold)] = 3
    labels[reference & boundary] = 2
    labels[reference & outside] = 1
    return labels


def measure(
    truth_px: DisparityMap, prediction: DisparityResult, reference: ValidMask, bad_threshold_px: float,
) -> dict[str, int | float | None]:
    """Errors on finite prediction intersection; never count missing values as zero."""
    if truth_px.shape != prediction.disparity_px.shape or reference.shape != truth_px.shape:
        raise ValueError("Evaluation shapes differ")
    if reference.dtype != np.bool_ or not np.isfinite(truth_px[reference]).all():
        raise ValueError("Invalid reference domain")
    if not np.isfinite(bad_threshold_px) or bad_threshold_px <= 0:
        raise ValueError("Invalid Bad threshold")
    if not np.isfinite(prediction.disparity_px[prediction.valid_mask]).all():
        raise ValueError("Nonfinite prediction marked valid")
    evaluated: ValidMask = reference & prediction.valid_mask
    residual: NDArray[np.float64] = prediction.disparity_px[evaluated].astype(np.float64) - truth_px[evaluated]
    absolute: NDArray[np.float64] = np.abs(residual)
    n: int = int(evaluated.sum())
    ref_n: int = int(reference.sum())
    return {"reference_pixels": ref_n, "evaluated_pixels": n, "missing_pixels": ref_n - n,
            "bad_pixels": int((absolute > bad_threshold_px).sum()),
            "absolute_error_sum_px": float(absolute.sum()), "squared_error_sum_px2": float(np.square(residual).sum()),
            "mae_px": float(absolute.mean()) if n else None,
            "bad_fraction": float((absolute > bad_threshold_px).mean()) if n else None,
            "coverage": n / ref_n if ref_n else None}


def pool(rows: Sequence[dict[str, object]]) -> dict[str, int | float | None]:
    """Pixel-pooled sums; a failed scene is rejected by the caller, never dropped."""
    result: dict[str, int | float | None] = {}
    key: str
    for key in ("reference_pixels", "evaluated_pixels", "missing_pixels", "bad_pixels"):
        result[key] = sum(int(row[key]) for row in rows)
    for key in ("absolute_error_sum_px", "squared_error_sum_px2"):
        result[key] = sum(float(row[key]) for row in rows)
    n: int = int(result["evaluated_pixels"])
    ref_n: int = int(result["reference_pixels"])
    result.update(mae_px=float(result["absolute_error_sum_px"]) / n if n else None,
                  rmse_px=float(np.sqrt(float(result["squared_error_sum_px2"]) / n)) if n else None,
                  bad_fraction=int(result["bad_pixels"]) / n if n else None,
                  coverage=n / ref_n if ref_n else None)
    return result


def write_json(path: Path, value: object) -> None:
    """Write standards-compliant JSON; None represents undefined statistics."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_scene(name: str, baseline: OfflineConfig, settings: DiagnosisConfig, raw: dict[str, object], out: Path) -> dict[str, object]:
    """Run original SGBM once, then inspect GT; save inputs to independent plotting."""
    scene: MiddleburyScene = load_middlebury_scene(
        baseline.image_root, "trainingQ", name, ground_truth_root=baseline.ground_truth_root)
    prediction: DisparityResult = compute_disparity(scene.rectified_pair, scene.geometry, baseline.sgbm)
    if scene.ground_truth is None:
        raise ValueError("Diagnosis requires GT, inference does not")
    truth: DisparityMap = scene.ground_truth.disparity_px
    reference: ValidMask = scene.ground_truth.non_occluded_mask & np.isfinite(truth)
    occluded: ValidMask = scene.ground_truth.valid_mask & ~reference
    local_std: DisparityMap = texture_std(scene.rectified_pair.left_bgr, settings.texture_window_px)
    regions: NDArray[np.uint8] = classify_regions(truth, reference, local_std, baseline.sgbm, settings)
    all_stats: dict = measure(truth, prediction, reference, settings.bad_threshold_px)
    current_metrics = evaluate_disparity(prediction, truth, reference)
    np.testing.assert_allclose(all_stats["mae_px"], current_metrics.mae_px, rtol=1e-12)
    assert all_stats["evaluated_pixels"] == current_metrics.compared_pixel_count
    strata: dict[str, dict] = {label: measure(truth, prediction, reference & (regions == code), settings.bad_threshold_px)
                              for code, label in REGIONS.items()}
    for key in ("reference_pixels", "evaluated_pixels", "missing_pixels", "bad_pixels"):
        assert sum(row[key] for row in strata.values()) == all_stats[key]
    errors: DisparityMap = np.full(truth.shape, np.nan, np.float32)
    common: ValidMask = reference & prediction.valid_mask
    errors[common] = np.abs(prediction.disparity_px[common] - truth[common])
    x, y, width, height = prediction.valid_roi
    roi: ValidMask = np.zeros_like(reference)
    roi[y:y + height, x:x + width] = True
    sensitivity: list[dict] = []
    for jump in raw["sensitivity_jumps_px"]:
        for radius in raw["sensitivity_radii_px"]:
            band: ValidMask = jump_band(truth, float(jump), int(radius))
            sensitivity.append({"jump_px": jump, "radius_px": radius,
                                "band": measure(truth, prediction, reference & band, settings.bad_threshold_px)})
    row: dict[str, object] = {"scene": name, "all": all_stats, "regions": strata,
        "occluded": measure(truth, prediction, occluded, settings.bad_threshold_px),
        "missing_outside_roi": int((reference & ~roi & ~prediction.valid_mask).sum()),
        "missing_inside_roi": int((reference & roi & ~prediction.valid_mask).sum()),
        "search_roi": list(prediction.valid_roi), "sensitivity": sensitivity}
    severity_edges: list[float] = [0, .5, 1, 2, 5, 10, float("inf")]
    row["severity"] = [{"label": f"{low:g}..{high:g}", "pixels": int(((errors > low) & (errors <= high)).sum())}
                       for low, high in zip(severity_edges[:-1], severity_edges[1:])]
    row["severity"][0]["pixels"] += int((errors == 0).sum())
    assert sum(item["pixels"] for item in row["severity"]) == all_stats["evaluated_pixels"]
    folder: Path = out / name
    folder.mkdir()
    write_json(folder / "report.json", row)
    np.savez_compressed(folder / "arrays.npz", left=scene.rectified_pair.left_bgr, right=scene.rectified_pair.right_bgr,
                        prediction=prediction.disparity_px, predicted_valid=prediction.valid_mask, truth=truth,
                        reference=reference, occluded=occluded, regions=regions, error=errors, texture_std=local_std)
    return row


def main(argv: Sequence[str] | None = None) -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/error_diagnosis.json"))
    args: argparse.Namespace = parser.parse_args(argv)
    raw: dict[str, object] = json.loads(args.config.read_text(encoding="utf-8"))
    baseline: OfflineConfig = load_offline_config(args.config.parent / str(raw["baseline_config"]))
    settings: DiagnosisConfig = DiagnosisConfig(**{key: raw[key] for key in DiagnosisConfig.__annotations__})
    out: Path = (args.config.parent / str(raw["output_root"])).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    rows: list[dict[str, object]] = []
    for name in raw["scenes"]:
        rows.append(run_scene(name, baseline, settings, raw, out))
        print(json.dumps({"scene": name, "status": "success", "bad2": rows[-1]["all"]["bad_fraction"]}), flush=True)
    overall: dict = pool([row["all"] for row in rows])
    region_summary: dict[str, dict] = {label: pool([row["regions"][label] for row in rows]) for label in REGIONS.values()}
    for row in region_summary.values():
        row["share_of_bad_pixels"] = row["bad_pixels"] / overall["bad_pixels"] if overall["bad_pixels"] else None
        row["share_of_absolute_error"] = row["absolute_error_sum_px"] / overall["absolute_error_sum_px"]
    sensitivity_summary: list[dict] = []
    for i, setting in enumerate(rows[0]["sensitivity"]):
        band: dict = pool([row["sensitivity"][i]["band"] for row in rows])
        sensitivity_summary.append({"jump_px": setting["jump_px"], "radius_px": setting["radius_px"],
            "band": band, "share_of_bad_pixels": band["bad_pixels"] / overall["bad_pixels"]})
    summary: dict = {"scene_count": len(rows), "failed_scenes": 0, "settings": raw,
        "sgbm": {key: getattr(baseline.sgbm, key) for key in SgbmConfig.__annotations__},
        "versions": {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__},
        "pooling": "pixel-pooled finite non-occluded GT intersected with valid predictions",
        "overall": overall, "scene_mean_mae_px": float(np.mean([row["all"]["mae_px"] for row in rows])),
        "regions": region_summary, "occluded": pool([row["occluded"] for row in rows]),
        "missing_outside_roi": sum(row["missing_outside_roi"] for row in rows),
        "missing_inside_roi": sum(row["missing_inside_roi"] for row in rows),
        "severity": [{"label": entry["label"], "pixels": sum(row["severity"][i]["pixels"] for row in rows)}
                     for i, entry in enumerate(rows[0]["severity"])],
        "sensitivity": sensitivity_summary,
        "scenes": [{key: value for key, value in row.items() if key != "sensitivity"} for row in rows]}
    write_json(out / "summary.json", summary)
    with (out / "regions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["region", *next(iter(region_summary.values())).keys()])
        writer.writeheader()
        for label, row in region_summary.items():
            writer.writerow({"region": label, **row})
    source_paths: list[str] = ["src/disparity.py", "src/datasets.py", "src/reprojection.py", "src/evaluation.py",
        "src/config.py", "src/pipeline.py", "config/offline.json", "environment.yml", "src/error_diagnosis.py", "config/error_diagnosis.json"]
    write_json(out / "provenance.json", {"baseline_commit": "54ce534e8cc82dda5b8789611e66785af7740ff0",
        "sha256": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in source_paths}})
    print(json.dumps({"status": "complete", "scenes": len(rows), "overall": overall}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
