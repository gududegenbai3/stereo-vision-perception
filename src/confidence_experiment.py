"""Opt-in research runner; keeps original CLI and baseline stages unchanged."""

import argparse
import hashlib
import json
import platform
import sys
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from .provenance import file_hashes, require_file_hashes
from time import perf_counter

import cv2
import numpy as np

from .confidence import combine_confidence, dps_confidence, left_right_confidence
from .config import OfflineConfig, load_offline_config
from .datasets import MiddleburyScene, load_middlebury_scene
from .disparity import compute_disparity
from .model import (
    BgrImage, ConfidenceExperimentResult, ConfidenceResult, DisparityMap,
    DisparityResult, RectificationGeometry, RectifiedPair, SgbmConfig, ValidMask,
)
from .reestimation import select_candidate
from .reprojection import reproject_disparity


def shifted_right_pair(
    pair: RectifiedPair, geometry: RectificationGeometry, shift: int,
) -> tuple[RectifiedPair, RectificationGeometry]:
    """Move right image content left by shift px, using zero padding, never wrap.

    Transform the right imaging ROI too; negative shift moves content right.
    Q is not used here: shifted predictions are only for consistency scoring.
    """
    width, height = geometry.image_size
    if type(shift) is not int or abs(shift) >= width:
        raise ValueError("Invalid image shift")
    shifted: BgrImage = np.zeros_like(pair.right_bgr)
    if shift >= 0:
        shifted[:, :width - shift] = pair.right_bgr[:, shift:]
    else:
        shifted[:, -shift:] = pair.right_bgr[:, :width + shift]
    x, y, roi_width, roi_height = geometry.roi_right
    start: int = max(0, x - shift)
    stop: int = min(width, x + roi_width - shift)
    if stop <= start:
        raise ValueError("Shift leaves no right image ROI")
    shifted_geometry: RectificationGeometry = replace(geometry, roi_right=(start, y, stop - start, roi_height))
    return RectifiedPair(pair.left_bgr, shifted), shifted_geometry


def _measure_confidence(
    pair: RectifiedPair, geometry: RectificationGeometry, config: SgbmConfig,
    anchor: DisparityResult, shifts: Sequence[int],
) -> tuple[ConfidenceResult, ConfidenceResult]:
    swapped: RectifiedPair = RectifiedPair(pair.right_bgr, pair.left_bgr)
    right_geometry: RectificationGeometry = replace(geometry, roi_left=geometry.roi_right, roi_right=geometry.roi_left)
    right_config: SgbmConfig = replace(config, min_disparity=-(config.min_disparity + config.num_disparities - 1))
    right: DisparityResult = compute_disparity(swapped, right_geometry, right_config)
    lrc: ConfidenceResult = left_right_confidence(anchor, right)
    predictions: dict[int, DisparityResult] = {}
    for shift in shifts:
        shifted_pair, shifted_geometry = shifted_right_pair(pair, geometry, shift)
        predictions[shift] = compute_disparity(
            shifted_pair, shifted_geometry,
            replace(config, min_disparity=config.min_disparity + shift),
        )
    return lrc, dps_confidence(anchor, predictions)


def predict_with_confidence(
    pair: RectifiedPair, geometry: RectificationGeometry,
    baseline_config: SgbmConfig, candidate_config: SgbmConfig,
    *, shifts: Sequence[int] = (-2, -1, 1, 2), threshold: float = 0.5, margin: float = 0.15,
) -> ConfidenceExperimentResult:
    """GT-free prediction API; all outputs retain the original left pixel grid.

    One baseline + one reversed + four shifted predictions per parameter set.
    Candidate parameter selection belongs to development, not this function.
    """
    if tuple(shifts) != (-2, -1, 1, 2):
        raise ValueError("Frozen DPS protocol requires shifts [-2,-1,1,2]")
    start: float = perf_counter()
    baseline: DisparityResult = compute_disparity(pair, geometry, baseline_config)
    matched: float = perf_counter()
    base_lrc, base_dps = _measure_confidence(pair, geometry, baseline_config, baseline, shifts)
    confidence_done: float = perf_counter()
    candidate: DisparityResult = compute_disparity(pair, geometry, candidate_config)
    candidate_lrc, candidate_dps = _measure_confidence(pair, geometry, candidate_config, candidate, shifts)
    candidates_done: float = perf_counter()
    base_combined: ConfidenceResult = combine_confidence(base_lrc, base_dps)
    candidate_combined: ConfidenceResult = combine_confidence(candidate_lrc, candidate_dps)
    refined = select_candidate(baseline, candidate, base_combined, candidate_combined, threshold=threshold, margin=margin)
    finished: float = perf_counter()
    return ConfidenceExperimentResult(
        baseline, candidate, base_lrc, base_dps, candidate_lrc, candidate_dps,
        base_combined, candidate_combined, refined,
        {"baseline_ms": (matched - start) * 1000,
         "baseline_confidence_ms": (confidence_done - matched) * 1000,
         "candidate_and_confidence_ms": (candidates_done - confidence_done) * 1000,
         "selection_ms": (finished - candidates_done) * 1000,
         "research_total_ms": (finished - start) * 1000},
    )


def evaluate_arrays(values: DisparityMap, mask: ValidMask, truth: DisparityMap, reference: ValidMask) -> dict[str, object]:
    """Fixed reference denominator; missing predictions never become zero errors."""
    valid: ValidMask = reference & mask & np.isfinite(values)
    count: int = int(valid.sum())
    reference_count: int = int(reference.sum())
    error: np.ndarray = np.abs(values[valid].astype(np.float64) - truth[valid])
    return {"reference_pixels": reference_count, "compared_pixels": count,
            "coverage": count / reference_count if reference_count else None,
            "mae_px": float(error.mean()) if count else None,
            "rmse_px": float(np.sqrt(np.mean(error**2))) if count else None,
            "bad2_fraction": float(np.mean(error > 2)) if count else None}


def risk_curves(
    prediction: DisparityResult, confidence: ConfidenceResult, truth: DisparityMap,
    reference: ValidMask, fractions: Sequence[float], seed: int,
) -> list[dict[str, object]]:
    """Compare ranking, random selection and oracle at exactly the same counts.

    Unknown confidence sorts last, not out of the evaluation domain. Oracle is
    diagnostic only; its mask never enters inference or refinement.
    """
    valid: ValidMask = reference & prediction.valid_mask & np.isfinite(prediction.disparity_px)
    errors: np.ndarray = np.abs(prediction.disparity_px[valid].astype(np.float64) - truth[valid])
    scores: np.ndarray = np.nan_to_num(confidence.score[valid], nan=-1.0)
    order: np.ndarray = np.argsort(-scores, kind="stable")
    random_order: np.ndarray = np.random.default_rng(seed).permutation(errors.size)
    oracle: np.ndarray = np.sort(errors)
    rows: list[dict[str, object]] = []
    for fraction in fractions:
        count: int = max(1, int(errors.size * fraction)) if errors.size else 0
        rows.append({"retained_fraction": fraction, "count": count,
                     "coverage": count / int(reference.sum()) if reference.any() else None,
                     "confidence_mae_px": float(errors[order[:count]].mean()) if count else None,
                     "random_mae_px": float(errors[random_order[:count]].mean()) if count else None,
                     "oracle_mae_px": float(oracle[:count].mean()) if count else None})
    return rows


def evaluate_prediction(scene: MiddleburyScene, result: ConfidenceExperimentResult, settings: dict[str, object]) -> dict[str, object]:
    """Evaluate only after GT-free predictions exist; save all methods' counts."""
    truth = scene.ground_truth
    if truth is None:
        raise ValueError("Research evaluation requires public training GT")
    reference: ValidMask = truth.non_occluded_mask & np.isfinite(truth.disparity_px)
    methods: dict[str, tuple[DisparityMap, ValidMask]] = {
        "baseline": (result.baseline.disparity_px, result.baseline.valid_mask),
        "lrc_filter": (result.baseline.disparity_px, result.baseline.valid_mask & (result.baseline_lrc.score >= settings["threshold"])),
        "dps_filter": (result.baseline.disparity_px, result.baseline.valid_mask & (result.baseline_dps.score >= settings["threshold"])),
        "refined": (result.refined.disparity.disparity_px, result.refined.disparity.valid_mask),
    }
    metrics: dict[str, object] = {name: evaluate_arrays(values, mask, truth.disparity_px, reference) for name, (values, mask) in methods.items()}
    common: ValidMask = reference & result.baseline.valid_mask & result.refined.disparity.valid_mask
    metrics["paired_common"] = {
        "baseline": evaluate_arrays(result.baseline.disparity_px, result.baseline.valid_mask, truth.disparity_px, common),
        "refined": evaluate_arrays(result.refined.disparity.disparity_px, result.refined.disparity.valid_mask, truth.disparity_px, common),
    }
    changed: ValidMask = reference & result.refined.replaced_mask
    metrics["replaced"] = {"pixels": int(changed.sum()),
        "baseline": evaluate_arrays(result.baseline.disparity_px, result.baseline.valid_mask, truth.disparity_px, changed),
        "refined": evaluate_arrays(result.refined.disparity.disparity_px, result.refined.disparity.valid_mask, truth.disparity_px, changed)}
    metrics["risk_curves"] = {name: risk_curves(result.baseline, confidence, truth.disparity_px, reference, settings["retained_fractions"], settings["random_seed"])
                             for name, confidence in (("lrc", result.baseline_lrc), ("dps", result.baseline_dps))}
    return metrics


def run_scene(settings: dict[str, object], baseline: OfflineConfig, scene_name: str, block: int, output: Path) -> dict[str, object]:
    scene: MiddleburyScene = load_middlebury_scene(baseline.image_root, "trainingQ", scene_name, ground_truth_root=baseline.ground_truth_root)
    result: ConfidenceExperimentResult = predict_with_confidence(
        scene.rectified_pair, scene.geometry, baseline.sgbm, replace(baseline.sgbm, block_size=block),
        shifts=settings["shifts"], threshold=settings["threshold"], margin=settings["margin"],
    )
    reconstruction = reproject_disparity(scene.rectified_pair, scene.geometry, result.refined.disparity)
    metrics: dict[str, object] = evaluate_prediction(scene, result, settings)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "arrays.npz",
        baseline=result.baseline.disparity_px, baseline_valid=result.baseline.valid_mask,
        candidate=result.candidate.disparity_px, candidate_valid=result.candidate.valid_mask,
        lrc=result.baseline_lrc.score, dps=result.baseline_dps.score,
        combined=result.baseline_combined.score, candidate_confidence=result.candidate_combined.score,
        final=result.refined.disparity.disparity_px, final_valid=result.refined.disparity.valid_mask,
        replaced=result.refined.replaced_mask, depth_m=reconstruction.depth_m,
        truth=scene.ground_truth.disparity_px, reference=scene.ground_truth.non_occluded_mask,
        left_bgr=scene.rectified_pair.left_bgr,
    )
    report: dict[str, object] = {"scene": scene_name, "status": "success", "candidate_block": block,
        "search_roi": list(result.baseline.valid_roi), "metrics": metrics, "timing_ms": result.timing_ms}
    write_json(output / "report.json", report)
    return report


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_settings(path: Path) -> tuple[dict[str, object], OfflineConfig]:
    settings: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    required: set[str] = {"baseline_config", "output_root", "development_scenes", "evaluation_scenes", "shifts", "candidate_blocks", "threshold", "margin", "retained_fractions", "random_seed"}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError("Invalid research configuration fields")
    if settings["shifts"] != [-2, -1, 1, 2] or settings["candidate_blocks"] != [3, 7, 9]:
        raise ValueError("Frozen protocol requires N=5 K=2 and candidate blocks 3,7,9")
    if settings["development_scenes"] != ["Adirondack", "ArtL", "Motorcycle", "MotorcycleE"]:
        raise ValueError("Development split is frozen")
    expected: set[str] = {"Jadeplant", "Piano", "PianoL", "Pipes", "Playroom", "Playtable", "PlaytableP", "Recycle", "Shelves", "Teddy", "Vintage"}
    if len(settings["evaluation_scenes"]) != 11 or set(settings["evaluation_scenes"]) != expected:
        raise ValueError("Evaluation split is frozen")
    if settings["threshold"] != 0.5 or settings["margin"] != 0.15 or settings["random_seed"] != 17:
        raise ValueError("Score thresholds and random seed are frozen")
    if settings["retained_fractions"] != [index / 10 for index in range(1, 11)]:
        raise ValueError("Retained fractions are frozen at 0.1 through 1.0")
    baseline: OfflineConfig = load_offline_config(path.parent / settings["baseline_config"])
    settings["output_root"] = str((path.parent / settings["output_root"]).resolve())
    return settings, baseline


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    """Scene-equal means require every scene; failed samples cannot disappear."""
    successful: list[dict[str, object]] = [row for row in rows if row["status"] == "success"]
    summary: dict[str, object] = {
        "total_scenes": len(rows),
        "successful_scenes": len(successful),
        "failed_scenes": len(rows) - len(successful),
    }
    for method in ("baseline", "lrc_filter", "dps_filter", "refined"):
        method_summary: dict[str, float | None] = {}
        for metric in ("mae_px", "rmse_px", "bad2_fraction", "coverage"):
            values: list[float | None] = [row["metrics"][method][metric] for row in successful]
            complete: bool = bool(rows) and len(successful) == len(rows)
            method_summary[metric] = (
                float(np.mean(values))
                if complete and all(value is not None for value in values)
                else None
            )
        summary[method] = method_summary
    return summary


def run_group(settings: dict[str, object], baseline: OfflineConfig, scenes: list[str], block: int, root: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for name in scenes:
        try:
            row: dict[str, object] = run_scene(settings, baseline, name, block, root / name)
        except (OSError, ValueError, cv2.error) as error:
            row = {"scene": name, "status": "failed", "reason": str(error)}
            (root / name).mkdir(parents=True, exist_ok=True)
            write_json(root / name / "failure.json", row)
        rows.append(row)
        print(json.dumps({"scene": name, "block": block, "status": row["status"]}), flush=True)
    report: dict[str, object] = {"scenes": rows, "summary": aggregate(rows)}
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "summary.json", report)
    return report


def validate_artifacts(root: Path, scenes: Sequence[str]) -> None:
    """Fail on missing scenes, illegal arrays or evidence of GT leakage."""
    for scene in scenes:
        report: dict[str, object] = json.loads((root / scene / "report.json").read_text(encoding="utf-8"))
        if report["scene"] != scene or report["status"] != "success":
            raise ValueError(f"Missing/failed scene {scene}")
        with np.load(root / scene / "arrays.npz") as arrays:
            for prefix in ("baseline", "candidate", "final"):
                values: np.ndarray = arrays[prefix]
                mask: np.ndarray = arrays[prefix + "_valid"]
                if mask.dtype != np.bool_ or values.dtype != np.float32 or mask.shape != values.shape or not np.isfinite(values[mask]).all():
                    raise ValueError(f"Nonfinite result marked valid: {scene}/{prefix}")
            if not np.array_equal(arrays["baseline_valid"], arrays["final_valid"]):
                raise ValueError("Refinement altered baseline coverage")
            if not np.array_equal(arrays["baseline"][~arrays["replaced"]], arrays["final"][~arrays["replaced"]], equal_nan=True):
                raise ValueError("Unselected pixel changed")
        if report.get("inference_used_ground_truth", False):
            raise ValueError("Ground truth used during prediction")


def main(argv: Sequence[str] | None = None) -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/confidence_experiment.json"))
    parser.add_argument("--phase", choices=("develop", "evaluate", "verify"), required=True)
    args: argparse.Namespace = parser.parse_args(argv)
    try:
        settings, baseline = load_settings(args.config)
        root: Path = Path(settings["output_root"])
        cv2.setNumThreads(1)
        fingerprint: str = hashlib.sha256(args.config.read_bytes()).hexdigest()
        dependencies: dict[str, Path] = {"experiment": args.config,
            "baseline": args.config.parent / settings["baseline_config"]}
        freeze_path: Path = root / "frozen.json"
        if args.phase == "develop":
            if freeze_path.exists():
                raise ValueError("Parameters already frozen; do not retune after evaluation")
            candidates: list[dict[str, object]] = []
            for block in settings["candidate_blocks"]:
                report = run_group(settings, baseline, settings["development_scenes"], block, root / "development" / f"block_{block}")
                candidates.append({"block": block, "summary": report["summary"]})
            eligible: list[dict[str, object]] = [row for row in candidates if row["summary"]["failed_scenes"] == 0 and row["summary"]["refined"]["mae_px"] is not None]
            if not eligible:
                raise ValueError("No complete development round; fix failures before freezing")
            chosen: dict[str, object] = min(eligible, key=lambda row: row["summary"]["refined"]["mae_px"])
            write_json(freeze_path, {"config_sha256": fingerprint, "configuration_sha256": file_hashes(dependencies), "candidate_block": chosen["block"], "development_rounds": candidates,
                "versions": {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__}})
            print("FROZEN", chosen["block"], flush=True)
        else:
            frozen: dict[str, object] = json.loads(freeze_path.read_text(encoding="utf-8"))
            if frozen["config_sha256"] != fingerprint:
                raise ValueError("Configuration changed after freeze")
            if "configuration_sha256" in frozen:
                require_file_hashes(frozen["configuration_sha256"], dependencies)
            elif args.phase == "evaluate":
                raise ValueError("Legacy record did not freeze baseline configuration; use a new output root")
            else:
                print("LEGACY: validating saved arrays only; baseline configuration identity was not recorded", flush=True)
            if args.phase == "evaluate":
                if (root / "evaluation").exists():
                    raise ValueError("Evaluation already exists; inspect it, do not tune or overwrite")
                report = run_group(settings, baseline, settings["evaluation_scenes"], frozen["candidate_block"], root / "evaluation")
                if report["summary"]["failed_scenes"]:
                    return 1
            validate_artifacts(root / "evaluation", settings["evaluation_scenes"])
            print("VERIFIED", len(settings["evaluation_scenes"]), "evaluation scenes", flush=True)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
