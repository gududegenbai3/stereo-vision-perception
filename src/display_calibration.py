"""Freeze screen-target sessions, fit training-only intrinsics/extrinsics, validate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .calibration import BoardGeometry, CornerPair, StereoCalibration, calibrate_stereo, canonical_corner_order, object_points
from .model import RawPair
from .provenance import file_hashes, save_source_snapshot
from .camera_io import digest, write_json
from .rectification import build_rectification, rectify_corner_pair, rectify_pair


def reprojection_check(pair: CornerPair, calibration: StereoCalibration) -> dict[str, Any]:
    """Estimate only a per-image target pose; never refit camera parameters."""
    points: NDArray[np.float32] = object_points(calibration.board)
    ok1, rvec1, tvec1 = cv2.solvePnP(points, pair.left_xy, calibration.k1, calibration.d1)
    ok2, rvec2, tvec2 = cv2.solvePnP(points, pair.right_xy, calibration.k2, calibration.d2)
    if not ok1 or not ok2:
        raise ValueError("Target pose estimation failed")
    rotation1: NDArray[np.float64] = cv2.Rodrigues(rvec1)[0]
    rotation2: NDArray[np.float64] = cv2.Rodrigues(rvec2)[0]
    predicted_r: NDArray[np.float64] = calibration.rotation @ rotation1
    predicted_t: NDArray[np.float64] = calibration.rotation @ tvec1 + calibration.translation_m
    projected1: NDArray[np.float64] = cv2.projectPoints(points, rvec1, tvec1, calibration.k1, calibration.d1)[0]
    projected2: NDArray[np.float64] = cv2.projectPoints(points, rvec2, tvec2, calibration.k2, calibration.d2)[0]
    stereo_projected: NDArray[np.float64] = cv2.projectPoints(points, cv2.Rodrigues(predicted_r)[0], predicted_t, calibration.k2, calibration.d2)[0]
    relative_r: NDArray[np.float64] = rotation2 @ rotation1.T
    relative_t: NDArray[np.float64] = tvec2 - relative_r @ tvec1
    return {"left_pose_rmse_px": float(np.sqrt(np.mean(np.sum((projected1 - pair.left_xy) ** 2, axis=2)))),
        "right_pose_rmse_px": float(np.sqrt(np.mean(np.sum((projected2 - pair.right_xy) ** 2, axis=2)))),
        "right_from_left_pose_rmse_px": float(np.sqrt(np.mean(np.sum((stereo_projected - pair.right_xy) ** 2, axis=2)))),
        "pose_derived_baseline_m": float(np.linalg.norm(relative_t)),
        "board_normal_left": rotation1[:, 2].tolist(), "target_origin_left_m": tvec1.ravel().tolist()}


def run(config: Path) -> None:
    settings: dict[str, Any] = json.loads(config.read_text(encoding="utf-8"))
    root: Path = (config.parent / settings["output_root"]).resolve()
    if root.exists():
        raise ValueError("Preserve existing calibration version; use a new output directory")
    target_path: Path = (config.parent / settings["target_config"]).resolve()
    target: dict[str, Any] = json.loads(target_path.read_text(encoding="utf-8"))["chessboard"]
    if target["square_size_m"] is None or settings["automatic_outlier_removal"] is not False:
        raise ValueError("Require measured target and explicit no-outlier-removal protocol")
    board: BoardGeometry = BoardGeometry(target["inner_cols"], target["inner_rows"], target["square_size_m"])
    object_points(board)
    pairs: list[CornerPair] = []
    observations: list[dict[str, Any]] = []
    input_paths: dict[str, Path] = {"config": config, "target_config": target_path}
    for session_name in settings["capture_sessions"]:
        session: Path = (config.parent / session_name).resolve()
        metadata: dict[str, Any] = json.loads((session / "session.json").read_text(encoding="utf-8"))
        done: dict[str, Any] = json.loads((session / "DONE.json").read_text(encoding="utf-8"))
        if metadata["config"]["rig_id"] != settings["rig_id"] or (metadata["physical_square_size_m"] is None or not np.isclose(metadata["physical_square_size_m"], board.square_size_m)):
            raise ValueError("Camera or measured target identity mismatch")
        if metadata["display_square_px"] != target["display_square_px"]:
            raise ValueError("Display target pixel size changed")
        with (session / "pairs.csv").open(encoding="utf-8") as stream:
            rows: list[dict[str, str]] = list(csv.DictReader(stream))
        if len(rows) != done["saved_pairs"]:
            raise ValueError("Completed capture count mismatch")
        for filename in ("session.json", "DONE.json", "pairs.csv", "displayed_pattern.png"):
            input_paths[f"{session.name}/{filename}"] = session / filename
        for row in rows:
            folder: Path = session / row["pair_id"]
            detail: dict[str, Any] = json.loads((folder / "pair.json").read_text(encoding="utf-8"))
            if detail["purpose"] not in ("calibration", "validation"):
                raise ValueError("Capture role was not assigned before calibration")
            image_size: tuple[int, int] = (int(row["width"]), int(row["height"]))
            if image_size != (1280, 720):
                raise ValueError("Unexpected image size")
            for filename in ("left.png", "right.png", "raw.png", "corners.npz", "pair.json"):
                path: Path = folder / filename
                input_paths[f"{session.name}/{row['pair_id']}/{filename}"] = path
                if filename.endswith(".png") and digest(path) != detail["sha256"][filename[:-4]]:
                    raise ValueError("Captured image changed")
            cached = np.load(folder / "corners.npz", allow_pickle=False)
            left: NDArray[np.float32] = canonical_corner_order(cached["left_xy"], board)
            right: NDArray[np.float32] = canonical_corner_order(cached["right_xy"], board)
            pair: CornerPair = CornerPair(f"{session.name}/{row['pair_id']}", left, right)
            pairs.append(pair)
            observations.append({"pair_id": pair.pair_id, "session": session.name, "path": str(folder), "purpose": detail["purpose"],
                "left_order_reversed": not np.array_equal(left, cached["left_xy"]), "right_order_reversed": not np.array_equal(right, cached["right_xy"])})
    fit: list[CornerPair] = [pair for pair, row in zip(pairs, observations) if row["purpose"] == "calibration"]
    validation: list[CornerPair] = [pair for pair, row in zip(pairs, observations) if row["purpose"] == "validation"]
    if len(fit) < settings["minimum_fit_pairs"] or len(validation) < settings["minimum_validation_pairs"]:
        raise ValueError("Insufficient fit or independent validation views")
    root.mkdir(parents=True)
    sources: dict[str, str] = file_hashes({name: Path(name) for name in ("src/calibration.py", "src/rectification.py", "src/display_calibration.py", "src/camera_io.py", "src/provenance.py")})
    save_source_snapshot(root, sources)
    protocol: dict[str, Any] = {"settings": settings, "board": target, "fit_pair_ids": [p.pair_id for p in fit], "validation_pair_ids": [p.pair_id for p in validation],
        "observations": observations, "source_sha256": sources, "input_sha256": file_hashes(input_paths),
        "opencv": cv2.__version__, "calibration_flags": "mono flags=0; stereo CALIB_FIX_INTRINSIC", "outlier_removal": False}
    write_json(root / "protocol.json", protocol)
    print(json.dumps({"fit_pairs": len(fit), "validation_pairs": len(validation), "square_size_m": board.square_size_m}), flush=True)
    cv2.setNumThreads(1)
    calibration: StereoCalibration = calibrate_stereo(fit, board, (1280, 720), settings["maximum_iterations"], settings["epsilon"])
    maps = build_rectification(calibration, settings["rectification_alpha"])
    np.savez_compressed(root / "calibration.npz", K1=calibration.k1, D1=calibration.d1, K2=calibration.k2, D2=calibration.d2,
        R=calibration.rotation, T=calibration.translation_m, E=calibration.essential, F=calibration.fundamental,
        image_size=np.asarray(calibration.image_size_wh), board_size=np.asarray([board.cols, board.rows]), square_size_m=board.square_size_m,
        rms_left=calibration.rms_left_px, rms_right=calibration.rms_right_px, rms_stereo=calibration.rms_stereo_px,
        fit_pair_ids=np.asarray([p.pair_id for p in fit]), validation_pair_ids=np.asarray([p.pair_id for p in validation]), opencv_version=cv2.__version__)
    np.savez_compressed(root / "rectification.npz", R1=maps.r1, R2=maps.r2, P1=maps.p1, P2=maps.p2, Q=maps.q,
        map1_x=maps.map1_x, map1_y=maps.map1_y, map2_x=maps.map2_x, map2_y=maps.map2_y,
        roi1=np.asarray(maps.roi1), roi2=np.asarray(maps.roi2), image_size=np.asarray(maps.image_size_wh))
    reports: list[dict[str, Any]] = []
    vertical_by_role: dict[str, list[NDArray[np.float64]]] = {"calibration": [], "validation": []}
    raw_by_role: dict[str, list[NDArray[np.float64]]] = {"calibration": [], "validation": []}
    contact: NDArray[np.uint8] = np.full((((len(pairs) + 5) // 6) * 180, 6 * 240, 3), 245, np.uint8)
    for index, (pair, row) in enumerate(zip(pairs, observations)):
        rect_left, rect_right = rectify_corner_pair(pair, calibration, maps)
        errors: NDArray[np.float64] = np.abs(rect_left[:, 1] - rect_right[:, 1])
        raw_errors: NDArray[np.float64] = np.abs(pair.left_xy[:, 0, 1] - pair.right_xy[:, 0, 1])
        role: str = row["purpose"]
        vertical_by_role[role].append(errors)
        raw_by_role[role].append(raw_errors)
        check: dict[str, Any] = reprojection_check(pair, calibration)
        in_frame: NDArray[np.bool_] = (rect_left[:, 0] >= 0) & (rect_left[:, 0] < 1280) & (rect_left[:, 1] >= 0) & (rect_left[:, 1] < 720) & (rect_right[:, 0] >= 0) & (rect_right[:, 0] < 1280) & (rect_right[:, 1] >= 0) & (rect_right[:, 1] < 720)
        reports.append({**row, **check, "raw_vertical_mean_px": float(raw_errors.mean()), "rectified_vertical_mean_px": float(errors.mean()),
            "rectified_vertical_p95_px": float(np.percentile(errors, 95)), "corners_in_both_rectified_images_fraction": float(in_frame.mean())})
        folder: Path = Path(row["path"])
        left_bgr: NDArray[np.uint8] = cv2.imread(str(folder / "left.png"))
        right_bgr: NDArray[np.uint8] = cv2.imread(str(folder / "right.png"))
        y, x = (index // 6) * 180, (index % 6) * 240
        contact[y + 25:y + 160, x:x + 240] = cv2.resize(left_bgr, (240, 135), interpolation=cv2.INTER_AREA)
        cv2.putText(contact, f"#{index + 1:02d} {role}", (x + 5, y + 19), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 80, 0) if role == "validation" else (80, 0, 0), 1)
        if role == "validation":
            corrected = rectify_pair(RawPair(left_bgr, right_bgr), maps)
            for name, image in (("left", corrected.left_bgr), ("right", corrected.right_bgr)):
                dest: Path = root / "validation" / f"view-{index + 1:02d}"
                dest.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dest / f"{name}.png"), image)
            before: NDArray[np.uint8] = np.concatenate([left_bgr, right_bgr], axis=1)
            after: NDArray[np.uint8] = np.concatenate([corrected.left_bgr, corrected.right_bgr], axis=1)
            for image in (before, after):
                for line_y in range(0, 720, 60):
                    cv2.line(image, (0, line_y), (2559, line_y), (0, 220, 0), 1)
            canvas: NDArray[np.uint8] = np.full((1530, 2560, 3), 245, np.uint8)
            canvas[40:760] = before
            canvas[810:1530] = after
            cv2.putText(canvas, f"Independent validation #{index+1}: RAW - mean vertical offset {raw_errors.mean():.2f} px", (15, 29), cv2.FONT_HERSHEY_SIMPLEX, .85, (0, 0, 0), 2)
            cv2.putText(canvas, f"CALIBRATED RECTIFICATION - mean {errors.mean():.2f} px; P95 {np.percentile(errors,95):.2f} px", (15, 795), cv2.FONT_HERSHEY_SIMPLEX, .85, (0, 0, 0), 2)
            cv2.imwrite(str(root / f"validation-{index+1:02d}.png"), canvas)
    cv2.imwrite(str(root / "capture_contact.png"), contact)
    metrics: dict[str, Any] = {"fit_pairs": len(fit), "validation_pairs": len(validation), "rms_left_px": calibration.rms_left_px,
        "rms_right_px": calibration.rms_right_px, "rms_stereo_px": calibration.rms_stereo_px, "baseline_estimate_m": float(np.linalg.norm(calibration.translation_m)),
        "fx_left_px": float(calibration.k1[0, 0]), "fx_right_px": float(calibration.k2[0, 0]),
        "estimated_translation_m": calibration.translation_m.ravel().tolist(), "physical_distance_accuracy": "not_evaluated",
        "scale_source": target["measurement"], "views": reports}
    for role in vertical_by_role:
        values: NDArray[np.float64] = np.concatenate(vertical_by_role[role])
        before_values: NDArray[np.float64] = np.concatenate(raw_by_role[role])
        metrics[role] = {"corners": len(values), "raw_vertical_mean_px": float(before_values.mean()), "raw_vertical_p95_px": float(np.percentile(before_values, 95)),
            "rectified_vertical_mean_px": float(values.mean()), "rectified_vertical_median_px": float(np.median(values)), "rectified_vertical_p95_px": float(np.percentile(values, 95)), "rectified_vertical_max_px": float(values.max())}
    metrics["warnings"] = []
    if calibration.rms_stereo_px > settings["training_rms_warning_px"]:
        metrics["warnings"].append("training_stereo_rms_above_diagnostic_threshold")
    if metrics["validation"]["rectified_vertical_p95_px"] > settings["validation_p95_warning_px"]:
        metrics["warnings"].append("independent_vertical_p95_above_diagnostic_threshold")
    if calibration.translation_m[0, 0] >= 0:
        metrics["warnings"].append("translation_sign_incompatible_with_expected_left_reference_disparity")
    write_json(root / "metrics.json", metrics)
    with np.load(root / "calibration.npz", allow_pickle=False) as saved:
        assert np.array_equal(saved["K1"], calibration.k1) and np.array_equal(saved["T"], calibration.translation_m)
        assert len(saved["fit_pair_ids"]) == len(fit) and not set(saved["fit_pair_ids"]) & set(saved["validation_pair_ids"])
    write_json(root / "verification.json", {"fit_count": len(fit), "validation_count": len(validation), "disjoint_roles": True,
        "no_validation_in_fit": True, "archive_reload_equal": True, "input_hashes_unchanged": file_hashes(input_paths) == protocol["input_sha256"],
        "calibration_sha256": digest(root / "calibration.npz"), "rectification_sha256": digest(root / "rectification.npz")})
    print(json.dumps({key: metrics[key] for key in ("fit_pairs", "validation_pairs", "rms_left_px", "rms_right_px", "rms_stereo_px", "baseline_estimate_m", "validation", "warnings")}), flush=True)


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/display_calibration.local.json"))
    args: argparse.Namespace = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
