"""Independent saved-parameter validation, figures and local calibration selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .camera_io import digest, write_json


def report(config: Path) -> None:
    settings: dict[str, Any] = json.loads(config.read_text(encoding="utf-8"))
    root: Path = (config.parent / settings["output_root"]).resolve()
    protocol: dict[str, Any] = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    metrics: dict[str, Any] = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    calibration = np.load(root / "calibration.npz", allow_pickle=False)
    rectification = np.load(root / "rectification.npz", allow_pickle=False)
    if set(protocol["fit_pair_ids"]) & set(protocol["validation_pair_ids"]):
        raise ValueError("Validation leaked into fitting")
    direct_errors: list[np.ndarray] = []
    validation_rows: list[dict[str, Any]] = []
    for index, row in enumerate(protocol["observations"]):
        if row["purpose"] != "validation":
            continue
        with np.load(Path(row["path"]) / "corners.npz", allow_pickle=False) as corner_file:
            left: np.ndarray = corner_file["left_xy"].copy()
            right: np.ndarray = corner_file["right_xy"].copy()
        if row["left_order_reversed"]:
            left = left[::-1].copy()
        if row["right_order_reversed"]:
            right = right[::-1].copy()
        corrected_left: np.ndarray = cv2.undistortPoints(left, calibration["K1"], calibration["D1"], R=rectification["R1"], P=rectification["P1"])
        corrected_right: np.ndarray = cv2.undistortPoints(right, calibration["K2"], calibration["D2"], R=rectification["R2"], P=rectification["P2"])
        errors: np.ndarray = np.abs(corrected_left[:, 0, 1].astype(np.float64) - corrected_right[:, 0, 1].astype(np.float64))
        direct_errors.append(errors)
        recorded: dict[str, Any] = metrics["views"][index]
        if not np.isclose(errors.mean(), recorded["rectified_vertical_mean_px"], atol=1e-8):
            raise ValueError("Stored validation metric differs from saved-parameter recomputation")
        validation_rows.append({"number": len(validation_rows) + 1, "capture_number": index + 1, "pair_id": row["pair_id"],
            "before_mean_px": recorded["raw_vertical_mean_px"], "after_mean_px": float(errors.mean()), "after_p95_px": float(np.percentile(errors, 95))})
    pooled: np.ndarray = np.concatenate(direct_errors)
    if not np.isclose(np.percentile(pooled, 95), metrics["validation"]["rectified_vertical_p95_px"], atol=1e-8):
        raise ValueError("Pooled independent validation metric mismatch")
    fit_points: list[np.ndarray] = []
    for row in protocol["observations"]:
        if row["purpose"] == "calibration":
            with np.load(Path(row["path"]) / "corners.npz") as values:
                fit_points.append(values["left_xy"].reshape(-1, 2))
    points: np.ndarray = np.concatenate(fit_points)
    audit: dict[str, Any] = {"independently_recomputed_validation_views": len(validation_rows), "validation_corners": int(pooled.size),
        "fit_validation_roles_disjoint": True, "saved_parameter_metrics_equal": True,
        "fit_left_corner_bounds_xy": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        "calibration_sha256": digest(root / "calibration.npz"), "rectification_sha256": digest(root / "rectification.npz"),
        "distance_accuracy": "not_evaluated"}
    write_json(root / "independent_audit.json", audit)
    fig, axis = plt.subplots(figsize=(10, 4.4), layout="constrained")
    x: np.ndarray = np.arange(len(validation_rows))
    before: list[float] = [row["before_mean_px"] for row in validation_rows]
    after: list[float] = [row["after_mean_px"] for row in validation_rows]
    for bars in (axis.bar(x - .18, before, .36, label="Before rectification", color="#dd8b39"),
                 axis.bar(x + .18, after, .36, label="Calibrated rectification", color="#2379b9")):
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axis.set_xticks(x, [f"V{i+1}" for i in x])
    axis.set_ylabel("Mean absolute vertical residual / pixels")
    axis.set_title(f"{len(validation_rows)} independent views: none used in fitting")
    axis.legend()
    axis.set_ylim(0, max(before) * 1.23)
    axis.grid(axis="y", alpha=.2)
    fig.savefig(root / "independent_validation.png", dpi=150)
    plt.close(fig)
    selector: dict[str, Any] = {"rig_id": settings["rig_id"], "status": "display_calibration_independently_checked_geometry_only",
        "calibration_npz": str((root / "calibration.npz").resolve()), "calibration_sha256": audit["calibration_sha256"],
        "rectification_npz": str((root / "rectification.npz").resolve()), "rectification_sha256": audit["rectification_sha256"],
        "image_size_wh": calibration["image_size"].tolist(), "distance_accuracy": "not_evaluated",
        "validity": "Only unchanged focus/inter-camera pose/capture mode. Prior scene photographs require a separate geometry compatibility check."}
    if metrics["warnings"]:
        selector["status"] = "diagnostic_warnings_do_not_use_without_review"
    write_json(config.parent / "current_calibration.local.json", selector)
    lines: list[str] = ["# 显示屏标定与独立几何验证", "",
        f"{metrics['fit_pairs']} 对拟合，{metrics['validation_pairs']} 对独立验证；单格 {float(calibration['square_size_m'])*1000:.3f} mm。",
        f"独立验证垂直差均值 {metrics['validation']['raw_vertical_mean_px']:.3f} → {pooled.mean():.3f} px，P95 {np.percentile(pooled,95):.3f} px。",
        "", "![独立验证](independent_validation.png)", "",
        f"基线估计 {metrics['baseline_estimate_m']*1000:.1f} mm，仅由标定估计，未独立实测。",
        "这是几何验证，不是距离精度验收。改变调焦、两镜头相对姿态或拍摄模式后须重查。",
        "[完整指标](metrics.json)；[协议](protocol.json)；[独立复算](independent_audit.json)。", ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"independent_views": len(validation_rows), "independent_corners": len(pooled), "current_selector": str(config.parent / 'current_calibration.local.json'), "report": str(root / 'report.md')}), flush=True)


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/display_calibration.local.json"))
    args: argparse.Namespace = parser.parse_args()
    report(args.config.resolve())


if __name__ == "__main__":
    main()
