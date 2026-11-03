"""Persist numeric results, display previews, RGB point cloud and run evidence."""
from __future__ import annotations

import csv
import json
import platform
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from stereo_vision.models import FloatMap, Image, PipelineResult, RunConfig, StereoScene


def write_png(path: Path, image: Image) -> None:
    """Save uint8 PNG or raise OSError containing the destination path."""
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot write PNG: {path}")


def save_point_cloud(path: Path, xyz_m: FloatMap, colors_rgb: Image) -> None:
    """Write finite float32 (N,3) XYZ metres and uint8 (N,3) RGB as ASCII PLY."""
    if xyz_m.ndim != 2 or xyz_m.shape[1] != 3 or xyz_m.shape != colors_rgb.shape or not np.isfinite(xyz_m).all():
        raise ValueError("Point cloud requires finite (N,3) XYZ and matching RGB colors")
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(
            "ply\nformat ascii 1.0\ncomment coordinates metres; rectified left camera: X right Y down Z forward\n"
            f"element vertex {len(xyz_m)}\nproperty float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        np.savetxt(stream, np.column_stack((xyz_m, colors_rgb)), fmt="%.7g %.7g %.7g %d %d %d")


def save_outputs(output_dir: Path, scene: StereoScene, result: PipelineResult, config: RunConfig) -> dict[str, object]:
    """Save full-size px/m arrays, bool masks, preview, PLY and JSON/CSV evidence.

    Display normalization never changes numeric arrays. Metrics explicitly mark
    unavailable GT/distance accuracy. Filesystem failures propagate as OSError.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, array in {
        "disparity_px": result.disparity.disparity_px,
        "depth_m": result.reconstruction.depth_m,
        "xyz_m": result.reconstruction.xyz_m,
        "valid_mask": result.reconstruction.valid_mask,
        "search_mask": result.disparity.search_mask,
    }.items():
        np.save(output_dir / f"{name}.npy", array)
    valid: np.ndarray = result.reconstruction.valid_mask
    write_png(output_dir / "valid_mask.png", valid.astype(np.uint8) * 255)
    preview_gray: Image = np.zeros(valid.shape, dtype=np.uint8)
    scale_px: float = max(float(np.percentile(result.disparity.disparity_px[valid], 99)), 1.0)
    preview_gray[valid] = np.clip(result.disparity.disparity_px[valid] / scale_px * 255, 0, 255).astype(np.uint8)
    preview_bgr: Image = cv2.applyColorMap(preview_gray, cv2.COLORMAP_TURBO)
    preview_bgr[~valid] = 0
    write_png(output_dir / "disparity_preview.png", preview_bgr)
    save_point_cloud(output_dir / "cloud.ply", result.reconstruction.xyz_m[valid], scene.pair.left_bgr[valid][:, ::-1])
    report: dict[str, object] = {
        "scene": scene.name, "split": scene.split, "status": "success",
        "source": str(scene.source_dir),
        "image_size_wh": [scene.pair.left_bgr.shape[1], scene.pair.left_bgr.shape[0]],
        "config": asdict(config) | {"data_root": str(config.data_root), "output_root": str(config.output_root)},
        "q_metres": scene.geometry.q.tolist(),
        "num_disparities": result.disparity.num_disparities,
        "matcher": {
            "min_disparity": 0, "mode": "STEREO_SGBM_MODE_SGBM",
            "p1": config.sgbm.p1_scale * config.sgbm.block_size**2,
            "p2": config.sgbm.p2_scale * config.sgbm.block_size**2,
        },
        "valid_points": int(valid.sum()),
        "search_pixels": int(result.disparity.search_mask.sum()),
        "valid_ratio_in_search": float(valid.sum() / result.disparity.search_mask.sum()),
        "disparity_evaluation": asdict(result.metrics) if result.metrics is not None else None,
        "evaluation_domain": "nonoccluded GT (mask == 255), finite positive GT, fixed search ROI",
        "evaluation_status": "evaluated" if result.metrics is not None else "not_evaluated_no_ground_truth",
        "distance_accuracy_status": "not_evaluated_no_independent_distance_reference",
        "stage_ms": result.stage_ms,
        "timing_scope": (
            "single pair: grayscale + SGBM; reprojection; "
            "excludes data loading, evaluation, display and output I/O; no FPS claim"
        ),
        "rectification": "provided by dataset; not rerun",
        "units": {"disparity": "px", "depth_and_xyz": "m", "time": "ms"},
        "versions": {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__},
        "device": {"system": platform.platform(), "machine": platform.machine(), "processor": platform.processor()},
        "preview_scale_max_px": scale_px,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    row: dict[str, object] = {
        "scene": scene.name, "split": scene.split, "status": "success", "failure_reason": "",
        "valid_points": int(valid.sum()), **result.stage_ms,
        **(asdict(result.metrics) if result.metrics is not None else {}),
    }
    with (output_dir / "samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer: csv.DictWriter = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return report
