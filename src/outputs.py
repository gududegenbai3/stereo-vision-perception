"""导出离线重建的数值结果、预览、彩色点云和指标。"""

import csv
import json
import platform
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeAlias
from uuid import uuid4

import cv2
import numpy as np
from numpy.typing import NDArray

from .model import (
    DepthMap,
    DisparityMap,
    ImageRoi,
    MiddleburySplit,
    PointMap,
    ReconstructionResult,
    SgbmConfig,
    ValidMask,
)

MetricValues: TypeAlias = Mapping[str, int | float | None]


def export_offline_result(
    output_root: Path,
    reconstruction: ReconstructionResult,
    metrics: MetricValues,
    *,
    split: MiddleburySplit,
    scene_name: str,
    sgbm: SgbmConfig,
    search_roi: ImageRoi,
) -> Path:
    """导出一次重建，返回 output_root 下新建的运行目录。

    视差为 float32 (H, W) 像素值，深度与 XYZ 为米制 float32，
    掩码为 bool (H, W)。数组格式不符或无有效点时抛出 ValueError，
    文件写入失败抛出 OSError。全部文件写完后才发布运行目录。
    """
    if split not in ("trainingQ", "testQ"):
        raise ValueError(f"不支持的 Middlebury 划分：{split!r}")
    if not scene_name.strip() or scene_name in (".", "..") or any(
        char in scene_name for char in "/\\\0"
    ):
        raise ValueError("scene_name 必须是单个目录名")

    disparity_px: DisparityMap = reconstruction.disparity_px
    depth_m: DepthMap = reconstruction.depth_m
    valid: ValidMask = reconstruction.valid_mask
    shape: tuple[int, ...] = depth_m.shape
    if (
        depth_m.ndim != 2 or depth_m.dtype != np.float32
        or disparity_px.shape != shape or disparity_px.dtype != np.float32
        or valid.shape != shape or valid.dtype != np.bool_
    ):
        raise ValueError("视差、深度及掩码须为同尺寸 (H, W) 的 float32、float32 及 bool 数组")
    if not valid.any():
        raise ValueError("没有有效三维点，无法导出")

    scene_dir: Path = output_root / split / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)
    run_dir: Path = scene_dir / f"run-{uuid4().hex}"
    with TemporaryDirectory(prefix=".staging-", dir=scene_dir) as temp_dir:
        staging: Path = Path(temp_dir) / "result"
        staging.mkdir()
        np.save(staging / "disparity.npy", disparity_px, allow_pickle=False)
        np.save(staging / "depth.npy", depth_m, allow_pickle=False)
        np.save(staging / "valid_mask.npy", valid, allow_pickle=False)
        _write_png(staging / "valid_mask.png", valid.astype(np.uint8) * 255)
        _write_preview(staging / "disparity_preview.png", disparity_px, np.isfinite(disparity_px))
        _write_preview(staging / "depth_preview.png", depth_m, valid)
        point_count: int = _write_point_cloud(staging / "point_cloud.ply", reconstruction)
        _write_metrics(staging, metrics, split, scene_name, sgbm, search_roi, shape, point_count)
        staging.rename(run_dir)
    return run_dir


def _write_png(path: Path, image: NDArray[np.uint8]) -> None:
    """写入灰度 (H, W) 或 BGR (H, W, 3) PNG。"""
    success: bool
    encoded: NDArray[np.uint8]
    try:
        success, encoded = cv2.imencode(".png", image)
    except cv2.error as error:
        raise OSError(f"无法编码 PNG：{path}") from error
    if not success:
        raise OSError(f"无法编码 PNG：{path}")
    path.write_bytes(encoded.tobytes())


def _write_preview(path: Path, values: NDArray[np.float32], valid: ValidMask) -> None:
    """按本图有效像素范围着色，黑色表示无效区域。"""
    scaled: NDArray[np.uint8] = np.zeros(values.shape, dtype=np.uint8)
    if valid.any():
        selected: NDArray[np.float64] = values[valid].astype(np.float64)
        minimum: float = float(selected.min())
        span: float = float(selected.max()) - minimum
        if span > 0:
            scaled[valid] = np.rint((selected - minimum) * (255 / span)).astype(np.uint8)
    color: NDArray[np.uint8] = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    _write_png(path, color)


def _write_point_cloud(path: Path, reconstruction: ReconstructionResult) -> int:
    valid: ValidMask = reconstruction.valid_mask
    xyz_m: PointMap = reconstruction.xyz_m
    left_bgr: NDArray[np.uint8] = reconstruction.rectified_pair.left_bgr
    expected_shape: tuple[int, ...] = (*valid.shape, 3)
    if xyz_m.shape != expected_shape or xyz_m.dtype != np.float32:
        raise ValueError("三维坐标须为与掩码对齐的 float32 (H, W, 3) 数组")
    if left_bgr.shape != expected_shape or left_bgr.dtype != np.uint8:
        raise ValueError("左校正图须为与掩码对齐的 uint8 BGR 图")

    # 按行提取有效点，坐标形状 (N, 3)，深度形状 (N,)，单位米。
    points_m: NDArray[np.float32] = xyz_m[valid]
    depth_m: NDArray[np.float32] = reconstruction.depth_m[valid]
    if not np.isfinite(points_m).all() or np.any(depth_m <= 0) or not np.array_equal(
        points_m[:, 2], depth_m
    ):
        raise ValueError("有效点必须有有限坐标及与 Z 一致的正深度")
    colors_rgb: NDArray[np.uint8] = left_bgr[valid][:, ::-1]
    point_count: int = points_m.shape[0]
    header: str = (
        "ply\nformat ascii 1.0\ncomment xyz_unit meter\n"
        f"element vertex {point_count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header"
    )
    vertices: NDArray[np.float64] = np.column_stack((points_m, colors_rgb)).astype(np.float64)
    np.savetxt(path, vertices, fmt="%.9g %.9g %.9g %d %d %d", header=header, comments="")
    return point_count


def _write_metrics(
    directory: Path,
    metrics: MetricValues,
    split: MiddleburySplit,
    scene_name: str,
    sgbm: SgbmConfig,
    search_roi: ImageRoi,
    shape: tuple[int, ...],
    point_count: int,
) -> None:
    height: int
    width: int
    height, width = shape
    sample: dict[str, str | int | float | None] = dict(metrics)
    sample.update(
        split=split, scene_name=scene_name, status="ok",
        valid_point_count=point_count,
    )
    document: dict[str, object] = {
        "sample": sample,
        "image_size": [width, height],
        "search_roi_xywh": search_roi,
        "sgbm": asdict(sgbm),
        "versions": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
    }
    (directory / "metrics.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (directory / "samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer: csv.DictWriter = csv.DictWriter(stream, fieldnames=list(sample))
        writer.writeheader()
        writer.writerow(sample)
