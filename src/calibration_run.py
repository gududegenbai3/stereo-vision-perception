"""编排双目标定：读取左右拼图、检测标靶、求解参数、校正并导出产物。"""

import csv
import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from .calibration import (
    PairDetection,
    StereoCalibration,
    calibrate_stereo,
    detect_target_pair,
    read_bgr_image,
    save_calibration,
    split_side_by_side,
    to_gray,
)
from .config import CalibrationConfig
from .model import (
    AsymmetricCircleGridSpec,
    BgrImage,
    ChessboardSpec,
    ImageRoi,
    TargetSpec,
)
from .rectification import (
    StereoRectification,
    VerticalResidual,
    build_rectification,
    rectify_pair,
    save_rectification,
    vertical_residual,
)

_PREVIEW_ROW_WIDTH: int = 960
_PREVIEW_LINE_STEP: int = 60


@dataclass(frozen=True, eq=False)
class PairRecord:
    """一对待检图像的检测结果，用于追溯接受或拒绝的原因。"""

    name: str
    source_file: str
    usable: bool
    target: str
    left_method: str | None
    right_method: str | None
    note: str


@dataclass(frozen=True, eq=False)
class CalibrationRunResult:
    """一次标定的结果摘要，产物已写入 output_dir。"""

    calibration: StereoCalibration
    rectification: StereoRectification
    vertical: VerticalResidual
    records: tuple[PairRecord, ...]
    ndisp: int
    output_dir: Path


def run_calibration(settings: CalibrationConfig) -> CalibrationRunResult:
    """执行一次完整标定并导出产物，返回摘要。

    输入目录缺失时抛出 FileNotFoundError；有效图像对不足、尺寸不一致
    或求解失败时抛出 ValueError；写入失败时抛出 OSError。
    """
    if not settings.source_dir.is_dir():
        raise FileNotFoundError(f"标定图像目录不存在：{settings.source_dir}")
    paths: list[Path] = sorted(settings.source_dir.glob(settings.image_glob))
    if not paths:
        raise FileNotFoundError(
            f"{settings.source_dir} 中没有匹配 {settings.image_glob} 的图像"
        )

    records: list[PairRecord] = []
    detections: list[PairDetection] = []
    image_size: tuple[int, int] | None = None
    preview: tuple[BgrImage, BgrImage] | None = None
    digests: dict[str, str] = {}
    for path in paths:
        digests[path.name] = _sha256(path)
        try:
            image: BgrImage = read_bgr_image(path)
        except (OSError, ValueError) as error:
            records.append(
                PairRecord(path.stem, path.name, False, "", None, None, f"读取失败：{error}")
            )
            continue
        left: BgrImage
        right: BgrImage
        try:
            left, right = split_side_by_side(image)
        except ValueError as error:
            records.append(
                PairRecord(path.stem, path.name, False, "", None, None, f"拆分失败：{error}")
            )
            continue
        pair_size: tuple[int, int] = (left.shape[1], left.shape[0])
        if image_size is None:
            image_size = pair_size
        elif pair_size != image_size:
            records.append(
                PairRecord(
                    path.stem,
                    path.name,
                    False,
                    "",
                    None,
                    None,
                    f"单目尺寸 {pair_size} 与首帧 {image_size} 不一致",
                )
            )
            continue

        left_gray = to_gray(left)
        right_gray = to_gray(right)
        detection: PairDetection | None = None
        for target in settings.targets:
            detection = detect_target_pair(path.stem, left_gray, right_gray, target)
            if detection is not None:
                break
        if detection is None:
            tried: str = "、".join(_target_kind(target) for target in settings.targets)
            records.append(
                PairRecord(
                    path.stem,
                    path.name,
                    False,
                    "",
                    None,
                    None,
                    f"未检出标靶（已尝试：{tried}）",
                )
            )
            continue
        records.append(
            PairRecord(
                path.stem,
                path.name,
                True,
                detection.target_name,
                detection.left.method,
                detection.right.method,
                "ok",
            )
        )
        detections.append(detection)
        if preview is None:
            preview = (left, right)

    if image_size is None:
        raise ValueError(f"{settings.source_dir} 中没有可读的左右拼图")
    if len(detections) < settings.min_pairs:
        raise ValueError(
            f"有效图像对 {len(detections)} 少于要求的最小值 {settings.min_pairs}；"
            "请检查标靶规格、清晰度与左右顺序"
        )
    if preview is None:
        raise ValueError("没有可用于校正预览的图像对")

    calibration: StereoCalibration = calibrate_stereo(
        detections,
        image_size,
        baseline_m=settings.baseline_m,
        fix_k3=settings.fix_k3,
    )
    rectification: StereoRectification = build_rectification(
        calibration, alpha=settings.alpha, zero_disparity=settings.zero_disparity
    )
    vertical: VerticalResidual = vertical_residual(calibration, rectification, detections)
    ndisp: int = _recommend_ndisp(
        calibration, rectification, near_depth_m=settings.near_depth_m
    )
    coverage: ImageRoi = _corner_coverage(detections, calibration.image_size)

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path: Path = settings.output_dir / "calibration.npz"
    save_calibration(calibration_path, calibration)
    save_rectification(
        settings.output_dir / "rectification.npz",
        rectification,
        calibration_source=f"calibration.npz sha256:{_sha256(calibration_path)}",
    )
    _write_calib_text(settings.output_dir / "calib.txt", calibration, rectification, ndisp)
    _write_preview(
        settings.output_dir / "rectified_preview.png",
        rectification,
        preview[0],
        preview[1],
    )
    _write_detections(settings.output_dir / "detections.csv", records, calibration)
    _write_metrics(
        settings.output_dir / "metrics.json",
        settings=settings,
        calibration=calibration,
        rectification=rectification,
        vertical=vertical,
        records=records,
        digests=digests,
        ndisp=ndisp,
        coverage=coverage,
    )
    _write_readme(
        settings.output_dir / "README.md",
        settings=settings,
        calibration=calibration,
        rectification=rectification,
        vertical=vertical,
        records=records,
        ndisp=ndisp,
        coverage=coverage,
    )
    return CalibrationRunResult(
        calibration=calibration,
        rectification=rectification,
        vertical=vertical,
        records=tuple(records),
        ndisp=ndisp,
        output_dir=settings.output_dir,
    )


def _target_kind(target: TargetSpec) -> str:
    """标靶类型的中文简称。"""
    if isinstance(target, ChessboardSpec):
        return "棋盘格"
    if isinstance(target, AsymmetricCircleGridSpec):
        return "圆点阵列"
    return "未知标靶"


def _target_label(name: str) -> str:
    """标靶类型名称的中文标签。"""
    labels: dict[str, str] = {
        "chessboard": "棋盘格",
        "asymmetric_circle_grid": "圆点阵列",
    }
    return labels.get(name, name)


def _describe_target(target: TargetSpec) -> str:
    """标靶规格的可读描述，单位毫米。"""
    if isinstance(target, ChessboardSpec):
        return (
            f"棋盘格 {target.cols}×{target.rows} 内角点，格长 "
            f"{target.square_size_m * 1000.0:.2f} mm"
        )
    if isinstance(target, AsymmetricCircleGridSpec):
        return (
            f"圆点阵列 {target.rows} 行、最多 {target.max_per_row} 点/行，"
            f"行距 {target.row_pitch_m * 1000.0:.2f} mm、行内间距 "
            f"{target.column_pitch_m * 1000.0:.2f} mm、最近点距 "
            f"{target.spacing_m * np.sqrt(2.0) * 1000.0:.2f} mm、直径 "
            f"{target.diameter_m * 1000.0:.2f} mm"
        )
    return "未知标靶"


def _recommend_ndisp(
    calibration: StereoCalibration,
    rectification: StereoRectification,
    *,
    near_depth_m: float,
) -> int:
    """按最近深度估算视差搜索范围，向上取 16 的整数倍。"""
    disparity: float = rectification.focal_px * calibration.baseline_m / near_depth_m
    steps: int = int(np.ceil(disparity / 16.0)) * 16
    return max(16, steps)


def _corner_coverage(detections: list[PairDetection], image_size: tuple[int, int]) -> ImageRoi:
    """左图全部图像点的外接范围 (x, y, width, height)。"""
    points: NDArray[np.float32] = np.concatenate(
        [detection.left.corners_px for detection in detections], axis=0
    )
    x_min: float = float(points[:, 0].min())
    y_min: float = float(points[:, 1].min())
    x_max: float = float(points[:, 0].max())
    y_max: float = float(points[:, 1].max())
    width: int = max(0, min(image_size[0], int(round(x_max)) - int(round(x_min))))
    height: int = max(0, min(image_size[1], int(round(y_max)) - int(round(y_min))))
    return (int(round(x_min)), int(round(y_min)), width, height)


def _write_calib_text(
    path: Path,
    calibration: StereoCalibration,
    rectification: StereoRectification,
    ndisp: int,
) -> None:
    """写与旧版 7 参数文本一致的 calib.txt，基线单位为毫米。"""
    p1: NDArray[np.float64] = rectification.projection_left
    p2: NDArray[np.float64] = rectification.projection_right
    lines: list[str] = [
        f"cam0=[{p1[0, 0]:.6f} 0.000000 {p1[0, 2]:.6f}; "
        f"0.000000 {p1[1, 1]:.6f} {p1[1, 2]:.6f}; 0 0 1]",
        f"cam1=[{p2[0, 0]:.6f} 0.000000 {p2[0, 2]:.6f}; "
        f"0.000000 {p2[1, 1]:.6f} {p2[1, 2]:.6f}; 0 0 1]",
        f"doffs={rectification.doffs_px:.6f}",
        f"baseline={calibration.baseline_m * 1000.0:.4f}",
        f"width={calibration.image_size[0]}",
        f"height={calibration.image_size[1]}",
        f"ndisp={ndisp}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_preview(
    path: Path,
    rectification: StereoRectification,
    left_bgr: BgrImage,
    right_bgr: BgrImage,
) -> None:
    """把一对图校正后缩放并叠加水平参考线，便于检查极线对齐。"""
    pair = rectify_pair(rectification, left_bgr, right_bgr)
    height: int
    width: int
    height, width = pair.left_bgr.shape[:2]
    scale: float = _PREVIEW_ROW_WIDTH / float(width)
    size: tuple[int, int] = (_PREVIEW_ROW_WIDTH, max(1, int(round(height * scale))))
    left_small: BgrImage = cv2.resize(pair.left_bgr, size, interpolation=cv2.INTER_AREA)
    right_small: BgrImage = cv2.resize(pair.right_bgr, size, interpolation=cv2.INTER_AREA)
    for row in (left_small, right_small):
        for y in range(_PREVIEW_LINE_STEP, row.shape[0], _PREVIEW_LINE_STEP):
            cv2.line(row, (0, y), (row.shape[1] - 1, y), (0, 255, 0), 1, cv2.LINE_8)
    canvas: BgrImage = np.hstack((left_small, right_small))
    success: bool
    encoded: NDArray[np.uint8]
    success, encoded = cv2.imencode(".png", canvas)
    if not success:
        raise OSError(f"无法编码预览图：{path}")
    path.write_bytes(encoded.tobytes())


def _write_detections(
    path: Path,
    records: list[PairRecord],
    calibration: StereoCalibration,
) -> None:
    """写逐对检测结果，附接受配对的左右重投影误差。"""
    errors: dict[str, tuple[float, float]] = {
        error.name: (error.left_rmse_px, error.right_rmse_px)
        for error in calibration.pair_errors
    }
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer: csv.DictWriter = csv.DictWriter(
            stream,
            fieldnames=[
                "pair_id",
                "source_file",
                "usable",
                "target",
                "left_method",
                "right_method",
                "left_rmse_px",
                "right_rmse_px",
                "note",
            ],
        )
        writer.writeheader()
        for index, record in enumerate(records, start=1):
            entry: tuple[float, float] | None = errors.get(record.name)
            writer.writerow(
                {
                    "pair_id": index,
                    "source_file": record.source_file,
                    "usable": record.usable,
                    "target": record.target,
                    "left_method": record.left_method or "",
                    "right_method": record.right_method or "",
                    "left_rmse_px": "" if entry is None else f"{entry[0]:.4f}",
                    "right_rmse_px": "" if entry is None else f"{entry[1]:.4f}",
                    "note": record.note,
                }
            )


def _write_metrics(
    path: Path,
    *,
    settings: CalibrationConfig,
    calibration: StereoCalibration,
    rectification: StereoRectification,
    vertical: VerticalResidual,
    records: list[PairRecord],
    digests: dict[str, str],
    ndisp: int,
    coverage: ImageRoi,
) -> None:
    """写汇总指标与全部标定参数。"""
    used: int = len(calibration.pair_errors)
    target_counts: dict[str, int] = {
        name: calibration.pair_targets.count(name) for name in calibration.targets
    }
    document: dict[str, object] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "provisional": settings.provisional,
        "source": {
            "directory": str(settings.source_dir),
            "image_glob": settings.image_glob,
            "capture_layout": settings.capture_layout,
            "image_size_wh": list(calibration.image_size),
            "files_total": len(records),
            "pairs_used": used,
            "rejected": [
                {"name": record.name, "reason": record.note}
                for record in records
                if not record.usable
            ],
            "sha256": digests,
        },
        "targets": {
            "configured": [_describe_target(target) for target in settings.targets],
            "used": target_counts,
        },
        "board": {
            "primary": _describe_target(settings.targets[0]),
            "fix_k3": calibration.fix_k3,
        },
        "calibration": {
            "rms_stereo_px": calibration.rms_px,
            "rms_left_px": calibration.left.rms_px,
            "rms_right_px": calibration.right.rms_px,
            "baseline_mm": calibration.baseline_m * 1000.0,
            "scale_source": calibration.scale_source,
            "per_pair": [
                {
                    "pair": error.name,
                    "target": error.target_name,
                    "left_rmse_px": error.left_rmse_px,
                    "right_rmse_px": error.right_rmse_px,
                }
                for error in calibration.pair_errors
            ],
            "K1": calibration.left.matrix.tolist(),
            "D1": calibration.left.distortion.ravel().tolist(),
            "K2": calibration.right.matrix.tolist(),
            "D2": calibration.right.distortion.ravel().tolist(),
            "R": calibration.rotation.tolist(),
            "T_m": calibration.translation_m.tolist(),
            "essential": calibration.essential.tolist(),
            "fundamental": calibration.fundamental.tolist(),
        },
        "rectification": {
            "alpha": rectification.alpha,
            "zero_disparity": rectification.zero_disparity,
            "focal_px": rectification.focal_px,
            "cx_left_px": float(rectification.projection_left[0, 2]),
            "cx_right_px": float(rectification.projection_right[0, 2]),
            "cy_px": float(rectification.projection_left[1, 2]),
            "doffs_px": rectification.doffs_px,
            "roi_left_xywh": list(rectification.roi_left),
            "roi_right_xywh": list(rectification.roi_right),
            "recommended_ndisp": ndisp,
            "near_depth_m": settings.near_depth_m,
            "Q": rectification.q.tolist(),
        },
        "vertical_residual_px": {
            "mean": vertical.mean_px,
            "p95": vertical.p95_px,
            "max": vertical.max_px,
            "sample_count": vertical.sample_count,
            "note": "在标定图自身角点上统计，不是独立验证集",
        },
        "corner_coverage_xywh": list(coverage),
        "versions": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
    }
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_readme(
    path: Path,
    *,
    settings: CalibrationConfig,
    calibration: StereoCalibration,
    rectification: StereoRectification,
    vertical: VerticalResidual,
    records: list[PairRecord],
    ndisp: int,
    coverage: ImageRoi,
) -> None:
    """生成便于阅读的标定说明与质量摘要。"""
    rejected: list[PairRecord] = [record for record in records if not record.usable]
    rejected_text: str = (
        "\n".join(f"- `{record.source_file}`：{record.note}" for record in rejected)
        if rejected
        else "无"
    )
    target_lines: str = "\n".join(
        f"- {_describe_target(target)}" for target in settings.targets
    )
    used_summary: str = "、".join(
        f"{_target_label(name)} {count} 对"
        for name, count in (
            (name, calibration.pair_targets.count(name)) for name in calibration.targets
        )
    )
    scale_lines: list[str]
    if calibration.scale_source == "baseline_override":
        scale_lines = [
            f"- 米制尺度由**声明基线** {calibration.baseline_m * 1000.0:.2f} mm 决定；"
            "标靶尺寸只用于构造对象点，不改变最终结果。",
            "- 声明基线应来自光心间距实测；与标靶尺寸推算值不一致时，深度会按两者比值整体缩放。",
        ]
    else:
        scale_lines = [
            "- 米制尺度由**标靶实测尺寸**决定，基线由 `T` 推出："
            f"**{calibration.baseline_m * 1000.0:.2f} mm**。",
            "- 标靶尺寸每偏差 1%，深度也偏差 1%；换间距或动过相机后必须重新标定。",
        ]
    lines: list[str] = [
        f"# 双目标定结果（{settings.output_dir.name}）",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"数据：`{settings.source_dir}`，{len(records)} 幅左右拼图，"
        f"按中线拆分为 {calibration.image_size[0]}×{calibration.image_size[1]} 单目图。",
        "",
        "标靶：",
        target_lines,
        "",
        "## 质量",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 有效图像对 | {len(calibration.pair_errors)} / {len(records)}（{used_summary}） |",
        f"| 单目 RMS（左 / 右） | {calibration.left.rms_px:.3f} / "
        f"{calibration.right.rms_px:.3f} px |",
        f"| 双目 RMS | {calibration.rms_px:.3f} px |",
        f"| 校正后垂直残差 mean / P95 / max | {vertical.mean_px:.3f} / "
        f"{vertical.p95_px:.3f} / {vertical.max_px:.3f} px |",
        f"| 图像点覆盖范围 (x, y, w, h) | {coverage} |",
        f"| 基线（尺度来源：{calibration.scale_source}） | "
        f"{calibration.baseline_m * 1000.0:.2f} mm |",
        "",
        "垂直残差在标定图自身统计，不是独立验证集，结果偏乐观。",
        "",
        "## 尺度",
        "",
        *scale_lines,
        "",
        "## 原始内参与相对姿态（标定坐标，未校正）",
        "",
        "```",
        f"K1={np.array2string(calibration.left.matrix, precision=4, suppress_small=True)}",
        f"D1={np.array2string(calibration.left.distortion.ravel(), precision=5)}",
        f"K2={np.array2string(calibration.right.matrix, precision=4, suppress_small=True)}",
        f"D2={np.array2string(calibration.right.distortion.ravel(), precision=5)}",
        f"R={np.array2string(calibration.rotation, precision=6, suppress_small=True)}",
        f"T_m={np.array2string(calibration.translation_m, precision=6)}",
        "```",
        "",
        "## 参数（校正后，零视差）",
        "",
        "```",
        f"cam0=[{rectification.projection_left[0, 0]:.6f} 0 {rectification.projection_left[0, 2]:.6f}; "
        f"0 {rectification.projection_left[1, 1]:.6f} {rectification.projection_left[1, 2]:.6f}; 0 0 1]",
        f"cam1=[{rectification.projection_right[0, 0]:.6f} 0 {rectification.projection_right[0, 2]:.6f}; "
        f"0 {rectification.projection_right[1, 1]:.6f} {rectification.projection_right[1, 2]:.6f}; 0 0 1]",
        f"doffs={rectification.doffs_px:.6f}",
        f"baseline={calibration.baseline_m * 1000.0:.4f}",
        f"width={calibration.image_size[0]}",
        f"height={calibration.image_size[1]}",
        f"ndisp={ndisp}",
        "```",
        "",
        "深度公式：`Z = f·B / (d + doffs)`，零视差校正下 `doffs = 0`，",
        "`Q` 可直接用于 `cv2.reprojectImageTo3D`。",
        "",
        "## 未采用或未能检出的图像",
        "",
        rejected_text,
        "",
        "## 文件",
        "",
        "- `calibration.npz`：K1、D1、K2、D2、R、T、E、F、尺寸、基线、逐对误差与标靶类型。",
        "- `rectification.npz`：R1、R2、P1、P2、Q、ROI、重映射表与标定文件摘要。",
        "- `calib.txt`：七参数文本，可直接被旧脚本读取。",
        "- `metrics.json`：全部指标、参数与环境版本。",
        "- `detections.csv`：逐图标靶类型、检测方式与重投影误差。",
        "- `rectified_preview.png`：校正后的左右预览，叠加水平参考线。",
    ]
    if settings.provisional:
        lines.insert(
            8,
            "> 本次结果标记为 provisional：部分尺寸或基线尚未实测，见「尺度」一节。",
        )
        lines.insert(9, "")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    """计算文件 SHA256 摘要。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
