"""读取并校验 V0 的 JSON 配置。"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .model import (
    AsymmetricCircleGridSpec,
    ChessboardSpec,
    MiddleburySplit,
    SgbmConfig,
    TargetSpec,
)


@dataclass(frozen=True)
class OfflineConfig:
    """Middlebury 离线实验参数，路径已转换为绝对路径。"""

    image_root: Path
    split: MiddleburySplit
    scene_name: str
    ground_truth_root: Path | None
    sgbm: SgbmConfig
    output_root: Path | None = None


@dataclass(frozen=True)
class CalibrationConfig:
    """双目标定运行参数，路径已转换为绝对路径。

    source_dir 存放左右拼图，capture_layout 描述拆分方式。targets 按顺序
    尝试，先命中的标靶类型生效。baseline_m 非空时按该基线重设平移尺度，
    否则使用标靶实测尺寸。near_depth_m 只用于推荐视差搜索范围。
    """

    source_dir: Path
    output_dir: Path
    image_glob: str
    capture_layout: str
    targets: tuple[TargetSpec, ...]
    fix_k3: bool
    baseline_m: float | None
    min_pairs: int
    alpha: float
    zero_disparity: bool
    near_depth_m: float
    provisional: bool


@dataclass(frozen=True)
class HardwareConfig:
    """运行前需要的设备身份与状态，未知基线保留为 None。"""

    rig_id: str
    baseline_m: float | None
    baseline_locked: bool
    left_right_layout: str
    calibration_status: str


def load_offline_config(
    path: Path,
    *,
    image_root_override: Path | None = None,
    ground_truth_root_override: Path | None = None,
    split_override: str | None = None,
    scene_override: str | None = None,
) -> OfflineConfig:
    """加载离线配置，数据相对路径以配置文件所在目录为准。

    显式覆盖值优先，覆盖路径相对于当前工作目录。
    文件不存在时抛出 FileNotFoundError。JSON、字段或参数无效时抛出
    ValueError。数据文件是否存在及其内容由数据读取层检查。
    """
    data: dict[str, object] = _read_json(path)
    if image_root_override is not None:
        data["image_root"] = str(image_root_override.resolve())
    if ground_truth_root_override is not None:
        data["ground_truth_root"] = str(ground_truth_root_override.resolve())
    if split_override is not None:
        data["split"] = split_override
    if scene_override is not None:
        data["scene_name"] = scene_override

    try:
        split_value: str = _required_text(data["split"], "split")
        if split_value not in ("trainingQ", "testQ"):
            raise ValueError("split 必须为 trainingQ 或 testQ")
        split: MiddleburySplit = cast(MiddleburySplit, split_value)

        scene_name: str = _required_text(data["scene_name"], "scene_name")
        if scene_name in (".", "..") or any(char in scene_name for char in "/\\\0"):
            raise ValueError("scene_name 必须是单个目录名")

        base_dir: Path = path.resolve().parent
        image_root: Path = _resolve_path(base_dir, data["image_root"], "image_root")
        truth_value: object = data.get("ground_truth_root")
        ground_truth_root: Path | None = None
        if truth_value is not None:
            ground_truth_root = _resolve_path(base_dir, truth_value, "ground_truth_root")
        if split == "trainingQ" and ground_truth_root is None:
            raise ValueError("trainingQ 必须指定 ground_truth_root")

        sgbm: SgbmConfig = _parse_sgbm(_require_object(data["sgbm"], "sgbm"))
        output_value: object = data.get("output_root")
        output_root: Path | None = (
            _resolve_path(base_dir, output_value, "output_root")
            if output_value is not None
            else None
        )
    except (KeyError, ValueError) as error:
        raise ValueError(f"{path}：离线配置无效（{error}）") from error

    return OfflineConfig(
        image_root=image_root,
        split=split,
        scene_name=scene_name,
        ground_truth_root=ground_truth_root,
        sgbm=sgbm,
        output_root=output_root,
    )


def load_calibration_config(
    path: Path,
    *,
    square_size_mm_override: float | None = None,
    baseline_mm_override: float | None = None,
    output_dir_override: Path | None = None,
) -> CalibrationConfig:
    """加载双目标定配置，路径相对配置文件所在目录解析。

    显式覆盖值优先，覆盖路径相对当前工作目录。文件不存在时抛出
    FileNotFoundError；JSON、字段或参数无效时抛出 ValueError。
    """
    data: dict[str, object] = _read_json(path)
    if square_size_mm_override is not None:
        data["targets"] = _override_square_size(data["targets"], float(square_size_mm_override))
    if baseline_mm_override is not None:
        data["baseline_mm"] = float(baseline_mm_override)
    if output_dir_override is not None:
        data["output_dir"] = str(output_dir_override.resolve())

    try:
        base_dir: Path = path.resolve().parent
        source_dir: Path = _resolve_path(base_dir, data["source_dir"], "source_dir")
        output_dir: Path = _resolve_path(base_dir, data["output_dir"], "output_dir")
        image_glob: str = _required_text(data["image_glob"], "image_glob")
        if any(char in image_glob for char in "/\\\0"):
            raise ValueError("image_glob 只能是文件名模式，不能包含路径分隔符")
        layout: str = _required_text(data["capture_layout"], "capture_layout")
        if layout != "side_by_side":
            raise ValueError("capture_layout 目前只支持 side_by_side")
        targets: tuple[TargetSpec, ...] = _parse_targets(data["targets"])
        distortion: dict[str, object] = _require_object(data["distortion"], "distortion")
        fix_k3: object = distortion["fix_k3"]
        if not isinstance(fix_k3, bool):
            raise ValueError("distortion.fix_k3 必须为布尔值")
        baseline_value: object = data.get("baseline_mm")
        baseline_m: float | None = None
        if baseline_value is not None:
            if not isinstance(baseline_value, (int, float)) or isinstance(baseline_value, bool):
                raise ValueError("baseline_mm 必须为正数或 null")
            baseline_m = float(baseline_value) / 1000.0
            if not math.isfinite(baseline_m) or baseline_m <= 0:
                raise ValueError("baseline_mm 必须为有限正数或 null")
        min_pairs: object = data["min_pairs"]
        if type(min_pairs) is not int or min_pairs < 3:
            raise ValueError("min_pairs 必须是不小于 3 的整数")
        rectify: dict[str, object] = _require_object(data["rectify"], "rectify")
        alpha: object = rectify["alpha"]
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
            raise ValueError("rectify.alpha 必须为数字")
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("rectify.alpha 必须位于 [0, 1]")
        zero_disparity: object = rectify["zero_disparity"]
        if not isinstance(zero_disparity, bool):
            raise ValueError("rectify.zero_disparity 必须为布尔值")
        near_depth: object = data.get("near_depth_m", 0.4)
        if not isinstance(near_depth, (int, float)) or isinstance(near_depth, bool):
            raise ValueError("near_depth_m 必须为数字")
        if not math.isfinite(float(near_depth)) or float(near_depth) <= 0:
            raise ValueError("near_depth_m 必须为有限正数")
        provisional: object = data.get("provisional", True)
        if not isinstance(provisional, bool):
            raise ValueError("provisional 必须为布尔值")
    except (KeyError, ValueError) as error:
        raise ValueError(f"{path}：标定配置无效（{error}）") from error

    return CalibrationConfig(
        source_dir=source_dir,
        output_dir=output_dir,
        image_glob=image_glob,
        capture_layout=layout,
        targets=targets,
        fix_k3=fix_k3,
        baseline_m=baseline_m,
        min_pairs=min_pairs,
        alpha=float(alpha),
        zero_disparity=zero_disparity,
        near_depth_m=float(near_depth),
        provisional=provisional,
    )


def _parse_targets(value: object) -> tuple[TargetSpec, ...]:
    """解析标靶列表，支持棋盘格与非对称圆点阵列。"""
    if not isinstance(value, list) or not value:
        raise ValueError("targets 必须是非空数组")
    targets: list[TargetSpec] = []
    for index, item in enumerate(value):
        entry: dict[str, object] = _require_object(item, f"targets[{index}]")
        kind: str = _required_text(entry["type"], f"targets[{index}].type")
        try:
            if kind == "chessboard":
                targets.append(_parse_chessboard(entry))
            elif kind == "asymmetric_circle_grid":
                targets.append(_parse_circle_grid(entry))
            else:
                raise ValueError(f"不支持的标靶类型 {kind!r}")
        except ValueError as error:
            raise ValueError(f"targets[{index}] 无效（{error}）") from error
    return tuple(targets)


def _parse_chessboard(data: dict[str, object]) -> ChessboardSpec:
    cols: object = data["cols"]
    rows: object = data["rows"]
    square_mm: object = data["square_size_mm"]
    if type(cols) is not int or type(rows) is not int:
        raise ValueError("cols 与 rows 必须为整数")
    if not isinstance(square_mm, (int, float)) or isinstance(square_mm, bool):
        raise ValueError("square_size_mm 必须为数字")
    if not math.isfinite(float(square_mm)) or float(square_mm) <= 0:
        raise ValueError("square_size_mm 必须为有限正数")
    return ChessboardSpec(cols=cols, rows=rows, square_size_m=float(square_mm) / 1000.0)


def _parse_circle_grid(data: dict[str, object]) -> AsymmetricCircleGridSpec:
    rows: object = data["rows"]
    per_row: object = data["max_per_row"]
    spacing_mm: object = data["spacing_mm"]
    diameter_mm: object = data["diameter_mm"]
    if type(rows) is not int or type(per_row) is not int:
        raise ValueError("rows 与 max_per_row 必须为整数")
    for name, value in (("spacing_mm", spacing_mm), ("diameter_mm", diameter_mm)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} 必须为数字")
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} 必须为有限正数")
    return AsymmetricCircleGridSpec(
        rows=rows,
        max_per_row=per_row,
        spacing_m=float(spacing_mm) / 1000.0,
        diameter_m=float(diameter_mm) / 1000.0,
    )


def _override_square_size(value: object, square_size_mm: float) -> list[object]:
    """把棋盘格格长覆盖值应用到全部棋盘格标靶。"""
    if not isinstance(value, list):
        raise ValueError("targets 必须是非空数组")
    updated: list[object] = []
    matched: bool = False
    for item in value:
        entry: dict[str, object] = _require_object(item, "targets")
        if entry.get("type") == "chessboard":
            entry = dict(entry)
            entry["square_size_mm"] = square_size_mm
            matched = True
        updated.append(entry)
    if not matched:
        raise ValueError("配置中没有棋盘格标靶，无法使用 --square-mm")
    return updated


def load_hardware_config(path: Path) -> HardwareConfig:
    """读取硬件档案中的运行相关字段，将毫米基线换算为米。

    文件不存在时抛出 FileNotFoundError。JSON 或所需字段无效时抛出
    ValueError。设备规格和备注由硬件档案保留。
    """
    data: dict[str, object] = _read_json(path)
    try:
        stereo: dict[str, object] = _require_object(data["stereo"], "stereo")
        calibration: dict[str, object] = _require_object(data["calibration"], "calibration")
        rig_id: str = _required_text(data["rig_id"], "rig_id")
        if data.get("schema_version") == "2.1.0":
            topology: object = stereo["device_topology"]
            layout: str = (
                "unknown" if topology is None else _required_text(topology, "stereo.device_topology")
            )
        else:
            layout = _required_text(stereo["left_right_layout"], "stereo.left_right_layout")
        status: str = _required_text(calibration["status"], "calibration.status")

        baseline_locked: object = stereo["baseline_locked"]
        if not isinstance(baseline_locked, bool):
            raise ValueError("stereo.baseline_locked 必须为布尔值")

        baseline_mm: object = stereo["baseline_mm"]
        baseline_m: float | None = None
        if baseline_mm is not None:
            if not isinstance(baseline_mm, (int, float)) or isinstance(baseline_mm, bool):
                raise ValueError("stereo.baseline_mm 必须为正数或 null")
            baseline_m = baseline_mm / 1000.0
            if not math.isfinite(baseline_m) or baseline_m <= 0:
                raise ValueError("stereo.baseline_mm 必须为有限正数或 null")
    except (KeyError, ValueError, OverflowError) as error:
        raise ValueError(f"{path}：硬件配置无效（{error}）") from error

    return HardwareConfig(rig_id, baseline_m, baseline_locked, layout, status)


def _read_json(path: Path) -> dict[str, object]:
    try:
        data: object = json.loads(path.read_text(encoding="utf-8"))
        return _require_object(data, "JSON 顶层")
    except ValueError as error:
        raise ValueError(f"{path}：无法读取 JSON 对象（{error}）") from error


def _require_object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} 必须为 JSON 对象")
    return value


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} 必须为非空字符串")
    return value


def _resolve_path(base_dir: Path, value: object, field: str) -> Path:
    path_text: str = _required_text(value, field)
    if "\0" in path_text:
        raise ValueError(f"{field} 不能包含空字符")
    return (base_dir / path_text).resolve()


def _parse_sgbm(data: dict[str, object]) -> SgbmConfig:
    min_disparity: object = data["min_disparity"]
    num_disparities: object = data["num_disparities"]
    block_size: object = data["block_size"]
    if type(min_disparity) is not int:
        raise ValueError("sgbm.min_disparity 必须为整数")
    if type(num_disparities) is not int or num_disparities <= 0 or num_disparities % 16:
        raise ValueError("sgbm.num_disparities 必须为 16 的正整数倍")
    if type(block_size) is not int or block_size <= 0 or block_size % 2 == 0:
        raise ValueError("sgbm.block_size 必须为正奇数")
    return SgbmConfig(min_disparity, num_disparities, block_size)
