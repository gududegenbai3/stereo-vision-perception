"""读取并校验 V0 的 JSON 配置。"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .model import MiddleburySplit, SgbmConfig


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
