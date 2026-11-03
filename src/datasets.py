"""读取 Middlebury Stereo v3 四分之一分辨率场景。"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from .model import (
    DisparityMap,
    ImageSize,
    MiddleburySplit,
    RectificationGeometry,
    RectifiedPair,
    ReprojectionMatrix,
    ValidMask,
)


@dataclass(frozen=True, eq=False)
class MiddleburyGroundTruth:
    """左图真值和两个 (H, W) 布尔掩码。

    disparity_px 为 float32 像素视差，未知值为正无穷。
    valid_mask 包含所有已知视差，non_occluded_mask 只包含双侧可见的点。
    """

    disparity_px: DisparityMap
    valid_mask: ValidMask
    non_occluded_mask: ValidMask


@dataclass(frozen=True, eq=False)
class MiddleburyScene:
    """已校正的 BGR 图像对、米制重建几何与可选真值。"""

    scene_name: str
    split: MiddleburySplit
    rectified_pair: RectifiedPair
    geometry: RectificationGeometry
    disparity_offset_px: float
    baseline_m: float
    disparity_search_hint_px: int
    ground_truth: MiddleburyGroundTruth | None


def load_middlebury_scene(
    image_root: Path,
    split: MiddleburySplit,
    scene_name: str,
    *,
    ground_truth_root: Path | None = None,
) -> MiddleburyScene:
    """读取指定场景，根目录下按 split / scene_name 查找文件。

    trainingQ 要求提供真值根目录，testQ 返回 ground_truth=None。
    缺失场景或文件抛出 FileNotFoundError，格式、尺寸或参数错误抛出 ValueError。
    """
    if split not in ("trainingQ", "testQ"):
        raise ValueError(f"不支持的 Middlebury 划分：{split!r}")
    if (
        not scene_name.strip()
        or scene_name in (".", "..")
        or any(separator in scene_name for separator in ("/", "\\", "\0"))
    ):
        raise ValueError(f"场景名必须是单个目录名：{scene_name!r}")
    if split == "trainingQ" and ground_truth_root is None:
        raise ValueError("读取 trainingQ 必须提供 ground_truth_root")

    scene_dir: Path = image_root / split / scene_name
    geometry: RectificationGeometry
    disparity_offset_px: float
    baseline_m: float
    search_hint_px: int
    geometry, disparity_offset_px, baseline_m, search_hint_px = _read_calibration(
        scene_dir / "calib.txt"
    )
    width: int
    height: int
    width, height = geometry.image_size
    pair: RectifiedPair = RectifiedPair(
        left_bgr=_read_png(scene_dir / "im0.png", (height, width, 3)),
        right_bgr=_read_png(scene_dir / "im1.png", (height, width, 3)),
    )

    ground_truth: MiddleburyGroundTruth | None = None
    if split == "trainingQ" and ground_truth_root is not None:
        ground_truth = _read_ground_truth(
            ground_truth_root / split / scene_name, geometry.image_size
        )
    return MiddleburyScene(
        scene_name, split, pair, geometry, disparity_offset_px, baseline_m,
        search_hint_px, ground_truth,
    )


def _read_calibration(
    path: Path,
) -> tuple[RectificationGeometry, float, float, int]:
    values: dict[str, str] = {}
    try:
        line: str
        key: str
        value: str
        for line in path.read_text(encoding="ascii").splitlines():
            if not line.strip():
                continue
            key, value = (part.strip() for part in line.split("=", 1))
            if key in values:
                raise ValueError(f"重复字段 {key!r}")
            values[key] = value

        cam_left: NDArray[np.float64] = _parse_camera(values["cam0"])
        cam_right: NDArray[np.float64] = _parse_camera(values["cam1"])
        width: int = int(values["width"])
        height: int = int(values["height"])
        search_hint_px: int = int(values["ndisp"])
        baseline_m: float = float(values["baseline"]) / 1000.0
        disparity_offset_px: float = float(values["doffs"])
    except (KeyError, ValueError) as error:
        raise ValueError(f"{path}：标定格式无效（{error}）") from error

    if min(width, height, search_hint_px) <= 0:
        raise ValueError(f"{path}：图像尺寸和 ndisp 必须为正整数")
    if not np.isfinite([baseline_m, disparity_offset_px]).all() or baseline_m <= 0:
        raise ValueError(f"{path}：基线必须为有限正数，doffs 必须有限")

    focal_px: float = float(cam_left[0, 0])
    cx_left_px: float = float(cam_left[0, 2])
    cx_right_px: float = float(cam_right[0, 2])
    cy_px: float = float(cam_left[1, 2])
    expected_left: NDArray[np.float64] = np.array(
        [[focal_px, 0, cx_left_px], [0, focal_px, cy_px], [0, 0, 1]]
    )
    expected_right: NDArray[np.float64] = expected_left.copy()
    expected_right[0, 2] = cx_right_px
    if (
        focal_px <= 0
        or not np.array_equal(cam_left, expected_left)
        or not np.array_equal(cam_right, expected_right)
        or not np.isclose(disparity_offset_px, cx_right_px - cx_left_px, rtol=0, atol=0.002)
    ):
        raise ValueError(f"{path}：内参或 doffs 不符合 Middlebury 水平校正模型")

    # 齐次除数为 (d + doffs) / baseline_m。数据集没有提供单独的校正 ROI。
    q: ReprojectionMatrix = np.array(
        [
            [1, 0, 0, -cx_left_px],
            [0, 1, 0, -cy_px],
            [0, 0, 0, focal_px],
            [0, 0, 1 / baseline_m, disparity_offset_px / baseline_m],
        ],
        dtype=np.float64,
    )
    if not np.isfinite(q).all():
        raise ValueError(f"{path}：校正几何超出浮点表示范围")
    roi: tuple[int, int, int, int] = (0, 0, width, height)
    geometry: RectificationGeometry = RectificationGeometry((width, height), q, roi, roi)
    return geometry, disparity_offset_px, baseline_m, search_hint_px


def _parse_camera(value: str) -> NDArray[np.float64]:
    if not (value.startswith("[") and value.endswith("]")):
        raise ValueError("内参矩阵缺少方括号")
    matrix: NDArray[np.float64] = np.array(
        [row.split() for row in value[1:-1].split(";")], dtype=np.float64
    )
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("内参必须为有限的 3×3 矩阵")
    return matrix


def _read_png(path: Path, expected_shape: tuple[int, ...]) -> NDArray[np.uint8]:
    encoded: bytes = path.read_bytes()
    if not encoded.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"{path}：文件不是 PNG")
    try:
        image: NDArray[np.uint8] | None = cv2.imdecode(
            np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
    except cv2.error as error:
        raise ValueError(f"{path}：无法解码 PNG") from error
    if image is None:
        raise ValueError(f"{path}：无法解码 PNG")
    if image.dtype != np.uint8 or image.shape != expected_shape:
        raise ValueError(f"{path}：期望 uint8 {expected_shape}，实际为 {image.dtype} {image.shape}")
    return image


def _read_ground_truth(scene_dir: Path, image_size: ImageSize) -> MiddleburyGroundTruth:
    width: int
    height: int
    width, height = image_size
    disparity_px: DisparityMap = _read_disparity(scene_dir / "disp0GT.pfm", image_size)
    mask_path: Path = scene_dir / "mask0nocc.png"
    mask: NDArray[np.uint8] = _read_png(mask_path, (height, width))
    valid_mask: ValidMask = np.isfinite(disparity_px)
    if not np.isin(mask, (0, 128, 255)).all():
        raise ValueError(f"{mask_path}：掩码只允许 0、128、255")
    if not np.array_equal(mask != 0, valid_mask):
        raise ValueError(f"{mask_path}：掩码标记的真值范围与 disp0GT.pfm 不一致")
    non_occluded_mask: ValidMask = mask == 255
    return MiddleburyGroundTruth(disparity_px, valid_mask, non_occluded_mask)


def _read_disparity(path: Path, image_size: ImageSize) -> DisparityMap:
    with path.open("rb") as stream:
        try:
            if stream.readline(128).strip() != b"Pf":
                raise ValueError("要求单通道 PFM（Pf）")
            width: int
            height: int
            width, height = map(int, stream.readline(128).split())
            scale: float = float(stream.readline(128))
            if not np.isfinite(scale) or scale == 0:
                raise ValueError("PFM 比例字段必须有限且非零")
        except ValueError as error:
            raise ValueError(f"{path}：PFM 头无效（{error}）") from error

        if (width, height) != image_size:
            raise ValueError(f"{path}：PFM 尺寸 {(width, height)} 与标定 {image_size} 不一致")
        byte_count: int = width * height * np.dtype(np.float32).itemsize
        payload: bytes = stream.read(byte_count + 1)
        if len(payload) != byte_count:
            raise ValueError(f"{path}：PFM 像素数据长度不符")

    # Middlebury 的像素值已经是 px，scale 的符号只用于选择字节序。
    byte_order: str = "<f4" if scale < 0 else ">f4"
    stored_rows: DisparityMap = np.frombuffer(payload, dtype=byte_order).reshape(height, width)
    disparity_px: DisparityMap = stored_rows[::-1].astype(np.float32)
    if np.isnan(disparity_px).any() or np.isneginf(disparity_px).any():
        raise ValueError(f"{path}：无效视差只允许用正无穷表示")
    return disparity_px
