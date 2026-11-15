"""极线校正：构建映射表、重投影矩阵与校正后垂直残差评价。"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from .calibration import PairDetection, StereoCalibration
from .model import BgrImage, ImageRoi, ImageSize, RectifiedPair, ReprojectionMatrix

# (3, 3) 校正旋转矩阵。
Matrix3: TypeAlias = NDArray[np.float64]
# (3, 4) 校正投影矩阵。
ProjectionMatrix: TypeAlias = NDArray[np.float64]
# (H, W) 重映射表。
RemapTable: TypeAlias = NDArray[np.float32]

_RECTIFICATION_SCHEMA: str = "1.0"


@dataclass(frozen=True, eq=False)
class VerticalResidual:
    """校正后同名角点的垂直方向残差，单位为像素。"""

    mean_px: float
    p95_px: float
    max_px: float
    sample_count: int


@dataclass(frozen=True, eq=False)
class StereoRectification:
    """一次极线校正的全部几何与映射表。

    左图为参考图，q 把 [x, y, disparity, 1] 映射到三维齐次坐标，
    disparity 是左图参考视差，零视差模式下无主点偏移。
    """

    image_size: ImageSize
    rotation_left: Matrix3
    rotation_right: Matrix3
    projection_left: ProjectionMatrix
    projection_right: ProjectionMatrix
    q: ReprojectionMatrix
    roi_left: ImageRoi
    roi_right: ImageRoi
    map_left_x: RemapTable
    map_left_y: RemapTable
    map_right_x: RemapTable
    map_right_y: RemapTable
    alpha: float
    zero_disparity: bool

    @property
    def doffs_px(self) -> float:
        """左右主点在视差轴上的偏移，零视差模式为 0。"""
        return float(self.projection_right[0, 2] - self.projection_left[0, 2])

    @property
    def focal_px(self) -> float:
        """校正后的等效焦距。"""
        return float(self.projection_left[0, 0])


def build_rectification(
    calibration: StereoCalibration,
    *,
    alpha: float = 0.0,
    zero_disparity: bool = True,
) -> StereoRectification:
    """由标定结果构建极线校正与重映射表。

    alpha 取 0 时只保留有效像素（无黑边），取 1 时保留全部源像素
    （出现黑边）；超出 [0, 1] 或标定尺寸非法时抛出 ValueError。
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha 必须位于 [0, 1]")
    width: int
    height: int
    width, height = calibration.image_size
    flags: int = cv2.CALIB_ZERO_DISPARITY if zero_disparity else 0
    rectified: tuple[
        Matrix3,
        Matrix3,
        ProjectionMatrix,
        ProjectionMatrix,
        ReprojectionMatrix,
        tuple[int, int, int, int],
        tuple[int, int, int, int],
    ] = cv2.stereoRectify(
        calibration.left.matrix,
        calibration.left.distortion,
        calibration.right.matrix,
        calibration.right.distortion,
        calibration.image_size,
        calibration.rotation,
        # OpenCV 5 的 stereoRectify 要求平移为 (3, 1)，(3,) 会在 gemm 中断言失败。
        calibration.translation_m.reshape(3, 1),
        flags=flags,
        alpha=float(alpha),
    )
    rotation_left, rotation_right, projection_left, projection_right, q, roi_left, roi_right = (
        rectified
    )
    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        calibration.left.matrix,
        calibration.left.distortion,
        rotation_left,
        projection_left[:3, :3],
        calibration.image_size,
        cv2.CV_32FC1,
    )
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        calibration.right.matrix,
        calibration.right.distortion,
        rotation_right,
        projection_right[:3, :3],
        calibration.image_size,
        cv2.CV_32FC1,
    )
    return StereoRectification(
        image_size=(width, height),
        rotation_left=rotation_left.astype(np.float64),
        rotation_right=rotation_right.astype(np.float64),
        projection_left=projection_left.astype(np.float64),
        projection_right=projection_right.astype(np.float64),
        q=q.astype(np.float64),
        roi_left=_as_roi(roi_left),
        roi_right=_as_roi(roi_right),
        map_left_x=map_left_x,
        map_left_y=map_left_y,
        map_right_x=map_right_x,
        map_right_y=map_right_y,
        alpha=float(alpha),
        zero_disparity=bool(zero_disparity),
    )


def rectify_pair(
    rectification: StereoRectification,
    left_bgr: BgrImage,
    right_bgr: BgrImage,
) -> RectifiedPair:
    """按映射表校正一对原始图像。

    输入必须是 (H, W, 3) uint8 BGR，且与校正尺寸一致；否则抛出
    ValueError。返回同尺寸的左右校正图。
    """
    width: int
    height: int
    width, height = rectification.image_size
    for image, tag in ((left_bgr, "左图"), (right_bgr, "右图")):
        if image.shape != (height, width, 3) or image.dtype != np.uint8:
            raise ValueError(f"{tag}必须是 uint8 BGR，形状 {height}x{width}x3")
    left: BgrImage = cv2.remap(
        left_bgr, rectification.map_left_x, rectification.map_left_y, cv2.INTER_LINEAR
    )
    right: BgrImage = cv2.remap(
        right_bgr, rectification.map_right_x, rectification.map_right_y, cv2.INTER_LINEAR
    )
    return RectifiedPair(left_bgr=left, right_bgr=right)


def vertical_residual(
    calibration: StereoCalibration,
    rectification: StereoRectification,
    detections: Sequence[PairDetection],
) -> VerticalResidual:
    """计算标定角点在校正图上的垂直残差绝对值统计。

    残差来自标定图自身，不是独立验证集，偏乐观。
    """
    residuals: list[NDArray[np.float64]] = []
    for detection in detections:
        left: NDArray[np.float32] = cv2.undistortPoints(
            detection.left.corners_px[None],
            calibration.left.matrix,
            calibration.left.distortion,
            R=rectification.rotation_left,
            P=rectification.projection_left,
        ).reshape(-1, 2)
        right: NDArray[np.float32] = cv2.undistortPoints(
            detection.right.corners_px[None],
            calibration.right.matrix,
            calibration.right.distortion,
            R=rectification.rotation_right,
            P=rectification.projection_right,
        ).reshape(-1, 2)
        if left.shape != right.shape:
            raise ValueError(f"{detection.name}：左右角点数量不一致")
        residuals.append(np.abs(left[:, 1] - right[:, 1]).astype(np.float64))
    if not residuals:
        raise ValueError("没有可用于垂直残差统计的图像对")
    values: NDArray[np.float64] = np.concatenate(residuals)
    return VerticalResidual(
        mean_px=float(values.mean()),
        p95_px=float(np.percentile(values, 95)),
        max_px=float(values.max()),
        sample_count=int(values.size),
    )


def save_rectification(path: Path, rectification: StereoRectification, *, calibration_source: str) -> None:
    """保存校正几何与映射表，并记录标定来源以便追溯。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema=np.array(_RECTIFICATION_SCHEMA),
        calibration_source=np.array(calibration_source),
        image_size_wh=np.array(rectification.image_size, dtype=np.int64),
        R1=rectification.rotation_left,
        R2=rectification.rotation_right,
        P1=rectification.projection_left,
        P2=rectification.projection_right,
        Q=rectification.q,
        roi1=np.array(rectification.roi_left, dtype=np.int64),
        roi2=np.array(rectification.roi_right, dtype=np.int64),
        map1x=rectification.map_left_x,
        map1y=rectification.map_left_y,
        map2x=rectification.map_right_x,
        map2y=rectification.map_right_y,
        alpha=np.array(rectification.alpha),
        zero_disparity=np.array(rectification.zero_disparity),
    )


def load_rectification(path: Path) -> StereoRectification:
    """读取校正 npz，字段缺失或形状不符时抛出 ValueError。"""
    if not path.is_file():
        raise FileNotFoundError(f"校正文件不存在：{path}")
    try:
        with np.load(path, allow_pickle=False) as data:
            rectification: StereoRectification = StereoRectification(
                image_size=(int(data["image_size_wh"][0]), int(data["image_size_wh"][1])),
                rotation_left=np.asarray(data["R1"], dtype=np.float64),
                rotation_right=np.asarray(data["R2"], dtype=np.float64),
                projection_left=np.asarray(data["P1"], dtype=np.float64),
                projection_right=np.asarray(data["P2"], dtype=np.float64),
                q=np.asarray(data["Q"], dtype=np.float64),
                roi_left=_as_roi(data["roi1"]),
                roi_right=_as_roi(data["roi2"]),
                map_left_x=np.asarray(data["map1x"], dtype=np.float32),
                map_left_y=np.asarray(data["map1y"], dtype=np.float32),
                map_right_x=np.asarray(data["map2x"], dtype=np.float32),
                map_right_y=np.asarray(data["map2y"], dtype=np.float32),
                alpha=float(np.asarray(data["alpha"])),
                zero_disparity=bool(np.asarray(data["zero_disparity"])),
            )
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"{path}：校正文件字段缺失或格式不符（{error}）") from error
    width: int
    height: int
    width, height = rectification.image_size
    expected: tuple[int, int] = (height, width)
    if any(
        table.shape != expected or table.dtype != np.float32
        for table in (
            rectification.map_left_x,
            rectification.map_left_y,
            rectification.map_right_x,
            rectification.map_right_y,
        )
    ):
        raise ValueError(f"{path}：重映射表必须为 float32 {height}x{width}")
    if not np.isfinite(rectification.q).all():
        raise ValueError(f"{path}：Q 包含非有限数值")
    return rectification


def _as_roi(value: object) -> ImageRoi:
    array: NDArray[np.int64] = np.asarray(value, dtype=np.int64).reshape(-1)
    if array.size != 4:
        raise ValueError("ROI 必须是 4 个整数 (x, y, width, height)")
    return (int(array[0]), int(array[1]), int(array[2]), int(array[3]))
