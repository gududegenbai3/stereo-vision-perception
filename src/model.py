"""双目流程共用的数据类型。调用方在构造前负责校验数据。"""

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

# (H, W, 3)，BGR 通道顺序，像素强度范围 0–255。
BgrImage: TypeAlias = NDArray[np.uint8]
# (H, W)，左图参考视差 x_left - x_right，单位像素。
DisparityMap: TypeAlias = NDArray[np.float32]
# (H, W)，沿校正后左相机 Z 轴的深度，单位米。
DepthMap: TypeAlias = NDArray[np.float32]
# (H, W, 3)，校正后左相机坐标系的 XYZ，单位米。
PointMap: TypeAlias = NDArray[np.float32]
# (H, W)，True 表示有效重建像素。
ValidMask: TypeAlias = NDArray[np.bool_]
# (4, 4)，将 [x_px, y_px, disparity_px, 1] 映射为三维齐次坐标。
ReprojectionMatrix: TypeAlias = NDArray[np.float64]
# (width, height)，单位像素。
ImageSize: TypeAlias = tuple[int, int]
# (x, y, width, height)，左上角为原点，单位像素。
ImageRoi: TypeAlias = tuple[int, int, int, int]


MiddleburySplit: TypeAlias = Literal["trainingQ", "testQ"]


@dataclass(frozen=True)
class SgbmConfig:
    """已校验的 SGBM 参数，视差和窗口边长均以像素为单位。"""

    min_disparity: int
    num_disparities: int
    block_size: int


@dataclass(frozen=True, eq=False)
class RawPair:
    """同一时刻或同一静态场景的未校正左右图，尺寸一致。"""

    left_bgr: BgrImage
    right_bgr: BgrImage


@dataclass(frozen=True, eq=False)
class RectifiedPair:
    """同一像素网格尺寸的左右校正图，采用水平极线布局。"""

    left_bgr: BgrImage
    right_bgr: BgrImage


@dataclass(frozen=True, eq=False)
class RectificationGeometry:
    """校正图的重建几何，与实际图像分辨率和裁剪方式绑定。

    q 使用米制平移，齐次除法后的 XYZ 单位为米。
    左右 ROI 分别描述对应校正图的有效成像区域。
    """

    image_size: ImageSize
    q: ReprojectionMatrix
    roi_left: ImageRoi
    roi_right: ImageRoi


@dataclass(frozen=True, eq=False)
class DisparityResult:
    """左校正图参考的匹配结果，无效视差为 NaN。

    valid_roi 是允许搜索的像素区域，valid_mask 表示其中通过匹配
    有效性检查的像素。重投影阶段还需检查深度和三维坐标。
    """

    disparity_px: DisparityMap
    valid_mask: ValidMask
    valid_roi: ImageRoi


@dataclass(frozen=True, eq=False)
class ReconstructionResult:
    """像素与左校正图对齐的稠密重建结果。

    xyz_m 的 X 向右、Y 向下、Z 向前。有效点的 depth_m 等于
    xyz_m 的 Z 分量，无效深度和 XYZ 用 NaN。使用 valid_mask 筛选有效
    坐标和对应的左图颜色，形成彩色点云。
    """

    rectified_pair: RectifiedPair
    disparity_px: DisparityMap
    depth_m: DepthMap
    xyz_m: PointMap
    valid_mask: ValidMask
