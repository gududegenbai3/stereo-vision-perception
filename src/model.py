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
class ChessboardSpec:
    """棋盘格规格，cols 与 rows 为内角点数，格长为米。"""

    cols: int
    rows: int
    square_size_m: float

    def __post_init__(self) -> None:
        if self.cols < 3 or self.rows < 3:
            raise ValueError("棋盘格内角点行数与列数都必须不小于 3")
        if not np.isfinite(self.square_size_m) or self.square_size_m <= 0:
            raise ValueError("棋盘格格长必须为有限正数")

    @property
    def corner_count(self) -> int:
        """单张图像应检测到的内角点数量。"""
        return self.cols * self.rows

    def object_points(self) -> NDArray[np.float32]:
        """返回 (N, 3) 对象点，单位为米，Z 为 0。"""
        points: NDArray[np.float32] = np.zeros((self.corner_count, 3), dtype=np.float32)
        grid: NDArray[np.float32] = np.mgrid[0 : self.cols, 0 : self.rows].T.reshape(-1, 2)
        points[:, :2] = grid.astype(np.float32) * np.float32(self.square_size_m)
        return points


@dataclass(frozen=True)
class AsymmetricCircleGridSpec:
    """非对称圆点阵列，圆点位于边长 spacing_m 的方点阵的隔点位置。

    第 i 行（从 0 开始，按从上到下计）圆点的 y 坐标为 i * spacing_m；
    奇数行是长行，含 max_per_row 个圆点，x 从 0 开始按 2 * spacing_m
    递增；偶数行是短行，少一个圆点，整体右移一个 spacing_m。因此行内
    间距为 2 * spacing_m，相邻行最近点距离为 sqrt(2) * spacing_m。
    rows 必须为偶数，point_count 为行数与点数交替后的总数。
    """

    rows: int
    max_per_row: int
    spacing_m: float
    diameter_m: float

    def __post_init__(self) -> None:
        if self.rows < 2 or self.rows % 2 != 0:
            raise ValueError("圆点阵列行数必须是大于等于 2 的偶数")
        if self.max_per_row < 2:
            raise ValueError("圆点阵列每行点数必须不小于 2")
        if not np.isfinite(self.spacing_m) or self.spacing_m <= 0:
            raise ValueError("圆点阵列间距必须为有限正数")
        if not np.isfinite(self.diameter_m) or self.diameter_m <= 0:
            raise ValueError("圆点直径必须为有限正数")

    @property
    def point_count(self) -> int:
        """整板圆点总数。"""
        return (self.rows // 2) * (2 * self.max_per_row - 1)

    @property
    def row_pitch_m(self) -> float:
        """相邻两行的 y 间距。"""
        return self.spacing_m

    @property
    def column_pitch_m(self) -> float:
        """同一行内相邻圆点的间距。"""
        return 2.0 * self.spacing_m

    def row_counts(self) -> tuple[int, ...]:
        """从上到下每一行的圆点数。"""
        return tuple(
            self.max_per_row if row % 2 == 1 else self.max_per_row - 1
            for row in range(self.rows)
        )

    def object_points(self) -> NDArray[np.float32]:
        """返回 (N, 3) 对象点，单位为米，Z 为 0，按行优先排列。"""
        coordinates: list[tuple[float, float, float]] = []
        for row, count in enumerate(self.row_counts()):
            offset = 0.0 if row % 2 == 1 else self.spacing_m
            for column in range(count):
                coordinates.append(
                    (offset + 2.0 * self.spacing_m * column, row * self.spacing_m, 0.0)
                )
        points: NDArray[np.float32] = np.array(coordinates, dtype=np.float32)
        return points


# 标定靶规格：棋盘格或非对称圆点阵列。
TargetSpec: TypeAlias = ChessboardSpec | AsymmetricCircleGridSpec


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
