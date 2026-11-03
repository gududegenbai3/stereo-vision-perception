"""使用 OpenCV SGBM 计算水平校正图像对的左参考视差。"""

import cv2
import numpy as np
from numpy.typing import NDArray

from .model import (
    DisparityMap,
    DisparityResult,
    ImageRoi,
    RectificationGeometry,
    RectifiedPair,
    SgbmConfig,
    ValidMask,
)

DISPARITY_SCALE: int = 16


def compute_disparity(
    pair: RectifiedPair,
    geometry: RectificationGeometry,
    config: SgbmConfig,
) -> DisparityResult:
    """返回 float32 像素视差和 bool 有效掩码，形状均为 (H, W)。

    左右输入是 uint8 BGR 图像 (H, W, 3)。配置层校验 SGBM 参数，
    此处检查参数是否适合图像和 OpenCV 的定点输出。图像尺寸、ROI
    或搜索范围不合法时抛出 ValueError。
    """
    width: int
    height: int
    width, height = geometry.image_size
    expected_shape: tuple[int, int, int] = (height, width, 3)
    if any(
        image.shape != expected_shape or image.dtype != np.uint8
        for image in (pair.left_bgr, pair.right_bgr)
    ):
        raise ValueError(f"左右校正图必须是 uint8 BGR，且形状为 {expected_shape}")

    roi: ImageRoi
    roi_x: int
    roi_y: int
    roi_width: int
    roi_height: int
    for roi in (geometry.roi_left, geometry.roi_right):
        roi_x, roi_y, roi_width, roi_height = roi
        if (
            min(roi_x, roi_y) < 0
            or min(roi_width, roi_height) <= 0
            or roi_x + roi_width > width
            or roi_y + roi_height > height
        ):
            raise ValueError(f"校正 ROI 为空或超出图像边界：{roi}")

    min_disparity: int = config.min_disparity
    num_disparities: int = config.num_disparities
    block_size: int = config.block_size
    invalid_raw: int = (min_disparity - 1) * DISPARITY_SCALE
    upper_raw: int = (min_disparity + num_disparities) * DISPARITY_SCALE
    if invalid_raw < -32768 or upper_raw > 32768:
        raise ValueError("SGBM 视差范围超出 int16 定点输出可表示的范围")
    search_width: int = max(0, min_disparity + num_disparities) - min(0, min_disparity)
    if search_width >= width:
        raise ValueError("SGBM 视差范围过宽，图像没有可搜索的列")
    if block_size > min(width, height) or 32 * block_size**2 > 2**31 - 1:
        raise ValueError("SGBM 窗口过大")

    valid_roi: ImageRoi = cv2.getValidDisparityROI(
        geometry.roi_left, geometry.roi_right, min_disparity, num_disparities, block_size
    )
    x: int
    y: int
    roi_w: int
    roi_h: int
    x, y, roi_w, roi_h = valid_roi

    # 负视差把右图候选窗口向右移动，确保窗口仍落在右侧 ROI 内。
    right_x: int = geometry.roi_right[0]
    right_width: int = geometry.roi_right[2]
    right_limit: int = right_x + right_width - block_size // 2 + min_disparity
    roi_w = min(x + roi_w, right_limit) - x
    if min(roi_w, roi_h) <= 0:
        raise ValueError("SGBM 有效搜索 ROI 为空，请检查视差范围、窗口与校正 ROI")
    valid_roi = (x, y, roi_w, roi_h)

    gray_left: NDArray[np.uint8] = cv2.cvtColor(pair.left_bgr, cv2.COLOR_BGR2GRAY)
    gray_right: NDArray[np.uint8] = cv2.cvtColor(pair.right_bgr, cv2.COLOR_BGR2GRAY)
    sgbm_p1: int = 8 * block_size**2
    sgbm_p2: int = 32 * block_size**2
    matcher: cv2.StereoSGBM = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=sgbm_p1,
        P2=sgbm_p2,
        mode=cv2.STEREO_SGBM_MODE_SGBM,
    )
    disparity_raw: NDArray[np.int16] = matcher.compute(gray_left, gray_right)

    valid_mask: ValidMask = np.zeros((height, width), dtype=np.bool_)
    roi_rows: slice = slice(y, y + roi_h)
    roi_cols: slice = slice(x, x + roi_w)
    roi_disparity: NDArray[np.int16] = disparity_raw[roi_rows, roi_cols]
    valid_mask[roi_rows, roi_cols] = (roi_disparity > invalid_raw) & (
        roi_disparity < upper_raw
    )

    disparity_px: DisparityMap = disparity_raw.astype(np.float32) / DISPARITY_SCALE
    disparity_px[~valid_mask] = np.nan
    return DisparityResult(disparity_px, valid_mask, valid_roi)
