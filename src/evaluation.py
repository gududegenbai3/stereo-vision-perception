"""离线双目视差评估。"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .model import DisparityMap, DisparityResult, ValidMask


@dataclass(frozen=True)
class DisparityMetrics:
    """覆盖率为 0 到 1 的比例，视差误差单位为像素。"""

    matched_pixel_count: int
    image_match_rate: float
    search_roi_match_rate: float
    reference_pixel_count: int | None
    compared_pixel_count: int | None
    reference_coverage: float | None
    mae_px: float | None
    rmse_px: float | None


def evaluate_disparity(
    result: DisparityResult,
    reference_px: DisparityMap | None = None,
    reference_mask: ValidMask | None = None,
) -> DisparityMetrics:
    """统计左参考视差的有效率和误差。

    预测和参考均为 float32 (H, W) 像素视差，掩码为 bool (H, W)。
    参考视差和参考掩码须一起提供。误差只在双方有限且通过掩码的像素上
    计算。无真值或有效交集为空时，无法定义的指标返回 None。
    格式、尺寸或搜索 ROI 无效时抛出 ValueError。
    """
    prediction_px: DisparityMap = result.disparity_px
    if prediction_px.ndim != 2 or prediction_px.dtype != np.float32:
        raise ValueError("预测视差必须是 float32 (H, W) 数组")
    if result.valid_mask.shape != prediction_px.shape or result.valid_mask.dtype != np.bool_:
        raise ValueError("匹配有效掩码必须是与预测视差同尺寸的 bool 数组")

    height: int
    width: int
    height, width = prediction_px.shape
    x: int
    y: int
    roi_width: int
    roi_height: int
    x, y, roi_width, roi_height = result.valid_roi
    if not (0 <= x < x + roi_width <= width and 0 <= y < y + roi_height <= height):
        raise ValueError("搜索 ROI 必须非空且位于视差图内")

    predicted_valid: ValidMask = result.valid_mask & np.isfinite(prediction_px)
    matched_count: int = int(np.count_nonzero(predicted_valid))
    image_rate: float = matched_count / prediction_px.size
    roi_rate: float = float(predicted_valid[y:y + roi_height, x:x + roi_width].mean())

    if reference_px is None and reference_mask is None:
        return DisparityMetrics(matched_count, image_rate, roi_rate, None, None, None, None, None)
    if reference_px is None or reference_mask is None:
        raise ValueError("参考视差和参考掩码必须一起提供")
    if reference_px.shape != prediction_px.shape or reference_px.dtype != np.float32:
        raise ValueError("参考视差必须是与预测视差同尺寸的 float32 数组")
    if reference_mask.shape != prediction_px.shape or reference_mask.dtype != np.bool_:
        raise ValueError("参考掩码必须是与预测视差同尺寸的 bool 数组")

    reference_valid: ValidMask = reference_mask & np.isfinite(reference_px)
    compared: ValidMask = reference_valid & predicted_valid
    reference_count: int = int(np.count_nonzero(reference_valid))
    compared_count: int = int(np.count_nonzero(compared))
    coverage: float | None = compared_count / reference_count if reference_count else None
    mae_px: float | None = None
    rmse_px: float | None = None
    if compared_count:
        errors_px: NDArray[np.float64] = (
            prediction_px[compared].astype(np.float64) - reference_px[compared]
        )
        mae_px = float(np.abs(errors_px).mean())
        rmse_px = float(np.sqrt(np.square(errors_px).mean()))

    return DisparityMetrics(
        matched_pixel_count=matched_count,
        image_match_rate=image_rate,
        search_roi_match_rate=roi_rate,
        reference_pixel_count=reference_count,
        compared_pixel_count=compared_count,
        reference_coverage=coverage,
        mae_px=mae_px,
        rmse_px=rmse_px,
    )
