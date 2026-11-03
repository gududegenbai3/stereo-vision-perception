"""将左参考像素视差重投影到校正后左相机坐标系。"""

import cv2
import numpy as np

from .model import (
    DepthMap,
    DisparityResult,
    PointMap,
    ReconstructionResult,
    RectificationGeometry,
    RectifiedPair,
    ReprojectionMatrix,
    ValidMask,
)


def reproject_disparity(
    pair: RectifiedPair,
    geometry: RectificationGeometry,
    disparity: DisparityResult,
) -> ReconstructionResult:
    """使用配套米制 Q 重建深度和三维点。

    输入左右图为 uint8 BGR (H, W, 3)，视差为 float32 (H, W)，单位
    像素，匹配掩码为 bool (H, W)。输出深度和 XYZ 为 float32，
    形状分别为 (H, W) 和 (H, W, 3)，单位米。无效深度和 XYZ
    写为 NaN。输入尺寸、格式或 Q 无效，或没有有效三维点时，
    抛出 ValueError。
    """
    width: int
    height: int
    width, height = geometry.image_size
    shape: tuple[int, int] = (height, width)
    if any(
        image.shape != (*shape, 3) or image.dtype != np.uint8
        for image in (pair.left_bgr, pair.right_bgr)
    ):
        raise ValueError(f"左右校正图必须是 uint8 BGR，且形状为 {(*shape, 3)}")
    if disparity.disparity_px.shape != shape or disparity.disparity_px.dtype != np.float32:
        raise ValueError(f"视差必须是形状为 {shape} 的 float32 像素数组")
    if disparity.valid_mask.shape != shape or disparity.valid_mask.dtype != np.bool_:
        raise ValueError(f"匹配有效掩码必须是形状为 {shape} 的 bool 数组")

    q: ReprojectionMatrix = geometry.q
    if q.shape != (4, 4) or q.dtype != np.float64 or not np.isfinite(q).all():
        raise ValueError("Q 必须是有限的 4×4 float64 矩阵")

    valid_mask: ValidMask = disparity.valid_mask & np.isfinite(disparity.disparity_px)
    if not valid_mask.any():
        raise ValueError("没有有效匹配视差，无法重建")

    xyz_m: PointMap = cv2.reprojectImageTo3D(disparity.disparity_px, q)
    depth_m: DepthMap = xyz_m[..., 2]
    valid_mask &= np.isfinite(xyz_m).all(axis=2) & (depth_m > 0)
    if not valid_mask.any():
        raise ValueError("重投影未得到有效三维点，请检查 Q 和视差")

    xyz_m[~valid_mask] = np.nan
    return ReconstructionResult(
        rectified_pair=pair,
        disparity_px=disparity.disparity_px,
        depth_m=depth_m,
        xyz_m=xyz_m,
        valid_mask=valid_mask,
    )
