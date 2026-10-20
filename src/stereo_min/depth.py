# -*- coding: utf-8 -*-
"""视差 → 深度、有效掩码。几何未知时只给视差。"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import GeometryCfg


def disparity_to_depth(disp: np.ndarray, geometry: GeometryCfg) -> Tuple[np.ndarray, np.ndarray]:
    """按 `Z = fx_px * baseline_m / d` 换算深度。

    返回 `(depth float32 (H, W) 米, valid bool (H, W))`：无效像素深度为 `NaN`。
    几何参数未配置时抛 `ConfigError`——深度必须由标定结果驱动。
    """
    if not geometry.usable:
        raise ValueError("未提供 fx_px / baseline_m，无法换算深度")
    d = disp.astype(np.float32, copy=False)
    if d.ndim != 2:
        raise ValueError(f"视差必须是二维数组，当前形状 {d.shape}")
    valid = np.isfinite(d) & (d > 0)
    depth = np.full(d.shape, np.nan, dtype=np.float32)
    depth[valid] = (geometry.fx_px * geometry.baseline_m) / d[valid]
    return depth, valid


def valid_mask(disp: np.ndarray, roi_margin_px: int = 0) -> np.ndarray:
    """有效视差掩码：视差为正且落在裁剪边界内（去掉 SGBM 在边界处不可靠的条带）。"""
    mask = np.isfinite(disp) & (disp > 0)
    if roi_margin_px > 0:
        h, w = mask.shape
        border = np.zeros_like(mask)
        border[roi_margin_px:h - roi_margin_px, roi_margin_px:w - roi_margin_px] = True
        mask &= border
    return mask


def depth_stats(depth: Optional[np.ndarray]) -> dict[str, float]:
    """深度汇总（米）：有效像素的比例与分位数，用于写入 metrics.json。"""
    if depth is None:
        return {}
    finite = np.isfinite(depth)
    if not finite.any():
        return {"depth_valid_ratio": 0.0}
    vals = depth[finite]
    return {
        "depth_valid_ratio": float(finite.mean()),
        "depth_m_min": float(vals.min()),
        "depth_m_median": float(np.median(vals)),
        "depth_m_max": float(vals.max()),
    }
