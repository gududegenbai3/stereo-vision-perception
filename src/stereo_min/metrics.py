# -*- coding: utf-8 -*-
"""评测：视差域指标。所有统计口径集中在此，避免各脚本各算一套。

口径（对应 V0 计划 A4）：
- 只在“预测有效 **且** 真值有效”的像素上统计误差；**无效预测不以 0 混入均值**。
- 覆盖率单独报告：有效达标像素 / 有真值的像素。
- 视差单位像素；不在此处换算深度（深度误差由独立参考距离评价，见计划 I2）。
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def valid_gt_mask(gt: np.ndarray) -> np.ndarray:
    """真值有效：有限且为正（`.pfm` 中无真值处约定为负值或 NaN）。"""
    g = gt.astype(np.float32, copy=False)
    return np.isfinite(g) & (g > 0)


def disparity_metrics(
    disp: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None,
    bad_thresholds: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> Dict[str, float]:
    """返回视差误差指标（像素）。

    `disp` 与 `gt` 形状一致；`valid` 是预测有效掩码（None 时用 `disp > 0`）。
    """
    d = disp.astype(np.float32, copy=False)
    g = gt.astype(np.float32, copy=False)
    if d.shape != g.shape:
        raise ValueError(f"视差与真值形状不一致：{d.shape} vs {g.shape}")

    gt_mask = valid_gt_mask(g)
    if valid is None:
        valid = np.isfinite(d) & (d > 0)
    eval_mask = gt_mask & valid
    gt_pixels = int(gt_mask.sum())
    eval_pixels = int(eval_mask.sum())

    out: Dict[str, float] = {
        "gt_pixels": float(gt_pixels),
        "eval_pixels": float(eval_pixels),
        "coverage": (eval_pixels / gt_pixels) if gt_pixels else float("nan"),
    }
    if eval_pixels == 0:
        out.update({"epe": float("nan"), "p95": float("nan")})
        for t in bad_thresholds:
            out[f"bad{int(t)}"] = float("nan")
        return out

    err = np.abs(d[eval_mask] - g[eval_mask])
    out["epe"] = float(err.mean())
    out["p95"] = float(np.percentile(err, 95))
    for t in bad_thresholds:
        out[f"bad{int(t)}"] = float((err > t).mean())
    return out


def depth_metrics(depth: np.ndarray, gt_depth: np.ndarray) -> Dict[str, float]:
    """深度域误差（米），仅用于有独立参考深度的场景；无效值 `NaN` 不参与统计。"""
    pred = depth.astype(np.float32, copy=False)
    ref = gt_depth.astype(np.float32, copy=False)
    if pred.shape != ref.shape:
        raise ValueError(f"深度与参考形状不一致：{pred.shape} vs {ref.shape}")
    mask = np.isfinite(pred) & np.isfinite(ref) & (ref > 0)
    if not mask.any():
        return {"depth_samples": 0.0, "depth_mae": float("nan"), "depth_rmse": float("nan")}
    diff = pred[mask] - ref[mask]
    return {
        "depth_samples": float(mask.sum()),
        "depth_mae": float(np.abs(diff).mean()),
        "depth_rmse": float(np.sqrt((diff ** 2).mean())),
        "depth_rel_mae": float((np.abs(diff) / ref[mask]).mean()),
    }
