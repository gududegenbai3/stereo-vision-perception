# -*- coding: utf-8 -*-
"""精算算子：对被选中的像素换一组更贵的 SGBM 参数重算。

两种实现：
  roi    —— 按窗口聚类，用**局部搜索范围**只重算被选中像素所在的窗口（代价随预算增长）
  global —— 整图 + 全范围重算（作为精度上界参考，代价与预算无关）

为什么 roi 要用局部搜索范围：SGBM 要求 图像宽度 > minDisparity + numDisparities，
直接在小窗口上用全范围（如 256）会让窗口几乎全部无效。把搜索范围收窄到该窗口的
粗算视差区间，既满足 OpenCV 的约束，也显著降低单窗口开销。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .config import RefineCfg, SgbmCfg
from .matcher import compute

MIN_H = 8


@dataclass
class RefineResult:
    disp: np.ndarray
    time_ms: float
    rois: int
    processed_px: int
    skipped: int = 0

    def as_dict(self) -> Dict[str, float]:
        return {"refine_time_ms": self.time_ms, "rois": self.rois,
                "processed_px": self.processed_px, "skipped_rois": self.skipped}


def _local_search_range(disp_crop: np.ndarray, fine_cfg: SgbmCfg, margin: int) -> Tuple[int, int]:
    """由粗算视差给出该窗口的搜索范围 (minDisparity, numDisparities)。"""
    vals = disp_crop[disp_crop > 0]
    if vals.size == 0:
        return 0, fine_cfg.num_disparities
    lo = max(0, int(np.floor(float(vals.min()))) - margin)
    hi = int(np.ceil(float(vals.max()))) + margin
    span = max(16, hi - lo + 1)
    nd = min(fine_cfg.num_disparities, int(np.ceil(span / 16.0)) * 16)
    return lo, nd


def refine_global(left: np.ndarray, right: np.ndarray, disp_base: np.ndarray, cfg: SgbmCfg) -> RefineResult:
    d_fine, dt = compute(left, right, cfg)
    merged = disp_base.copy()
    keep = d_fine > 0
    merged[keep] = d_fine[keep]
    return RefineResult(disp=merged, time_ms=dt, rois=1, processed_px=int(left.shape[0] * left.shape[1]))


def refine_roi(
    left: np.ndarray, right: np.ndarray, disp_base: np.ndarray,
    sel_mask: np.ndarray, fine_cfg: SgbmCfg, rc: RefineCfg,
) -> RefineResult:
    """只重算被选中像素所在窗口，返回合并后的视差图与真实开销。"""
    h, w = disp_base.shape
    out = disp_base.copy()
    total_ms, rois, processed, skipped = 0.0, 0, 0, 0
    step = rc.window

    for y0 in range(0, h, step):
        y1 = min(h, y0 + step)
        for x0 in range(0, w, step):
            x1 = min(w, x0 + step)
            if not sel_mask[y0:y1, x0:x1].any():
                continue

            py0, py1 = max(0, y0 - rc.pad), min(h, y1 + rc.pad)
            px0, px1 = max(0, x0 - rc.pad), min(w, x1 + rc.pad)
            if py1 - py0 < MIN_H:
                skipped += 1
                continue

            lo, nd = _local_search_range(disp_base[py0:py1, px0:px1], fine_cfg, rc.search_margin)
            need_w = lo + nd + fine_cfg.block_size + 8
            if need_w > w:                       # 图本身太窄，无法满足 SGBM 约束
                skipped += 1
                continue
            if px1 - px0 < need_w:               # 窗口加宽到 SGBM 能接受的最小宽度
                grow = need_w - (px1 - px0)
                px0 = max(0, px0 - grow // 2)
                px1 = min(w, px0 + need_w)
                px0 = max(0, px1 - need_w)

            sub_l = left[py0:py1, px0:px1]
            sub_r = right[py0:py1, px0:px1]
            d_fine, dt = compute(sub_l, sub_r, fine_cfg, min_disparity=lo, num_disparities=nd)
            total_ms += dt
            rois += 1
            processed += int(sub_l.shape[0] * sub_l.shape[1])

            patch = d_fine[y0 - py0:y1 - py0, x0 - px0:x1 - px0]
            keep = sel_mask[y0:y1, x0:x1] & (patch > 0)
            out[y0:y1, x0:x1][keep] = patch[keep]

    return RefineResult(disp=out, time_ms=total_ms, rois=rois, processed_px=processed, skipped=skipped)


def refine(
    left: np.ndarray, right: np.ndarray, disp_base: np.ndarray,
    sel_mask: np.ndarray, fine_cfg: SgbmCfg, rc: RefineCfg,
) -> RefineResult:
    """按配置选择算子；none 表示不做精算（用于 A 基线）。"""
    if rc.operator == "none" or not sel_mask.any():
        return RefineResult(disp=disp_base.copy(), time_ms=0.0, rois=0, processed_px=0)
    if rc.operator == "global":
        return refine_global(left, right, disp_base, fine_cfg)
    return refine_roi(left, right, disp_base, sel_mask, fine_cfg, rc)
