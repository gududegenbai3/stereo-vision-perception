# -*- coding: utf-8 -*-
"""单对图像的一次完整评测：视差 → （可选）深度 → 掩码 → 指标。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import RunCfg
from .depth import depth_stats, disparity_to_depth, valid_mask
from .matcher import compute_disparity
from . import metrics as M


@dataclass
class PairResult:
    """一对图像的处理结果与指标。"""

    disparity: np.ndarray                     # float32 (H, W)，像素，无效为 0
    mask: np.ndarray                          # bool (H, W)
    depth: np.ndarray | None                  # float32 (H, W) 米，无效为 NaN
    disparity_ms: float
    metrics: dict[str, float] = field(default_factory=dict)


def evaluate_pair(
    left: np.ndarray, right: np.ndarray, cfg: RunCfg,
    gt: np.ndarray | None = None, gt_depth: np.ndarray | None = None,
) -> PairResult:
    """处理一对已校正图像；有真值时同时给出视差/深度指标。

    注意：本函数不读文件、不读全局配置（AGENTS.md 规则 3）；所有参数已由调用层校验。
    """
    disp, elapsed_ms = compute_disparity(left, right, cfg.sgbm)
    mask = valid_mask(disp, cfg.roi_margin_px)

    depth: np.ndarray | None = None
    if cfg.geometry.usable:
        depth, _ = disparity_to_depth(disp, cfg.geometry)

    out: dict[str, float] = {
        "width": float(disp.shape[1]),
        "height": float(disp.shape[0]),
        "disparity_ms": float(elapsed_ms),
        "valid_ratio": float(mask.mean()),
        "disparity_max_px": float(disp[mask].max()) if mask.any() else float("nan"),
    }
    if gt is not None:
        out.update(M.disparity_metrics(disp, gt, mask))
    if depth is not None and gt_depth is not None:
        out.update(M.depth_metrics(depth, gt_depth))
    out.update(depth_stats(depth))
    return PairResult(disparity=disp, mask=mask, depth=depth,
                      disparity_ms=elapsed_ms, metrics=out)


def result_as_row(name: str, result: PairResult) -> dict[str, Any]:
    """把结果整理成 samples.csv 的一行。"""
    row: dict[str, Any] = {"pair_id": name}
    row.update({k: v for k, v in result.metrics.items()})
    return row
