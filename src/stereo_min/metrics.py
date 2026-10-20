# -*- coding: utf-8 -*-
"""评测模块：视差域指标 + 风险-覆盖率曲线。所有统计口径集中在这里，避免各脚本各算一套。"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def predict_valid(disp: np.ndarray) -> np.ndarray:
    return disp > 0


def error_map(disp: np.ndarray, gt: np.ndarray, penalty: float = 3.0) -> np.ndarray:
    """误差图。预测缺失（视差为 0）按 penalty 计，避免"拒绝输出"白拿好处。"""
    err = np.abs(disp.astype(np.float32) - gt.astype(np.float32))
    err[~predict_valid(disp)] = penalty
    return err


def summarize(disp: np.ndarray, gt: np.ndarray, penalty: float = 3.0) -> Dict[str, float]:
    """在 GT 有效像素上统计；覆盖率指预测有效的比例。"""
    gt_valid = gt > 0
    if not gt_valid.any():
        raise ValueError("GT 中没有有效像素")
    err = error_map(disp, gt, penalty)[gt_valid]
    pred_valid = predict_valid(disp)[gt_valid]
    return {
        "coverage": float(pred_valid.mean()),
        "epe": float(err.mean()),
        "epe_valid_only": float(err[pred_valid].mean()) if pred_valid.any() else float("nan"),
        "d1": float((err > 1.0).mean()),
        "bad2": float((err > 2.0).mean()),
        "p95": float(np.percentile(err, 95)),
        "gt_pixels": int(gt_valid.sum()),
    }


def region_summarize(disp: np.ndarray, gt: np.ndarray, region: np.ndarray, penalty: float = 3.0) -> Dict[str, float]:
    """按区域（例如纹理区/无纹理区）分组统计同一套指标。"""
    gt_valid = (gt > 0) & region
    if not gt_valid.any():
        return {"coverage": float("nan"), "epe": float("nan"), "d1": float("nan"), "pixels": 0}
    err = error_map(disp, gt, penalty)[gt_valid]
    pred_valid = predict_valid(disp)[gt_valid]
    return {
        "coverage": float(pred_valid.mean()),
        "epe": float(err.mean()),
        "d1": float((err > 1.0).mean()),
        "pixels": int(gt_valid.sum()),
    }


def risk_coverage(
    uncertainty: np.ndarray, err: np.ndarray, gt_valid: np.ndarray,
    levels: int = 20, penalty: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """按不确定度从小到大（最可靠先看）逐步提高覆盖率，返回 (覆盖率, 选择性风险) 两条曲线。"""
    idx = np.flatnonzero(gt_valid)
    if idx.size == 0:
        raise ValueError("没有 GT 有效像素")
    u = uncertainty.ravel()[idx]
    e = err.ravel()[idx]
    order = np.argsort(u, kind="stable")
    e_sorted = e[order]

    cov = np.linspace(1.0 / levels, 1.0, levels)
    risk = np.empty_like(cov)
    for i, c in enumerate(cov):
        k = max(1, int(round(c * idx.size)))
        risk[i] = float(e_sorted[:k].mean())
    return cov, risk


def aurc(cov: np.ndarray, risk: np.ndarray) -> float:
    """风险-覆盖率曲线下面积（按覆盖率区间归一化，等价于该区间内的平均风险），越小越好。

    手写梯形积分，避免依赖 numpy 1.x / 2.x 的 trapz / trapezoid 命名差异。
    """
    cov = np.asarray(cov, dtype=np.float64)
    risk = np.asarray(risk, dtype=np.float64)
    if cov.size < 2 or cov[-1] <= cov[0]:
        return float("nan")
    area = float(np.sum(0.5 * (risk[:-1] + risk[1:]) * np.diff(cov)))
    return area / float(cov[-1] - cov[0])


def sparsification_error(uncertainty: np.ndarray, err: np.ndarray, gt_valid: np.ndarray,
                         fraction: float = 0.5) -> float:
    """移除最不可靠的 fraction 比例像素后，剩余像素的平均误差。"""
    idx = np.flatnonzero(gt_valid)
    if idx.size == 0:
        return float("nan")
    u = uncertainty.ravel()[idx]
    e = err.ravel()[idx]
    order = np.argsort(u, kind="stable")
    keep = max(1, int(round((1.0 - fraction) * idx.size)))
    return float(e[order][:keep].mean())
