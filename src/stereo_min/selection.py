# -*- coding: utf-8 -*-
"""分配模块：把不确定度变成"这次精修哪些像素"。"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def budget_pixels(uncertainty: np.ndarray, fraction: float) -> int:
    n = int(round(float(fraction) * uncertainty.size))
    return max(0, min(uncertainty.size, n))


def select_top_k(uncertainty: np.ndarray, fraction: float) -> np.ndarray:
    """按不确定度从大到小取前 fraction 比例的像素。分数相同时按像素索引稳定排序，保证可复现。"""
    n = budget_pixels(uncertainty, fraction)
    flat = uncertainty.ravel()
    mask = np.zeros(flat.size, dtype=bool)
    if n > 0:
        order = np.argsort(-flat, kind="stable")
        mask[order[:n]] = True
    return mask.reshape(uncertainty.shape)


def select_random(uncertainty: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """对照组 R：同样预算，但随机挑。用于证明"自适应"不是"多算就更好"。"""
    n = budget_pixels(uncertainty, fraction)
    flat = np.zeros(uncertainty.size, dtype=bool)
    if n > 0:
        idx = rng.choice(uncertainty.size, size=n, replace=False)
        flat[idx] = True
    return flat.reshape(uncertainty.shape)


def select_by(
    uncertainty: np.ndarray, fraction: float, strategy: str = "adaptive",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, str]:
    if strategy == "adaptive":
        return select_top_k(uncertainty, fraction), strategy
    if strategy == "random":
        if rng is None:
            raise ValueError("random 策略需要传入 rng")
        return select_random(uncertainty, fraction, rng), strategy
    raise ValueError(f"未知策略 {strategy}")
