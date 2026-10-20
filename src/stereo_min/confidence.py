# -*- coding: utf-8 -*-
"""自查模块：把"这里算不准"量化成不确定度图（越大越不可靠）。

最小版本只用一个信号源——左右一致性（LRC），它同时能检出遮挡与误匹配。
后续要加"代价曲率 / 多峰性"时，只需再产出一张同形状的 uncertainty 在 pipeline 里融合。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np

from .config import SgbmCfg
from .matcher import compute

LARGE = 1e3  # 无效像素的不确定度，保证它们排在队伍最前面被精修


@dataclass
class ConfidenceResult:
    uncertainty: np.ndarray   # float32，越大越不可靠；>=LARGE 表示无效
    valid: np.ndarray         # bool，左右一致且两视差均有效
    lrc_err: np.ndarray       # float32，左右视差之差（无效处为 LARGE）
    disp: np.ndarray          # float32，左参考实现（粗算视差）
    time_ms: float            # 两遍匹配合计耗时


def lrc_pair(d_l: np.ndarray, d_r: np.ndarray, thresh: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """由左参考视差 d_l 与右参考视差 d_r 计算 (uncertainty, valid, lrc_err)。

    约定：d = x_L - x_R >= 0。对左图像素 x，其在右参考图中的对应横坐标为 x - d_l(x)。
    """
    if d_l.shape != d_r.shape:
        raise ValueError(f"左右视差图尺寸不一致：{d_l.shape} vs {d_r.shape}")
    h, w = d_l.shape
    xs = np.broadcast_to(np.arange(w, dtype=np.float32)[None, :], (h, w))
    ys = np.broadcast_to(np.arange(h, dtype=np.float32)[:, None], (h, w)).astype(np.int32)

    xi = np.rint(xs - d_l).astype(np.int32)
    inside = (d_l > 0) & (xi >= 0) & (xi < w)
    d_r_at = d_r[ys, np.clip(xi, 0, w - 1)]

    err = np.abs(d_l - d_r_at).astype(np.float32)
    valid = inside & (d_r_at > 0) & (err <= thresh)
    uncertainty = np.where(valid, err, LARGE).astype(np.float32)
    lrc_err = np.where(inside, err, LARGE).astype(np.float32)
    return uncertainty, valid, lrc_err


def lrc_confidence(left: np.ndarray, right: np.ndarray, cfg: SgbmCfg, thresh: float = 1.0) -> ConfidenceResult:
    """完整版：自己跑两遍匹配。pipeline 里若已有粗算结果，应改用 lrc_pair 省一遍。"""
    d_l, t_l = compute(left, right, cfg)
    d_r, t_r = compute(right, left, cfg)
    uncertainty, valid, lrc_err = lrc_pair(d_l, d_r, thresh)
    return ConfidenceResult(uncertainty=uncertainty, valid=valid, lrc_err=lrc_err,
                            disp=d_l, time_ms=t_l + t_r)


def texture_map(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """局部梯度能量，仅用于分析"可救性"，不参与挑选。"""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def summarize_confidence(res: ConfidenceResult) -> Dict[str, float]:
    total = res.valid.size
    return {
        "valid_ratio": float(res.valid.sum()) / total,
        "invalid_ratio": float((~res.valid).sum()) / total,
        "lrc_err_mean_valid_px": float(res.lrc_err[res.valid].mean()) if res.valid.any() else float("nan"),
    }
