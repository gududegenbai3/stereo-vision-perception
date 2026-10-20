# -*- coding: utf-8 -*-
"""SGBM 封装：只做一件事——给一对校正后的灰度图，返回浮点视差图与耗时。"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .config import MODE_ALIASES, SgbmCfg

DISP_SCALE = float(cv2.StereoMatcher_DISP_SCALE)  # = 16.0，OpenCV 的 1/16 px 定点


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def make_matcher(
    cfg: SgbmCfg,
    min_disparity: Optional[int] = None,
    num_disparities: Optional[int] = None,
) -> cv2.StereoSGBM:
    """min_disparity / num_disparities 可覆盖配置，用于"局部搜索范围"精算。"""
    mode = getattr(cv2, "STEREO_SGBM_" + MODE_ALIASES[cfg.mode])
    md = 0 if min_disparity is None else int(min_disparity)
    nd = cfg.num_disparities if num_disparities is None else int(num_disparities)
    if nd % 16 != 0:
        raise ValueError(f"numDisparities 必须是 16 的倍数，当前 {nd}")
    return cv2.StereoSGBM_create(
        minDisparity=md,
        numDisparities=nd,
        blockSize=cfg.block_size,
        P1=cfg.p1(),
        P2=cfg.p2(),
        disp12MaxDiff=cfg.disp12_max_diff,
        uniquenessRatio=cfg.uniqueness_ratio,
        speckleWindowSize=cfg.speckle_window,
        speckleRange=cfg.speckle_range,
        preFilterCap=cfg.pre_filter_cap,
        mode=mode,
    )


def compute(
    left: np.ndarray, right: np.ndarray, cfg: SgbmCfg,
    min_disparity: Optional[int] = None, num_disparities: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """返回 (视差图 float32，单位 px；无效像素为 0)，以及单次耗时（毫秒）。

    注意：OpenCV 的 SGBM 输出的是**绝对视差**（已包含 minDisparity），不需要再加偏移。
    """
    matcher = make_matcher(cfg, min_disparity, num_disparities)
    t0 = _now()
    raw = matcher.compute(to_gray(left), to_gray(right))
    dt_ms = (_now() - t0) * 1000.0
    disp = raw.astype(np.float32) / DISP_SCALE
    disp[disp <= 0] = 0.0
    return disp, dt_ms


def _now() -> float:
    import time
    return time.perf_counter()
