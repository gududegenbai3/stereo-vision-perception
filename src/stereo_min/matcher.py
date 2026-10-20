# -*- coding: utf-8 -*-
"""SGBM 视差：输入已校正的左右图，输出浮点视差（像素）与耗时。"""
from __future__ import annotations

import time
from typing import Tuple

import cv2
import numpy as np

from .config import MODE_ALIASES, SgbmCfg

# OpenCV 以 1/16 像素定点输出视差，读取时必须除以 DISP_SCALE
DISP_SCALE: float = float(cv2.StereoMatcher_DISP_SCALE)


def to_gray(image: np.ndarray) -> np.ndarray:
    """转灰度 `uint8 (H, W)`；彩色按 BGR 处理。"""
    if image.ndim == 3:
        if image.shape[2] != 3:
            raise ValueError(f"彩色图必须是 3 通道 BGR，当前形状 {image.shape}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2:
        raise ValueError(f"图像维度必须是 2，当前形状 {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def make_matcher(cfg: SgbmCfg) -> cv2.StereoSGBM:
    mode = getattr(cv2, "STEREO_SGBM_" + MODE_ALIASES[cfg.mode])
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=cfg.num_disparities,
        blockSize=cfg.block_size,
        P1=cfg.p1,
        P2=cfg.p2,
        disp12MaxDiff=cfg.disp12_max_diff,
        uniquenessRatio=cfg.uniqueness_ratio,
        speckleWindowSize=cfg.speckle_window,
        speckleRange=cfg.speckle_range,
        preFilterCap=cfg.pre_filter_cap,
        mode=mode,
    )


def compute_disparity(left: np.ndarray, right: np.ndarray, cfg: SgbmCfg) -> Tuple[np.ndarray, float]:
    """返回 `(disp float32 (H, W) 像素, 耗时 ms)`；无效像素为 0。

    约定：`d = x_L - x_R >= 0`；OpenCV 输出为 1/16 定点，已在此处除以 16。
    """
    gray_l, gray_r = to_gray(left), to_gray(right)
    if gray_l.shape != gray_r.shape:
        raise ValueError(f"左右图尺寸不一致：{gray_l.shape} vs {gray_r.shape}")
    h, w = gray_l.shape
    need = cfg.num_disparities + cfg.block_size + 8
    if w < need:
        raise ValueError(f"图像宽度 {w} 不足：num_disparities={cfg.num_disparities} 时至少需要 {need}")

    matcher = make_matcher(cfg)
    t0 = time.perf_counter()
    raw = matcher.compute(gray_l, gray_r)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    disp = raw.astype(np.float32) / DISP_SCALE
    disp[disp <= 0] = 0.0
    return disp, elapsed_ms
