# -*- coding: utf-8 -*-
"""合成图对：在没有任何外部数据时也能跑通并自检。

设计成三段竖直平面（不同视差），其中一段为弱纹理：
  * 段与段之间有视差跳变 —— 粗档的大窗口会跨边界取平均，于是边界附近会算错，
    这正是"可识别但算错"的一类误差，也是精算应当介入的地方；
  * 弱纹理段 —— 属于"不可识别"，精算也救不回来，用来验证"该放弃"的情形。
真值精确已知：先按 UP 倍超采样构造，按段做整数位移，再降采样回原尺寸。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import cv2
import numpy as np

UP = 4  # 超采样倍数：位移只能是 1/UP 像素的整数倍


@dataclass
class SyntheticPair:
    left: np.ndarray
    right: np.ndarray
    gt: np.ndarray          # float32 真值视差；-1 表示无真值（右边界/遮挡带）
    textured: np.ndarray    # bool，True=强纹理段（可识别）


DEFAULT_PLANES: Tuple[Tuple[float, float], ...] = ((0.36, 6.0), (0.30, 9.5), (0.34, 4.0))
DEFAULT_STRENGTH = (1.0, 1.0, 0.15)


def _texture_block(rng: np.random.Generator, h: int, w: int, strength: float) -> np.ndarray:
    """生成一块纹理：强纹理用高对比斑点+线段，弱纹理用低对比度噪声。"""
    block = np.full((h, w), 128.0, dtype=np.float32)
    if strength <= 0:
        return block
    if strength >= 0.5:
        for _ in range(max(1, int(h * w / 1200))):
            y, x = rng.integers(0, max(1, h - 8)), rng.integers(0, max(1, w - 8))
            block[y:y + 8, x:x + 8] = rng.integers(0, 256)
        for _ in range(20):
            p0 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            cv2.line(block, p0, p1, float(rng.integers(0, 256)), 3)
    else:
        block += strength * 10.0 * rng.standard_normal((h, w)).astype(np.float32)
    return np.clip(block, 0, 255)


def make_pair(
    height: int = 240, width: int = 480,
    planes: Sequence[Tuple[float, float]] = DEFAULT_PLANES,
    strength: Sequence[float] = DEFAULT_STRENGTH,
    seed: int = 0,
) -> SyntheticPair:
    """planes: [(宽度占比, 该段视差)]；strength: 各段纹理强度（0~1，<0.5 视为弱纹理）。"""
    if len(planes) != len(strength):
        raise ValueError("planes 与 strength 长度必须一致")
    if abs(sum(p[0] for p in planes) - 1.0) > 1e-6:
        raise ValueError("planes 的宽度占比之和必须为 1")

    rng = np.random.default_rng(seed)
    H, W = height * UP, width * UP

    edges, acc = [0], 0.0
    for frac, _ in planes:
        acc += frac
        edges.append(int(round(acc * W)))

    left_big = np.zeros((H, W), dtype=np.float32)
    for i, ((frac, disp), s) in enumerate(zip(planes, strength)):
        x0, x1 = edges[i], edges[i + 1]
        left_big[:, x0:x1] = _texture_block(rng, H, x1 - x0, s)

    right_big = np.full((H, W), 128.0, dtype=np.float32)
    gt = np.full((height, width), -1.0, dtype=np.float32)
    textured = np.zeros((height, width), dtype=bool)

    for i, ((frac, disp), s) in enumerate(zip(planes, strength)):
        x0, x1 = edges[i], edges[i + 1]
        shift = int(round(disp * UP))
        if shift * UP != int(round(disp * UP * UP)) or abs(disp * UP - shift) > 1e-9:
            raise ValueError(f"视差 {disp} 必须是 1/{UP} 的整数倍")
        # 右图中该段内容出现在更小的 x 处：R(x) = L(x + d)
        src = np.clip(np.arange(x0, x1) + shift, 0, W - 1)
        right_big[:, x0:x1] = left_big[:, src]

        gx0, gx1 = int(round(x0 / UP)), int(round(x1 / UP))
        gt[:, gx0:gx1] = disp
        if s >= 0.5:
            textured[:, gx0:gx1] = True

    # 遮挡/无边界的区域标为无真值：右边界 max(d) 列，以及每段左边界 1 px（遮挡）
    max_d = max(d for _, d in planes)
    margin = int(np.ceil(max_d)) + 2
    gt[:, width - margin:] = -1.0
    textured[:, width - margin:] = False
    for i in range(1, len(planes)):
        x = int(round(edges[i] / UP))
        gt[:, max(0, x - 1):min(width, x + 1)] = -1.0
        textured[:, max(0, x - 1):min(width, x + 1)] = False

    left = cv2.resize(left_big, (width, height), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right_big, (width, height), interpolation=cv2.INTER_AREA)
    return SyntheticPair(
        left=np.clip(left, 0, 255).astype(np.uint8),
        right=np.clip(right, 0, 255).astype(np.uint8),
        gt=gt, textured=textured,
    )
