# -*- coding: utf-8 -*-
"""读写工具。Windows 下路径可能含中文，因此不用 cv2.imread / imwrite，改用 imdecode / imencode。"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def imread(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    buf = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"无法解码图像：{p}")
    return img


def imwrite(path: str | Path, img: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"无法编码图像：{p}")
    buf.tofile(str(p))


def read_pfm(path: str | Path) -> np.ndarray:
    """读 Middlebury 的 .pfm 稠密真值（单通道 Pf 或多通道 PF），返回 float32 数组。"""
    p = Path(path)
    with open(p, "rb") as f:
        header = f.readline().rstrip()
        if header == b"PF":
            channels = 3
        elif header == b"Pf":
            channels = 1
        else:
            raise ValueError(f"不是 PFM 文件：{p}")
        width, height = (int(v) for v in f.readline().split())
        scale = float(f.readline().rstrip())
        endian = "<" if scale < 0 else ">"
        data = np.frombuffer(f.read(width * height * channels * 4), dtype=np.dtype(endian + "f4"))
    data = data.reshape((height, width, channels) if channels > 1 else (height, width))
    data = np.flipud(data).astype(np.float32)
    return data[:, :, 0] if channels == 3 else data


def read_array(path: str | Path) -> np.ndarray:
    """按扩展名读真值：.pfm / .npy / .png。"""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pfm":
        return read_pfm(p)
    if suffix == ".npy":
        return np.load(p).astype(np.float32)
    return imread(p).astype(np.float32)


def read_pair(left_path: str | Path, right_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    left = imread(left_path)
    right = imread(right_path)
    if left.shape[:2] != right.shape[:2]:
        raise ValueError(f"左右图尺寸不一致：{left.shape} vs {right.shape}")
    return left, right


def save_disp_preview(path: str | Path, disp: np.ndarray, max_disp: float | None = None) -> None:
    """把视差图存成可视化 PNG（0 无效显示为黑）。"""
    m = float(max_disp) if max_disp else float(disp[disp > 0].max() if (disp > 0).any() else 1.0)
    vis = np.clip(disp / max(m, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    vis[disp <= 0] = 0
    imwrite(path, vis)
