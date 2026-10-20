# -*- coding: utf-8 -*-
"""画图（可选依赖 matplotlib）。缺库时只写 CSV，不报错。"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def plot_budget_curve(rows: List[Dict[str, float]], path: str | Path, title: str = "预算-误差曲线") -> bool:
    plt = _plt()
    if plt is None:
        return False
    def series(tag: str) -> Tuple[List[float], List[float]]:
        xs, ys = [], []
        for r in rows:
            if str(r["variant"]).startswith(tag + "_b"):
                xs.append(float(r["budget"]))
                ys.append(float(r["epe"]))
        return xs, ys

    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    for tag, label, style in (("C", "C 自适应挑选", "o-"), ("R", "R 随机挑选（对照）", "s--")):
        xs, ys = series(tag)
        if xs:
            order = np.argsort(xs)
            ax.plot(np.array(xs)[order], np.array(ys)[order], style, label=label)
    for name, label in (("A_all_coarse", "A 全粗算"), ("B_all_fine", "B 全精算")):
        for r in rows:
            if r["variant"] == name:
                ax.axhline(float(r["epe"]), linestyle=":", linewidth=1, label=label, alpha=0.7)
    ax.set_xlabel("精修预算（占总像素比例）")
    ax.set_ylabel("EPE（px，视差域）")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return True


def plot_cost_curve(rows: List[Dict[str, float]], path: str | Path, title: str = "精度-开销（报告主图）") -> bool:
    """横轴总耗时、纵轴 EPE，把 A/B/C/R 画在同一张图上——这就是精度-开销权衡图。"""
    plt = _plt()
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    for tag, label, style in (("C_b", "C 自适应挑选", "o-"), ("R_b", "R 随机挑选（对照）", "s--")):
        pts = [(float(r["total_ms"]), float(r["epe"])) for r in rows if str(r["variant"]).startswith(tag)]
        if pts:
            pts.sort()
            xs, ys = zip(*pts)
            ax.plot(xs, ys, style, label=label)
    for name, label, marker in (("A_all_coarse", "A 全粗算", "*"), ("B_all_fine", "B 全精算", "P")):
        for r in rows:
            if r["variant"] == name:
                ax.scatter([float(r["total_ms"])], [float(r["epe"])], marker=marker, s=120, label=label, zorder=5)
    ax.set_xlabel("单帧总耗时（算法部分，ms）")
    ax.set_ylabel("EPE（px，视差域）")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return True


def plot_risk_coverage(cov: np.ndarray, risk: np.ndarray, path: str | Path,
                       aurc_value: float | None = None) -> bool:
    plt = _plt()
    if plt is None or cov.size == 0:
        return False
    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    ax.plot(cov, risk, "o-")
    ax.set_xlabel("覆盖率（按置信度从高到低保留的像素比例）")
    ax.set_ylabel("选择性风险（EPE, px）")
    title = "风险-覆盖率曲线"
    if aurc_value is not None and np.isfinite(aurc_value):
        title += f"（AURC={aurc_value:.3f}）"
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return True
