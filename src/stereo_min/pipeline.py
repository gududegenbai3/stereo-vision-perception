# -*- coding: utf-8 -*-
"""把四个模块串起来：粗算 → 自查 → 挑选 → 精算 → 评测。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import metrics as M
from .config import ExpCfg
from .confidence import LARGE, lrc_pair, texture_map
from .matcher import compute, to_gray
from .refine import refine
from .selection import select_by


def selection_score(uncertainty: np.ndarray, gray: np.ndarray, weight: float) -> np.ndarray:
    """挑选用的排序键。

    不确定度会大量饱和：所有判为无效的像素都取同一个大值（LARGE），此时按它排序等于
    按像素序号排，排序失去意义。做法是只在**饱和像素内部**用局部纹理强度做次级键：
    同样不可靠时，优先精修有纹理、更可能被救回来的区域，弱纹理/无纹理区排后面。
    有效像素的相对顺序保持不变（否则会破坏风险-覆盖率曲线的排序语义）。
    """
    if weight <= 0:
        return uncertainty
    tex = texture_map(gray)
    scale = float(np.percentile(tex, 99)) + 1e-6
    texn = np.clip(tex / scale, 0.0, 1.0).astype(np.float32)
    score = uncertainty.copy()
    saturated = score >= LARGE
    score[saturated] = LARGE + weight * texn[saturated]
    return score


@dataclass
class Variant:
    name: str
    budget: float
    disp: np.ndarray
    coarse_ms: float = 0.0
    selfcheck_ms: float = 0.0
    refine_ms: float = 0.0
    rois: int = 0
    processed_px: int = 0

    @property
    def total_ms(self) -> float:
        return self.coarse_ms + self.selfcheck_ms + self.refine_ms


@dataclass
class PairResult:
    rows: List[Dict[str, float]] = field(default_factory=list)
    variants: Dict[str, Variant] = field(default_factory=dict)
    uncertainty: Optional[np.ndarray] = None
    gt_valid: Optional[np.ndarray] = None
    risk_curve: Tuple[np.ndarray, np.ndarray] = (np.array([]), np.array([]))
    aurc: float = float("nan")


def _row(v: Variant, gt: np.ndarray, penalty: float) -> Dict[str, float]:
    s = M.summarize(v.disp, gt, penalty=penalty)
    valid_px = int((v.disp > 0).sum())
    return {
        "variant": v.name,
        "budget": round(v.budget, 4),
        "coverage": round(s["coverage"], 4),
        "epe": round(s["epe"], 4),
        "epe_valid_only": round(s["epe_valid_only"], 4),
        "d1": round(s["d1"], 4),
        "bad2": round(s["bad2"], 4),
        "p95": round(s["p95"], 4),
        "coarse_ms": round(v.coarse_ms, 2),
        "selfcheck_ms": round(v.selfcheck_ms, 2),
        "refine_ms": round(v.refine_ms, 2),
        "total_ms": round(v.total_ms, 2),
        "rois": v.rois,
        "processed_px": v.processed_px,
        "valid_px": valid_px,
        # 曲线三：每千个"输出有效"像素要花多少毫秒（能耗口径待硬件接入后同式替换）
        "ms_per_kvalid": round(v.total_ms / (valid_px / 1000.0), 4) if valid_px else float("nan"),
    }


def run_pair(
    left: np.ndarray, right: np.ndarray, gt: np.ndarray, cfg: ExpCfg,
    penalty: float = 3.0, region: Optional[np.ndarray] = None,
) -> PairResult:
    """一次完整实验：A（全粗）→ B（全精，上界）→ 对每个预算跑 C（自适应）与 R（随机对照）。"""
    rng = np.random.default_rng(cfg.seed)
    res = PairResult()

    # ---- 粗算一次（A 基线的时间口径就是这一次） ----
    d_coarse, t_coarse = compute(left, right, cfg.coarse)

    # ---- 自查：需要再跑一遍右参考，这部分是"自查的额外成本" ----
    d_right, t_right = compute(right, left, cfg.coarse)
    uncertainty, _valid_lrc, _err = lrc_pair(d_coarse, d_right, cfg.lrc_thresh)

    res.uncertainty = uncertainty
    res.gt_valid = gt > 0

    score = selection_score(uncertainty, to_gray(left),
                            cfg.tie_weight if cfg.texture_tiebreak else 0.0)

    a = Variant("A_all_coarse", 0.0, d_coarse, coarse_ms=t_coarse, selfcheck_ms=0.0)
    res.rows.append(_row(a, gt, penalty))
    res.variants["A"] = a

    # ---- B：整图换贵参数（精度上界参考） ----
    b = Variant("B_all_fine", 1.0, d_coarse.copy(), coarse_ms=t_coarse)
    b_res = refine(left, right, d_coarse, np.ones_like(d_coarse, dtype=bool), cfg.fine,
                   type(cfg.refine)(operator="global"))
    b.disp = b_res.disp
    b.refine_ms = b_res.time_ms
    b.rois = b_res.rois
    b.processed_px = b_res.processed_px
    res.rows.append(_row(b, gt, penalty))
    res.variants["B"] = b

    # ---- C / R：同一预算下自适应挑选 vs 随机挑选 ----
    for budget in cfg.budgets:
        for strategy, tag in (("adaptive", "C"), ("random", "R")):
            if budget <= 0.0 and strategy == "random":
                continue  # 预算为 0 时两者等价，只留 C 一行
            sel, _ = select_by(score, budget, strategy, rng)
            v = Variant(f"{tag}_b{budget:g}", budget, d_coarse.copy(),
                        coarse_ms=t_coarse, selfcheck_ms=t_right)
            out = refine(left, right, d_coarse, sel, cfg.fine, cfg.refine)
            v.disp = out.disp
            v.refine_ms = out.time_ms
            v.rois = out.rois
            v.processed_px = out.processed_px
            res.rows.append(_row(v, gt, penalty))
            res.variants[f"{tag}_b{budget:g}"] = v

    # ---- 风险-覆盖率曲线（用粗算结果 + 自查信号） ----
    err_map = M.error_map(d_coarse, gt, penalty=penalty)
    # 风险-覆盖率曲线用"置信度本身"排序（评估自查信号的质量），
    # 而挑选精修目标用 score（额外引入"可救性"先验），两者用途不同，不能混。
    cov, risk = M.risk_coverage(uncertainty, err_map, res.gt_valid, levels=cfg.coverage_levels)
    res.risk_curve = (cov, risk)
    res.aurc = M.aurc(cov, risk)

    if region is not None:
        for name, v in (("A", res.variants["A"]), ("B", res.variants["B"])):
            rs = M.region_summarize(v.disp, gt, region, penalty=penalty)
            valid_px = int((v.disp > 0).sum())
            res.rows.append({
                "variant": f"{name}_region_textured", "budget": 0.0,
                "coverage": round(rs["coverage"], 4), "epe": round(rs["epe"], 4),
                "d1": round(rs["d1"], 4), "epe_valid_only": float("nan"),
                "bad2": float("nan"), "p95": float("nan"),
                "coarse_ms": round(v.coarse_ms, 2), "selfcheck_ms": round(v.selfcheck_ms, 2),
                "refine_ms": round(v.refine_ms, 2), "total_ms": round(v.total_ms, 2),
                "rois": v.rois, "processed_px": v.processed_px, "valid_px": valid_px,
                "ms_per_kvalid": round(v.total_ms / (valid_px / 1000.0), 4) if valid_px else float("nan"),
                "region": "textured",
            })
    return res
