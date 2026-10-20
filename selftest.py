#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""自检：不依赖任何外部数据，验证整条链路能跑通且数字自洽。

断言只检查**结构性事实**（数量、单调性、口径一致），不假设"精算一定更好"——
精修是否有收益是实验问题，由 run_experiment.py 在真实数据上回答。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                            # noqa: E402

from stereo_min import metrics as M                           # noqa: E402
from stereo_min.config import ExpCfg, RefineCfg, SgbmCfg      # noqa: E402
from stereo_min.confidence import lrc_pair                    # noqa: E402
from stereo_min.pipeline import run_pair                      # noqa: E402
from stereo_min.selection import budget_pixels, select_top_k  # noqa: E402
from stereo_min.synthetic import make_pair                    # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def main() -> int:
    # ---- 自查模块：左一致 / 不一致 / 无对应 ----
    d = np.full((6, 8), 4.0, dtype=np.float32)
    u, valid, err = lrc_pair(d, d, thresh=1.0)
    check("LRC：一致视差的不确定度为 0", bool(np.all(u[:, 4:] == 0.0)))
    check("LRC：左侧无对应的像素判无效", bool((~valid[:, :4]).all()))
    # 左图像素 x=5 查的是右参考图在 x-d=1 处的视差；改那里才能触发不一致
    d_r = d.copy()
    d_r[2, 1] = 20.0
    _, valid2, _ = lrc_pair(d, d_r, thresh=1.0)
    check("LRC：超出阈值的像素被判无效", not bool(valid2[2, 5]))

    # ---- 挑选模块：数量精确 ----
    rng = np.random.default_rng(1)
    m = rng.random((40, 50)).astype(np.float32)
    for frac in (0.0, 0.05, 0.25, 1.0):
        mask = select_top_k(m, frac)
        check(f"挑选：budget={frac} 数量正确", int(mask.sum()) == budget_pixels(m, frac),
              f"实际 {int(mask.sum())}")

    # ---- 指标模块：口径 ----
    perfect = np.full((10, 10), 5.0, dtype=np.float32)
    gt = np.full((10, 10), 5.0, dtype=np.float32)
    s = M.summarize(perfect, gt)
    check("指标：完美视差 EPE=0", abs(s["epe"]) < 1e-6)
    check("指标：完美视差覆盖率=1", abs(s["coverage"] - 1.0) < 1e-9)
    s2 = M.summarize(np.zeros((10, 10), dtype=np.float32), gt, penalty=3.0)
    check("指标：预测缺失按 penalty 计", abs(s2["epe"] - 3.0) < 1e-6, f"epe={s2['epe']}")

    # ---- 端到端 ----
    pair = make_pair(height=160, width=320, seed=0)
    cfg = ExpCfg(
        coarse=SgbmCfg(name="coarse", num_disparities=64, block_size=7, mode="sgbm_3way",
                       uniqueness_ratio=10, speckle_window=100),
        fine=SgbmCfg(name="fine", num_disparities=64, block_size=3, mode="hh",
                     uniqueness_ratio=8, speckle_window=0),
        refine=RefineCfg(operator="roi", window=48, pad=12, search_margin=8),
        budgets=[0.0, 0.1, 1.0],
        coverage_levels=10,
        threads=1,
        seed=0,
    )
    res = run_pair(pair.left, pair.right, pair.gt, cfg, region=pair.textured)
    got = {r["variant"]: r for r in res.rows}
    check("端到端：A/B/全预算 均产出结果",
          all(k in got for k in ("A_all_coarse", "B_all_fine", "C_b1")))
    check("端到端：A 的 EPE 有限", np.isfinite(got["A_all_coarse"]["epe"]),
          f"epe={got['A_all_coarse']['epe']}")
    check("端到端：AURC 有限", np.isfinite(res.aurc), f"aurc={res.aurc:.4f}")
    check("端到端：覆盖率随预算单调不减",
          got["C_b0"]["coverage"] <= got["C_b0.1"]["coverage"] <= got["C_b1"]["coverage"] + 1e-9)
    check("端到端：全预算时 ROI 覆盖整图",
          got["C_b1"]["processed_px"] >= 0.9 * pair.left.shape[0] * pair.left.shape[1],
          f"processed={got['C_b1']['processed_px']}")
    check("端到端：精修耗时随预算单调不减",
          got["C_b0"]["refine_ms"] <= got["C_b0.1"]["refine_ms"] <= got["C_b1"]["refine_ms"] + 1e-6)
    check("端到端：全预算时自适应与随机选中同一批像素（结果必须完全一致）",
          abs(got["C_b1"]["epe"] - got["R_b1"]["epe"]) < 1e-9,
          f"C={got['C_b1']['epe']:.4f} R={got['R_b1']['epe']:.4f}")

    print()
    if FAILURES:
        print(f"自检失败 {len(FAILURES)} 项：{FAILURES}")
        return 1
    print("自检全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
