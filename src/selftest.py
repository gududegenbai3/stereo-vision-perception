#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""零数据自检：不依赖外部数据，验证视差 → 掩码 → 深度 → 指标这条链路自洽。

断言只检查结构性事实（数量、口径、公式），不判断精度好坏——精度需要真实数据与独立参考。
运行：python src/selftest.py（仓库根目录执行）
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import numpy as np                                       # noqa: E402

from stereo_min.config import ConfigError, GeometryCfg, RunCfg, SgbmCfg   # noqa: E402
from stereo_min.depth import disparity_to_depth, valid_mask               # noqa: E402
from stereo_min.evaluate import evaluate_pair                             # noqa: E402
from stereo_min import metrics as M                                       # noqa: E402
from stereo_min.synthetic import make_pair                                # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def raises(fn, exc_type: type[BaseException]) -> bool:
    try:
        fn()
    except exc_type:
        return True
    except Exception:
        return False
    return False


def main() -> int:
    # ---- 配置校验：非法输入必须报错 ----
    check("配置：num_disparities 非 16 倍数被拒",
          raises(lambda: SgbmCfg(num_disparities=100).validate(), ConfigError))
    check("配置：block_size 为偶数被拒",
          raises(lambda: SgbmCfg(block_size=4).validate(), ConfigError))
    check("配置：几何参数只给一半被拒",
          raises(lambda: GeometryCfg(fx_px=700.0).validate(), ConfigError))

    # ---- 深度公式：Z = fx * B / d ----
    geo = GeometryCfg(fx_px=1000.0, baseline_m=0.1)
    disp = np.array([[10.0, 0.0], [np.nan, -1.0]], dtype=np.float32)
    depth, valid = disparity_to_depth(disp, geo)
    check("深度：Z=fxB/d 数值正确", abs(float(depth[0, 0]) - 10.0) < 1e-5, f"Z={depth[0, 0]:.4f}")
    check("深度：无效视差得 NaN", bool(np.isnan(depth[0, 1]) and np.isnan(depth[1, 0]) and np.isnan(depth[1, 1])))
    check("深度：有效掩码只含正视差", bool(valid[0, 0]) and not bool(valid[1, 1]))

    # ---- 掩码：边界裁剪 ----
    m = valid_mask(np.ones((10, 10), dtype=np.float32), roi_margin_px=2)
    check("掩码：margin 裁掉边界", int(m.sum()) == 36, f"实际 {int(m.sum())}")

    # ---- 指标口径：完美视差、无效不混入均值 ----
    gt = np.full((6, 6), 5.0, dtype=np.float32)
    perfect = np.full((6, 6), 5.0, dtype=np.float32)
    mm = M.disparity_metrics(perfect, gt)
    check("指标：完美视差 EPE=0", abs(mm["epe"]) < 1e-6)
    check("指标：完美视差覆盖率=1", abs(mm["coverage"] - 1.0) < 1e-9)

    half = np.zeros((6, 6), dtype=np.float32)
    half[:, :3] = 5.0
    mh = M.disparity_metrics(half, gt)
    check("指标：无效预测不进误差均值", abs(mh["epe"]) < 1e-6, f"epe={mh['epe']}")
    check("指标：覆盖率反映无效比例", abs(mh["coverage"] - 0.5) < 1e-9, f"cov={mh['coverage']}")

    # ---- 端到端：合成图跑通 ----
    pair = make_pair(height=160, width=320, seed=0)
    cfg = RunCfg(sgbm=SgbmCfg(num_disparities=64, block_size=7), threads=1)
    res = evaluate_pair(pair.left, pair.right, cfg, gt=pair.gt)
    check("端到端：输出视差形状正确", res.disparity.shape == pair.left.shape[:2])
    check("端到端：EPE 有限且非负", bool(np.isfinite(res.metrics["epe"]) and res.metrics["epe"] >= 0),
          f"epe={res.metrics['epe']:.3f}")
    check("端到端：覆盖率在 (0,1]", 0.0 < res.metrics["coverage"] <= 1.0,
          f"cov={res.metrics['coverage']:.3f}")
    check("端到端：无几何参数时不输出深度", res.depth is None)

    cfg_geo = RunCfg(sgbm=cfg.sgbm, geometry=geo, threads=1)
    res2 = evaluate_pair(pair.left, pair.right, cfg_geo, gt=pair.gt)
    check("端到端：有几何参数时输出深度", res2.depth is not None and np.isfinite(res2.depth).any())

    print()
    if FAILURES:
        print(f"自检失败 {len(FAILURES)} 项：{FAILURES}")
        return 1
    print("自检全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
