#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""命令行入口：一条命令跑完整实验，并把结果全部落盘。

用法（三条命令任选其一）：

    python run_experiment.py                                  # 内置合成图，零外部数据
    python run_experiment.py --planes 0.36:6.0,0.30:9.5,0.34:4.0 --size 320x640
    python run_experiment.py --left L.png --right R.png --gt disp.pfm --config configs/default.json

产物（写到 --out 目录，默认 artifacts/run）：
    rows.csv            每个变体一行的指标
    risk_coverage.csv   风险-覆盖率曲线
    budget_curve.png    预算-误差曲线（C 自适应 vs R 随机对照）
    risk_coverage.png   风险-覆盖率曲线图
    config.json         本次运行的完整配置（可复现）
    summary.json        摘要（AURC 等）
    disp_A/B/C.png      视差预览
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from stereo_min import metrics as M                                       # noqa: E402
from stereo_min.config import dump_config, load_config                    # noqa: E402
from stereo_min.pipeline import run_pair                                  # noqa: E402
from stereo_min.plotting import plot_budget_curve, plot_cost_curve, plot_risk_coverage     # noqa: E402
from stereo_min.stereo_io import read_array, read_pair, save_disp_preview  # noqa: E402
from stereo_min.synthetic import DEFAULT_PLANES, DEFAULT_STRENGTH, make_pair  # noqa: E402

FIELDS = ["variant", "budget", "coverage", "epe", "epe_valid_only", "d1", "bad2", "p95",
          "coarse_ms", "selfcheck_ms", "refine_ms", "total_ms", "rois", "processed_px",
          "valid_px", "ms_per_kvalid"]


def parse_size(text: str) -> tuple[int, int]:
    h, w = (int(v) for v in text.lower().split("x"))
    return h, w


def parse_planes(text: str) -> tuple[tuple[float, float], ...]:
    """'0.36:6.0,0.30:9.5,0.34:4.0' -> ((0.36, 6.0), (0.30, 9.5), (0.34, 4.0))"""
    planes = []
    for chunk in text.split(","):
        frac, disp = chunk.split(":")
        planes.append((float(frac), float(disp)))
    return tuple(planes)


def parse_strength(text: str) -> tuple[float, ...]:
    return tuple(float(v) for v in text.split(","))


def git_state() -> tuple[str, bool]:
    """返回 (短 commit sha, 是否有未提交改动)；不在 git 仓库里时返回 ("nogit", False)。"""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=15)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, timeout=15)
        if sha.returncode != 0:
            return "nogit", False
        return sha.stdout.strip(), bool(dirty.stdout.strip())
    except (FileNotFoundError, subprocess.SubprocessError):
        return "nogit", False


def auto_out_dir(topic: str) -> Path:
    """证据归档命名：artifacts/<YYYYMMDD-HHMM>_<topic>_<sha7>/（口径见 docs/code.md）。"""
    sha, _ = git_state()
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    safe_topic = "".join(c if c.isalnum() or c in "-_" else "-" for c in topic) or "run"
    return Path("artifacts") / f"{stamp}_{safe_topic}_{sha[:7]}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="置信度驱动的主动深度感知：最小可运行实验")
    ap.add_argument("--config", default=str(ROOT / "configs" / "synthetic.json"))
    ap.add_argument("--left", help="左图路径（已校正）")
    ap.add_argument("--right", help="右图路径（已校正）")
    ap.add_argument("--gt", help="真值视差路径（.pfm / .npy / .png）")
    ap.add_argument("--size", default="240x480", help="合成图尺寸 HxW")
    ap.add_argument("--planes", default="0.36:6.0,0.30:9.5,0.34:4.0",
                    help="合成图各段“宽度占比:视差”，逗号分隔")
    ap.add_argument("--strength", default="1,1,0.15",
                    help="合成图各段纹理强度（0~1，<0.5 视为弱纹理）")
    ap.add_argument("--penalty", type=float, default=3.0,
                    help="预测缺失时计入的误差（px）；口径必须写进报告")
    ap.add_argument("--topic", default="min-refine", help="证据目录名里的主题词")
    ap.add_argument("--out", default=None,
                    help="输出目录；默认 artifacts/<YYYYMMDD-HHMM>_<topic>_<sha7>")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out_dir = Path(args.out) if args.out else auto_out_dir(args.topic)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.left and args.right:
        left, right = read_pair(args.left, args.right)
        if not args.gt:
            ap.error("真实数据必须提供 --gt 真值视差")
        gt = read_array(args.gt)
        region = None
        tag = "real"
        data_desc = {"left": args.left, "right": args.right, "gt": args.gt}
    else:
        planes = parse_planes(args.planes)
        strength = parse_strength(args.strength)
        h, w = parse_size(args.size)
        pair = make_pair(height=h, width=w, planes=planes, strength=strength, seed=cfg.seed)
        left, right, gt, region, tag = pair.left, pair.right, pair.gt, pair.textured, "synthetic"
        data_desc = {"planes": planes, "strength": strength, "size": [h, w]}

    res = run_pair(left, right, gt, cfg, penalty=args.penalty, region=region)

    with (out_dir / "rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in res.rows:
            writer.writerow({k: r.get(k, "") for k in FIELDS})

    cov, risk = res.risk_curve
    with (out_dir / "risk_coverage.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["coverage", "risk"])
        for c, r in zip(cov, risk):
            writer.writerow([f"{c:.4f}", f"{r:.4f}"])

    plot_budget_curve(res.rows, out_dir / "budget_curve.png")
    plot_cost_curve(res.rows, out_dir / "cost_curve.png")
    plot_risk_coverage(cov, risk, out_dir / "risk_coverage.png", res.aurc)

    (out_dir / "config.json").write_text(
        json.dumps(dump_config(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps({
        "data": tag,
        "aurc": round(res.aurc, 4),
        "gt_pixels": int(res.gt_valid.sum()) if res.gt_valid is not None else 0,
        "rows": len(res.rows),
        "penalty_px": args.penalty,
        "note": f"epe/d1 在 GT 有效像素上统计；预测缺失按 penalty={args.penalty:g}px 计。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 证据元数据：任何一个报告数字都能反查到确切的代码状态与环境
    import cv2
    import numpy as np
    sha, dirty = git_state()
    (out_dir / "meta.json").write_text(json.dumps({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": sha,
        "git_dirty": dirty,
        "data": tag,
        "data_source": data_desc,
        "image_hw": [int(left.shape[0]), int(left.shape[1])],
        "penalty_px": args.penalty,
        "config_path": str(args.config),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "opencv": cv2.__version__},
        "outputs": ["rows.csv", "risk_coverage.csv", "cost_curve.png",
                    "risk_coverage.png", "config.json", "summary.json", "meta.json"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    for key in ("A", "B"):
        save_disp_preview(out_dir / f"disp_{key}.png", res.variants[key].disp)
    full_budget = [k for k in res.variants if k.startswith("C_")]
    if full_budget:
        save_disp_preview(out_dir / "disp_C.png", res.variants[full_budget[-1]].disp)

    print(f"数据: {tag}   输出目录: {out_dir}")
    print(f"{'变体':<18}{'预算':>6}{'覆盖率':>9}{'EPE':>9}{'D1':>9}"
          f"{'粗算':>8}{'自查':>8}{'精算':>8}{'总ms':>8}{'ROI':>5}{'ms/k有效':>10}")
    for r in res.rows:
        print(f"{r['variant']:<18}{r['budget']:>6.2f}{r['coverage']:>9.3f}{r['epe']:>9.3f}"
              f"{r['d1']:>9.3f}{r['coarse_ms']:>8.1f}{r['selfcheck_ms']:>8.1f}"
              f"{r['refine_ms']:>8.1f}{r['total_ms']:>8.1f}{r['rois']:>5}{r['ms_per_kvalid']:>10.2f}")
    print(f"AURC(风险-覆盖率) = {res.aurc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
