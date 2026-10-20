#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""离线入口：一对已校正图像 → 视差 → （可选）深度 → 指标，结果落盘。

用法（仓库根目录）：

    python run_experiment.py                              # 合成图冒烟（无设备时验证程序行为）
    python run_experiment.py --left L.png --right R.png --gt disp.pfm
    python run_experiment.py --left L.png --right R.png --config config/stereo_default.json

产物（--out 目录，默认 artifacts/offline_<时间戳>/）：
    metrics.json   汇总与运行环境（可复现入口）
    samples.csv    逐样本结果（本版本一次一对）
    disp.png       视差预览；(depth.png 仅在配置了 geometry 时生成)
    config.json    本次运行参数；meta.json 记录 git 提交与依赖版本
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from stereo_min.config import ConfigError, dump_config, load_config   # noqa: E402
from stereo_min.evaluate import evaluate_pair, result_as_row           # noqa: E402
from stereo_min.stereo_io import imwrite, read_array, read_pair, save_disp_preview  # noqa: E402
from stereo_min.synthetic import make_pair                             # noqa: E402


def git_state() -> tuple[str, bool]:
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


def auto_out_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("artifacts") / f"offline_{stamp}"


def parse_size(text: str) -> tuple[int, int]:
    h, w = (int(v) for v in text.lower().split("x"))
    return h, w


def write_depth_preview(path: Path, depth: np.ndarray) -> None:
    """深度预览：有效像素线性映射为 0~255（近=亮），无效为黑。仅用于目视检查，不参与计算。"""
    vis = np.zeros(depth.shape, dtype=np.uint8)
    finite = np.isfinite(depth)
    if finite.any():
        lo, hi = float(depth[finite].min()), float(depth[finite].max())
        span = max(hi - lo, 1e-6)
        vis[finite] = np.clip((hi - depth[finite]) / span * 255.0, 0, 255).astype(np.uint8)
    imwrite(path, vis)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="离线双目视差评测")
    ap.add_argument("--config", default=str(ROOT / "config" / "stereo_synthetic.json"))
    ap.add_argument("--left", default=None, help="左图路径（已校正）")
    ap.add_argument("--right", default=None, help="右图路径（已校正）")
    ap.add_argument("--gt", default=None, help="真值视差（.pfm / .npy / .png）")
    ap.add_argument("--gt-depth", default=None, help="参考深度（.npy / .pfm），用于深度指标")
    ap.add_argument("--size", default="240x480", help="合成图尺寸 HxW")
    ap.add_argument("--planes", default="0.36:6.0,0.30:9.5,0.34:4.0",
                    help="合成图各段“宽度占比:视差”，逗号分隔")
    ap.add_argument("--out", default=None, help="输出目录；默认 artifacts/offline_<时间戳>")
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as ex:
        print(f"配置错误：{ex}")
        return 2

    out_dir = Path(args.out) if args.out else auto_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    if (args.left is None) != (args.right is None):
        print("错误：--left 与 --right 必须同时提供")
        return 2

    if args.left and args.right:
        left, right = read_pair(args.left, args.right)
        gt = read_array(args.gt) if args.gt else None
        gt_depth = read_array(args.gt_depth) if args.gt_depth else None
        pair_id = Path(args.left).stem
        source = {"kind": "files", "left": args.left, "right": args.right,
                  "gt": args.gt, "gt_depth": args.gt_depth}
    else:
        h, w = parse_size(args.size)
        planes = []
        for chunk in args.planes.split(","):
            frac, disp = chunk.split(":")
            planes.append((float(frac), float(disp)))
        pair = make_pair(height=h, width=w, planes=tuple(planes), seed=0)
        left, right, gt, gt_depth = pair.left, pair.right, pair.gt, None
        pair_id = "synthetic"
        source = {"kind": "synthetic", "size": [h, w], "planes": [list(p) for p in planes]}

    res = evaluate_pair(left, right, cfg, gt=gt, gt_depth=gt_depth)

    (out_dir / "metrics.json").write_text(json.dumps({
        "pair_id": pair_id,
        "metrics": {k: (None if isinstance(v, float) and v != v else v) for k, v in res.metrics.items()},
        "config": dump_config(cfg),
        "source": source,
        "note": "指标只在“预测有效且真值有效”的像素上统计；无效预测不以 0 混入均值。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    row = result_as_row(pair_id, res)
    with (out_dir / "samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    save_disp_preview(out_dir / "disp.png", res.disparity)
    if res.depth is not None:
        write_depth_preview(out_dir / "depth.png", res.depth)

    (out_dir / "config.json").write_text(
        json.dumps(dump_config(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    sha, dirty = git_state()
    (out_dir / "meta.json").write_text(json.dumps({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": sha,
        "git_dirty": dirty,
        "image_hw": [int(left.shape[0]), int(left.shape[1])],
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "opencv": cv2.__version__},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"pair={pair_id}  输出目录={out_dir}")
    for key in ("width", "height", "valid_ratio", "coverage", "epe", "bad1", "p95", "disparity_ms"):
        if key in res.metrics:
            print(f"  {key:14s} {res.metrics[key]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
