#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验收门：一条命令跑完全部检查，任一失败即非零退出（CI 调用同一条）。

    python src/check.py             # 全跑
    python src/check.py --quick     # 只跑自检 + 单元测试
    python src/check.py --out <目录>  # 指定证据目录（默认 artifacts/_gate）

顺序：① 零数据自检 → ② 单元测试 → ③ 小尺寸实验 → ④⑤ 卫生检查
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent        # <repo>/src
ROOT = SRC.parent                            # <repo>


def run(cmd: list[str], title: str) -> tuple[int, float]:
    print(f"\n=== {title} ===")
    print("$", " ".join(cmd))
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT)
    dt = time.perf_counter() - t0
    print(f"--- {title}: exit={proc.returncode} ({dt:.1f}s) ---")
    return proc.returncode, dt


def evidence_dir(explicit: str | None) -> str:
    """证据目录：默认 artifacts/_gate；不可写时用 --out 指定。"""
    if explicit:
        Path(explicit).mkdir(parents=True, exist_ok=True)
        return explicit
    target = ROOT / "artifacts"
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as ex:
        print(f"[FAIL] 无法写 {target}（{ex}）；用 --out <可写目录> 重跑")
        raise SystemExit(1)
    return "artifacts/_gate"


def hygiene_checks() -> list[str]:
    problems: list[str] = []

    # ① 暂存区不能含产物类文件
    try:
        out = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT,
                             capture_output=True, text=True, timeout=30)
        if out.returncode == 0 and out.stdout.strip():
            forbidden = [p for p in out.stdout.splitlines()
                         if p.startswith(("artifacts/", "data/", "outputs/", "logs/"))
                         or p.endswith((".npy", ".npz", ".ply"))]
            if forbidden:
                problems.append("暂存区包含不该提交的产物：" + ", ".join(forbidden[:5]))
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    # ② 源码与配置里不能出现本机绝对路径
    pat = re.compile(r"[A-Za-z]:\\\\|[A-Za-z]:\\Users|/home/|/mnt/")
    targets = list((SRC / "stereo_min").rglob("*.py")) + list((ROOT / "config").glob("*.json"))
    for p in targets:
        text = p.read_text(encoding="utf-8", errors="replace")
        if pat.search(text):
            problems.append(f"{p.relative_to(ROOT)} 里有本机绝对路径")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="验收门")
    ap.add_argument("--quick", action="store_true", help="跳过实验步骤")
    ap.add_argument("--out", default=None, help="实验证据目录（默认 artifacts/_gate）")
    args = ap.parse_args(argv)

    steps: list[tuple[str, int]] = []

    rc, _ = run([sys.executable, "src/selftest.py"], "① 零数据自检")
    steps.append(("selftest", rc))
    if rc != 0:
        return report(steps)

    rc, _ = run([sys.executable, "src/run_tests.py"], "② 单元测试（有 pytest 用 pytest）")
    steps.append(("unit_tests", rc))
    if rc != 0:
        return report(steps)

    if not args.quick:
        out = evidence_dir(args.out)
        rc, _ = run([sys.executable, "run_experiment.py", "--out", out], "③ 小尺寸实验")
        steps.append(("experiment", rc))
        if rc != 0:
            return report(steps)

    print("\n=== ④⑤ 卫生检查 ===")
    problems = hygiene_checks()
    for msg in problems:
        print("  [FAIL]", msg)
    if not problems:
        print("  [PASS] 暂存区与源码路径检查通过")
    steps.append(("hygiene", 1 if problems else 0))
    return report(steps)


def report(steps: list[tuple[str, int]]) -> int:
    print("\n===== 验收门结果 =====")
    for name, rc in steps:
        print(f"  {name:12s} {'PASS' if rc == 0 else 'FAIL'}")
    failed = [n for n, rc in steps if rc != 0]
    if failed:
        print(f"\n未通过：{', '.join(failed)}")
        return 1
    print("\n全部门通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
