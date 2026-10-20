#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""跑单元测试：有 pytest 就用 pytest，没有则用内置运行器（两者跑同一批断言）。

CI 一定走 pytest；本机没装 pytest 时不因此卡住，但输出会标明用了哪个后端。
运行：python src/run_tests.py（仓库根目录执行）
"""
from __future__ import annotations

import importlib.util
import inspect
import subprocess
import sys
import types
from pathlib import Path

SRC = Path(__file__).resolve().parent        # <repo>/src
ROOT = SRC.parent                            # <repo>
TESTS = SRC / "tests"
sys.path.insert(0, str(SRC))


def run_with_pytest() -> int:
    print("后端：pytest")
    return subprocess.run([sys.executable, "-m", "pytest", str(TESTS), "-q"], cwd=ROOT).returncode


class _Approx:
    def __init__(self, expected: float, rel: float = 1e-6, abs_: float = 1e-12) -> None:
        self.expected, self.rel, self.abs = expected, rel, abs_

    def __eq__(self, other: float) -> bool:
        return abs(other - self.expected) <= max(self.abs, self.rel * abs(self.expected))

    def __repr__(self) -> str:
        return f"approx({self.expected})"


class _Raises:
    def __init__(self, exc: type[BaseException]) -> None:
        self.exc = exc

    def __enter__(self) -> "_Raises":
        return self

    def __exit__(self, et, ev, tb) -> bool:
        if et is None:
            raise AssertionError(f"期望抛出 {self.exc.__name__}，但没有异常")
        return issubclass(et, self.exc)


def install_shim() -> None:
    fake = types.ModuleType("pytest")
    fake.approx = lambda v, **kw: _Approx(v, kw.get("rel", 1e-6), kw.get("abs", 1e-12))
    fake.raises = lambda exc: _Raises(exc)
    sys.modules["pytest"] = fake


def run_builtin() -> int:
    print("后端：内置运行器（未安装 pytest；CI 中会用 pytest）")
    install_shim()
    total = failed = 0
    for path in sorted(TESTS.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as ex:
            print(f"[FAIL] {path.name} 导入失败：{type(ex).__name__}: {ex}")
            failed += 1
            continue
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            total += 1
            try:
                fn()
                print(f"[PASS] {path.stem}::{name}")
            except Exception as ex:
                failed += 1
                print(f"[FAIL] {path.stem}::{name} -> {type(ex).__name__}: {ex}")
    print(f"\n{total} tests, {failed} failed")
    return 1 if failed else 0


def main() -> int:
    try:
        import pytest  # noqa: F401
        return run_with_pytest()
    except ImportError:
        return run_builtin()


if __name__ == "__main__":
    raise SystemExit(main())
