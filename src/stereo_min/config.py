# -*- coding: utf-8 -*-
"""配置对象：全部参数从 JSON 读入，代码里不写死数值。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

MODE_ALIASES = {
    "sgbm": "MODE_SGBM",
    "sgbm_3way": "MODE_SGBM_3WAY",
    "hh": "MODE_HH",
    "hh4": "MODE_HH4",
}


@dataclass
class SgbmCfg:
    """一组 SGBM 参数。粗档/贵档只是同一类的两组取值。"""

    name: str = "coarse"
    num_disparities: int = 128
    block_size: int = 7
    mode: str = "sgbm_3way"
    uniqueness_ratio: int = 10
    disp12_max_diff: int = 1
    speckle_window: int = 100
    speckle_range: int = 2
    pre_filter_cap: int = 31
    p1_scale: int = 8
    p2_scale: int = 32

    def p1(self) -> int:
        return self.p1_scale * self.block_size ** 2

    def p2(self) -> int:
        return self.p2_scale * self.block_size ** 2

    def validate(self) -> None:
        if self.num_disparities % 16 != 0:
            raise ValueError(f"{self.name}: num_disparities 必须是 16 的倍数，当前 {self.num_disparities}")
        if self.block_size % 2 == 0 or self.block_size < 3:
            raise ValueError(f"{self.name}: block_size 必须是 >=3 的奇数，当前 {self.block_size}")
        if self.mode not in MODE_ALIASES:
            raise ValueError(f"{self.name}: mode 只能是 {sorted(MODE_ALIASES)}")


@dataclass
class RefineCfg:
    """精算算子配置。roi: 只重算被选中像素所在窗口；global: 整图重算（上界参考）。"""

    operator: str = "roi"
    window: int = 64
    pad: int = 16
    search_margin: int = 8

    def validate(self) -> None:
        if self.operator not in ("roi", "global", "none"):
            raise ValueError("refine.operator 只能是 roi / global / none")
        if self.window <= 0 or self.pad < 0:
            raise ValueError("refine.window 必须为正，refine.pad 不能为负")
        if self.search_margin < 0:
            raise ValueError("refine.search_margin 不能为负")


@dataclass
class ExpCfg:
    coarse: SgbmCfg = field(default_factory=lambda: SgbmCfg(name="coarse"))
    fine: SgbmCfg = field(
        default_factory=lambda: SgbmCfg(
            name="fine", num_disparities=256, block_size=3, mode="hh", uniqueness_ratio=8
        )
    )
    refine: RefineCfg = field(default_factory=RefineCfg)
    budgets: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2, 0.4, 1.0])
    lrc_thresh: float = 1.0
    coverage_levels: int = 20
    texture_tiebreak: bool = True
    tie_weight: float = 0.01
    threads: int = 1
    seed: int = 0

    def validate(self) -> None:
        self.coarse.validate()
        self.fine.validate()
        self.refine.validate()
        if not self.budgets:
            raise ValueError("budgets 不能为空")
        for b in self.budgets:
            if not 0.0 <= b <= 1.0:
                raise ValueError(f"budget 必须落在 [0,1]，当前 {b}")
        if sorted(self.budgets) != list(self.budgets):
            raise ValueError("budgets 必须递增")


def load_config(path: str | Path) -> ExpCfg:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = ExpCfg(
        coarse=SgbmCfg(**raw.get("coarse", {})),
        fine=SgbmCfg(**raw.get("fine", {})),
        refine=RefineCfg(**raw.get("refine", {})),
        budgets=raw.get("budgets", ExpCfg().budgets),
        lrc_thresh=raw.get("lrc_thresh", 1.0),
        coverage_levels=raw.get("coverage_levels", 20),
        texture_tiebreak=raw.get("texture_tiebreak", True),
        tie_weight=raw.get("tie_weight", 0.01),
        threads=raw.get("threads", 1),
        seed=raw.get("seed", 0),
    )
    cfg.validate()
    return cfg


def dump_config(cfg: ExpCfg) -> dict:
    return asdict(cfg)
