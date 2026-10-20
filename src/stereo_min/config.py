# -*- coding: utf-8 -*-
"""运行配置：全部来自 `config/*.json`，此处只做类型承载与校验。

调用层负责读文件并校验，核心函数只接收已校验的对象（AGENTS.md 规则 3）。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MODE_ALIASES: dict[str, str] = {
    "sgbm": "MODE_SGBM",
    "sgbm_3way": "MODE_SGBM_3WAY",
    "hh": "MODE_HH",
    "hh4": "MODE_HH4",
}


class ConfigError(ValueError):
    """配置文件缺字段或取值非法。"""


@dataclass
class SgbmCfg:
    """一组 SGBM 参数。"""

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

    @property
    def p1(self) -> int:
        return self.p1_scale * self.block_size ** 2

    @property
    def p2(self) -> int:
        return self.p2_scale * self.block_size ** 2

    def validate(self) -> None:
        if self.num_disparities % 16 != 0:
            raise ConfigError(f"num_disparities 必须是 16 的倍数，当前 {self.num_disparities}")
        if self.num_disparities < 16:
            raise ConfigError(f"num_disparities 至少为 16，当前 {self.num_disparities}")
        if self.block_size < 3 or self.block_size % 2 == 0:
            raise ConfigError(f"block_size 必须是 >=3 的奇数，当前 {self.block_size}")
        if self.mode not in MODE_ALIASES:
            raise ConfigError(f"mode 只能是 {sorted(MODE_ALIASES)}，当前 {self.mode!r}")
        if not 0 <= self.uniqueness_ratio <= 100:
            raise ConfigError(f"uniqueness_ratio 必须在 0~100，当前 {self.uniqueness_ratio}")


@dataclass
class GeometryCfg:
    """校正好之后的相机几何。未知时留空，此时只输出视差，不输出深度。"""

    fx_px: float | None = None
    baseline_m: float | None = None

    @property
    def usable(self) -> bool:
        return (self.fx_px is not None and self.baseline_m is not None
                and self.fx_px > 0 and self.baseline_m > 0)

    def validate(self) -> None:
        if self.fx_px is not None and self.fx_px <= 0:
            raise ConfigError(f"fx_px 必须为正，当前 {self.fx_px}")
        if self.baseline_m is not None and self.baseline_m <= 0:
            raise ConfigError(f"baseline_m 必须为正，当前 {self.baseline_m}")
        if (self.fx_px is None) != (self.baseline_m is None):
            raise ConfigError("fx_px 与 baseline_m 必须同时给出或同时留空")


@dataclass
class RunCfg:
    """一次离线运行的全部参数。"""

    sgbm: SgbmCfg = field(default_factory=SgbmCfg)
    geometry: GeometryCfg = field(default_factory=GeometryCfg)
    threads: int = 1
    roi_margin_px: int = 0

    def validate(self) -> None:
        self.sgbm.validate()
        self.geometry.validate()
        if self.threads < 1:
            raise ConfigError(f"threads 必须 >=1，当前 {self.threads}")
        if self.roi_margin_px < 0:
            raise ConfigError(f"roi_margin_px 不能为负，当前 {self.roi_margin_px}")


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"{where} 缺少必需字段 {key!r}")
    return mapping[key]


def load_config(path: str | Path) -> RunCfg:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"配置文件不存在：{p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as ex:
        raise ConfigError(f"配置文件不是合法 JSON：{p}（{ex}）") from ex
    if not isinstance(raw, dict):
        raise ConfigError(f"配置文件顶层必须是对象：{p}")

    sgbm_raw = _require(raw, "sgbm", "配置")
    if not isinstance(sgbm_raw, dict):
        raise ConfigError("sgbm 必须是对象")
    known = set(SgbmCfg.__dataclass_fields__)
    unknown = set(sgbm_raw) - known
    if unknown:
        raise ConfigError(f"sgbm 含未知字段：{sorted(unknown)}；可用字段：{sorted(known)}")

    geo_raw = raw.get("geometry", {})
    if not isinstance(geo_raw, dict):
        raise ConfigError("geometry 必须是对象")
    geo_known = set(GeometryCfg.__dataclass_fields__)
    geo_unknown = set(geo_raw) - geo_known
    if geo_unknown:
        raise ConfigError(f"geometry 含未知字段：{sorted(geo_unknown)}；可用字段：{sorted(geo_known)}")

    cfg = RunCfg(
        sgbm=SgbmCfg(**sgbm_raw),
        geometry=GeometryCfg(**geo_raw),
        threads=int(raw.get("threads", 1)),
        roi_margin_px=int(raw.get("roi_margin_px", 0)),
    )
    cfg.validate()
    return cfg


def dump_config(cfg: RunCfg) -> dict[str, Any]:
    return asdict(cfg)
