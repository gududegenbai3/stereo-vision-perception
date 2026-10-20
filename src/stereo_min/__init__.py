# -*- coding: utf-8 -*-
"""stereo_min：置信度驱动的主动深度感知——最小可运行版本。

模块划分（与报告的三步流程一一对应）：
    matcher     粗算 / 精算的 SGBM 封装
    confidence  自查（最小版本用左右一致性 LRC）
    selection   分配（预算化挑选，含随机对照组）
    refine      精算算子（ROI 重算 / 整图重算）
    metrics     评测（视差域指标 + 风险-覆盖率 + AURC）
    pipeline    串起整条链路
"""
from .config import ExpCfg, RefineCfg, SgbmCfg, dump_config, load_config

__all__ = ["ExpCfg", "SgbmCfg", "RefineCfg", "load_config", "dump_config"]
__version__ = "0.1.0"
