# -*- coding: utf-8 -*-
"""stereo_min：离线双目视差最小实现（V0 · 算法路线 A0/A3/A4 的一部分）。

模块划分：
    config       配置承载与校验（全部来自 config/*.json）
    matcher      SGBM 视差（1/16 定点处理、无效置 0）
    depth        视差 → 深度（Z = fx·B/d）与有效掩码
    metrics      视差域指标（只在有效像素上统计）
    stereo_io    图像 / PFM / NPY 读写（兼容中文路径）
    synthetic    明确标记的合成图对（仅用于无设备时验证程序行为）
    evaluate     单对图像的一次完整评测

本阶段**不包含**置信度估计、在线自适应、学习式匹配、多帧融合。
"""
from .config import ConfigError, GeometryCfg, RunCfg, SgbmCfg, dump_config, load_config

__all__ = ["ConfigError", "RunCfg", "SgbmCfg", "GeometryCfg", "load_config", "dump_config"]
__version__ = "0.1.0"
