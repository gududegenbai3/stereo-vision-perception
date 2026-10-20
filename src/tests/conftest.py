# -*- coding: utf-8 -*-
"""让测试无需安装即可 import src/stereo_min。"""
import sys
from pathlib import Path

# 本文件位于 <repo>/src/tests/conftest.py，仓库 src 目录是 parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
