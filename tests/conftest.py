import sys
from pathlib import Path

# 让测试无需安装即可 import src/stereo_min
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
