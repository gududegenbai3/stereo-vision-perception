# Middlebury 离线最简实现

## 范围与入口

对应 A0 离线入口、A3 已校正数据重建及 A4 的视差评价部分。算法仍为 OpenCV SGBM 基线，采集、棋盘格标定、原始图像校正及独立测距尚待完成。

```sh
conda activate SVP_env
python run_experiment.py
python run_experiment.py --scene Australia --split testQ
python -m unittest discover -s tests -v
```

Notebook 使用 `src/offline_middlebury.ipynb`，选择 Python 3.11 (SVP_env) 内核。CLI 与 Notebook 共用以下顺序：

```python
config = load_config(config_path)
scene = load_middlebury_scene(config.data_root, config.scene, config.split)
result = reconstruct_scene(scene, config.sgbm)
save_outputs(output_dir, scene, result, config)
```

类型与接口定义见 `src/stereo_vision/models.py`。阶段函数仅接收数据和已校验参数；数据读取、运行配置、算法、评价与文件输出各有独立职责。导入模块不执行实验。

## 参数与坐标

运行参数维护于 `config/middlebury.json`。路径相对于该 JSON 所在目录；CLI 可覆盖数据根目录、场景、划分和输出目录。所有 SGBM 参数都记录在结果中；搜索上界来自场景 `ndisp`，向上取 16 的倍数，不使用 GT 或 `vmin/vmax` 选择参数。

图像适配层检查左右图、两侧 3×3 内参、图像尺寸、`doffs` 一致性、基线与真值尺寸。只支持当前数据集的水平校正针孔模型，无法适配时明确报错。源基线从 mm 转为 m；Q 保留 `doffs = cx_right - cx_left`，对应 `Z = fx × baseline / (d + doffs)`。重投影坐标系为左校正相机：X 右、Y 下、Z 前。

PFM 按字节序和自底向上行序读取，浮点样本直接作为像素视差，保留 Inf 无真值状态；不使用 OpenCV 读入时的额外缩放。非遮挡掩码取 `mask0nocc == 255`，排除 128 遮挡和 0 无真值。

## 有效性与评价

SGBM 定点视差除以 16。有效点同时满足固定几何搜索域、有效匹配、正视差、有限 XYZ 和正深度。无效视差/深度/坐标统一 NaN，掩码为 bool，所有数组保留原图大小。

- 重建有效率：有效三维点数 / 固定几何搜索域像素数。
- 评价参考域：固定搜索域与非遮挡、有限正真值的交集。
- 评价覆盖率：参考域内有效预测数 / 参考域像素数。
- EPE、P95 和 bad2 只在参考域内有有效预测的位置统计；缺预测不以零混入误差，另报数量和覆盖率。bad2 为当前 Q 分辨率下误差大于 2 px 的比例。
- 没有 GT 时，评价状态明确记录为未评估；没有独立距离参考时不宣称距离精度。

这是本地样例诊断口径，不等同于官方全分辨率榜单评价。参考 [Middlebury SDK](https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3/README.txt) 与 [标定格式](https://vision.middlebury.edu/stereo/data/scenes2014/)。

## 输出与复现

默认目录为 `artifacts/middlebury/<split>/<scene>/`，自动排除 Git 跟踪。

| 输出 | 内容 |
|---|---|
| `disparity_px.npy` | float32 (H,W)，px |
| `depth_m.npy`、`xyz_m.npy` | float32 (H,W)/(H,W,3)，m |
| `valid_mask.npy`、`search_mask.npy` | bool (H,W)，重建有效点与固定搜索域 |
| `valid_mask.png`、`disparity_preview.png` | 掩码与伪彩预览；预览不参与重建 |
| `cloud.ply` | ASCII RGB 点云，点数等于重建掩码有效点数，坐标 m |
| `metrics.json` | 参数、Q、数据身份、单位、版本、设备、评价口径与阶段耗时 |
| `samples.csv` | 单样本结果、有效点数、误差与阶段耗时 |
| `failure.json` | CLI 处理失败时保留场景身份与错误原因 |

计时覆盖灰度转换与 SGBM、重投影及二者合计，排除读文件、评价、显示和写输出。当前仅记录单次耗时；均值、P95 与真实设备帧率尚未评价。参数中的线程数由入口设置，不由核心算法更改。

## 当前验证

2026-11-03：Python 3.13、OpenCV 5.0.0、NumPy 2.5.1，7 项回归检查通过。Adirondack 参考域 295457 像素，参与评价 289052 像素，EPE 1.147992 px、P95 2.648650 px、bad2 0.057671、覆盖率 0.978322。Australia 无 GT 流程运行成功。旧版将 `mask > 0` 视作非遮挡的结果已被此口径取代。

目标 Python 3.11 / OpenCV 4.x、Notebook 内核、真实相机及独立测距尚待验证；不据此宣布 V0 完成或精度达标。
