# 离线视差最小实现（V0 · 算法路线 A0 / A3 / A4 的一部分）

本文件说明**已实现并跑通**的离线链路：一对已校正图像 → SGBM 视差 → 有效掩码 →（可选）深度 → 指标。
算法规格与验收要求以 [algorithm.md](algorithm.md) 为准；本文件只记录实现现状、接口约定与已知未实现项。

**范围**：本文件描述 V0 阶段 A0/A3/A4 范围内的离线视差实现，不含置信度估计、在线自适应、学习式匹配、多帧融合，也不含标定与极线校正（输入假定已校正）。

## 1. 运行方式

仓库根目录执行（Python 3.11）：

```sh
pip install -r requirements.txt

python src/check.py                      # 验收门：自检 → 单测 → 小尺寸实验 → 卫生检查
python run_experiment.py                 # 合成图冒烟（无设备时验证程序行为）
python run_experiment.py --left L.png --right R.png --gt disp.pfm
python run_experiment.py --left L.png --right R.png \
    --config config/stereo_default.json  # 真实（已校正）图像
```

真值支持 `.pfm`（Middlebury）、`.npy`、`.png`；无真值时不输出精度指标，只输出视差/掩码/深度与耗时。

## 2. 模块与接口

实现位于 `src/stereo_min/`。数据约定：

- 图像：`np.uint8`，形状 `(H, W)` 灰度或 `(H, W, 3)` BGR。
- 视差：`np.float32`，形状 `(H, W)`，单位**像素**；无效像素为 `0`。OpenCV 输出为 1/16 定点，读取时已除以 16。
- 深度：`np.float32`，形状 `(H, W)`，单位**米**；无效像素为 `NaN`。
- 掩码：`np.bool_`，形状 `(H, W)`。

| 函数 | 输入 | 输出 | 失败方式 |
|---|---|---|---|
| `config.load_config(path)` | JSON 路径 | `RunCfg` | `ConfigError`（缺字段、未知字段、非法取值、非 JSON、文件不存在） |
| `matcher.compute_disparity(left, right, sgbm_cfg)` | 图像对 + `SgbmCfg` | `disp (H,W) float32` px、耗时 ms | `ValueError`（尺寸不符、宽度不足以容纳搜索范围） |
| `depth.disparity_to_depth(disp, geometry)` | 视差 + `GeometryCfg` | `depth (H,W) float32` m、`valid (H,W) bool` | `ValueError`（几何未配置） |
| `depth.valid_mask(disp, roi_margin_px)` | 视差 | `mask (H,W) bool` | — |
| `metrics.disparity_metrics(disp, gt, valid)` | 视差、真值、掩码 | EPE / P95 / bad1 / bad2 / bad3 / coverage / eval_pixels | `ValueError`（形状不符） |
| `metrics.depth_metrics(depth, gt_depth)` | 深度、参考深度 | MAE / RMSE / 相对 MAE / 样本数 | `ValueError`（形状不符） |
| `evaluate.evaluate_pair(left, right, cfg, gt=None, gt_depth=None)` | 图像对、`RunCfg` | `PairResult`（视差、掩码、深度、指标） | 透传上述错误 |

核心函数**不读文件、不读全局配置**：配置由调用层解析校验后以 `RunCfg` 传入（AGENTS.md 规则 3）。

## 3. 统计口径

- 指标只在**预测有效且真值有效**的像素上统计；**无效预测不以 0 混入均值**（对应计划 A4 的要求）。
- `coverage` = 参与统计的像素 / 有真值的像素；`valid_ratio` = 预测有效像素 / 全图。
- 真值有效定义为“有限且为正”；`.pfm` 中无真值处（负值或 `NaN`）被排除。
- 视差单位像素；深度误差需独立参考深度，属于 I2 的范围。

## 4. 配置

| 文件 | 用途 |
|---|---|
| `config/stereo_synthetic.json` | 合成图冒烟（视差范围小） |
| `config/stereo_default.json` | 真实（已校正）图像起步参数 |

字段语义（数值以 JSON 为准，此处不重复维护）：

- `sgbm`：`num_disparities`、`block_size`、`mode`（`sgbm`/`sgbm_3way`/`hh`/`hh4`）、`uniqueness_ratio`、`disp12_max_diff`、`speckle_window`、`speckle_range`、`pre_filter_cap`、`p1_scale`、`p2_scale`；`P1/P2` 由 `scale × block_size²` 推出。
- `geometry`：`fx_px`（像素焦距）与 `baseline_m`（基线，米）。**必须同时给出或同时留空**；留空时只输出视差。
- `threads`：OpenCV 线程数（固定可复现）；`roi_margin_px`：裁掉边界不可靠条带。

`num_disparities` 由量程反推：`d_max = fx_px × baseline_m / Z_min`，再向上取 16 的倍数。**不要照抄默认值。**

## 5. 验收门与产物

`python src/check.py` 依次执行：① `src/selftest.py`（16 项结构性断言）→ ② `src/run_tests.py`（有 pytest 用 pytest，否则内置运行器，共 19 项）→ ③ 小尺寸实验 → ④ 暂存区不得含产物 → ⑤ 源码与配置不得含本机绝对路径。任一失败即非零退出。

每次运行产物（`--out`，默认 `artifacts/offline_<时间戳>/`，已被 `.gitignore` 排除）：

| 文件 | 内容 |
|---|---|
| `metrics.json` | 汇总指标 + 本次配置 + 数据来源 + 口径说明 |
| `samples.csv` | 逐样本结果（本版本一次一对） |
| `disp.png` | 视差预览（0 为黑） |
| `depth.png` | 深度预览（近=亮），仅当 `geometry` 已配置 |
| `config.json` / `meta.json` | 本次参数；git 提交、是否 dirty、图像尺寸、依赖版本 |

## 6. 合成数据实测（480×240，三段平面 6.0/9.5/4.0 px，`config/stereo_synthetic.json`）

验证环境：Python 3.11.9（`.venv` 虚拟环境）、NumPy 2.4.6、OpenCV 5.0.0，单线程。

| 指标 | 值 |
|---|---|
| EPE | 0.150 px |
| P95 | 0.375 px |
| bad1 / bad2 / bad3 | 1.98% / 1.92% / 1.69% |
| coverage | 0.855 |
| valid_ratio | 0.851 |
| SGBM 耗时 | 4.42 ms（单帧） |

该数据来自程序生成的合成图（低频纹理 + 三段平面），**只用于验证程序行为，不能作为本设备的测距精度结论**。

## 7. 未实现项

| 项 | 现状 |
|---|---|
| 相机采集适配（单设备拼图 / 双设备） | 未实现，属硬件路线 H1–H3 与 A0 的采集适配层 |
| 双目标定与极线校正（A1/A2） | 未实现；本实现假设输入已校正 |
| 点云与彩色 PLY | 未实现（A3 剩余部分） |
| 有效 ROI 的几何搜索（`getValidDisparityROI`） | 未实现；目前只用 `roi_margin_px` 裁边 |
| 距离精度评价、重复性、分阶段耗时 P95（A4 剩余部分） | 未实现；依赖独立参考距离数据（I2） |
| 能耗与端到端时延 | 未测量 |
