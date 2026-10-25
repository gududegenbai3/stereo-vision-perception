# 相机标定记录与版本

> 机器可读摘要见 [`config/hardware.json`](../../config/hardware.json) 的 `calibration.dry_run` 与 `calibration.next_version`。
> 当前状态：`calibration.status = not_calibrated`；`v1_dryrun_display` 仅为 dry run，正式标定等待 `v2_final_bracket`。
> 本文件记录每个标定版本的数据、指标、产物和限制；不覆盖历史版本。

## 1. 版本表

| 版本 | 日期 | 状态 | 标定目标 | 使用数据 | 关键指标 | 产物 |
|---|---|---|---|---|---|---|
| `v1_dryrun_display` | 2026-10-25 | provisional / dry run | 显示屏 9×6 内角点、28 mm 方格 | 15 对采集，10 对参与标定 | 双目 RMS 0.314 px；基线 128.04 mm；垂直残差均值 0.668 px、P95 0.923 px | `data/calibrations/hbvcam-4m2403hd-v22/display_dryrun_v1/` |
| `v2_final_bracket` | 待定 | planned | 打印板优先；支架固定后的正式双目结构 | 目标 15–20 对 + 5–8 对独立验证 | 待测 | 待定 |

## 2. v1_dryrun_display

### 2.1 目标与数据

- 目标类型：电脑显示屏棋盘格。
- 棋盘格：9×6 内角点（10×7 个方格），实测方格边长 **0.028 m**。
- 图像：1280×480 左右拼图，每目 640×480；左半=左相机，右半=右相机。
- 采集数据：15 对，源文件 `IMG_0013`、`IMG_0016`、`IMG_0018`–`IMG_0030`。
- 角点检测：`findChessboardCornersSB` 对 15 对左右均检测到 54 个角点；标准 `findChessboardCorners` 在该显示屏目标上失败。
- 数据问题：源 JPG 均缺少文件尾（truncated JPEG），OpenCV `imread` 失败；已用 Pillow `LOAD_TRUNCATED_IMAGES=True` 解码，并转成 PNG 后再标定。

### 2.2 使用哪些数据

- 参与标定：`IMG_0021`–`IMG_0030`，共 10 对。
- 排除：`IMG_0013`、`IMG_0016`、`IMG_0018`、`IMG_0019`、`IMG_0020`。
- 排除原因：全 15 对混合标定时双目 RMS 1.068 px、垂直残差均值 1.84 px、P95 4.04 px；前 5 对与后 10 对不一致，说明采集期间软连接/相机相对姿态发生变化。后 10 对内部一致，留一验证稳定。

### 2.3 指标

| 指标 | 值 |
|---|---:|
| 单目 RMS（左/右） | 0.053 / 0.059 px |
| 双目 RMS | 0.314 px |
| 基线估计 | 128.04 mm |
| 校正垂直残差（标定集） | 均值 0.668 px，P95 0.923 px，最大 1.10 px |
| 留一验证垂直残差 | 均值 0.746 px，P95 1.006 px，最大 1.19 px |
| 左/右焦距 `fx` | 562.60 / 566.32 px |

基线 sanity check：`143.6 mm（模组长度） - 14.0 mm（镜头筒直径） ≈ 129.6 mm`，与标定估计 128.04 mm 接近；仍必须以卡尺实测光心距为准。

### 2.4 已知限制

- 显示屏目标有摩尔纹、反光和屏幕像素结构；姿态多样性有限。
- 前 5 对与后 10 对几何不一致，说明软连接在采集期间发生过变化。
- 只有 10 对参与标定，且没有**同一几何下的独立采集验证集**；留一验证只是内部一致性检查。
- `calibration.npz` 是 provisional，不得用于米制深度或精度结论。
- 临时采集/标定脚本未入库；本次只同步数据记录、配置和指标。

### 2.5 产物

本地忽略目录（不提交 Git）：

```text
data/calibrations/hbvcam-4m2403hd-v22/display_dryrun_v1/
├── calibration.npz
├── rectification.npz
├── metrics.json
├── pairs.csv
├── session.json
├── validation_leave_one_out.json
├── validation_excluded5.json
└── rectified_preview_pair_0001.png
```

原始采集数据保存在本地标定数据集目录，未提交 Git。源文件名和排除列表已记录在 `session.json` 与 `config/hardware.json`。

## 3. v2_final_bracket 计划

支架到位后执行，不覆盖 v1。

1. 刚性固定两个镜头头，保持平行、同向；用手轻碰不改变姿态。
2. 用卡尺测量两镜头光心距，重复 3 次；记录测量方式。
3. 优先使用打印在硬板上的棋盘格；若仍用显示屏，必须确认无摩尔纹/反光/横纹。
4. 在一次连续采集内拍 15–20 对；每对左右都用 `findChessboardCornersSB` 检查 54 个角点。
5. 另拍 5–8 对独立验证图，不参与标定；同一刚性结构、同一分辨率。
6. 重新标定并记录：单目/双目 RMS、基线 vs 卡尺实测、独立验证垂直残差、校正预览。
7. 通过后更新 `config/hardware.json` 的 `calibration.status`、`calibration.baseline_mm`、`calibration_file`，并在本文件新增 `v2_final_bracket` 行。

### 版本规则

- 基线、左右相对姿态、分辨率或裁剪方式变化后，必须新建版本，不能覆盖旧版本。
- 每个版本使用独立目录：`data/calibrations/<rig_id>/<version>/`。
- `.npz`、原始图像和大图留在本地忽略目录；Git 只提交配置、指标、方法、限制和证据索引。
- 提交命名遵守 `AGENTS.md` 的 `V<版本号>_<GitHub用户名>_<修改次数>` 规则。
