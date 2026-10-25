# 硬件与接口：HBVCAM-4M2403HD V22

> 本文档是摄像头硬件的**人类可读说明**；机器可读参数独立维护于 [`config/hardware.json`](../../config/hardware.json)。
> 参数来源：厂商规格书第 3 页（Key Specifications）和第 4 页（Module Mechanical Dimension），版本 A，日期 2024-10-30；原图见 `assets/`。
> 用户确认：采购的是 **80° 无畸变版本**；两个摄像头之间是**软连接**，基线可调。
> 最后核对：2026-10-21（参数核对）；2026-10-25（设备 probe、左右确认与 v1 dry-run 标定）。参数值以 JSON 为准；本文不重复维护完整规格表。
> 标定版本、指标与 v2 计划见[标定记录](calibration_log.md)。

## 1. 独立硬件配置与 JSON 结构

硬件配置已提取至 [`config/hardware.json`](../../config/hardware.json)，当前 `schema_version` 为 `2.2.0`，作为机器可读参数的唯一维护入口。JSON 保留规范化的数值与单位字段，同时用 `document.sources`、分组 `source`、`raw`、`uncertainties`、`pending_summary` 和 `calibration.dry_run` 记录来源、未决项与临时标定。

顶层分组：

- `rig_id`、`vendor`、`model`、`user_description`：设备身份。
- `document`：规格书版本、页码、图片路径与 SHA256。
- `conventions`：单位、数组顺序、`null` 规则与禁止替代标定的字段。
- `interface`、`sensor`、`lens`、`output`、`mechanical`：接口、传感器、镜头、输出模式与机械规格。
- `stereo`：基线、软连接、设备拓扑与左右目顺序。
- `calibration`：正式标定状态、理论焦距、缺失参数，以及 `v1_dryrun_display` 临时标定摘要。
- `compliance`：装配、PCB 印字、包装与认证。
- `uncertainties`：U1–U8 的冲突、未知与推断项；未决项统一 `status: pending`，已确认项 `status: resolved`，每项含 `target_fields` 与 `resolution`。
- `pending_summary`：剩余未决项数量、ID 列表、已解决项与回填规则。
- `ai_usage`：给 AI/调用层的读取顺序、硬约束与错误处理要求。

## 2. 配置数据与调用边界

- `source` 指向 `document.sources` 中的 `spec_page3` / `spec_page4`，或 `user` / `project`；`raw` 只用于追溯原文，不参与数值计算。
- 数值字段使用规范单位；分辨率按 `[width_px, height_px]` 排列，模组尺寸用 `length`、`width`、`height` 三个键，避免数组顺序歧义。
- 未知值保持 `null`，不以零或猜测值替代；每个未知/冲突项在 `uncertainties` 中有 `id`、`field`、`status`、`current_handling`、`needed`、`impact`、`target_fields` 与 `resolution`。未决项为 `pending`，已确认项为 `resolved`。
- `resolution` 是回填区：未决项初始为 `value/evidence/recorded_at/note = null`；确认后填写，并按 `target_fields` 同步数据字段，保留原始值与来源。
- `lens.tv_distortion.page3_raw` 与 `.page4_raw` 保留两页原始值，`U1.resolution.value` 为 `null`；不得把任一原始值直接当作畸变系数（U1）。
- `output.max_resolution_px` 采用规格书的 `3840×1080`，但 `output.supported_modes` 保留 `3840×1520`；两者不一致，见 U2。`max_fps` 是标称上限，不是实测帧率（U8）。
- `stereo.device_topology = single_uvc_side_by_side`、`stereo.left_right_order = left_half_is_left_camera` 已确认（U4 resolved）；`stereo.baseline_mm` 仍为 `null`（U3）。
- `calibration.nominal_focal_px = 1500` 是 `EFL 3.0 mm / 像素 2 µm` 的理论值；`calibration.dry_run` 中 640×480 的实测 `fx` 约 562.6/566.3 px，不能替代正式标定结果。
- `interface.class = UVC` 已确认（`class_status = confirmed`，U5 resolved）。
- 读取层应先校验 `schema_version`、必需字段、单位与数组顺序；遇到 `null` 或 `uncertainties` 中 `status: pending` 的项时，明确报错或要求实测，不得静默使用默认值。`resolution.value` 未填写时不得当作已确认数据。核心算法不读取 Markdown，也不隐式读取该 JSON。
- 若暴露为 function calling，工具名称、参数 schema 与适配层另行声明；`config/hardware.json` 是配置数据，不是工具定义。

## 3. 输出模式与采集建议

完整模式列表见 JSON 的 `output.supported_modes`。这些值来自规格书的“Maximum Image Transfer Rate”，实际帧率受 USB 带宽、曝光时间、主机和驱动影响。

- 第一阶段优先使用 MJPEG（采集 FourCC 常用 `MJPG`）；高分辨率 YUY2 的标称帧率较低。
- `output.max_resolution_px` 与模式表存在 `3840×1080` / `3840×1520` 的不一致（U2）；实际模式必须逐档 probe。
- 2026-10-25 实测：`640×480`、`1280×720` 是单画面；`1280×480`、`2560×720`、`3840×1080`、`3040×1520`、`3840×1520` 是左右拼图，分界在宽度一半。
- 标定初期建议 `1280×480`（每目 640×480）；确认稳定后再尝试 2560×720（每目 1280×720）。
- 仅在左右拼图模式下，总图像宽度才等于单目宽度的两倍；双独立设备模式以各设备实测尺寸为准。

## 4. 双目相关的关键未知项

这些参数不能从规格书直接推定，必须接入后实测。JSON 中用 `null`、`uncertainties` 与 `pending_summary` 记录：

| ID | 未知项 / JSON 字段 | 为什么重要 | 如何确认 |
|---|---|---|---|
| U1 | `lens.tv_distortion` | 第 3 页 `<-97.2%`，第 4 页 `<5%`，两页值冲突；影响畸变模型和质量预期 | 向供应商确认或实测；确认前 `U1.resolution.value = null` |
| U2 | `output.max_resolution_px` 与 `supported_modes` | 影响采集模式、单目分辨率与 SGBM 搜索范围 | probe 实际可用模式 |
| U3 | `stereo.baseline_mm` | 可调，直接决定深度范围和误差；每次改变都会使旧标定失效 | 设定后用卷尺/卡尺量两镜头光心距，锁死并记录 |
| U4 | `stereo.device_topology` / `left_right_order` | 决定采集适配方式与视差符号 | **已确认（2026-10-25）**：单 UVC 左右拼图；左半=左相机，右半=右相机 |
| U5 | `interface.class` | 影响采集后端选择 | **已确认（2026-10-25）**：USB Video Device / usbvideo，UVC |
| U6 | `output.default_controls` | 不锁定会导致两帧亮度漂移，结果不可复现 | 用厂商软件或 UVC 控制工具检查手动曝光/增益/白平衡 |
| U7 | `mechanical.module_size_tolerance` | ±0.5 mm 是否适用于长度/宽度未确认 | 安装设计前向供应商确认或实测 |
| U8 | `output.supported_modes.max_fps` | 标称值不是实测帧率 | 连续读取 300 帧统计成功率和帧率 |

特别提醒：**143.6 mm 是模组整体长度，不是已经确认的光学基线。** 不能直接把模组长度写进深度公式。第 4 页图中的 46.00 mm 标注位于两个相机头之间，看起来是柔性段/头间距，不是两镜头光心距，也不能直接当作 `baseline_mm`（U3）。

### v1 dry-run 标定

2026-10-25 使用显示屏 9×6、28 mm 棋盘格完成 `v1_dryrun_display`：15 对采集中 10 对参与标定，双目 RMS 0.314 px、基线估计 128.04 mm、校正垂直残差均值 0.668 px/P95 0.923 px。该结果仅为 provisional；前 5 对与后 10 对几何不一致，且没有同一几何下的独立验证集。详细数据、限制与 v2 计划见[标定记录](calibration_log.md)。

## 5. 基线配置记录

基线可调时，每次调整都应留下记录。复制下面这一行，不要覆盖旧记录：

| 日期 | 基线 mm | 连接/锁定方式 | 是否锁死 | 标定文件 | 备注 |
|---|---:|---|---|---|---|
| 2026-10-17 | 待测 | 软连接 | 否 | 尚未标定 | 初次接入，范围和锁紧方式待确认 |
| 2026-10-21 | 待测 | 软连接 | 否 | 尚未标定 | 复核规格书后仍未测量；不得用 143.6 mm 模组长度或图纸 46.00 mm 代填 |
| 2026-10-25 | 待测 | 软连接/临时放置 | 否 | `v1_dryrun_display`（provisional） | 标定估计 128.04 mm，仅作 sanity check；支架未到，光心距仍未用卡尺实测 |

规则：

- 同一行基线记录对应一套标定结果。
- 改变基线后必须重新标定，不能复用旧 `calibration.npz`。
- 建议 `rig_id` 带基线值，例如 `hbvcam-4m2403hd-v22-b80mm`。
- 运行 manifest 中记录本次实际使用的 `baseline_mm`。

## 6. 对算法和采集的影响

1. **“80° 无畸变”不等于不需要标定。**  
   低畸变镜头仍然需要标定内参、外参和极线校正。`lens.tv_distortion` 仍有两个冲突值（U1）；在确认前不能据此选择畸变模型或宣称低畸变指标。

2. **默认 Auto 曝光是复现性的风险。**  
   规格书写的是 AGC/AEC/白平衡 Auto。正式采集前必须确认能否手动锁定；如果只能自动，需要在报告中记录这一限制（U6）。

3. **卷帘快门不适合快速运动场景。**  
   静态标定和静态深度演示没问题；运动目标可能产生时间错位。第一阶段先用静态场景。

4. **USB 2.0 高分辨率优先用 MJPEG（FourCC `MJPG`）。**  
   3840 × 1080 的 MJPEG 标称 30 fps，YUY2 只有 1 fps。不要用 YUY2 做实时双目。

5. **理论焦距只能做 sanity check。**  
   由 `EFL / pixel_size` 得到约 1500 px，但实际焦距会受镜头、分辨率和标定影响。深度计算以 `calibration.npz` 中的 `P1[0,0]` 为准。

6. **基线未知时不要宣称米级精度。**  
   在完成物理测量和双目标定前，只能输出“近似深度热力图”，不能把结果当作测距精度。`stereo.baseline_mm` 保持 `null`（U3）。

7. **基线可调意味着标定不是永久的。**  
   软连接允许调整基线，但运行和拍摄期间必须保持刚性、锁死。每次改变基线或左右目相对姿态后，旧标定立即失效，必须重新标定。建议在 `rig_id` 或标定目录名中带上基线值，例如 `hbvcam-4m2403hd-v22-b80mm`，并在 `manifest.json` 中记录本次运行使用的 `baseline_mm`。

8. **软连接不等于软同步。**  
   已确认是单个 UVC 设备输出左右拼图，但同一帧内的两个传感器仍可能有曝光/读出时间差。若两路自动曝光不同步，亮度漂移会直接破坏 SGBM 匹配（U4 已确认拓扑/左右，U6 仍需确认手动控制）。

9. **不要把 JSON 的未知项静默补成默认值。**  
   读取层必须处理 `null` 与 `uncertainties`；用模组尺寸、理论焦距或图纸尺寸代填会产生看似可运行但尺度错误的结果。`ai_usage.hard_constraints` 是给 AI/调用层的最小约束集。

## 7. 接入后的检查顺序

```text
0. 读取 config/hardware.json，确认 schema_version、conventions 与 uncertainties
1. 确认设备数量：一个 UVC 设备还是两个 UVC 设备
2. 确认画面：单路宽画面，还是左右拼图
3. 确认分辨率：实际输出宽度、单目宽度、FPS、FOURCC
4. 确认左右目顺序
5. 锁定曝光、增益、白平衡
6. 设定基线，锁死软连接，测量光心距
7. 记录基线值和连接方式
8. 采集棋盘格图像对
9. 双目标定，记录 RMS 和基线
10. 跑 SGBM，输出深度热力图
```

## 8. 原始规格页

- [第 3 页：Key Specifications](assets/specs-page3.jpg)
- [第 4 页：Module Mechanical Dimension](assets/specs-page4.jpg)

文件路径、页码与 SHA256 记录在 JSON 的 `document.sources`。原始文档中的拼写错误（如 `ROSH`、`Whiter`）按原样保留在 `raw` 字段；项目规范字段使用 `RoHS` 等正确拼写。

## 9. JSON 字段索引

| 分组 | 内容 | 使用注意 |
|---|---|---|
| `rig_id` / `vendor` / `model` / `user_description` | 设备身份 | `rig_id` 暂不含基线；确定后可加 `-b<baseline_mm>mm` |
| `document` | 规格书版本、页码、图片路径、SHA256 | 追溯来源 |
| `conventions` | 单位、数组顺序、`null` 规则、`pending_rule` | 读取前先看 |
| `interface` | USB、供电、OS | `class` 已确认 UVC |
| `sensor` | OV4689、像素、灵敏度、SNR、动态范围 | `max_output_resolution_note` 指向 `output` |
| `lens` | EFL、F/No、FOV、焦距、畸变、IR | TV distortion 冲突（U1） |
| `output` | 格式、默认控制、分辨率/帧率模式 | `max_fps` 为标称值（U8） |
| `mechanical` | 模组尺寸、镜头筒、PCB/FPC、公差、温度 | 143.6 mm 不是基线 |
| `stereo` | 基线、软连接、拓扑、左右顺序 | 拓扑/左右已确认；基线仍为 `null`（U3） |
| `calibration` | 标定状态、理论焦距、缺失参数、`dry_run` | `dry_run` 为 provisional，不能替代正式标定 |
| `compliance` | 装配、PCB 印字、包装、认证 | 原始拼写见 `raw` |
| `uncertainties` | U1–U8 冲突/未知/推断 | U4/U5 已 resolved；其余 `status: pending`；`resolution` 为回填区 |
| `pending_summary` | 未决项数量、ID 列表、已解决项、回填规则 | 与 `uncertainties` 同步维护 |
| `ai_usage` | 读取顺序、硬约束、错误处理 | AI/调用层入口；未填写 `resolution.value` 不得当已确认数据 |
