# 硬件与接口：HBVCAM-4M2403HD V22

> 本文档是本项目摄像头硬件的**人类可读说明 + 机器可读参数摘要**。
> 参数来源：厂商规格书第 3 页（Key Specifications）和第 4 页（Module Mechanical Dimension），版本 A，日期 2024-10-30。
> 用户确认：采购的是 **80° 无畸变版本**；两个摄像头之间是**软连接**，基线可调。
> 最后核对：2026-10-17。

## 1. 机器可读摘要

下面的 YAML 是给代码和 AI 直接读取的稳定字段。未知值保持 `null`，不要用猜测值填充。

```yaml
rig_id: hbvcam-4m2403hd-v22
vendor: "深圳市汇博视捷科技有限公司"
model: "HBVCAM-4M2403HD V22"
user_description: "80° 无畸变版本；两摄像头软连接，基线可调"

document:
  title: "Key Specifications / Module Mechanical Dimension"
  revision: "A"
  date: "2024-10-30"
  pages_used: [3, 4]

interface:
  type: "USB 2.0"
  class: "UVC"
  power_source: "USB bus power"
  supply_voltage: "DC 5V ±5%"
  max_power_mw: 790
  os_support: "Windows XP/7/8/Vista, Android 4.0+, macOS, Linux with UVC"

sensor:
  model: "OV4689"
  optical_format: "1/3 inch"
  package: "CSP, bare die"
  pixel_size_um: 2.0
  shutter: "electronic rolling shutter"
  max_output_resolution_px: [3840, 1080]
  sensitivity_mv_per_lux_s: 1900
  snr_db: 38.3
  dynamic_range_db: 64.6

lens:
  construction: "2G2P"
  efl_mm: 3.0
  f_number: 2.4
  fov_diagonal_deg: 80
  focus: "fixed"
  object_distance_cm: "30 to infinity"
  center_resolution_lw_ph: 600
  tv_distortion:
    page3_raw: "<97.2%"
    page4_raw: "<3%"
    status: "conflicting_in_datasheet"
    likely_value: "<3% (pending vendor confirmation)"
  ir_filter_nm: "650 ± 10"
  relative_illumination_percent: 70

output:
  formats: ["MJPG", "YUY2"]
  default_ae_agc_awb: "Auto"
  microphone: false

mechanical:
  module_size_mm: [143.6, 22.0, 16.5]
  module_size_tolerance_mm: 0.5
  lens_barrel_diameter_mm: 14.0
  pcb_thickness_mm: 1.60
  fpc_thickness_mm: 0.15
  operating_temperature_c: [-30, 70]
  stable_image_temperature_c: [0, 50]

stereo:
  baseline_mm: null
  baseline_adjustable: true
  baseline_range_mm: null
  mount_type: "flexible_soft_connection"
  baseline_locked: false
  left_right_layout: "unknown"
  expected_layout: "unknown_dual_or_side_by_side"
  notes: "两摄像头为软连接、基线可调；可能是两个独立 UVC 设备，也可能是单设备左右拼图。3840x1080 看起来像左右拼图，但接入后必须用 probe 确认。"

calibration:
  required: true
  status: "not_calibrated"
  nominal_focal_px: 1500
  nominal_focal_note: "EFL 3.0 mm / pixel 2 µm = 1500 px；这是理论值，实际以标定结果为准。"
```

## 2. 关键事实

| 项目 | 参数 |
|---|---|
| 型号 | HBVCAM-4M2403HD V22 |
| 接口 | USB 2.0，UVC 免驱 |
| 传感器 | OV4689，1/3″，CSP |
| 像素尺寸 | 2 µm × 2 µm |
| 最大输出分辨率 | 3840 × 1080 |
| 输出格式 | MJPG / YUY2 |
| 镜头 | EFL 3.0 mm，F/2.4，2G2P |
| 视场角 | 80°（对角，FOV(D)） |
| 对焦 | 固定焦距 |
| 物距 | 30 cm–无穷远 |
| 快门 | 电子卷帘快门 |
| 自动控制 | AEC / AGC / 白平衡默认 Auto |
| 麦克风 | 无 |
| 供电 | USB 总线供电，DC 5V ±5% |
| 最大功耗 | 790 mW |
| 模组尺寸 | 143.6 × 22 × 16.5 mm ±0.5 mm |
| 双目连接方式 | 软连接，基线可调；调整后必须锁死 |
| 镜头筒直径 | 约 14.0 mm |
| 工作温度 | -30°C–70°C |
| 稳定成像温度 | 0°C–50°C |

## 3. 支持的分辨率与帧率

以下是规格书标注的“Maximum Image Transfer Rate”。实际帧率会受到 USB 带宽、曝光时间、主机和驱动影响。

| 格式 | 分辨率 | 标称最大帧率 |
|---|---:|---:|
| MJPG | 1280 × 480 | 30 fps |
| MJPG | 2160 × 1080 | 30 fps |
| MJPG | 2560 × 720 | 30 fps |
| MJPG | 3040 × 1520 | 30 fps |
| MJPG | 3840 × 1080 | 30 fps |
| MJPG | 3840 × 1520 | 11 fps |
| YUY2 | 1280 × 480 | 10 fps |
| YUY2 | 2160 × 1080 | 3 fps |
| YUY2 | 2560 × 720 | 3 fps |
| YUY2 | 3040 × 1520 | 1 fps |
| YUY2 | 3840 × 1080 | 1 fps |
| YUY2 | 3840 × 1520 | 1 fps |

对双目第一阶段的建议：

- 优先使用 **MJPG**；YUY2 在高分辨率下帧率过低。
- 先用 **2560 × 720（每目 1280 × 720）** 或 **1280 × 480（每目 640 × 480）** 做 SGBM 基线，确认稳定后再提高分辨率。
- 如果设备实际以左右拼图输出，分辨率应配置为“单目宽度 × 2”。

## 4. 双目相关的关键未知项

这些参数不能从规格书直接推定，必须接入后实测：

| 未知项 | 为什么重要 | 如何确认 |
|---|---|---|
| 光学基线 | 可调，直接决定深度范围和误差；每次改变都会使旧标定失效 | 设定后用卷尺/卡尺量两个镜头光心距离，锁死并记录 |
| 左右目排布 | 决定 `mode: dual` 还是 `side_by_side` | 用 `probe` 看设备数量；看画面是否左右拼图 |
| 单目分辨率 | 决定 SGBM 的搜索范围和速度 | 逐档设置并读取实际分辨率 |
| 左右目顺序 | 接反会导致视差符号和深度异常 | 用有明显前后关系的场景确认 |
| 手动曝光/增益/白平衡 | 不锁定会导致两帧亮度漂移，结果不可复现 | 用厂商软件或 UVC 控制工具检查 |
| 实际可用帧率 | 规格书是上限，USB 带宽可能更低 | 连续读 300 帧统计 |
| 第 3 页 TV Distortion 的 `<97.2%` | 与第 4 页 `<3%` 冲突 | 向供应商确认，或实测畸变 |

特别提醒：**143.6 mm 是模组整体尺寸，不是已经确认的光学基线。** 不能直接把模组长度写进深度公式。

## 5. 基线配置记录

基线可调时，每次调整都应留下记录。复制下面这一行，不要覆盖旧记录：

| 日期 | 基线 mm | 连接/锁定方式 | 是否锁死 | 标定文件 | 备注 |
|---|---:|---|---|---|---|
| 2026-10-17 | 待测 | 软连接 | 否 | 尚未标定 | 初次接入，范围和锁紧方式待确认 |

规则：

- 同一行基线记录对应一套标定结果。
- 改变基线后必须重新标定，不能复用旧 `calibration.npz`。
- 建议 `rig_id` 带基线值，例如 `hbvcam-4m2403hd-v22-b80mm`。
- 运行 manifest 中记录本次实际使用的 `baseline_mm`。

## 6. 对算法和采集的影响

1. **“80° 无畸变”不等于不需要标定。**  
   低畸变镜头仍然需要标定内参、外参和极线校正。标定结果应保存到 `data/calibrations/<rig_id>/`，不要手写进本文档。

2. **默认 Auto 曝光是复现性的风险。**  
   规格书写的是 AEC/AGC/白平衡 Auto。正式采集前必须确认能否手动锁定；如果只能自动，需要在报告中记录这一限制。

3. **卷帘快门不适合快速运动场景。**  
   静态标定和静态深度演示没问题；运动目标可能产生时间错位。第一阶段先用静态场景。

4. **USB 2.0 高分辨率优先用 MJPG。**  
   3840 × 1080 的 MJPG 标称 30 fps，YUY2 只有 1 fps。不要用 YUY2 做实时双目。

5. **理论焦距只能做 sanity check。**  
   由 `EFL / pixel_size` 得到约 1500 px，但实际焦距会受镜头、分辨率和标定影响。深度计算以 `calibration.npz` 中的 `P1[0,0]` 为准。

6. **基线未知时不要宣称米级精度。**  
   在完成物理测量和双目标定前，只能输出“近似深度热力图”，不能把结果当作测距精度。

7. **基线可调意味着标定不是永久的。**  
   软连接允许调整基线，但运行和拍摄期间必须保持刚性、锁死。每次改变基线或左右目相对姿态后，旧标定立即失效，必须重新标定。建议在 `rig_id` 或标定目录名中带上基线值，例如 `hbvcam-4m2403hd-v22-b80mm`，并在 `manifest.json` 中记录本次运行使用的 `baseline_mm`。

8. **软连接不等于软同步。**  
   如果最终是两个独立 USB 摄像头，左右目是两路独立视频流，需要确认是否在同一帧循环读取，以及两路曝光是否能同时锁定。若两路自动曝光不同步，亮度漂移会直接破坏 SGBM 匹配。

## 7. 接入后的检查顺序

```text
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

- [第 3 页：Key Specifications](assets/hardware/specs-page3.jpg)
- [第 4 页：Module Mechanical Dimension](assets/hardware/specs-page4.jpg)

原始文档中的拼写错误（如 `ROSH`）按原样保留在来源说明中，不作为项目配置值使用；项目统一写 `RoHS`。
