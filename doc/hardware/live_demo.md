# 现场实时双目试验版

Windows 下激活 `SVP_env`，从仓库根目录运行 `.\start_live_demo.cmd`。启动器直接调用原生 FFmpeg 和 Python 窗口。摄像头断开会等待并重连。关闭窗口或按 Esc 退出。

上排显示左右目，下排并排显示 SGBM、BANet 和 IGEV。每次更新的五个画面来自同一帧。

| 操作 | 界面行为 |
| --- | --- |
| 暂停 / 继续 / 空格 | 暂停保留当前画面，继续后使用最新双目帧 |
| 深度 / 视差 | 五面板同步切换。灰色表示无有效输出 |
| 保存画面 | 保存合图、原始双目拼图和三路数值数组 |

原生 FFmpeg 经本机 TCP 交付完整 BGR24 帧。结果和日志位于 `artifacts/live_demo/`，首帧记录为 `camera-connected.json`。相机模式会自动重试。`.\start_live_replay.cmd` 从配置指定的录像进入同样的窗口；没有录像和相机时，使用[公开静态回放](../algorithm/pretrained_comparison.md#五面板快速验收)。

## 一次配置

先按[复现指南](../algorithm/pretrained_comparison.md)配置环境并验证模型。将 [live_demo.example.json](../../config/live_demo.example.json) 复制为同目录的 `live_demo.local.json`。引用已有的本地拍照配置、v2 标定选择文件和预训练配置，字段及相对路径约定见 [相机使用说明](camera_demo.md)。保持镜头焦点、摄像头间相对姿态和采集模式与 v2 标定相符。设置 `replay_file` 后可双击启动回放。

自有录像须使用与当前标定一致的原生拼接模式。在关闭实时窗口后，可从已激活环境的 Windows 终端运行下例，再将 AVI 路径填写到 `replay_file`。设备名或模式需要从本机采集配置取值。

```powershell
ffmpeg -f dshow -video_size 2560x720 -framerate 30 -vcodec mjpeg -i "video=USB Camera" -an -t 10 -c:v copy artifacts/stereo_replay.avi
```

BANet 和 IGEV 使用官方预训练权重，由独立进程在首次计算时加载。可用 `igev_iterations` 选择 8 或 32 次迭代。每轮从最新的完整双目帧计算三种结果，算完后一起更新，跳过中间帧。同屏刷新率受三种算法的总耗时影响，启动时先显示双目预览。

三方法统一采用每目 640×360 的校正图和 192 px 搜索上限。Q 映射到半尺寸坐标后再生成米制深度。界面中的帧率指应用更新的深度图，延迟从主机收齐帧起算。米制估计只适用于匹配有效的位置，独立测距精度待验收。

## 自动回放测速

从仓库根目录使用同一环境运行：

```powershell
python -s -m src.live_demo --run-dir artifacts/live_bench_8 --replay PATH_TO_AVI --benchmark-seconds 10 --sample-frame PATH_TO_RAW_PNG
python -s -m src.live_demo --run-dir artifacts/live_bench_32 --replay PATH_TO_AVI --benchmark-seconds 10 --sample-frame PATH_TO_RAW_PNG --igev-iterations 32
```

回放按拍照配置中的采集帧率定时，每个方法预热后依次测速。结果包含同帧深度与视差图、推理中位时间和接收到伪彩数组的延迟，Tk 控件显示开销由桌面回放单独观察。`--sample-frame` 可省略，默认展示回放首帧。实时和回放均可保留全输入的有效性差异；预测覆盖率不能作为精度。

2026-11-28 本机视频回放，CPU SGBM 与 RTX 5060 Laptop 推理结果如下。SGBM、BANet 和 IGEV8 来自同轮回放，IGEV32 来自同设置的另一次回放。BANet 本轮首次结果约需 9 秒，其中包括 CUDA 进程启动和加载权重。

| 方法 | 深度帧/秒 | 主机收到帧到伪彩数组 P95 |
| --- | ---: | ---: |
| SGBM | 12.3 | 113 ms |
| BANet | 14.6 | 110 ms |
| IGEV8 | 2.34 | 477 ms |
| IGEV32 | 0.93 | 1106 ms |

普通桌面通过外部 FFmpeg 实拍取得 2560×720 双目帧。新版桌面回放验证五个同帧面板、暂停、恢复与保存。使用 IGEV 8 次迭代时，回放连续 6 次同屏更新的速率约 2.5 次/秒，预热后收齐帧至提交 Tk 的中位耗时约 442 ms。

实时处理缓存固定校正的有效区域掩码，FFmpeg 按帧序号给原始视频流分配递增时间戳。用相同录像帧交替计时，完整 BGR 帧至 SGBM 深度伪彩的中位耗时从 91.99 ms 降至 81.50 ms。该计时覆盖校正、匹配、重投影和伪彩，实拍端到端表现仍以现场复测为准。
