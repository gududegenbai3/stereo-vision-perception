# USB 双目拍照、标定与预训练对照

本入口独立于默认 Middlebury CLI，适用于已确认左半为左相机的单 UVC 拼图。本机验证每目 1280×720；真实距离精度未验收。设备固定内参/外参与本机标定文件绑定，不代填 `hardware.json` 中尚未独立实测的参数。

## 环境与本机配置

先按[复现指南](../algorithm/pretrained_comparison.md)配置 Python、Pillow/Tk 和两份官方权重。Windows 实拍使用支持 DirectShow 的 FFmpeg。标定和 SGBM 使用 CPU。

将 `config/` 中以下模板复制为同目录的 `.local.json` 后编辑；本机配置自动忽略：

| 模板 | 本机文件 | 需要填写 |
|---|---|---|
| `capture_ui.example.json` | `capture_ui.local.json` | 设备名、FFmpeg 和本环境 pythonw 路径 |
| `calibration_capture.example.json` | `calibration_capture.local.json` | 同上；实测棋盘格长和测量来源 |
| `display_calibration.example.json` | `display_calibration.local.json` | 完成的标定会话目录、新版本输出目录 |
| `camera_comparison.example.json` | `camera_comparison.local.json` | 场景会话、新输出目录、确认标定仍适用 |

JSON 相对路径均按配置所在目录解释；拍照启动器中的裸可执行文件名从当前 PATH 查找，也支持绝对路径或相对配置目录的路径。先激活 `SVP_env`；`pythonw.exe` 必须来自同一个环境。

## 拍照与屏幕标定

```powershell
./src/start_capture.ps1
./src/start_capture.ps1 -Config config/calibration_capture.local.json
```

如本机策略禁止执行 PowerShell 脚本文件，可启动两个普通终端。第一个运行 `python -m src.capture_ui --config config/calibration_capture.local.json`（只拍场景时改用 `capture_ui.local.json`）。从配置指定的 `ready_file` 中读此会话的端口，然后在第二个终端运行下列命令。将端口和采集参数替换为本机值。采集完成后关闭窗口，结束 FFmpeg。

```powershell
ffmpeg -f dshow -video_size 2560x720 -framerate 30 -vcodec mjpeg -i "video=USB Camera" -an -fps_mode passthrough -pix_fmt bgr24 -f rawvideo tcp://127.0.0.1:12345
```

窗口按钮或空格拍照，“完成拍摄”结束。保存原始拼图、左右无损 PNG、哈希、配对 CSV 与 `DONE.json`；预览缩放不影响保存像素。主机接收时间不代表传感器曝光同步。

棋盘使用 9×6 内角点。先显示并量连续若干格，格长以米填写，测量来源放 `chessboard.measurement`。若预览时格长未知，先结束该次预览，保持屏幕分辨率、缩放和方格像素尺寸不变，填写测量后重新开始正式采集，避免擅自回写旧会话。**模板的空格长不能直接用于求解，也不能照抄别人的屏幕尺寸。**

移动整套相机改变目标角度、位置和距离，不拧镜头或改变两镜头相对位置。两侧均完整检测才保存；先选“标定图”采拟合数据，再选“独立验证图”。两类身份在求解前确定，不按误差剔图或重选。显示屏平面性、摩尔纹和手持稳定性仍可能影响结果。

```powershell
python -m src.display_calibration --config config/display_calibration.local.json
python -m src.display_calibration_report --config config/display_calibration.local.json
```

先分别单目标定，再固定内参求双目外参；保存米制 T、原尺寸映射和 Q。报告进程只回读参数，独立验证角点不参与拟合。两条命令分别运行，避免本机 OpenCV 标定与可选 Torch 的 OpenMP 运行库混用。已有标定输出拒绝覆盖。

报告生成 `config/current_calibration.local.json`，保存本机几何文件与哈希。警戒仅是诊断阈值，不能当作用户规定的精度标准。改变调焦、双目相对姿态或拍摄模式后须重查；整套刚性设备换拍摄位置不使标定自动失效。

## 三方法同输入对照

模型来源、SHA-256 与下载入口见[三模型复现指南](../algorithm/pretrained_comparison.md)。实拍对照和实时程序共用同一组官方权重。

```powershell
python -m src.camera_comparison --phase prepare
python -m src.camera_comparison --phase infer --method sgbm
python -m src.camera_comparison --phase infer --method banet
python -m src.camera_comparison --phase infer --method igev
python -m src.camera_report
```

准备时检查完成会话、左右身份、原图哈希与拼图拆分、标定身份及分辨率，冻结全部样例、展示编号、配置/源码/官方权重哈希。左右使用同一套固定标定映射，不逐对拟合；三方法读取同一份干净校正输入。原生或半尺寸由配置明确选择；192 px 搜索范围、IGEV 32 次迭代固定。两种分辨率须使用不同输出目录；近景在原生尺寸可能超出 192 px，不能忽略这个混杂因素。

报告校验全部输入和预测，生成全样例 `gallery.html` 及汇总。固定色标、共同支持和搜索区域；灰色为比较区域外或无输出。平滑、稠密及视觉轮廓都不等于正确率；没有真值时不输出 MAE、边界误差或模型精度排名。半尺寸视差不能直接套原生 Q 做米制重建。

## 现场实时预览

实时自动启动、帧率对照与同帧深度图见[现场实时试验版](live_demo.md)。

## 本次已验证与限制

2026-11-22：20 对拟合、6 对/324 角点独立验证；垂直差均值 3.879→0.640 px，P95 1.292 px。用户确认未再次调焦或改变双目相对位置后复用全部 22 对宿舍图，完成官方预训练三方法对照。1/2/14/15 号稀疏 SIFT 对应点仍有较大垂直残差，全部保留；这不直接证明相机移动。稀疏对应点不是真值。

精简代码已重新求解标定和重跑三模型；具体验证与提交范围见[交付记录](../exe_plans/completed/camera_demo_delivery.md)。原图、NPZ、环境、权重、完整预测和本机路径仅保留本地；仓库只收通用代码、配置模板、测试及文字记录。尚未独立实测光心距、锁定曝光/白平衡，也未验收绝对测距精度。
