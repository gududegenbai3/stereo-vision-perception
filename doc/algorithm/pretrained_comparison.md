# SGBM、BANet 和 IGEV 复现指南

默认离线入口运行 OpenCV SGBM。下列对比与实时演示使用独立入口，共用官方 Scene Flow 权重。

## 环境与模型

在仓库根目录运行。Windows 实时采集使用支持 DirectShow 的 FFmpeg 8.1.1。公开对比需要能解开 7z 的 bsdtar，Windows 自带的 `tar.exe` 已验证。

```powershell
conda env create -f environment.yml
conda activate SVP_env
conda install -c conda-forge pillow tk pip
python -m pip install -r requirements-stereo-cuda.txt
python -m src.prepare_pretrained
python -m src.prepare_pretrained --verify-only
```

已有 `SVP_env` 可从激活环境开始。参考环境为 Windows、Python 3.11.18、OpenCV 4.13.0、RTX 5060 Laptop、Torch 2.7.1+cu128、torchvision 0.22.1+cu128 和 timm 0.6.13。两个网络需要支持 CUDA 12.8 的驱动和兼容 NVIDIA GPU。CPU 可运行 `python -m src.cli --no-export` 和单元测试。

| 模型 | 官方固定源码 | 官方 Scene Flow 权重 |
| --- | --- | --- |
| BANet-2D | [7cb83f7](https://github.com/gangweix/BANet/tree/7cb83f7cca99a249e1fcec8ff556e4ac73facd70/banet-2d) | [官方文件](https://drive.google.com/file/d/1z5QgTU1lMb74SpbHyBA26CEILYrru77O/view) |
| IGEV-Stereo | [4a3d937](https://github.com/gangweix/IGEV/tree/4a3d937aab788b3ff3ed49b7b5c81340ab46a5eb/IGEV-Stereo) | [官方文件](https://drive.google.com/file/d/16e9NR_RfzFdYT5mPaGwpjccplCi82C2e/view) |

[模型配置](../../config/pretrained_stereo.example.json)记录下载目录、文件名和项目实测 SHA-256。上游推理源码逐文件校验。如果 Drive 自动下载受限，可在浏览器将两份权重保存到配置所列路径，再运行准备命令。推理严格加载全部参数，使用 FP32 和权重中的 BatchNorm 统计。

## 五面板快速验收

用仓库自带的已校正 Adirondack 图片和场景 Q 做静态回放。五个面板、深度/视差、暂停、继续和保存与 USB 路径共用。

```powershell
python -m src.live_demo --config config/live_demo.example.json --middlebury-replay --output-root artifacts/live_demo
```

接入 USB 相机时，先按[相机采集与标定](../hardware/camera_demo.md)生成该套相机的标定映射，再按[实时说明](../hardware/live_demo.md)启动。USB 设备须输出左右拼接流，双目顺序、镜头焦点和分辨率须与标定一致。静态回放可检验软件链路，真实同步、动态延迟和距离精度需要实物验证。

## 重跑 181 对公开数据对比

准备完模型后，从仓库根目录顺序执行。

```powershell
python -m src.stereo_fair_data
python -m src.stereo_fair_evaluation --phase freeze
python -m src.stereo_fair_evaluation --phase select-sgbm
python -m src.stereo_fair_evaluation --phase infer
python -m src.stereo_fair_evaluation --phase summarize
python -m src.stereo_fair_report
```

- [实验配置](../../config/stereo_fair_evaluation.json)固定参数和来源。[样本清单](../../config/stereo_fair_samples.json)固定 15 对选参、181 对评价、场景组件和规范化文件哈希。Middlebury 2014 Q 开发集使用仓库已有文件。Middlebury 2006、2021 Mobile 与 ETH3D 使用官方档案。Booster 通过 HTTP Range 获取所需 ZIP64 成员。缓存素材与预测需预留数 GB。
- 下载档案置于 `artifacts/fair_evaluation/data/sources`。如站点访问受限，可按配置 URL 手动下载 Middlebury 和 ETH3D 档案，将 2021 移动数据保存为 `mobile-all.zip`。Booster 可从官方 ZIP 取出清单场景的 `train/balanced/<scene>` 文件，保留目录层级放入 `sources/booster/`。已备齐本地档案时可用 `--offline`。
- 输出位于 `artifacts/fair_evaluation/pretrained_v1/`。`report.md` 为图文汇总，`per_sample.csv` 为逐样本指标。预测、协议、源码/权重哈希和计时保留在本机输出目录。重复实验使用新的 `output_root`。
- 主指标为固定、预测无关区域内误差超过 2 px 或缺失的比例，按场景组件和数据来源等权。实验含开发集选参 SGBM、IGEV 8/32 迭代对照和边界分层。Mobile 2021 使用已知 GT 域，因为该来源没有遮挡标签。

[固定评估协议、历史结果与限制](../exe_plans/completed/stereo_fair_evaluation.md)
