# Stereo Vision Perception

双目立体成像项目，使用 Python 3.11、OpenCV 和 NumPy，实现图像采集、双目标定、极线校正、SGBM 视差、深度与彩色点云重建，以及独立距离评估。后续研究方向为置信度驱动的主动深度感知。

## 当前状态

当前版本为 **V0**。已在本地 `SVP_env` 验证 Middlebury 单场景离线流程，包括读图、SGBM、米制重投影、视差评估与产物导出。CLI 提供一键运行，Notebook 用于逐步检查中间结果。真实设备采集、标定、校正及独立测距仍待联调，具体状态见 [V0 执行计划](doc/exe_plans/active/V0_stereo_pipeline.md)。

最新进展、验收证据、阻碍与下一步以 [V0 执行计划](doc/exe_plans/active/V0_stereo_pipeline.md) 为准。计划完成并通过验收后归入 `completed/`，管理方式见[执行计划索引](doc/exe_plans/README.md)。

## 项目结构

```text
.
├── AGENTS.md                   Agent 职责、导航与严格规则
├── ARCHITECTURE.md            Python 模块职责、命名与依赖约定
├── shortcuts.md                自然语言对应的 Git / GitHub 操作
├── .gitignore                  本地环境、缓存与运行产物的忽略规则
├── environment.yml             Python、算法与 JupyterLab 的 Conda 依赖
├── MiddEval3-Q/                 Middlebury 四分之一分辨率双目图和标定（本地）
├── MiddEval3-GT0/               Middlebury 左视差真值和遮挡掩码（本地）
├── doc/
│   ├── algorithm/algorithm.md  算法流程、接口与验收要求
│   ├── hardware/               硬件说明与规格图片
│   ├── literature/             论文资料，按明确指令读取
│   ├── chat/                   组内谈话记录，按明确指令读取
│   └── exe_plans/
│       ├── README.md           计划索引与管理约定
│       ├── active/             未完成计划及当前进展
│       └── completed/          已验收归档的计划
├── config/
│   ├── hardware.json           机器可读硬件参数与已知状态
│   └── offline.json            Middlebury 场景、数据路径与 SGBM 参数
├── notebook/environment.ipynb  分步运行和结果展示
└── src/                        数据模型、算法模块、流程编排及 CLI
```

`doc/` 保存说明与记录，`config/` 保存配置，`src/` 保存实现。硬件参数以 [hardware.json](config/hardware.json) 为唯一机器可读来源（schema v2.1）；`document` 记录规格书页码与图片 SHA256，分组 `source` 与 `raw` 保留来源和原文，`uncertainties` 记录 U1–U8 的 `pending` 状态、`target_fields` 与 `resolution` 预留字段，`pending_summary` 汇总未决项，`ai_usage` 给出读取顺序与硬约束。含义、未知项和接入流程见[硬件说明](doc/hardware/hardware.md)。

## V0 开发路线

硬件与 OpenCV 算法通过明确的数据接口分离，可由组员并行推进：

- **硬件路线 H1–H3**：确认单设备拼图或双设备模式、左右顺序与实际输出，固定基线并记录成像控制设置，再交付标定、独立验证及距离测试图像。
- **算法路线 A0–A4**：建立环境与类型明确的函数接口，先处理离线图像，逐步实现标定、校正、视差、深度、点云和评估。设备尚未就绪时，可用有来源的样例或明确标记的合成数据验证程序。
- **联调与验收 I1–I2**：两条路线按统一的图像、配置与标定格式汇合，用真实设备数据贯通流程，再完成独立距离评价。

详细流程遵循[算法规格](doc/algorithm/algorithm.md)。交付物包括原始图像、标定与校正参数、数值视差/深度、有效掩码、彩色 PLY、预览和评估记录。合成验证、流程跑通与真实精度达标分别记录；未设置验收阈值时只报告实测指标。

## 技术栈与运行准备

- Python：固定使用 **3.11**。
- OpenCV：采集、标定、校正、立体匹配与三维重投影。
- NumPy：数组、标定参数和数值结果处理。
- 标准库：配置解析、路径、CSV/JSON 记录与 PLY 输出。
- 开发环境：JupyterLab，使用 ipykernel 执行 Python，Matplotlib 显示图像、深度预览与指标。

依赖记录在 [environment.yml](environment.yml)，环境名为 `SVP_env`，统一使用 conda-forge。文件限定 Python 3.11 和其他包的版本范围，不是完整锁文件。本地已验证 Python 3.11.11、OpenCV 4.10.0 和清单中的其他 Python 包可导入；离线 Notebook 已用 `SVP_env` 内核验证。新机器首次创建环境时，在仓库根目录执行：

```sh
conda env create -f environment.yml
```

已有 `SVP_env` 时跳过创建。激活环境、按需注册内核并启动：

```sh
conda activate SVP_env
python -m ipykernel install --user --name SVP_env --display-name "Python 3.11 (SVP_env)"
jupyter lab
```

在 JupyterLab 中选择 `Python 3.11 (SVP_env)` 内核。安装后可在终端验证解释器、核心依赖和硬件配置：

```sh
python --version
python -c "import cv2, numpy, matplotlib, jupyterlab, ipykernel; print(cv2.__version__, numpy.__version__)"
python -m json.tool config/hardware.json
```

后续按依赖清单补装或更新现有环境使用 `conda env update -n SVP_env -f environment.yml`，由使用者自行执行。Notebook 用于实验与展示，可复用函数仍放入 `src/`，参数从 `config/` 读取。Notebook 中优先使用 Matplotlib 显示图像，OpenCV 的 BGR 彩色图显示前转换成 RGB。

配置读取与校验由调用层负责，核心算法接收明确类型的输入并返回结构化结果。后续 function calling 的工具声明、参数 schema 与实现适配层分别维护，硬件 JSON 作为配置数据使用。

## 离线验证数据

数据来自 [Middlebury Stereo v3 官方下载页](https://vision.middlebury.edu/stereo/submit3/)。首次获取时下载 `MiddEval3-data-Q.zip` 和 `MiddEval3-GT0-Q.zip`，分别解压到仓库根目录下的以下目录。路径均相对于仓库根目录。

| 本地目录 | 文件与用途 |
|---|---|
| `MiddEval3-Q/trainingQ/<场景>/` | `im0.png`、`im1.png` 和 `calib.txt`，用于计算视差和重建 |
| `MiddEval3-GT0/trainingQ/<场景>/` | `disp0GT.pfm` 和 `mask0nocc.png`，用于比较左图视差及筛选非遮挡区域 |
| `MiddEval3-Q/testQ/<场景>/` | 左右图与标定，可运行算法观察结果；官方未公开该划分的视差真值 |

首轮可用两个 `trainingQ` 目录中的 `Adirondack`。按同名场景配对，沿用图像自带的校正几何；深度换算使用配套 `calib.txt` 的主点偏移 `doffs` 与毫米基线 `baseline`。本地数据需单独获取。

Python 模块的命名、职责和导入方向见 [ARCHITECTURE.md](ARCHITECTURE.md)。

在 JupyterLab 打开 [environment.ipynb](notebook/environment.ipynb)，选择 `Python 3.11 (SVP_env)` 并从头运行。Notebook 会从仓库根目录或 `notebook/` 定位项目，直接调用 `src` 的 SGBM 和重投影实现，显示预测视差、匹配有效率和米制深度图。有真值时，还会显示真值对照、非遮挡真值覆盖率，以及在预测与真值有效交集上的视差 MAE 和 RMSE。无真值的测试场景仍显示全图与搜索 ROI 的有效匹配率。也可以在仓库根目录使用 `SVP_env` 调用单场景离线流程。

```python
from pathlib import Path

from src.config import OfflineConfig, load_offline_config
from src.pipeline import OfflineRunResult, run_offline_pipeline

settings: OfflineConfig = load_offline_config(Path("config/offline.json"))
run: OfflineRunResult = run_offline_pipeline(settings)
print(run.metrics)
print(run.output_dir)
```

Notebook 的最后一个单元格按 `output_root` 导出结果。默认每次运行写入新的 `outputs/<划分>/<场景>/run-*/`，并打印完整路径。将 `output_root` 设为 `null` 可跳过导出。

| 文件 | 内容 |
|---|---|
| `disparity.npy`、`depth.npy`、`valid_mask.npy` | 原分辨率像素视差、米制深度、布尔重建掩码 |
| `valid_mask.png`、`disparity_preview.png`、`depth_preview.png` | 0/255 掩码及按各图有效值范围着色的预览 |
| `point_cloud.ply` | 有效点的米制 XYZ 和左校正图 RGB |
| `metrics.json`、`samples.csv` | 本次场景指标与单样本记录，JSON 另附搜索 ROI、参数和版本 |

修改 `src/` 中的模块后，在 Notebook 中重启内核并运行全部单元格，使配置、数据和算法对象都使用同一版本的代码。

场景、数据与产物路径、SGBM 起始参数维护在 [offline.json](config/offline.json)。配置中的相对数据路径以配置文件所在目录为准。

`scene.geometry` 提供米制重投影矩阵和图像尺寸。`scene.ground_truth` 提供像素视差、全部真值有效掩码与非遮挡掩码，读取 `testQ` 时为 `None`。路径由调用方传入。文件缺失报 `FileNotFoundError`，场景名、格式或尺寸有误报 `ValueError`。

## 离线命令行入口

在仓库根目录激活 `SVP_env`，按需选择命令。

| 命令 | 用途 |
|---|---|
| `python -m src.cli` | 读取 `config/offline.json`，运行离线重建和评估，按配置导出 |
| `python -m src.cli --help` | 查看可用参数 |
| `python -m src.cli --config config/offline.json` | 显式指定配置文件，可替换为其他 JSON 路径 |
| `python -m src.cli --no-export` | 计算重建和评估，跳过写文件 |
| `python -m src.cli --preview --no-export` | 在桌面窗口查看视差和米制深度，跳过写文件 |

可用 `--split trainingQ|testQ` 和 `--scene 场景名` 选择图像对。例如：

```sh
python -m src.cli --split testQ --scene Australia --no-export
```

数据位于其他位置时，使用 `--image-root 路径` 和 `--ground-truth-root 路径` 覆盖配置中的根目录。训练场景需使用与图像匹配的真值目录。CLI 路径相对于当前工作目录，JSON 内的相对路径仍以配置文件所在目录为准。当前读取器支持 Middlebury Q 版目录结构。

CLI 打印场景、指标与导出目录的 JSON，跳过导出时 `output_dir` 为 `null`。预览需要交互式 Matplotlib 后端。已验证图像与色条生成及无交互后端的报错，桌面弹窗待实测。退出码与模块职责见 [CLI 指令](ARCHITECTURE.md#cli-指令)。

## 协作方式

- 开始工作先阅读 [AGENTS.md](AGENTS.md) 及对应活动计划，结束后按代码、运行结果与实测证据更新任务状态。
- Python 命名采用 `snake_case`，变量、函数参数和返回值明确类型，数组注明元素类型、形状与单位。
- `main` 保持为稳定基线，不直接提交实验性改动。
- 使用独立开发分支，默认前缀为 `codex/`，改动通过 PR 合并；说明改动内容、验证结果与已知限制。
- Git 上传修改按 `V<版本号>_<GitHub用户名>_<修改次数>` 命名，例如 `V0_gududegenbai3_1`；同一版本下，每位用户的修改次数从 1 开始递增。
- 拉取、提交、推送、创建 PR 和合并的自然语言入口及 CLI 流程见 [shortcuts.md](shortcuts.md)。
- 虚拟环境、缓存、本地配置、采集/标定数据和运行产物按 [.gitignore](.gitignore) 管理；共享代码、配置和文档保留在仓库，活动计划记录数据标识、复现方法与证据位置。

**上下文读取约定：** 未收到用户明确指令时，Agent 不读取或总结 `doc/literature/` 的论文及 `doc/chat/` 的组内谈话记录；全文搜索、批量读取与上下文收集默认排除这两类内容。日常开发优先查阅算法说明、硬件配置和执行计划。

## 已知限制

- 双目为软连接且基线可调；采集期间需要固定，改变基线或相对姿态后重新标定（U3）。
- 单设备左右拼图或双设备模式、光学基线、左右顺序、手动控制能力与实际帧率仍需实测（U3、U4、U6、U8）。
- 规格书存在畸变参数冲突（U1）和最大分辨率与模式表不一致（U2）；配置保留原始值，未采用任一值。
- 未决项在 JSON 中统一为 `status: pending`，`resolution.value/evidence/recorded_at/note` 为预留字段；在实测或供应商确认后回填，未填写时不得当作已确认数据。
- 设备读取时间不代表真实曝光同步；V0 优先使用静态场景。
- 理论焦距和模组外形尺寸不能替代真实标定参数，测距精度需由独立参考评价。
