# Stereo Vision Perception

双目立体成像项目，使用 Python 3.11、OpenCV 和 NumPy，实现图像采集、双目标定、极线校正、SGBM 视差、深度与彩色点云重建，以及独立距离评估。后续研究方向为置信度驱动的主动深度感知。

## 当前状态

当前版本为 **V0**，处于文档与接口准备阶段。已有算法规格、独立硬件配置、协作规则、执行计划及 Conda 依赖声明；`SVP_env` 环境已建立，已通过依赖导入和 Adirondack 读图检查，`src/` 已建立，Python 模块与可运行 demo 待实现，真实设备和算法精度尚无验证结论。

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
│   └── hardware.json           机器可读硬件参数与已知状态
└── src/                        已有实验 Notebook，Python 模块待实现
```

`doc/` 保存说明与记录，`config/` 保存配置，`src/` 保存实现。硬件参数以 [hardware.json](config/hardware.json) 为维护入口；含义、来源、未知项和接入流程见[硬件说明](doc/hardware/hardware.md)。

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

依赖记录在 [environment.yml](environment.yml)，环境名为 `SVP_env`，统一使用 conda-forge。文件限定 Python 3.11 和其他包的版本范围，不是完整锁文件。本地已验证 Python 3.11.11、OpenCV 4.10.0 和清单中的其他 Python 包可导入；JupyterLab 界面与内核启动待验证。新机器首次创建环境时，在仓库根目录执行：

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

后续按依赖清单补装或更新现有环境使用 `conda env update -n SVP_env -f environment.yml`，由使用者自行执行。Notebook 用于实验与展示，可复用函数仍放入 `src/`，参数从 `config/` 读取。Notebook 中优先使用 Matplotlib 显示图像，OpenCV 的 BGR 彩色图显示前转换成 RGB。目前仍无算法 demo 运行命令。

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

- 双目为软连接且基线可调；采集期间需要固定，改变基线或相对姿态后重新标定。
- 单设备左右拼图或双设备模式、光学基线、左右顺序及手动控制能力仍需实测。
- 规格书存在畸变参数冲突，配置保留原始值，待确认后更新。
- 设备读取时间不代表真实曝光同步；V0 优先使用静态场景。
- 理论焦距和模组外形尺寸不能替代真实标定参数，测距精度需由独立参考评价。
