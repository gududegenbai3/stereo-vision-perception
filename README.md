# Stereo Vision Perception

双目立体成像项目，使用 Python 3.11、OpenCV 和 NumPy，实现图像采集、双目标定、极线校正、SGBM 视差、深度与彩色点云重建，以及独立距离评估。后续研究方向为置信度驱动的主动深度感知。

## 当前状态

当前版本为 **V0**，处于文档与接口准备阶段。已有算法规格、独立硬件配置、协作规则和执行计划；相机参数已从规格书第 3、4 页整理为 `schema_version: 2.2.0` 的 JSON，包含来源、原始值、冲突/未知项与 AI 使用约束。U1–U8 中 U4/U5 已确认，其余未决项统一 `status: pending`，每项在 `resolution` 中预留 `value/evidence/recorded_at/note`。2026-10-25 已完成显示屏目标上的 `v1_dryrun_display` 临时标定：10 对有效图、双目 RMS 0.314 px、基线估计 128.04 mm；该结果仅为 provisional。`src/`、依赖清单与可运行 demo 尚未建立，正式标定和算法精度尚无验证结论。

最新进展、验收证据、阻碍与下一步以 [V0 执行计划](doc/exe_plans/active/V0_stereo_pipeline.md) 为准。计划完成并通过验收后归入 `completed/`，管理方式见[执行计划索引](doc/exe_plans/README.md)。

## 项目结构

```text
.
├── AGENTS.md                   Agent 职责、导航与严格规则
├── shortcuts.md                自然语言对应的 Git / GitHub 操作
├── .gitignore                  本地环境、缓存与运行产物的忽略规则
├── doc/
│   ├── algorithm/algorithm.md  算法流程、接口与验收要求
│   ├── hardware/
│   │   ├── hardware.md         硬件说明与规格图片
│   │   └── calibration_log.md  标定版本、指标、限制与 v2 计划
│   ├── literature/             论文资料，按明确指令读取
│   ├── chat/                   组内谈话记录，按明确指令读取
│   └── exe_plans/
│       ├── README.md           计划索引与管理约定
│       ├── active/             未完成计划及当前进展
│       └── completed/          已验收归档的计划
├── config/
│   └── hardware.json           相机参数 JSON：规范单位、来源、原始值、冲突/未知项、pending 预留与 AI 使用约束
└── src/                        Python demo 与实现模块，待建立
```

`doc/` 保存说明与记录，`config/` 保存配置，`src/` 保存实现。硬件参数以 [hardware.json](config/hardware.json) 为唯一机器可读来源（schema v2.2）；`document` 记录规格书页码与图片 SHA256，分组 `source` 与 `raw` 保留来源和原文，`uncertainties` 记录 U1–U8 的状态与 `resolution` 预留字段，`pending_summary` 汇总未决项，`calibration.dry_run` 记录 v1 临时标定，`ai_usage` 给出读取顺序与硬约束。含义、未知项和接入流程见[硬件说明](doc/hardware/hardware.md)；标定版本与 v2 计划见[标定记录](doc/hardware/calibration_log.md)。

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

OpenCV 与 NumPy 的具体版本将在 A0 中固定。目前尚无完整安装流程或 demo 运行命令。已具备 Python 3.11 时，可在仓库根目录检查解释器和配置：

```sh
python3.11 --version
python3.11 -m json.tool config/hardware.json
```

配置读取与校验由调用层负责，核心算法接收明确类型的输入并返回结构化结果。后续 function calling 的工具声明、参数 schema 与实现适配层分别维护，硬件 JSON 作为配置数据使用。

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
- 单设备左右拼图与左右顺序已确认（U4 resolved）；光学基线、手动控制能力与实际帧率仍需实测（U3、U6、U8）。
- 规格书存在畸变参数冲突（U1）和最大分辨率与模式表不一致（U2）；配置保留原始值，未采用任一值。
- `v1_dryrun_display` 只是显示屏目标上的 provisional 标定，且没有同一几何下的独立验证集；不得用于米制深度或精度结论，正式标定等待支架和 v2。
- 未决项在 JSON 中统一为 `status: pending`，`resolution.value/evidence/recorded_at/note` 为预留字段；在实测或供应商确认后回填，未填写时不得当作已确认数据。
- 设备读取时间不代表真实曝光同步；V0 优先使用静态场景。
- 理论焦距和模组外形尺寸不能替代真实标定参数，测距精度需由独立参考评价。
