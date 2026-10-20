# Stereo Vision Perception

双目视觉感知项目，目标是用 OpenCV 完成双目采集、标定、SGBM 视差和深度热力图，并在此基础上迭代“置信度驱动的主动深度感知”。

当前处于第一阶段：硬件规格、研究资料和项目计划已归档；**置信度驱动精算链路的最小可运行实现已加入**（见[代码说明](docs/code.md)），标定与采集仍需硬件到位后推进。

## 当前资料

- [硬件与接口](docs/hardware.md)：HBVCAM-4M2403HD V22 的规格、接口、采集建议和基线记录。
- [论文索引](docs/research/papers.md)：先读哪些论文、和本项目的关系。
- [第一阶段执行方案](plans/phase-1/plan.md)：硬件、标定、实验和日程。
- [代码说明](docs/code.md)：粗算 → 自查 → 预算化精算 → 评测的实现、口径与实测结果。
- [完整论文库](research/active-depth-perception/README.md)：46 篇论文的分类、获取状态和阅读顺序。
- `hardware/vendor-software/`：厂商摄像头工具和说明书。

## 代码与验收门

```bash
pip install -r requirements.txt
python scripts/check.py      # 一条命令：selftest → 单元测试 → 小尺寸实验 → 卫生检查
```

- CI：`.github/workflows/ci.yml`，push / PR 时跑同一道门；要拦住合并需在分支保护里设为 required check（需 admin）。
- 证据：`run_experiment.py` 默认落到 `artifacts/<时间>_<主题>_<sha>/`，内含 `meta.json`（git commit、数据来源、依赖版本），保证报告中的数字可回溯。
- 协作细则（分支、提交、红线、证据要求）：[AGENTS.md](AGENTS.md)。

## 第一阶段目标

1. 确认双目是“双 USB 设备”还是“单设备左右拼图”。
2. 设定并锁死可调基线，测量两个镜头的光心距离。
3. 锁定曝光、增益和白平衡。
4. 完成棋盘格双目标定和极线校正。
5. 用 SGBM 输出可检查的视差图与深度热力图。
6. 每次实验留下配置、图像和指标，避免只保留一张“看起来还行”的截图。

## 目录约定

| 路径 | 内容 |
|---|---|
| `docs/` | 可读的硬件事实、研究索引和决策说明 |
| `docs/assets/` | 文档引用的小型图片 |
| `docs/pr/` | 每个 PR 的四段记录（变更 / 验证 / 已知未知 / 风险） |
| `docs/research/notes/` | 论文笔记（一篇一个 md） |
| `plans/` | 执行计划、进度和决策记录 |
| `research/` | 论文全文、元数据和采集脚本 |
| `hardware/` | 厂商软件、说明书和硬件相关资料 |
| `src/` | 可运行代码（`src/stereo_min/`） |
| `tests/` | 单元测试 |
| `configs/` | 参数档（JSON，不含本机路径） |
| `scripts/` | 验收门等开发脚本 |
| `.github/` | CI 配置 |
| `data/`、`artifacts/` | 采集与运行产物，**不入库** |

## 协作方式

- `main` 保持为稳定基线，不直接提交实验性改动。
- 每个阶段使用独立分支，例如 `codex/harness-structure`。
- 改动通过 Pull Request 合并；PR 中写清验证命令、已知未知项和风险。
- 论文 PDF、原始采集和运行产物不放进 `docs/`。

## 已知限制

- 两个摄像头是软连接，基线可调；改变基线后必须重新标定。
- 规格书第 3 页与第 4 页的 TV Distortion 数值冲突，待供应商确认。
- 3840×1080 很可能是左右拼图，但尚未用设备确认。
- 光学基线、左右目顺序和手动曝光能力尚未实测。
