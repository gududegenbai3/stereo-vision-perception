# AGENTS.md — 本仓库的 AI 协作规则

> 目的：让任何 AI agent（Codex / Claude / DSH 等）在这个仓库里 **不需要人工解释** 就能正确干活。
> 规则来源：`README.md`（协作方式、目录约定）、`plans/phase-1/plan.md`（要交什么证据）。
> 冲突时以 `README.md` 与 `plans/` 为准；本文件只补充"怎么执行"。

## 0. 一句话

**所有改动必须是可验证的**：跑得起来、有自检、有产物、产物对得上 plan 里的曲线。做不到这四点的代码不提交。

## 1. 分支与提交

- `main` 只放已验证的稳定版本，**任何 agent 都不直接提交到 main**。
- 分支命名：`codex/<主题>` 或 `feat/<主题>`（例：`codex/min-refine-harness`）。
- 一次提交只做一件事；提交信息用中文、动词开头，说明"做了什么 + 验证命令"，例：
  `feat: 加置信度驱动精算最小实现（selftest 全过，rows.csv 见 artifacts 说明）`。
- 任何改动都走 PR。**PR 描述必须包含四段**：变更内容 / 验证命令与输出 / 已知未知项 / 风险（模板见 `docs/pr/`）。

## 2. 目录与文件

| 路径 | 放什么 | 禁止 |
|---|---|---|
| `src/` | 可运行代码 | 硬编码本机绝对路径、把参数散落在代码里 |
| `tests/` | 单元测试 | 依赖网络与大数据 |
| `configs/` | 参数档（JSON） | 写入本机路径、密钥 |
| `docs/` | 可读说明、决策记录、论文笔记 | **PDF、图片集合、运行产物** |
| `docs/research/notes/` | 每篇论文一个 md，命名 `YYYY_FirstAuthor_ShortTitle.md` | 直接抄摘要 |
| `plans/` | 阶段计划与进度 | 写无法验证的目标 |
| `research/` | 论文 PDF 与元数据（只读归档） | 修改/移动/重命名 PDF（会撑爆仓库历史） |
| `data/`、`artifacts/` | 采集与运行产物（**不入库**） | 提交 |

## 3. AI 工作循环（每次任务照此执行）

1. **读**：`README.md` → `plans/<当前阶段>/plan.md` → 本次要改的文件。确认要交付的证据是哪条曲线。
2. **分支**：`git switch -c codex/<主题>`。
3. **实现**：参数进 `configs/`，逻辑进 `src/`，边界情况写进 `tests/`。
4. **过门（gate）**：一条命令，全绿才能继续
   ```bash
   python scripts/check.py          # 依次跑 selftest → pytest → 小尺寸实验 + 卫生检查
   python scripts/check.py --quick  # 只跑 selftest + pytest（约 20 秒，改文案时用）
   ```
   CI 调用的就是这一条命令，因此"本地绿"与"CI 绿"是同一个判据。
5. **留证**：把 `artifacts/.../rows.csv` 的关键结论摘成 3–5 行写进 PR（**不要提交 artifacts 目录本身**）。
6. **写 PR**：四段齐备；不确定的地方写进"已知未知项"，不要藏着。
7. **收尾**：`git status` 确认没有 `data/`、`artifacts/`、`*.npy`、`__pycache__` 被加入。

## 4. 证据与计划的对应（不可省略）

`plans/phase-1/plan.md` 第五节要求四条证据，代码必须能产出：

| 计划要求的证据 | 由什么产出 | 状态 |
|---|---|---|
| 曲线一：精度-开销权衡（A/B/C） | `run_experiment.py` → `cost_curve.png`、`rows.csv` | 已有（开销以耗时计，能耗待硬件） |
| 曲线二：风险-覆盖率 | `risk_coverage.csv`、`risk_coverage.png` | 已有 |
| 曲线三：每可信像素开销 | `rows.csv` 的 `ms_per_kvalid` | 已有（毫秒口径；接 INA219 后换成 mJ/千像素） |
| 速度：算法时延 vs 端到端 | `rows.csv` 的 `selfcheck_ms`/`refine_ms`/`total_ms` | 只覆盖算法部分，端到端待采集链路接入 |

**若计划要求而代码给不出，写进 PR 的"已知未知项"，并在 `plans/` 里登记缺口。**

### 证据归档命名（强制）

`run_experiment.py` 不传 `--out` 时，自动落到：

```
artifacts/<YYYYMMDD-HHMM>_<topic>_<sha7>/
```

每个证据目录必须同时含 `meta.json`（时间戳、git commit 与是否 dirty、数据来源、图像尺寸、penalty、python/numpy/opencv 版本）。**报告里出现的任何一个数字，都要能从这里反查到确切的代码状态。** 产物本身不入库（`artifacts/` 已 gitignore），但结论必须带着目录名写进 PR 或 `docs/`。

## 5. 红线

1. 不提交：论文 PDF、原始采集、运行产物、模型权重、任何凭据。
2. 不对共享分支 `rebase` / `push --force`。
3. 不修改 `research/` 下的 PDF（只读归档）。
4. 不用自动曝光的图做实验结果（参数必须锁定，见 `docs/hardware.md`）。
5. 不在没有真值的情况下报告"精度提升"。
6. 不把"拒绝输出"当成免费胜利：缺测像素必须计入误差（当前按 3 px 计）。

## 6. 已知未知项怎么记

在 `plans/<阶段>/plan.md` 末尾追加一节 `## 已知未知项`（或新建 `plans/<阶段>/unknowns.md`），每条写：
`什么是未知 / 为什么现在无法确定 / 下一步怎么验证 / 谁负责`。
不要写在 PR 里就结束——PR 会沉底，计划文件不会。

## 7. 论文笔记的最小要求

新增/阅读一篇论文，就在 `docs/research/notes/` 留一条笔记，模板见 `docs/research/notes/README.md`。
笔记里必须有一节"与本项目的关系"，指出它对应本项目哪一步（粗算/自查/分配/精算/评测），否则等于没读。
