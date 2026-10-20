# 置信度驱动的主动深度感知（最小可运行版）

方案1 的第三步流程——**粗算 → 自查 → 预算化精算 → 评测**——的可运行最小实现。
目标不是刷精度，而是把"**该在哪里多花算力**"变成可以测量、可以画图、可以复现的实验。

- 只用 OpenCV SGBM：**不需要训练、不需要 GPU、不需要外部数据集**
- 一次运行产出计划要求的三条证据（精度-开销、风险-覆盖率、每可信像素开销）+ 时延拆分
- 配置外置、结果落盘、零数据自检，符合 `AGENTS.md` 的验收门

## 代码在仓库里的位置

| 路径 | 内容 |
|---|---|
| `src/stereo_min/` | 算法实现（粗算 / 自查 / 分配 / 精算 / 评测 / 合成数据 / IO） |
| `tests/` | 单元测试（13 项） |
| `configs/` | 参数档：`synthetic.json`（合成图）、`default.json`（真实数据起步） |
| `scripts/check.py` | **验收门**：selftest → 单元测试 → 小尺寸实验 → 卫生检查 |
| `scripts/run_tests.py` | 测试入口：有 pytest 用 pytest，没有则用内置运行器 |
| `.github/workflows/ci.yml` | CI：push / PR 时跑同一道门 |
| `docs/pr/feat-min-refine.md` | 本模块 PR 的四段记录 |
| `docs/research/notes/` | 论文笔记（本篇给了 URS-Stereo 的样例） |
| `run_experiment.py` / `selftest.py` | 实验入口 / 零数据自检 |

**不要提交**：`artifacts/`、`data/`、`*.npy`（已在 `.gitignore`）。

## 快速开始

```bash
pip install -r requirements.txt
python scripts/check.py       # 验收门：selftest + pytest + 小尺寸实验 + 卫生检查
python run_experiment.py      # 单独跑一次完整实验（产物落到 artifacts/<时间>_<主题>_<sha>）
```

真实数据（队友标定+校正之后）：

```bash
python run_experiment.py --config configs/default.json \
    --left data/pair/left.png --right data/pair/right.png \
    --gt data/pair/disp.pfm --out artifacts/2026xxxx_min_refine
```

真值支持 `.pfm`（Middlebury）/ `.npy` / `.png`；`--penalty` 设定"预测缺失"计入的误差（默认 3 px，**必须与报告口径一致**）。

## 输出物与计划证据的对应

| 文件 | 内容 | 对应计划第五节 |
|---|---|---|
| `rows.csv` | 每变体一行：覆盖率、EPE、D1、bad2、P95、粗算/自查/精算/总耗时、ROI 数、每千有效像素耗时 | 曲线一、曲线三、速度 |
| `cost_curve.png` | 精度-开销图：A/B 点 + C/R 曲线（需 matplotlib） | **曲线一（主图）** |
| `risk_coverage.csv` / `.png` | 风险-覆盖率曲线与 AURC | **曲线二** |
| `rows.csv` 的 `ms_per_kvalid` | 每千个有效像素的毫秒数（接 INA219 后同口径换成 mJ） | **曲线三** |
| `rows.csv` 的 `coarse_ms` / `selfcheck_ms` / `refine_ms` | 时延拆分：粗算、自查额外成本、精算 | 速度口径 |
| `config.json` / `summary.json` | 本次运行参数与摘要，用于复现 | — |
| `meta.json` | 时间戳、git commit 与 dirty 状态、数据来源、图像尺寸、penalty、依赖版本 | 证据可追溯 |

**证据目录命名（强制）**：不传 `--out` 时自动落到 `artifacts/<YYYYMMDD-HHMM>_<topic>_<sha7>/`，保证报告里的每个数字都能反查到确切的代码状态。

## 算法（每一步对应报告的一节）

| 步骤 | 实现 | 关键约定 |
|---|---|---|
| **粗算** | `matcher.compute()`，一组便宜的 SGBM 参数 | 视差 $d=x_L-x_R\ge0$；OpenCV 输出 **1/16 px 定点**，读取时 `/16` |
| **自查** | `confidence.lrc_pair()`：左右一致性 | 左图像素 $x$ 查右参考图的 $x-d_L(x)$；超阈值判无效。**这是额外一遍匹配，计入 `selfcheck_ms`** |
| **分配** | `selection.select_top_k()`：按不确定度取前 $b$ 比例 | 预算 $b$ 由配置给定；`R` 变体同预算随机挑选，是"多算就该更准"的对照组 |
| **精算** | `refine.refine_roi()`：按窗口、用**局部搜索范围**重算 | 用该窗口粗算视差的 min/max 定 `minDisparity`/`numDisparities`（`search_margin` 外扩），满足 SGBM 的窗口宽度约束，比全范围便宜一个量级 |
| **评测** | `metrics.*` | 预测缺失按 `penalty=3 px` 计，避免"拒绝输出"白拿好处 |

两个刻意的设计决定：

1. **排序键与置信度分离**：挑精修目标用「不确定度 + 纹理次级键」，风险-覆盖率曲线用**原始置信度**——前者含"可救性"先验，后者评估自查信号本身。
2. **次级键优先高纹理**：无效像素的不确定度全部饱和，不打破平局排序就退化成按像素序号。实测 20% 预算下 EPE：优先高纹理 0.812 / 优先低纹理 0.876 / 不打破平局 0.892。

## 合成图实测（`python run_experiment.py`，240×480，三段平面 6.0/9.5/4.0 px）

| 变体 | 预算 | 覆盖率 | EPE (px) | 总耗时 (ms) | ROI | ms/千有效像素 |
|---|---|---|---|---|---|---|
| A 全粗算 | 0 | 0.717 | 0.976 | 6.0 | 0 | 0.07 |
| B 全精算 | 1.00 | 0.723 | 0.966 | 41.7 | 1 | 0.50 |
| C 自适应 | 0.02 | 0.722 | **0.962** | 21.6 | 20 | 0.26 |
| R 随机 | 0.02 | 0.720 | 0.968 | 26.4 | 32 | 0.32 |
| C 自适应 | 0.10 | 0.743 | **0.900** | 21.7 | 20 | 0.25 |
| R 随机 | 0.10 | 0.731 | 0.934 | 26.7 | 32 | 0.32 |
| C 自适应 | 0.20 | 0.773 | **0.812** | 21.7 | 20 | 0.24 |
| R 随机 | 0.20 | 0.746 | 0.891 | 26.7 | 32 | 0.31 |
| C 自适应 | 0.40 | 0.790 | **0.763** | 26.7 | 32 | 0.29 |
| R 随机 | 0.40 | 0.774 | 0.809 | 27.3 | 32 | 0.31 |
| C = R | 1.00 | 0.860 | 0.558 | 26.8 | 32 | 0.27 |

可直接写进报告的四条结论：

1. **自适应挑选有效**：10% 预算下 C 的 EPE 比 R 低 3.6%，20% 时低 8.9%。**C−R 的差才是"聪明地算"的净收益**，A–B 连线区分不出来。
2. **自适应还更省**：小预算只需 20 个窗口，随机要 32 个——高不确定度像素是**成簇**的。
3. **曲线三给出性价比**：A 是 0.07 ms/千有效像素，C@20% 是 0.24（3.4×）换来 17% 的 EPE 下降；B 是 0.50（7×）只换来 1%。**C 明显落在 A–B 连线的下方（更好一侧）**。
4. **100% 预算时 C 与 R 必然相同**，这条不变量由 `selftest.py` 守住。

> 合成图只用于验证管线。它的纹理是随机构造的，统计特性与真实场景不同，**不能用它判断算法优劣**——那必须在 Middlebury/ETH3D 与自采数据上做。

## 已知限制

- 自查只有左右一致性（计划 §4.2 要求三种横向对比，目前完成一种）；
- 精算是"换一组更贵的 SGBM 参数 + 局部搜索范围"，不是换模型，也不是升分辨率；
- 只有单帧，无时域先验；无能耗测量，只有耗时（曲线三目前是毫秒口径）；
- 端到端时延缺失（不含采集/传输/显示）；
- `refine_roi` 的窗口须比搜索范围宽（SGBM 硬约束），视差范围很大时会退化成接近整图重算。

## 下一步（按性价比）

1. **自查升级**：加代价曲率 / 峰数（`confidence.py` 再加一个信号，pipeline 内融合），完成计划要求的三种横向对比；
2. **排序键升级**：用"每焦耳误差下降"的估计量替代不确定度（离线拟合小回归器）；
3. **精算升级**：局部升分辨率重算；或 ROI 内换学习模型（IGEV/LAS2），整图跑一次作上界；
4. **时域**：上一帧低置信区膨胀作为本帧候选，动态场景自动退回全图排序；
5. **测电**：把 `ms_per_kvalid` 换成 `mJ_per_kvalid`，口径分"算法-only / 采集到输出 / 整机"。

## 代码结构

```
src/stereo_min/
├── config.py      参数对象与校验（全从 JSON 读，无硬编码）
├── matcher.py     SGBM 封装（粗/贵两档，支持局部搜索范围）
├── confidence.py  自查：左右一致性 → 不确定度
├── selection.py   分配：预算化挑选 + 随机对照
├── refine.py      精算：ROI 局部重算 / 整图重算
├── metrics.py     评测：视差域指标、风险-覆盖率、AURC
├── synthetic.py   合成图对（真值精确，仅用于自检）
├── stereo_io.py   读写图像 / PFM 真值（兼容中文路径）
├── plotting.py    可选绘图（缺 matplotlib 时只写 CSV）
└── pipeline.py    串起整条链路
```

## CI（`.github/workflows/ci.yml`）

- 触发：任意分支 push、PR、手动；同分支连推会自动取消旧任务。
- 只做一件事：**安装依赖 → `python scripts/check.py`**，与本地完全同一条门。
- 证据：`artifacts/` 以 workflow artifact 形式上传，保留 7 天（失败也传，便于看现场）。
- 仓库是 public，标准 runner 免费；CI 里用 `opencv-python-headless`（无显示环境，体积更小）。
- **要让 CI 真正拦住合并**：仓库 Settings → Branches → `main` 保护规则 → 勾选 *Require status checks to pass*，选中 `ci / check`。这步需要 admin 权限（组长操作）。

## 后续改动怎么走

```bash
git switch -c feat/<主题>        # 不在 main 上开发
# 改代码 / 加参数档 / 补测试
python scripts/check.py           # 门必须全绿（CI 跑的是同一条）
git add <明确列出文件>             # 不要 git add -A，避免误加产物
git commit -m "feat: <做了什么>（check.py 全绿）"
git push -u origin feat/<主题>
```

PR 描述取 `docs/pr/feat-min-refine.md` 的四段结构：变更内容 / 验证命令与输出 / 已知未知项 / 风险。
