# AGENTS.md — AI 协作约束

> 稳定约束，不随阶段变化。阶段任务见 `plans/`，代码与口径说明见 `docs/code.md`。
> 与本文件冲突时，以 `README.md` 与 `plans/` 为准。

## 1. 目录归属

| 路径 | 放什么 |
|---|---|
| `src/` | 可运行代码 |
| `tests/` | 单元测试（不依赖网络与大数据） |
| `configs/` | 参数档（JSON，不含本机路径与密钥） |
| `scripts/` | 开发脚本（验收门等） |
| `docs/` | 可读说明、决策记录、论文笔记 |
| `docs/research/notes/` | 论文笔记：一篇一个 md，命名 `YYYY_FirstAuthor_ShortTitle.md` |
| `plans/` | 阶段计划与进度 |
| `research/` | 论文 PDF 与元数据（**只读归档**） |
| `data/`、`artifacts/` | 采集与运行产物（**不入库**） |

## 2. 验证

提交前必须过门，CI 跑的是同一条命令：

```bash
python scripts/check.py        # selftest → 单元测试 → 小尺寸实验 → 卫生检查
```

失败即不提交。CI 配置见 `.github/workflows/ci.yml`。

## 3. 红线

1. 不直接改 `main`；不对共享分支 `rebase` 或 `push --force`。
2. 不新增 PDF 到 git；不修改、移动、重命名 `research/` 下已有 PDF——git 保存每次改动的完整副本，删掉也不减小历史。
3. `docs/` 不放 PDF、截图集合、运行产物。
4. 不提交产物与数据：`artifacts/`、`data/`、`*.npy`、模型权重、任何凭据。
5. 不用自动曝光的图做实验结论（曝光/增益/白平衡需锁定，见 `docs/hardware.md`）。
6. 没有真值不报"精度提升"；缺测像素必须计入误差（口径见 `run_experiment.py` 的 `--penalty`）。

## 4. PR 格式

四段：变更内容 / 验证命令与输出 / 已知未知项 / 风险。模板与实例见 `docs/pr/`。

## 5. 不确定的事

写进 `plans/<阶段>/`（追加"已知未知项"或新建 `unknowns.md`）。只写在 PR 里会随 PR 沉底。
