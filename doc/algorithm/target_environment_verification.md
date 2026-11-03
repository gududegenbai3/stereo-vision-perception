# 目标环境验证记录

日期：2026-11-03。范围：V0 Middlebury 离线实现的环境、CLI 与 Notebook 内核。

## 环境来源

使用仓库原 `environment.yml`，通过 Micromamba 从 conda-forge 创建独立 SVP_env。原直接依赖范围和渠道未更改；此前 Python 3.13 / OpenCV 5 的运行不能代替本记录。环境、缓存与运行产物位于 Git 忽略的 `artifacts/`。

| 依赖 | 本次实际版本 | 项目规定 |
|---|---|---|
| Python | 3.11.18 | 3.11 |
| OpenCV | 4.13.0 | >=4.10,<5 |
| NumPy | 2.4.6 | >=1.26,<3 |
| Matplotlib | 3.11.2 | >=3.8,<4 |
| JupyterLab | 4.6.5 | >=4,<5 |
| ipykernel | 7.4.0 | >=6,<8 |

`environment-win-64.explicit.txt` 根据本次已安装的 Conda 包元数据生成，包含全部 271 个包的 conda-forge URL 与 MD5。该标准 `@EXPLICIT` 格式仅供 Windows x64 创建环境；尚未进行第二次干净重建测试。项目依赖范围仍以 `environment.yml` 为准。

```sh
# 新建 Windows 环境时可选择精确快照
conda create -n SVP_env --file environment-win-64.explicit.txt
# 在已经建立的目标环境执行
conda run -n SVP_env python -m unittest discover -s tests -v
conda run -n SVP_env python run_experiment.py --scene Adirondack
conda run -n SVP_env python run_experiment.py --split testQ --scene Australia
```

## 实际验证

- 8 项检查全部通过；新增 Python 3.11 / OpenCV >=4.10,<5 环境门禁，错误解释器不能通过完整检查。
- Adirondack 读入、SGBM、Q 重投影、评价与输出成功：EPE 1.147992 px、P95 2.648650 px、bad2 5.7671%、参考域内覆盖率 97.8322%。
- 目标环境生成的 PLY 有 299953 个顶点，与有效掩码和指标报告一致；无效深度为 NaN。
- Australia 测试场景无 GT 的流程成功，评价状态明确标记为未评估。
- `src/offline_middlebury.ipynb` 使用指向目标环境解释器的独立 `svp_env` 内核，由 nbclient 执行全部 4 个代码单元格，无错误输出；不是以普通 exec 代替内核。已执行副本保存在忽略目录，提交的 Notebook 保持未执行状态并显示实际内核版本。

本机验证日志、版本 JSON、执行后的 Notebook 和两个场景输出保存于 `artifacts/target_environment_verification/`。这些产物没有提交进 Git，后续协作者可按上述命令重新生成自己的记录。

## 本机环境处理与剩余范围

本机未预装 Conda/SVP_env，创建时使用项目内的缓存及短路径映射处理 Windows 长路径问题。ZeroMQ 在系统临时目录下触发 `epoll.cpp:73`，将测试进程的 TEMP/TMP 指向项目内可写目录后，本地套接字与 Notebook 内核执行均成功；未改动 ZeroMQ 版本。相关上游问题见 [libzmq #4730](https://github.com/zeromq/libzmq/issues/4730)，本记录仅陈述本机实测现象，不推断其他机器也会触发。

JupyterLab 图形界面尚未手工验收；真实相机采集、棋盘格标定、原始图像校正、独立距离精度与重复计时仍待完成。PR #8 已关闭，本次上传工作分支供重新审阅，不将离线验收称为 V0 全部完成。
