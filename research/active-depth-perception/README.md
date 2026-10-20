# 方案1 论文库 · 索引

> **方案1 = 会看重点的双目测距模块**（置信度驱动的主动双目深度感知）
> 采集日期：2026-10-04　｜　共 **46 篇全文 PDF**，371.9 MB
> 采集规则：**① 优先老师推荐清单中的论文　② 年份尽可能新**

---

## 一、采集规则与来源

| 优先级 | 范围 | 数量 | 年份 |
|---|---|---|---|
| **第一优先** | 老师推荐清单（75 条）中与方案1相关的论文，按年份从新到旧 | 21 篇 | 1975–2022 |
| **第二优先** | 方案1 技术支撑文献（立体置信度、不确定度、选择性预测、主动视觉、自适应计算、事件立体、能效） | 43 篇 | 2016–2026 |
| 不重复收录 | 项目已有的 `论文收集/`（30 篇）与 `核心与综述论文/`（13 篇） | — | 2024–2026 |

**全文来源**：arXiv 开放预印本；以及 Unpaywall 解析到的**合法开放获取版本**（出版社 OA、作者机构库）。付费墙文献不绕过，记录 DOI 与获取方式。

> 为什么是两个优先级：老师推荐清单以神经科学与认知科学为主，其中与双目测距直接相关的只有 21 篇，且大多是经典（年份偏老）；而支撑方案1的技术内容属于计算机视觉，清单没有覆盖。
> 所以——**第一优先提供认知立论，第二优先提供技术实现**，两边都需要。

---

## 二、目录结构

| 目录 | 内容 | 篇数 | 大小 |
|---|---|---|---|
| `01_老师推荐清单_立体与深度感知` | 第一优先 | 3 篇全文 + 18 篇索引 | 6.0 MB |
| `02_立体匹配置信度与不确定度` | **技术核心**：置信度怎么算、不确定度怎么建模 | 11 | 75.4 MB |
| `03_不确定度驱动的鲁棒与高效匹配` | 用不确定度指导计算分配 | 5 | 58.8 MB |
| `04_选择性预测与风险管理` | 元认知那一层：何时拒绝输出 | 7 | 42.2 MB |
| `05_主动视觉与中央凹感知` | "只看重点"的算法与仿生实现 | 7 | 82.9 MB |
| `06_自适应计算与anytime感知` | 算力/时间预算下的动态决策 | 6 | 22.9 MB |
| `07_事件相机立体视觉` | 后续研究路线 | 4 | 60.1 MB |
| `08_能效与嵌入式感知` | 能耗作为一等指标 | 3 | 23.6 MB |
| `_metadata` | 采集清单（CSV / JSON） | — | — |
| `_tools` | 采集脚本（可复现、可增量更新） | — | — |

---

## 三、第一优先：老师推荐清单（21 篇，按年份从新到旧）

标 ★ 的是与方案1关系最直接的。

| 年份 | 文献 | 出处 | 与方案1的关系 | 状态 |
|---|---|---|---|---|
| 2022 | Favila, Kuhl & Winawer — Spatial tuning properties in human visual cortex | Nature Communications | 低｜视觉皮层空间调谐 | ✅ 已下载 |
| 2016 | Moutsiana et al. — Cortical idiosyncrasies predict object size perception | Nature Communications | 低｜物体大小知觉 | ✅ 已下载 |
| 2012 | Nagata et al. — **Depth perception from image defocus in a jumping spider** ★ | Science | 高｜跳蛛靠离焦测距，仿生依据 | ✅ 已下载 |
| 2010 | Nardini, Bedford & Mareschal — Fusion of visual cues is not mandatory in children | PNAS | 高｜深度线索融合与权重 | 🔓 PMC |
| 2010 | Rossi & Roorda — Visual resolution and cone spacing in the human fovea | Nature Neuroscience | 中｜中央凹采样极限 | 🔓 PMC |
| 2008 | Nadler, Angelaki & DeAngelis — Depth from motion parallax | Nature | 高｜运动视差深度通路 | 🔓 PMC |
| 2007 | Parker — **Binocular depth perception and the cerebral cortex** ★ | Nature Rev. Neuroscience | 高｜立体视觉权威综述 | 📚 图书馆 |
| 2006 | Fernald — Casting a genetic light on the evolution of eyes | Science | 中｜眼的演化，仿生立论 | 📚 图书馆 |
| 2005 | Welchman et al. — **3D shape perception from combined depth cues** ★ | Nature Neuroscience | 高｜多深度线索融合 | 📚 图书馆 |
| 2002 | Ernst & Banks — Visual and haptic information integration | Nature | 中高｜最优融合的经典依据 | 📚 图书馆 |
| 2001 | Wexler et al. — Self-motion and the perception of stationary objects | Nature | 中｜自运动参与深度知觉 | 📚 图书馆 |
| 2001 | Ooi, Wu & He — Distance from angular declination below the horizon | Nature | 中高｜距离判断的多线索加权 | 📚 图书馆 |
| 2000 | Ernst, Banks & Bülthoff — Touch can change visual slant perception | Nature Neuroscience | 中高｜线索权重可被训练改变 | 📚 图书馆 |
| 2000 | Kaufman & Kaufman — Explaining the moon illusion | PNAS | 低｜距离知觉背景 | 🔓 PMC |
| 1999 | Roorda & Williams — Arrangement of the three cone classes | Nature | 中｜非均匀采样 | 📚 图书馆 |
| 1998 | DeAngelis, Cumming & Newsome — **MT and stereoscopic depth** ★ | Nature | 高｜视差神经编码的因果证据 | 📚 图书馆 |
| 1998 | Coppola et al. — Distribution of oriented contours in the real world | PNAS | 中｜自然图像统计 | 🔓 PMC |
| 1993 | Azzopardi & Cowey — **Preferential representation of the fovea in V1** ★ | Nature | 中高｜中央凹超表征＝只看重点的生物学依据 | 📚 图书馆 |
| 1983 | Rossel — **Binocular stereopsis in an insect** ★ | Nature | 高｜螳螂中央小区立体测距，核心仿生依据 | 📚 图书馆 |
| 1980 | Held, Birch & Gwiazda — Stereoacuity of human infants | PNAS | 高｜立体视锐度与精度量级 | 🔓 PMC |
| 1975 | Blake & Hirsch — Binocular depth deficits after monocular deprivation | Science | 中高｜双目细胞的必要性 | 📚 图书馆 |

**统计**：✅ 已下载 3 篇　｜　🔓 PMC 免费 6 篇　｜　📚 需校园网/图书馆 12 篇

DOI、卷期页码与获取链接见 `_metadata/老师推荐清单_获取情况.csv`。

---

## 四、第二优先：技术支撑文献（43 篇）

### 02_立体匹配置信度与不确定度（11 篇）—— 最先读这组

| 日期 | arXiv | 标题 |
|---|---|---|
| 2021-01-02 | 2101.00431 | **On the confidence of stereo matching in a deep-learning era** ← 本方向最重要 |
| 2024-01-22 | 2401.12001 | Modeling Stereo-Confidence via Disparity Plane Sweep |
| 2024-12-24 | 2412.18703 | Probabilistic Modeling of Disparity Uncertainty |
| 2024-08-05 | 2408.02796 | Gaussian Mixture based Evidential Learning |
| 2024-04-09 | 2404.06273 | Robust Confidence Intervals using Possibility Theory |
| 2023-08-01 | 2308.00728 | ELFNet: Evidential Local-global Fusion |
| 2023-03-31 | 2304.00152 | Learning the Distribution of Errors |
| 2020-08-14 | 2008.06447 | Self-adapting confidence estimation for stereo |
| 2019-05-17 | 1905.07287 | CNN-based Cost Volume Analysis as Confidence Measure |
| 2025-08-02 | 2508.01275 | Disparity Confidence in Unsupervised Stereo |
| 2025-05-31 | 2506.00324 | Uncertainty-Based Learning Difficulties |

### 03_不确定度驱动的鲁棒与高效匹配（5 篇）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-06-01 | 2606.02906 | Depth from Dual Differential Defocus and Stereo Consensus |
| 2025-08-02 | 2508.01303 | Uncertainty-guided Data Augmentation |
| 2023-07-26 | 2307.14071 | **Uncertainty Guided Adaptive Warping for Robust and Efficient Stereo** |
| 2023-07-31 | 2307.16509 | Uncertainty-based Pseudo-label |
| 2020-02-10 | 2002.03663 | Uncertainty Estimation via Probabilistic Deep Learning |

### 04_选择性预测与风险管理（7 篇）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2025-09-15 | 2509.11520 | Know What You Don't Know: Selective Prediction for Early Exit DNNs |
| 2025-08-11 | 2508.07556 | Uncertainty-Driven Reliability |
| 2024-10-31 | 2410.24029 | Joint Training for Selective Prediction |
| 2024-07-01 | 2407.01032 | **Overcoming Common Flaws in Evaluation of Selective Classification** |
| 2024-05-19 | 2405.11533 | Hierarchical Selective Classification |
| 2024-02-16 | 2402.10665 | Soft Dice Confidence |
| 2020-09-27 | 2009.12840 | Adaptive confidence thresholding for depth estimation |

### 05_主动视觉与中央凹感知（7 篇）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-07-01 | 2607.00780 | SpiralFovea: Input-Adaptive Foveated Tokenization |
| 2026-06-01 | 2606.02565 | Policy-based Foveated Imaging and Perception |
| 2026-03-23 | 2603.22570 | CanViT: Toward Active-Vision Foundation Models |
| 2026-03-10 | 2603.09613 | A Saccade-inspired Approach to Image Classification |
| 2026-02-03 | 2602.03766 | **FOVI: A biologically-inspired foveated interface** |
| 2025-07-21 | 2507.15833 | Look, Focus, Act: Human Gaze and Foveated ViT |
| 2025-02-10 | 2502.06747 | Wandering around: bioinspired visual attention |

### 06_自适应计算与anytime感知（6 篇）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-07-04 | 2607.05445 | BitFair: Learnable Early Termination Accelerator |
| 2026-03-06 | 2603.06351 | DC-DiT: Adaptive Compute and Elastic Inference |
| 2023-01-30 | 2301.13195 | Adaptive Computation with Elastic Input Sequence |
| 2022-08-25 | 2208.12181 | **Anytime-Lidar: Deadline-aware 3D Object Detection** |
| 2018-01-01 | 1801.00508 | Depth-Adaptive Computational Policies |
| 2016-12-07 | 1612.02297 | Spatially Adaptive Computation Time（源头工作） |

### 07_事件相机立体视觉（4 篇）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-04-02 | 2604.02331 | EventHub: Generalizable Event-Based Stereo Networks |
| 2025-09-18 | 2509.18184 | **URNet: Uncertainty-aware Event-based Stereo Depth** |
| 2025-06-07 | 2506.06918 | Reading in the Dark with Foveated Event Vision |
| 2024-09-26 | 2409.17680 | **Event-based Stereo Depth Estimation: A Survey** |

### 08_能效与嵌入式感知（3 篇）

| 日期 | arXiv | 标题 |
|---|---|---|
| 2026-08-16 | 2608.15830 | MITE-Net: SWaP-Optimized Edge Perception |
| 2026-07-06 | 2607.04921 | Efficient Perception with Neuromorphic Computing |
| 2026-05-07 | 2605.06927 | XiYOLO: Energy-Aware Object Detection |

---

## 五、建议阅读顺序

1. **认知立论**（第一优先 4 篇）：Rossel 1983 → Azzopardi & Cowey 1993 → Nagata 2012 → Parker 2007。讲清"生物为什么不做全域精细感知"，报告引言直接用。
2. **技术核心**（02 组）：2101.00431（置信度评测）→ 2401.12001（置信度建模）。
3. **算力该花在哪**（03 + 06 组）：2307.14071 → 2208.12181。
4. **元认知那层**（04 组）：2407.01032（评测方法学）→ 2410.24029（选择性预测）。
5. **只看重点的现成做法**（05 组）：2602.03766（FOVI）→ 2606.02565（策略化中央凹）。
6. **误差与标定**：项目已有 `核心与综述论文/` 里的鱼眼双目误差解析 + IEEE TIM 畸变模型。
7. **展望章节**（07 组）：2409.17680。

---

## 六、付费墙文献怎么拿到

12 篇 Nature / Science 经典没有合法开放版本，按顺序取：

1. **校园网直连**：多数高校订阅了 Nature 与 Science，校园网内按 DOI 直接下载，最快。
2. **图书馆文献传递 / 馆际互借**：提交 DOI，通常 1–2 个工作日。
3. **PMC 免费阅读**：标 🔓 的 6 篇在 PubMed Central 免费（需在浏览器打开，本机对 PMC 的 PDF 直链有限制）。
4. **引用**：拿不到全文也可正常引用，完整卷期页码见 CSV。

---

## 七、复现与增量更新

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File "research\active-depth-perception\_tools\collect_papers.ps1"
pwsh -NoProfile -ExecutionPolicy Bypass -File "research\active-depth-perception\_tools\resolve_tier1_oa.ps1"
```

**必须用 `pwsh`（PowerShell 7）**：Windows PowerShell 5.1 会把 UTF-8 无 BOM 脚本按 ANSI 解析，中文会乱码并报语法错误。

**采集清单**：`_metadata/采集清单.csv`、`_metadata/老师推荐清单_获取情况.csv`。

**增量更新**：在 `collect_papers.ps1` 的 `$tier2` 里加 arXiv ID 即可，脚本会跳过已下载的文件。
