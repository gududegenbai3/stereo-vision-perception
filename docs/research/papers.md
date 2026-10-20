# 论文索引

完整 46 篇论文的分类、获取状态和阅读顺序见
[research/active-depth-perception/README.md](../../research/active-depth-perception/README.md)。

本文只保留进入项目所需的最短路径。

## 先建立认知框架

- Rossel (1983), *Binocular stereopsis in an insect*：昆虫也能做立体测距，且只在很小的区域精细处理。
- Azzopardi & Cowey (1993), *Preferential representation of the fovea in V1*：中央凹在视觉皮层中的超表征。
- Nagata et al. (2012), *Depth perception from image defocus in a jumping spider*：离焦作为深度线索。
- Parker (2007), *Binocular depth perception and the cerebral cortex*：立体视觉的神经科学综述。

## 技术核心

- [立体匹配置信度与不确定度](../../research/active-depth-perception/papers/02_立体匹配置信度与不确定度/)：
  先读 `2101.00431`（置信度评测）和 `2401.12001`（置信度建模）。
- [不确定度驱动的鲁棒与高效匹配](../../research/active-depth-perception/papers/03_不确定度驱动的鲁棒与高效匹配/)：
  重点看 `2307.14071`，理解不确定度如何影响计算分配。
- [选择性预测与风险管理](../../research/active-depth-perception/papers/04_选择性预测与风险管理/)：
  重点看 `2407.01032`，学习如何正确评测“保留哪些像素”。
- [主动视觉与中央凹感知](../../research/active-depth-perception/papers/05_主动视觉与中央凹感知/)：
  重点看 `2602.03766`（FOVI）和 `2606.02565`（策略化中央凹）。

## 经典基线

- Scharstein & Szeliski (2002)：双目评测方法学。
- Hirschmüller (2008)：SGM/SGBM 原始论文。
- Zhang (2000)：张正友标定法。

## 阅读顺序

1. 认知框架：Rossel → Azzopardi & Cowey → Nagata → Parker。
2. 置信度：`2101.00431` → `2401.12001`。
3. 计算分配：`2307.14071` → `2208.12181`。
4. 选择性预测：`2407.01032` → `2410.24029`。
5. 主动视觉：`2602.03766` → `2606.02565`。

论文笔记写入 `docs/research/notes/`，不要修改 `research/` 下的 PDF。
