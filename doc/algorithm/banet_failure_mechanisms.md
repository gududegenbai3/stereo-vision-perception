# BANet失误阶段诊断与模型机制候选

核对日期：2026-11-15。范围为固定step1000候选的只读诊断与论文/源码研究，未引入新匹配器、未训练或调整参数。当前24对手机图已经观察，本记录不作为新的盲测结果。运行状态见[执行记录](../exe_plans/completed/banet_failure_mechanisms.md)。

## 1. 已运行验证的事实

原48对比较与权重不变。进一步使用24对手机图中SG192和BANet共同有效的33739212个GT像素，GT相邻差>4原图px定义视差突变种子，距种子>16px视为远离边界；大错误定义>10px。

- 共4417909个大错误，其中3528119个（79.8595%）远离GT边界；这部分占大错误绝对误差总量89.9651%。不能把主要失误概括成轮廓偏移。
- 去掉图像四周128px后，仍有2753768个大错误，其中76.8913%远离物体视差边界。外边框不是唯一解释。
- 在skates2/bandsaw2/curule2/skiboots1捕获spa_att、两分支最终logits、上采样logits，按实际公式重构输出：四例最大重构差0输入px；重跑最终预测与归档数组最大差0原图px。
- 保持模型eval与inference_mode，权重哈希不变；捕获的attention只代表分支权重，softmax只代表模型内部候选权重，均未做正确性概率校准。

进一步选择距图像外边框至少128px、距GT边界>16px且SGBM误差<=2px的内部失败点。它们是明确的事后诊断点，不是随机性能样本。

| 图像、原图坐标 | GT | 聚合后粗视差恢复到该位置 | 最终BANet | SGBM | 上采样粗候选范围 |
|---|---:|---:|---:|---:|---:|
| skates2 (1554,181) | 103.6420 | 8.2105 | 8.9306 | 104.0000 | 6.3714–11.4794 |
| bandsaw2 (1778,773) | 193.0231 | 10.2481 | 11.9977 | 191.4063 | 9.2390–16.1354 |
| skiboots1 (940,951) | 140.9208 | 6.3811 | 5.9413 | 140.2188 | 5.3965–6.7391 |

单位均为原图像素。skates2点距GT边界217px、距图像外边框181px。由保存的粗视差平台中心恢复上采样支持值的方法，已在四个原始hook点与直接捕获的低分辨率张量核对一致。非负和为1的9邻域权重及外部双线性恢复仍为凸组合；GT不在所有参与候选的范围内时，仅改上采样权重无法产生该正确视差。因此这些点的错误已在匹配/聚合后的粗视差中形成。

输出分布不是“总在两个正确表面之间取平均”的证据：原始skiboots1 probe中主要候选为0和24原图px，GT152.23px；最大内部softmax权重0.637仍然错误。正确control curule2的最大权重0.607，不能直接按此权重大小判断对错。该观察不等于完成全数据校准研究。

## 2. 源码核对与尚未证实的成因

BANet固定上游commit `7cb83f7cca99a249e1fcec8ff556e4ac73facd70`：

- [BANet.py L133–144](https://github.com/gangweix/BANet/blob/7cb83f7cca99a249e1fcec8ff556e4ac73facd70/banet-2d/core/BANet.py#L133)：1/4输入分辨率相关体、左图多尺度attention分流、两分支聚合、softmax期望视差。此attention没有语义分割或GT深度边界监督。
- [submodule.py L41–45](https://github.com/gangweix/BANet/blob/7cb83f7cca99a249e1fcec8ff556e4ac73facd70/banet-2d/core/submodule.py#L41)：沿48个候选计算期望值，不等于离散argmax。
- [submodule.py L141–153](https://github.com/gangweix/BANet/blob/7cb83f7cca99a249e1fcec8ff556e4ac73facd70/banet-2d/core/submodule.py#L141)：低网格3×3邻域展开，最近邻扩展后按学习权重求和；没有再查询左右匹配、修改粗视差的迭代环节。

当前mobile输入缩小一半，再在1/4网格匹配，相邻代价体采样点相距8原图px；这不是感受野仅8px，也不意味着输出只能是8的整数倍。薄结构更易混合的解释合理，但上述内部面的大错不能仅归于采样量化，正确GT均在允许搜索范围内。

本项目微调只有3个物理训练地点的14对灰度图、1个验证地点，验证MAE0.4210→0.4725。数据覆盖不足、训练域偏移、错误聚合先验和空间分辨率均是待消融因素。尚未分离“特征相关本身错”和“聚合把正确相关压掉”，没有证据宣布网络欠拟合或参数容量不足。

## 3. 候选机制及最小证伪实验

以下为官方论文与固定源码核验，除BANet诊断外均未在本机安装、推理或训练，论文数字不能代替项目结果。

### 优先：IGEV-Stereo，几何体加迭代修正

当前缺口是粗视差大错后缺乏重新查证据的过程。IGEV先用轻量3D几何聚合建立初始视差，再联合几何体与原始相关细节，逐次查询并更新`disp += delta_disp`；高分辨率左图特征参与最终上采样。3D几何聚合本身也可能平滑边缘，原始相关细节正用于补偿这一折中。

固定仓库`gangweiX/IGEV` commit `4a3d937aab788b3ff3ed49b7b5c81340ab46a5eb`：
- [igev_stereo.py L168–175](https://github.com/gangweiX/IGEV/blob/4a3d937aab788b3ff3ed49b7b5c81340ab46a5eb/IGEV-Stereo/core/igev_stereo.py#L168)：3D几何体和初始化；[L199–210](https://github.com/gangweiX/IGEV/blob/4a3d937aab788b3ff3ed49b7b5c81340ab46a5eb/IGEV-Stereo/core/igev_stereo.py#L199)：反复索引、更新及上采样。
- [geometry.py L34–58](https://github.com/gangweiX/IGEV/blob/4a3d937aab788b3ff3ed49b7b5c81340ab46a5eb/IGEV-Stereo/core/geometry.py#L34)：几何体与原始相关同时参与查询。不是每次在全部视差上重新穷举，也不能保证跳出错误解。
- [论文表1–3](https://arxiv.org/html/2303.06615v2)：同为12.60M参数的G+I+S和完整模型，EPE0.48→0.47；同一模型增加迭代改变计算量而不改变参数量。

最小试验：官方预训练固定权重，先在现有已观察图上测8/16/32次迭代，保持输入/尺度/GT域一致，比较全域、边界、弱纹理、显存与耗时。若迭代增加没有精度收益，则不支持“多做几轮求解就足够”。若候选无整体改善，不进入微调。正式泛化结论需新未观察来源；模型训练来源也需单独登记，不能把现有24对重新称新盲测。

代码MIT；下载checkpoint独立条款未核验。原项目依赖/3090记录不证明Python3.11、5060 Windows及8GB可直接训练；先做最小推理兼容探测，仍使用当前SVP_env约束。

### 结构对照：RAFT-Stereo，参数共享的反复更新

固定`princeton-vl/RAFT-Stereo` commit `6e93ed2169bd858dbb43033988563f3b0bb49506`。同行相关金字塔、多个尺度的2D ConvGRU传播信息并迭代修正；最终仍为凸组合上采样，因此关键差别是上采样前粗视差能够被修改，不是“彻底没有上采样混合”。

- [forward L90–133](https://github.com/princeton-vl/RAFT-Stereo/blob/6e93ed2169bd858dbb43033988563f3b0bb49506/core/raft_stereo.py#L90)，[上采样L55–67](https://github.com/princeton-vl/RAFT-Stereo/blob/6e93ed2169bd858dbb43033988563f3b0bb49506/core/raft_stereo.py#L55)，[论文§3](https://arxiv.org/html/2109.07547v1)。
- 可先用纯PyTorch`alt`相关路径以避免自定义CUDA扩展；该commit的`alt_cuda`类抛NotImplementedError，不应误选。代码MIT，权重独立条款未核验。

最小试验同IGEV，作为无3D卷积聚合的机制对照。不同官方权重的数据、骨干和训练预算仍是混杂因素，不能仅凭两模型胜负断言3D卷积的因果作用。本机兼容与显存待验证。

### 后续候选：FoundationStereo，强先验与大规模多样训练

固定`NVlabs/FoundationStereo` commit `6e8806816b533e4d13ddbb95ffa907b797060a62`。冻结DepthAnything侧路特征与CNN特征拼接，再用空间/视差分解3D卷积、视差维attention、迭代更新处理匹配；论文还使用约100万组合成训练对。它针对域外泛化，但优势不能全部归因于模型参数量。

- [extractor.py L333–369](https://github.com/NVlabs/FoundationStereo/blob/6e8806816b533e4d13ddbb95ffa907b797060a62/core/extractor.py#L333)、[foundation_stereo.py L213–253](https://github.com/NVlabs/FoundationStereo/blob/6e8806816b533e4d13ddbb95ffa907b797060a62/core/foundation_stereo.py#L213)、[论文](https://arxiv.org/html/2501.09898v2)。
- 仓库LICENSE明确非商业研究；模型与第三方基础特征许可仍须分别核验。文档测试Linux、列flash-attn/xformers等依赖，固定源码某attention层实际使用PyTorch SDPA；不能将README依赖列表直接等同每层必须。8GB/Windows未测。

最小试验仅固定官方预训练零样本推理作为上限探索，先测显存和兼容，不立刻在14对图上再训练大模型。若运行代价不适合本课程部署，则保留为离线参考，不移入默认主流程。

### 与既有论文衔接：不确定性与可信融合

[Lee2024 DPS](https://arxiv.org/html/2401.12001)通过输入平移后视差应相应变化的几何一致性检测不可靠预测，适合借鉴为外部检查，不会自动产生正确匹配。已有项目DPS实验针对SGBM，尚未证明对当前BANet同样有效。

本地已有[ELFNet论文](https://arxiv.org/html/2308.00728)提出对局部代价体模型与全局Transformer预测估计不确定性、据证据融合；本轮只核对论文描述，未核查其代码/权重许可或运行。可借鉴“用独立证据选择互补估计”，不直接复制双网络。最小试验应先在训练/验证中校准错误识别，评估风险-覆盖和固定误差预算的检测率；不能用测试GT逐点挑SGBM/BANet结果当作有效融合，也不能简单平均两个不同表面的视差。

## 4. 是否增大参数

当前BANet实际参数5456649，没有同数据/同训练预算的容量消融，无法认定容量是瓶颈。两项论文证据仅说明值得比较结构与规模：

- [BANet表1](https://arxiv.org/html/2503.03259v2)：2D→3D的SceneFlow EPE0.57→0.51，MACs39G→85G。这同时改变了视差维几何组织方式，不是单纯增宽；不能套到本项目边界或显存数字。
- IGEV表3：更换骨干、参数12.60M→18.04M，EPE0.47→0.45；存在收益但并不与参数增长成比例。这是论文配置，不是本机checkpoint统计。

当前建议先验证IGEV式几何加迭代、扩大真实物理场景覆盖，再考虑BANet-3D或增宽。增加迭代共享参数而增加求解计算量，可以与单纯加参数分开验证。模型扩大、训练数据扩大、计算预算扩大要分别记录；不能从现有BANet的4.72GiB训练显存推断替代模型也能训练。

## 5. 本轮复现与限制

`config/banet_failure_diagnosis.json`声明当前数据与诊断阈值；`src/banet_failure_diagnosis.py`只读捕获阶段并核对输出，`src/banet_failure_plots.py`进一步分离图像外边框、复核粗网格支持并绘图。

```sh
python -s -m src.banet_failure_diagnosis
python -s -m src.banet_failure_plots
```

产物：`artifacts/banet/failure_mechanisms/`中的distance_analysis、image_border_analysis、summary、central_interior_probes、stage数组及interior-stage-skates2.png。诊断已运行，替代模型只有源码/论文核验，未下载模型或安装依赖。没有正式边缘检测精度评测、容量消融、数据量消融或真实设备测距；本轮只定位失误阶段并给出可验证候选。
