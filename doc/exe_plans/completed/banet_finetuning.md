# BANet监督微调与本机训练验证

2026-11-15；状态：已验证并归档。官方数据获取、固定预算1000步监督微调、本机训练实测及独立数组核验完成。测试改善与验证退步并存，不设为默认匹配器；未购买云算力，未提交或推送。

## 完成条件

1. 从官方来源获取独立于现有Middlebury评价图的数据、真值与有效掩码，记录来源、许可、下载与文件哈希。
2. 按物理场景分组并冻结train/validation/test，禁止同一场景不同视角或裁剪跨集合；仅训练集增强，验证集选checkpoint，测试集不选参。
3. 同一SVP_env及已验证官方checkpoint；真实前向、反向、优化器更新测量显存和耗时，检查梯度有限及参数确实更新。合成负载只用于容量测试，不宣称精度。
4. 完成有固定预算、验证、日志、best/last checkpoint与断点状态的小规模监督微调；预训练与微调模型用完全相同评价协议比较。
5. 保留负结果、失败与限制，更新项目导航。数据与权重在忽略目录，默认SGBM不改，不提交或推送。

## 当前证据与决策

- RTX5060 Laptop 8151MiB，torch2.7.1+cu128；真实前向、反向与AdamW更新已通过，1000步训练未发生OOM。
- 官方训练用256×512裁剪、AdamW、OneCycleLR、192视差；损失为最终输出SmoothL1加0.3倍辅助输出SmoothL1。微调可采用较小batch与显式低学习率；它是独立的小规模实验，不声称复现作者200k/50k训练步骤。
- 现有15个Middlebury场景已经反复观察，未用于本轮训练或模型选择。已使用ETH3D官方公开训练数据，物理场景划分见下文。
- 硬件阻碍不依赖相机标定；若8GB能完成有用裁剪和batch的训练，不因作者使用3090直接建议租卡。云费用与购买需用户决定，本轮先提供测量证据。

## 官方数据与冻结协议

- [ETH3D下载](https://www.eth3d.net/datasets)、[格式文档](https://www.eth3d.net/documentation)：已校正灰度左右图，PFM float32像素视差，+inf无真值，mask=0/128/255。PNG以RGB三通道复制灰度读入，原尺度0..255，不另做ImageNet归一化。
- `two_view_training.7z` 14283045字节、`two_view_training_gt.7z` 14863383字节；官方直链及固定SHA256在`config/banet_finetuning.json`，下载记录在`artifacts/banet/data/eth3d/downloads.json`。数据依ETH3D首页[CC BY-NC-SA 4.0许可](https://www.eth3d.net/)使用。独立实测解压27对，图像/GT尺寸和有限值/掩码一致，所有非遮挡GT均在192范围内。
- 训练14对：delivery_area/electro/forest；验证4对：terrains；本地holdout测试9对：facade/playground/terrace。每个物理地点所有编号、l/s基线版本均在同一组；不是随机裁剪后分组。此“测试集”从官方公开训练集中划出，不是ETH3D隐藏真值官方测试集，也不是榜单成绩。
- 配置、全部左右图/GT/mask文件哈希、组别与样本名、执行源码哈希在`protocol.json`中冻结；数据源不与先前Middlebury评价集混用。训练前只检查测试文件格式和有效性，不计算测试预测误差。

## 实现与训练设置

- 同一Python3.11.18 SVP_env，没有更换环境或增装训练依赖；复用前轮严格加载的官方SceneFlow checkpoint。
- `stereo_training_data.py`独立读取数据、PFM底向上行序及mask，生成共享位置的左右裁剪；裁剪不缩放，视差单位不变，排除右对应点位于裁剪图之外的监督点。仅训练集使用相同的亮度/gamma变换；验证和测试原图原尺寸、不增强。
- `banet_training.py`实现final SmoothL1+0.3×auxiliary SmoothL1，显式应用已知GT与视差范围mask。上游train_stereo.py虽接收valid参数但未用于loss；本适配尊重它，防止稀疏GT无效区进入监督。训练允许已知遮挡GT；非遮挡评价严格使用mask=255。
- 冻结预算1000步、256×512裁剪、batch4、FP32、AdamW、OneCycle、最大学习率2e-5、梯度范数裁剪1.0。小batch保留预训练BN运行统计，BN仿射和其他非固定参数仍学习；这与原论文batch16训练设置有区别，不称为官方训练复现。
- 每100步全分辨率验证，按验证集像素池化MAE选checkpoint，含step0作为不退步对照；验证和测试评价域固定为有效非遮挡GT，预测越界/大误差不会被过滤，非有限预测直接失败。
- 保存训练日志、验证轨迹、best权重、last完整训练状态（模型、优化器、调度器、随机数、协议身份）。实现`--resume`，完整checkpoint可用weights_only=True载入且状态有限；尚未进行真实中断后恢复的端到端实验，不声称位级重现保证。

## 本机容量与实训证据

| 实验 | 输入/批量 | PyTorch峰值allocated | reserved | 同步耗时 |
|---|---|---:|---:|---:|
| 合成完整训练步 | 256×512 / 1 | 1253.62MiB | 1272MiB | 163.81ms/步 |
| 合成完整训练步 | 256×512 / 2 | 2441.75MiB | 2530MiB | 160.73ms/步 |
| 合成完整训练步 | 256×512 / 4 | 4829.45MiB | 4948MiB | 163.34ms/步 |
| ETH3D实际1000步 | 256×512 / 4 | 4833.80MiB | 4964MiB | 平均225.71ms/步 |

合成测量每批量2步预热+5步计时，包含前向/反向/梯度裁剪/AdamW，权重丢弃；仅作容量证据。真实训练计时还包含CPU增强和数据传输，含周期验证/状态保存总时长233.54秒（约3分54秒）。不外推更大裁剪、batch16、BANet-3D、全量从零训练或租卡费用。

实际1000步损失和梯度全部有限；与初始权重比较656个张量变化，633个BN统计张量保持不变，AdamW状态有限。训练前25步平均加权损失34.8656，后25步0.0611；这属于训练目标下降，不等于未见场景精度改善。

## 精度结果与模型选择

| 数据域/模型 | MAE/px | Bad1 | Bad2 |
|---|---:|---:|---:|
| 验证：预训练step0 | 0.4210 | 8.27% | 1.35% |
| 验证：固定step1000 | 0.4725 | 8.44% | 1.10% |
| 本地测试：预训练step0 | 62.4675 | 63.86% | 58.07% |
| 本地测试：固定step1000（次级诊断） | 0.4231 | 7.46% | 3.39% |

- 主协议按验证MAE选出**step0**，因此`test_report.json`中的before与after（验证选择）相同。未把验证退步隐瞒成稳定升级。
- 另以`banet_finetuning_last.py`检查固定预算终点step1000，记录在`fixed_last_test.json`。这项次级诊断在主选择结束后进行，未根据测试结果再训练、换参数或挑训练步数；不得据此倒改主选择规则。它显示测试场景大面积错配被修复，同时验证MAE退步，单一小验证域不足以覆盖泛化差异。
- 9个测试样本step1000自身MAE均低于起点，具体逐样本指标在JSON；例如playground_1s 22.0096→0.2685px，terrace_1s 99.0356→0.2226px。
- 独立`predict_banet`适配器复算初始terrace_1s，和训练评价入口保存的预测在共同有效点最大差0px；排除了该样例读图/缩放接口差异造成的大误差。对训练域/光照敏感的具体机制尚未做消融，不能把现象归因当成已证实原因。
- 测试像素1419966、验证像素1102283；按固定GT域池化，Bad1/Bad2严格大于1/2px。统计不能与旧Middlebury有效预测交集指标直接比较。

## 验证、产物与复现

```sh
python -s -m src.eth3d_prepare
python -s -m src.banet_training_probe
python -s -m src.banet_finetuning
python -s -m src.banet_finetuning_last
python -s -m src.banet_finetuning_plots
python -s -m src.banet_checkpoint_export
```

配置分别在`config/banet_finetuning.json`、`banet_training_probe.json`和`banet_finetuning_views.json`。数据准备需要7z-capable bsdtar，本机使用Windows自带tar3.8.9。已存在训练记录须显式resume；已有最终测试报告时拒绝训练覆盖。旧运行不能仅换超参后继续报告同一holdout为未见测试。

- `artifacts/banet/finetuning/eth3d-v1/`：protocol/model/status、训练/验证轨迹、test_report、fixed_last_test、audit、adapter_equivalence、原始预测数组及实图。验证曲线和两幅预定样本对照已渲染检查。
- `checkpoint_best.pth`为验证选择的step0；`checkpoint_last.pth`为step1000的完整可恢复状态；`checkpoint_finetuned.pth`为step1000纯模型权重，导出与原状态逐张量回读一致。SHA256与来源在export.json，推理载入参数在`config/banet_finetuned_model.json`，由既有`load_banet`校验使用。该配置明确标记非验证选择，不覆盖原模型。
- 导出权重通过既有`load_banet`严格加载，载入模型逐张量等于完整训练状态；确定性模式连续两次playground_1s推理最大差0px。归档次级诊断使用CUDA默认算法，与后续确定性复算存在最大0.02319px、平均0.000313px浮点差异，因此不宣称跨算法逐位预测复现；数值审计以归档数组为准。记录见export_inference_check.json。
- 39项unittest通过；新增6项覆盖物理场景泄漏拒绝、PFM行序/单位、成对裁剪/可见性、无效GT不进入loss、BN状态和仿射训练、固定评价域不丢弃难点。
- 9对测试的预训练/主选择/固定末步三组数组独立复算通过，协议源码哈希一致；原默认CLI复跑Adirondack MAE1.3039442220px、覆盖率82.125039%不变。未验证真实相机测距。

## 结论与下一步

本机足以完成本轮BANet-2D监督微调，无需为此租卡。已经证明有学习和部分跨场景改善，也发现验证退步，未达到稳定替换默认方案的结论。后续应增大真实场景数量和验证场景多样性，再冻结新实验；当前9对holdout已经揭示，后续若据此调整方案须用新的独立测试来源。ETH3D仅27对，且验证4对来自一个物理地点，不能代表通用室内或真实设备精度。更大数据/模型的训练需求再依据实测决定是否租算力。
