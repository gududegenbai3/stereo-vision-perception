# BANet-2D 预训练推理验证

日期：2026-11-15；状态：已验证并归档。预训练推理尝试和15场景比较已完成，未训练、未推送，整体精度未改善。

## 范围
- 保持项目Python3.11与现有SVP_env，依赖以requirements-banet.txt声明，增补支持RTX5060的CUDA PyTorch；不另换Python版本，不修改默认OpenCV流程。
- 官方BANet-2D代码与SceneFlow预训练权重用于推理；权重、上游源码和运行产物放artifacts/banet/，不提交大文件。核对来源、模型结构、严格加载及SHA256。
- 输入先采用已有Adirondack校正图，成功后扩展至同一批15诊断场景。按官方RGB与数值尺度、32倍数填充处理；移除填充后还原原尺寸视差，沿用原Q与评价。
- 显卡已核对为RTX5060 Laptop，8151MiB显存，驱动592.01；已验证CUDA卷积、矩阵乘法及完整模型推理，并测量峰值显存、同步耗时及误差，不将推理称训练。
- 对比原SGBM：同时报告共同有效像素误差、模型自身有效域、覆盖率和无效点。max_disp遵循官方模型结构，不使用GT选参。
- 若上游timm旧接口需要适配，仅修复版本兼容与关闭随后将被完整checkpoint覆盖的ImageNet权重下载；严格加载失败不得使用随机权重冒充预训练模型。
- 本轮不训练、不推送；面向用户的原理讲解与实图放会话，仓库只保留实现、方法约定和验证事实。

## 实现与环境

- `src/banet.py`提供`load_banet`和`predict_banet`：输入已校正BGR uint8图像对，返回现有`DisparityResult`（float32 H×W，单位px，无效NaN），以及单独的软分支权重、计时和显存。复用原重投影模块及原Q。
- `src/banet_experiment.py`读取`config/banet.json`，比较原SGBM、同为192搜索范围的SGBM及BANet；GT仅用于预测后的评估。`src/banet_plots.py`独立复算并绘图，配置为`config/banet_views.json`。
- 原SVP_env增补`requirements-banet.txt`：Python3.11.18，OpenCV4.13.0，NumPy2.4.6，torch2.7.1+cu128，torchvision0.22.1+cu128，timm0.6.13。CUDA运行时12.8，计算能力12.0（编译架构含sm_120）。使用`python -s`隔离用户级Python包，`pip check`通过。
- Windows显式注册当前Conda的Library/bin并保留DLL句柄。上游源码仅在忽略目录的运行副本适配两处：关闭将被完整checkpoint覆盖的ImageNet下载；timm已将ReLU6融合进bn1时使用Identity作为独立act1。实际bn1包含ReLU6，已检查。未改变参数形状、输入缩放或分支算法。
- 上游：[gangweix/BANet](https://github.com/gangweix/BANet)，固定提交`7cb83f7cca99a249e1fcec8ff556e4ac73facd70`；官方BANet-2D SceneFlow权重来源及SHA256维护于`config/banet.json`。文件22665679字节，全部参数严格加载成功，参数总数5456649。

## 验证方法与结果

Middlebury trainingQ全部15场景、非遮挡有限GT，原尺寸输入不缩放；对称复制填充至32倍数后移除。模型固定max_disp=192，不能直接改为128而沿用该checkpoint。浮点输出已经是原尺寸像素单位，不执行OpenCV的除16操作。匹配有效性仅检查有限值、非负、范围及右图投影不越界；未检查左右一致性，不能称为正确性置信度。

| 方法 | 自身有效域MAE/px | Bad2 | GT覆盖率 |
|---|---:|---:|---:|
| 原SGBM（128） | 2.7447 | 13.77% | 79.88% |
| 同范围SGBM（192） | 3.1875 | 13.87% | 70.51% |
| BANet-2D预训练 | 5.1409 | 22.12% | 98.07% |

按像素池化。Bad2指绝对误差严格大于2px；MAE/Bad2以各方法有效预测和非遮挡GT交集为分母，覆盖率以全体非遮挡有限GT为分母。SGBM有效性规则和搜索ROI与BANet不同，覆盖率不能单独衡量精度。

同一批像素比较：原SGBM与BANet共同有效像素MAE为2.7394/5.0810px，Bad2为13.74%/20.44%；同范围SGBM与BANet共同有效像素MAE为3.1875/5.1421px。更稠密的输出没有带来整体匹配改善。

- 逐场景自身有效域MAE有7/15下降：Adirondack 1.3039→0.7823px，Shelves 2.1748→1.2906px；负例Jadeplant 6.7048→14.6425px、PianoL 14.0230→43.5356px。这是数值误差，不是进程失败。
- 75次CUDA同步FP32前向平均42.92ms，P95 83.48ms；最大allocated150.10MiB，reserved214MiB。每场景预热1次、计时5次；不含IO、预处理、模型加载、GT评估和独立attention可视化前向，不代表相机端到端帧率或训练显存。分辨率随场景变化。
- 实际调用原`reproject_disparity`，保存米制depth和XYZ；未使用独立距离参考，真实设备测距精度待验证。
- `python -s -m unittest discover -s tests -v`：33项通过，0跳过。新增4项覆盖填充/原坐标恢复、像素单位/无越界回绕、共同像素与缺失分母、兼容补丁不修改原文件及异常接口拒绝。
- `python -s -m src.banet_plots`：15场景原SGBM数组和掩码与此前诊断产物逐像素一致；独立float64复算MAE、Bad2、缺失及成对计数通过。Adirondack、Shelves、PianoL实图已渲染检查。

## 复现与证据

原SVP_env安装可选依赖，克隆并固定上述上游提交至`artifacts/banet/upstream`，将官方权重保存到配置指定位置。命令：

```sh
python -s -m pip install -r requirements-banet.txt
python -s -m src.banet_experiment
python -s -m src.banet_plots
```

`artifacts/banet/results/`保留model.json、protocol.json、summary.json、provenance.json、audit.json、每场景report.json/arrays.npz及comparison-*.png。包含权重和源码哈希、版本、兼容修改。已有summary时拒绝覆盖，复跑应在新配置中选择新output_root并相应调整绘图配置。大文件、环境和产物均被忽略。

结论与下一步：5060运行预训练推理已获实测支持，整体精度提升未获支持；保留默认SGBM。后续优先检查失败场景及训练域/输入域差异，用无测试GT反馈的冻结方案比较其他权重或模型。光照/域差异是待验证原因，不能凭本轮断言；未进行微调、消融训练或BANet论文榜单复现。
