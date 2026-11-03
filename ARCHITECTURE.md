# V0 Python 代码架构

`src/model.py` 已实现共享数据模型，`src/datasets.py` 已实现 Middlebury 离线读取，`src/config.py` 已实现 JSON 配置读取与校验，`src/disparity.py` 已实现 SGBM 匹配，`src/reprojection.py` 已实现米制深度与三维重建，`src/evaluation.py` 已实现离线视差评估，`src/outputs.py` 已实现离线产物导出，`src/pipeline.py` 已串联单场景离线流程，`src/cli.py` 已提供命令行入口。下文定义该模块与后续模块的职责和依赖方向。算法、数据单位与验收指标以 [算法规格](doc/algorithm/algorithm.md) 为准，开发顺序见 [V0 执行计划](doc/exe_plans/active/V0_stereo_pipeline.md)。

## 模块职责

CLI 通过 `pipeline.py` 运行单场景离线重建，Notebook 逐步调用相同模块并展示中间结果。采集、真实相机标定和极线校正模块仍待实现。

| 模块 | 职责 |
|---|---|
| `model.py` | 声明共享的数据类和数组类型，区分原始图像对、校正图像对、校正几何及重建结果 |
| `config.py` | 读取并校验 `config/` 下的运行配置，返回有类型的参数 |
| `datasets.py` | 读取双目图、标定文件、真值和掩码，检查场景、尺寸和格式，把数据集约定转成内部类型 |
| `capture.py` | 从相机采集并保存左右图与配对信息 |
| `calibration.py` | 检测棋盘格角点，计算内参、畸变和双目外参 |
| `rectification.py` | 构建校正映射和重投影矩阵，对原始图像对应用校正 |
| `disparity.py` | 用 SGBM 计算像素单位视差，返回匹配有效性信息 |
| `reprojection.py` | 根据视差和校正几何计算深度、三维坐标及重建有效掩码 |
| `evaluation.py` | 根据预测、真值和掩码计算误差与覆盖率 |
| `outputs.py` | 保存数组、掩码、预览图、RGB 点云及指标 |
| `pipeline.py` | 编排已校正 Middlebury 图像的离线重建、评估与可选导出 |
| `cli.py` | 解析命令行参数、校验配置、调用流程并报告失败 |

其余模块随对应任务落地。函数实现集中在 `src/`，Notebook 中保留实验参数选择、调用与展示。

当前共享模型包括 `RawPair`、`RectifiedPair`、`RectificationGeometry`、`DisparityResult` 和 `ReconstructionResult`。校正几何保留匹配和重建所需的图像尺寸、`Q` 与双侧 ROI。旋转、投影矩阵和重映射表由后续校正模块负责。模型字段带类型和单位约定，调用层负责校验，NumPy 数组按引用传递。

## CLI 指令

先激活 `SVP_env`，再在仓库根目录运行。

| 指令 | 用途 |
|---|---|
| `python -m src.cli --help` | 查看参数 |
| `python -m src.cli` | 使用 `config/offline.json` 执行单场景离线流程，按配置导出 |
| `python -m src.cli --config config/offline.json` | 显式指定配置，也可传入其他 JSON 路径 |
| `python -m src.cli --no-export` | 计算重建和指标，跳过文件导出，可与 `--config` 组合 |
| `python -m src.cli --preview` | 计算完成后打开视差与深度预览窗口，可与 `--no-export` 组合 |

可用 `--split trainingQ|testQ` 和 `--scene 场景名` 选择图像对。例如：

```sh
python -m src.cli --split testQ --scene Australia --no-export
```

数据位于其他位置时，使用 `--image-root 路径` 和 `--ground-truth-root 路径` 覆盖配置中的根目录。训练场景需使用与图像匹配的真值目录。CLI 路径相对于当前工作目录，JSON 内的相对路径仍以配置文件所在目录为准。当前读取器支持 Middlebury Q 版目录结构。

`cli.py` 解析参数后调用 `load_offline_config` 和 `run_offline_pipeline`。成功时向标准输出写 JSON，包含场景、指标和导出目录，未导出时目录为 `null`。成功退出码为 0，运行失败为 1，命令行参数错误为 2。错误写到标准错误。配置内的相对路径以配置文件所在目录为准。预览使用交互式 Matplotlib 后端，关闭窗口后命令结束。窗口无法打开时报告预览失败并返回 1，已生成的重建结果仍可用。

## 导入方向

箭头表示左侧可以导入右侧。库依赖如标准库、NumPy 和 OpenCV 不在图中。

```mermaid
flowchart TD
    Entry["CLI / Notebook"] --> Config["配置"]
    Entry --> Pipeline["流程编排"]
    Pipeline --> Stages["数据读写与各算法阶段"]
    Config --> Models["共享类型"]
    Stages --> Models
    Pipeline --> Models
```

- 同层阶段独立，通过参数和返回值交换数据。需要先校正再匹配时，由 `pipeline.py` 顺序调用。
- 共享类型放在 `model.py`。底层模块不得导入流程入口；`__init__.py` 保持轻量，内部从定义模块显式导入。
- 新增跨模块导入时检查依赖方向。若两个模块需要互相导入，先由调用层编排；确有共用的数据类型时下移到 `model.py`。函数内延迟导入和 `TYPE_CHECKING` 不用于规避层级约束。

## 命名与可读性

| 对象 | 约定 | 示例 |
|---|---|---|
| 模块、函数、变量 | 使用 `snake_case`，模块名表达职责，函数名使用动词 | `rectification.py`、`compute_disparity` |
| 类型、常量 | 数据类用 `PascalCase`，模块常量用 `UPPER_SNAKE_CASE` | `RectifiedPair`、`DISPARITY_SCALE` |
| 左右顺序、颜色和单位 | 名称体现容易混淆的语义 | `left_bgr`、`right_gray`、`disparity_px`、`depth_m` |
| 导入和排版 | 使用四空格缩进，按标准库、第三方库、项目模块分组；避免通配导入 | `from src.model import RectifiedPair` |

函数参数、返回值和变量显式标注类型。对外函数的文档字符串说明数组的元素类型、形状、单位和坐标约定，并说明可能抛出的异常。项目内部名称沿用上述风格，调用 OpenCV 时保留其 API 的关键字拼写。

## 数据与错误边界

- 调用层显式传入路径和已校验参数；算法函数接收数据，导入模块时不读文件、不启动相机、不运行实验。
- 原始相机图先校正，再算视差。Middlebury 已校正图像从匹配阶段进入，数据适配层保留主点偏移并统一基线单位。
- 匹配结果使用 `float32 (H, W)`、单位 px；深度使用 `float32 (H, W)`、单位 m，无效值为 `NaN`；有效掩码使用 `bool (H, W)`。预览图在输出层生成。
- 缺失文件抛出 `FileNotFoundError`，非法形状、配置或数据格式抛出 `ValueError`。错误信息应能定位路径、场景或参数。批量实验在入口记录失败，保留样本身份。

`config.py` 和 `datasets.py` 各自只依赖项目内的 `model.py`。调用层将离线配置中的路径和场景交给数据读取层，将校验后的 `SgbmConfig` 交给后续匹配函数。硬件解析只提取设备身份和运行状态，规格字段仍保留在硬件档案。

Middlebury 适配器从校正后的内参与基线构造 `Q`，使用全图作为校正 ROI。PFM 按 [Middlebury 格式约定](https://vision.middlebury.edu/stereo/submit3/upload-format.html) 解码，比例字段的符号决定字节序，原始浮点值已是像素视差。

`disparity.py` 接收校正图像对、几何和已校验的 `SgbmConfig`。返回的 `DisparityResult` 包含像素视差、有效搜索 ROI 和匹配有效掩码。无效视差用 `NaN`；后续重投影阶段再检查正深度与有限三维坐标。

`reprojection.py` 只依赖项目内的 `model.py`。调用方传入 `DisparityResult`、校正图像对及配套几何。重投影结合匹配有效掩码、有限坐标和正深度生成 `ReconstructionResult`，供后续评估与点云导出使用。

`evaluation.py` 只依赖项目内的 `model.py`，由调用方提供预测、可选参考视差和参考掩码。该模块返回结构化指标，Notebook 负责显示，并将指标映射传给产物层。

`outputs.py` 接收 `ReconstructionResult`、指标映射、实验参数和目标路径。运行目录在所有文件写完后发布；每次调用生成独立目录。

`run_offline_pipeline` 接收已解析的 `OfflineConfig`，返回匹配、重建、指标和可选导出目录。Notebook 仍可分别调用各阶段来检查中间结果。
