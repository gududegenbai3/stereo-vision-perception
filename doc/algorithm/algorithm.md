# 双目立体视觉感知：最小可运行算法说明

> 版本：0.2 · 日期：2026-10-17  
> 目标：供 Codex 实现“采集双目图像 → 标定 → OpenCV 三维重建 → 成果评估”的基本流程。  
> 状态：本文是实现规格，尚未验证真实设备上的重建精度。算法基础见 [1,2]，OpenCV 接口见 [3–7]；模块划分、数据格式和验收流程是本项目的工程决策。

## 1. 本阶段交付目标

使用两台固定安装的普通相机，采集棋盘格和静态场景图像，通过 OpenCV 得到视差、深度与彩色点云，并用独立距离参考评价结果。

“场景重建”在本阶段指**单对双目图像的三维点云重建**。必需交付物为原始图像、标定结果、校正对照图、数值视差图、数值深度图、有效掩码、PLY 点云和评估报告。

仅使用 Python、OpenCV 与 NumPy；标准库负责配置、CSV/JSON 和 PLY 文件写入。无需模型训练或 GPU。第一阶段按静态场景、固定分辨率、水平双目布局实现。普通针孔镜头是当前成像模型假设，实际镜头不符合时应先修订模型。

## 2. 流程与模块

| 顺序 | 模块 | 输入 | 输出 | 主要 OpenCV 接口 |
| --- | --- | --- | --- | --- |
| 1 | `capture` | 两台相机、采集配置 | 成对原始图像、采集记录 | `VideoCapture`、`grab`、`retrieve` |
| 2 | `calibrate` | 棋盘格图像对、格子尺寸 | 双目内参、畸变、外参 | `findChessboardCorners`、`cornerSubPix`、`calibrateCamera`、`stereoCalibrate` |
| 3 | `rectify` | 标定参数、原始场景图像对 | 校正图像、映射表、投影矩阵与 `Q` | `stereoRectify`、`initUndistortRectifyMap`、`remap` |
| 4 | `reconstruct` | 校正图像、`Q`、匹配配置 | 视差、深度、掩码、点云 | `StereoSGBM_create`、`compute`、`reprojectImageTo3D` |
| 5 | `evaluate` | 标定与重建结果、独立参考距离 | 指标表和报告 | `projectPoints`、`undistortPoints`，配合 NumPy |

标定是离线步骤，完成后可复用；每次场景重建只需执行校正和重建。镜头调焦状态、相机相对位置、分辨率或裁剪方式变化后，需重新检查标定适用性。

## 3. 双摄像头采集

### 3.1 设备与配对

- 配置中明确 `left_camera_id`、`right_camera_id` 和实际图像尺寸；左右身份在全流程保持一致。
- 检查两台设备均成功打开，实际输出尺寸相同且符合配置；读取失败或尺寸不符时应明确报错。
- 每轮先对两台相机分别调用 `grab()`，再分别 `retrieve()`，随后以相同 `pair_id` 保存两张图像。两侧均成功才登记为完整图像对 [6]。
- 普通 USB 摄像头的软件配对不能保证曝光同步。记录主机读取时间并标明其含义，不能将读取时间差当成真实曝光时间差；第一阶段使用静态场景。

保存无损 PNG 原始图像，避免先裁剪或缩放。使用 `pairs.csv` 记录 `pair_id,left_path,right_path,left_read_time_ns,right_read_time_ns,width,height`；时间由同一主机时钟产生。程序退出时释放两台设备。

### 3.2 两类采集数据

**标定数据**：采集棋盘格在不同位置、距离和倾斜姿态下的图像对，左右图均须看见完整内角点。本项目建议先收集至少 15 对有效图像，另留若干独立图像对用于检查；这是采集起点，不是标定质量保证。

**场景数据**：标定完成后，保持双目结构不动，采集有纹理的静态场景。测距评价另采集至少三个不同已知距离的、正对参考相机的有纹理平面，每个距离采集多对图像以观察重复性。

## 4. 相机标定

### 4.1 基本原理

针孔模型把空间点投影到像素平面：

$$
s\begin{bmatrix}u\\v\\1\end{bmatrix}
=K[R_i\mid t_i]\begin{bmatrix}X_w\\Y_w\\Z_w\\1\end{bmatrix},
\qquad
K=\begin{bmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{bmatrix}.
$$

$K$ 包含像素单位的焦距和主点；$R_i,t_i$ 是棋盘格相对某台相机的姿态。真实镜头还有畸变，由 OpenCV 的畸变系数描述。

张正友方法利用已知平面标定板的多姿态观测估计相机参数，再通过非线性优化减小重投影误差 [1]。本阶段直接调用 OpenCV，不自行编写求解器。

### 4.2 棋盘格定义

配置 `board_cols`、`board_rows` 为**内角点**列数和行数，传给 OpenCV 的顺序为 `(board_cols, board_rows)`；不得填成黑白格子数量。配置 `square_size_m` 为实测相邻角点间距，单位米。

棋盘格三维点按与检测结果相同的顺序生成：

$$
P_{ij}=(i\,a,\ j\,a,\ 0),\qquad a=\texttt{square\_size\_m}.
$$

同一对左右图必须使用相同的角点身份和顺序；拒绝存在翻转或顺序不一致的配对。格子尺寸决定重建的绝对尺度，禁止使用未测量的占位尺寸进行正式测距。

### 4.3 实现步骤

1. 将每对棋盘格图像转为灰度，检测内角点，并用 `cornerSubPix` 精化。任一侧检测失败则整对跳过，记录失败原因。
2. 使用同一组有效配对，分别调用 `calibrateCamera`，得到 `K1,D1` 与 `K2,D2`，保留逐帧姿态以便计算重投影误差 [3]。
3. 将已估计的内参传入 `stereoCalibrate`，使用 `CALIB_FIX_INTRINSIC`，估计固定双目外参 `R,T`。约定空间点满足 $P_R=RP_L+T$ [4]。
4. 保存基线 $B=\|T\|$、拟合返回的 RMS 及有效配对数量；检查参数有限且 $B>0$。

使用 `np.savez` 保存 `calibration.npz`，至少包含 `K1,D1,K2,D2,R,T,image_size,board_size,square_size_m,rms_left,rms_right,rms_stereo`。另存有效 `pair_id` 和 OpenCV 实际版本。`image_size` 始终为 `(width,height)`。

## 5. 去畸变与极线校正

### 5.1 基本原理

极线校正把左右图像映射到共同的虚拟成像平面，使同一空间点的左右像素尽量位于同一行：

$$
v_L\approx v_R,\qquad d=u_L-u_R.
$$

这样即可沿水平方向搜索对应点。$d$ 是左参考视差，单位像素。

### 5.2 实现步骤

1. 调用 `stereoRectify`，设置 `flags=CALIB_ZERO_DISPARITY`、`alpha=0`，输出尺寸保持原尺寸，取得 `R1,R2,P1,P2,Q,roi1,roi2`。
2. 使用两侧内参、畸变、校正旋转和投影矩阵，分别调用 `initUndistortRectifyMap` 生成映射表，再用 `remap` 校正原始图像。
3. 保存 `rectification.npz` 与带水平辅助线的左右拼接预览；预览辅助线不写入匹配输入。检查输出确为水平极线布局；垂直布局不直接送入本流程。

上述参数及接口定义以 OpenCV 官方文档为准 [4]。不再对校正图做额外裁剪或缩放，以保持像素坐标和 `Q` 一致。

## 6. 使用 OpenCV 估计视差

### 6.1 算法原理

立体匹配寻找左右图像中的同名位置。SGM 利用像素匹配代价和相邻视差的平滑约束，通过多个方向的路径聚合估计视差 [2]。本阶段固定使用 OpenCV `StereoSGBM`，不自行实现动态规划。

OpenCV 的 SGBM 是 SGM 的工程变体，不能称为对 [2] 中互信息代价等全部细节的完整复现 [5]。

### 6.2 最少配置

输入为同尺寸的校正后 `uint8` 灰度图。以下为本项目启动参数，未经设备实验验证：

| 参数 | 初始设置 | 含义或约束 |
| --- | --- | --- |
| `minDisparity` | `0` | 当前采用零视差校正与常规前向水平布局 |
| `numDisparities` | `128` | 正整数且为 16 的倍数，需适合图像宽度和最近目标 |
| `blockSize` | `5` | 正奇数，匹配窗口边长 |
| `P1` | `8 * blockSize**2` | 灰度输入时的小幅视差变化惩罚 |
| `P2` | `32 * blockSize**2` | 灰度输入时的大幅变化惩罚，需大于 `P1` |
| `mode` | `cv2.STEREO_SGBM_MODE_SGBM` | 固定使用基本模式 |

`P1,P2` 是匹配器惩罚参数，与校正投影矩阵的同名变量不同；代码中建议分别命名为 `sgbm_p1,sgbm_p2` 和 `proj_left,proj_right`。其余匹配器选项先使用所安装版本默认值，并将实际参数一起存档 [5]。

在主点对齐后，近似关系为 $d=fB/Z$。据最近目标距离估算最大视差，给搜索范围留余量；若目标视差超出范围，应调整配置后重新计算，而非把错误结果解释为准确深度 [7]。

### 6.3 视差格式

```python
matcher = cv2.StereoSGBM_create(...)
disparity_raw = matcher.compute(gray_left, gray_right)
disparity_px = disparity_raw.astype(np.float32) / 16.0
```

OpenCV SGBM 输出带 16 倍尺度的定点视差，重投影前必须转换为真实像素单位 [4,8]。保存 `disparity.npy` 的浮点数组，另存可视化 PNG。归一化后的显示图不参与深度计算。

## 7. 深度与点云重建

### 7.1 原理

在水平校正且主点对齐的模型中：

$$
Z=\frac{fB}{d}.
$$

$f$ 是**校正后**的像素焦距，$B$ 为基线；$B$ 用米则 $Z$ 用米。视差越小，对应深度越大，小视差误差在远处会造成更大的深度误差。

实际统一调用 `reprojectImageTo3D(disparity_px,Q)`，得到校正后左相机坐标系下的三维点 [4]。其 $X$ 向右、$Y$ 向下、$Z$ 向前；`xyz[...,2]` 是轴向深度，不是到光心的直线距离。

### 7.2 最低限度的无效值处理

使用 `getValidDisparityROI(roi1,roi2,minDisparity,numDisparities,blockSize)` 得到几何搜索有效区域 [4]。构建掩码时同时要求：

- 像素位于上述区域；区域为空则报配置错误。
- 原始视差大于 `(minDisparity - 1) * 16`，且当前零视差配置下 `disparity_px > 0`。
- 重投影后的三维坐标均有限，且 `Z > 0`。

这是基本有效性检查，不能保证所有保留点均正确。无需另建置信度网络，也无需加入独立的左右匹配或额外后处理模块。

输出 `depth.npy`（`float32` 米）、`valid_mask.npy`（布尔值）和 `valid_mask.png`（0/255）。无效深度写为 `NaN`；`Q` 无法正确重建或无有效点时明确报错。视差、深度和掩码均保留原图尺寸。

将有效三维点及对应左校正图颜色写为 ASCII PLY。注意 OpenCV 图像颜色顺序为 BGR，PLY 颜色字段使用 RGB；点云坐标单位注明为米。直接写文件即可，不要求引入点云处理库。

## 8. 最小成果评估体系

评估由三部分组成：参数拟合质量、校正质量、真实距离重建质量。标定拟合使用的数据与独立验证图像分别记录。

### 8.1 标定重投影误差

用 `projectPoints` 将棋盘格三维点按各帧姿态投影回原图，和检测角点比较。自行汇总的像素 RMSE 定义为：

$$
RMSE_{reproj}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}\|m_i-\hat m_i\|_2^2}.
$$

左右相机分别报告，并保留每帧结果与 OpenCV 返回的 RMS。$N$ 为角点数量，不是图像数量。重投影误差小不单独证明测距尺度正确。

### 8.2 校正后的垂直残差

在未参与标定的棋盘格图像对中检测对应角点，用 `undistortPoints` 配合各侧校正旋转及投影矩阵转换到校正像素坐标，计算：

$$
e_{v,i}=|v_{L,i}-v_{R,i}|.
$$

报告均值和 P95，单位像素，并保存水平辅助线图。验证图像的左右角点仍需保持身份一致；不能直接用原始图像角点评价校正结果。

### 8.3 独立距离测试

每个有纹理平面样本记录已知轴向距离 `reference_z_m`、测量方式和左校正图上的固定 ROI。参考距离应相对于**校正后左相机光心及其前向轴**；必须说明物理测量如何换算到这个坐标系，不能直接把到相机外壳的距离当作参考值。

对 ROI 内通过掩码的深度取中位数 $\hat Z_i$，与独立参考 $Z_i$ 比较。若平面明显倾斜，应调整采集姿态，不能用同一常数参考评价整个平面。

$$
MAE_Z=\frac{1}{M}\sum_i|\hat Z_i-Z_i|,
\qquad
RMSE_Z=\sqrt{\frac{1}{M}\sum_i(\hat Z_i-Z_i)^2},
$$

$$
\text{相对误差均值}=\frac{100\%}{M}\sum_i\frac{|\hat Z_i-Z_i|}{Z_i}.
$$

$M$ 为有有效预测且有独立参考的测试图像对数量。按距离段列出误差、有效像素比例和重复采集的深度波动。无有效点的样本标记失败，不能用零填充后计算均值；另报总样本数、失败数和实际参与指标的数量。没有独立参考时，距离精度项标记“未评估”。

### 8.4 覆盖率和运行时间

- **有效像素比例**：全图有效点数除以固定的几何搜索有效区域像素数；距离测试还需报告 ROI 内有效点数除以完整 ROI 像素数。ROI 必须位于几何有效区域内，不能随结果移动以避开失败区域。
- **耗时**：至少记录每对图像的校正、SGBM、重投影耗时及三者总和，报告平均值和 P95，注明分辨率、视差范围和运行设备。磁盘保存、显示与采集耗时单独记录，不混称完整实时帧率。
- **结果检查**：确认视差和深度随目标距离变化方向合理，PLY 可读取且点数等于掩码有效点数。

`metrics.json` 保存汇总指标、参数、版本与计时范围；`samples.csv` 保存逐样本结果、失败原因、参考距离及 ROI。固定 ROI、距离参考与所有参数应能用于同一输入的重复评估。

## 9. Codex 实现约定

### 9.1 数据和接口

以下为建议接口，名称可随仓库结构调整，但输入输出含义保持一致：

| 接口 | 输入 | 输出 |
| --- | --- | --- |
| `capture_pairs(config, output_dir)` | 相机与分辨率配置 | PNG 图像对、`pairs.csv` |
| `calibrate_stereo(pair_list, board_config)` | 有效棋盘格图像对 | 标定参数、拟合误差及有效配对列表 |
| `build_rectification(calibration)` | 标定参数 | 映射表、`R1,R2,P1,P2,Q` 和 ROI |
| `reconstruct_pair(left, right, calibration, rectification, sgbm_config)` | 原始 BGR 图像对及参数 | 校正图、像素视差、深度、掩码、三维点 |
| `evaluate(results, references, config)` | 结果、独立参考及验证配置 | `metrics.json`、`samples.csv` |

配置至少包含：相机身份、尺寸、棋盘格内角点行列数、`square_size_m`、SGBM 参数和数据路径。参考距离与 ROI 单独记录，不硬编码进算法。所有数值结果使用米、像素和毫秒三种明确单位；配置中缺少实际硬件信息时明确要求填写，不虚构设备型号或尺寸。

### 9.2 实现与验收顺序

1. **采集可用**：两相机可打开，完整图像对可保存，失败可追踪。
2. **标定可复用**：可从保存图像生成参数文件；重新加载后可校正独立图像对。
3. **重建可输出**：生成数值视差、深度、掩码和彩色 PLY，尺度与坐标明确。
4. **评估可复现**：独立输入与参考生成指标文件，失败样本和未评估项均明确标记。

上述检查通过表示基本流程跑通。是否达到使用精度，需要另外填写允许的重投影误差、垂直残差、测距误差和覆盖率阈值；未设置阈值时只报告实测值，不自动宣布“精度达标”。先支持保存图像的离线处理，再复用同一重建函数处理相机采集结果。

本阶段不实现学习式匹配、置信度或不确定性估计、在线自适应、额外点云优化、多帧融合、网格重建及 SLAM。不要求人工重写 OpenCV 已提供的求解器。

## 10. 参考依据

文献解释基本原理，OpenCV 官方文档约束实际接口。实现时固定并记录使用的 OpenCV 版本；下列官方链接可能随站点更新。

1. Zhengyou Zhang. *A Flexible New Technique for Camera Calibration.* IEEE TPAMI, 2000. [DOI](https://doi.org/10.1109/34.888718)。项目文献：`doc/literature/calibration/A_flexible_new_technique_for_camera_calibration.pdf`。
2. Heiko Hirschmüller. *Stereo Processing by Semiglobal Matching and Mutual Information.* IEEE TPAMI, 2008. [DOI](https://doi.org/10.1109/TPAMI.2007.1166)。项目文献：`doc/literature/calibration/SvVYHP-Stereo_Processing_by_Semiglobal_Matching_and_Mutual_Information.pdf`。
3. OpenCV. [Camera Calibration（Python 教程）](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)。
4. OpenCV. [Camera Calibration and 3D Reconstruction](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)。
5. OpenCV. [StereoSGBM](https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html)。
6. OpenCV. [VideoCapture](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)。
7. OpenCV. [Depth Map from Stereo Images](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html)。
8. OpenCV. [StereoMatcher](https://docs.opencv.org/4.x/d2/d6e/classcv_1_1StereoMatcher.html)。
