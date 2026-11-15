"""双目标定：棋盘格与圆点阵列的角点检测、内外参求解与参数持久化。"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from .model import (
    AsymmetricCircleGridSpec,
    BgrImage,
    ChessboardSpec,
    ImageSize,
    TargetSpec,
)

# (N, 2)，角点像素坐标 (x, y)，按标靶规范顺序排列。
CornerArray: TypeAlias = NDArray[np.float32]
# (N, 3)，标靶对象点，单位米，Z 恒为 0。
ObjectPointArray: TypeAlias = NDArray[np.float32]
# (H, W)，uint8 单通道灰度图。
GrayImage: TypeAlias = NDArray[np.uint8]
# (3, 3) 内参矩阵或旋转矩阵。
Matrix3: TypeAlias = NDArray[np.float64]
# (1, 5) 畸变系数 k1、k2、p1、p2、k3。
DistortionCoefficients: TypeAlias = NDArray[np.float64]
# (3,) 平移向量，单位米。
Vector3: TypeAlias = NDArray[np.float64]

DetectionMethod: TypeAlias = Literal["sb", "classic", "blob"]
ScaleSource: TypeAlias = Literal["target_size", "baseline_override"]

_CALIBRATION_SCHEMA: str = "2.0"
# findChessboardCornersSB 使用全搜索与亚像素精度，适合清晰图像。
_SB_FLAGS: int = (
    cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
)
# 经典检测器对模糊图像更稳健，随后做亚像素精化。
_CLASSIC_FLAGS: int = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
_SUBPIX_CRITERIA: tuple[int, int, float] = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,
    60,
    1e-5,
)
_SOLVE_CRITERIA: tuple[int, int, float] = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,
    200,
    1e-9,
)
_JPEG_EOI: NDArray[np.uint8] = np.array([0xFF, 0xD9], dtype=np.uint8)


@dataclass(frozen=True, eq=False)
class BoardDetection:
    """单张图像上的标靶点及其检测方式。"""

    corners_px: CornerArray
    method: DetectionMethod


@dataclass(frozen=True, eq=False)
class PairDetection:
    """同一标靶姿态的左右图像点与对应对象点。

    object_points 与左右 corners_px 一一对应，单位米。target_name 记录
    该对图像使用的标靶类型。
    """

    name: str
    target_name: str
    object_points: ObjectPointArray
    left: BoardDetection
    right: BoardDetection


@dataclass(frozen=True, eq=False)
class CameraIntrinsics:
    """单目内参与逐视图重投影误差。"""

    matrix: Matrix3
    distortion: DistortionCoefficients
    rms_px: float
    per_view_rmse_px: tuple[float, ...]


@dataclass(frozen=True, eq=False)
class PairReprojectionError:
    """一对图像的左右单目重投影均方根误差。"""

    name: str
    target_name: str
    left_rmse_px: float
    right_rmse_px: float


@dataclass(frozen=True, eq=False)
class StereoCalibration:
    """双目标定结果，平移与基线单位为米。

    rotation 与 translation_m 把左相机坐标系变换到右相机坐标系。
    targets 记录参与标定的标靶类型，pair_targets 与逐对误差顺序一致。
    """

    left: CameraIntrinsics
    right: CameraIntrinsics
    rotation: Matrix3
    translation_m: Vector3
    essential: Matrix3
    fundamental: Matrix3
    rms_px: float
    image_size: ImageSize
    targets: tuple[str, ...]
    baseline_m: float
    scale_source: ScaleSource
    fix_k3: bool
    pair_errors: tuple[PairReprojectionError, ...]

    @property
    def pair_names(self) -> tuple[str, ...]:
        """参与标定的图像对名称。"""
        return tuple(error.name for error in self.pair_errors)

    @property
    def pair_targets(self) -> tuple[str, ...]:
        """参与标定的图像对所用标靶类型。"""
        return tuple(error.target_name for error in self.pair_errors)


def read_bgr_image(path: Path) -> BgrImage:
    """读取 BGR 图像 (H, W, 3) uint8。

    部分采集文件缺少 JPEG 结束标记，会在内存中补齐后重试，像素数据
    保持不变。文件不存在抛出 FileNotFoundError，无法解码抛出 ValueError。
    """
    if not path.is_file():
        raise FileNotFoundError(f"图像不存在：{path}")
    data: NDArray[np.uint8] = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"图像文件为空：{path}")
    image: BgrImage | None = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None and not _has_jpeg_eoi(data):
        image = cv2.imdecode(np.concatenate((data, _JPEG_EOI)), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法解码图像：{path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"图像不是三通道 BGR：{path}")
    return image


def split_side_by_side(image: BgrImage) -> tuple[BgrImage, BgrImage]:
    """把左右拼图沿中线拆成左右两幅图。

    拼图宽度必须为偶数且不小于两倍最小宽度，否则抛出 ValueError。
    """
    height: int
    width: int
    height, width = image.shape[:2]
    if width % 2 != 0:
        raise ValueError(f"左右拼图宽度必须为偶数，实际为 {width}")
    half: int = width // 2
    if half < 16 or height < 16:
        raise ValueError(f"拼图拆分后尺寸过小：{half}x{height}")
    return image[:, :half].copy(), image[:, half:].copy()


def to_gray(image: BgrImage) -> GrayImage:
    """把 BGR 图像转为单通道灰度图。"""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("转灰度需要 uint8 BGR 图像，形状 (H, W, 3)")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def detect_target_pair(
    name: str,
    left_gray: GrayImage,
    right_gray: GrayImage,
    target: TargetSpec,
) -> PairDetection | None:
    """检测一对图像上的同一标靶，返回带对象点的观测。

    棋盘格优先两侧同时使用 SB 检测器，其次两侧一起退回经典检测器加
    亚像素精化；混用两种检测器会产生不一致的角点顺序。圆点阵列用
    连通域圆点中心与行列聚类排序。任一侧失败时返回 None。
    """
    if left_gray.ndim != 2 or left_gray.dtype != np.uint8:
        raise ValueError("角点检测需要 (H, W) uint8 灰度图")
    if right_gray.ndim != 2 or right_gray.dtype != np.uint8:
        raise ValueError("角点检测需要 (H, W) uint8 灰度图")
    pair: tuple[BoardDetection, BoardDetection] | None
    target_name: str
    if isinstance(target, ChessboardSpec):
        pair = _detect_chessboard_pair(left_gray, right_gray, target)
        target_name = "chessboard"
    elif isinstance(target, AsymmetricCircleGridSpec):
        left_detection: BoardDetection | None = _detect_circle_grid(left_gray, target)
        right_detection: BoardDetection | None = _detect_circle_grid(right_gray, target)
        pair = (
            (left_detection, right_detection)
            if left_detection is not None and right_detection is not None
            else None
        )
        target_name = "asymmetric_circle_grid"
    else:
        raise TypeError(f"不支持的标靶类型：{type(target).__name__}")
    if pair is None:
        return None
    object_points: ObjectPointArray = target.object_points()
    if len(object_points) != len(pair[0].corners_px):
        raise ValueError("对象点数量与检测到的图像点数量不一致")
    return PairDetection(name, target_name, object_points, pair[0], pair[1])


def calibrate_camera(
    object_points: Sequence[ObjectPointArray],
    image_points: Sequence[CornerArray],
    image_size: ImageSize,
    *,
    fix_k3: bool,
) -> CameraIntrinsics:
    """由同一标靶的多视图点求解单目内参与畸变。

    object_points 为米制 (N, 3) 数组，image_points 为对应的 (N, 2)
    像素坐标。数组数量、尺寸或视图数不合法时抛出 ValueError。
    """
    _validate_observations(object_points, image_points, image_size)
    flags: int = cv2.CALIB_FIX_K3 if fix_k3 else 0
    result: tuple[
        float,
        NDArray[np.float64],
        NDArray[np.float64],
        list[NDArray[np.float64]],
        list[NDArray[np.float64]],
    ] = cv2.calibrateCamera(
        list(object_points),
        list(image_points),
        image_size,
        None,
        None,
        flags=flags,
        criteria=_SOLVE_CRITERIA,
    )
    rms_px, matrix, distortion, rotations, translations = result
    per_view: list[float] = []
    for index, corners in enumerate(image_points):
        projected: NDArray[np.float32]
        projected, _ = cv2.projectPoints(
            object_points[index], rotations[index], translations[index], matrix, distortion
        )
        residual: float = float(
            cv2.norm(corners, projected.reshape(-1, 2).astype(np.float32), cv2.NORM_L2)
        )
        per_view.append(residual / float(np.sqrt(corners.shape[0])))
    return CameraIntrinsics(
        matrix=matrix,
        distortion=distortion,
        rms_px=float(rms_px),
        per_view_rmse_px=tuple(per_view),
    )


def calibrate_stereo(
    detections: Sequence[PairDetection],
    image_size: ImageSize,
    *,
    baseline_m: float | None = None,
    fix_k3: bool = True,
) -> StereoCalibration:
    """由多对标靶观测求解双目内外参。

    先分别标定两目内参，再以固定内参求左右相对姿态。对象点单位为米，
    因此 baseline_m 为 None 时平移直接以米给出；给定时按该基线重设
    平移尺度。少于 3 对、点数不符或求解非有限值时抛出 ValueError。
    """
    if len(detections) < 3:
        raise ValueError("双目外参求解至少需要 3 对有效图像")
    if baseline_m is not None and (not np.isfinite(baseline_m) or baseline_m <= 0):
        raise ValueError("基线必须为有限正数")
    object_points: list[ObjectPointArray] = [detection.object_points for detection in detections]
    left_points: list[CornerArray] = [detection.left.corners_px for detection in detections]
    right_points: list[CornerArray] = [detection.right.corners_px for detection in detections]

    left: CameraIntrinsics = calibrate_camera(object_points, left_points, image_size, fix_k3=fix_k3)
    right: CameraIntrinsics = calibrate_camera(
        object_points, right_points, image_size, fix_k3=fix_k3
    )

    stereo: tuple[
        float,
        Matrix3,
        DistortionCoefficients,
        Matrix3,
        DistortionCoefficients,
        Matrix3,
        Vector3,
        Matrix3,
        Matrix3,
    ] = cv2.stereoCalibrate(
        object_points,
        left_points,
        right_points,
        left.matrix,
        left.distortion,
        right.matrix,
        right.distortion,
        image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=_SOLVE_CRITERIA,
    )
    rms_px, _, _, _, _, rotation, translation, essential, fundamental = stereo

    if baseline_m is None:
        baseline: float = float(np.linalg.norm(translation))
        scale_source: ScaleSource = "target_size"
    else:
        factor: float = baseline_m / float(np.linalg.norm(translation))
        translation = translation * factor
        essential = essential * factor
        baseline = float(baseline_m)
        scale_source = "baseline_override"

    pair_errors: tuple[PairReprojectionError, ...] = tuple(
        PairReprojectionError(
            detection.name,
            detection.target_name,
            left_rmse,
            right_rmse,
        )
        for detection, left_rmse, right_rmse in zip(
            detections, left.per_view_rmse_px, right.per_view_rmse_px, strict=True
        )
    )
    targets: tuple[str, ...] = tuple(dict.fromkeys(detection.target_name for detection in detections))
    calibration: StereoCalibration = StereoCalibration(
        left=left,
        right=right,
        rotation=rotation,
        translation_m=translation.reshape(3).astype(np.float64),
        essential=essential,
        fundamental=fundamental,
        rms_px=float(rms_px),
        image_size=(int(image_size[0]), int(image_size[1])),
        targets=targets,
        baseline_m=baseline,
        scale_source=scale_source,
        fix_k3=fix_k3,
        pair_errors=pair_errors,
    )
    _validate_calibration(calibration)
    return calibration


def save_calibration(path: Path, calibration: StereoCalibration) -> None:
    """把标定结果写入 npz，单位为米，图像尺寸顺序为 (width, height)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    errors: NDArray[np.float64] = np.array(
        [[error.left_rmse_px, error.right_rmse_px] for error in calibration.pair_errors],
        dtype=np.float64,
    )
    np.savez_compressed(
        path,
        schema=np.array(_CALIBRATION_SCHEMA),
        created_at=np.array(datetime.now().isoformat(timespec="seconds")),
        K1=calibration.left.matrix,
        D1=calibration.left.distortion,
        K2=calibration.right.matrix,
        D2=calibration.right.distortion,
        R=calibration.rotation,
        T=calibration.translation_m,
        E=calibration.essential,
        F=calibration.fundamental,
        image_size_wh=np.array(calibration.image_size, dtype=np.int64),
        targets=np.array(list(calibration.targets)),
        baseline_m=np.array(calibration.baseline_m),
        scale_source=np.array(calibration.scale_source),
        fix_k3=np.array(calibration.fix_k3),
        rms_left_px=np.array(calibration.left.rms_px),
        rms_right_px=np.array(calibration.right.rms_px),
        rms_stereo_px=np.array(calibration.rms_px),
        pair_names=np.array(list(calibration.pair_names)),
        pair_targets=np.array(list(calibration.pair_targets)),
        pair_errors_px=errors,
    )


def load_calibration(path: Path) -> StereoCalibration:
    """读取 npz 标定结果，缺少字段或数值无效时抛出 ValueError。"""
    if not path.is_file():
        raise FileNotFoundError(f"标定文件不存在：{path}")
    try:
        with np.load(path, allow_pickle=False) as data:
            image_size: tuple[int, int] = (
                int(data["image_size_wh"][0]),
                int(data["image_size_wh"][1]),
            )
            names: tuple[str, ...] = tuple(str(name) for name in data["pair_names"])
            pair_targets: tuple[str, ...] = tuple(str(name) for name in data["pair_targets"])
            errors: NDArray[np.float64] = np.asarray(data["pair_errors_px"], dtype=np.float64)
            calibration: StereoCalibration = StereoCalibration(
                left=CameraIntrinsics(
                    matrix=np.asarray(data["K1"], dtype=np.float64),
                    distortion=np.asarray(data["D1"], dtype=np.float64),
                    rms_px=float(np.asarray(data["rms_left_px"])),
                    per_view_rmse_px=tuple(float(value) for value in errors[:, 0]),
                ),
                right=CameraIntrinsics(
                    matrix=np.asarray(data["K2"], dtype=np.float64),
                    distortion=np.asarray(data["D2"], dtype=np.float64),
                    rms_px=float(np.asarray(data["rms_right_px"])),
                    per_view_rmse_px=tuple(float(value) for value in errors[:, 1]),
                ),
                rotation=np.asarray(data["R"], dtype=np.float64),
                translation_m=np.asarray(data["T"], dtype=np.float64).reshape(3),
                essential=np.asarray(data["E"], dtype=np.float64),
                fundamental=np.asarray(data["F"], dtype=np.float64),
                rms_px=float(np.asarray(data["rms_stereo_px"])),
                image_size=image_size,
                targets=tuple(str(name) for name in data["targets"]),
                baseline_m=float(np.asarray(data["baseline_m"])),
                scale_source=str(np.asarray(data["scale_source"])),  # type: ignore[arg-type]
                fix_k3=bool(np.asarray(data["fix_k3"])),
                pair_errors=tuple(
                    PairReprojectionError(name, target, float(row[0]), float(row[1]))
                    for name, target, row in zip(names, pair_targets, errors, strict=True)
                ),
            )
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"{path}：标定文件字段缺失或格式不符（{error}）") from error
    _validate_calibration(calibration)
    return calibration


def _detect_chessboard_pair(
    left_gray: GrayImage,
    right_gray: GrayImage,
    board: ChessboardSpec,
) -> tuple[BoardDetection, BoardDetection] | None:
    """检测棋盘格，按对统一检测器以保证角点顺序一致。"""
    left_sb: BoardDetection | None = _detect_chessboard_sb(left_gray, board)
    right_sb: BoardDetection | None = _detect_chessboard_sb(right_gray, board)
    if left_sb is not None and right_sb is not None:
        return left_sb, right_sb
    left_classic: BoardDetection | None = _detect_chessboard_classic(left_gray, board)
    right_classic: BoardDetection | None = _detect_chessboard_classic(right_gray, board)
    if left_classic is None or right_classic is None:
        return None
    return left_classic, right_classic


def _detect_chessboard_sb(
    gray: GrayImage, board: ChessboardSpec
) -> BoardDetection | None:
    """用 SB 检测器检测棋盘格内角点，失败返回 None。"""
    pattern: tuple[int, int] = (board.cols, board.rows)
    found: bool
    corners: NDArray[np.float32]
    found, corners = cv2.findChessboardCornersSB(gray, pattern, _SB_FLAGS)
    if not found or corners.size != board.corner_count * 2:
        return None
    return BoardDetection(corners.reshape(-1, 2).astype(np.float32), "sb")


def _detect_chessboard_classic(
    gray: GrayImage, board: ChessboardSpec
) -> BoardDetection | None:
    """用经典检测器加亚像素精化检测棋盘格内角点，失败返回 None。"""
    pattern: tuple[int, int] = (board.cols, board.rows)
    found: bool
    corners: NDArray[np.float32]
    found, corners = cv2.findChessboardCorners(gray, pattern, _CLASSIC_FLAGS)
    if not found or corners.size != board.corner_count * 2:
        return None
    refined: NDArray[np.float32] = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1), _SUBPIX_CRITERIA
    )
    return BoardDetection(refined.reshape(-1, 2).astype(np.float32), "classic")


def _detect_circle_grid(
    gray: GrayImage, spec: AsymmetricCircleGridSpec
) -> BoardDetection | None:
    """检测非对称圆点阵列并按规范顺序返回圆点中心。

    OpenCV 5.0 的 findCirclesGrid 在非对称模式下对标准合成板也会失败，
    因此这里用连通域圆点检测配合行列聚类自行排序：投影到点阵主轴后
    按垂直方向排序，再按已知的长短行点数切成 rows 行。点数不符、行长
    不成交替规律或某行离散过大时返回 None。
    """
    keypoints = _blob_detector().detect(gray)
    if not keypoints:
        return None
    sizes: NDArray[np.float64] = np.array([keypoint.size for keypoint in keypoints])
    median: float = float(np.median(sizes))
    keep: NDArray[np.bool_] = (sizes > 0.6 * median) & (sizes < 1.6 * median)
    points: NDArray[np.float64] = np.array(
        [keypoints[index].pt for index in np.flatnonzero(keep)], dtype=np.float64
    )
    if points.shape[0] != spec.point_count:
        return None
    rows: list[NDArray[np.float64]] | None = _order_circle_rows(points, spec)
    if rows is None:
        return None
    corners: CornerArray = np.concatenate(rows).astype(np.float32)
    return BoardDetection(corners, "blob")


def _blob_detector() -> cv2.SimpleBlobDetector:
    """构建用于检测黑色圆点的连通域检测器。"""
    parameters = cv2.SimpleBlobDetector_Params()
    parameters.minThreshold = 10
    parameters.maxThreshold = 200
    parameters.filterByColor = True
    parameters.blobColor = 0
    parameters.filterByArea = True
    parameters.minArea = 300
    parameters.maxArea = 6000
    parameters.filterByCircularity = True
    parameters.minCircularity = 0.6
    parameters.filterByConvexity = True
    parameters.minConvexity = 0.8
    parameters.filterByInertia = True
    parameters.minInertiaRatio = 0.5
    parameters.minDistBetweenBlobs = 20
    return cv2.SimpleBlobDetector_create(parameters)


def _order_circle_rows(
    points: NDArray[np.float64],
    spec: AsymmetricCircleGridSpec,
) -> list[NDArray[np.float64]] | None:
    """把圆点分成 rows 行，每行按从左到右排序，行按从上到下返回。

    先按点阵主轴投影，再按垂直方向排序并按已知的长短行点数切分。两种
    行序各算一次行内垂直离散，取较小者；切分后行内离散过大时返回 None。
    首行最终统一为短行，与对象点顺序一致。
    """
    centered: NDArray[np.float64] = points - points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(centered.T))
    axis: NDArray[np.float64] = vectors[:, int(np.argmax(values))]
    if axis[0] < 0:
        axis = -axis
    perpendicular: NDArray[np.float64] = np.array([-axis[1], axis[0]])
    along: NDArray[np.float64] = centered @ axis
    across: NDArray[np.float64] = centered @ perpendicular
    order: NDArray[np.int64] = np.argsort(across)
    counts: tuple[int, ...] = spec.row_counts()

    best: tuple[float, list[int], list[NDArray[np.int64]]] | None = None
    for shift in (0, 1):
        pattern: list[int] = [counts[(index + shift) % 2] for index in range(spec.rows)]
        offsets: list[int] = [0]
        for count in pattern:
            offsets.append(offsets[-1] + count)
        if offsets[-1] != len(order):
            continue
        groups: list[NDArray[np.int64]] = []
        cost: float = 0.0
        for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
            indices: NDArray[np.int64] = order[start:stop]
            spread: float = float(np.std(across[indices]))
            cost += spread * spread * len(indices)
            groups.append(indices)
        if best is None or cost < best[0]:
            best = (cost, pattern, groups)
    if best is None:
        return None
    _, pattern, groups = best

    pitch: float = float(np.ptp(across)) / float(spec.rows - 1)
    worst_spread: float = max(float(np.std(across[indices])) for indices in groups)
    if pitch <= 0 or worst_spread > 0.35 * pitch:
        return None

    rows: list[NDArray[np.float64]] = [
        points[indices[np.argsort(along[indices])]] for indices in groups
    ]
    if pattern[0] == spec.max_per_row:
        rows.reverse()
    expected: list[int] = list(spec.row_counts())
    if [len(row) for row in rows] != expected:
        return None
    return rows


def _has_jpeg_eoi(data: NDArray[np.uint8]) -> bool:
    """判断 JPEG 数据是否以 FFD9 结束。"""
    return bool(data.size >= 2 and data[-2] == 0xFF and data[-1] == 0xD9)


def _validate_observations(
    object_points: Sequence[ObjectPointArray],
    image_points: Sequence[CornerArray],
    image_size: ImageSize,
) -> None:
    if len(object_points) != len(image_points) or len(object_points) < 3:
        raise ValueError("对象点与图像点数量必须一致且不少于 3 组")
    width: int
    height: int
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("图像尺寸必须为正")
    for points, corners in zip(object_points, image_points, strict=True):
        if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(corners):
            raise ValueError("每组对象点与图像点数量必须一致")
        if corners.ndim != 2 or corners.shape[1] != 2 or corners.dtype != np.float32:
            raise ValueError("图像角点必须是 (N, 2) float32 数组")
        if not np.isfinite(points).all() or not np.isfinite(corners).all():
            raise ValueError("对象点与图像点必须是有限数值")


def _validate_calibration(calibration: StereoCalibration) -> None:
    arrays: tuple[NDArray[np.float64], ...] = (
        calibration.left.matrix,
        calibration.left.distortion,
        calibration.right.matrix,
        calibration.right.distortion,
        calibration.rotation,
        calibration.translation_m,
        calibration.essential,
        calibration.fundamental,
    )
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("标定结果包含非有限数值")
    if calibration.left.matrix.shape != (3, 3) or calibration.right.matrix.shape != (3, 3):
        raise ValueError("内参矩阵形状必须为 (3, 3)")
    if calibration.translation_m.shape != (3,):
        raise ValueError("平移向量形状必须为 (3,)")
    if calibration.baseline_m <= 0:
        raise ValueError("基线必须为正")
