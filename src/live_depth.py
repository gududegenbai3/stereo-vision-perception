"""Fixed-calibration live stereo on an identical grid for all predictors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from multiprocessing import get_context
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from threading import Event
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .camera_io import calibrated_inputs, digest, sgbm_predict

METHODS: tuple[str, ...] = ("sgbm", "banet", "igev")
LABELS: dict[str, str] = {"sgbm": "SGBM", "banet": "BANet", "igev": "IGEV"}
BgrImage = NDArray[np.uint8]  # H x W x 3 BGR, 0–255
ScalarMap = NDArray[np.float32]  # H x W, pixels or metres, invalid NaN


@dataclass(frozen=True)
class LiveGeometry:
    """Native float32 maps in pixels, and metre Q on the matching grid."""

    maps: dict[str, NDArray[Any]]
    native_wh: tuple[int, int]
    q: NDArray[np.float64]
    scale: float


@dataclass(frozen=True)
class LiveResult:
    """Paired H x W float32 disparity (px) and axial depth (m), invalid NaN."""

    disparity_px: ScalarMap
    depth_m: ScalarMap
    inference_ms: float


def scaled_q(q: NDArray[np.float64], scale: float) -> NDArray[np.float64]:
    """Convert Q to a uniformly resized grid, including pixel-centre offsets."""
    if q.shape != (4, 4) or not np.isfinite(q).all() or scale <= 0:
        raise ValueError("Invalid Q or image scale")
    inverse_resize: NDArray[np.float64] = np.diag([1 / scale, 1 / scale, 1 / scale, 1.0])
    inverse_resize[:2, 3] = 0.5 / scale - 0.5
    return q @ inverse_resize


def load_geometry(selector_path: Path, capture: dict[str, Any], scale: float) -> LiveGeometry:
    """Verify the exact v2 files against a selector and the configured camera mode."""
    selector: dict[str, Any] = json.loads(selector_path.read_text(encoding="utf-8"))
    if selector["status"] != "display_calibration_independently_checked_geometry_only":
        raise ValueError("标定未完成独立校正检查")
    width, height = selector["image_size_wh"]
    if (
        selector["rig_id"] != capture["rig_id"]
        or capture["physical_order"] != "left_half_is_left_camera"
        or capture["rectified"] is not False
        or capture["frame_wh"] != [2 * width, height]
    ):
        raise ValueError("相机、左右顺序或分辨率与标定不匹配")
    verified_paths: dict[str, Path] = {}
    for name in ("calibration", "rectification"):
        path: Path = selector_path.parent / selector[f"{name}_npz"]
        if digest(path) != selector[f"{name}_sha256"]:
            raise ValueError(f"标定文件校验失败：{name}")
        verified_paths[name] = path
    with np.load(verified_paths["rectification"], allow_pickle=False) as data:
        maps: dict[str, NDArray[Any]] = {
            name: data[name] for name in ("image_size", "map1_x", "map1_y", "map2_x", "map2_y")
        }
        if tuple(maps["image_size"]) != (width, height) or any(
            values.shape != (height, width) or values.dtype != np.float32 or not np.isfinite(values).all()
            for key, values in maps.items() if key != "image_size"
        ):
            raise ValueError("校正映射格式或尺寸有误")
        q: NDArray[np.float64] = scaled_q(np.asarray(data["Q"], dtype=np.float64), scale)
    return LiveGeometry(maps, (width, height), q, scale)


def make_depth(
    disparity: ScalarMap, inputs: dict[str, BgrImage], q: NDArray[np.float64], maximum: int,
) -> tuple[ScalarMap, ScalarMap]:
    """Keep valid left and corresponding right support; BGR is unused here."""
    left_support: NDArray[np.uint8] = inputs["left_support"]
    right_support: NDArray[np.uint8] = inputs["right_support"]
    if disparity.dtype != np.float32 or disparity.shape != left_support.shape:
        raise ValueError("预测视差尺寸或类型有误")
    width: int = disparity.shape[1]
    valid: NDArray[np.bool_] = (
        (left_support != 0) & np.isfinite(disparity) & (disparity > 0) & (disparity < maximum)
    )
    right_x: ScalarMap = np.arange(width, dtype=np.float32) - np.where(valid, disparity, 0)
    valid &= (right_x >= 0) & (right_x <= width - 1)
    lo: NDArray[np.int32] = np.clip(np.floor(right_x).astype(np.int32), 0, width - 1)
    hi: NDArray[np.int32] = np.clip(np.ceil(right_x).astype(np.int32), 0, width - 1)
    valid &= np.take_along_axis(right_support, lo, axis=1) != 0
    valid &= np.take_along_axis(right_support, hi, axis=1) != 0
    values: ScalarMap = np.where(valid, disparity, np.nan)
    depth: ScalarMap = cv2.reprojectImageTo3D(values, q)[..., 2].copy()
    valid &= np.isfinite(depth) & (depth > 0)
    depth[~valid] = np.nan
    values[~valid] = np.nan
    return values, depth


def _gpu_worker(connection: Connection, models_path: Path, runtime: Path, iterations: int) -> None:
    """Load models lazily on one CUDA process; requests are serialized."""
    models: dict[str, Callable[[dict[str, BgrImage]], ScalarMap]] = {}
    try:
        while True:
            method, inputs = connection.recv()
            try:
                if method not in models:
                    models[method] = _load_gpu_predictor(method, models_path, runtime, iterations)
                started: float = perf_counter()
                result: ScalarMap = models[method](inputs)
                connection.send((result, (perf_counter() - started) * 1000, None))
            except Exception as error:
                connection.send((None, 0.0, f"{type(error).__name__}: {error}"))
    except (EOFError, OSError):
        pass
    finally:
        connection.close()


def _load_gpu_predictor(
    method: str, config: Path, runtime: Path, iterations: int,
) -> Callable[[dict[str, BgrImage]], ScalarMap]:
    from .pretrained_stereo import environment_dll_handle, load_banet, load_igev, predict_banet_rgb, predict_igev

    dll_handle: object | None = environment_dll_handle()
    import torch

    raw: dict[str, Any] = json.loads(config.read_text(encoding="utf-8"))[method]
    if raw["precision"] != "float32" or raw["max_disparity"] != 192 or raw["batch_norm_policy"] != "running":
        raise ValueError("预训练配置须采用 FP32、192 px 和保存的 BN 统计")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if method == "banet":
        model, _ = load_banet(config.parent / raw["source_root"], runtime / method,
                              config.parent / raw["checkpoint"], raw["checkpoint_sha256"])
        model._svp_live_dll_handle = dll_handle
        return lambda arrays: predict_banet_rgb(model, arrays, 192)
    model, _ = load_igev(config, {**raw, "runtime_root": str(runtime / method)})
    model._svp_live_dll_handle = dll_handle
    return lambda arrays: predict_igev(model, arrays, iterations)


class LiveProcessor:
    """One background caller prepares and reconstructs input; SGBM is CPU-only."""

    def __init__(
        self, geometry: LiveGeometry, sgbm: dict[str, Any], models_path: Path,
        runtime: Path, iterations: int, stop: Event,
    ) -> None:
        self.geometry: LiveGeometry = geometry
        self.sgbm: dict[str, Any] = sgbm
        self.models_path: Path = models_path
        self.runtime: Path = runtime
        self.iterations: int = iterations
        self.stop: Event = stop
        self.support_masks: dict[str, NDArray[np.uint8]] | None = None
        self.connection: Connection | None = None
        self.worker: BaseProcess | None = None

    def prepare(self, raw: BgrImage) -> dict[str, BgrImage]:
        """H x 2W x 3 native BGR to H' x W' x 3 rectified BGR with 2-D masks."""
        width, height = self.geometry.native_wh
        if raw.shape != (height, width * 2, 3) or raw.dtype != np.uint8:
            raise ValueError("无效的双目拼接帧")
        inputs: dict[str, BgrImage] = calibrated_inputs(
            raw[:, :width], raw[:, width:], self.geometry.maps, self.geometry.scale,
            support_masks=self.support_masks,
        )
        if self.support_masks is None:
            self.support_masks = {key: inputs[key] for key in ("left_support", "right_support")}
        return inputs

    def reconstruct(self, inputs: dict[str, BgrImage], method: str) -> LiveResult:
        """Compute one predictor on the shared inputs, with depth in metres."""
        arrays: dict[str, BgrImage] = {
            side: cv2.cvtColor(inputs[side], cv2.COLOR_BGR2RGB) for side in ("left", "right")
        }
        if method == "sgbm":
            started: float = perf_counter()
            predicted: ScalarMap = sgbm_predict(arrays, self.sgbm, 192)
            inference_ms: float = (perf_counter() - started) * 1000
        else:
            predicted, inference_ms = self._predict_gpu(method, arrays)
        disparity, depth = make_depth(predicted, inputs, self.geometry.q, 192)
        return LiveResult(disparity, depth, inference_ms)

    def _predict_gpu(self, method: str, arrays: dict[str, BgrImage]) -> tuple[ScalarMap, float]:
        if self.connection is None:
            context = get_context("spawn")
            parent, child = context.Pipe()
            worker: BaseProcess = context.Process(
                target=_gpu_worker, args=(child, self.models_path, self.runtime, self.iterations), daemon=True,
            )
            try:
                worker.start()
            except BaseException:
                parent.close()
                raise
            finally:
                child.close()
            self.worker, self.connection = worker, parent
        connection: Connection = self.connection
        try:
            connection.send((method, arrays))
            deadline: float = perf_counter() + 90
            while not self.stop.is_set() and perf_counter() < deadline:
                if connection.poll(0.1):
                    values, elapsed, error = connection.recv()
                    if error:
                        raise ValueError(error)
                    return values, elapsed
        except (OSError, EOFError) as error:
            self.close()
            raise RuntimeError("CUDA 推理进程已退出") from error
        self.close()
        raise RuntimeError("CUDA 推理已停止或超时")

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.worker is not None:
            self.worker.join(0.3)
            if self.worker.is_alive():
                self.worker.terminate()
                self.worker.join()
            self.worker.close()
            self.worker = None
