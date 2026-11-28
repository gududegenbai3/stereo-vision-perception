"""Show three synchronized live stereo predictors or benchmark a replay."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime
import json
import os
from pathlib import Path
from shutil import which
from statistics import median
from threading import Event, Thread
from time import perf_counter
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageTk

from .camera_io import write_json
from .config import load_offline_config
from .datasets import load_middlebury_scene
from .live_depth import BgrImage, LABELS, LiveGeometry, LiveProcessor, LiveResult, METHODS, ScalarMap, load_geometry, scaled_q
from .live_source import Frame, LatestFrame, capture_ffmpeg, receive_camera, replay_video


@dataclass(frozen=True)
class Settings:
    capture: dict[str, Any]
    capture_config: Path
    replay_file: Path | None
    size: tuple[int, int]
    fps: int
    selector: Path
    models: Path
    scale: float
    sgbm: dict[str, Any]
    iterations: int
    near_m: float
    far_m: float


def load_settings(config: Path, iterations_override: int | None, *, public_replay: bool = False) -> Settings:
    raw: dict[str, Any] = json.loads(config.read_text(encoding="utf-8"))
    capture_name: str = "capture_ui.example.json" if public_replay else raw["capture_config"]
    capture_config: Path = config.parent / capture_name
    capture: dict[str, Any] = json.loads(capture_config.read_text(encoding="utf-8"))
    replay_file: str | None = raw.get("replay_file")
    width, height = capture["frame_wh"]
    fps: int = capture["requested_fps"]
    scale: float = raw["image_scale"]
    iterations: int = raw["igev_iterations"] if iterations_override is None else iterations_override
    near, far = raw["depth_display_m"]
    if any(type(value) is not int or value < 1 for value in (width, height, fps)) or width % 2:
        raise ValueError("无效的摄像头拼接尺寸或帧率")
    if scale != 0.5 or iterations not in (8, 32) or not 0 < near < far:
        raise ValueError("试验版需使用半尺寸、IGEV 8/32 次迭代和合法的深度显示范围")
    return Settings(
        capture, capture_config, config.parent / replay_file if replay_file else None, (width, height), fps,
        config.parent / raw["calibration_selector"], config.parent / raw["models_config"],
        scale, raw["sgbm"], iterations, near, far,
    )


@dataclass(frozen=True)
class Request:
    revision: int
    paused: bool = False


@dataclass(frozen=True)
class Display:
    request: Request
    frame: Frame
    left: BgrImage
    right: BgrImage
    rectified: bool
    results: dict[str, LiveResult]
    errors: dict[str, str]


def colorize(result: LiveResult, depth: bool, settings: Settings) -> BgrImage:
    """Fixed near-to-far colors; grey denotes invalid/out-of-support values."""
    values: ScalarMap = result.depth_m if depth else result.disparity_px
    if depth:
        normalized: ScalarMap = (settings.far_m - values) / (settings.far_m - settings.near_m)
    else:
        normalized = values / 192
    valid: NDArray[np.bool_] = np.isfinite(values)
    packed: NDArray[np.uint8] = (255 * np.clip(np.where(valid, normalized, 0), 0, 1)).astype(np.uint8)
    colored: BgrImage = cv2.applyColorMap(packed, cv2.COLORMAP_TURBO)
    colored[~valid] = (113, 113, 113)
    return colored


def save_frame_view(
    path: Path, view: Display, show_depth: bool, settings: Settings,
) -> None:
    """Save the same-frame panels with their fixed palette and English labels."""
    width, height = 520, 292
    images: list[BgrImage] = [cv2.resize(eye, (width, height)) for eye in (view.left, view.right)]
    image_kind: str = "Rectified" if view.rectified else "Raw"
    labels: list[str] = [f"{image_kind} left", f"{image_kind} right"]
    for method in METHODS:
        name: str = LABELS[method] + (f" {settings.iterations}" if method == "igev" else "")
        if method in view.results:
            result: LiveResult = view.results[method]
            images.append(cv2.resize(colorize(result, show_depth, settings), (width, height),
                                     interpolation=cv2.INTER_NEAREST))
            labels.append(f"{name}  |  {result.inference_ms:.0f} ms inference")
        else:
            images.append(np.full_like(images[0], 113))
            labels.append(f"{name} unavailable")
    content_height: int = (height + 40) * 2
    strip: BgrImage = np.full((content_height + 48, (width + 8) * 3 + 8, 3), 240, dtype=np.uint8)
    for index, (panel, label) in enumerate(zip(images, labels)):
        row: int = int(index >= 2)
        column: int = index - 2 if row else index
        x: int = 8 + column * (width + 8) + ((width + 8) // 2 if row == 0 else 0)
        y: int = row * (height + 40)
        strip[y + 36:y + 36 + height, x:x + width] = panel
        cv2.putText(strip, label, (x, y + 24), cv2.FONT_HERSHEY_SIMPLEX, .62, (20, 26, 35), 1, cv2.LINE_AA)
    ramp: NDArray[np.uint8] = np.repeat(np.arange(256, dtype=np.uint8)[None, :], 12, axis=0)
    strip[content_height + 15:content_height + 27, 8:264] = cv2.applyColorMap(ramp, cv2.COLORMAP_TURBO)
    legend: str = (
        f"{settings.far_m:g} to {settings.near_m:g} m estimated axial depth"
        if show_depth else "0 to 192 px disparity"
    )
    cv2.putText(strip, legend + "   |   gray = invalid   |   same input frame",
                (276, content_height + 26), cv2.FONT_HERSHEY_SIMPLEX, .56, (20, 26, 35), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), strip):
        raise OSError(f"无法保存 {path}")


class DemoWindow:
    def __init__(
        self, settings: Settings, processor: LiveProcessor | None, calibration_error: str,
        latest: LatestFrame, stop: Event, root: Path, replay: bool,
    ) -> None:
        self.settings: Settings = settings
        self.processor: LiveProcessor | None = processor
        self.calibration_error: str = calibration_error
        self.latest: LatestFrame = latest
        self.stop: Event = stop
        self.root: Path = root
        self.request: Request = Request(0)
        self.completed: Display | None = None
        self.shown: Display | None = None
        self.photos: list[ImageTk.PhotoImage] = []
        self.preview_sequence: int | None = None
        self.preview_at: float = 0.0
        self.timestamps: deque[float] = deque(maxlen=120)
        self.show_depth: bool = True

        self.window: tk.Tk = tk.Tk()
        self.window.title("双目实时演示" + (" · 视频回放" if replay else " · USB 相机"))
        self.window.configure(bg="#101721")
        self.window.protocol("WM_DELETE_WINDOW", self.finish)
        self.window.bind("<Escape>", lambda _event: self.finish())
        controls: ttk.Frame = ttk.Frame(self.window, padding=10)
        controls.pack(fill="x")
        self.pause_button: ttk.Button = ttk.Button(controls, text="暂停", command=self.toggle_pause, state="disabled")
        self.pause_button.pack(side="left", padx=3)
        self.pause_button.bind("<space>", self.toggle_pause)
        ttk.Button(controls, text="深度 / 视差", command=self.toggle).pack(side="left", padx=3)
        self.save_button: ttk.Button = ttk.Button(controls, text="保存画面", command=self.save, state="disabled")
        self.save_button.pack(side="right", padx=3)
        if not replay and os.name == "nt":
            ttk.Button(controls, text="相机权限", command=self.open_camera_settings).pack(side="right", padx=3)
        self.window.bind("<space>", self.toggle_pause)
        self.panels: tk.Frame = tk.Frame(self.window, bg="#101721")
        self.panels.pack(fill="both", expand=True, padx=8, pady=8)
        self.views: list[tk.Label] = []
        self.labels: list[tk.Label] = []
        for column in range(6):
            self.panels.columnconfigure(column, weight=1, uniform="stereo")
        locations: tuple[tuple[int, int], ...] = ((0, 1), (0, 3), (1, 0), (1, 2), (1, 4))
        for row, column in locations:
            panel: tk.Frame = tk.Frame(self.panels, bg="#101721")
            panel.grid(row=row, column=column, columnspan=2, padx=5, pady=4, sticky="nsew")
            title: tk.Label = tk.Label(panel, bg="#101721", fg="#eef1f5", font=("Microsoft YaHei UI", 11))
            title.pack()
            self.labels.append(title)
            preview: tk.Label = tk.Label(panel, bg="#273241", fg="white", borderwidth=0)
            preview.pack()
            self.views.append(preview)
        self.status: tk.StringVar = tk.StringVar(value="正在等待画面")

        tk.Label(self.window, textvariable=self.status, bg="#101721", fg="#eef1f5", anchor="w",
                 wraplength=1100, justify="left", font=("Microsoft YaHei UI", 11)).pack(fill="x", padx=16, pady=10)
        self.metric: tk.StringVar = tk.StringVar()
        tk.Label(self.window, textvariable=self.metric, bg="#101721", fg="#a2b1c1", anchor="w",
                 font=("Microsoft YaHei UI", 10)).pack(fill="x", padx=16, pady=(0, 10))
        self._legend()
        self.worker: Thread = Thread(target=self.process_frames, daemon=True)
        self.worker.start()
        self.window.after(15, self.refresh)

    def open_camera_settings(self) -> None:
        try:
            os.startfile("ms-settings:privacy-webcam")
        except OSError as error:
            messagebox.showinfo(
                "打开 Windows 相机设置",
                "请从 Windows 开始菜单打开「设置 → 隐私和安全性 → 相机」。\n"
                "也可以打开「设置 → 蓝牙和设备 → 相机 → USB Camera」查看系统预览。\n\n"
                f"应用跳转失败：{error}",
                parent=self.window,
            )

    def process_frames(self) -> None:
        last_revision: int = -1
        last_sequence: int = 0
        blocked_revision: int = -1
        try:
            while not self.stop.is_set():
                request: Request = self.request
                if request.paused or blocked_revision == request.revision:
                    self.stop.wait(0.08)
                    continue
                if request.revision != last_revision:
                    last_revision, last_sequence = request.revision, 0
                frame: Frame | None = self.latest.wait_after(last_sequence)
                if frame is None:
                    continue
                last_sequence = frame.sequence
                results: dict[str, LiveResult] = {}
                errors: dict[str, str] = {}
                midpoint: int = self.settings.size[0] // 2
                left: BgrImage = frame.image[:, :midpoint]
                right: BgrImage = frame.image[:, midpoint:]
                rectified: bool = False
                if self.processor is not None:
                    try:
                        inputs: dict[str, BgrImage] = self.processor.prepare(frame.image)
                        left, right = inputs["left"], inputs["right"]
                        rectified = True
                        for method in METHODS:
                            if self.stop.is_set() or request is not self.request:
                                break
                            try:
                                results[method] = self.processor.reconstruct(inputs, method)
                            except (RuntimeError, ValueError, OSError, cv2.error) as error:
                                errors[method] = str(error)
                    except (ValueError, cv2.error) as error:
                        errors["校正"] = str(error)
                if request is self.request:
                    self.completed = Display(request, frame, left, right, rectified, results, errors)
                    if errors and not results:
                        blocked_revision = request.revision
        finally:
            if self.processor is not None:
                self.processor.close()

    def toggle_pause(self, _event: tk.Event | None = None) -> str:
        if self.request.paused:
            self.request = Request(self.request.revision + 1)
            self.pause_button.configure(text="暂停")
            self.clear()
        elif self.shown is not None and self.shown.results:
            self.request = Request(self.request.revision + 1, paused=True)
            self.pause_button.configure(text="继续")
            self.status.set(f"已暂停 · 帧 {self.shown.frame.sequence} · 点击继续或按空格恢复实时")
        return "break"

    def clear(self) -> None:
        self.shown = None
        self.preview_sequence = None
        self.timestamps.clear()
        self.save_button.configure(state="disabled")
        self.pause_button.configure(state="disabled")
        for view in self.views:
            view.configure(image="", text="")
        for label in self.labels:
            label.configure(text="")
        self.photos.clear()

    def _legend(self) -> None:
        self.metric.set(
            f"轴向深度估计（米） · 红色近 {self.settings.near_m:g}    蓝色远 {self.settings.far_m:g}    灰色无有效值"
            if self.show_depth else "视差（像素） · 蓝色 0    红色 192    灰色无有效值"
        )

    def toggle(self) -> None:
        self.show_depth = not self.show_depth
        self._legend()
        if self.shown is not None:
            self.draw(self.shown)

    def draw(self, view: Display) -> None:
        state: str = "校正" if view.rectified else "原始"
        titles: list[str] = [f"左目 · {state}", f"右目 · {state}"]
        images: list[BgrImage | None] = [view.left, view.right]
        for method in METHODS:
            label: str = LABELS[method] + (f" {self.settings.iterations}" if method == "igev" else "")
            if method in view.results:
                result: LiveResult = view.results[method]
                quantity: str = "深度" if self.show_depth else "视差"
                titles.append(f"{label} · {quantity} · {result.inference_ms:.0f} ms")
                images.append(colorize(result, self.show_depth, self.settings))
            else:
                titles.append(label + " · 暂无结果")
                images.append(None)
        panel_width: int = min(640, (self.window.winfo_screenwidth() - 80) // 3)
        panel_height: int = max(100, (self.window.winfo_screenheight() - 200) // 2)
        photos: list[ImageTk.PhotoImage] = []
        for index, (title, image) in enumerate(zip(titles, images)):
            self.labels[index].configure(text=title)
            if image is None:
                self.views[index].configure(image="", text="—")
                continue
            pillow_image: Image.Image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            pillow_image.thumbnail((panel_width, panel_height), Image.Resampling.NEAREST)
            photo: ImageTk.PhotoImage = ImageTk.PhotoImage(pillow_image)
            self.views[index].configure(image=photo, text="")
            photos.append(photo)
        self.photos = photos
        controls_state: str = "normal" if view.results else "disabled"
        self.save_button.configure(state=controls_state)
        self.pause_button.configure(state=controls_state)

    def refresh(self) -> None:
        if self.request.paused:
            self.window.after(15, self.refresh)
            return
        frame, epoch, source_status = self.latest.snapshot()
        request: Request = self.request
        display: Display | None = self.completed
        valid: bool = (
            display is not None and display.request is request
            and frame is not None and display.frame.epoch == epoch
        )
        if valid and display is not None:
            if display is not self.shown:
                self.draw(display)
                self.shown = display
                self.timestamps.append(perf_counter())
            if self.timestamps:
                count: int = len(self.timestamps)
                fps: float = (count - 1) / max(0.001, self.timestamps[-1] - self.timestamps[0]) if count > 1 else 0
                lag_ms: float = (self.timestamps[-1] - display.frame.received_at) * 1000
                status: str = f"同帧对照 · 帧 {display.frame.sequence} · {fps:.1f} 帧/秒 · 收帧到提交界面 {lag_ms:.0f} ms"
            else:
                status = source_status
            if display.errors:
                status += " · " + " ／ ".join(f"{key}: {error}" for key, error in display.errors.items())
            self.status.set(status + (f" · {self.calibration_error}" if self.calibration_error else ""))
        else:
            if self.shown is not None:
                self.clear()
            preview: Frame | None = frame
            now: float = perf_counter()
            if preview is None and self.preview_sequence is not None:
                self.clear()
            elif preview is not None and preview.sequence != self.preview_sequence and now - self.preview_at > 0.07:
                midpoint: int = self.settings.size[0] // 2
                self.draw(Display(request, preview, preview.image[:, :midpoint], preview.image[:, midpoint:],
                                  False, {}, {}))
                self.preview_sequence = preview.sequence
                self.preview_at = now
            self.status.set(source_status if frame is None else "正在计算同帧结果，首次加载模型需要一些时间…")
        self.window.after(15, self.refresh)

    def save(self) -> None:
        if self.shown is None:
            return
        view: Display = self.shown
        stamp: str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        folder: Path = self.root / stamp
        try:
            folder.mkdir()
            save_frame_view(folder / "comparison.png", view, self.show_depth, self.settings)
            if not cv2.imwrite(str(folder / "raw.png"), view.frame.image):
                raise OSError("无法保存原始帧")
            for method, result in view.results.items():
                np.savez_compressed(folder / f"{method}.npz", disparity_px=result.disparity_px, depth_m=result.depth_m)
            self.metric.set(f"已保存 · {folder}")
        except (OSError, cv2.error) as error:
            messagebox.showerror("保存失败", str(error), parent=self.window)

    def finish(self) -> None:
        self.stop.set()
        self.window.destroy()


def benchmark(
    processor: LiveProcessor, latest: LatestFrame, seconds: float, settings: Settings,
    root: Path, sample_path: Path | None,
) -> None:
    """Replay independently per method, including rectification, depth and coloring."""
    if (root / "benchmark.json").exists():
        raise FileExistsError("回放计时请使用新的输出目录")
    first: Frame | None = latest.wait_after(0, 8)
    if first is None:
        raise RuntimeError(latest.snapshot()[2])
    if sample_path is not None:
        image: BgrImage | None = cv2.imread(str(sample_path))
        if image is None:
            raise OSError(f"无法读取指定的对照帧：{sample_path}")
        first = Frame(image, 0, 0, perf_counter())
    inputs: dict[str, BgrImage] = processor.prepare(first.image)
    same_frame: dict[str, LiveResult] = {}
    errors: dict[str, str] = {}
    metrics: dict[str, Any] = {}
    for method in METHODS:
        cold_start: float = perf_counter()
        try:
            processor.reconstruct(inputs, method)
            cold_ms: float = (perf_counter() - cold_start) * 1000
            same_frame[method] = processor.reconstruct(inputs, method)
            started: float = perf_counter()
            current: Frame | None = latest.snapshot()[0]
            sequence: int = current.sequence if current is not None else 0
            durations: list[float] = []
            latency: list[float] = []
            inference: list[float] = []
            while perf_counter() - started < seconds:
                frame: Frame | None = latest.wait_after(sequence)
                if frame is None:
                    continue
                sequence = frame.sequence
                loop_start: float = perf_counter()
                result: LiveResult = processor.reconstruct(processor.prepare(frame.image), method)
                colorize(result, True, settings)
                finished: float = perf_counter()
                durations.append((finished - loop_start) * 1000)
                latency.append((finished - frame.received_at) * 1000)
                inference.append(result.inference_ms)
            elapsed: float = perf_counter() - started
            if not durations:
                raise RuntimeError("回放期间没有完成一帧")
            metrics[method] = {
                "frames": len(durations), "seconds": elapsed, "depth_fps": len(durations) / elapsed,
                "median_inference_ms": median(inference), "median_pipeline_ms": median(durations),
                "p95_host_receive_to_render_ms": float(np.percentile(latency, 95)),
                "first_result_from_prepared_ms": cold_ms,
                "same_frame_valid_fraction": float(np.isfinite(same_frame[method].depth_m).mean()),
            }
            print(json.dumps({"method": method, **metrics[method]}), flush=True)
        except (RuntimeError, ValueError, OSError, cv2.error) as error:
            errors[method] = str(error)
            print(json.dumps({"method": method, "error": str(error)}, ensure_ascii=False), flush=True)
    view: Display = Display(Request(0), first, inputs["left"], inputs["right"], True, same_frame, errors)
    for depth, name in ((True, "same-frame-depth.png"), (False, "same-frame-disparity.png")):
        save_frame_view(root / name, view, depth, settings)
    write_json(root / "benchmark.json", {
        "igev_iterations": settings.iterations, "input_wh_per_eye": list(inputs["left"].shape[1::-1]),
        "source_fps": settings.fps, "timing_scope": "host replay to rendered array, without Tk presentation",
        "sample": str(sample_path) if sample_path is not None else "first paced replay frame",
        "methods": metrics, "errors": errors,
    })
    if errors:
        raise RuntimeError("部分方法运行失败，见 benchmark.json")


def public_replay_pair(config: Path, scale: float) -> tuple[BgrImage, LiveGeometry]:
    """Use the repository's already rectified Middlebury pair and its own metric Q."""
    offline = load_offline_config(config)
    scene = load_middlebury_scene(
        offline.image_root, offline.split, offline.scene_name, ground_truth_root=offline.ground_truth_root,
    )
    geometry = scene.geometry
    width, height = geometry.image_size
    if width % 2 or height % 2:
        raise ValueError("公开示范图的宽高须均为偶数以使用半尺寸重投影")
    y, x = np.indices((height, width), dtype=np.float32)
    maps: dict[str, NDArray[Any]] = {
        "image_size": np.array([width, height]), "map1_x": x, "map1_y": y, "map2_x": x, "map2_y": y,
    }
    pair = scene.rectified_pair
    frame: BgrImage = np.concatenate((pair.left_bgr, pair.right_bgr), axis=1)
    return frame, LiveGeometry(maps, (width, height), scaled_q(geometry.q, scale), scale)


def replay_static(frame: BgrImage, fps: int, latest: LatestFrame, stop: Event) -> None:
    """Publish the same public stereo pair at the replay clock rate."""
    while not stop.is_set():
        latest.publish(frame)
        stop.wait(1 / fps)


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/live_demo.local.json"))
    output: argparse._MutuallyExclusiveGroup = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--run-dir", type=Path)
    output.add_argument("--output-root", type=Path)
    source: argparse._MutuallyExclusiveGroup = parser.add_mutually_exclusive_group()
    source.add_argument("--replay", type=Path)
    source.add_argument("--saved-replay", action="store_true")
    source.add_argument("--middlebury-replay", action="store_true")
    source.add_argument("--direct-camera", action="store_true")
    source.add_argument("--native-launch-settings", action="store_true")
    parser.add_argument("--benchmark-seconds", type=float)
    parser.add_argument("--sample-frame", type=Path)
    parser.add_argument("--igev-iterations", type=int, choices=(8, 32))
    options: argparse.Namespace = parser.parse_args()
    config_path: Path = options.config.resolve()
    settings: Settings = load_settings(config_path, options.igev_iterations, public_replay=options.middlebury_replay)
    if options.native_launch_settings:
        if options.output_root is None:
            parser.error("--native-launch-settings requires --output-root")
        reference: Path = Path(settings.capture["ffmpeg_path"])
        if reference.parent != Path("."):
            reference = (settings.capture_config.parent / reference).resolve()
        ffmpeg: str | None = which(str(reference))
        if ffmpeg is None:
            raise FileNotFoundError(f"找不到 FFmpeg：{reference}")
        run_dir: Path = options.output_root.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        values: dict[str, str] = {
            "STEREO_FFMPEG": ffmpeg,
            "STEREO_DEVICE": settings.capture["device_name"],
            "STEREO_FRAME_WIDTH": str(settings.size[0]),
            "STEREO_FRAME_HEIGHT": str(settings.size[1]),
            "STEREO_FRAME_RATE": str(settings.fps),
            "STEREO_RUN_DIR": str(run_dir),
        }
        if any(any(character in value for character in '"\r\n') for value in values.values()):
            raise ValueError("相机启动配置的路径或名称包含无效字符")
        run_dir.mkdir(parents=True)
        for key, value in values.items():
            print(f"{key}={value}")
        return
    if options.saved_replay:
        if settings.replay_file is None:
            raise ValueError("本机配置缺少 replay_file，无法启动回放")
        options.replay = settings.replay_file
    if options.replay is not None and not options.replay.is_file():
        raise FileNotFoundError(f"找不到回放视频：{options.replay}")
    is_replay: bool = options.replay is not None or options.middlebury_replay
    seconds: float | None = options.benchmark_seconds
    if seconds is not None and (not is_replay or not np.isfinite(seconds) or seconds <= 0):
        parser.error("benchmark requires a replay source and a positive test duration")
    if options.sample_frame is not None and seconds is None:
        parser.error("--sample-frame is only for the benchmark comparison")
    stop: Event = Event()
    latest: LatestFrame = LatestFrame()
    if options.output_root is None:
        root: Path = options.run_dir.resolve()
        root.mkdir(parents=True, exist_ok=True)
    else:
        root = options.output_root.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        root.mkdir(parents=True)
        write_json(root / "startup.json", {"ui_pid": os.getpid(), "source": "replay" if is_replay else "camera"})
    cv2.setNumThreads(1)
    processor: LiveProcessor | None = None
    calibration_error: str = ""
    public_frame: BgrImage | None = None
    try:
        if options.middlebury_replay:
            public_frame, geometry = public_replay_pair(config_path.parent / "offline.json", settings.scale)
            settings = replace(settings, size=(public_frame.shape[1], public_frame.shape[0]))
        else:
            geometry = load_geometry(settings.selector, settings.capture, settings.scale)
        processor = LiveProcessor(geometry, settings.sgbm, settings.models, root / "runtime", settings.iterations, stop)
    except (OSError, KeyError, ValueError) as error:
        if seconds is not None or options.middlebury_replay:
            raise
        calibration_error = f"深度暂不可用：{error}"
    if options.direct_camera:
        executable: Path = Path(settings.capture["ffmpeg_path"])
        if executable.parent != Path("."):
            executable = (settings.capture_config.parent / executable).resolve()
        producer: Thread = Thread(
            target=capture_ffmpeg,
            args=(executable, settings.capture["device_name"], settings.fps, settings.size, root, latest, stop),
            daemon=True,
        )
    elif public_frame is not None:
        producer = Thread(target=replay_static, args=(public_frame, settings.fps, latest, stop), daemon=True)
    elif options.replay is None:
        producer = Thread(target=receive_camera, args=(root, settings.size, latest, stop), daemon=True)
    else:
        producer = Thread(
            target=replay_video, args=(options.replay.resolve(), settings.fps, settings.size, latest, stop), daemon=True,
        )
    producer.start()
    try:
        if seconds is not None and processor is not None:
            benchmark(processor, latest, seconds, settings, root, options.sample_frame)
        else:
            app: DemoWindow = DemoWindow(settings, processor, calibration_error, latest, stop, root, is_replay)
            app.window.mainloop()
            stop.set()
            app.worker.join(3.0)
            processor = None  # The processing thread owns its process cleanup.
    finally:
        stop.set()
        producer.join(3.0)
        if processor is not None:
            processor.close()


if __name__ == "__main__":
    main()
