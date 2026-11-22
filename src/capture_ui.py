"""Button-driven raw stereo capture; external native FFmpeg supplies loopback BGR.

No model, calibration or GT dependency. Preview scaling never changes saved pixels.
Transport arrays are uint8 (H,2W,3) BGR; positional halves are raw, unrectified.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import threading
import time
import tkinter as tk
from tkinter import messagebox
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageTk

from .calibration_pattern import detect_paired_corners, make_chessboard


@dataclass(frozen=True)
class CaptureSettings:
    """Validated dimensions in pixels, roots absolute; same-frame L/R halves."""

    frame_wh: tuple[int, int]
    preview_wh: tuple[int, int]
    capture_root: Path
    ready_file: Path
    metadata: dict[str, Any]


def load_settings(path: Path) -> CaptureSettings:
    """Parse machine-readable capture settings; reject ambiguous order or geometry."""
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    for key in ("frame_wh", "preview_wh"):
        if len(raw[key]) != 2 or any(type(v) is not int or v <= 0 for v in raw[key]):
            raise ValueError(f"Invalid {key}")
    if raw["frame_wh"][0] % 2 or raw["physical_order"] != "left_half_is_left_camera":
        raise ValueError("Require even-width, confirmed left/right side-by-side input")
    if raw["rectified"] is not False or raw["calibration_file"] is not None:
        raise ValueError("Capture UI saves raw unrectified pixels only")
    if "chessboard" in raw:
        board: dict[str, Any] = raw["chessboard"]
        for key in ("inner_cols", "inner_rows", "target_calibration_pairs", "target_validation_pairs"):
            if type(board[key]) is not int or board[key] < 1:
                raise ValueError(f"Invalid chessboard {key}")
        square_size: object = board.get("square_size_m")
        if square_size is not None and (type(square_size) not in (int, float) or not np.isfinite(square_size) or square_size <= 0):
            raise ValueError("Physical square size must be null or positive metres")
        cell_px: object = board.get("display_square_px")
        if cell_px is not None and (type(cell_px) is not int or cell_px < 1):
            raise ValueError("Display square size must be positive integer pixels")
    return CaptureSettings(tuple(raw["frame_wh"]), tuple(raw["preview_wh"]),
        (path.parent / raw["capture_root"]).resolve(), (path.parent / raw["ready_file"]).resolve(), raw)


def save_raw_pair(
    session_root: Path, frame_bgr: NDArray[np.uint8], pair_id: str, read_time_ns: int,
) -> dict[str, Any]:
    """Write lossless raw/L/R PNGs, then register a complete pair in CSV.

    uint8 BGR input is H,2W,3; output mono dimensions are W,H. Read timestamp is
    the completed host receipt of one combined frame, not exposure synchronization.
    Failures raise and never append an incomplete pair to pairs.csv.
    """
    if frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.shape[1] % 2:
        raise ValueError("Expected uint8 H,2W,3 BGR")
    folder: Path = session_root / pair_id
    folder.mkdir(exist_ok=False)
    height, total_width = frame_bgr.shape[:2]
    width: int = total_width // 2
    hashes: dict[str, str] = {}
    images: dict[str, NDArray[np.uint8]] = {"raw": frame_bgr, "left": frame_bgr[:, :width], "right": frame_bgr[:, width:]}
    for name, values in images.items():
        destination: Path = folder / f"{name}.png"
        if not cv2.imwrite(str(destination), values):
            raise OSError(f"Could not write {destination}")
        with destination.open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    record: dict[str, Any] = {"pair_id": pair_id, "left_path": f"{pair_id}/left.png", "right_path": f"{pair_id}/right.png",
        "left_read_time_ns": read_time_ns, "right_read_time_ns": read_time_ns, "width": width, "height": height}
    with (session_root / "pairs.csv").open("a", newline="", encoding="utf-8") as stream:
        writer: csv.DictWriter = csv.DictWriter(stream, fieldnames=list(record))
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(record)
    detail: dict[str, Any] = {**record, "raw_path": f"{pair_id}/raw.png", "sha256": hashes,
        "saved_at": datetime.now(timezone.utc).isoformat(), "rectified": False}
    (folder / "pair.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
    return detail


class CaptureWindow:
    """Main-thread Tk UI, background socket reader keeps only the latest raw frame."""

    def __init__(self, settings: CaptureSettings) -> None:
        self.settings: CaptureSettings = settings
        stamp: str = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
        self.session_root: Path = settings.capture_root / stamp
        self.session_root.mkdir(parents=True)
        self.stop: threading.Event = threading.Event()
        self.lock: threading.Lock = threading.Lock()
        self.frame: NDArray[np.uint8] | None = None
        self.frame_time_ns: int = 0
        self.sequence: int = 0
        self.last_saved_sequence: int = -1
        self.count: int = 0
        self.reader_error: str = ""
        self.connection: socket.socket | None = None
        self.server: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(1)
        self.server.settimeout(.5)
        self.window: tk.Tk = tk.Tk()
        self.window.title("双目拍照 — 点击拍照保存左右图")
        self.board: dict[str, Any] | None = settings.metadata.get("chessboard")
        self.purpose: tk.StringVar = tk.StringVar(value="calibration" if self.board else "scene")
        self.purpose_counts: dict[str, int] = {"calibration": 0, "validation": 0, "scene": 0}
        if self.board:
            self.window.title("屏幕棋盘标定拍照 — 停稳后按空格")
            self.window.attributes("-fullscreen", True)
        self.window.protocol("WM_DELETE_WINDOW", self.finish)
        self.status: tk.StringVar = tk.StringVar(value="正在连接相机……")
        self.preview: tk.Label = tk.Label(self.window, bg="#151515")
        self.preview.pack(padx=10, pady=10)
        controls: tk.Frame = tk.Frame(self.window)
        controls.pack(fill="x", padx=12, pady=5)
        self.shutter: tk.Button = tk.Button(controls, text="📷 拍照 / 空格", font=("Microsoft YaHei UI", 17), bg="#1670c5", fg="white", width=20, command=self.capture, state="disabled")
        self.shutter.pack(side="left", padx=8)
        tk.Button(controls, text="完成拍摄", font=("Microsoft YaHei UI", 15), command=self.finish).pack(side="right", padx=8)
        tk.Label(self.window, textvariable=self.status, font=("Microsoft YaHei UI", 13), anchor="w").pack(fill="x", padx=16, pady=8)
        instruction: str = "左半＝左镜头　右半＝右镜头｜停稳后拍照，每个场景拍 1～2 张即可"
        self.camera_thumbnail: tk.Label | None = None
        self.pattern_photo: ImageTk.PhotoImage | None = None
        self.square_px: int | None = None
        if self.board:
            instruction = "屏幕和格子大小保持不变｜移动整套相机，别改变两个镜头之间的位置｜两边都看全棋盘｜拍完标定图再选独立验证"
            available_cell: int = max(12, min((self.window.winfo_screenwidth() - 40) // (self.board["inner_cols"] + 3), (self.window.winfo_screenheight() - 260) // (self.board["inner_rows"] + 3)))
            self.square_px = self.board.get("display_square_px", available_cell)
            if self.square_px > available_cell:
                raise ValueError("Measured target no longer fits; restore original display mode instead of rescaling it")
            if settings.metadata.get("continuation_of_session"):
                instruction = f"补拍目标：标定 {self.board['target_calibration_pairs']} 对、独立验证 {self.board['target_validation_pairs']} 对；旧照片已保留｜镜头、支架、屏幕大小保持不变"
            pattern: NDArray[np.uint8] = make_chessboard(self.board["inner_cols"], self.board["inner_rows"], self.square_px)
            if not cv2.imwrite(str(self.session_root / "displayed_pattern.png"), pattern):
                raise OSError("Could not save displayed chessboard identity")
            self.pattern_photo = ImageTk.PhotoImage(Image.fromarray(pattern))
            self.preview.configure(image=self.pattern_photo, bg="white")
            self.window.configure(bg="white")
            for value, title in (("calibration", "标定图"), ("validation", "独立验证图")):
                tk.Radiobutton(controls, text=title, value=value, variable=self.purpose, font=("Microsoft YaHei UI", 12)).pack(side="left")
            self.camera_thumbnail = tk.Label(controls)
            self.camera_thumbnail.pack(side="left", padx=8)
        tk.Label(self.window, text=instruction, font=("Microsoft YaHei UI", 11), wraplength=1400).pack(pady=3)
        self.window.bind("<space>", lambda _event: self.capture())
        self.photo: ImageTk.PhotoImage | None = None
        session: dict[str, Any] = {"config": settings.metadata, "session_root": str(self.session_root), "rectified": False,
            "timestamp_meaning": "Host monotonic time after receiving one combined frame; not sensor exposure time", "saved_pairs": 0}
        if self.board:
            session.update(display_square_px=self.square_px, physical_square_size_m=self.board["square_size_m"], target_type="display", phase_counts=self.purpose_counts)
        (self.session_root / "session.json").write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")
        settings.ready_file.parent.mkdir(parents=True, exist_ok=True)
        settings.ready_file.write_text(json.dumps({"port": self.server.getsockname()[1], "pid": os.getpid(), "session_root": str(self.session_root)}), encoding="utf-8")
        threading.Thread(target=self.receive, daemon=True).start()
        self.window.after(80, self.refresh)

    def receive(self) -> None:
        """Assemble complete fixed-size raw frames; discard partial EOF frames."""
        width, height = self.settings.frame_wh
        size: int = width * height * 3
        try:
            while not self.stop.is_set():
                try:
                    self.connection, _ = self.server.accept()
                    break
                except socket.timeout:
                    continue
            if self.connection is None:
                return
            self.connection.settimeout(.5)
            while not self.stop.is_set():
                payload: bytearray = bytearray(size)
                offset: int = 0
                while offset < size and not self.stop.is_set():
                    try:
                        received: int = self.connection.recv_into(memoryview(payload)[offset:])
                    except socket.timeout:
                        continue
                    if received == 0:
                        raise EOFError("相机连接已结束")
                    offset += received
                if offset != size:
                    return
                frame: NDArray[np.uint8] = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
                with self.lock:
                    self.frame = frame
                    self.frame_time_ns = time.perf_counter_ns()
                    self.sequence += 1
        except (OSError, EOFError) as error:
            if not self.stop.is_set():
                self.reader_error = str(error)

    def refresh(self) -> None:
        """Render an independent preview; reject stale frames for capture."""
        if self.stop.is_set():
            return
        with self.lock:
            frame: NDArray[np.uint8] | None = self.frame
            frame_time_ns: int = self.frame_time_ns
        fresh: bool = frame is not None and time.perf_counter_ns() - frame_time_ns < 1_000_000_000
        self.shutter.configure(state="normal" if fresh else "disabled")
        if frame is not None:
            image: Image.Image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image.thumbnail((240, 68) if self.board else self.settings.preview_wh)
            self.photo = ImageTk.PhotoImage(image)
            if self.camera_thumbnail is not None:
                self.camera_thumbnail.configure(image=self.photo)
            else:
                self.preview.configure(image=self.photo)
            if self.count == 0 and self.status.get() == "正在连接相机……":
                self.status.set(f"相机已连接｜本窗口目标：标定 {self.board['target_calibration_pairs']} 对，验证 {self.board['target_validation_pairs']} 对｜停稳 1 秒再按空格。" if self.board else "相机已连接｜已保存 0 对。点击蓝色“拍照”按钮或按空格。")
        if self.reader_error:
            self.status.set(f"连接中断：{self.reader_error}；已保存 {self.count} 对，不再接收拍照。")
            self.shutter.configure(state="disabled")
        self.window.after(80, self.refresh)

    def capture(self) -> None:
        """Save exactly one fresh combined frame for a user shutter action."""
        with self.lock:
            if self.frame is None or time.perf_counter_ns() - self.frame_time_ns > 1_000_000_000:
                self.status.set("还没有实时画面，请稍等。")
                return
            frame: NDArray[np.uint8] = self.frame.copy()
            timestamp: int = self.frame_time_ns
            sequence: int = self.sequence
        if sequence == self.last_saved_sequence:
            return
        corners_left: NDArray[np.float32] | None = None
        corners_right: NDArray[np.float32] | None = None
        if self.board:
            self.status.set("正在检查左右完整棋盘，请继续保持相机静止……")
            self.window.update_idletasks()
            corners_left, corners_right = detect_paired_corners(frame, self.board["inner_cols"], self.board["inner_rows"])
            if corners_left is None or corners_right is None:
                failed: str = "左、右" if corners_left is None and corners_right is None else ("左" if corners_left is None else "右")
                self.status.set(f"本次未保存：{failed}侧没检测全棋盘。退远一点、减少倾斜，并检查反光/清晰度后重拍。")
                return
        pair_id: str = f"pair_{self.count + 1:04d}_{timestamp}"
        try:
            detail: dict[str, Any] = save_raw_pair(self.session_root, frame, pair_id, timestamp)
            detail["purpose"] = self.purpose.get()
            if self.board:
                detail["board_inner_cols_rows"] = [self.board["inner_cols"], self.board["inner_rows"]]
                detail["square_size_m"] = self.board["square_size_m"]
                np.savez_compressed(self.session_root / pair_id / "corners.npz", left_xy=corners_left, right_xy=corners_right)
            (self.session_root / pair_id / "pair.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
        except (OSError, ValueError, cv2.error) as error:
            messagebox.showerror("保存失败", str(error), parent=self.window)
            return
        self.count += 1
        self.purpose_counts[self.purpose.get()] += 1
        self.last_saved_sequence = sequence
        self.status.set(f"✓ 已保存：标定 {self.purpose_counts['calibration']} 对 / 独立验证 {self.purpose_counts['validation']} 对。换位置或倾角，停稳后再拍。" if self.board else f"✓ 已保存 {self.count} 对原始左右图。可以换个场景继续拍。")
        print(json.dumps({"saved": self.count, "pair_id": pair_id, "session_root": str(self.session_root)}), flush=True)

    def finish(self) -> None:
        """Keep all saved pairs and mark the session done; close source socket."""
        self.stop.set()
        self.server.close()
        if self.connection is not None:
            self.connection.close()
        ended_at: str = datetime.now(timezone.utc).isoformat()
        session_file: Path = self.session_root / "session.json"
        session_record: dict[str, Any] = json.loads(session_file.read_text(encoding="utf-8"))
        session_record.update(saved_pairs=self.count, ended_at=ended_at, status="complete", phase_counts=self.purpose_counts)
        session_file.write_text(json.dumps(session_record, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.session_root / "DONE.json").write_text(json.dumps({"saved_pairs": self.count, "ended_at": ended_at}), encoding="utf-8")
        self.window.destroy()


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/capture_ui.local.json"))
    args: argparse.Namespace = parser.parse_args()
    settings: CaptureSettings = load_settings(args.config.resolve())
    cv2.setNumThreads(1)
    ui: CaptureWindow = CaptureWindow(settings)
    ui.window.mainloop()


if __name__ == "__main__":
    main()
