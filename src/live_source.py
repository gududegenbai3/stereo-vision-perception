"""Receive only complete side-by-side camera frames, or replay an AVI at camera rate."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import socket
import subprocess
from threading import Condition, Event, Thread
from time import perf_counter
from typing import BinaryIO

import cv2
import numpy as np
from numpy.typing import NDArray

from .camera_io import write_json


@dataclass(frozen=True)
class Frame:
    """One native H x 2W x 3 BGR uint8 frame and host monotonic receive time."""

    image: NDArray[np.uint8]
    sequence: int
    epoch: int
    received_at: float


class LatestFrame:
    """A one-slot handoff. A disconnect invalidates all earlier results."""

    def __init__(self) -> None:
        self.condition: Condition = Condition()
        self.frame: Frame | None = None
        self.epoch: int = 0
        self.sequence: int = 0
        self.status: str = "等待相机或视频"

    def publish(self, image: NDArray[np.uint8]) -> None:
        with self.condition:
            self.sequence += 1
            self.frame = Frame(image, self.sequence, self.epoch, perf_counter())
            self.status = "已收到双目画面"
            self.condition.notify_all()

    def disconnect(self, status: str) -> None:
        with self.condition:
            self.epoch += 1
            self.frame = None
            self.status = status
            self.condition.notify_all()

    def snapshot(self) -> tuple[Frame | None, int, str]:
        with self.condition:
            return self.frame, self.epoch, self.status

    def wait_after(self, sequence: int, timeout: float = 0.1) -> Frame | None:
        with self.condition:
            if not self.condition.wait_for(
                lambda: self.frame is not None and self.frame.sequence > sequence, timeout,
            ):
                return None
            return self.frame


def receive_camera(root: Path, size: tuple[int, int], latest: LatestFrame, stop: Event) -> None:
    """Publish the loopback port, then accept successive native FFmpeg sessions."""
    width, height = size
    frame_bytes: int = width * height * 3
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            server.settimeout(0.25)
            ready_temp: Path = root / "receiver.tmp"
            write_json(ready_temp, {"port": server.getsockname()[1]})
            ready_temp.replace(root / "receiver.json")
            while not stop.is_set():
                try:
                    connection, _ = server.accept()
                except socket.timeout:
                    continue
                with connection:
                    connection.settimeout(0.25)
                    last_frame: float = perf_counter()
                    connected: bool = False
                    try:
                        while not stop.is_set():
                            data: bytearray = bytearray(frame_bytes)
                            view: memoryview = memoryview(data)
                            offset: int = 0
                            while offset < frame_bytes and not stop.is_set():
                                try:
                                    received: int = connection.recv_into(view[offset:])
                                except socket.timeout:
                                    if perf_counter() - last_frame < 2.0:
                                        continue
                                    raise TimeoutError("相机画面超时")
                                if received == 0:
                                    raise EOFError
                                offset += received
                            if offset == frame_bytes:
                                latest.publish(np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3))
                                last_frame = perf_counter()
                                if not connected:
                                    frame: Frame | None = latest.snapshot()[0]
                                    if frame is not None:
                                        write_json(root / "camera-connected.json", {
                                            "backend": "external-ffmpeg-tcp", "frame": frame.sequence,
                                            "size_wh": [width, height],
                                        })
                                    connected = True
                    except (OSError, EOFError):
                        latest.disconnect("连接中断，等待相机重新接入")
    except OSError as error:
        latest.disconnect(f"接收器启动失败：{error}")


def read_raw_frames(stream: BinaryIO, size: tuple[int, int], latest: LatestFrame, stop: Event) -> None:
    """Publish complete BGR24 frames from a byte stream, ignoring a partial EOF."""
    width, height = size
    frame_bytes: int = width * height * 3
    while not stop.is_set():
        data: bytearray = bytearray(frame_bytes)
        view: memoryview = memoryview(data)
        offset: int = 0
        while offset < frame_bytes:
            count: int = stream.readinto(view[offset:])
            if not count or stop.is_set():
                return
            offset += count
        latest.publish(np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3))


def camera_privacy_settings() -> dict[str, str]:
    """Read Windows camera consent settings visible to the current process."""
    if os.name != "nt":
        return {}
    import winreg

    webcam: str = r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam"
    settings: dict[str, str] = {}
    for name, hive, subkey in (
        ("device", winreg.HKEY_LOCAL_MACHINE, webcam),
        ("apps", winreg.HKEY_CURRENT_USER, webcam),
        ("desktop_apps", winreg.HKEY_CURRENT_USER, webcam + r"\NonPackaged"),
    ):
        try:
            with winreg.OpenKey(hive, subkey) as key:
                value, _ = winreg.QueryValueEx(key, "Value")
                settings[name] = str(value)
        except FileNotFoundError:
            settings[name] = "not_configured"
        except OSError:
            settings[name] = "not_readable"
    return settings


def capture_ffmpeg(
    executable: Path, device: str, fps: int, size: tuple[int, int], root: Path,
    latest: LatestFrame, stop: Event,
) -> None:
    """Supervise local camera workers and retry with the calibrated USB alternative."""
    width, height = size
    ffmpeg: str | None = shutil.which(str(executable))
    if ffmpeg is None:
        latest.disconnect(f"未找到 FFmpeg：{executable}")
        return
    command: list[str] = [
        ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "warning",
        "-f", "dshow", "-rtbufsize", "128M", "-video_size", f"{width}x{height}",
        "-framerate", str(fps), "-vcodec", "mjpeg", "-i", f"video={device}",
        "-an", "-vf", f"settb=expr=1/{fps},setpts=N", "-enc_time_base", f"1:{fps}",
        "-fps_mode", "passthrough", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
    ]
    backend: str = "DirectShow"
    log: Path = root / "camera.stderr.log"
    previous_consent: dict[str, str] | None = None
    try:
        with log.open("ab", buffering=0) as stderr:
            while not stop.is_set():
                stderr.write(f"[{backend}] opening {width}x{height}\n".encode("utf-8"))
                with subprocess.Popen(
                    command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=stderr,
                    bufsize=0, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                ) as process:
                    if process.stdout is None:
                        raise RuntimeError(f"无法读取 {backend} 输出")
                    reader: Thread = Thread(
                        target=read_raw_frames, args=(process.stdout, size, latest, stop), daemon=True,
                    )
                    started: float = perf_counter()
                    connected: bool = False
                    reader.start()
                    try:
                        while not stop.wait(0.1) and reader.is_alive():
                            frame: Frame | None = latest.snapshot()[0]
                            if frame is not None and not connected:
                                connected = True
                                write_json(root / "camera-connected.json", {
                                    "backend": backend, "frame": frame.sequence, "size_wh": [width, height],
                                })
                            if perf_counter() - (frame.received_at if frame is not None else started) > 5:
                                break
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(1)
                            except subprocess.TimeoutExpired:
                                process.kill()
                        reader.join(1)
                if not stop.is_set():
                    consent: dict[str, str] = camera_privacy_settings()
                    if consent != previous_consent:
                        write_json(root / "camera-access.json", consent)
                        previous_consent = consent
                    message: str = (
                        "Windows 相机权限已关闭。请点「相机权限」开启，应用会自动重试。"
                        if "Deny" in consent.values()
                        else f"{backend} 未收到双目帧，正在重试。详情见 camera.stderr.log。"
                    )
                    latest.disconnect(message)
                    stop.wait(1)
    except OSError as error:
        latest.disconnect(f"相机采集进程失败：{error}")


def replay_video(path: Path, fps: float, size: tuple[int, int], latest: LatestFrame, stop: Event) -> None:
    """Use the recorded acquisition FPS rather than the AVI container's timebase."""
    capture: cv2.VideoCapture = cv2.VideoCapture(str(path))
    deadline: float = perf_counter()
    previous_position: int = 0
    try:
        if not capture.isOpened():
            raise OSError(f"无法打开回放视频：{path}")
        while not stop.is_set():
            ok, image = capture.read()
            if not ok:
                if previous_position == 0 or not capture.set(cv2.CAP_PROP_POS_FRAMES, 0):
                    raise OSError("回放读取失败")
                previous_position = 0
                continue
            if image.shape != (size[1], size[0], 3) or image.dtype != np.uint8:
                raise ValueError("回放视频尺寸与采集配置不符")
            previous_position += 1
            if stop.wait(max(0.0, deadline - perf_counter())):
                return
            latest.publish(image)
            deadline = max(deadline + 1 / fps, perf_counter())
    except (OSError, ValueError, cv2.error) as error:
        latest.disconnect(str(error))
    finally:
        capture.release()
