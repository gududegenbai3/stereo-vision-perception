"""Exact square-pixel chessboard and paired corner detection for screen capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


def make_chessboard(inner_cols: int, inner_rows: int, square_px: int) -> NDArray[np.uint8]:
    """Return uint8 HW gray pattern, one-square white border, square pixel cells.

    Physical cell size is unknown until measured on the actual display. Pixel
    size must never be substituted for millimetres in camera calibration.
    """
    if any(type(value) is not int or value < 1 for value in (inner_cols, inner_rows, square_px)):
        raise ValueError("Positive integer board dimensions required")
    board: NDArray[np.uint8] = np.full(((inner_rows + 3) * square_px, (inner_cols + 3) * square_px), 240, np.uint8)
    for row in range(inner_rows + 1):
        for col in range(inner_cols + 1):
            if (row + col) % 2 == 0:
                board[(row + 1) * square_px:(row + 2) * square_px, (col + 1) * square_px:(col + 2) * square_px] = 20
    return board


def detect_paired_corners(
    frame_bgr: NDArray[np.uint8], inner_cols: int, inner_rows: int,
) -> tuple[NDArray[np.float32] | None, NDArray[np.float32] | None]:
    """Detect both raw halves; float32 N,1,2 XY positions in each mono pixel grid.

    These are detected image coordinates only, not solved calibration parameters.
    Left/right ordering consistency still requires checking at calibration time.
    """
    if frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.shape[1] % 2:
        raise ValueError("Require combined BGR uint8 H,2W,3 image")
    width: int = frame_bgr.shape[1] // 2
    results: list[NDArray[np.float32] | None] = []
    for eye in (frame_bgr[:, :width], frame_bgr[:, width:]):
        gray: NDArray[np.uint8] = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCornersSB(gray, (inner_cols, inner_rows), flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        results.append(corners.astype(np.float32) if found else None)
    return results[0], results[1]


def show_measurement_pattern(session_root: Path) -> None:
    """Reopen saved target at native Tk pixel size; no camera access or resampling.

    Re-measurement is valid only on the same physical display with unchanged
    display mode/scaling. A red five-cell ruler is drawn outside the target grid.
    """
    import tkinter as tk
    from PIL import Image, ImageTk

    metadata: dict = json.loads((session_root / "session.json").read_text(encoding="utf-8"))
    cell: int = metadata["display_square_px"]
    pattern: NDArray[np.uint8] = cv2.imread(str(session_root / "displayed_pattern.png"), cv2.IMREAD_GRAYSCALE)
    target: dict = metadata["config"]["chessboard"]
    if pattern is None or not np.array_equal(pattern, make_chessboard(target["inner_cols"], target["inner_rows"], cell)):
        raise ValueError("Saved display pattern identity mismatch")
    window: tk.Tk = tk.Tk()
    window.title("补量棋盘格长 — 与原采集相同大小")
    window.attributes("-fullscreen", True)
    window.configure(bg="white")
    image: ImageTk.PhotoImage = ImageTk.PhotoImage(Image.fromarray(pattern))
    canvas: tk.Canvas = tk.Canvas(window, width=pattern.shape[1], height=pattern.shape[0], bg="white", highlightthickness=0)
    canvas.pack(pady=10)
    canvas.create_image(0, 0, image=image, anchor="nw")
    ruler_y: int = cell // 2
    canvas.create_line(cell, ruler_y, 6 * cell, ruler_y, fill="#c00000", width=3)
    for x in (cell, 6 * cell):
        canvas.create_line(x, ruler_y - 10, x, ruler_y + 10, fill="#c00000", width=3)
    canvas.create_text(7 * cell, ruler_y, text="红线两端＝5格", anchor="w", fill="#c00000", font=("Microsoft YaHei UI", 14))
    tk.Label(window, text="请量红线两端的距离，记录毫米数。屏幕分辨率和系统缩放须与拍摄时相同。", bg="white", font=("Microsoft YaHei UI", 15)).pack(pady=8)
    tk.Button(window, text="关闭（Esc）", font=("Microsoft YaHei UI", 14), command=window.destroy).pack(pady=8)
    window.bind("<Escape>", lambda _event: window.destroy())
    window.mainloop()


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure-session", type=Path, required=True)
    args: argparse.Namespace = parser.parse_args()
    show_measurement_pattern(args.measure_session.resolve())


if __name__ == "__main__":
    main()
