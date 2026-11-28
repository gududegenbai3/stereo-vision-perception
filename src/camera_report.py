"""Verified qualitative gallery; deliberately does not calculate accuracy without GT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .camera_comparison import verify
from .camera_io import capture_inventory, digest, write_json


def common_domain(folder: Path, maximum: int) -> NDArray[np.bool_]:
    """Shared HxW bool region supported by both inputs for every candidate d."""
    left: NDArray[np.uint8] = cv2.imread(str(folder / "left_support.png"), cv2.IMREAD_GRAYSCALE)
    right: NDArray[np.uint8] = cv2.imread(str(folder / "right_support.png"), cv2.IMREAD_GRAYSCALE)
    valid: NDArray[np.bool_] = (left == 255) & (cv2.erode(right, np.ones((1, maximum + 1), np.uint8), anchor=(maximum, 0), borderType=cv2.BORDER_CONSTANT, borderValue=0) == 255)
    valid[:3] = False
    valid[-3:] = False
    valid[:, :maximum + 2] = False
    valid[:, -3:] = False
    if not valid.any():
        raise ValueError("Empty shared comparison domain")
    return valid


def report(config: Path) -> None:
    """Verify exact input/output identities before displaying all frozen samples."""
    protocol, root = verify(config)
    if capture_inventory(Path(protocol["session_root"])) != [{key: row[key] for key in ("index", "pair_id", "sha256", "wh")} for row in protocol["samples"]]:
        raise ValueError("Original capture inventory changed")
    methods: tuple[str, ...] = ("sgbm", "banet", "igev")
    records: dict[str, list[dict[str, Any]]] = {}
    timings: dict[str, float] = {}
    for method in methods:
        execution: dict[str, Any] = json.loads((root / f"{method}_execution.json").read_text(encoding="utf-8"))
        if execution["training"] is not False or [r["pair_id"] for r in execution["records"]] != [s["pair_id"] for s in protocol["samples"]]:
            raise ValueError("Incomplete/wrong inference sample set")
        if method != "sgbm" and execution["model"]["checkpoint_sha256"] != protocol["models"][method]["checkpoint_sha256"]:
            raise ValueError("Wrong official checkpoint")
        records[method] = execution["records"]
        timings[method] = float(np.median([r["elapsed_ms"] for r in records[method]]))
    for sample in protocol["samples"]:
        number: int = sample["index"] + 1
        folder: Path = root / sample["pair_id"]
        for name, expected in sample["input_sha256"].items():
            if digest(folder / f"{name}.png") != expected:
                raise ValueError("Shared input/mask changed")
        domain: NDArray[np.bool_] = common_domain(folder, protocol["settings"]["max_disparity"])
        fig, axes = plt.subplots(1, 4, figsize=(17, 3.6), layout="constrained")
        axes[0].imshow(cv2.cvtColor(cv2.imread(str(folder / "left.png")), cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Capture #{number:02d}: left RGB")
        for axis, method, title in zip(axes[1:], methods, ("SGBM", "BANet official pretrained", "IGEV official pretrained, 32")):
            file: Path = folder / f"{method}.npy"
            if digest(file) != records[method][sample["index"]]["sha256"]:
                raise ValueError("Prediction changed after inference")
            values: NDArray[np.float32] = np.load(file, allow_pickle=False)
            if values.dtype != np.float32 or values.shape != domain.shape or np.isinf(values).any():
                raise ValueError("Wrong prediction shape/dtype/values")
            colour = axis.imshow(np.ma.masked_invalid(np.where(domain, values, np.nan)), vmin=0, vmax=192, cmap=plt.get_cmap("turbo").with_extremes(bad="#888888"))
            axis.set_title(title)
        for axis in axes:
            axis.axis("off")
        fig.colorbar(colour, ax=axes[1:], shrink=.65, label="disparity / input pixel")
        fig.suptitle("Fixed calibration, shared inputs; NO GT. Gray = outside common ROI or missing.")
        fig.savefig(root / f"capture-{number:02d}.png", dpi=135)
        plt.close(fig)
    summary: dict[str, Any] = {"pairs": len(protocol["samples"]), "predictions_verified": len(protocol["samples"]) * 3,
        "display_pair_numbers": [i + 1 for i in protocol["display_indices"]], "median_times_ms": timings,
        "training": False, "accuracy": "not_evaluated_no_ground_truth", "scope": protocol["scope"]}
    write_json(root / "summary.json", summary)
    html: str = '<!doctype html><meta charset="utf-8"><title>固定标定三方法对照</title><style>body{font:17px sans-serif;max-width:1600px;margin:25px auto}img{width:100%}section{margin:30px 0}</style><h1>固定标定：SGBM / 官方预训练 BANet / IGEV</h1><p>每行：左图 / SGBM / BANet / IGEV-32。相同输入与色标。灰色为共同比较区域外或无输出；平滑不等于正确，没有视差真值或米制精度结论。预抽图号：' + ', '.join(map(str, summary['display_pair_numbers'])) + '。</p>'
    for sample in protocol["samples"]:
        number = sample["index"] + 1
        html += f'<section><h2>#{number:02d}</h2><img loading="lazy" src="capture-{number:02d}.png"></section>'
    (root / "gallery.html").write_text(html, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/camera_comparison.local.json"))
    args: argparse.Namespace = parser.parse_args()
    report(args.config.resolve())
