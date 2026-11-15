"""Inspect central failure points from saved stages; no inference or fitting."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .banet import environment_dll_handle, padding_for_shape
from .banet_failure_diagnosis import boundary_distance
from .error_diagnosis import write_json


def support_from_saved_coarse(coarse_native: np.ndarray, x: int, y: int, scale: float) -> list[float]:
    """Recover true low-grid values at plateau centers after nearest expansion.

    This is valid for the saved half-scale/quarter-grid pipeline; center samples
    remain inside constant plateaus after native bilinear expansion. Verified
    against the actual captured low tensors on all four original probes.
    """
    height, width = coarse_native.shape
    left_pad, _, top_pad, _ = padding_for_shape(round(height * scale), round(width * scale))
    input_x: float = (x + .5) * scale - .5
    input_y: float = (y + .5) * scale - .5
    values: list[float] = []
    for iy in {math.floor(input_y), math.ceil(input_y)}:
        for ix in {math.floor(input_x), math.ceil(input_x)}:
            low_y: int = (iy + top_pad) // 4
            low_x: int = (ix + left_pad) // 4
            for cy in range(low_y - 1, low_y + 2):
                for cx in range(low_x - 1, low_x + 2):
                    ny: int = round((4 * cy + 2 - top_pad) / scale - .5)
                    nx: int = round((4 * cx + 2 - left_pad) / scale - .5)
                    if not 0 <= ny < height or not 0 <= nx < width:
                        raise ValueError("Diagnostic support reaches image boundary")
                    values.append(float(coarse_native[ny, nx]))
    return [min(values), max(values)]


def analyze_image_borders(comparison_root: Path, jump: float) -> dict:
    """Separate image-frame artifacts from object-boundary errors on all24 pairs."""
    buckets: dict = {str(margin): {"pixels": 0, "large": 0, "far_object_large": 0} for margin in (0, 32, 64, 128)}
    for path in sorted((comparison_root / "mobile2021_new").glob("*/arrays.npz")):
        with np.load(path) as arrays:
            reference: np.ndarray = arrays["reference"] & arrays["sgbm192_valid"]
            errors: np.ndarray = np.abs(arrays["banet"].astype(np.float64) - arrays["truth"].astype(np.float64))
            large: np.ndarray = reference & (errors > 10)
            distance: np.ndarray = boundary_distance(arrays["truth"], jump)
            height, width = errors.shape
            rows: np.ndarray = np.arange(height, dtype=np.int32)
            columns: np.ndarray = np.arange(width, dtype=np.int32)
            border: np.ndarray = np.minimum(np.minimum(rows, height - 1 - rows)[:, None], np.minimum(columns, width - 1 - columns)[None, :])
            for margin, values in buckets.items():
                mask: np.ndarray = reference & (border >= int(margin))
                values["pixels"] += int(mask.sum())
                values["large"] += int((large & mask).sum())
                values["far_object_large"] += int((large & mask & (distance > 16)).sum())
    for values in buckets.values():
        values["large_fraction"] = values["large"] / values["pixels"]
        values["large_far_object_fraction"] = values["far_object_large"] / values["large"]
    return buckets


def main() -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/banet_failure_diagnosis.json"))
    args: argparse.Namespace = parser.parse_args()
    raw: dict = json.loads(args.config.read_text(encoding="utf-8"))
    root: Path = (args.config.parent / raw["output_root"]).resolve()
    comparison_config: Path = args.config.parent / raw["comparison_config"]
    comparison_settings: dict = json.loads(comparison_config.read_text(encoding="utf-8"))
    comparison_root: Path = (comparison_config.parent / comparison_settings["output_root"]).resolve()
    write_json(root / "image_border_analysis.json", analyze_image_borders(comparison_root, raw["gt_jump_native_px"]))
    summary: dict = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    for probe in summary["probes"]:
        with np.load(root / f"stages-{probe['sample']}.npz") as stages:
            x, y = probe["native_xy"]
            np.testing.assert_allclose(support_from_saved_coarse(stages["coarse"], x, y, .5), probe["upsample_support_native_px"], rtol=1e-6)
    records: list[dict] = []
    for name in ("skates2", "bandsaw2", "skiboots1"):
        with np.load(root / f"stages-{name}.npz") as stages, np.load(comparison_root / "mobile2021_new" / name / "arrays.npz") as prior:
            truth: np.ndarray = stages["truth"]
            height, width = truth.shape
            error: np.ndarray = np.abs(stages["final"] - truth)
            distance: np.ndarray = boundary_distance(truth, raw["gt_jump_native_px"])
            candidate: np.ndarray = prior["reference"] & prior["sgbm192_valid"] & (np.abs(prior["sgbm192"] - truth) <= 2) & (distance > 16)
            candidate[:128] = candidate[-128:] = False
            candidate[:, :128] = candidate[:, -128:] = False
            y, x = (int(value) for value in np.unravel_index(np.argmax(np.where(candidate, error, -np.inf)), error.shape))
            records.append({"sample": name, "native_xy": [x, y], "gt_native_px": float(truth[y, x]),
                "coarse_resized_native_px": float(stages["coarse"][y, x]), "final_native_px": float(stages["final"][y, x]),
                "sgbm_native_px": float(prior["sgbm192"][y, x]), "distance_to_gt_jump_px": float(distance[y, x]),
                "image_border_margin_px": int(min(x, y, width - 1 - x, height - 1 - y)),
                "coarse_neighbor_support_native_px": support_from_saved_coarse(stages["coarse"], x, y, .5)})
    write_json(root / "central_interior_probes.json", records)
    dll_handle: object | None = environment_dll_handle()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    point: dict = records[0]
    with np.load(root / "stages-skates2.npz") as stages:
        x, y = point["native_xy"]
        x0, x1 = x - 128, x + 128
        y0, y1 = y - 128, y + 128
        truth_crop: np.ndarray = np.where(stages["reference"][y0:y1, x0:x1], stages["truth"][y0:y1, x0:x1], np.nan)
        low: float = 0.
        high: float = math.ceil(float(np.nanpercentile(truth_crop, 99)) / 10) * 10
        cmap = plt.get_cmap("turbo").copy()
        cmap.set_bad("#999999")
        fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.5), layout="constrained")
        axes[0, 0].imshow(stages["left"][y0:y1, x0:x1, ::-1])
        axes[0, 0].set_title("挡板内部原图局部")
        image = axes[0, 1].imshow(truth_crop, cmap=cmap, vmin=low, vmax=high)
        axes[0, 1].set_title(f"真值：{point['gt_native_px']:.2f}px")
        for axis, key, label, value in ((axes[1, 0], "coarse", "聚合后粗视差（已换回原图单位）", point["coarse_resized_native_px"]),
                                        (axes[1, 1], "final", "最终输出", point["final_native_px"])):
            axis.imshow(stages[key][y0:y1, x0:x1], cmap=cmap, vmin=low, vmax=high)
            axis.set_title(f"{label}\n标记点：{value:.2f}px")
        for axis in axes.flat:
            axis.plot(128, 128, "+", color="#ff0000", markersize=15, markeredgewidth=2)
            axis.set_axis_off()
        fig.colorbar(image, ax=axes[:, :], shrink=.7, label="原图像素视差（所有图同一色标）")
        fig.suptitle("skates2：错误在粗视差阶段已经形成\n标记点距GT视差突变约217px，距图像外边框181px", fontsize=13)
        fig.savefig(root / "interior-stage-skates2.png", dpi=155)
        plt.close(fig)
    print(json.dumps({"central_probes": records, "support_recovery_verified_against_captured_tensors": True,
                      "figure": str(root / "interior-stage-skates2.png")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
