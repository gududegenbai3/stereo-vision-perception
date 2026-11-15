"""Render post-hoc figures and independently check saved diagnostic counts."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import cv2
import matplotlib
import numpy as np


from .error_diagnosis import REGIONS, jump_band, write_json

LABELS: dict[str, str] = {"outside_disparity_range": "真值超出搜索范围", "disparity_jump_band": "真值视差突变带",
                         "low_texture_interior": "非边界低纹理", "other_interior": "其余区域"}


def read_arrays(root: Path, name: str) -> dict[str, np.ndarray]:
    """Return saved baseline data, images uint8; disparities/errors float32 px."""
    with np.load(root / name / "arrays.npz") as archive:
        return {key: archive[key] for key in archive.files}


def audit(root: Path, summary: dict, views: dict) -> dict:
    """Recompute counts from arrays, not scene reports; include matched-input pairs."""
    counts: dict[int, int] = {key: 0 for key in REGIONS}
    severity_count: np.ndarray = np.zeros(6, np.int64)
    severity_sum: np.ndarray = np.zeros(6, np.float64)
    bins: list[float] = [0, .5, 1, 2, 5, 10, float("inf")]
    sensitivity: dict[tuple[float, int], np.ndarray] = {
        (float(jump), int(radius)): np.zeros(4, np.int64)
        for jump in summary["settings"]["sensitivity_jumps_px"] for radius in summary["settings"]["sensitivity_radii_px"]}
    adir: dict = {}
    for row in summary["scenes"]:
        a: dict = read_arrays(root, row["scene"])
        common: np.ndarray = a["reference"] & a["predicted_valid"]
        error: np.ndarray = np.abs(a["prediction"].astype(np.float64) - a["truth"].astype(np.float64))
        bad: np.ndarray = common & (error > summary["settings"]["bad_threshold_px"])
        assert int(common.sum()) == row["all"]["evaluated_pixels"]
        assert int(bad.sum()) == row["all"]["bad_pixels"]
        np.testing.assert_allclose(error[common].sum(), row["all"]["absolute_error_sum_px"], rtol=1e-12)
        for code, label in REGIONS.items():
            n: int = int((bad & (a["regions"] == code)).sum())
            assert n == row["regions"][label]["bad_pixels"]
            counts[code] += n
        for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
            selected: np.ndarray = common & (error >= low if i == 0 else error > low) & (error <= high)
            severity_count[i] += int(selected.sum())
            severity_sum[i] += float(error[selected].sum())
        outside: np.ndarray = a["regions"] == 1
        low: np.ndarray = a["texture_std"] < summary["settings"]["texture_std_threshold"]
        for (jump, radius), group_counts in sensitivity.items():
            boundary: np.ndarray = jump_band(a["truth"], jump, radius) & ~outside
            labels: np.ndarray = np.where(outside, 0, np.where(boundary, 1, np.where(low, 2, 3)))
            for i in range(4):
                group_counts[i] += int((bad & (labels == i)).sum())
        if row["scene"] == "Adirondack":
            adir = {"region_of_C": int(a["regions"][270, 480]), "predicted_C": float(a["prediction"][270, 480]),
                    "truth_C": float(a["truth"][270, 480])}
    for code, label in REGIONS.items():
        assert counts[code] == summary["regions"][label]["bad_pixels"]
    assert int(severity_count.sum()) == summary["overall"]["evaluated_pixels"]
    assert int(severity_count[3:].sum()) == summary["overall"]["bad_pixels"]
    pair_rows: list[dict] = []
    for base, variant in views["pairs"]:
        a, b = read_arrays(root, base), read_arrays(root, variant)
        paired: np.ndarray = a["reference"] & b["reference"] & a["predicted_valid"] & b["predicted_valid"]
        ea: np.ndarray = np.abs(a["prediction"][paired].astype(np.float64) - a["truth"][paired])
        eb: np.ndarray = np.abs(b["prediction"][paired].astype(np.float64) - b["truth"][paired])
        pair_rows.append({"base": base, "variant": variant, "same_left": bool(np.array_equal(a["left"], b["left"])),
            "same_gt": bool(np.array_equal(a["truth"], b["truth"], equal_nan=True)),
            "same_reference_mask": bool(np.array_equal(a["reference"], b["reference"])),
            "same_right": bool(np.array_equal(a["right"], b["right"])), "paired_pixels": int(paired.sum()),
            "paired_base_bad2": float((ea > 2).mean()), "paired_variant_bad2": float((eb > 2).mean()),
            "paired_base_mae": float(ea.mean()), "paired_variant_mae": float(eb.mean())})
    return {"checked_scenes": len(summary["scenes"]), "region_counts_verified": True,
        "severity": [{"label": item["label"], "count": int(severity_count[i]),
                      "share_of_error_sum": float(severity_sum[i] / severity_sum.sum())}
                     for i, item in enumerate(summary["severity"])],
        "sensitivity_excluding_range": [{"jump_px": jump, "radius_px": radius,
            "bad_counts_range_boundary_low_other": value.tolist(),
            "shares": (value / summary["overall"]["bad_pixels"]).tolist()} for (jump, radius), value in sensitivity.items()],
        "pairs": pair_rows, "Adirondack": adir}


def plot_overview(root: Path, summary: dict) -> None:
    """Rank erroneous-pixel shares and show severity on the same baseline domain."""
    import matplotlib.pyplot as plt
    ordered: list[str] = sorted(summary["regions"], key=lambda key: summary["regions"][key]["bad_pixels"], reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    shares: list[float] = [summary["regions"][key]["share_of_bad_pixels"] * 100 for key in ordered]
    axes[0].barh([LABELS[key] for key in ordered], shares, color=["#5279aa", "#dc8b37", "#76996a", "#a87070"])
    axes[0].invert_yaxis()
    for i, key in enumerate(ordered):
        axes[0].text(shares[i] + .6, i, f"{shares[i]:.2f}%", va="center")
    axes[0].set(xlim=(0, 65), xlabel=f"占全部 {summary['overall']['bad_pixels']:,} 个 Bad-2 错误的比例 / %", title="错误主要落在哪里（互斥区域）")
    severity: list[dict] = summary["severity"]
    fractions: list[float] = [row["pixels"] / summary["overall"]["evaluated_pixels"] * 100 for row in severity]
    axes[1].barh(["≤0.5", "0.5–1", "1–2", "2–5", "5–10", ">10"], fractions,
                 color=["#76996a"] * 3 + ["#dc8b37"] * 2 + ["#a34e4e"])
    axes[1].invert_yaxis()
    for i, fraction in enumerate(fractions):
        axes[1].text(fraction + .6, i, f"{fraction:.2f}%", va="center")
    axes[1].set(xlim=(0, 82), xlabel=f"占全部 {summary['overall']['evaluated_pixels']:,} 个可评价预测的比例 / %", ylabel="绝对视差误差 / px",
                title="误差大小：超过 2 px 定义为本次 Bad-2")
    fig.savefig(root / "error-overview.png", dpi=160)
    plt.close(fig)


def plot_examples(root: Path, summary: dict, views: dict) -> list[dict]:
    """Show real image crops; post-hoc selection never influences baseline results."""
    import matplotlib.pyplot as plt
    examples: list[dict] = []
    fig, axes = plt.subplots(len(views["examples"]), 4, figsize=(13, 12), layout="constrained")
    for row_index, example in enumerate(views["examples"]):
        name, code = example["scene"], example["region"]
        a: dict = read_arrays(root, name)
        domain: np.ndarray = a["reference"] & a["predicted_valid"] & (a["regions"] == code)
        bad: np.ndarray = domain & (a["error"] > 2)
        if example["center_xy"] is None:
            window: int = views["selection_window_px"]
            density: np.ndarray = cv2.boxFilter(bad.astype(np.float32), -1, (window, window), normalize=False,
                                              borderType=cv2.BORDER_CONSTANT)
            cy, cx = (int(value) for value in np.unravel_index(np.argmax(density), density.shape))
        else:
            cx, cy = example["center_xy"]
        radius: int = views["crop_radius_px"]
        x0, x1 = max(0, cx - radius), min(a["truth"].shape[1], cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(a["truth"].shape[0], cy + radius + 1)
        region = np.s_[y0:y1, x0:x1]
        truth: np.ndarray = np.where(np.isfinite(a["truth"][region]), a["truth"][region], np.nan)
        lo, hi = float(np.nanmin(truth)), float(np.nanmax(truth))
        context: np.ndarray = domain[region]
        n: int = int(context.sum())
        bad_n: int = int(bad[region].sum())
        title: str = f"{LABELS[REGIONS[code]]} · {name}\n裁剪内该类 {bad_n}/{n} 点为 Bad-2"
        panels: list[tuple] = [(cv2.cvtColor(a["left"][region], cv2.COLOR_BGR2RGB), None, None, "左图"),
            (truth, "turbo", (lo, hi), "视差真值 / px"),
            (a["prediction"][region], "turbo", (lo, hi), "原 SGBM 视差 / px"),
            (a["error"][region], "magma", (0, views["error_display_max_px"]), "绝对误差 / px")]
        for column, (values, cmap, bounds, label) in enumerate(panels):
            ax = axes[row_index, column]
            if cmap is None:
                ax.imshow(values)
            else:
                im = ax.imshow(np.ma.masked_invalid(values), cmap=plt.get_cmap(cmap).with_extremes(bad="0.65"),
                               vmin=bounds[0], vmax=bounds[1])
                fig.colorbar(im, ax=ax, shrink=.72, pad=.01)
            if example["center_xy"] is not None:
                ax.scatter([cx - x0], [cy - y0], s=80, facecolors="none", edgecolors="cyan", linewidths=1.7)
            ax.set_title(title if column == 0 else label, fontsize=10)
            ax.axis("off")
        examples.append({"scene": name, "region": REGIONS[code], "crop_xyxy": [x0, y0, x1, y1],
            "region_evaluated_pixels_in_crop": n, "region_bad2_pixels_in_crop": bad_n,
            "selection": "fixed earlier C pixel" if example["center_xy"] else "largest local Bad-2 count in configured scene/stratum"})
    fig.savefig(root / "error-examples.png", dpi=160)
    plt.close(fig)
    return examples


def plot_pair(root: Path, summary: dict) -> None:
    """Display the same-left/same-GT Piano pair with changed right illumination."""
    import matplotlib.pyplot as plt
    a, b = read_arrays(root, "Piano"), read_arrays(root, "PianoL")
    assert np.array_equal(a["left"], b["left"])
    assert np.array_equal(a["truth"], b["truth"], equal_nan=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for ax, values, label in zip(axes[0], [a["left"], a["right"], b["right"]], ["共同左图（逐像素相同）", "Piano：右图", "PianoL：右图，光照改变"]):
        ax.imshow(cv2.cvtColor(values, cv2.COLOR_BGR2RGB))
        ax.set_title(label)
        ax.axis("off")
    paired: np.ndarray = a["reference"] & b["reference"] & a["predicted_valid"] & b["predicted_valid"]
    first_error: np.ndarray = np.where(paired, a["error"], np.nan)
    second_error: np.ndarray = np.where(paired, b["error"], np.nan)
    first_bad: float = float((first_error[paired] > summary["settings"]["bad_threshold_px"]).mean())
    second_bad: float = float((second_error[paired] > summary["settings"]["bad_threshold_px"]).mean())
    for ax, values, cmap, bounds, label in [
        (axes[1, 0], np.where(np.isfinite(a["truth"]), a["truth"], np.nan), "turbo", (0, 80), "共同视差真值（逐像素相同）"),
        (axes[1, 1], first_error, "magma", (0, 30), f"共同有效域 · Piano：Bad-2 {first_bad:.2%}"),
        (axes[1, 2], second_error, "magma", (0, 30), f"共同有效域 · PianoL：Bad-2 {second_bad:.2%}"),
    ]:
        im = ax.imshow(np.ma.masked_invalid(values), cmap=plt.get_cmap(cmap).with_extremes(bad="0.65"),
                       vmin=bounds[0], vmax=bounds[1])
        fig.colorbar(im, ax=ax, shrink=.75, label="px")
        ax.set_title(label)
        ax.axis("off")
    fig.savefig(root / "illumination-pair.png", dpi=160)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/error_diagnosis_views.json"))
    args: argparse.Namespace = parser.parse_args(argv)
    views: dict = json.loads(args.config.read_text(encoding="utf-8"))
    root: Path = (args.config.parent / views["diagnosis_root"]).resolve()
    summary: dict = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False})
    checked: dict = audit(root, summary, views)
    plot_overview(root, summary)
    checked["examples"] = plot_examples(root, summary, views)
    plot_pair(root, summary)
    write_json(root / "independent_audit.json", checked)
    print(json.dumps({"audit": "passed", "scenes": checked["checked_scenes"], "pairs": checked["pairs"],
                      "examples": checked["examples"], "severity": checked["severity"]}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
