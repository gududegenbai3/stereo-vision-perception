"""Independently audit all saved pixels, then report source-balanced results and failures."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .stereo_fair_data import load_arrays


LABELS: dict[str, str] = {"sgbm_original": "SGBM original", "sgbm_selected": "SGBM dev-selected", "banet": "BANet pretrained", "igev8": "IGEV 8", "igev32": "IGEV 32"}
SOURCES: dict[str, str] = {"middlebury2006_new": "Middlebury 2006 (new)", "eth3d_development": "ETH3D", "mobile_development": "Mobile 2021", "booster": "Booster"}


def independent_audit(root: Path, protocol: dict[str, object], summary: dict[str, object]) -> dict[str, object]:
    """Rebuild masks and counts without importing production metric/aggregation helpers."""
    raw: dict[str, object] = protocol["settings"]
    samples: list[dict[str, object]] = [row for row in protocol["samples"] if row["role"] == "evaluation"]
    grouped: dict[str, dict[str, dict[str, list[float]]]] = {method: {} for method in LABELS}
    observations: dict[str, list[dict[str, object]]] = {method: [] for method in LABELS}
    checked: int = 0
    for index, sample in enumerate(samples):
        arrays: dict[str, np.ndarray] = load_arrays(sample["path"])
        truth: np.ndarray = arrays["disparity"]
        height, width = truth.shape
        grid_x: np.ndarray = np.arange(width)[None, :]
        grid_y: np.ndarray = np.arange(height)[:, None]
        known: np.ndarray = arrays["known"] & np.isfinite(truth) & (truth > 0) & (truth < 192) & (grid_x >= truth)
        nonocc: np.ndarray = known & arrays["non_occluded"]
        fixed: np.ndarray = nonocc & (grid_x >= 194) & (grid_x < width-3) & (grid_y >= 3) & (grid_y < height-3)
        seeds: np.ndarray = np.zeros_like(known)
        horizontal: np.ndarray = arrays["known"][:, 1:] & arrays["known"][:, :-1] & (np.abs(truth[:, 1:]-truth[:, :-1]) > raw["boundary_jump_px"])
        vertical: np.ndarray = arrays["known"][1:] & arrays["known"][:-1] & (np.abs(truth[1:]-truth[:-1]) > raw["boundary_jump_px"])
        seeds[:, 1:] |= horizontal
        seeds[:, :-1] |= horizontal
        seeds[1:] |= vertical
        seeds[:-1] |= vertical
        band: np.ndarray = cv2.dilate(seeds.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        gray: np.ndarray = cv2.cvtColor(arrays["left"], cv2.COLOR_RGB2GRAY).astype(np.float32)
        mean: np.ndarray = cv2.boxFilter(gray, -1, (5, 5), borderType=cv2.BORDER_REFLECT_101)
        variance: np.ndarray = np.maximum(cv2.boxFilter(gray*gray, -1, (5, 5), borderType=cv2.BORDER_REFLECT_101)-mean*mean, 0)
        weak: np.ndarray = np.sqrt(variance) < 5
        predictions: dict[str, np.ndarray] = {method: np.load(root/f"sample-{index:03d}/{method}.npy") for method in LABELS}
        joint: np.ndarray = nonocc.copy()
        for values in predictions.values():
            joint &= np.isfinite(values)
        masks: dict[str, np.ndarray] = {"all_known": known, "non_occluded": nonocc, "fixed_roi": fixed, "joint_finite": joint,
            "boundary": fixed & band, "interior": fixed & ~band, "weak_texture": fixed & weak, "textured": fixed & ~weak}
        saved: dict[str, object] = json.loads((root/f"sample-{index:03d}/metrics.json").read_text(encoding="utf-8"))["methods"]
        for method, values in predictions.items():
            if values.dtype != np.float32 or values.shape != truth.shape:
                raise ValueError("Output dtype or pixel grid changed")
            for name, mask in masks.items():
                total: int = int(mask.sum())
                record: dict[str, object] = saved[method][name]
                assert record["reference"] == total
                if total == 0:
                    continue
                finite: np.ndarray = mask & np.isfinite(values)
                error: np.ndarray = np.abs(values[finite].astype(np.float64)-truth[finite].astype(np.float64))
                missing: int = total-error.size
                assert record["missing"] == missing
                failure: float = (int((error > 2).sum())+missing)/total
                np.testing.assert_allclose(failure, record["bad2_or_missing"], rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(error.size/total, record["coverage"], rtol=1e-12)
                if error.size:
                    np.testing.assert_allclose(error.mean(), record["mae_px"], rtol=1e-12, atol=1e-12)
                if name == "fixed_roi":
                    grouped[method].setdefault(sample["dataset"], {}).setdefault(sample["group"], []).append(failure)
            observations[method].append({"index": index, **sample, "score": saved[method]["fixed_roi"]["bad2_or_missing"], "mae": saved[method]["joint_finite"]["mae_px"]})
            checked += 1
    for method in LABELS:
        per_source: dict[str, float] = {source: float(np.mean([np.mean(values) for values in groups.values()])) for source, groups in grouped[method].items()}
        expected: dict[str, object] = summary["methods"][method]["fixed_roi"]["bad2_or_missing"]
        for source, value in per_source.items():
            np.testing.assert_allclose(value, expected["per_source"][source], rtol=1e-12)
        np.testing.assert_allclose(np.mean(list(per_source.values())), expected["source_equal_macro"], rtol=1e-12)
    selection: dict[str, object] = json.loads((root/"sgbm_selection.json").read_text())
    development_paths: set[str] = {row["path"] for row in protocol["samples"] if row["role"] == "development"}
    evaluation_paths: set[str] = {row["path"] for row in samples}
    assert set(selection["selection_inputs"]) == development_paths and not development_paths & evaluation_paths
    lock: dict[str, object] = json.loads((root/"evaluation_lock.json").read_text())
    assert lock["selection_sha256"] == hashlib.sha256((root/"sgbm_selection.json").read_bytes()).hexdigest()
    audit: dict[str, object] = {"passed": True, "samples": len(samples), "method_sample_arrays": checked, "domains_independently_checked": list(masks),
        "selection_inputs_dev_only": True, "selected_parameters_unchanged": True, "source_equal_macro_recomputed": True,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "limit": protocol["scope"]}
    (root/"independent_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (root/"per_observation.json").write_text(json.dumps(observations, indent=2), encoding="utf-8")
    return observations


def plot_case(root: Path, sample: dict[str, object], index: int, stem: str, subtitle: str) -> Path:
    import matplotlib.pyplot as plt

    arrays: dict[str, np.ndarray] = load_arrays(sample["path"])
    truth: np.ndarray = arrays["disparity"]
    height, width = truth.shape
    reference: np.ndarray = arrays["known"] & arrays["non_occluded"] & (truth > 0) & (truth < 192) & (np.arange(width)[None, :]-truth >= 0)
    scores: dict[str, object] = json.loads((root/f"sample-{index:03d}/metrics.json").read_text())["methods"]
    predictions: dict[str, np.ndarray] = {method: np.load(root/f"sample-{index:03d}/{method}.npy") for method in LABELS}
    vmax: float = float(max(8, min(192, np.ceil(np.quantile(truth[reference], .99)/8)*8)))
    fig, axes = plt.subplots(2, 5, figsize=(22, 10))
    fig.subplots_adjust(left=.02, right=.985, top=.86, bottom=.14, wspace=.12, hspace=.3)
    axes[0, 0].imshow(arrays["left"])
    axes[0, 0].set_title("Left RGB")
    panels: list[tuple] = [(axes[0, 1], np.where(arrays["known"], truth, np.nan), "GT disparity", "turbo", vmax)]
    for axis, method in ((axes[0, 2], "igev8"), (axes[0, 3], "igev32"), (axes[0, 4], "banet"), (axes[1, 0], "sgbm_original"), (axes[1, 1], "sgbm_selected")):
        failure: float = scores[method]["fixed_roi"]["bad2_or_missing"]
        panels.append((axis, predictions[method], f"{LABELS[method]}\nFixed-ROI Bad2+missing {failure:.1%}", "turbo", vmax))
    for axis, method in ((axes[1, 2], "igev32"), (axes[1, 3], "banet"), (axes[1, 4], "sgbm_selected")):
        error: np.ndarray = np.where(reference, np.abs(predictions[method]-truth), np.nan)
        panels.append((axis, error, f"{LABELS[method]} error", "magma", 5))
    for axis, values, title, cmap, maximum in panels:
        artist = axis.imshow(np.ma.masked_invalid(values), cmap=plt.get_cmap(cmap).with_extremes(bad=".65"), vmin=0, vmax=maximum)
        axis.set_title(title, fontsize=10, pad=8)
        if cmap == "turbo":
            disparity_artist = artist
        else:
            error_artist = artist
    for axis in axes.flat:
        axis.axis("off")
    fig.colorbar(disparity_artist, cax=fig.add_axes([.08, .065, .35, .018]), orientation="horizontal", label="Disparity / px; shared GT-defined scale within this figure", extend="max")
    fig.colorbar(error_artist, cax=fig.add_axes([.57, .065, .35, .018]), orientation="horizontal", label="Absolute error / px; all figures capped at 5", extend="max")
    credit: str = "Middlebury / ETH3D / Booster source attribution in report; Booster CC BY-NC 4.0"
    fig.suptitle(f"{sample['dataset']} / {sample['scene']} | {subtitle}\n{credit}. Gray: unknown GT or no output; no per-model color autoscaling.", fontsize=11)
    target: Path = root/f"{stem}.png"
    fig.savefig(target, dpi=130, bbox_inches="tight", pad_inches=.15)
    plt.close(fig)
    return target


def main() -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/stereo_fair_evaluation.json"))
    args: argparse.Namespace = parser.parse_args()
    raw: dict[str, object] = json.loads(args.config.read_text())
    root: Path = (args.config.parent/raw["output_root"]).resolve()
    protocol: dict[str, object] = json.loads((root/"protocol.json").read_text())
    summary: dict[str, object] = json.loads((root/"summary.json").read_text())
    observations: dict = independent_audit(root, protocol, summary)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    samples: list[dict[str, object]] = [row for row in protocol["samples"] if row["role"] == "evaluation"]
    view_links: list[str] = []
    for number, view in enumerate(protocol["random_views"], 1):
        index: int = next(i for i, row in enumerate(samples) if row["path"] == view["path"])
        target: Path = plot_case(root, samples[index], index, f"random-{number:02d}-{samples[index]['scene']}", "group drawn before scoring; first observation")
        view_links.append(f"![{view['group']}]({target.as_posix()})")
    # Separately expose failures using declared outcome-based selection, never replace random views.
    worst: dict = max(observations["igev32"], key=lambda row: (row["score"], -row["index"]))
    gap_rows: list[dict] = [{**row, "gap": row["score"]-observations["sgbm_selected"][row["index"]]["score"]} for row in observations["igev32"]]
    loss: dict = max(gap_rows, key=lambda row: (row["gap"], -row["index"]))
    failures: list[str] = []
    for label, row in (("worst-igev", worst), ("largest-loss-to-sgbm", loss)):
        target = plot_case(root, samples[row["index"]], row["index"], label, "FAILURE DIAGNOSIS: selected by declared highest error / largest loss")
        failures.append(f"![{label}]({target.as_posix()})")
    sources: list[str] = raw["evaluation_datasets"]
    fig, axis = plt.subplots(figsize=(11, 5), layout="constrained")
    for offset, (method, label) in enumerate(LABELS.items()):
        values: list[float] = [summary["methods"][method]["fixed_roi"]["bad2_or_missing"]["per_source"][source] for source in sources]
        axis.bar(np.arange(len(sources))+(offset-2)*.15, values, width=.15, label=label)
    axis.set_xticks(range(len(sources)), [SOURCES[source] for source in sources], rotation=10)
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.set_ylabel("Bad2 + missing, fixed prediction-independent ROI")
    axis.set_title("Within each source: observations -> conservative groups -> equal group mean")
    axis.legend(fontsize=8)
    fig.savefig(root/"source-comparison.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(6, 6), layout="constrained")
    for source in sources:
        x: dict = summary["methods"]["sgbm_selected"]["fixed_roi"]["bad2_or_missing"]["per_group"][source]
        y: dict = summary["methods"]["igev32"]["fixed_roi"]["bad2_or_missing"]["per_group"][source]
        axis.scatter([x[group] for group in sorted(x)], [y[group] for group in sorted(x)], label=SOURCES[source], alpha=.8)
    axis.plot([0, 1], [0, 1], color="gray", linestyle="--")
    axis.set(xlabel="Development-selected SGBM failure rate", ylabel="IGEV32 failure rate", title="60 paired components; below diagonal: IGEV32 better", xlim=(0, 1), ylim=(0, 1))
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.legend(fontsize=8)
    fig.savefig(root/"paired-components.png", dpi=160)
    plt.close(fig)
    timing: dict = json.loads((root/"timing.json").read_text())
    fig, axis = plt.subplots(figsize=(8, 4), layout="constrained")
    for index, (method, label) in enumerate(LABELS.items()):
        row: dict = timing["methods"][method]
        axis.barh(index, row["median_seconds"]*1000)
        axis.errorbar(row["median_seconds"]*1000, index, xerr=np.array([[row["median_seconds"]-row["q25_q75_seconds"][0]], [row["q25_q75_seconds"][1]-row["median_seconds"]]])*1000, color="black", capsize=3)
    axis.set_yticks(range(len(LABELS)), list(LABELS.values()))
    axis.set(xlabel="End-to-end latency / ms, median and IQR", title="Same 256x512 crop, 3 warmup + 10 repeats; CPU SGBM / GPU networks")
    fig.savefig(root/"latency.png", dpi=160)
    plt.close(fig)
    lines: list[str] = ["# 预训练模型重新评测", "", "模型不训练，使用固定官方Scene Flow权重。15对Middlebury2014只用于13套SGBM设置选参；181对正式评价含本轮新下载的21对Middlebury2006，以及已查看的ETH3D27、Mobile2021 24、Booster109。60个保守关联组不是60个已证实独立房间。新增源未用于本项目调参；作者历史预训练资料无法完全核实，不宣称绝对无重叠。", "", "主指标为非遮挡GT、固定搜索ROI内的误差>2px或缺失比例。ROI由192搜索范围和最大候选block7决定，与任何预测值无关；失败输出不会被删掉。先观察等权到组、组等权到来源，最后四来源等权。其他域/阈值、共同有限像素MAE、边界/内部/弱纹理均在summary与逐图CSV中。", "", "| 数据来源 | 图像对/组 | SGBM原版 | 开发集选参SGBM | BANet | IGEV8 | IGEV32 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for source in sources:
        counts: dict = summary["datasets"][source]
        points: list[str] = [f"{summary['methods'][method]['fixed_roi']['bad2_or_missing']['per_source'][source]:.2%}" for method in LABELS]
        lines.append(f"| {SOURCES[source]} | {counts['pairs']}/{counts['groups']} | "+" | ".join(points)+" |")
    lines.append("| 四来源等权 | 181/60 | "+" | ".join(f"{summary['methods'][method]['fixed_roi']['bad2_or_missing']['source_equal_macro']:.2%}" for method in LABELS)+" |")
    lines.extend(["", "## 不同评价域的敏感性", "", "下表均先按组、再按来源等权；完整非遮挡域保留左边缘等原SGM未输出位置，全部已知域另包含已知遮挡点。共同有限域MAE只作辅助，不能用它掩盖缺失。主表和本表是不同口径，不能直接与上轮仅Booster的像素池化率相减。", "", "| 方法 | 完整非遮挡坏点含缺失 | 全部已知坏点含缺失 | 完整非遮挡覆盖率 | 共同有限域MAE/px |", "|---|---:|---:|---:|---:|"])
    for method, label in LABELS.items():
        row: dict = summary["methods"][method]
        lines.append(f"| {label} | {row['non_occluded']['bad2_or_missing']['source_equal_macro']:.2%} | {row['all_known']['bad2_or_missing']['source_equal_macro']:.2%} | {row['non_occluded']['coverage']['source_equal_macro']:.2%} | {row['joint_finite']['mae_px']['source_equal_macro']:.4f} |")
    lines.extend(["", "| 来源的共同有限域MAE/px | 原SGM | 选参SGM | BANet | IGEV8 | IGEV32 |", "|---|---:|---:|---:|---:|---:|"])
    for source in sources:
        lines.append(f"| {SOURCES[source]} | "+" | ".join(f"{summary['methods'][method]['joint_finite']['mae_px']['per_source'][source]:.4f}" for method in LABELS)+" |")
    lines.extend(["", "## 边界、内部与纹理", "", "固定ROI内分别统计；边界与内部互斥，弱纹理与边界可能重叠，不能把两种比例相加当错误贡献。无有效参考点的组不参与相应层的均值，各层分组明细在summary中。", "", "| 方法 | 边界坏点含缺失 | 内部坏点含缺失 | 弱纹理坏点含缺失 | 纹理丰富坏点含缺失 |", "|---|---:|---:|---:|---:|"])
    for method, label in LABELS.items():
        points: list[str] = []
        for domain in ("boundary", "interior", "weak_texture", "textured"):
            value: float | None = summary["methods"][method][domain]["bad2_or_missing"]["source_equal_macro"]
            points.append(f"{value:.2%}" if value is not None else "无参考点")
        lines.append(f"| {label} | "+" | ".join(points)+" |")
    lines.extend(["", "![按来源结果]("+(root/"source-comparison.png").as_posix()+")", "", "![分组胜负]("+(root/"paired-components.png").as_posix()+")", "", "## 配对统计", "", "差值为IGEV32减对照，负值为IGEV32更低。区间为按来源分层的组重采样描述区间，不是对所有未知场景的保证。", ""])
    for method, row in summary["paired_igev32_minus_other"].items():
        lines.append(f"- 对{LABELS[method]}：差{row['difference_source_equal_macro']*100:.2f}个百分点，95%区间[{row['component_bootstrap_95'][0]*100:.2f}, {row['component_bootstrap_95'][1]*100:.2f}]；60组赢/输/平={row['component_wins']}/{row['component_losses']}/{row['component_ties']}。")
    lines.extend(["", "## 相同输入计时", "", "![计时]("+(root/"latency.png").as_posix()+")", "", "速度与精度分别测量：计时为同一256×512开发图裁剪，精度使用各来源规定输入尺度。比较同机CPU与GPU实际耗时，不能据此推断纯算法FLOPs效率。", "", "所有误差为各输入尺度的视差像素，不是相机测距的米制精度。小视差场景的2px阈值较宽，需同时看共同域MAE与完整JSON中的Bad1；真实硬件精度仍需独立标定/测距。", "", "## 评分前固定的八组图", "", "视差颜色上限仅由本例GT定义，同一图中各算法共用；误差颜色统一0–5px并截断。灰色是未知GT或无输出，不代表正确。图内主指标为固定ROI坏点含缺失比例。", "", *view_links, "", "## 显式失败案例", "", "以下两例按IGEV32最高错误率、相对选参SGBM最大劣势选取，属于失败诊断；没有替换上面的随机图，也没有用于调参。", "", *failures, "", "来源：[Middlebury2006及编码说明](https://vision.middlebury.edu/stereo/data/scenes2006/)、[ETH3D](https://www.eth3d.net/datasets)、[Middlebury2021](https://vision.middlebury.edu/stereo/data/scenes2021/)、[Booster](https://amsacta.unibo.it/id/eprint/6876/)（Zama Ramirez等，CC BY-NC 4.0）。Middlebury引用Scharstein/Pal与Hirschmüller/Scharstein 2007。", "", "证据：protocol.json、image_audit.json、sgbm_selection.json、evaluation_lock.json、summary.json、per_sample.csv、timing.json、independent_audit.json及原始预测数组。全部数值源文件与模型哈希在执行前冻结。"])
    (root/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"audit_passed": True, "random_views": len(view_links), "failure_views": len(failures), "report": str(root/"report.md")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
