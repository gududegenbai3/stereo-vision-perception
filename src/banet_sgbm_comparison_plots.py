"""Audit all saved comparison pixels and display the pre-drawn physical scenes."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from .banet import environment_dll_handle
from .config import load_offline_config
from .error_diagnosis import write_json
from .provenance import verify_recorded_sources
from .stereo_comparison_data import choose_random_views


def assert_metrics(truth: np.ndarray, prediction: np.ndarray, domain: np.ndarray, row: dict, threshold: float) -> np.ndarray:
    """Float64 independent count/sum audit on an explicit, fixed pixel domain."""
    valid: np.ndarray = domain & np.isfinite(prediction)
    errors: np.ndarray = np.abs(prediction[valid].astype(np.float64) - truth[valid].astype(np.float64))
    assert int(domain.sum()) == row["reference_pixels"]
    assert errors.size == row["evaluated_pixels"]
    assert int(domain.sum()) - errors.size == row["missing_pixels"]
    assert int((errors > threshold).sum()) == row["bad_pixels"]
    np.testing.assert_allclose(float(errors.sum()), row["absolute_error_sum_px"], rtol=1e-12)
    if errors.size:
        np.testing.assert_allclose(float(errors.mean()), row["mae_px"], rtol=1e-12)
    return errors


def audit_results(root: Path, config_path: Path) -> dict:
    """Verify source/model identity, all48 prediction metrics and old baseline pixels."""
    protocol: dict = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    summary: dict = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    settings: dict = protocol["settings"]
    assert summary["scene_count"] == 48 and summary["failed_scenes"] == 0
    changed_sources: list[str] = verify_recorded_sources(root, protocol["source_sha256"])
    model_path: Path = config_path.parent / settings["model_config"]
    model_config: dict = json.loads(model_path.read_text(encoding="utf-8"))
    digest: str = hashlib.sha256((model_path.parent / model_config["checkpoint"]).read_bytes()).hexdigest()
    assert digest == model_config["checkpoint_sha256"] == summary["checkpoint_sha256"]
    names: list[str] = [row["name"] for row in protocol["sample_inventory"] if row["dataset"] == "mobile2021_new"]
    assert choose_random_views(names, settings["random_seed"], settings["random_physical_scene_count"]) == summary["random_views"]
    mobile_root: Path = (config_path.parent / settings["mobile_root"]).resolve()
    archive_path: Path = (config_path.parent / settings["mobile_archive"]).resolve()
    member_count: int = 0
    with zipfile.ZipFile(archive_path) as archive:
        for name in names:
            for filename in ("im0.png", "im1.png", "disp0.pfm", "disp1.pfm", "calib.txt"):
                expected: str = hashlib.sha256(archive.read(f"data/{name}/{filename}")).hexdigest()
                assert hashlib.sha256((mobile_root / name / filename).read_bytes()).hexdigest() == expected
                member_count += 1
    severity: dict = {dataset: {method: {"pixels": 0, "over10": 0, "over20": 0, "error_sum": 0., "over10_error_sum": 0.}
                                for method in ("sgbm192", "banet")} for dataset in summary["datasets"]}
    checked: int = 0
    old_baseline_checked: int = 0
    for row in summary["scenes"]:
        with np.load(root / row["dataset"] / row["sample"] / "arrays.npz") as stored:
            arrays: dict[str, np.ndarray] = {key: stored[key] for key in stored.files}
        reference: np.ndarray = arrays["reference"]
        truth: np.ndarray = arrays["truth"]
        for method in ("sgbm128", "sgbm192", "banet"):
            np.testing.assert_array_equal(np.isfinite(arrays[method]), arrays[method + "_valid"])
            own: dict = row["metrics"]["methods"][method]
            assert_metrics(truth, arrays[method], reference, own, settings["bad_threshold_native_px"])
            assert own["bad_or_missing_fraction"] == (own["bad_pixels"] + own["missing_pixels"]) / own["reference_pixels"]
        for sgbm in ("sgbm128", "sgbm192"):
            common: np.ndarray = reference & arrays[sgbm + "_valid"] & arrays["banet_valid"]
            for method in (sgbm, "banet"):
                errors: np.ndarray = assert_metrics(truth, arrays[method], common, row["metrics"]["pairs"][sgbm][method], settings["bad_threshold_native_px"])
                if sgbm == "sgbm192":
                    bucket: dict = severity[row["dataset"]][method]
                    bucket["pixels"] += errors.size
                    bucket["over10"] += int((errors > 10).sum())
                    bucket["over20"] += int((errors > 20).sum())
                    bucket["error_sum"] += float(errors.sum())
                    bucket["over10_error_sum"] += float(errors[errors > 10].sum())
                for code, label in ((1, "boundary"), (2, "low_texture"), (3, "other")):
                    assert_metrics(truth, arrays[method], common & (arrays["regions"] == code),
                                   row["metrics"]["pairs"][sgbm]["regions"][label][method], settings["bad_threshold_native_px"])
        if row["dataset"] == "middlebury_previous_development":
            with np.load(Path("artifacts/error_diagnosis") / row["sample"] / "arrays.npz") as old:
                np.testing.assert_array_equal(arrays["sgbm128"], old["prediction"])
                np.testing.assert_array_equal(arrays["left"], old["left"])
                np.testing.assert_array_equal(truth, old["truth"])
            old_baseline_checked += 1
        checked += 1
    for dataset in severity.values():
        for values in dataset.values():
            values["over10_fraction"] = values["over10"] / values["pixels"]
            values["over20_fraction"] = values["over20"] / values["pixels"]
            values["over10_share_of_absolute_error"] = values["over10_error_sum"] / values["error_sum"]
    write_json(root / "error_severity.json", severity)
    return {"passed": True, "pairs_checked": checked, "original_sgbm128_baselines_identical": old_baseline_checked,
        "archive_members_identical": member_count, "model_hash_unchanged": True, "source_hashes_match": True,
        "current_sources_changed_since_execution": changed_sources,
        "random_draw_reproduced": True, "all_metrics_recomputed_from_arrays": True}


def draw_scene(root: Path, name: str, index: int, error_input_limit: float) -> Path:
    """Two rows: left/GT, SGBM disparity/error, BANet disparity/error; shared scales."""
    import matplotlib.pyplot as plt

    folder: Path = root / "mobile2021_new" / name
    with np.load(folder / "arrays.npz") as stored:
        arrays: dict[str, np.ndarray] = {key: stored[key] for key in stored.files}
    row: dict = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    reference: np.ndarray = arrays["reference"]
    common: np.ndarray = reference & arrays["sgbm192_valid"] & arrays["banet_valid"]
    truth: np.ndarray = np.where(reference, arrays["truth"], np.nan)
    low, high = np.percentile(truth[reference], [1, 99])
    error_limit: float = error_input_limit / row["scale"]
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("#999999")
    error_cmap = plt.get_cmap("magma").copy()
    error_cmap.set_bad("#999999")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.1), layout="constrained")
    axes[0, 0].imshow(arrays["left"][..., ::-1])
    axes[0, 0].set_title("原始左图")
    disparity_image = axes[1, 0].imshow(truth, cmap=cmap, vmin=low, vmax=high, interpolation="nearest")
    axes[1, 0].set_title("视差真值")
    for column, method, label in ((1, "sgbm192", "SGBM（192范围）"), (2, "banet", "微调 BANet（固定1000步）")):
        prediction: np.ndarray = np.where(reference & arrays[method + "_valid"], arrays[method], np.nan)
        axes[0, column].imshow(prediction, cmap=cmap, vmin=low, vmax=high, interpolation="nearest")
        coverage: float = row["metrics"]["methods"][method]["coverage"]
        axes[0, column].set_title(f"{label}\n输出覆盖 {coverage:.1%}")
        error: np.ndarray = np.where(common, np.abs(arrays[method] - arrays["truth"]), np.nan)
        error_image = axes[1, column].imshow(error, cmap=error_cmap, vmin=0, vmax=error_limit, interpolation="nearest")
        metrics: dict = row["metrics"]["pairs"]["sgbm192"][method]
        axes[1, column].set_title(f"同一批像素误差：MAE {metrics['mae_px']:.2f}px\n误差>2px：{metrics['bad_fraction']:.1%}")
    fig.colorbar(disparity_image, ax=axes[:, 0], shrink=.65, label="原图视差 / px（同一色标）")
    fig.colorbar(error_image, ax=axes[:, 1:], shrink=.65, label=f"原图绝对误差 / px（封顶{error_limit:g}）")
    for axis in axes.flat:
        axis.set_axis_off()
    fig.suptitle(f"随机不同场景 {index}/6：{name}\n灰色=不评价或无输出；下排两张误差图比较完全相同的像素", fontsize=13)
    target: Path = root / f"random-{index:02d}-{name}.png"
    fig.savefig(target, dpi=145)
    plt.close(fig)
    return target


def main() -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/banet_sgbm_comparison.json"))
    args: argparse.Namespace = parser.parse_args()
    raw: dict = json.loads(args.config.read_text(encoding="utf-8"))
    root: Path = (args.config.parent / raw["output_root"]).resolve()
    audit: dict = audit_results(root, args.config)
    write_json(root / "independent_audit.json", audit)
    dll_handle: object | None = environment_dll_handle()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    protocol: dict = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    paths: list[str] = [str(draw_scene(root, name, index, raw["error_display_input_px"]))
                        for index, name in enumerate(protocol["random_views"], start=1)]
    write_json(root / "random_figures.json", {"seed": raw["random_seed"], "views": protocol["random_views"], "files": paths})
    print(json.dumps({"audit": audit, "figures": paths}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
