"""Independent fixed-test-domain audit and actual fine-tuning comparisons."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .banet import environment_dll_handle
from .error_diagnosis import write_json
from .provenance import verify_recorded_sources


def audit_test(root: Path) -> dict:
    """Recompute all held-out errors without the training/evaluation helpers."""
    report: dict = json.loads((root / "test_report.json").read_text(encoding="utf-8"))
    protocol: dict = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    assert report["identity_sha256"] == protocol["identity_sha256"]
    groups: dict = protocol["settings"]["split_groups"]
    all_groups: list[str] = [group for values in groups.values() for group in values]
    assert len(all_groups) == len(set(all_groups))
    changed_sources: list[str] = verify_recorded_sources(root, protocol["source_sha256"])
    checks: list[dict] = []
    for method in ("before", "after"):
        total: int = 0
        error_sum: float = 0.
        bad1: int = 0
        bad2: int = 0
        for row in report[method]["samples"]:
            assert row["sample"] in protocol["splits"]["test"]
            with np.load(root / f"test-{row['sample']}.npz") as arrays:
                reference: np.ndarray = arrays["reference"]
                error: np.ndarray = np.abs(arrays[method][reference].astype(np.float64) - arrays["truth"][reference].astype(np.float64))
            assert error.size == row["pixels"] and np.isfinite(error).all()
            assert int((error > 1).sum()) == row["bad1_pixels"]
            assert int((error > 2).sum()) == row["bad2_pixels"]
            np.testing.assert_allclose(error.mean(), row["mae_px"], rtol=1e-12)
            total += error.size
            error_sum += float(error.sum())
            bad1 += int((error > 1).sum())
            bad2 += int((error > 2).sum())
        expected: dict = report[method]["pooled"]
        assert total == expected["pixels"]
        np.testing.assert_allclose([error_sum / total, bad1 / total, bad2 / total],
            [expected["mae_px"], expected["bad1_fraction"], expected["bad2_fraction"]], rtol=1e-12)
        checks.append({"method": method, "samples": len(report[method]["samples"]), "pixels": int(total)})
    if (root / "fixed_last_test.json").exists():
        last: dict = json.loads((root / "fixed_last_test.json").read_text(encoding="utf-8"))
        assert last["step"] == protocol["settings"]["steps"]
        assert hashlib.sha256((root / "checkpoint_last.pth").read_bytes()).hexdigest() == last["checkpoint_sha256"]
        total, absolute_sum, bad1, bad2 = 0, 0., 0, 0
        for row in last["samples"]:
            assert row["sample"] in protocol["splits"]["test"]
            with np.load(root / f"test-{row['sample']}.npz") as arrays, np.load(root / f"last-test-{row['sample']}.npz") as prediction:
                reference = arrays["reference"]
                errors: np.ndarray = np.abs(prediction["prediction"][reference].astype(np.float64) - arrays["truth"][reference].astype(np.float64))
            assert errors.size == row["pixels"]
            assert int((errors > 1).sum()) == row["bad1_pixels"]
            assert int((errors > 2).sum()) == row["bad2_pixels"]
            np.testing.assert_allclose(errors.mean(), row["mae_px"], rtol=1e-12)
            total += errors.size
            absolute_sum += float(errors.sum())
            bad1 += int((errors > 1).sum())
            bad2 += int((errors > 2).sum())
        np.testing.assert_allclose([absolute_sum / total, bad1 / total, bad2 / total],
            [last["pooled"]["mae_px"], last["pooled"]["bad1_fraction"], last["pooled"]["bad2_fraction"]], rtol=1e-12)
        checks.append({"method": "fixed_last_secondary", "samples": len(last["samples"]), "pixels": int(total)})
    return {"passed": True, "physical_groups_disjoint": True, "source_hashes_match_run": True,
        "current_sources_changed_since_execution": changed_sources, "test_checks": checks}


def draw_sample(root: Path, name: str, error_limit: float, comparison_model: str) -> Path:
    """Original image and GT, shared disparity/error scales, fixed GT validity."""
    import matplotlib.pyplot as plt

    with np.load(root / f"test-{name}.npz") as stored:
        arrays: dict[str, np.ndarray] = {key: stored[key] for key in stored.files}
    if comparison_model == "fixed_last":
        with np.load(root / f"last-test-{name}.npz") as fixed_last:
            arrays["after"] = fixed_last["prediction"].copy()
    elif comparison_model != "validation_selected":
        raise ValueError("Unknown comparison checkpoint")
    reference: np.ndarray = arrays["reference"]
    truth: np.ndarray = np.where(reference, arrays["truth"], np.nan)
    low, high = np.percentile(truth[reference], [1, 99])
    disparity_cmap = plt.get_cmap("turbo").copy()
    disparity_cmap.set_bad("#777777")
    error_cmap = plt.get_cmap("magma").copy()
    error_cmap.set_bad("#777777")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    axes[0, 0].imshow(arrays["left"])
    axes[0, 0].set_title("Left image (held-out site)")
    disparity_image = axes[1, 0].imshow(truth, cmap=disparity_cmap, vmin=low, vmax=high)
    axes[1, 0].set_title("Ground-truth disparity")
    label_after: str = "Fixed step 1000 (secondary)" if comparison_model == "fixed_last" else "Validation-selected"
    for column, method, label in ((1, "before", "Pretrained"), (2, "after", label_after)):
        prediction: np.ndarray = np.where(reference, arrays[method], np.nan)
        error: np.ndarray = np.abs(prediction - truth)
        axes[0, column].imshow(prediction, cmap=disparity_cmap, vmin=low, vmax=high)
        axes[0, column].set_title(label + " disparity")
        error_image = axes[1, column].imshow(error, cmap=error_cmap, vmin=0, vmax=error_limit)
        axes[1, column].set_title(f"MAE {np.nanmean(error):.3f}px; Bad1 {(error[reference] > 1).mean():.2%}")
    fig.colorbar(disparity_image, ax=axes[:, 0], shrink=.65, label="Disparity (px)")
    fig.colorbar(error_image, ax=axes[:, 1:], shrink=.65, label=f"Absolute error (px, clipped at {error_limit:g})")
    for axis in axes.flat:
        axis.set_axis_off()
    fig.suptitle(f"ETH3D {name}: identical non-occluded GT pixels, no prediction rejection", fontsize=13)
    target: Path = root / f"{comparison_model}-comparison-{name}.png"
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


def main() -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/banet_finetuning_views.json"))
    args: argparse.Namespace = parser.parse_args()
    raw: dict = json.loads(args.config.read_text(encoding="utf-8"))
    root: Path = (args.config.parent / raw["result_root"]).resolve()
    audit: dict = audit_test(root)
    write_json(root / "audit.json", audit)
    dll_handle: object | None = environment_dll_handle()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    validation: list[dict] = json.loads((root / "validation_history.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for axis, field, label, scale in ((axes[0], "mae_px", "Validation MAE (px)", 1),
                                     (axes[1], "bad1_fraction", "Validation Bad1 (%)", 100)):
        axis.plot([row["step"] for row in validation], [row["pooled"][field] * scale for row in validation], "o-")
        axis.set(xlabel="Optimizer steps", ylabel=label)
        axis.grid(alpha=.25)
    fig.suptitle("Validation only: checkpoint selected by MAE, including pretrained step 0")
    fig.savefig(root / "validation-curve.png", dpi=150)
    plt.close(fig)
    paths: list[str] = [str(draw_sample(root, name, raw["error_display_max_px"], raw["comparison_model"])) for name in raw["samples"]]
    print(json.dumps({"audit": audit, "figures": paths}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
