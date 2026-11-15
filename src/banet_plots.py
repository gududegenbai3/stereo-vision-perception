"""Independently audit saved BANet comparisons and show successes and failures."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .banet import environment_dll_handle
from .error_diagnosis import write_json


def audit_arrays(root: Path, baseline_root: Path) -> dict:
    """Recount float32 (H,W) px arrays with float64 errors and original GT masks.

    Raises AssertionError for numerical/count drift or a changed baseline prediction.
    The audit never feeds GT or its masks into model inference.
    """
    summary: dict = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    checked: list[str] = []
    for row in summary["scenes"]:
        name: str = row["scene"]
        with np.load(root / name / "arrays.npz") as arrays, np.load(baseline_root / name / "arrays.npz") as original:
            np.testing.assert_array_equal(arrays["baseline"], original["prediction"])
            np.testing.assert_array_equal(arrays["baseline_valid"], original["predicted_valid"])
            np.testing.assert_array_equal(arrays["reference"], original["reference"])
            for method in ("baseline", "same_range", "banet"):
                reference: np.ndarray = arrays["reference"]
                valid: np.ndarray = reference & arrays[method + "_valid"]
                error: np.ndarray = np.abs(arrays[method][valid].astype(np.float64) - arrays["truth"][valid].astype(np.float64))
                recorded: dict = row["metrics"][method]
                assert int(valid.sum()) == recorded["evaluated_pixels"]
                assert int((reference & ~valid).sum()) == recorded["missing_pixels"]
                assert int((error > 2).sum()) == recorded["bad_pixels"]
                np.testing.assert_allclose(error.mean(), recorded["mae_px"], rtol=1e-12)
                common: np.ndarray = valid & arrays["banet_valid"]
                for side, prediction in (("method", method), ("banet", "banet")):
                    paired: np.ndarray = np.abs(arrays[prediction][common].astype(np.float64) - arrays["truth"][common].astype(np.float64))
                    saved: dict = recorded["paired_with_banet"][side]
                    assert int(common.sum()) == saved["evaluated_pixels"]
                    assert int((paired > 2).sum()) == saved["bad_pixels"]
                    np.testing.assert_allclose(paired.mean(), saved["mae_px"], rtol=1e-12)
        checked.append(name)
    return {"passed": True, "scenes": checked, "checks": [
        "original SGBM arrays and masks unchanged", "own-domain counts and float64 MAE",
        "Bad2 strictly greater than 2px", "identical-pixel paired comparisons"]}


def draw_comparison(root: Path, view: dict, error_limit: float) -> Path:
    """Plot actual RGB, GT, predictions and errors; gray denotes missing output."""
    import matplotlib.pyplot as plt

    name: str = view["scene"]
    with np.load(root / name / "arrays.npz") as stored:
        arrays: dict[str, np.ndarray] = {key: stored[key] for key in stored.files}
    report: dict = json.loads((root / name / "report.json").read_text(encoding="utf-8"))
    crop: list[int] | None = view["crop_xyxy"]
    if crop is not None:
        x0, y0, x1, y1 = crop
        arrays = {key: value[y0:y1, x0:x1] for key, value in arrays.items()}
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 7.3), layout="constrained")
    reference: np.ndarray = arrays["reference"]
    truth: np.ndarray = np.where(reference, arrays["truth"], np.nan)
    finite: np.ndarray = truth[np.isfinite(truth)]
    low, high = np.percentile(finite, [1, 99])
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("#777777")
    error_cmap = plt.get_cmap("magma").copy()
    error_cmap.set_bad("#777777")
    axes[0, 0].imshow(arrays["left"][..., ::-1])
    axes[0, 0].set_title("Left image" + (" (crop)" if crop else ""))
    image = axes[1, 0].imshow(truth, cmap=cmap, vmin=low, vmax=high, interpolation="nearest")
    axes[1, 0].set_title("GT disparity (non-occluded)")
    fig.colorbar(image, ax=axes[:, 0], shrink=.65, label="Disparity (px)")
    for column, method in enumerate(("baseline", "banet"), start=1):
        valid: np.ndarray = reference & arrays[method + "_valid"]
        prediction: np.ndarray = np.where(valid, arrays[method], np.nan)
        error: np.ndarray = np.abs(prediction - truth)
        label: str = "Original SGBM" if method == "baseline" else "BANet-2D"
        axes[0, column].imshow(prediction, cmap=cmap, vmin=low, vmax=high, interpolation="nearest")
        axes[0, column].set_title(label + " disparity")
        error_image = axes[1, column].imshow(error, cmap=error_cmap, vmin=0, vmax=error_limit, interpolation="nearest")
        local_mae: float = float(np.nanmean(error))
        coverage: float = float(valid.sum() / reference.sum())
        axes[1, column].set_title(f"Error: MAE {local_mae:.2f}px; coverage {coverage:.1%}")
    fig.colorbar(error_image, ax=axes[:, 1:], shrink=.65, label=f"Absolute error (px; clipped at {error_limit:g})")
    for axis in axes.flat:
        axis.set_axis_off()
    original: dict = report["metrics"]["baseline"]
    banet: dict = report["metrics"]["banet"]
    fig.suptitle(f"{name} | whole-scene MAE: SGBM {original['mae_px']:.2f}px / BANet {banet['mae_px']:.2f}px\n"
                 "Same disparity/error scales; gray = missing prediction or excluded GT", fontsize=13)
    target: Path = root / f"comparison-{name.lower()}.png"
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/banet_views.json"))
    args: argparse.Namespace = parser.parse_args(argv)
    raw: dict = json.loads(args.config.read_text(encoding="utf-8"))
    root: Path = (args.config.parent / raw["result_root"]).resolve()
    baseline_root: Path = (args.config.parent / raw["baseline_diagnosis_root"]).resolve()
    audit: dict = audit_arrays(root, baseline_root)
    write_json(root / "audit.json", audit)
    dll_handle: object | None = environment_dll_handle()
    import matplotlib
    matplotlib.use("Agg")
    paths: list[str] = [str(draw_comparison(root, view, float(raw["error_display_max_px"]))) for view in raw["figures"]]
    print(json.dumps({"audit_passed": audit["passed"], "scenes_checked": len(audit["scenes"]), "figures": paths}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
