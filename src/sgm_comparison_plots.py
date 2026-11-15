"""Audit comparison arrays and plot predefined examples with common color scales."""

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

import cv2
import matplotlib
import numpy as np


from .error_diagnosis import write_json

LABELS: dict[str,str] = {"opencv_baseline":"原 OpenCV SGBM", "opencv_options":"OpenCV 8路径＋筛选", "opencv_range":"按标定提示扩搜索范围",
    "sgm_fixed":"像素 SGM：固定 P2", "sgm_adaptive":"像素 SGM：自适应 P2", "sgm_matched_mean":"像素 SGM：平均惩罚固定",
    "sgm_fixed_raw":"固定 P2：未筛选", "sgm_adaptive_raw":"自适应 P2：未筛选", "sgm_matched_mean_raw":"平均惩罚固定：未筛选"}


def load_arrays(root: Path, scene: str) -> dict[str,np.ndarray]:
    with np.load(root/scene/"arrays.npz") as saved:
        return {key:saved[key] for key in saved.files}


def verify(root: Path, summary: dict, diagnosis_root: Path | None = None) -> dict:
    """Recompute own/paired metrics and unchanged original predictions from arrays."""
    methods: list[str] = list(summary["methods"])
    diagnosis_root = diagnosis_root if diagnosis_root is not None else root.parent / "error_diagnosis"
    counts: dict[str,list[float]] = {method:[0,0,0,0] for method in methods}
    outcomes: dict[str,dict[str,int]] = {method:{"improved":0,"worsened":0,"equal":0,"lost":0,"gained":0} for method in methods}
    paired_control: dict[str,int] = {"improved":0,"worsened":0,"equal":0}
    for row in summary["scenes"]:
        a: dict = load_arrays(root,row["scene"])
        ref: np.ndarray = a["reference"]
        base_valid: np.ndarray = ref & a["opencv_baseline_valid"]
        with np.load(diagnosis_root/row["scene"]/"arrays.npz") as previous:
            np.testing.assert_array_equal(a["opencv_baseline"],previous["prediction"])
        for method in methods:
            mask: np.ndarray = a[method+"_valid"]
            assert mask.dtype == np.bool_ and np.isfinite(a[method][mask]).all()
            common: np.ndarray = ref & mask
            error: np.ndarray = np.abs(a[method][common].astype(np.float64)-a["truth"][common])
            metrics: dict = row["metrics"][method]
            assert int(common.sum()) == metrics["evaluated_pixels"]
            assert int((error>2).sum()) == metrics["bad_pixels"]
            np.testing.assert_allclose(error.mean(),metrics["mae_px"],rtol=1e-12)
            counts[method][0] += common.sum()
            counts[method][1] += (error>2).sum()
            counts[method][2] += error.sum()
            counts[method][3] += ref.sum()
            paired: np.ndarray = base_valid & mask
            e0: np.ndarray = np.abs(a["opencv_baseline"][paired].astype(np.float64)-a["truth"][paired])
            e1: np.ndarray = np.abs(a[method][paired].astype(np.float64)-a["truth"][paired])
            np.testing.assert_allclose(e0.mean(),metrics["paired_baseline"]["baseline"]["mae_px"],rtol=1e-12)
            np.testing.assert_allclose(e1.mean(),metrics["paired_baseline"]["method"]["mae_px"],rtol=1e-12)
            outcome: dict = outcomes[method]
            outcome["improved"] += int((e1<e0-1e-6).sum())
            outcome["worsened"] += int((e1>e0+1e-6).sum())
            outcome["equal"] += int((np.abs(e1-e0)<=1e-6).sum())
            outcome["lost"] += int((base_valid & ~mask).sum())
            outcome["gained"] += int((common & ~a["opencv_baseline_valid"]).sum())
        paired: np.ndarray = ref & a["sgm_fixed_raw_valid"] & a["sgm_adaptive_raw_valid"]
        e0 = np.abs(a["sgm_fixed_raw"][paired].astype(np.float64)-a["truth"][paired])
        e1 = np.abs(a["sgm_adaptive_raw"][paired].astype(np.float64)-a["truth"][paired])
        paired_control["improved"] += int((e1<e0-1e-6).sum())
        paired_control["worsened"] += int((e1>e0+1e-6).sum())
        paired_control["equal"] += int((np.abs(e1-e0)<=1e-6).sum())
    for method,(n,bad,total,ref_n) in counts.items():
        metrics: dict = summary["methods"][method]
        np.testing.assert_allclose([n,bad,total/n,n/ref_n,(bad+ref_n-n)/ref_n],
            [metrics["evaluated_pixels"],metrics["bad_pixels"],metrics["mae_px"],metrics["coverage"],metrics["bad_with_missing_fraction"]],rtol=1e-12)
    return {"scenes":len(summary["scenes"]),"methods":len(methods),"all_metrics_recomputed":True,
        "original_predictions_identical_to_diagnosis":True,"outcomes_vs_opencv":outcomes,"raw_adaptive_vs_fixed":paired_control}


def render(root: Path, spec: dict, error_max: float) -> None:
    """RGB and GT at left; before/after predictions and errors align in columns."""
    import matplotlib.pyplot as plt
    a: dict = load_arrays(root,spec["scene"])
    region = np.s_[:,:] if spec["crop_xyxy"] is None else np.s_[spec["crop_xyxy"][1]:spec["crop_xyxy"][3],spec["crop_xyxy"][0]:spec["crop_xyxy"][2]]
    truth: np.ndarray = np.where(np.isfinite(a["truth"][region]),a["truth"][region],np.nan)
    lo,hi = float(np.nanmin(truth)),float(np.nanmax(truth))
    before,after = spec["before"],spec["after"]
    fig,axes = plt.subplots(2,3,figsize=(12,7.2),layout="constrained")
    axes[0,0].imshow(cv2.cvtColor(a["left"][region],cv2.COLOR_BGR2RGB))
    axes[0,0].set_title(spec["scene"]+"：左图")
    im = axes[1,0].imshow(np.ma.masked_invalid(truth),cmap=plt.get_cmap("turbo").with_extremes(bad="0.65"),vmin=lo,vmax=hi)
    fig.colorbar(im,ax=axes[1,0],shrink=.72,label="px")
    axes[1,0].set_title("视差真值")
    for column,method in enumerate((before,after),start=1):
        pred: np.ndarray = a[method][region]
        ref: np.ndarray = a["reference"][region]
        error: np.ndarray = np.where(ref,np.abs(pred-truth),np.nan)
        im = axes[0,column].imshow(np.ma.masked_invalid(pred),cmap=plt.get_cmap("turbo").with_extremes(bad="0.65"),vmin=lo,vmax=hi)
        fig.colorbar(im,ax=axes[0,column],shrink=.72,label="px")
        axes[0,column].set_title(LABELS[method])
        im = axes[1,column].imshow(np.ma.masked_invalid(error),cmap=plt.get_cmap("magma").with_extremes(bad="0.65"),vmin=0,vmax=error_max)
        fig.colorbar(im,ax=axes[1,column],shrink=.72,label="px",extend="max")
        axes[1,column].set_title("绝对误差；灰色 = 无有效评价")
    for ax in axes.flat:
        ax.axis("off")
    fig.savefig(root/spec["filename"],dpi=150)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("config/sgm_comparison_views.json"))
    args: argparse.Namespace = parser.parse_args(argv)
    views: dict = json.loads(args.config.read_text(encoding="utf-8"))
    root: Path = (args.config.parent/views["result_root"]).resolve()
    summary: dict = json.loads((root/"summary.json").read_text(encoding="utf-8"))
    checked: dict = verify(root,summary,(args.config.parent/views["diagnosis_root"]).resolve())
    plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","DejaVu Sans"],"axes.unicode_minus":False})
    for figure in views["figures"]:
        render(root,figure,views["error_display_max_px"])
    write_json(root/"independent_audit.json",checked)
    with (root/"metrics.csv").open("w",newline="",encoding="utf-8-sig") as handle:
        columns: list[str] = ["method","mae_px","bad_fraction","coverage","bad_with_missing_fraction","mean_time_ms"]
        writer = csv.DictWriter(handle,fieldnames=columns)
        writer.writeheader()
        for name,row in summary["methods"].items():
            writer.writerow({key:name if key=="method" else row[key] for key in columns})
    print(json.dumps(checked),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
