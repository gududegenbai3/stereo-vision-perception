"""Predefined traditional-SGM comparison; input-only inference, post-hoc GT metrics."""

import argparse
import json
import platform
import time
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from .config import load_offline_config
from .datasets import load_middlebury_scene
from .disparity import compute_disparity
from .error_diagnosis import DiagnosisConfig, REGIONS, classify_regions, measure, pool, texture_std, write_json
from .model import DisparityResult, RectificationGeometry, RectifiedPair, SgbmConfig
from .sgm import SgmSettings, compute_sgm_disparity
from .provenance import file_hashes, save_source_snapshot

METHODS: tuple[str, ...] = ("opencv_baseline", "opencv_options", "opencv_range", "sgm_fixed", "sgm_adaptive", "sgm_matched_mean")


def opencv_variant(pair: RectifiedPair, geometry: RectificationGeometry, base: SgbmConfig,
                   options: dict[str,int], search_hint: int, range_only: bool) -> DisparityResult:
    """Explicit OpenCV controls; no spatially adaptive P2 is claimed."""
    disparities: int = max(base.num_disparities, int(np.ceil(search_hint / 16)) * 16) if range_only else base.num_disparities
    roi = cv2.getValidDisparityROI(geometry.roi_left,geometry.roi_right,base.min_disparity,disparities,base.block_size)
    x,y,w,h = roi
    if min(w,h) <= 0:
        raise ValueError("Empty OpenCV variant ROI")
    kwargs: dict = {} if range_only else dict(uniquenessRatio=options["uniqueness_ratio"],speckleWindowSize=options["speckle_window_size"],
        speckleRange=options["speckle_range"],disp12MaxDiff=options["disp12_max_diff"],preFilterCap=options["pre_filter_cap"])
    matcher = cv2.StereoSGBM_create(minDisparity=base.min_disparity,numDisparities=disparities,blockSize=base.block_size,
        P1=8*base.block_size**2,P2=32*base.block_size**2,mode=cv2.STEREO_SGBM_MODE_SGBM if range_only else cv2.STEREO_SGBM_MODE_HH,**kwargs)
    raw: np.ndarray = matcher.compute(cv2.cvtColor(pair.left_bgr,cv2.COLOR_BGR2GRAY),cv2.cvtColor(pair.right_bgr,cv2.COLOR_BGR2GRAY))
    valid: np.ndarray = np.zeros(raw.shape,bool)
    valid[y:y+h,x:x+w] = True
    valid &= (raw > (base.min_disparity-1)*16) & (raw < (base.min_disparity+disparities)*16)
    values: np.ndarray = raw.astype(np.float32)/16
    values[~valid] = np.nan
    return DisparityResult(values,valid,roi)


def evaluate_methods(results: dict[str,DisparityResult], truth: np.ndarray, reference: np.ndarray,
                     regions: np.ndarray, threshold: float) -> dict:
    """Separate own-domain metrics, identical-pixel comparisons, and missing-as-bad."""
    metrics: dict = {}
    anchor: DisparityResult = results["opencv_baseline"]
    fixed: DisparityResult = results["sgm_fixed"]
    for name,result in results.items():
        row: dict = measure(truth,result,reference,threshold)
        row["bad_with_missing_fraction"] = (row["bad_pixels"]+row["missing_pixels"])/row["reference_pixels"]
        row["regions"] = {label:measure(truth,result,reference & (regions==code),threshold) for code,label in REGIONS.items()}
        common: np.ndarray = reference & anchor.valid_mask & result.valid_mask
        common_fixed: np.ndarray = reference & fixed.valid_mask & result.valid_mask
        row["paired_baseline"] = {"baseline":measure(truth,anchor,common,threshold),"method":measure(truth,result,common,threshold)}
        row["paired_fixed"] = {"fixed":measure(truth,fixed,common_fixed,threshold),"method":measure(truth,result,common_fixed,threshold)}
        metrics[name] = row
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("config/sgm_comparison.json"))
    args: argparse.Namespace = parser.parse_args(argv)
    raw: dict = json.loads(args.config.read_text(encoding="utf-8"))
    base = load_offline_config(args.config.parent/raw["baseline_config"])
    diagnosis: dict = json.loads((args.config.parent/raw["diagnosis_config"]).read_text(encoding="utf-8"))
    diag: DiagnosisConfig = DiagnosisConfig(**{key:diagnosis[key] for key in DiagnosisConfig.__annotations__})
    settings: SgmSettings = SgmSettings(**raw["sgm"])
    root: Path = (args.config.parent/raw["output_root"]).resolve()
    root.mkdir(parents=True,exist_ok=True)
    protocol: dict = {"settings": raw, "configuration_sha256": file_hashes({
        "experiment": args.config, "baseline": args.config.parent/raw["baseline_config"],
        "diagnosis": args.config.parent/raw["diagnosis_config"]}),
        "source_sha256": file_hashes({name: Path(name) for name in (
            "src/sgm_comparison.py", "src/sgm.py", "src/disparity.py", "src/datasets.py",
            "src/config.py", "src/model.py", "src/error_diagnosis.py", "src/provenance.py")})}
    if (root/"summary.json").exists():
        raise ValueError("Completed comparison exists; do not overwrite or tune against GT")
    if (root/"protocol.json").exists() and json.loads((root/"protocol.json").read_text(encoding="utf-8")) != protocol:
        raise ValueError("Existing comparison protocol differs")
    save_source_snapshot(root, protocol["source_sha256"])
    write_json(root/"protocol.json",protocol)
    cv2.setNumThreads(1)
    rows: list[dict] = []
    for name in raw["scenes"]:
        folder: Path = root/name
        if (folder/"report.json").exists() and (folder/"arrays.npz").exists():
            if json.loads((folder/"protocol.json").read_text(encoding="utf-8")) != protocol:
                raise ValueError("Resume protocol differs")
            rows.append(json.loads((folder/"report.json").read_text(encoding="utf-8")))
            print(json.dumps({"scene":name,"status":"resumed"}),flush=True)
            continue
        folder.mkdir(exist_ok=True)
        write_json(folder/"protocol.json",protocol)
        scene = load_middlebury_scene(base.image_root,"trainingQ",name,ground_truth_root=base.ground_truth_root)
        predictions: dict[str,DisparityResult] = {}
        timings: dict[str,float] = {}
        means: dict[str,float] = {}
        for method in METHODS:
            start: float = time.perf_counter()
            if method == "opencv_baseline":
                predictions[method] = compute_disparity(scene.rectified_pair,scene.geometry,base.sgbm)
            elif method.startswith("opencv_"):
                predictions[method] = opencv_variant(scene.rectified_pair,scene.geometry,base.sgbm,raw["opencv_options"],scene.disparity_search_hint_px,method=="opencv_range")
            else:
                mode: str = method.removeprefix("sgm_")
                output = compute_sgm_disparity(scene.rectified_pair,scene.geometry,settings,mode)
                predictions[method] = output.final
                predictions[method+"_raw"] = output.raw
                means[method] = output.mean_p2_left
            timings[method] = (time.perf_counter()-start)*1000
            print(json.dumps({"scene":name,"method":method,"ms":round(timings[method])}),flush=True)
        truth: np.ndarray = scene.ground_truth.disparity_px
        reference: np.ndarray = scene.ground_truth.non_occluded_mask & np.isfinite(truth)
        regions: np.ndarray = classify_regions(truth,reference,texture_std(scene.rectified_pair.left_bgr,diag.texture_window_px),base.sgbm,diag)
        metrics: dict = evaluate_methods(predictions,truth,reference,regions,diag.bad_threshold_px)
        row: dict = {"scene":name,"status":"success","metrics":metrics,"timing_ms":timings,"input_mean_p2":means,
            "range_hint_px":scene.disparity_search_hint_px}
        arrays: dict = {"left":scene.rectified_pair.left_bgr,"truth":truth,"reference":reference,"regions":regions}
        for method,result in predictions.items():
            arrays[method] = result.disparity_px
            arrays[method+"_valid"] = result.valid_mask
        np.savez_compressed(folder/"arrays.npz",**arrays)
        write_json(folder/"report.json",row)
        rows.append(row)
    summary: dict = {"scene_count":len(rows),"failed_scenes":0,"settings":raw,"versions":{
        "python":platform.python_version(),"opencv":cv2.__version__,"numpy":np.__version__},"methods":{},"scenes":rows}
    for method in rows[0]["metrics"]:
        pooled: dict = pool([row["metrics"][method] for row in rows])
        pooled["bad_with_missing_fraction"] = (pooled["bad_pixels"]+pooled["missing_pixels"])/pooled["reference_pixels"]
        pooled["regions"] = {label:pool([row["metrics"][method]["regions"][label] for row in rows]) for label in REGIONS.values()}
        for comparison in ("paired_baseline","paired_fixed"):
            pooled[comparison] = {side:pool([row["metrics"][method][comparison][side] for row in rows]) for side in rows[0]["metrics"][method][comparison]}
        pooled["mean_time_ms"] = float(np.mean([row["timing_ms"].get(method,np.nan) for row in rows])) if method in METHODS else None
        summary["methods"][method] = pooled
    write_json(root/"summary.json",summary)
    print(json.dumps({"status":"complete","scene_count":len(rows),"methods":{m:{k:r[k] for k in ("mae_px","bad_fraction","coverage","bad_with_missing_fraction")} for m,r in summary["methods"].items()}}),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
