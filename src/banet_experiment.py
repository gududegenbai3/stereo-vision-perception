"""Official pretrained BANet inference and existing SGBM comparison, no training."""

import argparse
import hashlib
import json
import os
import platform
import time
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from .banet import load_banet,predict_banet
from .config import load_offline_config
from .datasets import load_middlebury_scene
from .disparity import compute_disparity
from .error_diagnosis import DiagnosisConfig,REGIONS,classify_regions,measure,pool,texture_std,write_json
from .model import DisparityResult,SgbmConfig
from .reprojection import reproject_disparity


def evaluate_predictions(results: dict[str,DisparityResult],truth: np.ndarray,reference: np.ndarray,regions: np.ndarray) -> dict:
    """Own-domain and identical-pixel comparisons; missing outputs remain visible."""
    metrics: dict = {}
    for method,result in results.items():
        row: dict = measure(truth,result,reference,2)
        row["bad_with_missing_fraction"] = (row["bad_pixels"]+row["missing_pixels"])/row["reference_pixels"]
        row["regions"] = {label:measure(truth,result,reference & (regions==code),2) for code,label in REGIONS.items()}
        common: np.ndarray = reference & results["banet"].valid_mask & result.valid_mask
        row["paired_with_banet"] = {"method":measure(truth,result,common,2),"banet":measure(truth,results["banet"],common,2)}
        metrics[method] = row
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("config/banet.json"))
    args: argparse.Namespace = parser.parse_args(argv)
    raw: dict = json.loads(args.config.read_text(encoding="utf-8"))
    if raw["training"] is not False or raw["precision"] != "float32":
        raise ValueError("This protocol is pretrained float32 inference only")
    base = load_offline_config(args.config.parent/raw["baseline_config"])
    diag_raw: dict = json.loads((args.config.parent/raw["diagnosis_config"]).read_text(encoding="utf-8"))
    diag: DiagnosisConfig = DiagnosisConfig(**{key:diag_raw[key] for key in DiagnosisConfig.__annotations__})
    root: Path = (args.config.parent/raw["output_root"]).resolve()
    if (root/"summary.json").exists():
        raise ValueError("Completed experiment exists; inspect rather than overwrite")
    root.mkdir(parents=True,exist_ok=True)
    os.environ["CUDA_CACHE_PATH"] = str(root.parent/"cuda-cache")
    model,model_info = load_banet((args.config.parent/raw["source_root"]).resolve(),(args.config.parent/raw["runtime_root"]).resolve(),
        (args.config.parent/raw["checkpoint"]).resolve(),raw["checkpoint_sha256"])
    import torch
    torch.backends.cudnn.benchmark = False
    cv2.setNumThreads(1)
    write_json(root/"model.json",model_info)
    write_json(root/"protocol.json",raw)
    rows: list[dict] = []
    for name in raw["scenes"]:
        folder: Path = root/name
        folder.mkdir(exist_ok=True)
        scene = load_middlebury_scene(base.image_root,"trainingQ",name,ground_truth_root=base.ground_truth_root)
        predicted = predict_banet(model,scene.rectified_pair,raw["max_disparity"],raw["warmup_runs"],raw["timed_runs"])
        start: float = time.perf_counter()
        original: DisparityResult = compute_disparity(scene.rectified_pair,scene.geometry,base.sgbm)
        baseline_ms: float = (time.perf_counter()-start)*1000
        same_range: DisparityResult = compute_disparity(scene.rectified_pair,scene.geometry,
            SgbmConfig(base.sgbm.min_disparity,raw["range_control_num_disparities"],base.sgbm.block_size))
        reconstructed = reproject_disparity(scene.rectified_pair,scene.geometry,predicted.disparity)
        truth: np.ndarray = scene.ground_truth.disparity_px
        reference: np.ndarray = scene.ground_truth.non_occluded_mask & np.isfinite(truth)
        regions: np.ndarray = classify_regions(truth,reference,texture_std(scene.rectified_pair.left_bgr,diag.texture_window_px),base.sgbm,diag)
        predictions: dict[str,DisparityResult] = {"baseline":original,"same_range":same_range,"banet":predicted.disparity}
        metrics: dict = evaluate_predictions(predictions,truth,reference,regions)
        row: dict = {"scene":name,"status":"success","metrics":metrics,"padded_shape":list(predicted.padded_shape),
            "timing_ms":list(predicted.timing_ms),"baseline_ms":baseline_ms,"peak_allocated_mib":predicted.peak_allocated_mib,
            "peak_reserved_mib":predicted.peak_reserved_mib,"valid_3d_points":int(reconstructed.valid_mask.sum())}
        arrays: dict = {"left":scene.rectified_pair.left_bgr,"truth":truth,"reference":reference,"regions":regions,
            "attention":predicted.attention,"depth_m":reconstructed.depth_m,"xyz_m":reconstructed.xyz_m}
        for method,result in predictions.items():
            arrays[method] = result.disparity_px
            arrays[method+"_valid"] = result.valid_mask
        np.savez_compressed(folder/"arrays.npz",**arrays)
        write_json(folder/"report.json",row)
        rows.append(row)
        print(json.dumps({"scene":name,"mae":metrics["banet"]["mae_px"],"coverage":metrics["banet"]["coverage"],
            "mean_ms":float(np.mean(predicted.timing_ms)),"memory_mib":predicted.peak_allocated_mib}),flush=True)
    summary: dict = {"scenes":rows,"scene_count":len(rows),"failed_scenes":0,"model":model_info,"settings":raw,
        "versions":{"python":platform.python_version(),"opencv":cv2.__version__,"numpy":np.__version__},"methods":{}}
    for method in ("baseline","same_range","banet"):
        pooled: dict = pool([row["metrics"][method] for row in rows])
        pooled["bad_with_missing_fraction"] = (pooled["bad_pixels"]+pooled["missing_pixels"])/pooled["reference_pixels"]
        pooled["regions"] = {label:pool([row["metrics"][method]["regions"][label] for row in rows]) for label in REGIONS.values()}
        pooled["paired_with_banet"] = {side:pool([row["metrics"][method]["paired_with_banet"][side] for row in rows]) for side in ("method","banet")}
        summary["methods"][method] = pooled
    times: np.ndarray = np.array([value for row in rows for value in row["timing_ms"]])
    summary["profiling"] = {"mean_forward_ms":float(times.mean()),"p95_forward_ms":float(np.percentile(times,95)),
        "max_allocated_mib":max(row["peak_allocated_mib"] for row in rows),"max_reserved_mib":max(row["peak_reserved_mib"] for row in rows),
        "scope":"CUDA synchronized FP32 forward only; excludes IO, preprocessing, model load, GT evaluation and visualization; not camera FPS"}
    write_json(root/"summary.json",summary)
    source_paths: list[str] = ["src/banet.py","src/banet_experiment.py","config/banet.json","requirements-banet.txt"]
    write_json(root/"provenance.json",{"upstream_commit":raw["upstream_commit"],"checkpoint_source":raw["checkpoint_url"],
        "checkpoint_sha256":raw["checkpoint_sha256"],"source_sha256":{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_paths}})
    print(json.dumps({"status":"complete","scenes":len(rows),"profiling":summary["profiling"],
        "methods":{m:{k:v[k] for k in ("mae_px","bad_fraction","coverage","bad_with_missing_fraction")} for m,v in summary["methods"].items()}}),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
