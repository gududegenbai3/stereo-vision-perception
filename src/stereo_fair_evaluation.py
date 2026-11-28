"""Frozen, multi-source pretrained evaluation with development-only SGBM selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .camera_io import digest, write_json
from .prepare_pretrained import prepare as prepare_pretrained
from .pretrained_stereo import environment_dll_handle, load_banet, load_igev, predict_banet_rgb, predict_igev
from .provenance import file_hashes, save_source_snapshot, verify_recorded_sources
from .stereo_fair_metrics import evaluation_domains, jump_band, paired_bootstrap, scene_source_summary, score_prediction, texture_std
from .stereo_fair_data import decoded_hash, load_arrays, perceptual_hash


METHODS: tuple[str, ...] = ("sgbm_original", "sgbm_selected", "banet", "igev8", "igev32")


def settings(config: Path) -> tuple[dict[str, object], Path]:
    raw: dict[str, object] = json.loads(config.read_text(encoding="utf-8"))
    if raw["training"] is not False or raw["precision"] != "float32" or raw["primary_igev_iterations"] != 32 or raw["max_disparity"] != 192:
        raise ValueError("Fixed pretrained FP32/192 comparison, no training or test-driven model selection")
    return raw, (config.parent/raw["output_root"]).resolve()


def original_sgbm() -> dict[str, object]:
    return {"block": 5, "input": "gray", "mode": "SGBM", "uniqueness_ratio": 0, "speckle_window_size": 0, "speckle_range": 0, "disp12_max_diff": 0}


def sgbm_predict(arrays: dict[str, np.ndarray], parameters: dict[str, object], maximum: int = 192) -> np.ndarray:
    """Float32 HW pixel disparity. Explicit OpenCV ROI and native invalid code -> NaN."""
    block: int = parameters["block"]
    channels: int = 1 if parameters["input"] == "gray" else 3
    convert: int = cv2.COLOR_RGB2GRAY if channels == 1 else cv2.COLOR_RGB2BGR
    mode: int = cv2.STEREO_SGBM_MODE_SGBM if parameters["mode"] == "SGBM" else cv2.STEREO_SGBM_MODE_HH
    matcher: cv2.StereoSGBM = cv2.StereoSGBM_create(minDisparity=0, numDisparities=maximum, blockSize=block, P1=8*channels*block**2, P2=32*channels*block**2,
        mode=mode, uniquenessRatio=parameters["uniqueness_ratio"], speckleWindowSize=parameters["speckle_window_size"], speckleRange=parameters["speckle_range"], disp12MaxDiff=parameters["disp12_max_diff"])
    raw: np.ndarray = matcher.compute(cv2.cvtColor(arrays["left"], convert), cv2.cvtColor(arrays["right"], convert))
    height, width = raw.shape
    x, y, w, h = cv2.getValidDisparityROI((0, 0, width, height), (0, 0, width, height), 0, maximum, block)
    valid: np.ndarray = np.zeros(raw.shape, bool)
    valid[y:y+h, x:x+w] = True
    valid &= (raw > -16) & (raw < maximum*16)
    return np.where(valid, raw.astype(np.float32)/16, np.nan).astype(np.float32)


def verify(config: Path) -> tuple[dict[str, object], Path, dict[str, object]]:
    raw, root = settings(config)
    protocol: dict[str, object] = json.loads((root/"protocol.json").read_text(encoding="utf-8"))
    if hashlib.sha256(config.read_bytes()).hexdigest() != protocol["config_sha256"] or verify_recorded_sources(root, protocol["source_sha256"]):
        raise ValueError("Configuration or numerical source differs from frozen protocol")
    if any(digest(Path(path)) != expected for path, expected in protocol["external_sha256"].items()):
        raise ValueError("External model source or weight differs from frozen protocol")
    return raw, root, protocol


def freeze(config: Path) -> dict[str, object]:
    raw, root = settings(config)
    if (root/"protocol.json").exists():
        raise ValueError("Protocol already frozen")
    root.mkdir(parents=True, exist_ok=True)
    sample_manifest: Path = (config.parent / raw["samples"]).resolve()
    data_root: Path = (config.parent / raw["normalized_root"]).resolve()
    rows: list[dict[str, object]] = [
        {**row, "path": str(data_root / row["path"])}
        for row in json.loads(sample_manifest.read_text(encoding="utf-8"))
    ]
    if len(rows) != 196 or len([row for row in rows if row["role"] == "development"]) != 15:
        raise ValueError("Unexpected sample inventory")
    development_groups: set[str] = {row["group"] for row in rows if row["role"] == "development"}
    if development_groups & {row["group"] for row in rows if row["role"] == "evaluation"}:
        raise ValueError("Development/evaluation group collision")
    images: list[dict[str, object]] = []
    for row in rows:
        path: Path = Path(row["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Normalized file identity mismatch")
        arrays: dict[str, np.ndarray] = load_arrays(str(path))
        for view in ("left", "right"):
            images.append({"dataset": row["dataset"], "role": row["role"], "group": row["group"], "path": row["path"], "view": view,
                "decoded_sha256": decoded_hash(arrays[view]), "phash": perceptual_hash(arrays[view])})
    candidates: list[dict[str, object]] = []
    duplicates: list[dict[str, object]] = []
    for i, left in enumerate(images):
        for right in images[i+1:]:
            should_check: bool = left["role"] != right["role"] or ((left["dataset"] == "middlebury2006_new") != (right["dataset"] == "middlebury2006_new"))
            if not should_check:
                continue
            distance: int = (left["phash"]^right["phash"]).bit_count()
            if left["decoded_sha256"] == right["decoded_sha256"]:
                duplicates.append({"left": left, "right": right})
            if distance <= 8:
                candidates.append({"left": left, "right": right, "distance": distance})
    image_audit: dict[str, object] = {"exact_duplicates": duplicates, "phash_candidates": candidates, "threshold": 8, "image_count": len(images), "scope": "Development vs evaluation and new2006 vs existing sources; no author training archive available"}
    write_json(root/"image_audit.json", image_audit)
    if candidates or duplicates:
        raise ValueError("Potential cross-role/new-source duplication requires review before freezing")
    rng: np.random.Generator = np.random.default_rng(raw["random_display_seed"])
    views: list[dict[str, object]] = []
    for dataset in raw["evaluation_datasets"]:
        names: list[str] = sorted({row["group"] for row in rows if row["dataset"] == dataset})
        for group in sorted(rng.choice(names, min(raw["random_groups_per_dataset"], len(names)), replace=False).tolist()):
            sample: dict[str, object] = sorted([row for row in rows if row["group"] == group], key=lambda row: row["path"])[0]
            views.append({"dataset": dataset, "group": group, "path": sample["path"]})
    dependencies: list[Path] = [sample_manifest, Path("requirements-stereo-cuda.txt")]
    dependencies.extend(Path("src") / name for name in (
        "stereo_fair_evaluation.py", "stereo_fair_metrics.py", "stereo_fair_data.py",
        "stereo_fair_archives.py", "pretrained_stereo.py", "prepare_pretrained.py",
        "camera_io.py", "provenance.py", "stereo_fair_report.py",
    ))
    model_config: Path = (config.parent / raw["models_config"]).resolve()
    models: dict[str, object] = json.loads(model_config.read_text(encoding="utf-8"))
    prepare_pretrained(model_config, verify_only=True)
    external: dict[str, str] = {}
    dependencies.append(model_config)
    for model in models.values():
        checkpoint: Path = (model_config.parent / model["checkpoint"]).resolve()
        source_root: Path = (model_config.parent / model["source_root"]).resolve()
        external[str(checkpoint)] = model["checkpoint_sha256"]
        external.update({str(source_root / name): sha for name, sha in model["source_sha256"].items()})
    hashes: dict[str, str] = file_hashes({path.resolve().relative_to(Path.cwd()).as_posix(): path for path in dependencies})
    protocol: dict[str, object] = {"settings": raw, "samples": rows, "random_views": views, "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(), "source_sha256": hashes, "external_sha256": external,
        "scope": raw["comparison_scope"], "primary_metric": "source-equal, group-equal Bad2+missing on fixed prediction-independent nonocc search ROI", "secondary": "all-known/full-nocc domains, joint-finite MAE, boundary/interior/texture, timing, iteration control"}
    save_source_snapshot(root, hashes)
    write_json(root/"protocol.json", protocol)
    return {"development": 15, "evaluation": 181, "new": 21, "groups": len({row["group"] for row in rows if row["role"] == "evaluation"}), "random_views": views, "duplicates": 0}


def select_sgbm(config: Path) -> dict[str, object]:
    raw, root, protocol = verify(config)
    if (root/"sgbm_selection.json").exists():
        raise ValueError("Development selection already fixed")
    cv2.setNumThreads(1)
    candidates: list[dict[str, object]] = [original_sgbm()]
    for block in raw["sgbm_blocks"]:
        for mode in raw["sgbm_modes"]:
            for image_type in raw["sgbm_inputs"]:
                candidates.append({"block": block, "input": image_type, "mode": mode, **raw["sgbm_standard_filters"]})
    samples: list[dict[str, object]] = [row for row in protocol["samples"] if row["role"] == "development"]
    scores: list[dict[str, object]] = []
    for index, parameters in enumerate(candidates):
        records: list[dict[str, object]] = []
        for sample in samples:
            arrays: dict[str, np.ndarray] = load_arrays(sample["path"])
            values: np.ndarray = sgbm_predict(arrays, parameters, raw["max_disparity"])
            domain: np.ndarray = evaluation_domains(arrays, raw["max_disparity"])["fixed_roi"]
            records.append({**sample, "metrics": {"fixed_roi": score_prediction(values, arrays["disparity"], domain)}})
        score: float = scene_source_summary(records, "fixed_roi", "bad2_or_missing")["source_equal_macro"]
        scores.append({"candidate": index, "parameters": parameters, "development_bad2_or_missing": score})
        print(json.dumps(scores[-1]), flush=True)
    winner: dict[str, object] = min(scores, key=lambda row: (row["development_bad2_or_missing"], row["candidate"]))
    result: dict[str, object] = {"winner": winner, "candidates": scores, "selection_inputs": [row["path"] for row in samples], "evaluation_images_opened": False, "criterion": raw["selection_metric"]}
    write_json(root/"sgbm_selection.json", result)
    return result


def infer(config: Path) -> dict[str, object]:
    raw, root, protocol = verify(config)
    if (root/"inference_complete.json").exists():
        raise ValueError("Inference already completed")
    selection: dict[str, object] = json.loads((root/"sgbm_selection.json").read_text(encoding="utf-8"))
    lock: dict[str, object] = {"selection_sha256": hashlib.sha256((root/"sgbm_selection.json").read_bytes()).hexdigest(), "protocol_sha256": hashlib.sha256((root/"protocol.json").read_bytes()).hexdigest()}
    if (root/"evaluation_lock.json").exists() and json.loads((root/"evaluation_lock.json").read_text()) != lock:
        raise ValueError("Evaluation selection changed")
    write_json(root/"evaluation_lock.json", lock)
    samples: list[dict[str, object]] = [row for row in protocol["samples"] if row["role"] == "evaluation"]
    for row in samples:
        if hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Evaluation file changed")
    dll_handle: object | None = environment_dll_handle()
    import torch

    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.cuda.set_per_process_memory_fraction(raw["memory_limit_fraction"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timing_sample: dict[str, object] = next(row for row in protocol["samples"] if row["role"] == "development" and row["scene"] == raw["timing"]["scene"])
    base: dict[str, np.ndarray] = load_arrays(timing_sample["path"])
    th, tw = raw["timing"]["crop_hw"]
    benchmark: dict[str, np.ndarray] = {key: np.ascontiguousarray(value[:th, :tw]) for key, value in base.items()}
    timing: dict[str, object] = {}
    model_config: Path = (config.parent / raw["models_config"]).resolve()
    models: dict[str, dict] = json.loads(model_config.read_text(encoding="utf-8"))
    for method in METHODS:
        model = None
        if method == "banet":
            mr = models["banet"]
            model, _ = load_banet(model_config.parent / mr["source_root"], root / "runtime/banet",
                                  model_config.parent / mr["checkpoint"], mr["checkpoint_sha256"])
        elif method.startswith("igev"):
            model, _ = load_igev(model_config, {**models["igev"], "runtime_root": str(root / "runtime/igev")})
        if model is not None:
            model._svp_fair_dll_handle = dll_handle
        def call(arrays: dict[str, np.ndarray]) -> np.ndarray:
            if method == "sgbm_original":
                return sgbm_predict(arrays, original_sgbm(), raw["max_disparity"])
            if method == "sgbm_selected":
                return sgbm_predict(arrays, selection["winner"]["parameters"], raw["max_disparity"])
            if method == "banet":
                return predict_banet_rgb(model, arrays, raw["max_disparity"])
            return predict_igev(model, arrays, int(method[4:]))
        for index, sample in enumerate(samples):
            folder: Path = root/f"sample-{index:03d}"
            folder.mkdir(exist_ok=True)
            values: np.ndarray = call(load_arrays(sample["path"]))
            np.save(folder/f"{method}.npy", values)
            if index % 25 == 0:
                print(json.dumps({"method": method, "completed": index+1, "total": len(samples)}), flush=True)
        torch.cuda.reset_peak_memory_stats()
        for _ in range(raw["timing"]["warmup"]):
            call(benchmark)
        times: list[float] = []
        for _ in range(raw["timing"]["repeats"]):
            torch.cuda.synchronize()
            started: float = time.perf_counter()
            call(benchmark)
            torch.cuda.synchronize()
            times.append(time.perf_counter()-started)
        timing[method] = {"seconds": times, "median_seconds": float(np.median(times)), "q25_q75_seconds": np.quantile(times, [.25, .75]).tolist(), "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2 if model is not None else None}
        del model
        torch.cuda.empty_cache()
        write_json(root/"timing.json", {"input_hw": [th, tw], "torch_threads": 4, "opencv_threads": 1, "warmup": raw["timing"]["warmup"], "repeats": raw["timing"]["repeats"], "methods": timing,
            "scope": "Same real image crop, conversion/padding/host-device/output transfer included, image/model loading excluded; CPU SGBM vs GPU neural methods"})
    write_json(root/"inference_complete.json", {"samples": len(samples), "methods": list(METHODS), "training": False})
    return {"samples": len(samples), "training": False}


def summarize(config: Path) -> dict[str, object]:
    raw, root, protocol = verify(config)
    if not (root/"inference_complete.json").exists():
        raise ValueError("Incomplete predictions")
    samples: list[dict[str, object]] = [row for row in protocol["samples"] if row["role"] == "evaluation"]
    records: dict[str, list[dict[str, object]]] = {method: [] for method in METHODS}
    for index, sample in enumerate(samples):
        arrays: dict[str, np.ndarray] = load_arrays(sample["path"])
        truth: np.ndarray = arrays["disparity"]
        domains: dict[str, np.ndarray] = evaluation_domains(arrays, raw["max_disparity"])
        predictions: dict[str, np.ndarray] = {method: np.load(root/f"sample-{index:03d}/{method}.npy") for method in METHODS}
        common: np.ndarray = domains["non_occluded"].copy()
        for values in predictions.values():
            common &= np.isfinite(values)
        domains["joint_finite"] = common
        band: np.ndarray = jump_band(np.where(arrays["known"], truth, np.nan), raw["boundary_jump_px"], raw["boundary_radius_px"])
        weak: np.ndarray = texture_std(cv2.cvtColor(arrays["left"], cv2.COLOR_RGB2BGR), raw["texture_window_px"]) < raw["texture_std_threshold"]
        domains["boundary"] = domains["fixed_roi"] & band
        domains["interior"] = domains["fixed_roi"] & ~band
        domains["weak_texture"] = domains["fixed_roi"] & weak
        domains["textured"] = domains["fixed_roi"] & ~weak
        metrics: dict[str, object] = {}
        for method, values in predictions.items():
            result: dict[str, object] = {domain: score_prediction(values, truth, mask, raw["capped_error_px"]) for domain, mask in domains.items()}
            records[method].append({"index": index, "dataset": sample["dataset"], "group": sample["group"], "scene": sample["scene"], "path": sample["path"], "metrics": result})
            metrics[method] = result
        write_json(root/f"sample-{index:03d}/metrics.json", {"sample": sample, "methods": metrics})
    summarized: dict[str, object] = {}
    for method in METHODS:
        summarized[method] = {domain: {metric: scene_source_summary(records[method], domain, metric) for metric in ("bad1_or_missing", "bad2_or_missing", "bad3_or_missing", "bad5_or_missing", "d1_or_missing", "coverage", "mae_px", "capped_mae20_or_missing")} for domain in records[method][0]["metrics"]}
    paired: dict[str, object] = {}
    for method in ("sgbm_original", "sgbm_selected", "banet", "igev8"):
        paired[method] = paired_bootstrap(summarized["igev32"]["fixed_roi"]["bad2_or_missing"]["per_group"], summarized[method]["fixed_roi"]["bad2_or_missing"]["per_group"], raw["bootstrap_seed"], raw["bootstrap_replicates"])
    summary: dict[str, object] = {"samples": len(samples), "groups": len({row["group"] for row in samples}), "datasets": {source: {"pairs": sum(row["dataset"] == source for row in samples), "groups": len({row["group"] for row in samples if row["dataset"] == source})} for source in raw["evaluation_datasets"]},
        "methods": summarized, "paired_igev32_minus_other": paired, "scope": protocol["scope"], "training": False}
    write_json(root/"summary.json", summary)
    with (root/"per_sample.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer: csv.writer = csv.writer(stream)
        writer.writerow(["dataset", "group", "scene", "method", "fixed_roi_bad2_or_missing", "full_nocc_bad2_or_missing", "full_nocc_coverage", "joint_finite_mae_px", "boundary_bad2_or_missing", "interior_bad2_or_missing"])
        for method, rows in records.items():
            for row in rows:
                m: dict[str, object] = row["metrics"]
                writer.writerow([row["dataset"], row["group"], row["scene"], method, m["fixed_roi"]["bad2_or_missing"], m["non_occluded"]["bad2_or_missing"], m["non_occluded"]["coverage"], m["joint_finite"]["mae_px"], m["boundary"]["bad2_or_missing"], m["interior"]["bad2_or_missing"]])
    print(json.dumps({method: summarized[method]["fixed_roi"]["bad2_or_missing"]["source_equal_macro"] for method in METHODS}), flush=True)
    return summary


def main() -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/stereo_fair_evaluation.json"))
    parser.add_argument("--phase", required=True, choices=("freeze", "select-sgbm", "infer", "summarize"))
    args: argparse.Namespace = parser.parse_args()
    result: dict[str, object] = {"freeze": freeze, "select-sgbm": select_sgbm, "infer": infer, "summarize": summarize}[args.phase](args.config)
    if args.phase == "freeze":
        print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
