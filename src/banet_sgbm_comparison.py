"""Frozen 48-pair comparison: local leakage audit, raw BANet and two SGBM ranges."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .banet import load_banet, padding_for_shape
from .config import load_offline_config
from .disparity import compute_disparity
from .error_diagnosis import jump_band, measure, pool, texture_std, write_json
from .model import DisparityResult, RectifiedPair, SgbmConfig
from .provenance import save_source_snapshot
from .stereo_comparison_data import ComparisonScene, choose_random_views, load_eth, load_mobile, load_previous, resize_pair, restore_native
from .stereo_leakage_audit import audit_evaluation_images, audit_training_run

if TYPE_CHECKING:
    import torch


def infer_dense_banet(model: torch.nn.Module, pair: RectifiedPair, max_disparity: int) -> np.ndarray:
    """RGB 0..255 FP32, replicate pad32, output raw float32 (H,W) input px.

    Interface accepts images only: no labels, validity masks, scene names or GT.
    No confidence/right-projection filtering may remove an erroneous prediction.
    """
    import torch
    import torch.nn.functional as functional

    if max_disparity != 192 or pair.left_bgr.shape != pair.right_bgr.shape:
        raise ValueError("Expected equal images and official 192-disparity model")
    height, width = pair.left_bgr.shape[:2]
    padding: tuple[int, int, int, int] = padding_for_shape(height, width)
    left_pad, _, top_pad, _ = padding
    images: list[torch.Tensor] = []
    for image in (pair.left_bgr, pair.right_bgr):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected BGR uint8 (H,W,3)")
        rgb: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor: torch.Tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to("cuda")
        images.append(functional.pad(tensor, padding, mode="replicate"))
    model.eval()
    with torch.inference_mode():
        output: torch.Tensor = model(images[0], images[1], max_disparity)
        values: np.ndarray = output[0, 0, top_pad:top_pad + height, left_pad:left_pad + width].float().cpu().numpy().copy()
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite BANet output; cannot silently drop sample/pixels")
    return values


def with_failure_fraction(row: dict) -> dict:
    """Missing outputs are failures on the full GT domain, never zero error."""
    result: dict = dict(row)
    result["bad_or_missing_fraction"] = ((row["bad_pixels"] + row["missing_pixels"]) / row["reference_pixels"]
                                          if row["reference_pixels"] else None)
    return result


def compare_predictions(truth: np.ndarray, reference: np.ndarray, predictions: dict[str, DisparityResult],
                        regions: np.ndarray, threshold: float) -> dict:
    """Full GT coverage/failure and exact paired domains; regions are GT-only analysis."""
    rows: dict = {}
    for method, prediction in predictions.items():
        own: dict = with_failure_fraction(measure(truth, prediction, reference, threshold))
        own["regions"] = {label: with_failure_fraction(measure(truth, prediction, reference & (regions == code), threshold))
                          for code, label in ((1, "boundary"), (2, "low_texture"), (3, "other"))}
        rows[method] = own
    pairs: dict = {}
    for sgbm in ("sgbm128", "sgbm192"):
        common: np.ndarray = reference & predictions[sgbm].valid_mask & predictions["banet"].valid_mask
        pairs[sgbm] = {method: measure(truth, predictions[method], common, threshold) for method in (sgbm, "banet")}
        pairs[sgbm]["regions"] = {label: {method: measure(truth, predictions[method], common & (regions == code), threshold)
                                         for method in (sgbm, "banet")}
                                  for code, label in ((1, "boundary"), (2, "low_texture"), (3, "other"))}
    return {"methods": rows, "pairs": pairs}


def analyze_regions(scene: ComparisonScene, raw: dict) -> np.ndarray:
    """Thresholds in input-grid px; convert lengths to native GT resolution."""
    boundary: np.ndarray = jump_band(scene.truth, raw["boundary_jump_input_px"] / scene.scale,
                                    math.ceil(raw["boundary_radius_input_px"] / scene.scale))
    window: int = math.ceil(raw["texture_window_input_px"] / scene.scale)
    if window % 2 == 0:
        window += 1
    texture: np.ndarray = texture_std(scene.pair.left_bgr, window)
    labels: np.ndarray = np.zeros(scene.truth.shape, np.uint8)
    labels[scene.reference] = 3
    labels[scene.reference & (texture < raw["texture_std_threshold"])] = 2
    labels[scene.reference & boundary] = 1
    return labels


def pool_dataset(rows: list[dict]) -> dict:
    """Pool sufficient sums, additionally give equal weight to physical scenes."""
    methods: dict = {method: with_failure_fraction(pool([row["metrics"]["methods"][method] for row in rows]))
                     for method in ("sgbm128", "sgbm192", "banet")}
    pairs: dict = {}
    for sgbm in ("sgbm128", "sgbm192"):
        pairs[sgbm] = {method: pool([row["metrics"]["pairs"][sgbm][method] for row in rows]) for method in (sgbm, "banet")}
        pairs[sgbm]["regions"] = {label: {method: pool([row["metrics"]["pairs"][sgbm]["regions"][label][method] for row in rows])
                                         for method in (sgbm, "banet")}
                                  for label in ("boundary", "low_texture", "other")}
    groups: list[str] = sorted({row["group"] for row in rows})
    per_group: list[dict] = []
    for group in groups:
        selected: list[dict] = [row for row in rows if row["group"] == group]
        paired: dict = {method: pool([row["metrics"]["pairs"]["sgbm192"][method] for row in selected])
                        for method in ("sgbm192", "banet")}
        per_group.append({"group": group, "pairs": len(selected), "sgbm192_mae_px": paired["sgbm192"]["mae_px"],
                          "banet_mae_px": paired["banet"]["mae_px"]})
    wins: int = sum(row["metrics"]["pairs"]["sgbm192"]["banet"]["mae_px"] < row["metrics"]["pairs"]["sgbm192"]["sgbm192"]["mae_px"] for row in rows)
    return {"pair_count": len(rows), "physical_group_count": len(groups), "banet_mae_wins_against_sgbm192": wins,
        "methods": methods, "paired": pairs, "physical_groups": per_group,
        "equal_physical_group_mean_mae_px": {method: float(np.mean([row[method + "_mae_px"] for row in per_group]))
                                             for method in ("sgbm192", "banet")}}


def load_scenes(raw: dict, config_parent: Path) -> list[ComparisonScene]:
    """Inventory is fixed before inference; never choose a sample from its score."""
    mobile_root: Path = (config_parent / raw["mobile_root"]).resolve()
    data_root: Path = (config_parent / raw["training_data_root"]).resolve()
    training: dict = json.loads((config_parent / raw["training_protocol"]).read_text(encoding="utf-8"))
    base = load_offline_config(config_parent / raw["baseline_config"])
    mobile_names: list[str] = sorted(path.parent.name for path in mobile_root.glob("*/im0.png"))
    scenes: list[ComparisonScene] = [load_mobile(mobile_root, name, raw["mobile_scale"],
        tuple(raw["mobile_image_size_overrides"][name]) if name in raw["mobile_image_size_overrides"] else None) for name in mobile_names]
    scenes.extend(load_eth(data_root, name) for name in training["splits"]["test"])
    scenes.extend(load_previous(base.image_root, base.ground_truth_root, path.name) for path in sorted((base.image_root / "trainingQ").iterdir())
                  if path.is_dir() and (path / "im0.png").exists())
    counts: dict[str, int] = {dataset: sum(scene.dataset == dataset for scene in scenes) for dataset in raw["expected_dataset_counts"]}
    if counts != raw["expected_dataset_counts"]:
        raise ValueError(f"Incomplete dataset inventory: {counts}")
    return scenes


def main() -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/banet_sgbm_comparison.json"))
    args: argparse.Namespace = parser.parse_args()
    raw: dict = json.loads(args.config.read_text(encoding="utf-8"))
    root: Path = (args.config.parent / raw["output_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "protocol.json").exists():
        raise ValueError("Frozen comparison exists; inspect it, do not overwrite or tune")
    if hashlib.sha256((args.config.parent / raw["mobile_archive"]).read_bytes()).hexdigest() != raw["mobile_archive_sha256"]:
        raise ValueError("New evaluation archive hash differs")
    scenes: list[ComparisonScene] = load_scenes(raw, args.config.parent)
    random_views: list[str] = choose_random_views([scene.name for scene in scenes if scene.dataset == "mobile2021_new"],
                                               raw["random_seed"], raw["random_physical_scene_count"])
    source_names: list[str] = ["src/banet_sgbm_comparison.py", "src/stereo_comparison_data.py", "src/stereo_leakage_audit.py",
                               "src/banet.py", "src/disparity.py", "src/datasets.py", "src/error_diagnosis.py", "src/provenance.py"]
    protocol: dict = {"settings": raw, "random_views": random_views, "sample_inventory": [
        {"dataset": scene.dataset, "name": scene.name, "group": scene.group, "native_shape": list(scene.truth.shape),
         "scale": scene.scale, "reference_policy": scene.reference_policy} for scene in scenes],
        "source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in source_names}}
    save_source_snapshot(root, protocol["source_sha256"])
    write_json(root / "protocol.json", protocol)
    training_path: Path = (args.config.parent / raw["training_protocol"]).resolve()
    data_root: Path = (args.config.parent / raw["training_data_root"]).resolve()
    replay: dict = audit_training_run(training_path, (args.config.parent / raw["training_checkpoint"]).resolve(), data_root)
    write_json(root / "training_replay_trace.json", replay.pop("trace"))
    write_json(root / "training_audit.json", replay)
    training_protocol: dict = json.loads(training_path.read_text(encoding="utf-8"))
    leakage: dict = audit_evaluation_images(training_protocol, data_root, scenes, raw["perceptual_hash_review_distance"])
    write_json(root / "image_leakage_audit.json", leakage)
    print(json.dumps({"training_audit_passed": replay["passed"], "image_exact_duplicates": len(leakage["exact_duplicates"]),
        "perceptual_candidates": len(leakage["perceptual_review_candidates"]), "random_views": random_views}), flush=True)
    model_path: Path = args.config.parent / raw["model_config"]
    model_raw: dict = json.loads(model_path.read_text(encoding="utf-8"))
    model, info = load_banet((model_path.parent / model_raw["source_root"]).resolve(), (model_path.parent / model_raw["runtime_root"]).resolve(),
        (model_path.parent / model_raw["checkpoint"]).resolve(), model_raw["checkpoint_sha256"])
    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cv2.setNumThreads(1)
    write_json(root / "model.json", info)
    rows: list[dict] = []
    for scene in scenes:
        pair, geometry = resize_pair(scene.pair, scene.geometry, scene.scale)
        # Prediction functions see images and public geometry/config only.
        start: float = time.perf_counter()
        dense: np.ndarray = infer_dense_banet(model, pair, raw["max_disparity"])
        elapsed: float = time.perf_counter() - start
        predictions: dict[str, DisparityResult] = {"banet": restore_native(dense, np.ones_like(dense, bool), scene.truth.shape, scene.scale)}
        for count in raw["sgbm"]["num_disparities"]:
            settings: SgbmConfig = SgbmConfig(raw["sgbm"]["min_disparity"], count, raw["sgbm"]["block_size"])
            result: DisparityResult = compute_disparity(pair, geometry, settings)
            predictions[f"sgbm{count}"] = restore_native(result.disparity_px, result.valid_mask, scene.truth.shape, scene.scale)
        regions: np.ndarray = analyze_regions(scene, raw)
        metrics: dict = compare_predictions(scene.truth, scene.reference, predictions, regions, raw["bad_threshold_native_px"])
        row: dict = {"dataset": scene.dataset, "sample": scene.name, "group": scene.group, "status": "success",
            "native_shape": list(scene.truth.shape), "input_shape": list(dense.shape), "scale": scene.scale,
            "gt_outside_192_input_pixels": int((scene.reference & (scene.truth * scene.scale >= 192)).sum()),
            "metrics": metrics, "banet_stage_ms": elapsed * 1000}
        folder: Path = root / scene.dataset / scene.name
        folder.mkdir(parents=True, exist_ok=True)
        arrays: dict = {"left": scene.pair.left_bgr, "truth": scene.truth, "reference": scene.reference, "regions": regions}
        for method, prediction in predictions.items():
            arrays[method] = prediction.disparity_px
            arrays[method + "_valid"] = prediction.valid_mask
        np.savez_compressed(folder / "arrays.npz", **arrays)
        write_json(folder / "report.json", row)
        rows.append(row)
        paired: dict = metrics["pairs"]["sgbm192"]
        print(json.dumps({"dataset": scene.dataset, "sample": scene.name, "sgbm192_common_mae": paired["sgbm192"]["mae_px"],
            "banet_common_mae": paired["banet"]["mae_px"]}), flush=True)
    summary: dict = {"scene_count": len(rows), "failed_scenes": 0, "datasets": {
        dataset: pool_dataset([row for row in rows if row["dataset"] == dataset]) for dataset in raw["expected_dataset_counts"]},
        "random_views": random_views, "checkpoint_sha256": model_raw["checkpoint_sha256"], "scenes": rows,
        "metrics_units": "Original native image px; mobile inputs half-size then predictions bilinear restored and divided by scale. Dataset policies differ: do not pool all datasets into one claim."}
    write_json(root / "summary.json", summary)
    with (root / "per_scene.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields: list[str] = ["dataset", "sample", "group", "sgbm192_common_mae_px", "banet_common_mae_px", "sgbm192_common_bad2", "banet_common_bad2",
                              "sgbm192_coverage", "banet_coverage", "sgbm192_bad_or_missing", "banet_bad_or_missing"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            paired = row["metrics"]["pairs"]["sgbm192"]
            own: dict = row["metrics"]["methods"]
            writer.writerow({"dataset": row["dataset"], "sample": row["sample"], "group": row["group"],
                "sgbm192_common_mae_px": paired["sgbm192"]["mae_px"], "banet_common_mae_px": paired["banet"]["mae_px"],
                "sgbm192_common_bad2": paired["sgbm192"]["bad_fraction"], "banet_common_bad2": paired["banet"]["bad_fraction"],
                "sgbm192_coverage": own["sgbm192"]["coverage"], "banet_coverage": own["banet"]["coverage"],
                "sgbm192_bad_or_missing": own["sgbm192"]["bad_or_missing_fraction"], "banet_bad_or_missing": own["banet"]["bad_or_missing_fraction"]})
    print(json.dumps({"status": "complete", "pairs": len(rows), "random_views": random_views}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
