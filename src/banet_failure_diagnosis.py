"""Read-only post-hoc BANet stage diagnosis; never train or change predictions."""

import argparse
import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from .banet import load_banet, padding_for_shape
from .error_diagnosis import jump_band, write_json
from .stereo_comparison_data import load_mobile, resize_pair, restore_native


def boundary_distance(truth: np.ndarray, jump: float) -> np.ndarray:
    """Native-pixel distance to finite GT disparity jump seeds, no predicted edges."""
    seeds: np.ndarray = jump_band(truth, jump, 0)
    return cv2.distanceTransform((~seeds).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def main() -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/banet_failure_diagnosis.json"))
    args: argparse.Namespace = parser.parse_args()
    raw: dict = json.loads(args.config.read_text(encoding="utf-8"))
    comparison_path: Path = args.config.parent / raw["comparison_config"]
    settings: dict = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison_root: Path = (comparison_path.parent / settings["output_root"]).resolve()
    root: Path = (args.config.parent / raw["output_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "summary.json").exists():
        raise ValueError("Diagnosis exists; do not silently overwrite")
    total: dict = {"reference_common_pixels": 0, "large_errors": 0, "large_errors_far_from_boundary": 0,
                   "large_error_sum": 0., "large_error_sum_far_from_boundary": 0.}
    per_scene: list[dict] = []
    selected_points: dict[str, tuple[int, int]] = {}
    names: list[str] = sorted(path.name for path in (comparison_root / "mobile2021_new").iterdir() if path.is_dir())
    for name in names:
        with np.load(comparison_root / "mobile2021_new" / name / "arrays.npz") as data:
            truth: np.ndarray = data["truth"]
            common: np.ndarray = data["reference"] & data["sgbm192_valid"] & data["banet_valid"]
            distance: np.ndarray = boundary_distance(truth, raw["gt_jump_native_px"])
            error: np.ndarray = np.abs(data["banet"] - truth)
            large: np.ndarray = common & (error > raw["large_error_native_px"])
            far: np.ndarray = distance > raw["far_from_gt_boundary_native_px"]
            row: dict = {"sample": name, "reference_common_pixels": int(common.sum()), "large_errors": int(large.sum()),
                "large_errors_far_from_boundary": int((large & far).sum()), "large_error_sum": float(error[large].astype(np.float64).sum()),
                "large_error_sum_far_from_boundary": float(error[large & far].astype(np.float64).sum())}
            for field in total:
                total[field] += row[field]
            per_scene.append(row)
            if name in raw["inspect_scenes"]:
                sgbm_error: np.ndarray = np.abs(data["sgbm192"] - truth)
                candidate: np.ndarray = common & far & (sgbm_error <= raw["point_min_sgbm_agreement_error_native_px"])
                candidate[:32] = candidate[-32:] = False
                candidate[:, :32] = candidate[:, -32:] = False
                if not candidate.any():
                    raise ValueError(f"No diagnostic interior point for {name}")
                score: np.ndarray = np.where(candidate, -error if name == "curule2" else error, -np.inf)
                selected_points[name] = tuple(int(value) for value in np.unravel_index(np.argmax(score), score.shape))
    total["large_error_fraction_far_from_boundary"] = total["large_errors_far_from_boundary"] / total["large_errors"]
    total["large_error_sum_fraction_far_from_boundary"] = total["large_error_sum_far_from_boundary"] / total["large_error_sum"]
    write_json(root / "distance_analysis.json", {"settings": raw, "total": total, "scenes": per_scene,
        "point_selection": "Post-hoc interior probes: >16px from GT jumps, SGBM within2px; worst BANet error except good control curule2"})
    model_path: Path = args.config.parent / raw["model_config"]
    model_config: dict = json.loads(model_path.read_text(encoding="utf-8"))
    model, info = load_banet((model_path.parent / model_config["source_root"]).resolve(),
        (model_path.parent / model_config["runtime_root"]).resolve(), (model_path.parent / model_config["checkpoint"]).resolve(),
        model_config["checkpoint_sha256"])
    import torch
    import torch.nn.functional as functional

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    records: list[dict] = []
    for name in raw["inspect_scenes"]:
        scene = load_mobile((comparison_path.parent / settings["mobile_root"]).resolve(), name, settings["mobile_scale"])
        pair, _ = resize_pair(scene.pair, scene.geometry, scene.scale)
        height, width = pair.left_bgr.shape[:2]
        padding: tuple[int, int, int, int] = padding_for_shape(height, width)
        left_pad, right_pad, top_pad, bottom_pad = padding
        inputs: list[torch.Tensor] = []
        for image in (pair.left_bgr, pair.right_bgr):
            rgb: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            value: torch.Tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to("cuda")
            inputs.append(functional.pad(value, padding, mode="replicate"))
        captured: dict[str, torch.Tensor] = {}
        def hook(label: str) -> Callable[[torch.nn.Module, tuple[object, ...], torch.Tensor], None]:
            def capture(_module: torch.nn.Module, _inputs: tuple[object, ...], output: torch.Tensor) -> None:
                captured[label] = output.detach().clone()
            return capture
        handles: list = [getattr(model, layer).register_forward_hook(hook(label)) for layer, label in
                         (("build_cv", "correlation"), ("spa_att", "attention"), ("cost_agg0", "branch0"),
                          ("cost_agg1", "branch1"), ("spx", "upsample_logits"))]
        try:
            with torch.inference_mode():
                output: torch.Tensor = model(inputs[0], inputs[1], 192)
                attention: torch.Tensor = captured["attention"]
                logits: torch.Tensor = attention * captured["branch0"] + (1 - attention) * captured["branch1"]
                probability: torch.Tensor = torch.softmax(logits, dim=1)
                levels: torch.Tensor = torch.arange(48, device="cuda").view(1, 48, 1, 1)
                low: torch.Tensor = (probability * levels).sum(1, keepdim=True)
                weights: torch.Tensor = torch.softmax(captured["upsample_logits"], dim=1)
                neighbors: torch.Tensor = functional.unfold(low, kernel_size=3, padding=1).reshape(1, 9, *low.shape[-2:])
                expanded: torch.Tensor = functional.interpolate(neighbors, scale_factor=4, mode="nearest")
                reconstructed: torch.Tensor = (expanded * weights).sum(1, keepdim=True) * 4
                reconstruction_delta: float = float(torch.max(torch.abs(reconstructed - output)))
                if reconstruction_delta > 1e-4:
                    raise ValueError("Captured stages do not reconstruct the actual model output")
                final_input: np.ndarray = output[0, 0, top_pad:top_pad + height, left_pad:left_pad + width].cpu().numpy()
                native: np.ndarray = restore_native(final_input, np.ones_like(final_input, bool), scene.truth.shape, scene.scale).disparity_px
                nearest_low = functional.interpolate(low * 4, output.shape[-2:], mode="nearest")
                nearest_input: np.ndarray = nearest_low[0, 0, top_pad:top_pad + height, left_pad:left_pad + width].cpu().numpy()
                coarse_native: np.ndarray = restore_native(nearest_input, np.ones_like(nearest_input, bool), scene.truth.shape, scene.scale).disparity_px
                probability_array: np.ndarray = probability[0].cpu().numpy()
                low_array: np.ndarray = low[0, 0].cpu().numpy() * 4 / scene.scale
                attention_input = functional.interpolate(attention, output.shape[-2:], mode="bilinear", align_corners=False)
                attention_map: np.ndarray = attention_input[0, 0, top_pad:top_pad + height, left_pad:left_pad + width].cpu().numpy()
        finally:
            for handle in handles:
                handle.remove()
        y, x = selected_points[name]
        input_x: float = (x + .5) * scene.scale - .5
        input_y: float = (y + .5) * scene.scale - .5
        coarse_x: int = int((round(input_x) + left_pad) // 4)
        coarse_y: int = int((round(input_y) + top_pad) // 4)
        probs: np.ndarray = probability_array[:, coarse_y, coarse_x]
        best_indices: np.ndarray = np.argsort(probs)[-3:][::-1]
        footprint: list[float] = []
        for input_row in {math.floor(input_y), math.ceil(input_y)}:
            for input_col in {math.floor(input_x), math.ceil(input_x)}:
                center_y: int = (input_row + top_pad) // 4
                center_x: int = (input_col + left_pad) // 4
                footprint.extend(float(value) for value in low_array[center_y - 1:center_y + 2, center_x - 1:center_x + 2].reshape(-1))
        gt: float = float(scene.truth[y, x])
        lower, upper = min(footprint), max(footprint)
        with np.load(comparison_root / "mobile2021_new" / name / "arrays.npz") as prior:
            archived_delta: float = float(np.max(np.abs(native - prior["banet"])))
        record: dict = {"sample": name, "native_xy": [x, y], "gt_disparity_native_px": gt,
            "final_native_px": float(native[y, x]), "coarse_expectation_native_px": float(low_array[coarse_y, coarse_x]),
            "coarse_mode_native_px": float(best_indices[0] * 4 / scene.scale), "max_softmax_weight": float(probs.max()),
            "attention_branch_weight": float(captured["attention"][0, 0, coarse_y, coarse_x].cpu()),
            "top3_disparities_native_px": [float(index * 4 / scene.scale) for index in best_indices],
            "top3_weights": [float(probs[index]) for index in best_indices],
            "upsample_support_native_px": [lower, upper], "gt_outside_upsample_support": gt < lower or gt > upper,
            "output_reconstruction_max_error_input_px": reconstruction_delta, "archived_prediction_max_difference_native_px": archived_delta}
        records.append(record)
        np.savez_compressed(root / f"stages-{name}.npz", left=scene.pair.left_bgr, truth=scene.truth, reference=scene.reference,
                            final=native, coarse=coarse_native, attention=attention_map, probabilities=probs, point=np.array([x, y]))
        print(json.dumps(record), flush=True)
        captured.clear()
    write_json(root / "summary.json", {"distance_analysis": total, "probes": records, "model": info,
        "settings": raw, "checkpoint_sha256": model_config["checkpoint_sha256"],
        "scope": "Observed-scene diagnostic probes, no architecture/data-factor causal ablation; softmax weights are not calibrated confidence"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
