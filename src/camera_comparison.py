"""Saved calibrated camera comparison: prepare once, infer each method separately."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

from .camera_io import calibrated_inputs, capture_inventory, digest, sgbm_predict, write_json
from .provenance import file_hashes, save_source_snapshot, verify_recorded_sources

if TYPE_CHECKING:
    import torch


def settings(config: Path) -> tuple[dict[str, Any], Path, Path]:
    """Load fixed comparison controls, raise on training or ambiguous rig state."""
    raw: dict[str, Any] = json.loads(config.read_text(encoding="utf-8"))
    if raw["training"] is not False or raw["image_scale"] not in (.5, 1.0) or raw["max_disparity"] != 192 or raw["igev_iterations"] != 32:
        raise ValueError("Require fixed FP32 official pretrained comparison")
    if raw["unchanged_focus_and_inter_camera_pose_confirmed"] is not True:
        raise ValueError("Confirm unchanged focus/inter-camera geometry in the local config")
    if not 0 < raw["memory_limit_fraction"] <= 1 or type(raw["display_count"]) is not int or raw["display_count"] < 1:
        raise ValueError("Invalid memory/display controls")
    return raw, (config.parent / raw["session_root"]).resolve(), (config.parent / raw["output_root"]).resolve()


def prepare(config: Path) -> None:
    """Verify raw pairs and calibrated geometry; freeze samples before prediction."""
    raw, session, root = settings(config)
    if root.exists():
        raise ValueError("Preserve existing run; choose a new output directory")
    selector_path: Path = (config.parent / raw["calibration_selector"]).resolve()
    selector: dict[str, Any] = json.loads(selector_path.read_text(encoding="utf-8"))
    if selector["status"] != "display_calibration_independently_checked_geometry_only":
        raise ValueError("Review calibration warnings before comparison")
    capture: dict[str, Any] = json.loads((session / "session.json").read_text(encoding="utf-8"))
    if capture["config"]["rig_id"] != selector["rig_id"] or capture.get("rectified") is not False:
        raise ValueError("Camera identity mismatch or images already rectified")
    dependencies: dict[str, Path] = {"config": config, "selector": selector_path}
    for name in ("calibration", "rectification"):
        path: Path = (selector_path.parent / selector[f"{name}_npz"]).resolve()
        if digest(path) != selector[f"{name}_sha256"]:
            raise ValueError("Calibration SHA256 mismatch")
        dependencies[name] = path
    models_path: Path = (config.parent / raw["models_config"]).resolve()
    models: dict[str, Any] = json.loads(models_path.read_text(encoding="utf-8"))
    dependencies["models_config"] = models_path
    for name in ("banet", "igev"):
        model: dict[str, Any] = models[name]
        if model["precision"] != "float32" or model["max_disparity"] != 192 or model["batch_norm_policy"] != "running":
            raise ValueError("Require official FP32 weights and saved BatchNorm statistics")
        model["checkpoint"] = str((models_path.parent / model["checkpoint"]).resolve())
        model["source_root"] = str((models_path.parent / model["source_root"]).resolve())
        if digest(Path(model["checkpoint"])) != model["checkpoint_sha256"]:
            raise ValueError("Official weight identity mismatch")
        dependencies[f"{name}_checkpoint"] = Path(model["checkpoint"])
        source_root: Path = Path(model["source_root"])
        for file in sorted(source_root.rglob("*.py")):
            dependencies[f"{name}_source/{file.relative_to(source_root).as_posix()}"] = file
    samples: list[dict[str, Any]] = capture_inventory(session)
    if any(sample["wh"] != selector["image_size_wh"] for sample in samples):
        raise ValueError("Calibration resolution does not match raw captures")
    selected: list[int] = sorted(np.random.default_rng(raw["random_seed"]).choice(len(samples), min(raw["display_count"], len(samples)), replace=False).tolist())
    source_files: dict[str, Path] = {f"src/{name}.py": Path(f"src/{name}.py") for name in ("camera_io", "camera_comparison", "pretrained_stereo", "provenance")}
    protocol: dict[str, Any] = {"settings": raw, "session_root": str(session), "samples": [], "display_indices": selected, "models": models,
        "dependency_paths": {key: str(path.resolve()) for key, path in dependencies.items()}, "dependency_sha256": file_hashes(dependencies),
        "source_sha256": file_hashes(source_files), "training": False,
        "scope": "All completed captures, fixed calibration, shared inputs; no GT accuracy or metric distance claim."}
    root.mkdir(parents=True)
    save_source_snapshot(root, protocol["source_sha256"])
    with np.load(dependencies["rectification"], allow_pickle=False) as maps:
        for sample in samples:
            source: Path = session / sample["pair_id"]
            output: Path = root / sample["pair_id"]
            output.mkdir()
            inputs: dict[str, NDArray[np.uint8]] = calibrated_inputs(cv2.imread(str(source / "left.png")), cv2.imread(str(source / "right.png")), maps, raw["image_scale"])
            for name, image in inputs.items():
                if not cv2.imwrite(str(output / f"{name}.png"), image):
                    raise OSError("Failed to save shared calibrated input")
            protocol["samples"].append({**sample, "input_sha256": {name: digest(output / f"{name}.png") for name in inputs}, "output_wh": list(inputs["left"].shape[1::-1])})
    write_json(root / "protocol.json", protocol)
    print(json.dumps({"prepared_pairs": len(samples), "display_indices": selected}), flush=True)


def verify(config: Path) -> tuple[dict[str, Any], Path]:
    """Reject changed live source, model bytes, calibration or controls."""
    _, _, root = settings(config)
    protocol: dict[str, Any] = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    if verify_recorded_sources(root, protocol["source_sha256"]):
        raise ValueError("Live source differs from frozen run")
    for name, path in protocol["dependency_paths"].items():
        if digest(Path(path)) != protocol["dependency_sha256"][name]:
            raise ValueError(f"Frozen dependency changed: {name}")
    return protocol, root


def infer(config: Path, method: str) -> None:
    """One saved float32 HxW disparity map per pair; invalid SGBM pixels are NaN."""
    protocol, root = verify(config)
    raw: dict[str, Any] = protocol["settings"]
    if (root / f"{method}_execution.json").exists():
        raise ValueError("Preserve completed inference")
    cv2.setNumThreads(1)
    model: Any = None
    info: dict[str, Any] = {}
    if method != "sgbm":
        from .pretrained_stereo import environment_dll_handle, load_banet, load_igev
        dll_handle: object | None = environment_dll_handle()
        import torch

        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(raw["memory_limit_fraction"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model_raw: dict[str, Any] = dict(protocol["models"][method])
        model_raw["runtime_root"] = str(root / f"{method}_runtime")
        if method == "banet":
            model, info = load_banet(Path(model_raw["source_root"]), Path(model_raw["runtime_root"]), Path(model_raw["checkpoint"]), model_raw["checkpoint_sha256"])
        else:
            model, info = load_igev(config, model_raw)
        model._svp_live_dll_handle = dll_handle
    records: list[dict[str, Any]] = []
    for index, sample in enumerate(protocol["samples"]):
        folder: Path = root / sample["pair_id"]
        arrays: dict[str, NDArray[np.uint8]] = {}
        for name in ("left", "right"):
            if digest(folder / f"{name}.png") != sample["input_sha256"][name]:
                raise ValueError("Shared input PNG changed")
            arrays[name] = cv2.cvtColor(cv2.imread(str(folder / f"{name}.png")), cv2.COLOR_BGR2RGB)

        def call() -> NDArray[np.float32]:
            if method == "sgbm":
                return sgbm_predict(arrays, raw["sgbm"], raw["max_disparity"])
            from .pretrained_stereo import predict_banet_rgb, predict_igev
            return predict_banet_rgb(model, arrays, 192) if method == "banet" else predict_igev(model, arrays, raw["igev_iterations"])

        if index == 0:
            call()
        if method != "sgbm":
            torch.cuda.synchronize()
        started: float = time.perf_counter()
        prediction: NDArray[np.float32] = call()
        if method != "sgbm":
            torch.cuda.synchronize()
        elapsed_ms: float = (time.perf_counter() - started) * 1000
        if prediction.dtype != np.float32 or prediction.shape != arrays["left"].shape[:2] or np.isinf(prediction).any():
            raise ValueError("Wrong disparity shape/dtype/values")
        np.save(folder / f"{method}.npy", prediction)
        records.append({"pair_id": sample["pair_id"], "elapsed_ms": elapsed_ms, "sha256": digest(folder / f"{method}.npy")})
        print(json.dumps({"method": method, "completed": index + 1, "total": len(protocol["samples"])}), flush=True)
    write_json(root / f"{method}_execution.json", {"method": method, "model": info, "records": records, "training": False})


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/camera_comparison.local.json"))
    parser.add_argument("--phase", choices=("prepare", "infer"), required=True)
    parser.add_argument("--method", choices=("sgbm", "banet", "igev"))
    args: argparse.Namespace = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.config.resolve())
    elif args.method is None:
        parser.error("--method required for inference")
    else:
        infer(args.config.resolve(), args.method)
