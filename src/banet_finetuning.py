"""Frozen scene-disjoint ETH3D pilot: pretrained -> supervised fine-tuning -> test."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .banet import load_banet
from .banet_training import fixed_domain_metrics, pool_training_metrics, predict_raw, set_training_mode, supervised_loss
from .error_diagnosis import write_json
from .provenance import save_source_snapshot
from .stereo_training_data import data_fingerprint, freeze_split, load_training_sample, training_crop


def evaluate(model: object, samples: list, max_disparity: int) -> dict:
    """Full original resolution, fixed GT domain, no random validation transforms."""
    rows: list[dict] = [fixed_domain_metrics(predict_raw(model, sample, max_disparity), sample, max_disparity)
                        for sample in samples]
    return {"pooled": pool_training_metrics(rows), "samples": rows}


def main() -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/banet_finetuning.json"))
    parser.add_argument("--resume", action="store_true", help="Restore optimizer, schedule and RNG at last validation")
    args: argparse.Namespace = parser.parse_args()
    raw: dict = json.loads(args.config.read_text(encoding="utf-8"))
    model_path: Path = args.config.parent / raw["model_config"]
    model_raw: dict = json.loads(model_path.read_text(encoding="utf-8"))
    if raw["precision"] != "float32" or raw["max_disparity"] != 192:
        raise ValueError("This frozen pilot uses FP32 and the official 192-disparity architecture")
    if raw["batch_size"] < 1 or raw["steps"] < 1 or raw["validation_interval"] < 1:
        raise ValueError("Positive batch size, steps and validation interval required")
    if any(value <= 0 or value % 32 for value in raw["crop_hw"]):
        raise ValueError("Training crop dimensions must be positive multiples of 32")
    root: Path = (args.config.parent / raw["output_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "test_report.json").exists():
        raise ValueError("Test has already been evaluated; do not repeatedly tune on this holdout")
    if (root / "protocol.json").exists() and not args.resume:
        raise ValueError("Started run exists; use --resume or a fresh output_root")
    data_root: Path = (args.config.parent / raw["data_root"]).resolve()
    for archive in raw["archives"]:
        digest: str = hashlib.sha256((data_root / archive["filename"]).read_bytes()).hexdigest()
        if digest != archive["sha256"]:
            raise ValueError(f"Dataset archive hash mismatch: {archive['filename']}")
    names: list[str] = sorted(folder.parent.name for folder in data_root.glob("*/im0.png"))
    splits: dict[str, list[str]] = freeze_split(names, raw["split_groups"])
    if {key: len(values) for key, values in splits.items()} != raw["expected_split_counts"]:
        raise ValueError("Unexpected dataset/split sample counts")
    source_paths: list[str] = ["src/banet.py", "src/stereo_training_data.py", "src/banet_training.py", "src/banet_finetuning.py", "src/provenance.py", "requirements-banet.txt"]
    identity: dict = {"settings": raw, "splits": splits, "files_sha256": data_fingerprint(data_root, names),
        "model_config_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in source_paths}}
    identity_hash: str = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if args.resume:
        stored: dict = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
        if stored["identity_sha256"] != identity_hash:
            raise ValueError("Resume data/config/source differs from frozen protocol")
    else:
        save_source_snapshot(root, identity["source_sha256"])
        write_json(root / "protocol.json", {"identity_sha256": identity_hash, **identity})
    model, info = load_banet((model_path.parent / model_raw["source_root"]).resolve(),
        (model_path.parent / model_raw["runtime_root"]).resolve(), (model_path.parent / model_raw["checkpoint"]).resolve(),
        model_raw["checkpoint_sha256"])
    import torch

    cv2.setNumThreads(1)
    torch.manual_seed(raw["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rng: np.random.Generator = np.random.default_rng(raw["seed"])
    write_json(root / "model.json", info)
    initial: dict = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    train_samples: list = [load_training_sample(data_root, name) for name in splits["train"]]
    validation_samples: list = [load_training_sample(data_root, name) for name in splits["validation"]]
    optimizer = torch.optim.AdamW(model.parameters(), lr=raw["max_learning_rate"], weight_decay=raw["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=raw["max_learning_rate"], total_steps=raw["steps"],
        pct_start=raw["one_cycle_pct_start"], cycle_momentum=False, anneal_strategy="cos", div_factor=10., final_div_factor=100.)
    first_step: int = 0
    history: list[dict] = []
    validation_history: list[dict] = []
    best_step: int = 0
    elapsed_previous: float = 0.
    peak_allocated_previous: float = 0.
    peak_reserved_previous: float = 0.
    if args.resume:
        saved: dict = torch.load(root / "checkpoint_last.pth", map_location="cpu", weights_only=True)
        if saved["identity_sha256"] != identity_hash:
            raise ValueError("Resume checkpoint identity mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        first_step, best_step = saved["step"], saved["best_step"]
        best_score: float = saved["best_score"]
        history, validation_history = saved["history"], saved["validation_history"]
        elapsed_previous = saved["elapsed_seconds"]
        peak_allocated_previous = saved["peak_allocated_mib"]
        peak_reserved_previous = saved["peak_reserved_mib"]
    else:
        baseline_validation: dict = evaluate(model, validation_samples, raw["max_disparity"])
        best_score = baseline_validation["pooled"]["mae_px"]
        validation_history.append({"step": 0, **baseline_validation})
        torch.save(initial, root / "checkpoint_best.pth")
        write_json(root / "baseline_validation.json", baseline_validation)
        print(json.dumps({"step": 0, "validation": baseline_validation["pooled"]}), flush=True)
    started: float = time.perf_counter()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for step in range(first_step + 1, raw["steps"] + 1):
        step_start: float = time.perf_counter()
        set_training_mode(model, raw["freeze_batch_norm_statistics"])
        crops: list[tuple[np.ndarray, ...]] = [training_crop(train_samples[int(rng.integers(len(train_samples)))],
            tuple(raw["crop_hw"]), raw["max_disparity"], rng, tuple(raw["brightness_range"]), tuple(raw["gamma_range"]))
            for _ in range(raw["batch_size"])]
        tensors = [torch.from_numpy(np.stack([crop[index] for crop in crops])).to("cuda") for index in range(4)]
        optimizer.zero_grad(set_to_none=True)
        outputs = model(tensors[0], tensors[1], raw["max_disparity"])
        loss = supervised_loss(outputs, tensors[2], tensors[3], raw["max_disparity"], raw["auxiliary_loss_weight"])
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite loss at step {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), raw["gradient_clip_norm"], error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        row: dict = {"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm),
            "learning_rate": float(scheduler.get_last_lr()[0]), "step_ms": (time.perf_counter() - step_start) * 1000,
            "supervised_pixels": int(tensors[3].sum())}
        history.append(row)
        del outputs, loss, norm, tensors
        if step % raw["log_interval"] == 0:
            print(json.dumps(row), flush=True)
            write_json(root / "training_history.json", history)
        if step % raw["validation_interval"] == 0 or step == raw["steps"]:
            result: dict = evaluate(model, validation_samples, raw["max_disparity"])
            validation_history.append({"step": step, **result})
            score: float = result["pooled"]["mae_px"]
            if score < best_score:
                best_score, best_step = score, step
                torch.save({name: value.detach().cpu().clone() for name, value in model.state_dict().items()}, root / "checkpoint_best.pth")
            elapsed: float = elapsed_previous + time.perf_counter() - started
            peak_allocated: float = max(peak_allocated_previous, torch.cuda.max_memory_allocated() / 1024**2)
            peak_reserved: float = max(peak_reserved_previous, torch.cuda.max_memory_reserved() / 1024**2)
            torch.save({"identity_sha256": identity_hash, "step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(), "best_step": best_step, "best_score": best_score, "history": history,
                "validation_history": validation_history, "elapsed_seconds": elapsed, "peak_allocated_mib": peak_allocated,
                "peak_reserved_mib": peak_reserved}, root / "checkpoint_last.pth")
            write_json(root / "validation_history.json", validation_history)
            write_json(root / "status.json", {"step": step, "best_step": best_step, "best_validation_mae_px": best_score,
                "elapsed_seconds": elapsed, "peak_allocated_mib": peak_allocated, "peak_reserved_mib": peak_reserved})
            print(json.dumps({"step": step, "validation": result["pooled"], "best_step": best_step}), flush=True)
    final_status: dict = json.loads((root / "status.json").read_text(encoding="utf-8"))
    # Only after the training budget and validation-only checkpoint choice are fixed, evaluate test.
    test_samples: list = [load_training_sample(data_root, name) for name in splits["test"]]
    selected: dict = torch.load(root / "checkpoint_best.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(selected, strict=True)
    after: dict = evaluate(model, test_samples, raw["max_disparity"])
    after_predictions: dict[str, np.ndarray] = {sample.name: predict_raw(model, sample, raw["max_disparity"]) for sample in test_samples}
    model.load_state_dict(initial, strict=True)
    before: dict = evaluate(model, test_samples, raw["max_disparity"])
    for sample in test_samples:
        before_prediction: np.ndarray = predict_raw(model, sample, raw["max_disparity"])
        np.savez_compressed(root / f"test-{sample.name}.npz", left=sample.left_rgb, truth=sample.disparity_px,
            reference=sample.non_occluded_mask & (sample.disparity_px > 0) & (sample.disparity_px < raw["max_disparity"]),
            before=before_prediction, after=after_predictions[sample.name])
    report: dict = {"status": "complete", "identity_sha256": identity_hash, "best_step": best_step,
        "before": before, "after": after, "training": final_status,
        "best_checkpoint_sha256": hashlib.sha256((root / "checkpoint_best.pth").read_bytes()).hexdigest(),
        "mean_training_step_ms": float(np.mean([row["step_ms"] for row in history])),
        "all_losses_finite": all(np.isfinite(row["loss"]) for row in history),
        "scope": "Single frozen 1000-step small-data supervised pilot; validation selects checkpoint including step 0; test evaluated after selection; not official leaderboard reproduction"}
    write_json(root / "test_report.json", report)
    print(json.dumps({"status": "complete", "best_step": best_step, "before": before["pooled"], "after": after["pooled"], "training": final_status}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
