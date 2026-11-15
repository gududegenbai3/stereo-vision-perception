"""Secondary diagnosis of fixed final-step weights; never select by test metrics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .banet import load_banet
from .banet_training import fixed_domain_metrics, pool_training_metrics, predict_raw
from .error_diagnosis import write_json
from .stereo_training_data import load_training_sample


def main() -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/banet_finetuning.json"))
    args: argparse.Namespace = parser.parse_args()
    raw: dict = json.loads(args.config.read_text(encoding="utf-8"))
    root: Path = (args.config.parent / raw["output_root"]).resolve()
    if (root / "fixed_last_test.json").exists():
        raise ValueError("Final-step diagnosis already exists")
    protocol: dict = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    primary: dict = json.loads((root / "test_report.json").read_text(encoding="utf-8"))
    model_path: Path = args.config.parent / raw["model_config"]
    model_raw: dict = json.loads(model_path.read_text(encoding="utf-8"))
    model, _ = load_banet((model_path.parent / model_raw["source_root"]).resolve(),
        (model_path.parent / model_raw["runtime_root"]).resolve(), (model_path.parent / model_raw["checkpoint"]).resolve(),
        model_raw["checkpoint_sha256"])
    import torch

    state: dict = torch.load(root / "checkpoint_last.pth", map_location="cpu", weights_only=True)
    if state["identity_sha256"] != protocol["identity_sha256"] or state["step"] != raw["steps"]:
        raise ValueError("Final checkpoint is not the frozen-budget endpoint")
    initial: dict = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    changed: list[str] = [name for name, value in initial.items() if not torch.equal(value, state["model"][name])]
    statistics: list[str] = [name for name in initial if name.endswith(("running_mean", "running_var", "num_batches_tracked"))]
    if not changed or any(not torch.equal(initial[name], state["model"][name]) for name in statistics):
        raise ValueError("Expected updated weights with frozen BN statistics")
    for values in state["optimizer"]["state"].values():
        for value in values.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError("Nonfinite optimizer state")
    model.load_state_dict(state["model"], strict=True)
    rows: list[dict] = []
    data_root: Path = (args.config.parent / raw["data_root"]).resolve()
    for name in protocol["splits"]["test"]:
        sample = load_training_sample(data_root, name)
        prediction: np.ndarray = predict_raw(model, sample, raw["max_disparity"])
        rows.append(fixed_domain_metrics(prediction, sample, raw["max_disparity"]))
        np.savez_compressed(root / f"last-test-{name}.npz", prediction=prediction)
    report: dict = {"scope": "Secondary diagnosis of fixed final-budget checkpoint after primary selection; no retraining or test-based model selection",
        "step": state["step"], "validation_selected_step": primary["best_step"], "pooled": pool_training_metrics(rows), "samples": rows,
        "parameters_changed_tensors": len(changed), "bn_statistics_unchanged": len(statistics), "optimizer_state_finite": True,
        "checkpoint_sha256": hashlib.sha256((root / "checkpoint_last.pth").read_bytes()).hexdigest()}
    write_json(root / "fixed_last_test.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
