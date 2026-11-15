"""Measure full BANet FP32 forward/backward/AdamW on disposable synthetic data."""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

from .banet import load_banet
from .error_diagnosis import write_json


def main() -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/banet_training_probe.json"))
    args: argparse.Namespace = parser.parse_args()
    settings: dict = json.loads(args.config.read_text(encoding="utf-8"))
    model_config: Path = args.config.parent / settings["model_config"]
    model_settings: dict = json.loads(model_config.read_text(encoding="utf-8"))
    model, info = load_banet((model_config.parent / model_settings["source_root"]).resolve(),
        (model_config.parent / model_settings["runtime_root"]).resolve(),
        (model_config.parent / model_settings["checkpoint"]).resolve(), model_settings["checkpoint_sha256"])
    import torch
    import torch.nn.functional as functional

    torch.manual_seed(settings["seed"])
    torch.backends.cudnn.benchmark = False
    root: Path = (args.config.parent / settings["output_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "summary.json").exists():
        raise ValueError("Probe already completed; use a new output_root")
    height, width = settings["crop_hw"]
    rows: list[dict] = []
    initial: dict = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    for batch in settings["batch_sizes"]:
        gc.collect()
        torch.cuda.empty_cache()
        model.load_state_dict(initial, strict=True)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=1e-5)
        before = next(model.parameters()).detach().clone()
        try:
            left = torch.rand(batch, 3, height, width, device="cuda") * 255
            right = torch.rand_like(left) * 255
            target = torch.full((batch, 1, height, width), 16., device="cuda")
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times: list[float] = []
            for step in range(settings["warmup_steps"] + settings["timed_steps"]):
                start: float = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                output = model(left, right, 192)
                loss = functional.smooth_l1_loss(output[0], target) + .3 * functional.smooth_l1_loss(output[1], target)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                if step >= settings["warmup_steps"]:
                    times.append((time.perf_counter() - start) * 1000)
            changed: bool = not torch.equal(before, next(model.parameters()).detach())
            if not changed:
                raise ValueError("Training did not update parameters")
            row: dict = {"batch_size": batch, "status": "success", "mean_step_ms": float(np.mean(times)),
                "step_ms": times, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                "last_loss": float(loss.detach()), "last_grad_norm": float(norm), "parameters_updated": changed}
            del left, right, target, output, loss, norm
        except torch.OutOfMemoryError as error:
            row = {"batch_size": batch, "status": "out_of_memory", "error": str(error)}
        rows.append(row)
        write_json(root / "progress.json", {"batches": rows})
        print(json.dumps(row), flush=True)
        del optimizer, before
    write_json(root / "summary.json", {"model": info, "settings": settings, "batches": rows,
        "scope": "Synthetic random images and constant target; full FP32 training step including backward, clipping and AdamW; capacity only, weights discarded, not quality evidence"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
