"""Export final training weights for the existing strict BANet inference loader."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from .banet import environment_dll_handle
from .error_diagnosis import write_json


def export_model_weights(checkpoint: Path, output: Path, expected_sha256: str) -> str:
    """Require original checkpoint hash and finite tensor state; never overwrite output."""
    dll_handle: object | None = environment_dll_handle()
    import torch

    if output.exists():
        raise ValueError(f"Weights already exported: {output}")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Training checkpoint SHA256 mismatch")
    state: dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    weights: dict = state["model"]
    if not weights or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("Invalid/nonfinite model state")
    torch.save(weights, output)
    restored: dict = torch.load(output, map_location="cpu", weights_only=True)
    if restored.keys() != weights.keys() or any(not torch.equal(weights[name], restored[name]) for name in weights):
        raise ValueError("Exported weights differ from final training state")
    return hashlib.sha256(output.read_bytes()).hexdigest()


def write_inference_config(template_path: Path, output: Path, weights: Path, digest: str, protocol: Path) -> None:
    """Create a portable local config for this export, preserving the recorded template."""
    if output.exists():
        raise ValueError(f"Local inference configuration already exists: {output}")
    raw: dict = json.loads(template_path.read_text(encoding="utf-8"))
    for field in ("source_root", "runtime_root"):
        raw[field] = Path(os.path.relpath((template_path.parent/raw[field]).resolve(), output.parent.resolve())).as_posix()
    raw.update({"checkpoint": Path(os.path.relpath(weights.resolve(), output.parent.resolve())).as_posix(),
        "checkpoint_sha256": digest,
        "checkpoint_provenance": Path(os.path.relpath((weights.parent/"export.json").resolve(), output.parent.resolve())).as_posix(),
        "training_protocol": Path(os.path.relpath(protocol.resolve(), output.parent.resolve())).as_posix(),
        "validation_selected": False, "role": "Fixed final-budget experimental candidate"})
    write_json(output, raw)


def main() -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/banet_finetuning.json"))
    parser.add_argument("--model-config-out", type=Path, default=Path("config/banet_finetuned_model.local.json"))
    args: argparse.Namespace = parser.parse_args()
    raw: dict = json.loads(args.config.read_text(encoding="utf-8"))
    root: Path = (args.config.parent / raw["output_root"]).resolve()
    last: dict = json.loads((root / "fixed_last_test.json").read_text(encoding="utf-8"))
    target: Path = root / "checkpoint_finetuned.pth"
    if args.model_config_out.exists():
        raise ValueError(f"Local model config already exists: {args.model_config_out}")
    digest: str = export_model_weights(root / "checkpoint_last.pth", target, last["checkpoint_sha256"])
    report: dict = {"file": target.name, "sha256": digest, "bytes": target.stat().st_size,
        "step": last["step"], "validation_selected_step": last["validation_selected_step"],
        "source_checkpoint_sha256": last["checkpoint_sha256"], "tensor_roundtrip_identical": True,
        "purpose": "Fixed final-step candidate, compatible with load_banet; not the validation-selected default"}
    write_json(root / "export.json", report)
    write_inference_config(args.config.parent/raw["model_config"], args.model_config_out, target, digest, args.config)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
