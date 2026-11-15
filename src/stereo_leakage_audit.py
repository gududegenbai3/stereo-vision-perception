"""Reconstruct local training exposure and audit evaluation-image fingerprints."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .banet import environment_dll_handle
from .provenance import verify_recorded_sources
from .stereo_comparison_data import ComparisonScene
from .stereo_training_data import load_training_sample


def image_fingerprint(path: Path) -> dict:
    """Encoded SHA, decoded RGB SHA and 63-bit perceptual DCT hash.

    Exact hashes detect byte/pixel duplicates; perceptual distance is a screening
    tool for resized or recompressed copies, not proof of scene independence.
    """
    encoded: bytes = path.read_bytes()
    image: np.ndarray | None = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    rgb: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small: np.ndarray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coefficients: np.ndarray = cv2.dct(small)[:8, :8].reshape(-1)[1:]
    bits: np.ndarray = coefficients > np.median(coefficients)
    perceptual: int = sum(int(bit) << index for index, bit in enumerate(bits))
    header: bytes = str(rgb.shape).encode()
    return {"path": str(path.resolve()), "encoded_sha256": hashlib.sha256(encoded).hexdigest(),
        "decoded_sha256": hashlib.sha256(header + rgb.tobytes()).hexdigest(), "phash63": f"{perceptual:016x}"}


def audit_training_run(protocol_path: Path, checkpoint_path: Path, data_root: Path) -> dict:
    """Require unchanged frozen code/data and replay every crop supervision count."""
    protocol: dict = json.loads(protocol_path.read_text(encoding="utf-8"))
    identity_payload: dict = {key: value for key, value in protocol.items() if key != "identity_sha256"}
    if hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest() != protocol["identity_sha256"]:
        raise ValueError("Frozen training protocol identity no longer matches its contents")
    changed_sources: list[str] = verify_recorded_sources(protocol_path.parent, protocol["source_sha256"])
    for filename, expected in protocol["files_sha256"].items():
        if hashlib.sha256((data_root / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Training/validation/holdout file changed: {filename}")
    settings: dict = protocol["settings"]
    groups: dict = settings["split_groups"]
    flattened: list[str] = [group for values in groups.values() for group in values]
    if len(set(flattened)) != len(flattened):
        raise ValueError("Overlapping train/validation/test physical groups")
    dll_handle: object | None = environment_dll_handle()
    import torch

    state: dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if state["identity_sha256"] != protocol["identity_sha256"] or state["step"] != settings["steps"]:
        raise ValueError("Checkpoint is not the frozen completed training run")
    samples: list = [load_training_sample(data_root, name) for name in protocol["splits"]["train"]]
    history: list[dict] = json.loads((protocol_path.parent / "training_history.json").read_text(encoding="utf-8"))
    if state["history"] != history:
        raise ValueError("Checkpoint and external training history differ")
    rng: np.random.Generator = np.random.default_rng(settings["seed"])
    height, width = settings["crop_hw"]
    columns: np.ndarray = np.arange(width)[None, :]
    seen: set[str] = set()
    trace: list[dict] = []
    for step in range(settings["steps"]):
        count: int = 0
        for _ in range(settings["batch_size"]):
            sample = samples[int(rng.integers(len(samples)))]
            y: int = int(rng.integers(sample.disparity_px.shape[0] - height + 1))
            x: int = int(rng.integers(sample.disparity_px.shape[1] - width + 1))
            gain: float = float(rng.uniform(*settings["brightness_range"]))
            gamma: float = float(rng.uniform(*settings["gamma_range"]))
            disparity: np.ndarray = sample.disparity_px[y:y + height, x:x + width]
            valid: np.ndarray = sample.known_mask[y:y + height, x:x + width] & (disparity > 0) & (disparity < 192) & (columns - disparity >= 0)
            count += int(valid.sum())
            seen.add(sample.name)
            trace.append({"step": step + 1, "sample": sample.name, "x": x, "y": y, "brightness": gain, "gamma": gamma})
        if count != history[step]["supervised_pixels"]:
            raise ValueError(f"Training crop replay differs at step {step + 1}")
    if rng.bit_generator.state != state["numpy_rng"]:
        raise ValueError("Replayed sampling RNG does not match saved training state")
    forbidden: set[str] = set(protocol["splits"]["validation"] + protocol["splits"]["test"])
    if seen & forbidden:
        raise ValueError("Validation/test sample used in training")
    return {"passed": True, "steps_replayed": settings["steps"], "crops_replayed": len(trace),
        "all_step_supervised_counts_match": True, "final_sampling_rng_matches": True,
        "seen_samples": sorted(seen), "forbidden_samples_seen": sorted(seen & forbidden),
        "frozen_source_and_data_hashes_match": True, "current_sources_changed_since_execution": changed_sources,
        "physical_groups_disjoint": True,
        "crop_trace_sha256": hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(), "trace": trace}


def audit_evaluation_images(protocol: dict, data_root: Path, scenes: list[ComparisonScene], review_distance: int) -> dict:
    """Compare all evaluation views against both local training and selection views."""
    training: list[dict] = []
    for split in ("train", "validation"):
        for name in protocol["splits"][split]:
            for filename in ("im0.png", "im1.png"):
                training.append({"split": split, **image_fingerprint(data_root / name / filename)})
    comparisons: list[dict] = []
    duplicates: list[dict] = []
    candidates: list[dict] = []
    for scene in scenes:
        for path in scene.image_paths:
            current: dict = image_fingerprint(path)
            for previous in training:
                if current["encoded_sha256"] == previous["encoded_sha256"] or current["decoded_sha256"] == previous["decoded_sha256"]:
                    duplicates.append({"evaluation": current["path"], "training_or_validation": previous["path"]})
            nearest: dict = min(training, key=lambda item: (int(current["phash63"], 16) ^ int(item["phash63"], 16)).bit_count())
            distance: int = (int(current["phash63"], 16) ^ int(nearest["phash63"], 16)).bit_count()
            row: dict = {"dataset": scene.dataset, "sample": scene.name, "physical_group": scene.group,
                **current, "nearest_training_or_validation": nearest["path"], "phash_distance": distance}
            comparisons.append(row)
            if distance <= review_distance:
                candidates.append(row)
    if duplicates:
        raise ValueError(f"Exact training/validation image duplicates in evaluation: {duplicates}")
    return {"passed_exact_duplicate_check": True, "evaluation_image_count": len(comparisons),
        "training_and_validation_image_count": len(training), "exact_duplicates": duplicates,
        "perceptual_review_candidates": candidates, "nearest_comparisons": comparisons,
        "limit": "No local fine-tuning/selection overlap detected. Perceptual screening is not a proof for all physical scenes or undisclosed author pretraining data."}
