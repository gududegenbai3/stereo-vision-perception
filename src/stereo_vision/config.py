"""Parse configuration once at the application boundary."""
from __future__ import annotations

import json
from pathlib import Path

from stereo_vision.models import RunConfig, SgbmConfig


def load_config(path: Path) -> RunConfig:
    """Read strict JSON; relative paths resolve against the configuration file.

    Raises FileNotFoundError for missing files, ValueError for invalid settings.
    """
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    required: set[str] = {"data_root", "output_root", "scene", "split", "threads", "sgbm"}
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError(f"{path}: expected fields {sorted(required)}")
    settings: object = raw["sgbm"]
    if not isinstance(settings, dict) or set(settings) != set(SgbmConfig.__dataclass_fields__):
        raise ValueError(f"{path}: invalid sgbm fields")
    for key, value in settings.items():
        if type(value) is not int:
            raise ValueError(f"{path}: sgbm.{key} must be an integer")
    sgbm: SgbmConfig = SgbmConfig(**settings)
    if sgbm.block_size < 1 or sgbm.block_size % 2 != 1:
        raise ValueError(f"{path}: block_size must be a positive odd integer")
    if not 0 < sgbm.p1_scale < sgbm.p2_scale:
        raise ValueError(f"{path}: require 0 < p1_scale < p2_scale")
    if not 0 <= sgbm.uniqueness_ratio <= 100 or min(sgbm.speckle_window_size, sgbm.speckle_range) < 0:
        raise ValueError(f"{path}: invalid matching filters")
    if type(raw["threads"]) is not int or raw["threads"] < 1:
        raise ValueError(f"{path}: threads must be a positive integer")
    for key in ("data_root", "output_root", "scene", "split"):
        if not isinstance(raw[key], str) or not raw[key]:
            raise ValueError(f"{path}: {key} must be a nonempty string")
    if raw["split"] not in {"trainingQ", "testQ"}:
        raise ValueError(f"{path}: split must be trainingQ or testQ")
    if Path(raw["scene"]).name != raw["scene"] or raw["scene"] in {".", ".."}:
        raise ValueError(f"{path}: scene must be a single directory name")
    return RunConfig(
        data_root=(path.parent / raw["data_root"]).resolve(),
        output_root=(path.parent / raw["output_root"]).resolve(),
        scene=raw["scene"], split=raw["split"], threads=raw["threads"], sgbm=sgbm,
    )
