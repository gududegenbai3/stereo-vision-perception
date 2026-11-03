"""CLI selects inputs and configuration; algorithms are shared with Notebook."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import cv2

from stereo_vision.config import load_config
from stereo_vision.datasets import load_middlebury_scene
from stereo_vision.models import PipelineResult, RunConfig, StereoScene
from stereo_vision.outputs import save_outputs
from stereo_vision.pipeline import reconstruct_scene

DEFAULT_CONFIG: Path = Path(__file__).resolve().parents[2] / "config" / "middlebury.json"


def main(argv: list[str] | None = None) -> int:
    """Run one scene; return 2 and preserve scene identity when processing fails."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Middlebury OpenCV SGBM baseline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, help="override dataset root")
    parser.add_argument("--scene", help="override configured scene")
    parser.add_argument("--split", choices=["trainingQ", "testQ"])
    parser.add_argument("--out", type=Path, help="exact output directory; default output_root/split/scene")
    args: argparse.Namespace = parser.parse_args(argv)
    try:
        config: RunConfig = load_config(args.config)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Configuration error: {exc}\n")
    config = replace(
        config, data_root=args.root.resolve() if args.root else config.data_root,
        scene=args.scene if args.scene is not None else config.scene,
        split=args.split if args.split is not None else config.split,
    )
    if not config.scene or Path(config.scene).name != config.scene or config.scene in {".", ".."}:
        parser.exit(2, "Scene must be a single directory name\n")
    output_dir: Path = args.out if args.out is not None else config.output_root / config.split / config.scene
    try:
        cv2.setNumThreads(config.threads)
        scene: StereoScene = load_middlebury_scene(config.data_root, config.scene, config.split)
        result: PipelineResult = reconstruct_scene(scene, config.sgbm)
        save_outputs(output_dir, scene, result, config)
    except (OSError, ValueError, cv2.error) as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "failure.json").write_text(json.dumps({
            "scene": config.scene, "split": config.split, "status": "failed", "reason": str(exc),
        }, indent=2), encoding="utf-8")
        parser.exit(2, f"Error in {config.split}/{config.scene}: {exc}\n")
    (output_dir / "failure.json").unlink(missing_ok=True)
    print(f"Saved {config.split}/{config.scene} to {output_dir}")
    if result.metrics is not None:
        print(f"Disparity evaluation: {result.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
