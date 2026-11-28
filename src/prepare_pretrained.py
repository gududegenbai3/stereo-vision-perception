"""Fetch pinned official stereo inference inputs and verify SHA-256 before use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .provenance import download_checked


def prepare(config: Path, verify_only: bool) -> dict[str, str]:
    models: dict = json.loads(config.read_text(encoding="utf-8"))
    status: dict[str, str] = {}
    for method, spec in models.items():
        repository: str = spec["upstream_repository"].removeprefix("https://github.com/")
        base: str = f"https://raw.githubusercontent.com/{repository}/{spec['upstream_commit']}/{spec['upstream_core']}"
        source: Path = (config.parent / spec["source_root"]).resolve()
        for name, expected in spec["source_sha256"].items():
            path: Path = source / name
            if verify_only and not path.is_file():
                raise FileNotFoundError(path)
            download_checked(f"{base}/{name}", path, expected)
        checkpoint: Path = (config.parent / spec["checkpoint"]).resolve()
        if verify_only and not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        try:
            download_checked(spec["checkpoint_download_url"], checkpoint, spec["checkpoint_sha256"])
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Check the configured {method} weight. Official download: {spec['checkpoint_url']}, target: {checkpoint}") from error
        status[method] = spec["checkpoint_sha256"]
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/pretrained_stereo.example.json"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        status = prepare(args.config.resolve(), args.verify_only)
    except (OSError, RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
