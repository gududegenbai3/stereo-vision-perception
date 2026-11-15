"""Prepare the pinned official mobile evaluation archive; no model computation."""

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

from .error_diagnosis import write_json


def main() -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/banet_sgbm_comparison.json"))
    args: argparse.Namespace = parser.parse_args()
    raw: dict = json.loads(args.config.read_text(encoding="utf-8"))
    archive_path: Path = (args.config.parent / raw["mobile_archive"]).resolve()
    destination: Path = (args.config.parent / raw["mobile_root"]).resolve().parent
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        request: urllib.request.Request = urllib.request.Request(raw["mobile_archive_url"],
            headers={"User-Agent": "Mozilla/5.0", "Referer": raw["mobile_source"]})
        temporary: Path = archive_path.with_suffix(".part")
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as target:
            while block := response.read(4 * 1024 * 1024):
                target.write(block)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != raw["mobile_archive_sha256"]:
            raise ValueError("Downloaded mobile archive hash mismatch")
        temporary.replace(archive_path)
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != raw["mobile_archive_sha256"]:
        raise ValueError("Local mobile archive hash mismatch")
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if not (destination / name).resolve().is_relative_to(destination):
                raise ValueError(f"Archive path escapes dataset root: {name}")
        archive.extractall(destination)
    result: dict = {"url": raw["mobile_archive_url"], "sha256": raw["mobile_archive_sha256"],
        "bytes": archive_path.stat().st_size, "pairs": len(list((args.config.parent / raw["mobile_root"]).glob("*/im0.png")))}
    if result["pairs"] != raw["expected_dataset_counts"]["mobile2021_new"]:
        raise ValueError("Incomplete external evaluation dataset")
    write_json(archive_path.parent / "download.json", result)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
