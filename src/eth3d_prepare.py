"""Download pinned official ETH3D archives and extract using a 7z-capable bsdtar."""

import argparse
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

from .error_diagnosis import write_json


def prepare_archive(root: Path, filename: str, url: str, expected_sha256: str) -> dict:
    """Download to a temporary file, require SHA256, reject paths escaping root.

    Network/file errors propagate. A local bsdtar supporting 7z is required;
    Windows' bundled tar was used for this experiment, with no new Python deps.
    """
    root.mkdir(parents=True, exist_ok=True)
    archive: Path = root / filename
    if not archive.exists():
        temporary: Path = archive.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as target:
            while block := response.read(1024 * 1024):
                target.write(block)
        digest: str = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError(f"Downloaded archive hash mismatch: {filename}")
        temporary.replace(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"Local archive hash mismatch: {filename}")
    members: list[str] = subprocess.check_output(["tar", "-tf", str(archive)], text=True).splitlines()
    for member in members:
        if not (root / member).resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Unsafe archive path: {member}")
    subprocess.run(["tar", "-xf", str(archive), "-C", str(root)], check=True)
    return {"filename": filename, "url": url, "bytes": archive.stat().st_size, "sha256": digest}


def main() -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/banet_finetuning.json"))
    args: argparse.Namespace = parser.parse_args()
    raw: dict = json.loads(args.config.read_text(encoding="utf-8"))
    root: Path = (args.config.parent / raw["data_root"]).resolve()
    rows: list[dict] = [prepare_archive(root, entry["filename"], entry["url"], entry["sha256"]) for entry in raw["archives"]]
    write_json(root / "downloads.json", {"dataset_url": raw["dataset_url"], "license": raw["license"], "archives": rows})
    print(json.dumps({"data_root": str(root), "pairs": len(list(root.glob("*/im0.png"))), "archives": rows}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
