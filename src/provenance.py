"""Freeze configuration identities and retain the source bytes used by experiments."""

import hashlib
from collections.abc import Mapping
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
import urllib.request


def file_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    """Hash explicit dependency files; missing files raise FileNotFoundError."""
    return {label: hashlib.sha256(path.read_bytes()).hexdigest() for label, path in paths.items()}


def require_file_hashes(recorded: Mapping[str, str], paths: Mapping[str, Path]) -> None:
    """Reject changed or incomplete experiment dependencies before evaluation/resume."""
    if dict(recorded) != file_hashes(paths):
        raise ValueError("Experiment dependency files changed or were not frozen")


def save_source_snapshot(output_root: Path, hashes: Mapping[str, str]) -> None:
    """Copy exact project-relative source bytes at protocol creation, never replace history."""
    for filename, expected in hashes.items():
        relative: Path = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Snapshot source must be project-relative: {filename}")
        payload: bytes = relative.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Source changed while freezing protocol: {filename}")
        target: Path = output_root / "source_snapshot" / relative
        if target.exists() and target.read_bytes() != payload:
            raise ValueError(f"Refusing to replace recorded source: {filename}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)


def verify_recorded_sources(output_root: Path, hashes: Mapping[str, str]) -> list[str]:
    """Verify executed snapshots; report later working-source edits separately.

    Legacy records without snapshots are accepted only when current bytes still
    exactly match the recorded hash. This checks historical evidence, not whether
    the current code can resume a run; resume must check its full live identity.
    """
    changed: list[str] = []
    for filename, expected in hashes.items():
        relative: Path = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid recorded source path: {filename}")
        snapshot: Path = output_root / "source_snapshot" / relative
        evidence: Path = snapshot if snapshot.exists() else relative
        if hashlib.sha256(evidence.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Executed source evidence mismatch: {filename}")
        if not relative.exists() or hashlib.sha256(relative.read_bytes()).hexdigest() != expected:
            changed.append(filename)
    return changed


def download_checked(url: str, target: Path, expected_sha256: str) -> Path:
    """Return a verified cached file or atomically install one from its source."""
    def matches(path: Path) -> bool:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest() == expected_sha256

    if target.is_file():
        if not matches(target):
            raise ValueError(f"Wrong file SHA-256: {target}")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Stereo course research"})
    with NamedTemporaryFile(dir=target.parent, suffix=".download", delete=False) as stream:
        partial: Path = Path(stream.name)
    try:
        with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if not matches(partial):
            raise ValueError(f"Downloaded file SHA-256 differs: {url}")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target
