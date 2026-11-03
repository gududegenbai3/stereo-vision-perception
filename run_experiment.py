"""Repository-root launcher for the stereo_vision package CLI."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from stereo_vision.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
