"""Protect experiment identity and historical evidence independently of model output."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from src.provenance import file_hashes, require_file_hashes, save_source_snapshot, verify_recorded_sources


class ExperimentProvenanceContracts(unittest.TestCase):
    def test_external_baseline_and_diagnosis_changes_reject_frozen_identity(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            paths: dict[str, Path] = {name: Path(folder) / name for name in ("experiment", "baseline", "diagnosis")}
            for name, path in paths.items():
                path.write_text(name, encoding="utf-8")
            frozen: dict[str, str] = file_hashes(paths)
            require_file_hashes(frozen, paths)
            for name in ("baseline", "diagnosis"):
                original: bytes = paths[name].read_bytes()
                paths[name].write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "dependency"):
                    require_file_hashes(frozen, paths)
                paths[name].write_bytes(original)
            with self.assertRaises(ValueError):
                require_file_hashes({}, paths)

    def test_snapshot_exact_bytes_and_tamper_detection(self) -> None:
        source: str = "src/provenance.py"
        frozen: dict[str, str] = file_hashes({source: Path(source)})
        with tempfile.TemporaryDirectory() as folder:
            root: Path = Path(folder)
            save_source_snapshot(root, frozen)
            snapshot: Path = root / "source_snapshot" / source
            self.assertEqual(snapshot.read_bytes(), Path(source).read_bytes())
            self.assertEqual(verify_recorded_sources(root, frozen), [])
            snapshot.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "evidence mismatch"):
                verify_recorded_sources(root, frozen)
            with self.assertRaisesRegex(ValueError, "Refusing"):
                save_source_snapshot(root, frozen)

    def test_historical_snapshot_distinguishes_later_source_and_missing_legacy(self) -> None:
        source: str = "src/provenance.py"
        previous: bytes = b"historical executed bytes"
        hashes: dict[str, str] = {source: hashlib.sha256(previous).hexdigest()}
        with tempfile.TemporaryDirectory() as folder:
            root: Path = Path(folder)
            with self.assertRaises(ValueError):
                verify_recorded_sources(root, hashes)
            snapshot: Path = root / "source_snapshot" / source
            snapshot.parent.mkdir(parents=True)
            snapshot.write_bytes(previous)
            self.assertEqual(verify_recorded_sources(root, hashes), [source])

    def test_snapshot_rejects_escaping_path_and_changed_live_source(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root: Path = Path(folder)
            with self.assertRaises(ValueError):
                save_source_snapshot(root, {"../outside.py": "unused"})
            with self.assertRaises(ValueError):
                save_source_snapshot(root, {"src/provenance.py": "0" * 64})


if __name__ == "__main__":
    unittest.main()
