"""Keep OpenCV calibration and optional Torch OpenMP runtimes in separate processes."""

from pathlib import Path
import subprocess
import sys
import unittest


class IsolatedCalibrationTests(unittest.TestCase):
    def test_known_camera_geometry_and_scale_in_clean_opencv_process(self) -> None:
        result: subprocess.CompletedProcess[str] = subprocess.run(
            [sys.executable, "-s", "-m", "tests.calibration_checks", "-q"],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Ran 4 tests", result.stderr)


if __name__ == "__main__":
    unittest.main()
