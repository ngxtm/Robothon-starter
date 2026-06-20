import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from cinematic import camera_for_step, camera_sequence_metadata


class CinematicTests(unittest.TestCase):
    def test_default_camera_when_cinematic_disabled(self):
        self.assertEqual(camera_for_step(20, 100, "demo_camera", False), "demo_camera")

    def test_cinematic_camera_sequence(self):
        self.assertEqual(camera_for_step(0, 100, "demo_camera", True), "demo_camera")
        self.assertEqual(camera_for_step(30, 100, "demo_camera", True), "topdown_camera")
        self.assertEqual(camera_for_step(55, 100, "demo_camera", True), "wrist_camera")
        self.assertEqual(camera_for_step(85, 100, "demo_camera", True), "demo_camera")

    def test_sequence_metadata(self):
        metadata = camera_sequence_metadata(100, "demo_camera", True)
        self.assertEqual(metadata["enabled"], True)
        self.assertEqual(len(metadata["segments"]), 4)
        self.assertEqual(metadata["segments"][2]["camera"], "wrist_camera")


if __name__ == "__main__":
    unittest.main()
