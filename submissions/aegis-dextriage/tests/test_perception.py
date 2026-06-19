import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from perception import SimulatedObservation, classify_observation


class PerceptionTests(unittest.TestCase):
    def test_red_box_classifies_as_emergency(self):
        observation = SimulatedObservation(
            object_name="red_emergency_box",
            shape="box",
            color="red",
            marker="cross",
            damaged=False,
            expired=False,
        )

        result = classify_observation(observation)

        self.assertEqual(result.category, "emergency")
        self.assertEqual(result.confidence, 0.98)
        self.assertEqual(result.evidence["shape"], "box")
        self.assertEqual(result.evidence["marker"], "cross")

    def test_unknown_or_damaged_object_routes_to_inspect(self):
        observation = SimulatedObservation(
            object_name="white_unknown_marker",
            shape="sphere",
            color="white",
            marker="inspection_dot",
            damaged=True,
            expired=False,
        )

        result = classify_observation(observation)

        self.assertEqual(result.category, "inspect")
        self.assertLess(result.confidence, 0.80)
        self.assertTrue(result.requires_inspection)


if __name__ == "__main__":
    unittest.main()
