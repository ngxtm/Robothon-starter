import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from perception import PerceptionResult
from triage_policy import decide_triage_action


class TriagePolicyTests(unittest.TestCase):
    def test_emergency_gets_highest_priority(self):
        perception = PerceptionResult(
            object_name="red_emergency_box",
            category="emergency",
            confidence=0.98,
            shape="box",
            color="red",
            marker="cross",
            requires_inspection=False,
            evidence={},
        )

        action = decide_triage_action(perception)

        self.assertEqual(action.target_category, "emergency")
        self.assertEqual(action.priority, 0)
        self.assertEqual(action.grasp_override, "side_pinch")
        self.assertTrue(action.requires_verification)

    def test_low_confidence_routes_to_inspect(self):
        perception = PerceptionResult(
            object_name="ambiguous_item",
            category="antibiotic",
            confidence=0.55,
            shape="cylinder",
            color="blue",
            marker="unknown",
            requires_inspection=False,
            evidence={},
        )

        action = decide_triage_action(perception)

        self.assertEqual(action.target_category, "inspect")
        self.assertEqual(action.reason, "low_confidence")
        self.assertEqual(action.priority, 3)


if __name__ == "__main__":
    unittest.main()
