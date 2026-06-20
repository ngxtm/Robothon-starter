import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from perception_report import build_perception_report


class PerceptionReportTests(unittest.TestCase):
    def test_groups_scan_evidence_by_object(self):
        summary = {"version": "3.0", "trial_index": 0, "trial_seed": 7}
        result = {
            "events": [
                {
                    "kind": "SCAN",
                    "object": "blue_antibiotic_bottle",
                    "perceived_category": "antibiotic",
                    "perception_confidence": 0.94,
                    "perception_evidence": {
                        "shape": "cylinder",
                        "color": "blue",
                        "marker": "cap_band",
                        "damaged": False,
                        "expired": False,
                    },
                    "policy_target_category": "antibiotic",
                    "triage_reason": "classified",
                },
                {
                    "kind": "GRASP",
                    "object": "blue_antibiotic_bottle",
                    "perceived_category": "antibiotic",
                },
            ],
            "per_object_metrics": {
                "blue_antibiotic_bottle": {
                    "label": "Antibiotic Bottle",
                    "perceived_category": "antibiotic",
                    "perception_confidence": 0.94,
                    "verified": True,
                }
            },
        }

        report = build_perception_report(summary, result)

        self.assertEqual(report["version"], "3.0")
        item = report["objects"]["blue_antibiotic_bottle"]
        self.assertEqual(item["detected_cues"]["shape"], "cylinder")
        self.assertEqual(item["policy_target_category"], "antibiotic")
        self.assertEqual(item["triage_reason"], "classified")
        self.assertTrue(item["verified"])


if __name__ == "__main__":
    unittest.main()
