import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from audit import build_audit_package


def sample_summary():
    return {
        "project": "Aegis DexTriage",
        "version": "3.0",
        "success": True,
        "sorted_count": 2,
        "object_count": 2,
        "recovery_count": 1,
    }


def sample_result():
    return {
        "events": [
            {
                "step": 0,
                "kind": "SCAN",
                "object": "red_emergency_box",
                "phase": "approach",
                "category": "emergency",
                "perceived_category": "emergency",
                "perception_confidence": 0.98,
                "perception_evidence": {"shape": "box", "color": "red", "marker": "cross"},
                "triage_priority": 0,
                "triage_reason": "classified",
                "policy_target_category": "emergency",
                "target_tray": "tray_emergency",
                "contact_confirmed": False,
                "placement_success": None,
            },
            {
                "step": 42,
                "kind": "PLACE",
                "object": "red_emergency_box",
                "phase": "release",
                "category": "emergency",
                "perceived_category": "emergency",
                "policy_target_category": "emergency",
                "target_tray": "tray_emergency",
                "contact_confirmed": True,
                "placement_success": True,
            },
        ],
        "per_object_metrics": {
            "red_emergency_box": {
                "label": "Emergency Box",
                "perceived_category": "emergency",
                "perception_confidence": 0.98,
                "target_tray": "tray_emergency",
                "target_label": "Emergency",
                "selected_grasp": "side_pinch",
                "placement_error_m": 0.01,
                "sorted": True,
                "verified": True,
                "contact_confirmations": 3,
            },
            "white_unknown_marker": {
                "label": "Unknown Marker",
                "perceived_category": "inspect",
                "perception_confidence": 0.86,
                "target_tray": "tray_inspect",
                "target_label": "Inspect",
                "selected_grasp": "enclosing",
                "placement_error_m": 0.02,
                "sorted": True,
                "verified": True,
                "contact_confirmations": 2,
            },
        },
    }


class AuditTests(unittest.TestCase):
    def test_builds_clinical_audit_package(self):
        package = build_audit_package(
            sample_summary(),
            sample_result(),
            artifact_paths={"audit_report": "outputs/audit_report.json"},
        )

        audit = package["audit_report"]
        self.assertEqual(audit["version"], "3.0")
        self.assertEqual(audit["sorted_count"], 2)
        self.assertEqual(audit["objects"][0]["risk_score"], 100)
        self.assertEqual(audit["objects"][0]["priority"], 0)
        self.assertTrue(audit["objects"][0]["verified"])

        timeline = package["triage_timeline"]
        self.assertEqual(timeline["events"][0]["kind"], "SCAN")
        self.assertEqual(timeline["events"][1]["verified"], True)

        matrix = package["confusion_matrix"]
        self.assertEqual(matrix["matrix"]["emergency"]["emergency"], 1)
        self.assertEqual(matrix["matrix"]["inspect"]["inspect"], 1)

        submission = package["submission_summary"]
        self.assertIn("audit_report", submission["artifacts"])
        self.assertEqual(submission["rubric_evidence"]["clinical_audit"], "available")


if __name__ == "__main__":
    unittest.main()
