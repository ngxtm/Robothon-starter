import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark import summarize_benchmark


class BenchmarkTests(unittest.TestCase):
    def test_summarizes_success_rate_and_error(self):
        trials = [
            {
                "trial_index": 0,
                "success": True,
                "sorted_count": 4,
                "object_count": 4,
                "per_object_metrics": {
                    "red": {
                        "placement_error_m": 0.01,
                        "perceived_category": "emergency",
                        "target_label": "Emergency",
                    },
                    "blue": {
                        "placement_error_m": 0.02,
                        "perceived_category": "antibiotic",
                        "target_label": "Antibiotic",
                    },
                },
                "recovery_count": 1,
            },
            {
                "trial_index": 1,
                "success": False,
                "sorted_count": 3,
                "object_count": 4,
                "per_object_metrics": {
                    "red": {
                        "placement_error_m": 0.03,
                        "perceived_category": "emergency",
                        "target_label": "Emergency",
                    },
                },
                "recovery_count": 0,
            },
        ]

        summary = summarize_benchmark(trials)

        self.assertEqual(summary["trials"], 2)
        self.assertEqual(summary["successful_trials"], 1)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["mean_placement_error_m"], 0.02)
        self.assertEqual(summary["total_recoveries"], 1)


if __name__ == "__main__":
    unittest.main()
