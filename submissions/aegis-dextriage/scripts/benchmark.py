from __future__ import annotations

from typing import Any


def summarize_benchmark(trials: list[dict[str, Any]]) -> dict[str, Any]:
    placement_errors: list[float] = []
    total_recoveries = 0
    classified_objects = 0
    recognized_objects = 0
    verified_objects = 0

    for trial in trials:
        total_recoveries += int(trial.get("recovery_count", 0))
        for metrics in trial.get("per_object_metrics", {}).values():
            if "placement_error_m" in metrics:
                placement_errors.append(float(metrics["placement_error_m"]))
            perceived = metrics.get("perceived_category")
            if perceived is not None:
                classified_objects += 1
                if perceived in {"emergency", "antibiotic", "pain_relief", "inspect"}:
                    recognized_objects += 1
            if metrics.get("verified", metrics.get("sorted", False)):
                verified_objects += 1

    successful_trials = sum(1 for trial in trials if trial.get("success"))
    trial_count = len(trials)
    mean_error = sum(placement_errors) / len(placement_errors) if placement_errors else 0.0

    return {
        "trials": trial_count,
        "successful_trials": successful_trials,
        "success_rate": round(successful_trials / trial_count, 3) if trial_count else 0.0,
        "mean_placement_error_m": round(mean_error, 5),
        "total_recoveries": total_recoveries,
        "sorted_objects": sum(int(trial.get("sorted_count", 0)) for trial in trials),
        "total_objects": sum(int(trial.get("object_count", 0)) for trial in trials),
        "verified_objects": verified_objects,
        "classified_objects": classified_objects,
        "classification_accuracy": round(recognized_objects / classified_objects, 3) if classified_objects else 0.0,
    }
