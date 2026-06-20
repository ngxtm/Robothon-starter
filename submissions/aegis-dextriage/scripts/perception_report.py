from __future__ import annotations

from typing import Any


def _scan_event(events: list[dict[str, Any]], object_name: str) -> dict[str, Any]:
    for event in events:
        if event.get("object") == object_name and event.get("kind") == "SCAN":
            return event
    return {}


def build_perception_report(summary: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    events = list(result.get("events", summary.get("events", [])))
    per_object_metrics = dict(result.get("per_object_metrics", summary.get("per_object_metrics", {})))
    objects: dict[str, dict[str, Any]] = {}

    for object_name, metrics in per_object_metrics.items():
        scan = _scan_event(events, object_name)
        evidence = dict(scan.get("perception_evidence") or {})
        confidence = metrics.get("perception_confidence", scan.get("perception_confidence", 0.0))
        objects[object_name] = {
            "label": metrics.get("label"),
            "detected_category": metrics.get("perceived_category", scan.get("perceived_category")),
            "confidence": confidence,
            "detected_cues": {
                "shape": evidence.get("shape", metrics.get("shape")),
                "color": evidence.get("color"),
                "marker": evidence.get("marker"),
                "damaged": bool(evidence.get("damaged", False)),
                "expired": bool(evidence.get("expired", False)),
            },
            "policy_target_category": scan.get("policy_target_category"),
            "triage_priority": scan.get("triage_priority"),
            "triage_reason": scan.get("triage_reason", "classified"),
            "requires_verification": bool(scan.get("requires_verification", True)),
            "verified": bool(metrics.get("verified", metrics.get("sorted", False))),
        }

    low_confidence = [name for name, item in objects.items() if float(item.get("confidence") or 0.0) < 0.8]
    routed_to_inspect = [
        name
        for name, item in objects.items()
        if item.get("policy_target_category") == "inspect" or item.get("detected_category") == "inspect"
    ]
    return {
        "project": summary.get("project", "Aegis DexTriage"),
        "version": "3.0",
        "trial_index": summary.get("trial_index"),
        "trial_seed": summary.get("trial_seed"),
        "objects": objects,
        "low_confidence_objects": low_confidence,
        "inspect_route_objects": routed_to_inspect,
    }
