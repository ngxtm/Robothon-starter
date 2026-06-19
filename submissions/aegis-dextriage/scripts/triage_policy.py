from __future__ import annotations

from dataclasses import dataclass

from perception import PerceptionResult


@dataclass(frozen=True)
class TriageAction:
    object_name: str
    target_category: str
    priority: int
    grasp_override: str | None
    requires_verification: bool
    reason: str


_PRIORITY = {
    "emergency": 0,
    "antibiotic": 1,
    "pain_relief": 2,
    "inspect": 3,
}

_GRASP_BY_SHAPE = {
    "box": "side_pinch",
    "cylinder": "wrap",
    "sphere": "enclosing",
}


def decide_triage_action(perception: PerceptionResult) -> TriageAction:
    target = perception.category
    reason = "classified"

    if perception.confidence < 0.80:
        target = "inspect"
        reason = "low_confidence"
    elif perception.requires_inspection:
        target = "inspect"
        reason = "requires_inspection"

    return TriageAction(
        object_name=perception.object_name,
        target_category=target,
        priority=_PRIORITY[target],
        grasp_override=_GRASP_BY_SHAPE.get(perception.shape),
        requires_verification=True,
        reason=reason,
    )
