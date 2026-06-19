from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SimulatedObservation:
    object_name: str
    shape: str
    color: str
    marker: str
    damaged: bool = False
    expired: bool = False


@dataclass(frozen=True)
class PerceptionResult:
    object_name: str
    category: str
    confidence: float
    shape: str
    color: str
    marker: str
    requires_inspection: bool
    evidence: dict[str, Any]


_RULES: dict[tuple[str, str, str], tuple[str, float]] = {
    ("box", "red", "cross"): ("emergency", 0.98),
    ("cylinder", "blue", "cap_band"): ("antibiotic", 0.94),
    ("box", "yellow", "pill"): ("pain_relief", 0.93),
    ("sphere", "white", "inspection_dot"): ("inspect", 0.86),
}


def classify_observation(observation: SimulatedObservation) -> PerceptionResult:
    key = (observation.shape, observation.color, observation.marker)
    category, confidence = _RULES.get(key, ("inspect", 0.62))
    requires_inspection = category == "inspect" or observation.damaged or observation.expired

    if observation.damaged or observation.expired:
        category = "inspect"
        confidence = min(confidence, 0.74)

    return PerceptionResult(
        object_name=observation.object_name,
        category=category,
        confidence=round(confidence, 2),
        shape=observation.shape,
        color=observation.color,
        marker=observation.marker,
        requires_inspection=requires_inspection,
        evidence={
            "shape": observation.shape,
            "color": observation.color,
            "marker": observation.marker,
            "damaged": observation.damaged,
            "expired": observation.expired,
        },
    )
