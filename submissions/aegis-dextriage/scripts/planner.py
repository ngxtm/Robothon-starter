from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MedicationItem:
    name: str
    label: str
    shape: str
    color: str
    marker: str
    category: str
    grasp: str
    initial_pos: tuple[float, float, float]
    held_offset: tuple[float, float, float]
    drop_pos: tuple[float, float, float]


@dataclass(frozen=True)
class Tray:
    name: str
    label: str
    category: str
    center: tuple[float, float, float]
    half_extents: tuple[float, float, float]


TRAYS: tuple[Tray, ...] = (
    Tray(
        name="tray_emergency",
        label="Emergency",
        category="emergency",
        center=(-0.48, 0.31, 0.455),
        half_extents=(0.13, 0.105, 0.055),
    ),
    Tray(
        name="tray_antibiotic",
        label="Antibiotic",
        category="antibiotic",
        center=(-0.16, 0.31, 0.455),
        half_extents=(0.13, 0.105, 0.055),
    ),
    Tray(
        name="tray_pain_relief",
        label="Pain Relief",
        category="pain_relief",
        center=(0.16, 0.31, 0.455),
        half_extents=(0.13, 0.105, 0.055),
    ),
    Tray(
        name="tray_inspect",
        label="Inspect",
        category="inspect",
        center=(0.48, 0.31, 0.455),
        half_extents=(0.13, 0.105, 0.055),
    ),
)

MEDICATION_ITEMS: tuple[MedicationItem, ...] = (
    MedicationItem(
        name="red_emergency_box",
        label="Emergency Box",
        shape="box",
        color="red",
        marker="cross",
        category="emergency",
        grasp="side_pinch",
        initial_pos=(-0.36, -0.14, 0.425),
        held_offset=(0.0, 0.0, -0.155),
        drop_pos=(-0.48, 0.31, 0.470),
    ),
    MedicationItem(
        name="blue_antibiotic_bottle",
        label="Antibiotic Bottle",
        shape="cylinder",
        color="blue",
        marker="cap_band",
        category="antibiotic",
        grasp="wrap",
        initial_pos=(-0.08, -0.13, 0.465),
        held_offset=(0.0, 0.0, -0.170),
        drop_pos=(-0.16, 0.31, 0.510),
    ),
    MedicationItem(
        name="yellow_pain_relief_box",
        label="Pain Relief Box",
        shape="box",
        color="yellow",
        marker="pill",
        category="pain_relief",
        grasp="side_pinch",
        initial_pos=(0.20, -0.16, 0.420),
        held_offset=(0.0, 0.0, -0.150),
        drop_pos=(0.16, 0.31, 0.465),
    ),
    MedicationItem(
        name="white_unknown_marker",
        label="Unknown Marker",
        shape="sphere",
        color="white",
        marker="inspection_dot",
        category="inspect",
        grasp="enclosing",
        initial_pos=(0.42, -0.10, 0.425),
        held_offset=(0.0, 0.0, -0.148),
        drop_pos=(0.48, 0.31, 0.475),
    ),
)


def trial_offset(item_name: str, trial_index: int, seed: int) -> np.ndarray:
    stable = sum(ord(ch) for ch in item_name) + seed * 31 + trial_index * 17
    dx = ((stable % 5) - 2) * 0.006
    dy = (((stable // 5) % 5) - 2) * 0.005
    return np.array([dx, dy, 0.0], dtype=float)


class DexTriagePlanner:
    """Deterministic task planner for the medication sorting demo."""

    def __init__(self, items: Iterable[MedicationItem] = MEDICATION_ITEMS, trays: Iterable[Tray] = TRAYS):
        self.items = tuple(items)
        self.trays = tuple(trays)
        self._trays_by_category = {tray.category: tray for tray in self.trays}

    def sorting_order(self) -> tuple[MedicationItem, ...]:
        return self.items

    def tray_for(self, item: MedicationItem) -> Tray:
        try:
            return self._trays_by_category[item.category]
        except KeyError as exc:
            raise ValueError(f"No tray configured for category {item.category!r}") from exc

    def is_inside_target_tray(self, item: MedicationItem, position: np.ndarray) -> bool:
        tray = self.tray_for(item)
        center = np.asarray(tray.center, dtype=float)
        half_extents = np.asarray(tray.half_extents, dtype=float)
        delta = np.abs(np.asarray(position, dtype=float) - center)
        return bool(np.all(delta <= half_extents + np.array([0.02, 0.02, 0.05])))

    def placement_error(self, item: MedicationItem, position: np.ndarray) -> float:
        target = np.asarray(item.drop_pos, dtype=float)
        return float(np.linalg.norm(np.asarray(position, dtype=float) - target))

    def observation_for(self, item: MedicationItem, *, damaged: bool = False, expired: bool = False):
        from perception import SimulatedObservation

        return SimulatedObservation(
            object_name=item.name,
            shape=item.shape,
            color=item.color,
            marker=item.marker,
            damaged=damaged,
            expired=expired,
        )
