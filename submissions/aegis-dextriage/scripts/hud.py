from __future__ import annotations

from typing import Any

import numpy as np

_CATEGORY_COLORS = {
    "emergency": (240, 40, 35),
    "antibiotic": (45, 115, 235),
    "pain_relief": (245, 190, 45),
    "inspect": (230, 230, 210),
}


def _draw_rect(
    frame: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    h, w = frame.shape[:2]
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    if x0 >= x1 or y0 >= y1:
        return

    overlay = np.array(color, dtype=np.float32)
    frame[y0:y1, x0:x1, :] = (
        frame[y0:y1, x0:x1, :] * (1.0 - alpha) + overlay * alpha
    ).astype(np.uint8)


def apply_hud(frame: np.ndarray, state: dict[str, Any], sorted_count: int, object_count: int) -> np.ndarray:
    output = frame.copy()
    h, w = output.shape[:2]
    category = str(state.get("perceived_category") or state.get("target_tray") or "inspect")
    color = _CATEGORY_COLORS.get(category, (80, 220, 190))
    top_h = max(44, h // 14)
    progress_w = int(w * sorted_count / max(1, object_count))

    _draw_rect(output, 0, 0, w, top_h, (8, 12, 18), 0.72)
    _draw_rect(output, 0, top_h, progress_w, max(top_h + 10, h // 12), color, 0.92)
    _draw_rect(output, 16, 14, 170, 34, color, 0.85)
    _draw_rect(output, w - 190, 14, w - 16, 34, (255, 255, 255), 0.82)
    return output
