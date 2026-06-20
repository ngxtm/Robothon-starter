from __future__ import annotations

from typing import Any

_SEGMENTS = (
    (0.00, 0.25, "demo_camera", "establishing_view"),
    (0.25, 0.50, "topdown_camera", "classification_layout"),
    (0.50, 0.75, "wrist_camera", "dexterous_contact_view"),
    (0.75, 1.01, "demo_camera", "verification_summary"),
)


def camera_for_step(step: int, total_steps: int, default_camera: str, cinematic: bool) -> str:
    if not cinematic:
        return default_camera
    progress = step / max(1, total_steps - 1)
    for start, end, camera, _label in _SEGMENTS:
        if start <= progress < end:
            return camera if camera != "demo_camera" else default_camera
    return default_camera


def camera_sequence_metadata(total_steps: int, default_camera: str, cinematic: bool) -> dict[str, Any]:
    if not cinematic:
        return {
            "enabled": False,
            "segments": [
                {
                    "start_step": 0,
                    "end_step": max(0, total_steps - 1),
                    "camera": default_camera,
                    "purpose": "static_render",
                }
            ],
        }

    segments: list[dict[str, Any]] = []
    for start, end, camera, label in _SEGMENTS:
        start_step = int(round(start * max(1, total_steps - 1)))
        end_step = int(round(min(end, 1.0) * max(1, total_steps - 1)))
        segments.append(
            {
                "start_step": start_step,
                "end_step": end_step,
                "camera": camera if camera != "demo_camera" else default_camera,
                "purpose": label,
            }
        )
    return {"enabled": True, "segments": segments}
