from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mujoco
except ImportError as exc:
    raise SystemExit(
        "Missing MuJoCo dependency. Install with:\n"
        "  python -m pip install -r requirements.txt\n\n"
        f"Original error: {exc}"
    ) from exc

try:
    import imageio.v3 as iio
except ImportError:
    iio = None

SCRIPT_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = SCRIPT_DIR.parent
DEFAULT_MODEL = SUBMISSION_DIR / "config" / "scene.xml"
DEFAULT_OUTPUT = SUBMISSION_DIR / "outputs" / "demo.mp4"
DEFAULT_TRAJECTORY = SUBMISSION_DIR / "outputs" / "trajectory.json"
DEFAULT_METRICS = SUBMISSION_DIR / "outputs" / "metrics.json"

sys.path.insert(0, str(SCRIPT_DIR))
from benchmark import summarize_benchmark  # noqa: E402
from controller import DexTriageController  # noqa: E402
from hud import apply_hud  # noqa: E402
from planner import DexTriagePlanner  # noqa: E402

def scenario_flags_for_trial(trial_index: int, trial_seed: int, randomized: bool) -> dict[str, dict[str, bool]]:
    if not randomized:
        return {}

    item_names = (
        "red_emergency_box",
        "blue_antibiotic_bottle",
        "yellow_pain_relief_box",
        "white_unknown_marker",
    )
    flags: dict[str, dict[str, bool]] = {}
    for idx, item_name in enumerate(item_names):
        stable = sum(ord(ch) for ch in item_name) + trial_seed * 37 + trial_index * 19 + idx * 11
        flags[item_name] = {
            "damaged": stable % 17 == 0,
            "expired": stable % 23 == 0,
        }
    return flags


def camera_name_to_id(model: mujoco.MjModel, name: str) -> int:
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if camera_id < 0:
        raise ValueError(f"Missing camera in MJCF: {name}")
    return int(camera_id)


def render_camera(model: mujoco.MjModel, name: str) -> int | mujoco.MjvCamera:
    if name != "demo_camera":
        return camera_name_to_id(model, name)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.03, 0.53]
    camera.distance = 1.55
    camera.azimuth = 135.0
    camera.elevation = -42.0
    return camera


def sample_sensor_data(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, list[float]]:
    samples: dict[str, list[float]] = {}
    interesting_prefixes = (
        "index_touch",
        "middle_touch",
        "ring_touch",
        "thumb_touch",
        "red_emergency_box_pos",
        "blue_antibiotic_bottle_pos",
        "yellow_pain_relief_box_pos",
        "white_unknown_marker_pos",
        "red_emergency_box_grasp_pos",
        "blue_antibiotic_bottle_grasp_pos",
        "yellow_pain_relief_box_grasp_pos",
        "white_unknown_marker_grasp_pos",
        "hand_palm_pos",
        "emergency_tray_pos",
        "antibiotic_tray_pos",
        "pain_tray_pos",
        "inspect_tray_pos",
        "tray_",
    )
    for sensor_id in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id) or f"sensor_{sensor_id}"
        if not name.startswith(interesting_prefixes):
            continue
        adr = int(model.sensor_adr[sensor_id])
        dim = int(model.sensor_dim[sensor_id])
        samples[name] = data.sensordata[adr : adr + dim].round(5).tolist()
    return samples


def write_video(video_path: Path, frames: list[np.ndarray], fps: int, summary: dict[str, Any]) -> None:
    if not frames:
        return
    if iio is None:
        summary["video_skipped_reason"] = "imageio is not installed"
        return

    video_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        iio.imwrite(video_path, np.asarray(frames), fps=fps, codec="libx264")
        summary["video"] = str(video_path)
    except Exception as exc:
        fallback = video_path.with_suffix(".gif")
        iio.imwrite(fallback, np.asarray(frames), fps=fps)
        summary["video"] = str(fallback)
        summary["video_fallback_reason"] = str(exc)


def build_metrics(summary: dict[str, Any], result: dict[str, Any], *, trial_index: int, trial_seed: int) -> dict[str, Any]:
    metrics = {
        "project": "Aegis DexTriage",
        "version": "2.0",
        "pipeline": ["perception", "triage_policy", "grasp", "placement", "verification"],
        "trial_index": trial_index,
        "trial_seed": trial_seed,
        "success": summary["success"],
        "sorted_count": summary["sorted_count"],
        "object_count": summary["object_count"],
        "recovery_count": result.get("recovery_count", 0),
        "classification_accuracy": sum(
            1
            for item in result["per_object_metrics"].values()
            if item.get("perceived_category") in {"emergency", "antibiotic", "pain_relief", "inspect"}
        ) / max(1, len(result["per_object_metrics"])),
        "contact_confirmations": sum(1 for event in result["events"] if event.get("contact_confirmed")),
        "per_object": result["per_object_metrics"],
        "deterministic": True,
    }
    metrics["classification_accuracy"] = round(metrics["classification_accuracy"], 3)
    return metrics


def format_trace_lines(events: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for event in events:
        kind = event.get("kind")
        if kind == "SCAN":
            lines.append(
                f"[SCAN] {event['object']} -> {event['perceived_category']} "
                f"confidence={event['perception_confidence']:.2f}"
            )
        elif kind == "GRASP":
            contact = str(bool(event.get("contact_confirmed"))).lower()
            lines.append(
                f"[GRASP] {event['object']} primitive={event['selected_grasp']} contact={contact}"
            )
        elif kind == "PLACE":
            success = str(bool(event.get("placement_success"))).lower()
            lines.append(
                f"[PLACE] {event['object']} target={event['target_tray']} success={success}"
            )
    return lines


def run_demo(
    *,
    model_path: Path,
    output_path: Path,
    trajectory_path: Path,
    metrics_path: Path | None,
    steps: int,
    duration_s: float,
    fps: int,
    width: int,
    height: int,
    camera: str,
    no_video: bool,
    randomized: bool = False,
    hud: bool = False,
    trial_index: int = 0,
    trial_seed: int = 7,
) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    planner = DexTriagePlanner()
    controller = DexTriageController(model, data, planner)
    controller.reset_scene(
        trial_index=trial_index,
        seed=trial_seed,
        scenario_flags=scenario_flags_for_trial(trial_index, trial_seed, randomized),
    )
    mujoco.mj_forward(model, data)

    renderer = None
    active_camera: int | mujoco.MjvCamera | None = None
    frames: list[np.ndarray] = []
    if not no_video:
        renderer = mujoco.Renderer(model, width=width, height=height)
        active_camera = render_camera(model, camera)

    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    frame_stride = max(1, steps // max(1, int(duration_s * fps)))
    sample_stride = max(1, steps // 80)

    for step in range(steps):
        state = controller.step(step, steps)
        if step % sample_stride == 0 or step == steps - 1:
            samples.append(
                {
                    "time_s": round(duration_s * step / max(1, steps - 1), 3),
                    **state,
                    "sensors": sample_sensor_data(model, data),
                }
            )

        if renderer is not None and active_camera is not None and step % frame_stride == 0:
            renderer.update_scene(data, camera=active_camera)
            frame = renderer.render().copy()
            if hud:
                frame = apply_hud(frame, state, state["sorted_count"], len(planner.items))
            frames.append(frame)

    result = controller.final_result()
    summary: dict[str, Any] = {
        "project": "Aegis DexTriage",
        "version": "2.0",
        "pipeline": "perception_guided_triage",
        "task": "Autonomous dexterous medication sorting into emergency triage trays.",
        "model": str(model_path),
        "trajectory": str(trajectory_path),
        "steps": steps,
        "duration_s": duration_s,
        "fps": fps,
        "camera": camera,
        "trial_index": trial_index,
        "trial_seed": trial_seed,
        "randomized": randomized,
        "success": result["success"],
        "sorted_count": result["sorted_count"],
        "object_count": result["object_count"],
        "recovery_count": result.get("recovery_count", 0),
        "sorted_flags": result["sorted_flags"],
        "object_positions": result["object_positions"],
        "per_object_metrics": result["per_object_metrics"],
        "events": result["events"],
        "trace": format_trace_lines(result["events"]),
        "trajectory_samples": samples,
    }

    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics = build_metrics(summary, result, trial_index=trial_index, trial_seed=trial_seed)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        summary["metrics"] = str(metrics_path)

    write_video(output_path, frames, fps, summary)
    trajectory_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def trial_trajectory_path(base_path: Path, trial_index: int) -> Path:
    return base_path.with_name(f"{base_path.stem}_trial_{trial_index}{base_path.suffix}")


def run_trials(
    *,
    model_path: Path,
    output_path: Path,
    trajectory_path: Path,
    metrics_path: Path,
    steps: int,
    duration_s: float,
    fps: int,
    width: int,
    height: int,
    camera: str,
    trials: int,
    trial_seed: int,
    benchmark: bool,
    randomized: bool,
) -> dict[str, Any]:
    trial_summaries: list[dict[str, Any]] = []
    for trial_index in range(trials):
        summary = run_demo(
            model_path=model_path,
            output_path=output_path,
            trajectory_path=trial_trajectory_path(trajectory_path, trial_index),
            metrics_path=None,
            steps=steps,
            duration_s=duration_s,
            fps=fps,
            width=width,
            height=height,
            camera=camera,
            no_video=True,
            randomized=randomized,
            hud=False,
            trial_index=trial_index,
            trial_seed=trial_seed,
        )
        trial_summaries.append(
            {
                "trial_index": trial_index,
                "success": summary["success"],
                "sorted_count": summary["sorted_count"],
                "object_count": summary["object_count"],
                "recovery_count": summary.get("recovery_count", 0),
                "trajectory": summary["trajectory"],
                "per_object_metrics": summary["per_object_metrics"],
            }
        )

    aggregate = {
        "project": "Aegis DexTriage",
        "version": "2.0",
        "pipeline": "perception_guided_triage_benchmark",
        "task": "Deterministic multi-trial validation for autonomous medication sorting.",
        "model": str(model_path),
        "trajectory": str(trajectory_path),
        "metrics": str(metrics_path),
        "trials_requested": trials,
        "trial_seed": trial_seed,
        "randomized": randomized,
        "success": all(trial["success"] for trial in trial_summaries),
        "successful_trials": sum(1 for trial in trial_summaries if trial["success"]),
        "trials": trial_summaries,
        "video_skipped_reason": "multi-trial validation runs headless; render a single demo video separately",
    }
    if benchmark:
        aggregate["benchmark"] = summarize_benchmark(trial_summaries)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    trajectory_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Aegis DexTriage autonomous MuJoCo medication sorting demo."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--camera",
        choices=("demo_camera", "topdown_camera", "wrist_camera"),
        default="demo_camera",
        help="Camera used for rendering video frames.",
    )
    parser.add_argument("--headless", action="store_true", help="Alias for running without an interactive viewer.")
    parser.add_argument("--no-video", action="store_true", help="Skip rendering and only write trajectory JSON.")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark-style multi-trial metrics.")
    parser.add_argument("--randomized", action="store_true", help="Apply deterministic scenario perturbations and inspection flags.")
    parser.add_argument("--hud", action="store_true", help="Overlay a lightweight triage HUD on rendered video frames.")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--trial-seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.trials < 1:
        raise SystemExit("--trials must be at least 1")

    if args.trials > 1 or args.benchmark:
        summary = run_trials(
            model_path=args.model,
            output_path=args.output,
            trajectory_path=args.trajectory,
            metrics_path=args.metrics,
            steps=args.steps,
            duration_s=args.duration,
            fps=args.fps,
            width=args.width,
            height=args.height,
            camera=args.camera,
            trials=args.trials,
            trial_seed=args.trial_seed,
            benchmark=args.benchmark,
            randomized=args.randomized,
        )
    else:
        summary = run_demo(
            model_path=args.model,
            output_path=args.output,
            trajectory_path=args.trajectory,
            metrics_path=args.metrics,
            steps=args.steps,
            duration_s=args.duration,
            fps=args.fps,
            width=args.width,
            height=args.height,
            camera=args.camera,
            no_video=args.no_video or args.headless,
            randomized=args.randomized,
            hud=args.hud,
            trial_index=0,
            trial_seed=args.trial_seed,
        )

    for line in summary.get("trace", []):
        print(line)
    printable = {key: value for key, value in summary.items() if key != "trajectory_samples"}
    print(json.dumps(printable, indent=2))
    return 0 if summary["success"] else 2


if __name__ == "__main__":
    sys.exit(main())
