# Aegis DexTriage

Aegis DexTriage is an autonomous MuJoCo medication triage station. A simplified multi-finger dexterous hand sorts emergency medication objects into labeled trays using deterministic task planning, shape-specific grasp primitives, touch sensors, frame sensors, and reproducible metrics.

The project is intentionally built for reliable Robothon judging: no model training, no external mesh dependency, and a one-command demo that generates trajectory and metrics artifacts.

## Version 2.0 Perception-Guided Multi-Trial Triage

This version upgrades the original runnable MVP into a perception-guided, benchmarkable Robothon submission:

- Simulated perception classifies each object from shape, color cue, marker, and deterministic condition flags.
- A triage policy assigns category, priority, grasp override, verification requirement, and recovery reason.
- Benchmark mode runs seeded multi-trial validation with optional randomized damaged/expired conditions.
- HUD presentation video overlays category color and task progress without adding OpenCV/Pillow dependencies.
- Textured tray label boards using PNG assets for `EMERGENCY`, `ANTIBIOTIC`, `PAIN RELIEF`, and `INSPECT`.
- UV-mapped label meshes on the front boards and table zones so category names remain readable from angled and top-down camera views.
- Color-coded classification zone pads under each tray so the target areas are visible from the demo camera.
- Visual medication markers built from MJCF primitive geoms.
- A wide default demo camera plus optional `topdown_camera` and `wrist_camera` renders.
- Structured console evidence with `[SCAN]`, `[GRASP]`, and `[PLACE]` trace lines.
- `outputs/metrics.json` with v2 pipeline, classification accuracy, contact confirmations, recovery count, placement error, and per-object results.
- Deterministic multi-trial validation through `--benchmark`, `--trials`, `--randomized`, and `--trial-seed`.
- Additional grasp/target sites and frame sensors for stronger MuJoCo evidence.

## Version 2.0 Pipeline

```text
MuJoCo scene state + camera view
        |
        v
Simulated perception: shape + color + marker + condition flags
        |
        v
Triage policy: category, priority, verification requirement
        |
        v
Dexterous controller: scan, approach, grasp, lift, transfer, place
        |
        v
Verification: tray containment, placement error, contact evidence
        |
        v
Benchmark metrics and HUD video evidence
```

## Task Goal

A field medical station starts with medication items scattered across a table. The robot must classify each item and place it into the correct tray:

| Object | Shape | Visual cue | Grasp primitive | Target tray |
|---|---|---|---|---|
| `red_emergency_box` | Box | Red box with white cross | Side pinch | Emergency |
| `blue_antibiotic_bottle` | Cylinder | Blue bottle with cap and band | Wrap grasp | Antibiotic |
| `yellow_pain_relief_box` | Box | Yellow box with pill marker | Side pinch | Pain Relief |
| `white_unknown_marker` | Sphere | White marker with inspection dot | Enclosing grasp | Inspect |

A successful run sorts all medication objects into their assigned trays and writes JSON evidence.

## Robot Platform

The MVP uses a custom MJCF dexterous hand instead of a large imported robot model. The hand includes:

- A scripted freejoint palm base.
- Four articulated fingers.
- Two hinge joints per finger.
- Position actuators for every finger joint.
- Fingertip touch sites and touch sensors.
- Collision geoms on the palm, phalanges, fingertips, medication objects, trays, and table.

The name Aegis refers to the emergency support system. The first version focuses on a stable triage workstation rather than locomotion.

## Technical Approach

The controller is autonomous and deterministic:

1. Load medication and tray metadata from `scripts/planner.py`.
2. Build a simulated observation for each object and classify it with `scripts/perception.py`.
3. Convert perception into a triage action with `scripts/triage_policy.py`.
4. Move the hand through scan, approach, grasp, lift, transfer, place, and retreat phases.
5. Script the held object's freejoint during transfer for reproducible contact-safe presentation.
6. Verify tray containment, contact evidence, placement error, and recovery-aware event fields.

This is a rule-based planner rather than reinforcement learning, which keeps the demo reproducible on judge machines.

## MuJoCo Features Used

| MuJoCo feature | Where used | Evidence |
|---|---|---|
| `body` / `geom` | Table, trays, label boards, classification pads, medication objects, palm, phalanges, fingertips | `config/scene.xml` worldbody |
| `freejoint` | Palm base and medication objects | `hand_base_freejoint`, object freejoints |
| `joint` | Finger knuckles and distal joints | `index_knuckle`, `thumb_distal`, etc. |
| `actuator/position` | Per-finger control targets | `index_knuckle_act`, `thumb_distal_act`, etc. |
| `sensor/touch` | Fingertip contact evidence | `index_touch`, `middle_touch`, `ring_touch`, `thumb_touch` |
| `sensor/jointpos` / `sensor/jointvel` | Hand state tracking | finger joint sensors |
| `sensor/framepos` | Objects, grasp sites, tray targets, palm | trajectory samples and `metrics.json` |
| `texture` / `material` / `mesh` | UV-mapped PNG label boards for visible tray names | `assets/labels/*.png`, `assets/meshes/*.obj`, and `scene.xml` assets |
| `camera` | Demo, top-down, and wrist views | `--camera demo_camera`, `--camera topdown_camera`, `--camera wrist_camera` |

## How To Run

From the repository root:

```powershell
python -m pip install -r requirements.txt
python submissions/aegis-dextriage/scripts/run_demo.py --headless --steps 2400 --no-video
```

If the `python` launcher is unavailable on Windows, use the Python executable from your virtual environment:

```powershell
.\.venv\Scripts\python.exe submissions/aegis-dextriage/scripts/run_demo.py --headless --steps 2400 --no-video
```

Render the main demo video:

```powershell
python submissions/aegis-dextriage/scripts/run_demo.py --steps 2400 --duration 12 --fps 24 --width 960 --height 720
```

Run deterministic multi-trial validation:

```powershell
python submissions/aegis-dextriage/scripts/run_demo.py --headless --trials 3 --no-video
```

Run a 20-trial v2 benchmark with deterministic condition perturbations:

```powershell
python submissions/aegis-dextriage/scripts/run_demo.py --headless --benchmark --trials 20 --randomized --no-video
```

Render the v2 HUD presentation video:

```powershell
python submissions/aegis-dextriage/scripts/run_demo.py --hud --steps 2400 --duration 12 --fps 24 --width 960 --height 720
```

Render alternate camera views:

```powershell
python submissions/aegis-dextriage/scripts/run_demo.py --camera topdown_camera --steps 2400 --duration 12 --fps 12 --width 640 --height 640 --output submissions/aegis-dextriage/outputs/demo_topdown.mp4
python submissions/aegis-dextriage/scripts/run_demo.py --camera wrist_camera --steps 2400 --duration 12 --fps 12 --width 640 --height 480 --output submissions/aegis-dextriage/outputs/demo_wrist.mp4
```

Visual label assets are stored with the submission:

```text
submissions/aegis-dextriage/assets/labels/emergency_label.png
submissions/aegis-dextriage/assets/labels/antibiotic_label.png
submissions/aegis-dextriage/assets/labels/pain_relief_label.png
submissions/aegis-dextriage/assets/labels/inspect_label.png
submissions/aegis-dextriage/assets/meshes/label_board_front.obj
submissions/aegis-dextriage/assets/meshes/label_board_top.obj
```

Outputs are written to:

```text
submissions/aegis-dextriage/outputs/trajectory.json
submissions/aegis-dextriage/outputs/metrics.json
submissions/aegis-dextriage/outputs/demo.mp4
```

If MP4 encoding is unavailable, the script writes a GIF fallback.

## Expected Console Summary

A successful run prints judge-friendly trace lines and final JSON:

```text
[SCAN] red_emergency_box -> Emergency
[GRASP] red_emergency_box primitive=side_pinch contact=true
[PLACE] red_emergency_box target=tray_emergency success=true
```

```json
{
  "project": "Aegis DexTriage",
  "success": true,
  "sorted_count": 4,
  "object_count": 4
}
```

## Metrics Evidence

`outputs/metrics.json` records:

- `version`: current submission schema, `2.0`.
- `pipeline`: perception, triage policy, grasp, placement, and verification stages.
- `success`: full-task result.
- `sorted_count` and `object_count`.
- `classification_accuracy`: deterministic category recognition score for known medication classes.
- `recovery_count`: number of recovery-aware grasp checks triggered during the run.
- `contact_confirmations`: grasp/contact evidence.
- `per_object`: target tray, selected grasp, placement error, and sorted status.
- `trial_seed` and `trial_index` for reproducibility.

For benchmark mode, the metrics file stores a `trials` array plus aggregate success rate, mean placement error, sorted object count, and total recoveries.

## Rubric Mapping

| Rubric area | Evidence in this submission |
|---|---|
| Runnability | Deterministic one-command demo plus 20-trial benchmark mode with seeded randomized conditions. |
| Depth of MuJoCo Use | MJCF joints, actuators, touch sensors, frame sensors, cameras, collision geoms, target sites, and grasp sites. |
| Task Design | Emergency medication triage is clear, meaningful, and visually easy to inspect. |
| Control | Perception-guided triage policy, phase-based hand motion, contact evidence, placement verification, and recovery-aware events. |
| Dexterous Manipulation | Multi-finger hand coordinates different grasps for box, cylinder, and sphere objects. |
| Engineering Quality | Focused modules for perception, triage policy, benchmark aggregation, HUD rendering, control, planning, and reproducible artifacts. |
| Presentation | HUD video, textured labels, zone pads, trace logs, and metrics artifacts show the full perception-to-action loop. |
| Innovation | Applies dexterous manipulation to field medication triage instead of generic pick-and-place. |

## Current Limitations

- The MVP uses scripted waypoints rather than learned policies.
- Object recognition uses simulated observations from task metadata and MuJoCo state instead of raw pixel classification.
- The hand is a simplified dexterous model designed for reproducibility, not a hardware-accurate Shadow Hand.

## Future Improvements

- Add stronger seeded object randomization once the deterministic baseline remains stable.
- Add a split-screen video showing demo and wrist camera views.
- Add an imported LEAP Hand or Shadow Hand variant after the custom-hand submission is finalized.
- Add split-screen overlays if the contest environment allows edited presentation videos beyond raw MuJoCo renders.
