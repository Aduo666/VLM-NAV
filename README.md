<div align="center">
  <img src="docs/assets/hero.png" alt="VLM-NAV hero" width="100%">

  <h1>VLM-NAV</h1>

  <p>
    <b>An Isaac Sim testbed for vision-language navigation with a custom Ackermann-style mobile robot.</b>
  </p>

  <p>
    VLM-NAV connects language instructions, RGB-D perception, LiDAR mapping, semantic grounding,
    VLM reasoning, and low-level wheel/steering control into a runnable closed-loop navigation stack.
  </p>

  <p>
    <img alt="Isaac Sim" src="https://img.shields.io/badge/Isaac%20Sim-5.1-76B900?style=for-the-badge">
    <img alt="Python" src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white">
    <img alt="Robot" src="https://img.shields.io/badge/Robot-Custom%20USD-0F172A?style=for-the-badge">
    <img alt="Navigation" src="https://img.shields.io/badge/VLN%20%2F%20ObjectNav-Zero--shot-FFB000?style=for-the-badge">
  </p>
</div>

## Why This Project Exists

Vision-and-Language Navigation (VLN) asks an embodied agent to follow natural-language instructions through previously unseen 3D environments. Classic VLN benchmarks such as R2R formalized the problem around language-conditioned action sequences in real building-scale scenes, while VLN-CE pushed the setting toward continuous navigation with realistic motion and collision constraints.

This repository focuses on the engineering gap between those benchmark formulations and an Isaac Sim robot stack:

- running a custom robot USD instead of a benchmark agent;
- using Isaac Sim RGB-D, LiDAR, odometry, and articulation control;
- keeping the high-level policy replaceable through VLM/LLM APIs;
- exposing maps, semantic anchors, candidates, and robot actions for debugging.

VLM-NAV is not a claim of state-of-the-art benchmark performance. It is a practical platform for testing VLN-style ideas on a wheel-driven robot in Isaac Sim.

## Core Idea

<p align="center">
  <img src="docs/assets/overview.png" alt="VLM-NAV method overview" width="100%">
</p>

The stack follows a simple separation of responsibilities:

```text
language instruction
    -> visual / metric observation
    -> semantic grounding + landmark memory
    -> occupancy-aware candidate generation
    -> VLM local decision
    -> discrete Ackermann action primitive
```

The design borrows several principles from recent VLN work:

- **Think global, act local**: maintain a global map and landmark memory, but execute short local decisions.
- **Ground before acting**: visual detections are checked with depth and VLM semantic verification before becoming stable landmarks.
- **Structure the execution loop**: instruction parsing, progress tracking, candidate selection, safety checks, and low-level control are separate modules.
- **Keep robot control deterministic**: the VLM selects from constrained candidates; the controller executes fixed primitives rather than continuous free-form commands.

## Demo

The video below shows an early successful closed-loop run in Isaac Sim. The robot follows a language-conditioned route using the current RGB-D, LiDAR map, semantic landmark memory, VLM candidate selection, and Ackermann primitive controller.

<video src="docs/assets/firstsuccess.mp4" controls width="100%"></video>

[Open the demo video](docs/assets/firstsuccess.mp4) if your Markdown viewer does not render embedded local MP4 files.

This demo is meant to show the system running end-to-end rather than final navigation robustness. The current policy can still oscillate when adjacent left/right candidates receive similar scores, which may cause local spinning or slow progress in narrow areas.

## Features

- **Isaac Sim 5.1 standalone runtime** with scene USD loading, custom robot USD loading, cameras, LiDAR, and viewport setup.
- **Custom Ackermann-style robot control** through wheel velocity targets and steering joint position targets.
- **Task JSON workflow** for scene path, start pose, and natural-language instruction selection.
- **Global occupancy map** from LiDAR, rendered as a north-up map for both debugging and VLM context.
- **Semantic landmark tracking** with GroundingDINO, optional SAM hooks, VLM verification, bbox/depth projection, and confirmed landmark anchors.
- **VLM candidate policy** using a first-person candidate view plus map context.
- **Discrete action primitives** by default: `FORWARD`, `BACKWARD`, `TURN_LEFT`, `TURN_RIGHT`, `STOP`.
- **Debug artifacts** including `vln_debug.png`, VLM request images, structured logs, and timestamped MP4 debug videos.

## System Architecture

```text
src/vln_nav/
├── app/
│   └── nav.py                    # Isaac Sim main loop
├── control/
│   └── controller.py             # wheel / steering articulation control
├── navigation/
│   ├── instruction_manager.py    # route and landmark parsing
│   ├── observation_adapter.py    # RGB-D, pose, feedback packaging
│   ├── lidar_sensor.py           # Isaac / PhysX LiDAR access
│   ├── occupancy_grid.py         # global occupancy memory
│   ├── map_candidate_generator.py
│   ├── candidate_waypoint_predictor.py
│   ├── target_grounding.py       # detector/VLM grounding interface
│   ├── target_tracker.py         # bbox tracking between grounding calls
│   ├── landmark_tracker.py       # stable landmark state and arrival checks
│   ├── landmark_map.py           # confirmed landmark memory
│   ├── progress_manager.py       # progress/stall signals
│   ├── depth_safety.py           # near-field safety override
│   ├── vln_reasoner.py           # VLM prompt and response parsing
│   └── vln_policy.py             # high-level navigation policy
└── runtime/
    ├── config.py
    ├── debug_video.py
    └── isaac_runtime_utils.py
```

## Repository Layout

```text
.
├── configs/
│   ├── default.yaml              # runtime, perception, control, map, and debug config
│   └── instructions.json         # navigation task list
├── docs/assets/
│   ├── hero.png                  # README cover image
│   └── overview.png              # method overview image
├── exts/vln.nav.task_editor/     # Isaac Sim task authoring panel
├── scripts/
│   ├── run_nav.sh                # main navigation launcher
│   ├── run_task_editor.sh        # task authoring launcher
│   ├── setup_grounded_sam_local.sh
│   └── test_llm_call.py          # offline VLM request test
├── src/
│   ├── ros2_car_model_ws/        # robot model workspace / USD assets
│   └── vln_nav/                  # navigation package
├── third_party/GroundingDINO/    # optional detector checkout
├── usd/                          # scene USD files
└── weights/grounded_sam/         # detector / segmenter weights
```

## Installation

The project assumes an existing conda environment:

```bash
conda activate env_isaaclab
```

Runtime assumptions:

- Ubuntu/Linux workstation with NVIDIA GPU.
- Python 3.10.
- Isaac Sim 5.1 installed in `env_isaaclab`.
- A SiliconFlow/OpenAI-compatible VLM API key.
- Optional local GroundingDINO weights for faster grounding.

Set the API key through an environment variable:

```bash
export SILICONFLOW_API_KEY="your_api_key"
```

If you use non-headless mode, start from a terminal with a valid desktop display. For multi-GPU machines, `sim.active_gpu: auto` tries to choose the display GPU; override it when necessary:

```bash
ACTIVE_GPU=1 bash scripts/run_nav.sh
```

## Running Navigation

Run the default enabled task:

```bash
bash scripts/run_nav.sh
```

List available tasks:

```bash
bash scripts/run_nav.sh --list-tasks
```

Run a specific task:

```bash
bash scripts/run_nav.sh --task hospital_002
```

Run headless:

```bash
bash scripts/run_nav.sh --headless
```

## Task Format

Navigation episodes are defined in `configs/instructions.json`:

```json
{
  "id": "hospital_001",
  "scene_usd_path": "usd/environment/hospital_flat.usd",
  "robot_start_position": [1.5, 4.0, 0.0],
  "robot_start_orientation_deg": [0.0, 0.0, 0.0],
  "instruction": "向前走找到轮椅，然后左转再直走找到楼梯"
}
```

Notes:

- `scene_usd_path` is resolved relative to the repository root unless absolute.
- `robot_start_position` is applied to `/World/Robot`.
- `robot_start_position[2]` should normally remain `0.0` in the current setup.
- The instruction can be Chinese or English; landmarks are normalized internally for grounding and tracking.

## Creating New Tasks

Launch the lightweight Isaac Sim task editor:

```bash
bash scripts/run_task_editor.sh
```

The editor loads the configured scene and robot, then opens a `VLM-NAV Task Editor` panel. Move `/World/Robot` in the viewport, enter a task id and instruction, and save. The panel writes to `configs/instructions.json`; it does not modify the original scene USD unless you explicitly save the USD from Isaac Sim.

The task editor scene and robot paths are configured at the top of `scripts/run_task_editor.sh`.

## GroundingDINO / SAM

The default grounding backend is:

```yaml
target_grounding:
  backend: "auto"
```

In `auto` mode, the system tries local GroundingDINO first and falls back to VLM bbox grounding if the detector is unavailable.

Expected local paths:

```text
third_party/GroundingDINO/
weights/grounded_sam/GroundingDINO_SwinT_OGC.py
weights/grounded_sam/groundingdino_swint_ogc.pth
```

Set up local files:

```bash
bash scripts/setup_grounded_sam_local.sh
```

If you already have compatible weights, symlink them:

```bash
ln -sf /path/to/GroundingDINO_SwinT_OGC.py weights/grounded_sam/GroundingDINO_SwinT_OGC.py
ln -sf /path/to/groundingdino_swint_ogc.pth weights/grounded_sam/groundingdino_swint_ogc.pth
```

SAM is optional and disabled by default:

```yaml
target_grounding:
  sam_enabled: false
```

## Key Configuration

Most runtime knobs live in `configs/default.yaml`.

```yaml
llm:
  api_base_url: "https://api.siliconflow.cn/v1"
  model_name: "Qwen/Qwen3-VL-30B-A3B-Instruct"
  timeout_sec: 30

control:
  use_discrete_action_primitives: true
  wheel_drive_velocity: 14.0
  turn_drive_velocity: 6.0
  action_forward_distance_m: 0.55
  action_turn_angle_deg: 15.0

occupancy_grid:
  enabled: true
  resolution_m: 0.08
  size_m: 80.0
  semantic_planner_enabled: true

lidar:
  enabled: true
  mode: "single"
  horizontal_fov_deg: 180.0
  max_range_m: 10.0

target_grounding:
  enabled: true
  backend: "auto"
  call_period_sec: 1.5
  semantic_verification_enabled: true

opennav:
  arrival_distance_m: 1.4
  success_distance_m: 2.0
  debug_video_enabled: true
```

To try continuous subgoal tracking instead of primitive execution:

```yaml
control:
  use_discrete_action_primitives: false
```

## Debug Outputs

Runtime outputs are intentionally concentrated:

```text
logs/nav_latest.log
debug/vln_debug.png
debug/videos/vln_debug_YYYYMMDD_HHMMSS.mp4
debug/llm_request.png
logs/last_candidate_choice_request.json
```

- `debug/vln_debug.png`: human-facing combined view. It shows the north-up map, robot pose, landmarks, first-person target overlays, and current subgoal/action.
- `debug/videos/vln_debug_YYYYMMDD_HHMMSS.mp4`: timestamped debug video for post-run review.
- `debug/llm_request.png`: the exact image used for VLM candidate selection.
- `logs/nav_latest.log`: policy, grounding, landmark, safety, control, and simulator logs.
- `logs/last_candidate_choice_request.json`: latest candidate-choice request payload for prompt debugging.

Test the latest VLM prompt without launching Isaac Sim:

```bash
python scripts/test_llm_call.py
```

Manual image test:

```bash
python scripts/test_llm_call.py --mode manual --image debug/llm_request.png
```

## Current Scope

VLM-NAV is a development platform, not a polished benchmark submission. The current implementation prioritizes:

- interpretable closed-loop behavior in Isaac Sim;
- custom robot control rather than benchmark-only action spaces;
- explicit semantic and metric memory;
- debuggable candidate selection and action execution;
- fast iteration on prompts, detectors, maps, and controller settings.

Known limitations:

- semantic grounding may produce false positives in visually ambiguous scenes;
- landmark world anchors depend on camera/depth calibration and valid depth pixels;
- API-based VLM calls introduce latency;
- primitive execution is inspectable, but may require tuning distance and turning thresholds for each robot model;
- candidate selection can flicker between neighboring left/right options, causing oscillation or in-place turning in cluttered layouts;
- this repository does not yet provide standardized R2R/RxR/VLN-CE benchmark evaluation.

## Research Context

This project is inspired by the following lines of work:

- [R2R / Vision-and-Language Navigation](https://openaccess.thecvf.com/content_cvpr_2018/papers/Anderson_Vision-and-Language_Navigation_Interpreting_CVPR_2018_paper.pdf): language-conditioned navigation in real building-scale environments.
- [VLN-CE](https://github.com/jacobkrantz/VLN-CE): continuous-environment VLN with realistic observations and unconstrained navigation.
- [DUET](https://github.com/cshizhe/VLN-DUET): global map reasoning plus fine-grained local cross-modal grounding.
- [NavGPT](https://github.com/GengzeZhou/NavGPT): explicit LLM reasoning for VLN.
- [Open-Nav](https://github.com/YanyuanQiao/Open-Nav): zero-shot continuous VLN with open-source LLMs and structured reasoning.
- [Safe-VLN](https://arxiv.org/abs/2311.02817): collision-aware VLN-CE with occupancy-informed waypoint selection.
- [CA-Nav](https://arxiv.org/abs/2412.10137): constraint-aware zero-shot VLN-CE with sub-instruction progress tracking.

## Acknowledgement

VLM-NAV adapts ideas from VLN, ObjectNav, open-vocabulary grounding, and robotics navigation into an Isaac Sim workflow for a custom wheel-driven robot. The repository is intended as an experimental base for building, testing, and debugging embodied navigation systems rather than as a finished product.
