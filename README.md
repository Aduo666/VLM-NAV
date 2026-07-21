VLM-NAV
=======

**An Isaac Sim 5.1 vision-language navigation testbed for a custom 4-wheel-steering robot.**

VLM-NAV connects natural-language instructions, RGB-D perception, LiDAR mapping, open-vocabulary grounding, VLM reasoning, and low-level wheel/steering control into a runnable closed-loop navigation stack.

| | |
|---|---|
| Simulator | Isaac Sim 5.1 from pip |
| Python | Conda environment `vlmnav` |
| Robot | Custom 4WS USD |
| Policy | VLM-guided ObjectNav / VLN prototype |
| Actions | Discrete atomic primitives |

![VLM-NAV overview](docs/assets/hero.png)

## Features

- **Isaac Sim 5.1 pip runtime**: scene USD loading, custom robot USD, cameras, LiDAR, viewport setup.
- **Custom 4-wheel-steering control**: FORWARD / TURN_LEFT / TURN_RIGHT atomic primitives with odometry-based completion.
- **Semantic grounding + memory**: mandatory local GroundingDINO proposals plus VLM verification maintain confirmed landmark anchors; reached anchors remain persistent.
- **Typed instruction graph**: one startup parse separates trusted scene context from executable `GO_TO`, `REGION_TRANSITION`, and `TURN` steps; validated plans are cached by instruction and schema version.
- **Aligned action representation**: at stopped decision keyframes, the VLM chooses exactly one safe candidate outcome that maps one-to-one to TURN_LEFT / FORWARD / TURN_RIGHT.
- **Region-aware execution**: room transitions require consistent visual region evidence, odometry displacement, and a topology-node change rather than keyword matching alone.
- **Search / approach separation**: VLM exploration is used while a landmark is unknown; fresh verified image evidence directly centers/approaches the target, while a confirmed off-screen anchor is followed through occupancy-grid A*.
- **Fast primitive execution**: during an atomic action, expensive decision-time perception is skipped; odometry, depth safety, drive control, and LiDAR/map updates remain active.
- **Local and evaluation flows**: normal debugging uses `debug/local` and `logs/nav_latest.log`; dataset evaluation uses `debug/eval` plus eval outputs.

![Core method](docs/assets/overview.png)

## Environment Setup

The project now uses only the conda / pip Isaac Sim workflow. The project environment name is `vlmnav`.

### 1. Create The Conda Environment

Isaac Sim 5.1 uses Python 3.11.

```bash
conda create -n vlmnav python=3.11 -y
conda activate vlmnav
python -m pip install --upgrade pip setuptools wheel
```

### 2. Install Isaac Sim And Core Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` contains:

- `isaacsim[all,extscache]==5.1.0`
- `numpy==1.26.0`
- `PyYAML==6.0.2`
- `Pillow==11.3.0`
- `opencv-contrib-python-headless==4.11.0.86`

The NVIDIA PyPI index is already included in `requirements.txt`.

### 3. Configure SiliconFlow / VLM API

```bash
export SILICONFLOW_API_KEY="your_api_key"
```

The default model and API endpoint are configured in `configs/default.yaml`.

### 4. Install Local GroundingDINO

GroundingDINO is required by the current main navigation method. It proposes open-vocabulary boxes; the VLM verifies each proposal before it may create a confirmed landmark anchor. The navigation path no longer falls back to VLM-generated boxes.

```bash
conda activate vlmnav

# If torch / torchvision are missing in your conda environment, install a CUDA build first.
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Clone / patch GroundingDINO and install its Python dependencies.
bash scripts/setup_grounded_sam_local.sh
```

Required model files:

- `third_party/GroundingDINO/`
- `weights/grounded_sam/GroundingDINO_SwinT_OGC.py`
- `weights/grounded_sam/groundingdino_swint_ogc.pth`
- `weights/grounded_sam/bert-base-uncased/config.json` and the accompanying local BERT snapshot

`scripts/setup_grounded_sam_local.sh` prints symlink commands if you already have the CA-Nav weights locally.

## Run

### Local Navigation

```bash
conda activate vlmnav
export SILICONFLOW_API_KEY="your_api_key"

# Run the default task from configs/default.yaml / configs/instructions.json.
bash scripts/run_nav.sh

# Run a named task.
bash scripts/run_nav.sh --task hospital_001

# List available tasks.
bash scripts/run_nav.sh --list-tasks

# Headless mode.
bash scripts/run_nav.sh --headless
```

The launch scripts use the currently active conda environment. If you need to launch from another shell without activating first, override the environment explicitly:

```bash
CONDA_ENV_NAME=vlmnav bash scripts/run_nav.sh
```

### Task Authoring Panel

```bash
conda activate vlmnav
bash scripts/run_task_editor.sh
```

This loads the configured scene and robot, then opens the in-simulator task editor panel for adding instructions to `configs/instructions.json`.

## Benchmark Evaluation

### InteriorNav

```bash
bash eval/interiornav/run.sh
SPLIT=val_unseen EPISODE_INDEX=0 bash eval/interiornav/run.sh
```

Data root: `/Data/qiang/interiornav_data`

### VLN-PE / R2R

```bash
bash eval/vlnpe/run.sh
SPLIT=val_unseen EPISODE_INDEX=5 bash eval/vlnpe/run.sh
```

Data root: `/Data/qiang/InternData-N1/vln_pe`

Scene root: `/Data/qiang/mp3d_pe`

Each dataset has an adapter under `eval/<dataset>/adapter.py` that normalizes dataset episodes into the task format consumed by `nav.py`.

## Current Pipeline

```text
English language instruction
  -> one validated typed semantic graph (cached)
       scene context: localization evidence only
       route steps: GO_TO / REGION_TRANSITION / TURN
  -> RGB-D + LiDAR + odometry observation
  -> GO_TO: GroundingDINO proposal + VLM semantic verification
       visible target -> deterministic centering/approach
       confirmed off-screen anchor -> occupancy A*
       unknown target -> one VLM candidate choice at a stopped keyframe
  -> REGION_TRANSITION: candidate choice + region evidence
       complete only after semantic + odometry + topology agreement
  -> TURN: bounded repeated 15-degree primitives until heading alignment
  -> low-level 4WS controller
```

During primitive execution, the loop does not request VLM decisions or run heavy grounding. It continues odometry completion, LiDAR hit/miss mapping, hard safety checks, and low-level control. A stopped decision keyframe permits at most one VLM request: semantic verification or candidate/region reasoning, never both.

The VLM input is intentionally split by role:

- A first-person candidate image for visual choice.
- Structured text for semantic map, landmark memory, obstacle distances, route context, and candidate action metadata.
- No raw occupancy image is sent to the VLM by default; occupancy visualization is mainly for human debugging.

Instruction parsing is fail-closed and runs before Isaac Sim starts. It uses a dedicated 90-second timeout, performs one parse request, deterministic schema validation, and at most one repair request. If neither produces a valid graph and no matching cache exists, navigation does not silently substitute a heuristic plan. Normal navigation requests continue using `llm.timeout_sec`.

## Key Configuration

All main configuration lives in `configs/default.yaml`.

```yaml
llm:
  model_name: "Qwen/Qwen3-VL-30B-A3B-Instruct"
  api_base_url: "https://api.siliconflow.cn/v1"

control:
  use_discrete_action_primitives: true
  action_forward_distance_m: 0.55
  action_turn_angle_deg: 15.0
  action_odom_completion_enabled: true

target_grounding:
  enabled: true
  backend: "grounding_dino"
  grounding_dino_text_encoder_path: "weights/grounded_sam/bert-base-uncased"
  semantic_verification_enabled: true
  semantic_verification_min_confidence: 0.50

landmark_tracker:
  confirm_min_observations: 2
  confirm_max_anchor_jump_m: 0.75

opennav:
  instruction_parse_max_tokens: 768
  instruction_parse_timeout_sec: 90
  instruction_cache_enabled: true
  instruction_cache_path: ".cache/instruction_plans.json"
  region_transition_min_displacement_m: 1.0
  region_transition_confirmations: 2
  region_transition_min_confidence: 0.55
  semantic_turn_min_yaw_deg: 45.0
  semantic_turn_max_primitives: 6
  visual_target_center_deadband_deg: 15.0
  arrival_distance_m: 2.0
  success_distance_m: 3.0

occupancy_grid:
  unknown_is_traversable: false

map_candidates:
  max_candidates: 3
```

Tasks are stored in `configs/instructions.json`. Each task can specify scene path, robot start pose, instruction text, and task metadata.

## Debug Outputs

```text
debug/
  local/
    vln_debug.png
    llm_request.png
    videos/vln_debug_YYYYMMDD_HHMMSS.mp4
  eval/
    vln_debug.png
    videos/vln_debug_YYYYMMDD_HHMMSS.mp4

logs/
  vln.log
  last_candidate_choice_request.json
```

The console intentionally shows only navigation milestones: `[PLAN]`, `[NAV_STATE]`, `[LANDMARK_CONFIRMED]`, `[STEP_COMPLETE]`, `[SAFETY_HOLD]`, `[RECOVERY]`, `[ERROR]`, and `[ROUTE_COMPLETE]`. Full component logs are overwritten in `logs/vln.log` on each run. The human debug frame keeps its status bar outside the imagery: north-up global map on the left and uncropped first-person RGB on the right.

## Demo

The current prototype can complete simple landmark-driven navigation, but candidate choice can still oscillate in cluttered scenes.

[![Watch the VLM-NAV demo](docs/assets/display_thumb.png)](docs/assets/display.mp4)

Click the preview to watch the H.264 demo directly with GitHub's video player.

## Known Limitations

- VLM API latency is still the dominant delay between atomic primitives.
- GroundingDINO proposals can still be ambiguous; VLM verification reduces but cannot eliminate false positives.
- Detections without usable depth can guide image-space centering, but cannot create a world anchor or complete a landmark.
- Search decisions still depend on VLM quality, while confirmed-target approach is deterministic.
- The system is a research prototype, not a benchmark-final VLN agent.

## Research Context

VLM-NAV adapts ideas from VLN, ObjectNav, open-vocabulary grounding, semantic memory, and map-aware robot navigation into an Isaac Sim workflow for a custom wheel-driven robot.

Inspired by R2R, VLN-CE, DUET, NavGPT, Open-Nav, Safe-VLN, and CA-Nav.
