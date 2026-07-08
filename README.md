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
- **Semantic grounding + memory**: GroundingDINO and VLM verification maintain landmark observations and map anchors.
- **Map-aware VLM decision**: the VLM receives a clean first-person candidate image, structured map text, route context, and an explicit candidate action table.
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

Grounding is part of the current main navigation method. GroundingDINO provides the local open-vocabulary detections used by landmark memory, semantic map updates, and arrival checks. VLM bbox fallback can help when the detector misses, but it is not the intended primary grounding path.

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

`scripts/setup_grounded_sam_local.sh` prints symlink commands if you already have the CA-Nav weights locally.

## Run

### Local Navigation

```bash
conda activate vlmnav
export SILICONFLOW_API_KEY="your_api_key"
export CONDA_ENV_NAME=vlmnav

# Run the default task from configs/default.yaml / configs/instructions.json.
bash scripts/run_nav.sh

# Run a named task.
bash scripts/run_nav.sh --task hospital_001

# List available tasks.
bash scripts/run_nav.sh --list-tasks

# Headless mode.
bash scripts/run_nav.sh --headless
```

The launch scripts read `CONDA_ENV_NAME`. Keep it set to `vlmnav` because the old `env_isaaclab` environment is no longer used:

```bash
CONDA_ENV_NAME=vlmnav bash scripts/run_nav.sh
```

### Task Authoring Panel

```bash
conda activate vlmnav
export CONDA_ENV_NAME=vlmnav
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
language instruction
  -> instruction parsing
  -> RGB-D + LiDAR + odometry observation
  -> target grounding and landmark memory
  -> structured map / route / obstacle summary
  -> occupancy-aware candidate generation
  -> VLM local decision
  -> discrete action primitive
  -> low-level 4WS controller
```

During primitive execution, the loop does not request new VLM decisions or run heavy grounding. It continues only the work needed for safe physical execution and map maintenance.

The VLM input is intentionally split by role:

- A clean first-person candidate image for visual choice.
- Structured text for semantic map, landmark memory, obstacle distances, route context, and candidate action metadata.
- No raw occupancy image is sent to the VLM by default; occupancy visualization is mainly for human debugging.

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
  backend: "auto"
  semantic_verification_enabled: true
  semantic_verification_min_confidence: 0.50

opennav:
  candidate_slot_bearings_deg: [35, 18, 0, -18, -35]
  arrival_distance_m: 2.0
  success_distance_m: 3.0
```

Tasks are stored in `configs/instructions.json`. Each task can specify scene path, robot start pose, instruction text, and task metadata.

## Debug Outputs

```text
debug/
  local/
    vln_debug.png
    llm_request.png
    depth_debug.png
    videos/vln_debug_YYYYMMDD_HHMMSS.mp4
  eval/
    vln_debug.png
    depth_debug.png
    videos/vln_debug_YYYYMMDD_HHMMSS.mp4

logs/
  nav_latest.log
  last_candidate_choice_request.json
```

## Demo

The current prototype can complete simple landmark-driven navigation, but candidate choice can still oscillate in cluttered scenes.

<video src="docs/assets/firstsuccess.mp4" controls width="720"></video>

If GitHub does not render the MP4 inline, open `docs/assets/firstsuccess.mp4` directly from the repository file browser.

## Known Limitations

- VLM API latency is still the dominant delay between atomic primitives.
- GroundingDINO can produce false positives for visually ambiguous objects such as stairs, shadows, and furniture edges.
- Detections without usable depth cannot create reliable world anchors.
- Candidate selection may still oscillate between adjacent left/right actions in tight layouts.
- The system is a research prototype, not a benchmark-final VLN agent.

## Research Context

VLM-NAV adapts ideas from VLN, ObjectNav, open-vocabulary grounding, semantic memory, and map-aware robot navigation into an Isaac Sim workflow for a custom wheel-driven robot.

Inspired by R2R, VLN-CE, DUET, NavGPT, Open-Nav, Safe-VLN, and CA-Nav.
