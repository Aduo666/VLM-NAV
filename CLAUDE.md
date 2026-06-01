# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

VLM-NAV is an Isaac Sim testbed for vision-language navigation with a custom 4-wheel-steering robot. A VLM (Qwen3-VL via SiliconFlow API) receives front camera images and obstacle context, then selects discrete atomic primitives (FORWARD/TURN_LEFT/TURN_RIGHT/STOP). The system runs in Isaac Sim 5.1 standalone mode with RGB-D, LiDAR, and occupancy grid mapping.

**This is a research prototype, not a production system.** Performance is not state-of-the-art; the platform exists to test VLN ideas in simulation.

## Commands

```bash
# Run navigation (activates conda env_isaaclab automatically)
bash scripts/run_nav.sh --task hospital_001

# Run with specific config
CONFIG_PATH=configs/default.yaml bash scripts/run_nav.sh --task hospital_001

# Run InteriorNav evaluation
SPLIT=val_seen EPISODE_INDEX=0 bash eval/run_interiornav.sh

# Test LLM API call
python scripts/test_llm_call.py

# Syntax check all modified Python files
python3 -c "import ast; [ast.parse(open(f).read()) for f in ['src/vln_nav/app/nav.py', 'src/vln_nav/navigation/vln_policy.py', 'src/vln_nav/navigation/direct_action_reasoner.py']]"
```

Conda environment: `env_isaaclab` (set via `CONDA_ENV_NAME` env var to override).

## Architecture

The pipeline is: **perception → VLM decision → atomic primitive → safety check → control**.

### Entry point and main loop

`src/vln_nav/app/nav.py` — single file containing the entire main loop. Each frame:
1. Isaac Sim `world.step()` — physics + render
2. Read RGB, depth, LiDAR → update occupancy grid
3. Call `policy.step(observation)` → get action
4. Safety layer may override action (depth-based obstacle avoidance)
5. `drive_controller.apply()` → set joint velocities

### Policy (VLM decision)

`src/vln_nav/navigation/vln_policy.py` — `OpenNavVLNPolicy.step()` (2334 lines) checks in priority order:
1. Route complete → STOP
2. Active primitive executing → continue (odom-based completion)
3. Need new decision → call `OpenNavReasoner.choose_candidate()`

Module-level helper functions extracted to `policy_helpers.py`.
VLM prompt shortened from 17 rules + 15-step procedure to 6 core rules.

`src/vln_nav/navigation/vln_reasoner.py` — sends annotated candidate image (LiDAR map + front camera with candidate markers) + structured prompt to Qwen3-VL, expects JSON with `{reason, target_visible, target_bearing_hint, target_memory_state, selected_action, selected_candidate_id}`.

### Perception modules

- `target_grounding.py` — GroundingDINO for visual target detection (requires CUDA-compatible PyTorch)
- `target_tracker.py` — OpenCV CSRT tracker for continuous bbox tracking between grounding calls
- `landmark_tracker.py` — accumulates observations into landmark state (evidence levels: none → potential → confirmed)
- `occupancy_grid.py` — log-odds occupancy mapping from LiDAR beams
- `depth_safety.py` — front/left/right depth check, overrides unsafe actions to STOP

### Control

`src/vln_nav/control/controller.py` — maps action names to 4-wheel-steering joint targets. Supports spin turn mode (`spin_4ws`) and Ackermann arc turns. Primitive completion is odom-based (FORWARD 0.55m, TURN 15°).

### Configuration

All config in `configs/default.yaml`. Task definitions in `configs/instructions.json` (each task: scene USD, start pose, instruction text). Config values are read as flat attributes on `AppConfig` dataclass in `runtime/config.py`.

Environment variables override config: `CONFIG_PATH`, `LLM_API_KEY`, `MODEL_NAME`, `LOG_PATH`.

## Key Design Decisions

- **Discrete primitives over continuous control**: VLM chooses from {FORWARD, TURN_LEFT, TURN_RIGHT, STOP}, not continuous waypoints. BACKWARD has been removed entirely — when stuck or blocked, the robot redirects to opposite turn direction.
- **Candidate-based mode**: `vln_reasoner.py` sends annotated candidate images (LiDAR map + front camera overlaid with candidate markers) to VLM. VLM selects from filtered "safe menu" of options rather than free-form actions.
- **Occupancy grid is LiDAR-only**: Depth camera is used for safety checks and target depth estimation, not for occupancy mapping.
- **LLM uses SiliconFlow API**: `open_nav_client.py` wraps the OpenAI-compatible chat API. Model, API key, and base URL are in config.

## Debugging

- `debug/vln_debug.png` — composite image: left=occupancy map, right=front camera with annotations
- `debug/model_front.png` — image sent to VLM as input
- `debug/last_direct_action_request.json` — full prompt + metadata of last VLM call
- `logs/nav_latest.log` — all runtime logs
- `debug/videos/vln_debug.mp4` — recorded debug video

## Known Issues

- GroundingDINO C++ CUDA extension (`_C` module) was never compiled. The code has been patched to fall back to a pure-Python deformable attention implementation. Works correctly on GPU but slower than the compiled CUDA kernel.
- Chinese instruction parsing relies on LLM preparse. If LLM is unavailable, heuristic parser only recognizes English landmarks (wheelchair, stairs, door, etc.) and English connectors (then, and then).
- The VLM tends to hallucinate `target_visible=true` even when the target is not in frame. Prompt explicitly instructs honesty but results vary by model.
- BACKWARD action has been completely removed from all code paths. Safety layer now redirects blocked turns to the opposite direction. Stuck recovery uses pause-then-turn instead of backward.
