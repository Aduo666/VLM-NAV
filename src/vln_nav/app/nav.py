from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..control.controller import (
    SteeringDriveController,
    resolve_front_steering_joint_groups,
    resolve_hold_joint_names,
    resolve_wheel_joint_groups,
)
from ..navigation.candidate_waypoint_predictor import CandidateWaypointPredictor, CandidateWaypointPredictorConfig
from ..navigation.depth_safety import DepthSafetyConfig, DepthSafetyMonitor
from ..navigation.instruction_manager import InstructionManager
from ..navigation.landmark_map import LandmarkMap, LandmarkMapConfig, LandmarkMapEntry
from ..navigation.landmark_tracker import LandmarkState, LandmarkTracker, LandmarkTrackerConfig
from ..navigation.lidar_sensor import (
    MultiLidarConfig,
    MultiLidarReader,
    MultiLidarSectorConfig,
    PhysXLidarConfig,
    PhysXLidarReader,
)
from ..navigation.map_candidate_generator import MapCandidateGenerator, MapCandidateGeneratorConfig
from ..navigation.navigation_map_visualizer import NavigationMapConfig, NavigationMapVisualizer
from ..navigation.occupancy_grid import OccupancyGridConfig, OccupancyGridMapper
from ..navigation.observation_adapter import ObservationAdapter
from ..navigation.odometry import SimOdometrySensor
from ..navigation.open_nav_client import OpenNavLLMClient
from ..navigation.progress_manager import ProgressManager
from ..navigation.spatial_transform import log_spatial_transform_self_check
from ..navigation.stuck_detector import StuckDetector, StuckDetectorConfig
from ..navigation.target_grounding import TargetGrounder, TargetGroundingConfig, TargetObservation
from ..navigation.target_tracker import TargetTracker, TargetTrackerConfig
from ..navigation.topological_memory import TopologicalMemory, TopologicalMemoryConfig
from ..navigation.vln_policy import ExecutionFeedback, OpenNavPolicyConfig, OpenNavVLNPolicy, PolicyStepResult
from ..navigation.vln_reasoner import OpenNavReasoner
from ..runtime.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, AppConfig, load_config, setup_logging
from ..runtime.debug_video import DebugVideoRecorder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isaac Sim Open-Nav style VLN runtime")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path")
    parser.add_argument("--task", default=os.getenv("TASK_ID", ""), help="Task id from configs/instructions.json")
    parser.add_argument("--tasks-file", default=str(PROJECT_ROOT / "configs" / "instructions.json"), help="JSON task list path")
    parser.add_argument("--list-tasks", action="store_true", help="List available task ids and exit")
    parser.add_argument("--headless", action="store_true", help="Force headless simulation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.headless:
        config.headless = True

    logger = setup_logging(config.log_level)
    if args.list_tasks:
        _print_available_tasks(tasks_file=Path(args.tasks_file), logger=logger)
        return 0
    applied_task_id = _apply_task_overrides(
        config=config,
        task_id=str(args.task).strip(),
        tasks_file=Path(args.tasks_file),
        logger=logger,
    )
    logger.info("Loaded config from %s", Path(args.config).resolve())
    if applied_task_id:
        logger.info("Loaded task from %s: %s", Path(args.tasks_file).resolve(), applied_task_id)
    logger.info("Robot USD path: %s", config.robot_usd_path)
    logger.info("LLM model=%s api_base=%s key_present=%s", config.model_name, config.api_base_url, bool(config.llm_api_key))
    if not config.llm_api_key:
        logger.error(
            "LLM API key is missing. Set SILICONFLOW_API_KEY or LLM_API_KEY, "
            "or put api_key under the llm section in configs/default.yaml. "
            "This project has no local navigation fallback, so the robot will not move without the VLM."
        )
        return 2

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    simulation_settings = _build_simulation_app_settings(config=config, logger=logger)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(simulation_settings)

    try:
        run_navigation(simulation_app=simulation_app, config=config)
    except Exception:
        logger.exception("Navigation loop crashed.")
        return 1
    finally:
        simulation_app.close()

    return 0


def _apply_task_overrides(*, config: AppConfig, task_id: str, tasks_file: Path, logger) -> str:
    tasks_path = tasks_file if tasks_file.is_absolute() else PROJECT_ROOT / tasks_file
    if not tasks_path.exists():
        if task_id:
            raise FileNotFoundError(f"Task file does not exist: {tasks_path}")
        if not str(config.instruction).strip():
            raise RuntimeError(
                "No task selected and config has no opennav.instruction. "
                "Use --task <id> or set INSTRUCTION."
            )
        logger.info("[Task] no task file found; using instruction from config/env")
        return ""

    tasks = _load_tasks(tasks_path)
    selected = None
    requested = str(task_id).strip()
    if requested:
        # Exact match first, then prefix match
        for task in tasks:
            if str(task.get("id", "")).strip() == requested:
                selected = task
                break
        if selected is None:
            candidates = [t for t in tasks if str(t.get("id", "")).strip().startswith(requested)]
            if len(candidates) == 1:
                selected = candidates[0]
            elif len(candidates) > 1:
                ids = [str(t.get("id", "")) for t in candidates]
                raise ValueError(f"Ambiguous task prefix '{requested}': {', '.join(ids)}")
        if selected is None:
            available = ", ".join(str(task.get("id", "")) for task in tasks)
            raise ValueError(f"Task id not found: {requested}. Available tasks: {available}")
    elif not str(config.instruction).strip() and tasks:
        if len(tasks) == 1:
            selected = tasks[0]
        else:
            selected = _interactive_task_selection(tasks=tasks)
            if selected is None:
                raise SystemExit("No task selected. Exiting.")

    if selected is None:
        logger.info("[Task] no task selected; using instruction from config/env")
        return ""

    selected_id = str(selected.get("id", "")).strip() or "unnamed_task"
    instruction = str(selected.get("instruction", "")).strip()
    if not instruction:
        raise ValueError(f"Task {selected_id} has empty instruction")
    config.instruction = instruction

    scene_usd_path = str(selected.get("scene_usd_path", "")).strip()
    if scene_usd_path:
        config.scene_usd_path = scene_usd_path

    start_position = selected.get("robot_start_position", None)
    if isinstance(start_position, list) and len(start_position) >= 3:
        config.robot_init_position = [float(start_position[0]), float(start_position[1]), float(start_position[2])]

    start_orientation = selected.get("robot_start_orientation_deg", None)
    if isinstance(start_orientation, list) and len(start_orientation) >= 3:
        config.robot_init_orientation_deg = [
            float(start_orientation[0]),
            float(start_orientation[1]),
            float(start_orientation[2]),
        ]

    logger.info(
        "[Task] selected=%s scene=%s start=%s rpy_deg=%s instruction=%s",
        selected_id,
        config.scene_usd_path,
        config.robot_init_position,
        config.robot_init_orientation_deg,
        config.instruction,
    )
    return selected_id


def _interactive_task_selection(*, tasks: list[dict]) -> dict | None:
    print("\n" + "=" * 60)
    print("  Available Tasks")
    print("=" * 60)
    for i, task in enumerate(tasks):
        tid = str(task.get("id", ""))
        instruction = str(task.get("instruction", ""))
        scene = str(task.get("scene_usd_path", ""))
        enabled = task.get("enabled", True)
        marker = " " if enabled else " [disabled]"
        print(f"  [{i + 1}] {tid}{marker}")
        print(f"      instruction: {instruction}")
        if scene:
            print(f"      scene: {scene}")
    print("=" * 60)
    print("  Enter number or task id (q to quit): ", end="", flush=True)
    try:
        choice = input().strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not choice or choice.lower() == "q":
        return None
    # Try as number
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(tasks):
            return tasks[idx]
        print(f"  Invalid number: {choice}")
        return None
    # Try as task id (exact, then prefix)
    for task in tasks:
        if str(task.get("id", "")).strip() == choice:
            return task
    candidates = [t for t in tasks if str(t.get("id", "")).strip().startswith(choice)]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        ids = [str(t.get("id", "")) for t in candidates]
        print(f"  Ambiguous: {', '.join(ids)}")
    else:
        print(f"  Not found: {choice}")
    return None


def _print_available_tasks(*, tasks_file: Path, logger) -> None:
    tasks_path = tasks_file if tasks_file.is_absolute() else PROJECT_ROOT / tasks_file
    tasks = _load_tasks(tasks_path)
    logger.info("Available tasks in %s:", tasks_path)
    for task in tasks:
        logger.info(
            "  %s | scene=%s | start=%s | instruction=%s",
            str(task.get("id", "")),
            str(task.get("scene_usd_path", "")),
            task.get("robot_start_position", []),
            str(task.get("instruction", "")),
        )


def _load_tasks(tasks_path: Path) -> list[dict]:
    with tasks_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    tasks = data.get("tasks", data.get("episodes", [])) if isinstance(data, dict) else []
    if not isinstance(tasks, list):
        raise ValueError(f"Invalid task file, expected tasks list: {tasks_path}")
    return [task for task in tasks if isinstance(task, dict)]


def _build_simulation_app_settings(*, config: AppConfig, logger) -> dict:
    active_gpu = _resolve_active_gpu_for_display(config=config, logger=logger)
    settings = {
        "headless": bool(config.headless),
        "width": int(config.sim_width),
        "height": int(config.sim_height),
        "multi_gpu": bool(config.multi_gpu),
    }
    if active_gpu is not None:
        settings["active_gpu"] = int(active_gpu)
    if config.physics_gpu is not None:
        settings["physics_gpu"] = int(config.physics_gpu)
    logger.info(
        "[SimulationApp] headless=%s active_gpu=%s physics_gpu=%s multi_gpu=%s DISPLAY=%s WAYLAND_DISPLAY=%s CUDA_VISIBLE_DEVICES=%s",
        int(bool(config.headless)),
        "auto" if active_gpu is None else int(active_gpu),
        "auto" if config.physics_gpu is None else int(config.physics_gpu),
        int(bool(config.multi_gpu)),
        os.environ.get("DISPLAY", "<unset>"),
        os.environ.get("WAYLAND_DISPLAY", "<unset>"),
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    )
    return settings


def _resolve_active_gpu_for_display(*, config: AppConfig, logger) -> int | None:
    configured_gpu = config.active_gpu
    if bool(config.headless):
        return configured_gpu

    display_env = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    if not display_env:
        raise RuntimeError(
            "Non-headless Isaac Sim requires DISPLAY or WAYLAND_DISPLAY, but neither is set. "
            "Run from a desktop/NoMachine/VNC terminal, export DISPLAY correctly, or use --headless."
        )

    display_gpu = _detect_display_active_gpu(logger=logger)
    if configured_gpu is not None:
        if display_gpu is not None and int(configured_gpu) != int(display_gpu):
            logger.warning(
                "[DisplayGPU] configured active_gpu=%s but nvidia-smi reports display_active GPU=%s. "
                "Non-headless windows usually require active_gpu to match the display GPU.",
                int(configured_gpu),
                int(display_gpu),
            )
        return int(configured_gpu)

    if display_gpu is not None:
        logger.info("[DisplayGPU] auto-selected active_gpu=%s from nvidia-smi display_active.", int(display_gpu))
        return int(display_gpu)

    logger.warning(
        "[DisplayGPU] could not auto-detect a display-active GPU. Isaac Sim will choose its default renderer GPU. "
        "If the window does not appear, set ACTIVE_GPU to the GPU with Disp.A/Display Active enabled."
    )
    return None


def _detect_display_active_gpu(*, logger) -> int | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,display_active",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception as exc:
        logger.warning("[DisplayGPU] nvidia-smi display query failed: %s", exc)
        return None
    if completed.returncode != 0:
        logger.warning("[DisplayGPU] nvidia-smi display query failed: %s", completed.stderr.strip())
        return None
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        state = parts[1].strip().lower()
        if state in {"enabled", "active", "yes", "on", "1", "true"}:
            try:
                return int(parts[0])
            except ValueError:
                continue
    return None


def run_navigation(*, simulation_app, config: AppConfig) -> None:
    logger = setup_logging(config.log_level)

    llm_client = OpenNavLLMClient(
        api_key=config.llm_api_key,
        model_name=config.model_name,
        base_url=config.api_base_url,
        timeout_sec=config.llm_timeout_sec,
        retry_cooldown_sec=config.llm_retry_cooldown_sec,
        upload_max_edge_px=config.llm_upload_max_edge_px,
        image_detail=config.camera_image_detail,
        logger=logger,
    )
    target_grounding_llm_client = OpenNavLLMClient(
        api_key=config.llm_api_key,
        model_name=config.model_name,
        base_url=config.api_base_url,
        timeout_sec=config.llm_timeout_sec,
        retry_cooldown_sec=config.llm_retry_cooldown_sec,
        upload_max_edge_px=config.target_grounding_upload_max_edge_px,
        image_detail=config.camera_image_detail,
        logger=logger,
    )
    instruction_manager = InstructionManager(
        llm_client=llm_client,
        logger=logger,
        preparse_enabled=bool(config.instruction_preparse_enabled),
        max_tokens=int(config.instruction_preparse_max_tokens),
    )
    instruction_context = instruction_manager.parse(config.instruction)
    logger.info("[Goal] instruction=%s", instruction_context.raw_text)
    logger.info(
        "[Goal] parse_source=%s parse_reason=%s task_type=%s route_steps=%s",
        instruction_context.parsing_source,
        instruction_context.parsing_reason,
        instruction_context.task_type,
        [
            {
                "landmark": step.landmark,
                "directive": step.directive,
                "behavior_type": step.behavior_type,
            }
            for step in instruction_context.route_steps
        ],
    )
    logger.info("[Goal] actions=%s", instruction_context.actions)
    logger.info("[Goal] landmarks=%s", instruction_context.landmarks)
    logger.info(
        "[Route] current_landmark=%s ordered_landmarks=%s arrival_distance_m=%.2f success_distance_m=%.2f tracker_min_visible=%s recovery_enabled=%s recovery_stalled_steps=%s",
        instruction_context.goal_phrase or "none",
        instruction_context.landmarks,
        float(config.arrival_distance_m),
        float(config.success_distance_m),
        int(config.landmark_tracker_min_visible_count),
        int(config.recovery_enabled),
        int(config.recovery_trigger_stalled_steps),
    )
    logger.info(
        "[GroundingConfig] enabled=%s backend=%s call_period_sec=%.2f reuse_sec=%.2f pause_active=%s min_conf=%.2f depth_range=[%.2f, %.2f] coord=%s depth_region=%s min_depth_valid_ratio=%.2f semantic_verify=%s verify_min=%.2f dino_checkpoint=%s sam_enabled=%s sam_checkpoint=%s sam_model=%s",
        int(bool(config.target_grounding_enabled)),
        str(config.target_grounding_backend),
        float(config.target_grounding_call_period_sec),
        float(config.target_grounding_reuse_last_observation_sec),
        int(bool(config.target_grounding_pause_during_active_primitive)),
        float(config.target_grounding_min_confidence),
        float(config.target_grounding_min_depth_m),
        float(config.target_grounding_max_depth_m),
        str(config.target_grounding_coordinate_mode),
        str(config.target_grounding_depth_region),
        float(config.target_grounding_min_depth_valid_ratio),
        int(bool(config.target_grounding_semantic_verification_enabled)),
        float(config.target_grounding_semantic_verification_min_confidence),
        str(config.target_grounding_dino_checkpoint_path),
        int(bool(config.target_grounding_sam_enabled)),
        str(config.target_grounding_sam_checkpoint_path),
        str(config.target_grounding_sam_model_type),
    )
    logger.info(
        "[TargetTrackerConfig] enabled=%s type=%s lost_frames=%s lost_sec=%.2f min_depth_valid_ratio=%.2f",
        int(bool(config.target_tracker_enabled)),
        str(config.target_tracker_type),
        int(config.target_tracker_max_lost_frames),
        float(config.target_tracker_lost_confirmed_sec),
        float(config.target_tracker_min_depth_valid_ratio),
    )
    logger.info(
        "[LandmarkTrackerConfig] arrival_distance=%.2f success_distance=%.2f min_visible=%s confirmations=%s landmark_map=%s map_ttl=%.1f reacquire_guard=%s recent_visible_sec=%.1f",
        float(config.arrival_distance_m),
        float(config.success_distance_m),
        int(config.landmark_tracker_min_visible_count),
        int(config.landmark_tracker_arrival_confirmations),
        int(bool(config.landmark_map_enabled)),
        float(config.landmark_map_anchor_ttl_sec),
        int(bool(config.reacquire_guard_enabled)),
        float(config.reacquire_guard_recent_visible_sec),
    )

    observation_adapter = ObservationAdapter(
        camera_width=int(config.camera_width),
        camera_height=int(config.camera_height),
        horizontal_fov_deg=float(config.camera_horizontal_fov_deg),
    )
    log_spatial_transform_self_check(
        logger=logger,
        image_width=int(config.camera_width),
        image_height=int(config.camera_height),
        horizontal_fov_deg=float(config.camera_horizontal_fov_deg),
    )
    candidate_predictor = CandidateWaypointPredictor(
        CandidateWaypointPredictorConfig(
            image_width=int(config.camera_width),
            image_height=int(config.camera_height),
            horizontal_fov_deg=float(config.camera_horizontal_fov_deg),
            roi_vertical_range=(float(config.roi_vertical_range[0]), float(config.roi_vertical_range[1])),
            valid_depth_range=(float(config.valid_depth_range_m[0]), float(config.valid_depth_range_m[1])),
            clearance_threshold_m=float(config.clearance_threshold_m),
            slot_bearings_deg=tuple(float(value) for value in config.candidate_slot_bearings_deg),
            min_travel_distance_m=float(config.candidate_min_travel_distance_m),
            preferred_distance_m=float(config.candidate_preferred_distance_m),
            max_travel_distance_m=float(config.candidate_max_travel_distance_m),
            min_free_ratio=float(config.candidate_min_free_ratio),
            column_half_width_px=int(config.candidate_column_half_width_px),
            debug_image_path=str(config.candidate_debug_image_path),
            adaptive_enabled=bool(config.candidate_adaptive_enabled),
            adaptive_count=int(config.candidate_adaptive_count),
            adaptive_min_spacing_deg=float(config.candidate_adaptive_min_spacing_deg),
            adaptive_smoothing_px=int(config.candidate_adaptive_smoothing_px),
        ),
        logger=logger,
    )
    depth_safety_monitor = DepthSafetyMonitor(
        config=DepthSafetyConfig(
            enabled=bool(config.obstacle_avoidance_enabled),
            stop_distance_m=float(config.obstacle_stop_distance_m),
            roi_vertical_range=(
                float(config.obstacle_roi_vertical_range[0]),
                float(config.obstacle_roi_vertical_range[1]),
            ),
            center_width_ratio=float(config.obstacle_center_width_ratio),
            side_width_ratio=float(config.obstacle_side_width_ratio),
            near_field_memory_enabled=bool(config.near_field_memory_enabled),
            near_field_memory_sec=float(config.near_field_memory_sec),
        )
    )
    stuck_detector = StuckDetector(
        config=StuckDetectorConfig(
            enabled=bool(config.stuck_detection_enabled),
            window_sec=float(config.stuck_window_sec),
            displacement_threshold_m=float(config.stuck_displacement_threshold_m),
            yaw_threshold_deg=float(config.stuck_yaw_threshold_deg),
            recovery_pause_sec=float(config.stuck_recovery_pause_sec),
            recovery_turn_sec=float(config.stuck_recovery_turn_sec),
            recovery_turn_action=str(config.stuck_recovery_turn_action),
        ),
        logger=logger,
    )
    progress_manager = ProgressManager(
        history_length=int(config.progress_history_length),
        stall_distance_m=float(config.progress_stall_distance_m),
        spin_yaw_deg=float(config.progress_spin_yaw_deg),
    )
    target_grounder = TargetGrounder(
        config=TargetGroundingConfig(
            enabled=bool(config.target_grounding_enabled),
            backend=str(config.target_grounding_backend),
            call_period_sec=float(config.target_grounding_call_period_sec),
            reuse_last_observation_sec=float(config.target_grounding_reuse_last_observation_sec),
            upload_max_edge_px=int(config.target_grounding_upload_max_edge_px),
            min_bbox_area_ratio=float(config.target_grounding_min_bbox_area_ratio),
            max_bbox_area_ratio=float(config.target_grounding_max_bbox_area_ratio),
            min_bbox_width_px=int(config.target_grounding_min_bbox_width_px),
            min_bbox_height_px=int(config.target_grounding_min_bbox_height_px),
            max_bbox_aspect_ratio=float(config.target_grounding_max_bbox_aspect_ratio),
            min_depth_valid_ratio=float(config.target_grounding_min_depth_valid_ratio),
            depth_statistic=str(config.target_grounding_depth_statistic),
            center_patch_ratio=float(config.target_grounding_center_patch_ratio),
            min_depth_m=float(config.target_grounding_min_depth_m),
            max_depth_m=float(config.target_grounding_max_depth_m),
            min_confidence=float(config.target_grounding_min_confidence),
            horizontal_fov_deg=float(config.camera_horizontal_fov_deg),
            coordinate_mode=str(config.target_grounding_coordinate_mode),
            allow_normalized_bbox=bool(config.target_grounding_allow_normalized_bbox),
            normalized_bbox_scale=float(config.target_grounding_normalized_bbox_scale),
            depth_region=str(config.target_grounding_depth_region),
            depth_mask_tolerance_ratio=float(config.target_grounding_depth_mask_tolerance_ratio),
            depth_mask_min_pixels=int(config.target_grounding_depth_mask_min_pixels),
            grounding_dino_repo_path=str(config.target_grounding_dino_repo_path),
            grounding_dino_config_path=str(config.target_grounding_dino_config_path),
            grounding_dino_checkpoint_path=str(config.target_grounding_dino_checkpoint_path),
            grounding_dino_device=str(config.target_grounding_dino_device),
            grounding_dino_box_threshold=float(config.target_grounding_dino_box_threshold),
            grounding_dino_text_threshold=float(config.target_grounding_dino_text_threshold),
            sam_checkpoint_path=str(config.target_grounding_sam_checkpoint_path),
            sam_model_type=str(config.target_grounding_sam_model_type),
            sam_device=str(config.target_grounding_sam_device),
            sam_min_mask_area_ratio=float(config.target_grounding_sam_min_mask_area_ratio),
            sam_enabled=bool(config.target_grounding_sam_enabled),
            max_result_age_sec=float(config.target_grounding_max_result_age_sec),
            max_result_pose_delta_m=float(config.target_grounding_max_result_pose_delta_m),
            max_result_yaw_delta_deg=float(config.target_grounding_max_result_yaw_delta_deg),
            qwen_fallback_confirm_allowed=bool(config.target_grounding_qwen_fallback_confirm_allowed),
            semantic_verification_enabled=bool(config.target_grounding_semantic_verification_enabled),
            semantic_verification_min_confidence=float(config.target_grounding_semantic_verification_min_confidence),
            semantic_verification_max_edge_px=int(config.target_grounding_semantic_verification_max_edge_px),
            semantic_verification_auto_accept_confidence=float(config.target_grounding_semantic_verification_auto_accept_confidence),
        ),
        llm_client=target_grounding_llm_client,
        logger=logger,
    )
    target_tracker = TargetTracker(
        config=TargetTrackerConfig(
            enabled=bool(config.target_tracker_enabled),
            tracker_type=str(config.target_tracker_type),
            max_lost_frames=int(config.target_tracker_max_lost_frames),
            lost_confirmed_sec=float(config.target_tracker_lost_confirmed_sec),
            min_depth_valid_ratio=float(config.target_tracker_min_depth_valid_ratio),
            center_patch_ratio=float(config.target_tracker_center_patch_ratio),
            min_depth_m=float(config.target_grounding_min_depth_m),
            max_depth_m=float(config.target_grounding_max_depth_m),
            horizontal_fov_deg=float(config.camera_horizontal_fov_deg),
            min_bbox_area_ratio=float(config.target_grounding_min_bbox_area_ratio),
            max_bbox_area_ratio=float(config.target_grounding_max_bbox_area_ratio),
            confidence_decay_per_lost_frame=float(config.target_tracker_confidence_decay_per_lost_frame),
            min_tracking_confidence=float(config.target_tracker_min_tracking_confidence),
        ),
        logger=logger,
    )
    landmark_tracker = LandmarkTracker(
        config=LandmarkTrackerConfig(
            arrival_visible_depth_m=float(config.arrival_distance_m),
            arrival_anchor_radius_m=float(config.arrival_distance_m),
            min_visible_count=int(config.landmark_tracker_min_visible_count),
            arrival_confirmations=int(config.landmark_tracker_arrival_confirmations),
            arrival_window_size=int(config.landmark_tracker_arrival_window_size),
            arrival_recent_visible_sec=float(config.landmark_tracker_arrival_recent_visible_sec),
            max_lost_count=int(config.landmark_tracker_max_lost_count),
            anchor_update_min_consistent_observations=int(
                config.landmark_tracker_anchor_update_min_consistent_observations
            ),
            anchor_update_max_jump_m=float(config.landmark_tracker_anchor_update_max_jump_m),
            confirm_confidence_threshold=float(config.landmark_tracker_confirm_confidence_threshold),
            confirm_min_observations=int(config.landmark_tracker_confirm_min_observations),
            confirm_max_anchor_jump_m=float(config.landmark_tracker_confirm_max_anchor_jump_m),
            confirm_max_depth_jump_m=float(config.landmark_tracker_confirm_max_depth_jump_m),
            potential_confidence_threshold=float(config.landmark_tracker_potential_confidence_threshold),
        ),
        logger=logger,
    )
    landmark_map = LandmarkMap(
        config=LandmarkMapConfig(
            enabled=bool(config.landmark_map_enabled),
            anchor_ttl_sec=float(config.landmark_map_anchor_ttl_sec),
            min_confidence=float(config.landmark_map_min_confidence),
            log_period_sec=float(config.landmark_map_log_period_sec),
        ),
        logger=logger,
    )
    topological_memory = TopologicalMemory(
        config=TopologicalMemoryConfig(
            enabled=bool(config.topology_enabled),
            node_spacing_m=float(config.topology_node_spacing_m),
            turn_threshold_deg=float(config.topology_turn_threshold_deg),
            merge_radius_m=float(config.topology_merge_radius_m),
            landmark_bind_radius_m=float(config.topology_landmark_bind_radius_m),
            max_nodes=int(config.topology_max_nodes),
            recent_node_count=int(config.topology_recent_node_count),
        ),
        logger=logger,
    )
    odom_sensor = SimOdometrySensor(logger=logger, source=str(config.odom_source))
    occupancy_grid = OccupancyGridMapper(
        config=OccupancyGridConfig(
            enabled=bool(config.occupancy_grid_enabled),
            resolution_m=float(config.occupancy_grid_resolution_m),
            size_m=float(config.occupancy_grid_size_m),
            update_period_sec=float(config.occupancy_grid_update_period_sec),
            depth_sample_stride_px=int(config.occupancy_grid_depth_sample_stride_px),
            depth_roi_vertical_range=(
                float(config.occupancy_grid_depth_roi_vertical_range[0]),
                float(config.occupancy_grid_depth_roi_vertical_range[1]),
            ),
            min_depth_m=float(config.occupancy_grid_min_depth_m),
            max_depth_m=float(config.occupancy_grid_max_depth_m),
            inflation_radius_m=float(config.occupancy_grid_inflation_radius_m),
            occupied_min_neighbors=int(config.occupancy_grid_occupied_min_neighbors),
            robot_radius_m=float(config.occupancy_grid_robot_radius_m),
            path_corridor_radius_m=float(config.occupancy_grid_path_corridor_radius_m),
            turn_clearance_radius_m=float(config.occupancy_grid_turn_clearance_radius_m),
            unknown_is_traversable=bool(config.occupancy_grid_unknown_is_traversable),
            unknown_cell_cost=float(config.occupancy_grid_unknown_cell_cost),
            debug_enabled=bool(config.occupancy_grid_debug_enabled),
            debug_image_path=str(config.occupancy_grid_debug_image_path),
            debug_update_period_sec=float(config.occupancy_grid_debug_update_period_sec),
            debug_width=int(config.occupancy_grid_debug_width),
            debug_height=int(config.occupancy_grid_debug_height),
            debug_view_size_m=float(config.occupancy_grid_debug_view_size_m),
            debug_show_free_space=bool(config.occupancy_grid_debug_show_free_space),
            semantic_planner_enabled=bool(config.occupancy_grid_semantic_planner_enabled),
            semantic_goal_step_m=float(config.occupancy_grid_semantic_goal_step_m),
            semantic_goal_min_step_m=float(config.occupancy_grid_semantic_goal_min_step_m),
            semantic_goal_lateral_search_m=float(config.occupancy_grid_semantic_goal_lateral_search_m),
        ),
        logger=logger,
    )
    _lidar_mode = str(config.lidar_mode).strip().lower()
    if _lidar_mode == "multi" and bool(config.lidar_enabled):
        _sector_count = min(
            int(config.lidar_multi_sector_count),
            len(config.lidar_multi_child_names),
            len(config.lidar_multi_sector_yaw_offsets_deg),
        )
        multi_sectors = [
            MultiLidarSectorConfig(
                child_name=str(config.lidar_multi_child_names[i]),
                yaw_offset_deg=float(config.lidar_multi_sector_yaw_offsets_deg[i]),
                horizontal_fov_deg=float(config.lidar_multi_sector_fov_deg),
            )
            for i in range(_sector_count)
        ]
        lidar_reader: PhysXLidarReader | MultiLidarReader = MultiLidarReader(
            config=MultiLidarConfig(
                enabled=bool(config.lidar_enabled),
                prim_path=str(config.lidar_prim_path),
                vertical_fov_deg=float(config.lidar_vertical_fov_deg),
                horizontal_resolution_deg=float(config.lidar_horizontal_resolution_deg),
                vertical_resolution_deg=float(config.lidar_vertical_resolution_deg),
                rotation_rate_hz=float(config.lidar_rotation_rate_hz),
                min_range_m=float(config.lidar_min_range_m),
                max_range_m=float(config.lidar_max_range_m),
                draw_points=bool(config.lidar_draw_points),
                draw_lines=bool(config.lidar_draw_lines),
                sectors=multi_sectors,
            ),
            robot_root_path=str(config.robot_prim_path),
            logger=logger,
        )
    else:
        lidar_reader = PhysXLidarReader(
            config=PhysXLidarConfig(
                enabled=bool(config.lidar_enabled),
                prim_path=str(config.lidar_prim_path),
                child_name=str(config.lidar_child_name),
                horizontal_fov_deg=float(config.lidar_horizontal_fov_deg),
                vertical_fov_deg=float(config.lidar_vertical_fov_deg),
                horizontal_resolution_deg=float(config.lidar_horizontal_resolution_deg),
                vertical_resolution_deg=float(config.lidar_vertical_resolution_deg),
                rotation_rate_hz=float(config.lidar_rotation_rate_hz),
                min_range_m=float(config.lidar_min_range_m),
                max_range_m=float(config.lidar_max_range_m),
                yaw_offset_deg=float(config.lidar_yaw_offset_deg),
                sector_merge_keep_sec=float(config.lidar_sector_merge_keep_sec),
                draw_points=bool(config.lidar_draw_points),
                draw_lines=bool(config.lidar_draw_lines),
            ),
            robot_root_path=str(config.robot_prim_path),
            logger=logger,
        )
    map_candidate_generator = MapCandidateGenerator(
        config=MapCandidateGeneratorConfig(
            enabled=bool(config.map_candidates_enabled),
            max_candidates=int(config.map_candidates_max_candidates),
            min_distance_m=float(config.map_candidates_min_distance_m),
            max_distance_m=float(config.map_candidates_max_distance_m),
            semantic_enabled=bool(config.map_candidates_semantic_enabled),
            frontier_enabled=bool(config.map_candidates_frontier_enabled),
            escape_enabled=bool(config.map_candidates_escape_enabled),
            min_clearance_m=float(config.map_candidates_min_clearance_m),
            unknown_gain_weight=float(config.map_candidates_unknown_gain_weight),
            semantic_weight=float(config.map_candidates_semantic_weight),
        ),
        logger=logger,
    )
    navigation_map_visualizer = NavigationMapVisualizer(
        config=NavigationMapConfig(
            enabled=bool(config.navigation_map_enabled),
            output_path=str(config.navigation_map_image_path),
            update_period_sec=float(config.navigation_map_update_period_sec),
            width=int(config.navigation_map_width),
            height=int(config.navigation_map_height),
            depth_sample_stride_px=int(config.navigation_map_depth_sample_stride_px),
            max_depth_obstacle_points=int(config.navigation_map_max_depth_obstacle_points),
        ),
        logger=logger,
    )
    reasoner = OpenNavReasoner(llm_client=llm_client, logger=logger)
    policy = OpenNavVLNPolicy(
        instruction_context=instruction_context,
        candidate_predictor=candidate_predictor,
        occupancy_grid=occupancy_grid if bool(config.occupancy_grid_enabled) else None,
        map_candidate_generator=map_candidate_generator,
        progress_manager=progress_manager,
        reasoner=reasoner,
        config=OpenNavPolicyConfig(
            execution_steps=int(config.policy_execution_steps),
            stop_distance_m=float(config.waypoint_stop_distance_m),
            heading_threshold_rad=float(config.waypoint_heading_threshold_rad),
            turn_speed_scale=float(config.waypoint_turn_speed_scale),
            forward_speed_scale=float(config.waypoint_forward_speed_scale),
            replanning_cooldown_sec=float(config.replanning_cooldown_sec),
            active_forward_duration_sec=float(config.action_forward_duration_sec),
            active_turn_duration_sec=float(config.action_turn_duration_sec),
            active_turn_angle_deg=float(config.action_turn_angle_deg),
            active_forward_distance_m=float(config.action_forward_distance_m),
            active_distance_tolerance_m=float(config.action_distance_tolerance_m),
            active_turn_tolerance_deg=float(config.action_turn_tolerance_deg),
            active_odom_completion_enabled=bool(config.action_odom_completion_enabled),
            active_track_max_yaw_deg=float(config.active_track_max_yaw_deg),
            active_track_min_duration_sec=float(config.active_track_min_duration_sec),
            use_discrete_action_primitives=bool(config.use_discrete_action_primitives),
            turn_at_max_turns=int(config.turn_at_max_turns),
            turn_at_complete_max_distance_m=float(config.landmark_tracker_turn_at_complete_max_distance_m),
            turn_at_complete_requires_confirmed_anchor=bool(config.landmark_tracker_turn_at_complete_requires_confirmed_anchor),
            confirm_min_observations=int(config.landmark_tracker_confirm_min_observations),
            confirm_confidence_threshold=float(config.landmark_tracker_confirm_confidence_threshold),
            landmark_arrival_confirmations=int(config.landmark_arrival_confirmations),
            landmark_arrival_window_size=int(config.landmark_tracker_arrival_window_size),
            reacquire_guard_enabled=bool(config.reacquire_guard_enabled),
            reacquire_guard_recent_visible_sec=float(config.reacquire_guard_recent_visible_sec),
            arrival_distance_m=float(config.arrival_distance_m),
            success_distance_m=float(config.success_distance_m),
            recovery_enabled=bool(config.recovery_enabled),
            recovery_trigger_stalled_steps=int(config.recovery_trigger_stalled_steps),
            recovery_execution_steps=int(config.recovery_execution_steps),
        ),
        logger=logger,
    )

    try:
        from isaacsim.core.prims import SingleArticulation as RobotArticulation
    except ImportError:
        from isaacsim.core.prims import Articulation as RobotArticulation

    from isaacsim.core.api import World
    from isaacsim.core.utils.numpy import rotations as rot_utils
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage, is_stage_loading
    from isaacsim.sensors.camera import Camera

    world = World(
        physics_dt=config.physics_dt,
        rendering_dt=config.rendering_dt,
        stage_units_in_meters=1.0,
    )
    stage = get_current_stage()

    has_environment = _load_environment(
        stage=stage,
        add_reference_to_stage=add_reference_to_stage,
        config=config,
        logger=logger,
    )
    if has_environment:
        _wait_for_stage_assets(
            simulation_app=simulation_app,
            is_stage_loading=is_stage_loading,
            timeout_sec=float(config.stage_load_timeout_sec),
            logger=logger,
            label="environment",
        )
    if config.add_default_ground_plane:
        world.scene.add_default_ground_plane()
        logger.info("Added default ground plane.")

    if getattr(config, "add_dome_light", False):
        try:
            from pxr import UsdLux, Gf, UsdGeom
            # Dome light for ambient outdoor lighting
            dome = UsdLux.DomeLight.Define(stage, "/World/defaultDomeLight")
            dome.CreateIntensityAttr().Set(3000.0)
            dome.CreateExposureAttr().Set(2.0)

            # Headlight attached to robot camera, follows the robot through rooms
            headlight = UsdLux.SphereLight.Define(stage, f"{config.robot_prim_path}/headlight")
            headlight.CreateIntensityAttr().Set(3000.0)
            headlight.CreateExposureAttr().Set(2.0)
            headlight.CreateRadiusAttr().Set(0.3)
            # Position slightly ahead and above the camera
            xform = UsdGeom.Xformable(headlight.GetPrim())
            xform.AddTranslateOp().Set(Gf.Vec3f(1.5, 0.0, 1.8))

            logger.info("Added dome+robot-headlight (add_dome_light=true)")
        except Exception as exc:
            logger.warning("Failed to add lights: %s", exc)

    add_reference_to_stage(str(config.robot_usd_path), config.robot_prim_path)
    _wait_for_stage_assets(
        simulation_app=simulation_app,
        is_stage_loading=is_stage_loading,
        timeout_sec=float(config.stage_load_timeout_sec),
        logger=logger,
        label="robot",
    )
    _set_local_xform(
        stage=stage,
        prim_path=config.robot_prim_path,
        translation=config.robot_init_position,
        rotation_deg=config.robot_init_orientation_deg,
    )

    articulation_path = config.robot_articulation_path or _find_articulation_root_path(
        stage=stage,
        root_path=config.robot_prim_path,
    )
    logger.info("Using articulation prim: %s", articulation_path)

    robot = world.scene.add(RobotArticulation(prim_path=articulation_path, name="custom_robot"))
    front_camera_prim_path = _resolve_robot_relative_prim_path(
        robot_root_path=config.robot_prim_path,
        prim_path=config.front_camera_prim_path,
    )
    observer_camera_prim_path = _resolve_robot_relative_prim_path(
        robot_root_path=config.robot_prim_path,
        prim_path=config.observer_camera_prim_path,
    )
    front_camera_prim_path = _resolve_existing_prim_path(
        stage=stage,
        prim_path=front_camera_prim_path,
        kind="front camera",
    )
    if config.observer_camera_enabled:
        observer_camera_prim_path = _resolve_existing_prim_path(
            stage=stage,
            prim_path=observer_camera_prim_path,
            kind="observer camera",
        )
    logger.info("Using front camera prim: %s", front_camera_prim_path)
    if config.observer_camera_enabled:
        logger.info("Using observer camera prim: %s", observer_camera_prim_path)

    camera = Camera(
        prim_path=front_camera_prim_path,
        frequency=max(1, int(config.sensor_frequency_hz)),
        resolution=(config.camera_width, config.camera_height),
    )
    observer_camera = None
    if config.observer_camera_enabled:
        observer_camera = Camera(
            prim_path=observer_camera_prim_path,
            frequency=max(1, int(config.sensor_frequency_hz)),
            resolution=(config.camera_width, config.camera_height),
        )

    _set_viewport_camera(config=config, logger=logger)

    logger.info("Calling world.reset()")
    world.reset()
    logger.info("world.reset() complete")
    robot.initialize()
    logger.info("robot.initialize() complete")
    camera.initialize()
    logger.info("front camera initialize() complete")
    _attach_camera_annotators(camera=camera, logger=logger, enable_depth=True)
    if observer_camera is not None:
        observer_camera.initialize()
        logger.info("observer camera initialize() complete")
        _attach_camera_annotators(camera=observer_camera, logger=logger, enable_depth=False)
    lidar_reader.initialize(stage=stage)

    _bind_viewport_to_camera(
        camera_prim_path=front_camera_prim_path,
        logger=logger,
        enabled=config.follow_front_camera_in_gui,
    )
    if observer_camera is not None:
        _bind_viewport_to_camera(
            camera_prim_path=observer_camera_prim_path,
            logger=logger,
            enabled=config.follow_observer_camera_in_gui,
        )

    available_joint_names = _get_joint_names(robot)
    logger.info("Detected articulation DOFs: %s", available_joint_names)
    left_joint_names, right_joint_names, steering_joint_names, error_message = resolve_wheel_joint_groups(
        available_joint_names=available_joint_names,
        configured_left=config.left_wheel_joint_names,
        configured_right=config.right_wheel_joint_names,
        configured_steering=config.steering_joint_names,
    )
    if error_message:
        raise RuntimeError(
            f"{error_message}\n"
            f"Available articulation DOFs: {available_joint_names}\n"
            "Example config for this robot:\n"
            "left_wheel_joint_names: [fl_wheel, rl_wheel]\n"
            "right_wheel_joint_names: [fr_wheel, rr_wheel]"
        )

    front_steering_joint_names, passive_steering_joint_names = resolve_front_steering_joint_groups(
        steering_joint_names
    )
    hold_joint_names = resolve_hold_joint_names(
        available_joint_names=available_joint_names,
        wheel_joint_names=[*left_joint_names, *right_joint_names],
        steering_joint_names=steering_joint_names,
    )
    hold_joint_targets = _build_hold_joint_targets(
        robot=robot,
        available_joint_names=available_joint_names,
        hold_joint_names=hold_joint_names,
    )
    drive_controller = SteeringDriveController(
        stage=stage,
        articulation_path=articulation_path,
        robot_root_path=config.robot_prim_path,
        left_wheel_joint_names=left_joint_names,
        right_wheel_joint_names=right_joint_names,
        steering_joint_names=front_steering_joint_names,
        passive_steering_joint_names=passive_steering_joint_names,
        hold_joint_targets=hold_joint_targets,
        wheel_drive_velocity=config.wheel_drive_velocity,
        turn_drive_velocity=config.turn_drive_velocity,
        steering_angle=config.steering_angle,
        turn_mode=config.turn_mode,
        turn_steering_angle=config.turn_steering_angle,
        rear_steering_ratio=config.rear_steering_ratio,
        wheel_drive_damping=config.wheel_drive_damping,
        steering_stiffness=config.steering_stiffness,
        steering_damping=config.steering_damping,
        hold_joint_stiffness=config.hold_joint_stiffness,
        hold_joint_damping=config.hold_joint_damping,
        track_steering_sign=config.track_steering_sign,
        logger=logger,
    )
    logger.info(
        "[DriveConfig] left_wheels=%s right_wheels=%s front_steering=%s rear_steering=%s turn_mode=%s turn_steering_angle=%.3f rear_steering_ratio=%.2f track_steering_sign=%.1f discrete_primitives=%s",
        left_joint_names,
        right_joint_names,
        front_steering_joint_names,
        passive_steering_joint_names,
        config.turn_mode,
        float(config.turn_steering_angle),
        float(config.rear_steering_ratio),
        float(config.track_steering_sign),
        int(bool(config.use_discrete_action_primitives)),
    )
    logger.info(
        "[OdomConfig] enabled=%s source=%s primitive_completion=%s forward_distance=%.2fm turn_angle=%.1fdeg tolerance=%.1fdeg",
        int(bool(config.odom_enabled)),
        str(config.odom_source),
        int(bool(config.action_odom_completion_enabled)),
        float(config.action_forward_distance_m),
        float(config.action_turn_angle_deg),
        float(config.action_turn_tolerance_deg),
    )
    depth_debug_output_path = _resolve_output_path(config.depth_debug_image_path)
    target_debug_output_path = _resolve_output_path(config.target_debug_image_path)
    target_debug_video_path = _resolve_output_path(config.debug_video_path)
    debug_video_recorder = DebugVideoRecorder(
        enabled=bool(config.debug_video_enabled) and bool(config.target_debug_enabled),
        output_path=target_debug_video_path,
        fps=float(config.debug_video_fps),
        logger=logger,
    )
    logger.info("VLM request image output: %s", _resolve_output_path(config.candidate_debug_image_path))
    if config.target_debug_enabled:
        logger.info("Runtime debug image output: %s", target_debug_output_path)
    if config.occupancy_grid_enabled:
        logger.info("Occupancy map is rendered in runtime debug only; VLM receives structured map text.")
    if config.debug_video_enabled and config.target_debug_enabled:
        logger.info("Runtime debug video output: %s", debug_video_recorder.output_path)
    if config.depth_debug_enabled:
        logger.info("Depth debug output: %s", depth_debug_output_path)
    if config.navigation_map_enabled:
        logger.info("Navigation map output: %s", navigation_map_visualizer.output_path)
    if config.topology_enabled:
        logger.info(
            "[TopologyConfig] enabled=1 node_spacing=%.2fm turn_threshold=%.1fdeg merge_radius=%.2fm bind_radius=%.2fm max_nodes=%s recent_nodes=%s",
            float(config.topology_node_spacing_m),
            float(config.topology_turn_threshold_deg),
            float(config.topology_merge_radius_m),
            float(config.topology_landmark_bind_radius_m),
            int(config.topology_max_nodes),
            int(config.topology_recent_node_count),
        )
    if config.occupancy_grid_enabled:
        logger.info(
            "[OccupancyGridConfig] enabled=1 source=%s resolution=%.2fm size=%.1fm inflation=%.2fm min_neighbors=%s path_corridor=%.2fm roi=%s debug=%s",
            "lidar" if bool(config.lidar_enabled) else "depth",
            float(config.occupancy_grid_resolution_m),
            float(config.occupancy_grid_size_m),
            float(config.occupancy_grid_inflation_radius_m),
            int(config.occupancy_grid_occupied_min_neighbors),
            float(config.occupancy_grid_path_corridor_radius_m),
            config.occupancy_grid_depth_roi_vertical_range,
            occupancy_grid.output_path,
        )
    if config.lidar_enabled:
        if _lidar_mode == "multi":
            logger.info(
                "[LidarConfig] mode=multi prim_path=%s sector_count=%s sector_fov=%.1f "
                "total_fov=%.1f res=[%.2f,%.2f] range=[%.2f,%.2f] yaw_offsets=%s child_names=%s",
                str(config.lidar_prim_path),
                int(config.lidar_multi_sector_count),
                float(config.lidar_multi_sector_fov_deg),
                float(config.lidar_multi_sector_count) * float(config.lidar_multi_sector_fov_deg),
                float(config.lidar_horizontal_resolution_deg),
                float(config.lidar_vertical_resolution_deg),
                float(config.lidar_min_range_m),
                float(config.lidar_max_range_m),
                config.lidar_multi_sector_yaw_offsets_deg,
                config.lidar_multi_child_names,
            )
        else:
            logger.info(
                "[LidarConfig] mode=single prim_path=%s child=%s fov=[%.1f,%.1f] res=[%.2f,%.2f] range=[%.2f,%.2f] yaw_offset=%.1f",
                str(config.lidar_prim_path),
                str(config.lidar_child_name),
                float(config.lidar_horizontal_fov_deg),
                float(config.lidar_vertical_fov_deg),
                float(config.lidar_horizontal_resolution_deg),
                float(config.lidar_vertical_resolution_deg),
                float(config.lidar_min_range_m),
                float(config.lidar_max_range_m),
                float(config.lidar_yaw_offset_deg),
            )

    last_depth_image: np.ndarray | None = None
    last_depth_eval_time = 0.0
    frame_index = 0
    last_action_name = "STOP"
    last_applied_wheel_velocity = 0.0
    last_applied_steering_angle = 0.0
    last_motion_log_time = 0.0
    last_logged_position = np.asarray(config.robot_init_position, dtype=np.float32)
    last_logged_policy_signature: tuple | None = None
    last_policy_feedback: ExecutionFeedback | None = None
    active_step_result = None
    active_command_token = 0
    initial_robot_orientation_xyzw: np.ndarray | None = None
    depth_warning_emitted = False
    depth_debug_save_pending = False
    last_safety_log_signature: tuple[str, str] | None = None
    last_nav_debug_save_time = 0.0
    last_depth_debug_save_time = 0.0
    sim_time_sec = 0.0
    route_complete_handled = False
    navigation_finished = False
    landmark_map_reached_synced: set[str] = set()
    global_pose_trace: list[tuple[float, float]] = []
    latest_global_map_image: Image.Image | None = None

    for _ in range(config.warmup_steps):
        world.step(render=True)
        sim_time_sec += float(config.physics_dt)
        drive_controller.apply("STOP", speed_scale=0.0)
    logger.info("Warmup complete, controller is armed.")
    occupancy_bootstrap_end_sec = sim_time_sec
    occupancy_bootstrap_deadline_sec = sim_time_sec
    occupancy_bootstrap_logged = False
    occupancy_bootstrap_done_logged = False
    occupancy_bootstrap_last_fov_bucket = -1
    if bool(config.occupancy_grid_enabled) and bool(config.lidar_enabled):
        occupancy_bootstrap_end_sec = sim_time_sec + max(0.0, float(config.occupancy_grid_lidar_bootstrap_sec))
        occupancy_bootstrap_deadline_sec = occupancy_bootstrap_end_sec + max(
            0.0,
            float(config.occupancy_grid_lidar_bootstrap_max_wait_sec),
        )

    while simulation_app.is_running():
        world.step(render=True)
        sim_time_sec += float(config.physics_dt)
        frame_index += 1
        loop_now = time.monotonic()
        position, orientation = _get_world_pose(robot)
        position = np.asarray(position, dtype=np.float32)
        orientation = np.asarray(orientation, dtype=np.float32)
        odom_state = odom_sensor.update(
            world_position_xyz=position,
            world_orientation_wxyz=orientation,
            timestamp_sec=sim_time_sec,
        )

        if initial_robot_orientation_xyzw is None:
            initial_robot_orientation_xyzw = orientation.copy()

        if config.tip_stop_enabled:
            roll_deg, pitch_deg = _relative_roll_pitch_deg(
                initial_quaternion_xyzw=initial_robot_orientation_xyzw,
                current_quaternion_xyzw=orientation,
            )
            if (
                abs(float(roll_deg)) >= float(config.tip_stop_roll_pitch_deg)
                or abs(float(pitch_deg)) >= float(config.tip_stop_roll_pitch_deg)
            ):
                logger.warning(
                    "Robot tip detected relative_roll=%.1f relative_pitch=%.1f. Stopping simulation.",
                    float(roll_deg),
                    float(pitch_deg),
                )
                drive_controller.apply("STOP", speed_scale=0.0)
                break

        if navigation_finished:
            drive_controller.apply("STOP", speed_scale=0.0)
            last_action_name = "STOP"
            continue

        if last_depth_image is None or (loop_now - last_depth_eval_time) >= float(config.depth_eval_period_sec):
            last_depth_image = _get_depth_image(camera=camera, logger=logger)
            last_depth_eval_time = loop_now
            depth_debug_save_pending = last_depth_image is not None
            if last_depth_image is None and not depth_warning_emitted:
                logger.warning("Depth frame unavailable, Open-Nav candidate predictor is waiting for the first valid frame.")
                depth_warning_emitted = True

        front_rgb = _get_rgb_image(camera=camera)
        observation = observation_adapter.adapt(
            rgb=front_rgb,
            depth=last_depth_image,
            robot_position_xyz=position,
            robot_orientation_xyzw=orientation,
            frame_index=frame_index,
            timestamp_sec=sim_time_sec,
            last_action=last_action_name,
            wheel_velocity=last_applied_wheel_velocity,
            steering_angle=last_applied_steering_angle,
            odom_pose=odom_state.pose if bool(config.odom_enabled) else None,
            odom_meta={
                "enabled": bool(config.odom_enabled),
                "source": odom_state.source,
                "linear_velocity_mps": float(odom_state.linear_velocity_mps),
                "angular_velocity_radps": float(odom_state.angular_velocity_radps),
                "delta_distance_m": float(odom_state.delta_distance_m),
                "delta_yaw_rad": float(odom_state.delta_yaw_rad),
                "total_distance_m": float(odom_state.total_distance_m),
            },
        )
        lidar_scan = lidar_reader.read(pose=observation.pose, timestamp_sec=sim_time_sec) if bool(config.lidar_enabled) else None
        if bool(config.lidar_enabled) and lidar_scan is not None:
            current_beam_endpoints = getattr(lidar_scan, "current_beam_endpoints_local_xy", lidar_scan.beam_endpoints_local_xy)
            current_beam_hits = getattr(lidar_scan, "current_beam_hit_mask", lidar_scan.beam_hit_mask)
            current_points = getattr(lidar_scan, "current_points_local_xy", lidar_scan.points_local_xy)
            current_beam_endpoints_np = np.asarray(current_beam_endpoints, dtype=np.float32)
            current_beam_hits_np = np.asarray(current_beam_hits, dtype=bool).reshape(-1)
            if (
                current_beam_endpoints_np.ndim == 2
                and current_beam_endpoints_np.shape[0] == current_beam_hits_np.shape[0]
                and np.any(current_beam_hits_np)
            ):
                map_beam_endpoints = current_beam_endpoints_np[current_beam_hits_np]
                map_beam_hits = np.ones((map_beam_endpoints.shape[0],), dtype=bool)
                map_points = map_beam_endpoints
            else:
                map_beam_endpoints = np.zeros((0, 2), dtype=np.float32)
                map_beam_hits = np.zeros((0,), dtype=bool)
                map_points = np.zeros((0, 2), dtype=np.float32)
            occupancy_grid.update_from_lidar(
                robot_pose=observation.pose,
                points_local_xy=map_points,
                beam_endpoints_local_xy=map_beam_endpoints,
                beam_hit_mask=map_beam_hits,
                timestamp_sec=sim_time_sec,
            )
            observation.meta["lidar"] = {
                "enabled": True,
                "source": lidar_scan.source,
                "sensor_path": lidar_scan.sensor_path,
                "point_count": int(lidar_scan.point_count),
                "current_point_count": int(getattr(lidar_scan, "current_point_count", lidar_scan.point_count)),
                "merged_beam_count": int(lidar_scan.beam_endpoints_local_xy.shape[0]),
                "raw_beam_count": int(lidar_scan.raw_beam_count),
                "valid_beam_count": int(lidar_scan.valid_beam_count),
                "effective_fov_deg": float(lidar_scan.effective_fov_deg),
            }
        elif bool(config.lidar_enabled):
            observation.meta["lidar"] = {
                "enabled": True,
                "source": "unavailable",
                "sensor_path": lidar_reader.sensor_path,
                "point_count": 0,
            }
        else:
            occupancy_grid.update(observation=observation)
            observation.meta["lidar"] = {
                "enabled": False,
                "source": "depth",
                "sensor_path": lidar_reader.sensor_path,
                "point_count": 0,
            }

        current_lidar_fov_deg = 0.0 if lidar_scan is None else float(lidar_scan.effective_fov_deg)
        bootstrap_min_fov_deg = max(0.0, float(config.occupancy_grid_lidar_bootstrap_min_fov_deg))
        bootstrap_fov_ready = (
            not bool(config.lidar_enabled)
            or bootstrap_min_fov_deg <= 0.0
            or current_lidar_fov_deg >= bootstrap_min_fov_deg
        )
        bootstrap_time_ready = sim_time_sec >= float(occupancy_bootstrap_end_sec)
        bootstrap_deadline_reached = sim_time_sec >= float(occupancy_bootstrap_deadline_sec)
        if not (bootstrap_time_ready and (bootstrap_fov_ready or bootstrap_deadline_reached)):
            if not occupancy_bootstrap_logged:
                logger.info(
                    "[OccupancyBootstrap] accumulating initial lidar map before first policy step duration_sec=%.2f min_fov_deg=%.1f max_wait_sec=%.2f",
                    max(0.0, float(config.occupancy_grid_lidar_bootstrap_sec)),
                    float(bootstrap_min_fov_deg),
                    float(config.occupancy_grid_lidar_bootstrap_max_wait_sec),
                )
                occupancy_bootstrap_logged = True
            fov_bucket = int(current_lidar_fov_deg // 10.0)
            if fov_bucket != occupancy_bootstrap_last_fov_bucket:
                logger.info(
                    "[OccupancyBootstrap] waiting time_ready=%s fov_ready=%s deadline=%s current_fov_deg=%.1f threshold_deg=%.1f",
                    int(bool(bootstrap_time_ready)),
                    int(bool(bootstrap_fov_ready)),
                    int(bool(bootstrap_deadline_reached)),
                    float(current_lidar_fov_deg),
                    float(bootstrap_min_fov_deg),
                )
                occupancy_bootstrap_last_fov_bucket = fov_bucket
            drive_controller.apply("STOP", speed_scale=0.0)
            last_action_name = "STOP"
            continue
        if occupancy_bootstrap_logged and not occupancy_bootstrap_done_logged:
            logger.info(
                "[OccupancyBootstrap] complete; starting VLN policy. fov_deg=%.1f deadline_reached=%s",
                float(current_lidar_fov_deg),
                int(bool(bootstrap_deadline_reached)),
            )
            occupancy_bootstrap_done_logged = True

        # --- Fast path during active primitive execution ---
        is_motion_active = _is_executing_motion_step(active_step_result)
        if is_motion_active:
            proposed_step_result = policy.step(observation, execution_feedback=last_policy_feedback)
            active_step_result = proposed_step_result
            step_result = active_step_result
            for completed_label in step_result.completed_landmarks:
                completed_key = str(completed_label).strip().lower()
                if completed_key and completed_key not in landmark_map_reached_synced:
                    landmark_map.mark_reached(completed_key, timestamp_sec=sim_time_sec)
                    landmark_map_reached_synced.add(completed_key)
            safety_decision = depth_safety_monitor.apply(
                observation=observation,
                action_name=step_result.action_name,
                speed_scale=step_result.speed_scale,
                local_goal_xy=step_result.local_goal_xy,
            )
            if safety_decision.action_name == "TRACK_TARGET":
                last_applied_wheel_velocity, last_applied_steering_angle = drive_controller.track_local_target(
                    local_forward_m=float(step_result.local_goal_xy[0]),
                    local_left_m=float(step_result.local_goal_xy[1]),
                    stop_distance_m=float(config.waypoint_stop_distance_m),
                    min_speed_scale=float(config.track_min_speed_scale),
                    max_speed_scale=float(config.track_max_speed_scale) * float(safety_decision.speed_scale),
                    heading_gain=float(config.track_heading_gain),
                    slow_heading_rad=float(config.track_slow_heading_rad),
                    smoothing_alpha=float(config.track_smoothing_alpha),
                )
            else:
                last_applied_wheel_velocity, last_applied_steering_angle = drive_controller.apply(
                    safety_decision.action_name,
                    speed_scale=safety_decision.speed_scale,
                )
            last_policy_feedback = ExecutionFeedback(
                applied_action=safety_decision.action_name,
                wheel_velocity=last_applied_wheel_velocity,
                steering_angle=last_applied_steering_angle,
            )
            policy.record_action(safety_decision.action_name, _action_result_text(step_result), sim_time_sec)
            stuck_detector.update(
                observation=observation,
                executed_action=safety_decision.action_name,
                speed_scale=safety_decision.speed_scale,
            )
            if step_result.route_complete and not route_complete_handled:
                route_complete_handled = True
                navigation_finished = True
                drive_controller.apply("STOP", speed_scale=0.0)
                target_grounder.close()
                debug_video_recorder.close()
            continue

        current_landmark = policy.current_landmark
        target_observation: TargetObservation | None = None
        landmark_map_entry = None
        if current_landmark:
            target_tracker.set_target_label(current_landmark)
            tracker_can_cover_motion = target_tracker.lifecycle in {"TRACKING", "LOST_TEMP"}
            perception_motion_active = (
                bool(config.target_grounding_pause_during_active_primitive)
                and _is_executing_motion_step(active_step_result)
                and bool(tracker_can_cover_motion)
            )
            allow_new_grounding = (
                not perception_motion_active
                and target_tracker.should_request_grounding(
                    target_label=current_landmark,
                    timestamp_sec=sim_time_sec,
                )
            )
            grounding_observation = target_grounder.observe(
                rgb=front_rgb,
                depth=last_depth_image,
                target_label=current_landmark,
                robot_pose=observation.pose,
                timestamp_sec=sim_time_sec,
                frame_index=frame_index,
                allow_new_request=allow_new_grounding,
            )
            if bool(grounding_observation.visible) and bool(getattr(grounding_observation, "can_confirm", False)):
                target_tracker.initialize_from_grounding(
                    observation=grounding_observation,
                    rgb=front_rgb,
                    timestamp_sec=sim_time_sec,
                )
            tracker_observation = target_tracker.update(
                rgb=front_rgb,
                depth=last_depth_image,
                robot_pose=observation.pose,
                timestamp_sec=sim_time_sec,
                frame_index=frame_index,
            )
            target_observation = _select_target_observation(
                grounding_observation=grounding_observation,
                tracker_observation=tracker_observation,
            )
            if perception_motion_active and frame_index % 30 == 0:
                logger.info(
                    "[PerceptionScheduler] target=%s grounding=paused_during_active_primitive tracker=%s selected=%s reason=%s",
                    current_landmark,
                    target_tracker.lifecycle,
                    target_observation.source if target_observation is not None else "none",
                    target_observation.reason if target_observation is not None else "none",
                )
            landmark_state = landmark_tracker.update(
                target_label=current_landmark,
                behavior_type=policy.current_behavior_type,
                observation=target_observation,
                robot_pose=observation.pose,
                timestamp_sec=sim_time_sec,
                arrival_distance_m=float(policy.current_arrival_distance_m),
            )
            landmark_map_entry = landmark_map.update_from_tracker(
                state=landmark_state,
                robot_pose=observation.pose,
                timestamp_sec=sim_time_sec,
            )
            policy.set_landmark_state(landmark_state)
            policy.set_landmark_map_entry(landmark_map_entry)
            policy.set_landmark_map_summary(
                landmark_map.summary(
                    target_label=current_landmark,
                    robot_pose=observation.pose,
                    timestamp_sec=sim_time_sec,
                )
            )
        else:
            target_tracker.reset()
            policy.set_landmark_state(None)
            policy.set_landmark_map_entry(None)
            policy.set_landmark_map_summary("")
        # Merge confirmed landmarks with current detection overlays
        all_entries = list(landmark_map.entries())
        detection_entries = _collect_detection_entries(
            landmark_state=landmark_state if current_landmark else None,
            timestamp_sec=sim_time_sec,
        )
        all_entries.extend(detection_entries)
        policy.set_all_landmark_map_entries(all_entries)

        topological_memory.update(
            robot_pose=observation.pose,
            timestamp_sec=sim_time_sec,
            current_landmark=current_landmark,
            landmark_entry=landmark_map_entry if current_landmark else None,
            progress_signal=progress_manager.state.progress_signal,
        )
        policy.set_topology_summary(
            topological_memory.summary(
                current_target=current_landmark,
                robot_pose=observation.pose,
            )
        )
        policy.set_topology_revisit_world_xy(
            topological_memory.suggest_revisit_world_xy(
                current_target=current_landmark,
                robot_pose=observation.pose,
            )
        )

        obstacle_summary = _extract_obstacle_summary(
            depth=last_depth_image,
            occupancy_grid=occupancy_grid if bool(config.occupancy_grid_enabled) else None,
            robot_pose=observation.pose,
            stop_distance_m=float(config.obstacle_stop_distance_m),
        )
        # Layout classification
        if bool(config.occupancy_grid_enabled):
            policy.set_layout_description(
                _classify_occupancy_layout(occupancy_grid=occupancy_grid, robot_pose=observation.pose)
            )

        semantic_map_summary = _build_semantic_map_summary(
            current_target=current_landmark or "",
            landmark_entries=list(landmark_map.entries()),
            robot_pose=observation.pose,
            obstacle_summary=obstacle_summary,
            completed_anchors=policy.landmark_anchors,
            direction_clause=policy.current_direction_clause,
            action_history=policy.action_history_text,
            layout_description=policy.current_layout_description,
        )
        policy.set_obstacle_summary(obstacle_summary)
        policy.set_semantic_map_summary(semantic_map_summary)
        should_refresh_policy = True
        recovery_command = stuck_detector.recovery_command(now_sec=sim_time_sec)
        if recovery_command.active:
            policy.cancel_active_goal()
            proposed_step_result = _runtime_recovery_step_result(
                recovery_command=recovery_command,
                previous_step_result=active_step_result,
            )
            proposed_command_token = active_command_token
        else:
            proposed_step_result = policy.step(observation, execution_feedback=last_policy_feedback)
            proposed_command_token = active_command_token + int(bool(proposed_step_result.replanned))
            confirmation_result = _confirm_policy_step(
                enabled=bool(config.interactive_confirmation_enabled),
                step_result=proposed_step_result,
                command_token=proposed_command_token,
                logger=logger,
            )
            if confirmation_result == "quit":
                logger.info("Interactive confirmation requested exit. Stopping navigation loop.")
                drive_controller.apply("STOP", speed_scale=0.0)
                break
            if confirmation_result == "hold":
                policy.cancel_active_goal()
                proposed_step_result = _user_hold_step_result(proposed_step_result)
        active_step_result = proposed_step_result
        active_command_token = proposed_command_token
        step_result = active_step_result
        for completed_label in step_result.completed_landmarks:
            completed_key = str(completed_label).strip().lower()
            if completed_key and completed_key not in landmark_map_reached_synced:
                landmark_map.mark_reached(completed_key, timestamp_sec=sim_time_sec)
                landmark_map_reached_synced.add(completed_key)
        policy.set_all_landmark_map_entries(landmark_map.entries())
        safety_decision = depth_safety_monitor.apply(
            observation=observation,
            action_name=step_result.action_name,
            speed_scale=step_result.speed_scale,
            local_goal_xy=step_result.local_goal_xy,
        )
        policy.record_safety_result(
            overridden=bool(safety_decision.overridden),
            desired_action=step_result.action_name,
            applied_action=safety_decision.action_name,
            reason=safety_decision.reason,
        )
        if safety_decision.overridden:
            policy.cancel_active_goal()
            safety_context = (
                f"SAFETY_STOP previous_desired={step_result.action_name}; "
                f"runtime_applied={safety_decision.action_name}; "
                f"reason={safety_decision.reason}; "
                f"depth_m front={float(safety_decision.front_depth_m):.2f}, "
                f"left={float(safety_decision.left_depth_m):.2f}, "
                f"right={float(safety_decision.right_depth_m):.2f}; "
                f"stop_threshold_m={float(config.obstacle_stop_distance_m):.2f}. "
                "The runtime overrode the unsafe action; choose a fresh safe candidate from the image."
            )
            policy.set_safety_context(safety_context)
            safety_signature = (str(step_result.action_name), str(safety_decision.reason))
            if safety_signature != last_safety_log_signature:
                logger.warning(
                    "!!! SAFETY_OVERRIDE !!! desired=%s applied=%s reason=%s depth_m(front=%.2f,left=%.2f,right=%.2f) threshold=%.2f next=LLM_REPLAN",
                    step_result.action_name,
                    safety_decision.action_name,
                    safety_decision.reason,
                    float(safety_decision.front_depth_m),
                    float(safety_decision.left_depth_m),
                    float(safety_decision.right_depth_m),
                    float(config.obstacle_stop_distance_m),
                )
                last_safety_log_signature = safety_signature
        else:
            last_safety_log_signature = None
        if safety_decision.action_name == "TRACK_TARGET":
            last_applied_wheel_velocity, last_applied_steering_angle = drive_controller.track_local_target(
                local_forward_m=float(step_result.local_goal_xy[0]),
                local_left_m=float(step_result.local_goal_xy[1]),
                stop_distance_m=float(config.waypoint_stop_distance_m),
                min_speed_scale=float(config.track_min_speed_scale),
                max_speed_scale=float(config.track_max_speed_scale) * float(safety_decision.speed_scale),
                heading_gain=float(config.track_heading_gain),
                slow_heading_rad=float(config.track_slow_heading_rad),
                smoothing_alpha=float(config.track_smoothing_alpha),
            )
        else:
            last_applied_wheel_velocity, last_applied_steering_angle = drive_controller.apply(
                safety_decision.action_name,
                speed_scale=safety_decision.speed_scale,
            )
        last_policy_feedback = ExecutionFeedback(
            applied_action=safety_decision.action_name,
            wheel_velocity=last_applied_wheel_velocity,
            steering_angle=last_applied_steering_angle,
        )
        policy.record_action(safety_decision.action_name, _action_result_text(step_result), sim_time_sec)
        stuck_event = stuck_detector.update(
            observation=observation,
            executed_action=safety_decision.action_name,
            speed_scale=safety_decision.speed_scale,
        )
        if stuck_event.triggered:
            policy.cancel_active_goal()
            policy.set_safety_context(
                "STUCK_DETECTED "
                f"reason={stuck_event.reason}; "
                f"displacement_m={stuck_event.displacement_m:.3f}; "
                f"yaw_delta_deg={stuck_event.yaw_delta_deg:.1f}. "
                "Runtime will back up and turn before requesting a fresh VLM decision."
            )

        policy_signature = (
            step_result.current_landmark,
            int(active_command_token),
            step_result.action_name,
            step_result.selected_candidate_id,
            step_result.decision_source,
            tuple(step_result.completed_landmarks),
            bool(step_result.route_complete),
        )
        if should_refresh_policy and policy_signature != last_logged_policy_signature:
            logger.info(
                "[Decision] landmark=%s completed=%s route_complete=%s action=%s speed_scale=%.2f source=%s token=%s candidate=%s remaining=%s progress=%s replanned=%s local_goal_xy=%s goal_xy=%s",
                step_result.current_landmark or "none",
                step_result.completed_landmarks,
                int(step_result.route_complete),
                step_result.action_name,
                float(step_result.speed_scale),
                step_result.decision_source,
                active_command_token,
                step_result.selected_candidate_id,
                step_result.execution_steps_remaining,
                step_result.progress_signal,
                int(step_result.replanned),
                step_result.local_goal_xy,
                step_result.world_goal_xy,
            )
            logger.info("[Reasoning] %s", step_result.reasoning_summary)
            logger.info(
                "[Execution] desired=%s executing=%s token=%s mode=%s",
                step_result.action_name,
                safety_decision.action_name,
                active_command_token,
                "discrete_primitive" if bool(config.use_discrete_action_primitives) else "continuous_waypoint",
            )
            if step_result.debug_image_path is not None:
                logger.info("[Debug] candidate_image=%s", step_result.debug_image_path)
            last_logged_policy_signature = policy_signature

        navigation_map_visualizer.update(
            observation=observation,
            step_result=step_result,
            executed_action=safety_decision.action_name,
            landmark_entries=landmark_map.entries(),
            timestamp_sec=sim_time_sec,
            occupancy_snapshot=occupancy_grid.snapshot() if bool(config.occupancy_grid_enabled) else None,
        )
        if bool(config.occupancy_grid_enabled):
            global_pose_trace.append((float(observation.pose.x), float(observation.pose.y)))
            if len(global_pose_trace) > 8000:
                del global_pose_trace[: len(global_pose_trace) - 8000]

        debug_update_period_sec = max(0.10, float(config.debug_image_update_period_sec))
        force_final_debug_frame = bool(step_result.route_complete and not route_complete_handled)
        should_save_target_debug = (
            bool(config.target_debug_enabled)
            and (force_final_debug_frame or (loop_now - last_nav_debug_save_time) >= debug_update_period_sec)
        )
        if should_save_target_debug:
            if bool(config.occupancy_grid_enabled):
                latest_global_map_image = _save_global_occupancy_debug(
                    occupancy_snapshot=occupancy_grid.snapshot(),
                    robot_pose=observation.pose,
                    trajectory_xy=global_pose_trace,
                    landmark_entries=landmark_map.entries(),
                    step_result=step_result,
                    output_path=None,
                )
            debug_image = _save_current_target_debug_image(
                rgb=front_rgb,
                step_result=step_result,
                target_observation=target_observation,
                executed_action=safety_decision.action_name,
                output_path=target_debug_output_path,
                map_image=latest_global_map_image,
                map_image_path=navigation_map_visualizer.output_path if bool(config.navigation_map_enabled) else None,
                horizontal_fov_deg=float(config.camera_horizontal_fov_deg),
            )
            debug_video_recorder.append(debug_image)
            last_nav_debug_save_time = loop_now

        if step_result.route_complete and not route_complete_handled:
            route_complete_handled = True
            navigation_finished = True
            drive_controller.apply("STOP", speed_scale=0.0)
            last_action_name = "STOP"
            target_grounder.close()
            debug_video_recorder.close()
            logger.info("Route complete — navigation stopped; Isaac Sim scene remains open. Debug video saved: %s", debug_video_recorder.output_path)
            continue

        if (
            config.depth_debug_enabled
            and last_depth_image is not None
            and depth_debug_save_pending
            and (loop_now - last_depth_debug_save_time) >= debug_update_period_sec
        ):
            depth_debug = _render_depth_debug(depth_image=last_depth_image)
            _save_rgb_debug_image(depth_debug, depth_debug_output_path)
            depth_debug_save_pending = False
            last_depth_debug_save_time = loop_now

        if loop_now - last_motion_log_time >= float(config.motion_log_period_sec):
            delta = position - last_logged_position
            joint_readback = drive_controller.readback()
            logger.info(
                "[Motion] policy_action=%s exec_action=%s token=%s mode=%s world_pos=[%.3f, %.3f, %.3f] odom=[%.3f, %.3f, %.1fdeg] delta=[%.3f, %.3f, %.3f] wheel_joint_target_rad_s=%.3f steering_joint_target_rad=%.3f steering_actual=%s wheel_actual=%s",
                step_result.action_name,
                safety_decision.action_name,
                active_command_token,
                "discrete_primitive" if bool(config.use_discrete_action_primitives) else "continuous_waypoint",
                float(position[0]),
                float(position[1]),
                float(position[2]),
                float(observation.pose.x),
                float(observation.pose.y),
                float(np.degrees(observation.pose.yaw_rad)),
                float(delta[0]),
                float(delta[1]),
                float(delta[2]),
                float(last_applied_wheel_velocity),
                float(last_applied_steering_angle),
                _format_joint_map(joint_readback["steering_position"]),
                _format_joint_map(joint_readback["wheel_velocity"]),
            )
            last_motion_log_time = loop_now
            last_logged_position = position.copy()

        last_action_name = safety_decision.action_name

    target_grounder.close()
    debug_video_recorder.close()


def _collect_detection_entries(
    *,
    landmark_state: LandmarkState | None,
    timestamp_sec: float,
) -> list[LandmarkMapEntry]:
    """Convert potential tracker observations into detection overlays for the map."""
    if landmark_state is None:
        return []
    state = landmark_state
    # Only show detections that have a potential world position and are not confirmed
    world_xy = state.potential_world_xy or state.candidate_world_xy
    if world_xy is None:
        return []
    if str(state.evidence_level) == "confirmed":
        return []  # confirmed entries already come from LandmarkMap
    obs = state.last_observation
    return [LandmarkMapEntry(
        label=str(state.label),
        behavior_type=str(state.behavior_type),
        world_xy=[float(world_xy[0]), float(world_xy[1])],
        confidence=float(state.potential_confidence),
        first_seen_time_sec=float(state.last_seen_time_sec or timestamp_sec),
        last_seen_time_sec=float(state.last_seen_time_sec or timestamp_sec),
        update_count=0,
        status="detection",
        source=str(state.potential_source) if state.potential_source != "unknown" else "tracker",
        depth_m=None if obs is None else obs.depth_m,
        bbox=state.potential_bbox,
        anchor_distance_m=float(state.anchor_distance_m) if state.anchor_distance_m is not None else None,
        robot_forward_m=None,
        robot_left_m=None,
        reached=False,
        confirmed_observation_count=0,
        evidence_level="detection",
    )]


def _load_environment(*, stage, add_reference_to_stage, config: AppConfig, logger) -> bool:
    usd_path = _resolve_environment_usd_path(config=config)
    if not usd_path:
        logger.info("No environment USD configured; using minimal local scene.")
        return False
    if not Path(usd_path).exists():
        raise FileNotFoundError(
            f"Configured environment USD does not exist: {usd_path}. "
            "Check scene.usd_path in configs/default.yaml."
        )
    add_reference_to_stage(usd_path, config.scene_prim_path)
    logger.info("Loaded environment USD: %s", usd_path)
    return True


def _confirm_policy_step(
    *,
    enabled: bool,
    step_result: PolicyStepResult,
    command_token: int,
    logger,
) -> str:
    if not enabled:
        return "execute"
    if not bool(step_result.replanned):
        return "execute"
    if str(step_result.action_name).strip().upper() == "STOP":
        return "execute"

    logger.info(
        "[Confirm] token=%s action=%s source=%s candidate=%s goal_xy=%s",
        command_token,
        step_result.action_name,
        step_result.decision_source,
        step_result.selected_candidate_id,
        step_result.world_goal_xy,
    )
    logger.info("[ConfirmReasoning] %s", step_result.reasoning_summary)

    prompt = (
        "\n"
        f"[Confirm] token={command_token} action={step_result.action_name} "
        f"source={step_result.decision_source} candidate={step_result.selected_candidate_id}\n"
        f"[Reasoning] {step_result.reasoning_summary}\n"
        "Execute this action? [y/N/q]: "
    )
    try:
        response = _read_confirmation_from_terminal(prompt)
    except EOFError:
        logger.warning("Interactive confirmation stdin unavailable. Holding action by default.")
        return "hold"

    if response == "y":
        return "execute"
    if response == "q":
        return "quit"
    logger.info("[Confirm] token=%s held by user input=%r", command_token, response)
    return "hold"


def _read_confirmation_from_terminal(prompt: str) -> str:
    try:
        with open("/dev/tty", "r+", encoding="utf-8") as tty:
            tty.write(prompt)
            tty.flush()
            return tty.readline().strip().lower()
    except OSError:
        return input(prompt).strip().lower()


def _user_hold_step_result(step_result: PolicyStepResult) -> PolicyStepResult:
    return PolicyStepResult(
        action_name="STOP",
        speed_scale=0.0,
        decision_source=f"{step_result.decision_source}|USER_HOLD",
        selected_candidate_id=step_result.selected_candidate_id,
        local_goal_xy=[0.0, 0.0],
        world_goal_xy=None,
        execution_steps_remaining=0,
        reasoning_summary=f"{step_result.reasoning_summary} | user_hold",
        progress_signal=step_result.progress_signal,
        replanned=step_result.replanned,
        debug_image_path=step_result.debug_image_path,
        current_landmark=step_result.current_landmark,
        completed_landmarks=list(step_result.completed_landmarks),
        route_complete=step_result.route_complete,
    )


def _runtime_recovery_step_result(*, recovery_command, previous_step_result) -> PolicyStepResult:
    return PolicyStepResult(
        action_name=str(recovery_command.action_name),
        speed_scale=float(recovery_command.speed_scale),
        decision_source="STUCK_RECOVERY",
        selected_candidate_id=None,
        local_goal_xy=[0.0, 0.0],
        world_goal_xy=None,
        execution_steps_remaining=0,
        reasoning_summary=str(recovery_command.reason),
        progress_signal="STUCK_RECOVERY",
        replanned=False,
        debug_image_path=None,
        current_landmark="" if previous_step_result is None else str(previous_step_result.current_landmark),
        completed_landmarks=[] if previous_step_result is None else list(previous_step_result.completed_landmarks),
        route_complete=False if previous_step_result is None else bool(previous_step_result.route_complete),
    )


def _wait_for_stage_assets(*, simulation_app, is_stage_loading, timeout_sec: float, logger, label: str) -> None:
    start_time = time.monotonic()
    last_log_time = start_time - 10.0
    while is_stage_loading():
        simulation_app.update()
        now = time.monotonic()
        if now - last_log_time >= 5.0:
            logger.info("Waiting for %s assets to finish loading...", label)
            last_log_time = now
        if now - start_time > timeout_sec:
            raise TimeoutError(f"Timed out while waiting for {label} assets to load.")


def _set_viewport_camera(*, config: AppConfig, logger) -> None:
    if config.headless:
        return
    try:
        from isaacsim.core.rendering_manager import ViewportManager
    except ImportError:
        try:
            from isaacsim.core.utils.viewports import set_camera_view
        except ImportError:
            logger.warning("Viewport camera helper is unavailable; keeping default GUI view.")
            return

        set_camera_view(
            eye=np.asarray(config.viewport_eye, dtype=np.float32),
            target=np.asarray(config.viewport_target, dtype=np.float32),
            camera_prim_path="/OmniverseKit_Persp",
        )
        logger.info("Set GUI viewport camera using fallback viewport utils.")
        return

    ViewportManager.set_camera_view(
        "/OmniverseKit_Persp",
        eye=np.asarray(config.viewport_eye, dtype=np.float32),
        target=np.asarray(config.viewport_target, dtype=np.float32),
    )
    logger.info("Set GUI viewport camera to eye=%s target=%s", config.viewport_eye, config.viewport_target)


def _bind_viewport_to_camera(*, camera_prim_path: str, logger, enabled: bool) -> bool:
    if not enabled:
        return False
    try:
        from omni.kit.viewport.utility import get_active_viewport
    except ImportError:
        logger.warning("Viewport API binding is unavailable; keeping perspective camera.")
        return False
    viewport_api = get_active_viewport()
    if viewport_api is None:
        logger.warning("Active viewport not found; keeping perspective camera.")
        return False
    try:
        viewport_api.camera_path = camera_prim_path
    except Exception as exc:
        logger.warning("Failed to bind active viewport to %s: %s", camera_prim_path, exc)
        return False
    logger.info("Bound active viewport to camera: %s", camera_prim_path)
    return True


def _resolve_output_path(path_str: str) -> Path:
    candidate = Path(path_str)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def _is_executing_motion_step(step_result) -> bool:
    if step_result is None:
        return False
    action = str(getattr(step_result, "action_name", "STOP")).strip().upper()
    if action not in {"FORWARD", "TURN_LEFT", "TURN_RIGHT", "TRACK_TARGET"}:
        return False
    remaining = int(getattr(step_result, "execution_steps_remaining", 0) or 0)
    return remaining > 0


def _select_target_observation(
    *,
    grounding_observation: TargetObservation,
    tracker_observation: TargetObservation,
) -> TargetObservation:
    if bool(grounding_observation.visible) and bool(getattr(grounding_observation, "can_confirm", False)):
        return grounding_observation
    if bool(tracker_observation.visible):
        return tracker_observation
    if _is_transient_grounding_observation(grounding_observation):
        return tracker_observation if bool(tracker_observation.visible) else grounding_observation
    return grounding_observation


def _is_transient_grounding_observation(observation: TargetObservation | None) -> bool:
    if observation is None:
        return False
    reason = str(observation.reason).strip().lower()
    source = str(observation.source).strip().lower()
    return (
        "reused" in source
        or reason
        in {
            "grounding_request_started",
            "grounding_request_in_flight",
            "grounding_call_period",
            "grounding_paused",
            "rgb_unavailable",
            "image_prepare_failed",
        }
    )


def _save_rgb_debug_image(image_rgb: np.ndarray, output_path: Path) -> None:
    image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8))
    _save_image_atomic(image, output_path)


def _save_current_target_debug_image(
    *,
    rgb: np.ndarray | None,
    step_result: PolicyStepResult,
    target_observation: TargetObservation | None,
    executed_action: str,
    output_path: Path,
    map_image: Image.Image | None,
    map_image_path: Path | None,
    horizontal_fov_deg: float,
) -> Image.Image | None:
    if rgb is None:
        return None
    image_rgb = np.asarray(rgb, dtype=np.uint8)
    if image_rgb.ndim != 3 or image_rgb.shape[-1] < 3:
        return None

    fp_image = Image.fromarray(image_rgb[..., :3].copy())
    _overlay_target_mask(image=fp_image, observation=target_observation)
    fp_draw = ImageDraw.Draw(fp_image)
    fp_width, fp_height = fp_image.size
    center_u = fp_width // 2
    origin = (center_u, int(fp_height * 0.88))
    fp_draw.line((center_u, 0, center_u, fp_height), fill=(255, 255, 255), width=1)
    _draw_grounded_target_overlay(draw=fp_draw, observation=target_observation, image_size=(fp_width, fp_height))

    local_forward = float(step_result.local_goal_xy[0]) if step_result.local_goal_xy else 0.0
    local_left = float(step_result.local_goal_xy[1]) if step_result.local_goal_xy else 0.0
    distance_m = float(np.hypot(local_forward, local_left))
    heading_rad = float(np.arctan2(local_left, max(local_forward, 1e-6)))
    half_fov_rad = max(1e-6, float(np.deg2rad(horizontal_fov_deg)) * 0.5)

    if step_result.world_goal_xy is not None and local_forward > 0.0:
        normalized = np.tan(heading_rad) / max(np.tan(half_fov_rad), 1e-6)
        # Robot local +Y is left, while image u increases to the right.
        target_u = int(round(center_u - normalized * center_u))
        target_u = max(18, min(fp_width - 18, target_u))
        target_v = int(fp_height * 0.68)
        color = (255, 220, 0) if abs(heading_rad) <= half_fov_rad else (255, 120, 0)
        fp_draw.line((origin[0], origin[1], target_u, target_v), fill=color, width=4)
        fp_draw.ellipse((target_u - 16, target_v - 16, target_u + 16, target_v + 16), outline=color, width=4)
        fp_draw.text((target_u + 18, max(0, target_v - 14)), "active subgoal", fill=color, font=_debug_font(15))
    else:
        color = (255, 80, 80)
        fp_draw.line((origin[0] - 35, origin[1], origin[0] + 35, origin[1]), fill=color, width=4)
        fp_draw.text((origin[0] + 42, origin[1] - 12), "no active target", fill=color, font=_debug_font(15))

    heading_deg = float(np.degrees(heading_rad))
    overlay_lines = [
        f"candidate={step_result.selected_candidate_id} source={step_result.decision_source}",
        f"landmark={step_result.current_landmark or 'none'} completed={step_result.completed_landmarks}",
        f"policy={step_result.action_name} executed={executed_action}",
        f"local_goal forward={local_forward:.2f}m left={local_left:.2f}m",
        f"distance={distance_m:.2f}m heading={heading_deg:+.1f}deg",
        "yellow marker=active subgoal; colored box=visual grounding/tracker",
    ]
    margin = 10
    overlay_font = _debug_font(15)
    line_height = 22
    box_height = margin * 2 + line_height * len(overlay_lines)
    fp_draw.rectangle((0, 0, fp_width, box_height), fill=(0, 0, 0))
    for index, text in enumerate(overlay_lines):
        fp_draw.text((margin, margin + index * line_height), text, fill=(255, 255, 255), font=overlay_font)

    occ_width = max(540, int(fp_width))
    occ_image = _load_occupancy_debug_panel(
        occupancy_image=map_image,
        occupancy_image_path=map_image_path,
        width=occ_width,
        height=fp_height,
    )
    canvas = Image.new("RGB", (occ_width + fp_width, fp_height), (0, 0, 0))
    canvas.paste(occ_image, (0, 0))
    canvas.paste(fp_image, (occ_width, 0))

    _save_image_atomic(canvas, output_path)
    return canvas


def _overlay_target_mask(*, image: Image.Image, observation: TargetObservation | None) -> None:
    if observation is None or observation.mask is None:
        return
    if not observation.is_current_frame:
        return
    mask = np.asarray(observation.mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[0] != image.height or mask.shape[1] != image.width:
        return
    mask_alpha = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 90, mode="L")
    green = Image.new("RGB", image.size, (40, 255, 110))
    blended = Image.blend(image, green, 0.35)
    image.paste(blended, (0, 0), mask_alpha)


def _draw_grounded_target_overlay(
    *,
    draw: ImageDraw.ImageDraw,
    observation: TargetObservation | None,
    image_size: tuple[int, int],
) -> None:
    width, height = [int(value) for value in image_size]
    label_font = _debug_font(15)
    if observation is None:
        draw.text((12, height - 30), "grounding: no current landmark", fill=(255, 210, 90), font=label_font)
        return
    visible = bool(observation.visible)
    reason = str(observation.reason)
    source = str(observation.source)
    segmentation_source = str(getattr(observation, "segmentation_source", "none"))
    depth_text = "none" if observation.depth_m is None else f"{float(observation.depth_m):.2f}m"
    if visible and observation.bbox is not None:
        is_current = bool(observation.is_current_frame)
        if is_current:
            xmin, ymin, xmax, ymax = [int(value) for value in observation.bbox]
            xmin = max(0, min(width - 1, xmin))
            ymin = max(0, min(height - 1, ymin))
            xmax = max(0, min(width, xmax))
            ymax = max(0, min(height, ymax))
            can_confirm = bool(getattr(observation, "can_confirm", False))
            color = (70, 255, 120) if can_confirm else (255, 190, 55)
            draw.rectangle((xmin, ymin, xmax, ymax), outline=color, width=4)
            if observation.center_uv is not None:
                center_u, center_v = [int(round(float(value))) for value in observation.center_uv]
                draw.line((center_u - 10, center_v, center_u + 10, center_v), fill=color, width=3)
                draw.line((center_u, center_v - 10, center_u, center_v + 10), fill=color, width=3)
            verify_text = (
                f"verify={int(bool(getattr(observation, 'vlm_verified', False)))}"
                f"/{float(getattr(observation, 'vlm_verification_confidence', 0.0)):.2f}"
            )
            label = (
                f"grounded {observation.label} depth={depth_text} conf={observation.confidence:.2f} "
                f"can_confirm={int(can_confirm)} {verify_text} seg={segmentation_source}"
            )
            draw.rectangle((xmin, max(0, ymin - 28), min(width, xmin + 620), ymin), fill=(0, 0, 0))
            draw.text((xmin + 4, max(0, ymin - 24)), label[:110], fill=color, font=label_font)
        else:
            stale_color = (255, 160, 60)
            age_text = f"{float(observation.result_age_sec):.1f}s" if observation.result_age_sec > 0.0 else "?"
            stale_text = (
                f"STALE grounding: {observation.label} age={age_text} "
                f"depth={depth_text} conf={observation.confidence:.2f} "
                f"frame={observation.request_frame_index}"
            )
            draw.rectangle((0, height - 44, width, height), fill=(0, 0, 0))
            draw.text((12, height - 32), stale_text[:120], fill=stale_color, font=label_font)
        return
    text = (
        f"grounding: lost/hidden label={observation.label} reason={reason} "
        f"source={source} seg={segmentation_source} depth={depth_text}"
    )
    draw.rectangle((0, height - 44, width, height), fill=(0, 0, 0))
    draw.text((12, height - 32), text[:120], fill=(255, 110, 90), font=label_font)


def _load_occupancy_debug_panel(
    *,
    occupancy_image: Image.Image | None = None,
    occupancy_image_path: Path | None,
    width: int,
    height: int,
) -> Image.Image:
    if occupancy_image is not None:
        try:
            source = occupancy_image.convert("RGB")
            source.thumbnail((int(width), int(height)), Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS)
            panel = Image.new("RGB", (int(width), int(height)), (14, 18, 22))
            offset = ((int(width) - source.width) // 2, (int(height) - source.height) // 2)
            panel.paste(source, offset)
        except Exception:
            panel = Image.new("RGB", (int(width), int(height)), (14, 18, 22))
    elif occupancy_image_path is not None and Path(occupancy_image_path).exists():
        try:
            source = Image.open(occupancy_image_path).convert("RGB")
            source.thumbnail((int(width), int(height)), Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS)
            panel = Image.new("RGB", (int(width), int(height)), (14, 18, 22))
            offset = ((int(width) - source.width) // 2, (int(height) - source.height) // 2)
            panel.paste(source, offset)
        except Exception:
            panel = Image.new("RGB", (int(width), int(height)), (14, 18, 22))
    else:
        panel = Image.new("RGB", (int(width), int(height)), (14, 18, 22))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, int(width), 34), fill=(0, 0, 0))
    draw.text((12, 8), "NORTH-UP GLOBAL OCCUPANCY MAP", fill=(245, 245, 245), font=_debug_font(16))
    return panel


def _debug_font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, int(size))
        except Exception:
            continue
    return ImageFont.load_default()


def _save_image_atomic(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=f"{output_path.name}.", suffix=".tmp", dir=str(output_path.parent))
    os.close(fd)
    try:
        image.save(temp_path, format="PNG")
        os.replace(temp_path, output_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _render_depth_debug(*, depth_image: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_image, dtype=np.float32)
    valid_mask = np.isfinite(depth)
    if not np.any(valid_mask):
        return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

    valid_depth = depth[valid_mask]
    near = float(np.percentile(valid_depth, 5.0))
    far = float(np.percentile(valid_depth, 95.0))
    if far - near < 1e-3:
        far = near + 1.0

    normalized = np.zeros_like(depth, dtype=np.float32)
    normalized[valid_mask] = np.clip((depth[valid_mask] - near) / (far - near), 0.0, 1.0)
    grayscale = np.zeros_like(depth, dtype=np.uint8)
    grayscale[valid_mask] = np.asarray(255.0 * (1.0 - normalized[valid_mask]), dtype=np.uint8)
    rgb = np.repeat(grayscale[..., None], 3, axis=2)
    rgb[~valid_mask] = np.asarray([0, 0, 0], dtype=np.uint8)
    return rgb


def _get_rgb_image(camera) -> np.ndarray | None:
    try:
        rgba = camera.get_rgba()
    except Exception:
        return None
    if rgba is None:
        return None
    rgb = _to_numpy_array(rgba)
    if np.size(rgb) == 0:
        return None
    if rgb.ndim == 4 and rgb.shape[0] == 1:
        rgb = rgb[0]
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        return None
    rgb = np.asarray(rgb[..., :3])
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0.0, 255.0)
        if float(np.max(rgb)) <= 1.5:
            rgb = rgb * 255.0
        rgb = rgb.astype(np.uint8)
    return rgb


def _attach_camera_annotators(*, camera, logger, enable_depth: bool) -> None:
    add_rgba = getattr(camera, "add_rgba_to_frame", None)
    if callable(add_rgba):
        try:
            add_rgba()
        except Exception as exc:
            logger.debug("Failed to attach RGBA annotator: %s", exc)
    if not enable_depth:
        return
    for method_name in ("add_distance_to_image_plane_to_frame", "add_depth_to_frame"):
        method = getattr(camera, method_name, None)
        if not callable(method):
            continue
        try:
            method()
            logger.info("Attached camera depth annotator using %s.", method_name)
            return
        except Exception as exc:
            logger.warning("Failed to attach depth annotator via %s: %s", method_name, exc)
    logger.warning("No compatible depth annotator attachment method was found on the camera object.")


def _get_depth_image(*, camera, logger) -> np.ndarray | None:
    try:
        depth = camera.get_depth()
    except Exception as exc:
        logger.debug("Depth frame unavailable: %s", exc)
        return None
    if depth is None:
        return None
    depth_np = _to_numpy_array(depth)
    if depth_np.ndim == 4 and depth_np.shape[0] == 1:
        depth_np = depth_np[0]
    if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
        depth_np = depth_np[..., 0]
    if depth_np.ndim != 2:
        return None
    depth_np = np.asarray(depth_np, dtype=np.float32)
    depth_np[~np.isfinite(depth_np)] = np.nan
    return depth_np


def _to_numpy_array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _relative_roll_pitch_deg(
    *,
    initial_quaternion_xyzw: np.ndarray,
    current_quaternion_xyzw: np.ndarray,
) -> tuple[float, float]:
    # Isaac Sim core APIs return quaternions in scalar-first wxyz order.
    relative_quaternion = _quaternion_multiply_wxyz(
        _quaternion_conjugate_wxyz(initial_quaternion_xyzw),
        current_quaternion_xyzw,
    )
    euler_xyz_deg = _quaternion_wxyz_to_euler_xyz_deg(relative_quaternion)
    return float(euler_xyz_deg[0]), float(euler_xyz_deg[1])


def _quaternion_conjugate_wxyz(quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(value) for value in quaternion_wxyz]
    return np.asarray([w, -x, -y, -z], dtype=np.float32)


def _quaternion_multiply_wxyz(lhs_wxyz: np.ndarray, rhs_wxyz: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = [float(value) for value in lhs_wxyz]
    rw, rx, ry, rz = [float(value) for value in rhs_wxyz]
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float32,
    )


def _quaternion_wxyz_to_euler_xyz_deg(quaternion_wxyz: np.ndarray) -> list[float]:
    w, x, y, z = [float(value) for value in quaternion_wxyz]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = np.sign(sinp) * (np.pi / 2.0)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return [float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))]


def _resolve_environment_usd_path(*, config: AppConfig) -> str:
    if config.scene_usd_path:
        candidate = Path(str(config.scene_usd_path))
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        return str(candidate)
    return ""


def _resolve_robot_relative_prim_path(*, robot_root_path: str, prim_path: str) -> str:
    candidate = str(prim_path).strip()
    if not candidate:
        return str(robot_root_path)
    if candidate.startswith("/"):
        return candidate
    robot_name = Path(str(robot_root_path).rstrip("/")).name
    if candidate == robot_name:
        return str(robot_root_path)
    if candidate.startswith(f"{robot_name}/"):
        candidate = candidate[len(robot_name) + 1 :]
    return f"{str(robot_root_path).rstrip('/')}/{candidate.lstrip('/')}"


def _resolve_existing_prim_path(*, stage, prim_path: str, kind: str) -> str:
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        return prim_path

    head, slash, tail = prim_path.rpartition("/")
    if slash and tail:
        fallback_names = []
        if tail == "camera":
            fallback_names.append("Camera")
        elif tail == "Camera":
            fallback_names.append("camera")

        for fallback_name in fallback_names:
            fallback_path = f"{head}/{fallback_name}"
            fallback_prim = stage.GetPrimAtPath(fallback_path)
            if fallback_prim and fallback_prim.IsValid():
                return fallback_path

    discovered = _discover_matching_prim_path(stage=stage, requested_path=prim_path)
    if discovered:
        return discovered

    root_hint = _stage_root_hint(prim_path)
    available = _available_camera_like_prim_paths(stage=stage, root_hint=root_hint)
    detail = "\n".join(f"  - {path}" for path in available[:40]) if available else "  none"
    raise RuntimeError(
        f"Configured {kind} prim does not exist: {prim_path}\n"
        f"Camera-like prims under {root_hint or '/'}:\n{detail}"
    )


def _discover_matching_prim_path(*, stage, requested_path: str) -> str:
    requested = str(requested_path).rstrip("/")
    requested_lower = requested.lower()
    root_hint = _stage_root_hint(requested)
    suffixes = _candidate_path_suffixes(requested)
    matches: list[tuple[int, str]] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if root_hint and not path.startswith(root_hint.rstrip("/") + "/") and path != root_hint:
            continue
        path_lower = path.lower()
        if path_lower == requested_lower:
            return path
        if any(path_lower.endswith(suffix) for suffix in suffixes):
            matches.append((0, path))
            continue
        if _is_camera_like_prim(prim) and path_lower.rsplit("/", 1)[-1] == requested_lower.rsplit("/", 1)[-1]:
            matches.append((1, path))
    if not matches:
        return ""
    matches.sort(key=lambda item: (item[0], len(item[1])))
    return matches[0][1]


def _candidate_path_suffixes(requested_path: str) -> list[str]:
    parts = [part for part in str(requested_path).strip("/").split("/") if part]
    suffixes: list[str] = []
    for count in range(min(4, len(parts)), 0, -1):
        suffix = "/" + "/".join(parts[-count:]).lower()
        if suffix not in suffixes:
            suffixes.append(suffix)
        tail = parts[-1]
        swapped_tail = "camera" if tail == "Camera" else "Camera" if tail == "camera" else ""
        if swapped_tail:
            swapped = "/" + "/".join([*parts[-count:-1], swapped_tail]).lower()
            if swapped not in suffixes:
                suffixes.append(swapped)
    return suffixes


def _available_camera_like_prim_paths(*, stage, root_hint: str) -> list[str]:
    paths: list[str] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if root_hint and not path.startswith(root_hint.rstrip("/") + "/") and path != root_hint:
            continue
        if _is_camera_like_prim(prim):
            paths.append(path)
    return paths


def _is_camera_like_prim(prim) -> bool:
    try:
        type_name = str(prim.GetTypeName())
        path = str(prim.GetPath())
    except Exception:
        return False
    lowered = path.lower()
    return type_name == "Camera" or "camera" in lowered


def _stage_root_hint(path: str) -> str:
    parts = [part for part in str(path).strip("/").split("/") if part]
    if len(parts) >= 2:
        return "/" + "/".join(parts[:2])
    if len(parts) == 1:
        return "/" + parts[0]
    return ""


def _find_articulation_root_path(*, stage, root_path: str) -> str:
    from pxr import UsdPhysics

    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim or not root_prim.IsValid():
        raise RuntimeError(f"Robot prim path is invalid: {root_path}")

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if not prim_path.startswith(root_path):
            continue
        if UsdPhysics.ArticulationRootAPI(prim):
            return prim_path
    return root_path


def _set_local_xform(*, stage, prim_path: str, translation: list[float], rotation_deg: list[float]) -> None:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Invalid prim path for transform setup: {prim_path}")

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    translate_op = xformable.AddTranslateOp()
    rotate_op = xformable.AddRotateXYZOp()
    translate_op.Set(tuple(float(value) for value in translation))
    rotate_op.Set(tuple(float(value) for value in rotation_deg))


def _build_hold_joint_targets(*, robot, available_joint_names: list[str], hold_joint_names: list[str]) -> dict[str, float]:
    joint_positions = _get_joint_positions(robot)
    return {
        name: float(joint_positions[available_joint_names.index(name)])
        for name in hold_joint_names
        if name in available_joint_names
    }


def _get_joint_names(robot) -> list[str]:
    try:
        names = robot.dof_names
    except Exception as exc:
        raise RuntimeError(f"Unable to query articulation DOF names: {exc}") from exc
    return [str(name) for name in names]


def _get_joint_positions(robot) -> np.ndarray:
    positions = robot.get_joint_positions()
    if positions is None:
        return np.zeros((0,), dtype=np.float32)
    positions = _to_numpy_array(positions)
    return np.asarray(positions, dtype=np.float32).reshape(-1)


def _get_world_pose(robot) -> tuple[np.ndarray, np.ndarray]:
    pose = robot.get_world_pose()
    if isinstance(pose, tuple) and len(pose) == 2:
        position, orientation = pose
    else:
        raise RuntimeError("Unexpected world pose return value from articulation.")
    return np.asarray(position, dtype=np.float32), np.asarray(orientation, dtype=np.float32)


def _save_global_occupancy_debug(
    *,
    occupancy_snapshot,
    robot_pose,
    trajectory_xy: list[tuple[float, float]],
    landmark_entries,
    step_result: PolicyStepResult | None,
    output_path: Path | None,
) -> Image.Image | None:
    if occupancy_snapshot is None:
        return None
    log_odds = np.asarray(getattr(occupancy_snapshot, "log_odds", None), dtype=np.float32)
    if log_odds.ndim != 2 or log_odds.size == 0:
        return None

    resolution = max(1.0e-6, float(getattr(occupancy_snapshot, "resolution_m", 0.08)))
    origin_x = float(getattr(occupancy_snapshot, "origin_x", 0.0))
    origin_y = float(getattr(occupancy_snapshot, "origin_y", 0.0))
    free_mask = np.asarray(getattr(occupancy_snapshot, "free_mask", log_odds <= -0.5), dtype=bool)
    occupied_mask = log_odds >= 1.2
    known_mask = np.abs(log_odds) >= 0.20

    world_points: list[tuple[float, float]] = [(float(robot_pose.x), float(robot_pose.y))]
    world_points.extend((float(x), float(y)) for x, y in trajectory_xy[-2000:])
    if step_result is not None and step_result.world_goal_xy is not None:
        world_points.append((float(step_result.world_goal_xy[0]), float(step_result.world_goal_xy[1])))
    for entry in landmark_entries or []:
        world_xy = getattr(entry, "world_xy", None)
        if world_xy:
            world_points.append((float(world_xy[0]), float(world_xy[1])))

    if np.any(known_mask):
        rows, cols = np.where(known_mask)
        world_points.append((origin_x + float(np.min(cols)) * resolution, origin_y + float(np.min(rows)) * resolution))
        world_points.append((origin_x + float(np.max(cols) + 1) * resolution, origin_y + float(np.max(rows) + 1) * resolution))

    xs = [item[0] for item in world_points]
    ys = [item[1] for item in world_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y, 12.0)
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    padding = max(1.5, 0.08 * span)
    span = span + 2.0 * padding
    min_x = cx - 0.5 * span
    max_x = cx + 0.5 * span
    min_y = cy - 0.5 * span
    max_y = cy + 0.5 * span

    width = 900
    height = 900
    header_h = 82
    map_px = min(width - 24, height - header_h - 18)
    map_x0 = (width - map_px) // 2
    map_y0 = header_h + (height - header_h - map_px) // 2
    scale = float(map_px) / max(span, 1.0e-6)

    canvas = np.full((height, width, 3), 180, dtype=np.uint8)
    canvas[:header_h, :, :] = 20
    yy, xx = np.indices((map_px, map_px), dtype=np.float32)
    world_x = min_x + (xx + 0.5) / max(scale, 1.0e-6)
    world_y = max_y - (yy + 0.5) / max(scale, 1.0e-6)
    cols = np.floor((world_x - origin_x) / resolution).astype(np.intp)
    rows = np.floor((world_y - origin_y) / resolution).astype(np.intp)
    inside = (rows >= 0) & (rows < log_odds.shape[0]) & (cols >= 0) & (cols < log_odds.shape[1])
    sub = np.full((map_px, map_px, 3), 180, dtype=np.uint8)
    sampled_free = np.zeros((map_px, map_px), dtype=bool)
    sampled_occupied = np.zeros((map_px, map_px), dtype=bool)
    sampled_free[inside] = free_mask[rows[inside], cols[inside]]
    sampled_occupied[inside] = occupied_mask[rows[inside], cols[inside]]
    sub[sampled_free] = 255
    sub[sampled_occupied] = 0
    canvas[map_y0 : map_y0 + map_px, map_x0 : map_x0 + map_px, :] = sub

    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    font_title = _debug_font(22)
    font_label = _debug_font(15)
    font_small = _debug_font(12)

    def world_to_px(x_m: float, y_m: float) -> tuple[int, int]:
        px = int(round(map_x0 + (float(x_m) - min_x) * scale))
        py = int(round(map_y0 + (max_y - float(y_m)) * scale))
        return px, py

    draw.text((12, 10), "NORTH-UP GLOBAL OCCUPANCY MAP", fill=(245, 245, 245), font=font_title)
    draw.text(
        (12, 42),
        "black=occupied  white=free  gray=unknown  cyan=trajectory  blue=robot  green=DONE  magenta=confirmed  amber=potential  yellow=active_subgoal",
        fill=(210, 220, 230),
        font=font_label,
    )

    for gx in np.arange(np.floor(min_x), np.ceil(max_x) + 0.01, 1.0):
        px0, py0 = world_to_px(float(gx), min_y)
        px1, py1 = world_to_px(float(gx), max_y)
        draw.line((px0, py0, px1, py1), fill=(150, 150, 150), width=1)
    for gy in np.arange(np.floor(min_y), np.ceil(max_y) + 0.01, 1.0):
        px0, py0 = world_to_px(min_x, float(gy))
        px1, py1 = world_to_px(max_x, float(gy))
        draw.line((px0, py0, px1, py1), fill=(150, 150, 150), width=1)
    draw.rectangle((map_x0, map_y0, map_x0 + map_px, map_y0 + map_px), outline=(35, 35, 35), width=2)

    if len(trajectory_xy) >= 2:
        points_px = [world_to_px(float(x), float(y)) for x, y in trajectory_xy[-2000:]]
        draw.line(points_px, fill=(0, 190, 220), width=3)

    if step_result is not None and step_result.world_goal_xy is not None:
        gx, gy = world_to_px(float(step_result.world_goal_xy[0]), float(step_result.world_goal_xy[1]))
        draw.ellipse((gx - 8, gy - 8, gx + 8, gy + 8), fill=(255, 220, 0), outline=(0, 0, 0), width=2)
        draw.text((gx + 10, gy - 9), "active_subgoal", fill=(255, 220, 0), font=font_label)

    for entry in landmark_entries or []:
        world_xy = getattr(entry, "world_xy", None)
        if not world_xy:
            continue
        lx, ly = world_to_px(float(world_xy[0]), float(world_xy[1]))
        reached = bool(getattr(entry, "reached", False))
        evidence = str(getattr(entry, "evidence_level", "")).strip().lower()
        confirmed = bool(getattr(entry, "confirmed", False)) or evidence == "confirmed"
        label = str(getattr(entry, "label", "landmark"))
        if reached:
            color = (40, 235, 95)
            draw.ellipse((lx - 11, ly - 11, lx + 11, ly + 11), fill=color, outline=(255, 255, 255), width=3)
            draw.line((lx - 6, ly, lx - 1, ly + 6), fill=(0, 0, 0), width=3)
            draw.line((lx - 1, ly + 6, lx + 8, ly - 7), fill=(0, 0, 0), width=3)
            draw.text((lx + 13, ly - 10), f"{label} DONE", fill=color, font=font_label)
        elif confirmed:
            color = (220, 60, 220)
            draw.ellipse((lx - 9, ly - 9, lx + 9, ly + 9), fill=color, outline=(255, 255, 255), width=2)
            draw.text((lx + 11, ly - 10), f"{label} confirmed", fill=color, font=font_label)
        else:
            color = (255, 175, 40)
            draw.ellipse((lx - 7, ly - 7, lx + 7, ly + 7), outline=color, width=3)
            draw.text((lx + 10, ly - 10), f"{label} potential", fill=color, font=font_label)

    yaw = float(robot_pose.yaw_rad)
    robot_x = float(robot_pose.x)
    robot_y = float(robot_pose.y)
    tip = world_to_px(robot_x + np.cos(yaw) * 0.40, robot_y + np.sin(yaw) * 0.40)
    rear_left = world_to_px(
        robot_x - np.cos(yaw) * 0.25 - np.sin(yaw) * 0.20,
        robot_y - np.sin(yaw) * 0.25 + np.cos(yaw) * 0.20,
    )
    rear_right = world_to_px(
        robot_x - np.cos(yaw) * 0.25 + np.sin(yaw) * 0.20,
        robot_y - np.sin(yaw) * 0.25 - np.cos(yaw) * 0.20,
    )
    draw.polygon([tip, rear_left, rear_right], fill=(40, 140, 255), outline=(255, 255, 255))
    rx, ry = world_to_px(robot_x, robot_y)
    draw.text((rx + 12, ry + 8), "robot", fill=(40, 140, 255), font=font_label)
    draw.text(
        (12, height - 22),
        f"view=[{min_x:.1f},{max_x:.1f}]x[{min_y:.1f},{max_y:.1f}]m  pose=({robot_x:.2f},{robot_y:.2f}) yaw={np.degrees(yaw):+.1f}deg",
        fill=(20, 20, 20),
        font=font_small,
    )

    if output_path is not None:
        _save_image_atomic(image, output_path)
    return image


def _save_lidar_raw_scan(
    *,
    beam_endpoints: np.ndarray,
    beam_hits: np.ndarray,
    current_beam_endpoints: np.ndarray | None = None,
    current_beam_hits: np.ndarray | None = None,
    output_path: Path,
    fov_deg: float,
) -> None:
    if beam_endpoints.size == 0:
        return
    size = 620
    margin = 40
    max_range = 8.0
    img = Image.new("RGB", (size, size), (14, 18, 22))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    scale = (size - 2 * margin) / (2.0 * max_range)

    def to_pixel(fwd: float, left: float) -> tuple[int, int]:
        px = int(cx + left * scale)
        py = int(cy - fwd * scale)
        return max(0, min(size - 1, px)), max(0, min(size - 1, py))

    # Grid circles
    for r_m in [2.0, 4.0, 6.0, 8.0]:
        r_px = int(r_m * scale)
        draw.ellipse((cx - r_px, cy - r_px, cx + r_px, cy + r_px), outline=(40, 45, 50), width=1)
    # Forward line
    draw.line((cx, cy, cx, margin), fill=(60, 65, 70), width=1)
    draw.text((cx + 4, margin), "FWD", fill=(80, 90, 100), font=_debug_font(12))

    # Draw merged binned endpoints first, then current-sector endpoints on top.
    hits = beam_hits.astype(bool) if beam_hits.size else np.zeros((beam_endpoints.shape[0],), dtype=bool)
    for i in range(beam_endpoints.shape[0]):
        fwd, left = float(beam_endpoints[i, 0]), float(beam_endpoints[i, 1])
        px, py = to_pixel(fwd, left)
        if i < len(hits) and hits[i]:
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(70, 210, 255))
        else:
            draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=(60, 60, 70))

    current_endpoints = np.asarray(current_beam_endpoints, dtype=np.float32) if current_beam_endpoints is not None else np.zeros((0, 2), dtype=np.float32)
    current_hits = np.asarray(current_beam_hits, dtype=bool).reshape(-1) if current_beam_hits is not None else np.zeros((0,), dtype=bool)
    if current_endpoints.ndim == 2 and current_endpoints.shape[1] >= 2:
        for i in range(current_endpoints.shape[0]):
            fwd, left = float(current_endpoints[i, 0]), float(current_endpoints[i, 1])
            px, py = to_pixel(fwd, left)
            if i < len(current_hits) and current_hits[i]:
                draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(40, 245, 100))
            else:
                draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(120, 120, 130))

    # Robot marker
    draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=(220, 220, 220), outline=(255, 80, 80), width=2)

    # Info text
    merged_hit_count = int(np.count_nonzero(hits))
    merged_total = int(beam_endpoints.shape[0])
    current_hit_count = int(np.count_nonzero(current_hits))
    current_total = int(current_endpoints.shape[0]) if current_endpoints.ndim == 2 else 0
    draw.rectangle((0, 0, size, 50), fill=(0, 0, 0))
    draw.text((8, 6), f"LIDAR DEBUG  merged_binned={merged_total}/{merged_hit_count} current={current_total}/{current_hit_count} fov={fov_deg:.0f}deg", fill=(230, 230, 230), font=_debug_font(13))
    draw.text((8, 27), "green=current hits  cyan=motion-compensated binned hits  gray=miss/max-range", fill=(190, 200, 210), font=_debug_font(12))

    _save_image_atomic(img, output_path)


def _format_joint_map(values: dict[str, float]) -> str:
    ordered_items = sorted(values.items(), key=lambda item: item[0])
    return ",".join(f"{name}={float(value):+.3f}" for name, value in ordered_items)


def _extract_obstacle_summary(
    *,
    depth: np.ndarray | None,
    occupancy_grid,
    robot_pose,
    stop_distance_m: float,
) -> str:
    """Extract obstacle distances in 5 directions for VLM text prompt."""
    rays = {"front": 0.0, "front_left": -25.0, "left": -70.0, "front_right": 25.0, "right": 70.0}
    parts: list[str] = []
    max_check_m = 8.0
    for name, angle_deg in rays.items():
        if depth is not None:
            h, w = depth.shape[:2]
            cx = w // 2
            angle_per_pixel = 90.0 / w  # approximate
            offset_px = int(angle_deg / angle_per_pixel)
            sx = max(0, min(w - 1, cx + offset_px))
            col = depth[max(0, h // 2):, sx]
            valid = col[np.isfinite(col) & (col > 0.1)]
            dist_m = float(np.percentile(valid, 10.0)) if valid.size > 0 else float("inf")
        else:
            dist_m = float("inf")
        display = f"{dist_m:.1f}m" if dist_m < max_check_m else "clear"
        if dist_m <= stop_distance_m:
            display += " BLOCKED"
        parts.append(f"{name}={display}")
    return ", ".join(parts)


def _extract_direction_words(text: str) -> set[str]:
    t = str(text).lower()
    words: set[str] = set()
    if "left" in t:
        words.add("left")
    if "right" in t:
        words.add("right")
    if "forward" in t or "straight" in t or "ahead" in t or "直" in t or "前" in t:
        words.add("forward")
    return words


def _action_result_text(step_result) -> str:
    action = str(getattr(step_result, "action_name", "STOP"))
    source = str(getattr(step_result, "decision_source", ""))
    if "COMPLETE" in source or "COMPLETE" in action:
        return "completed"
    if action in ("TURN_LEFT", "TURN_RIGHT"):
        return f"turn {action.split('_')[1].lower()}"
    if action == "FORWARD":
        return "forward"
    return action.lower()


def _classify_occupancy_layout(*, occupancy_grid, robot_pose) -> str:
    """Analyze occupancy grid free space around robot and classify layout."""
    try:
        snapshot = occupancy_grid.snapshot()
        if snapshot is None:
            return ""
        resolution = float(getattr(snapshot, "resolution_m", 0.08))
        robot_x, robot_y = float(robot_pose.x), float(robot_pose.y)
        robot_yaw = float(robot_pose.yaw_rad)
        check_dist_m = 3.0
        check_steps = int(check_dist_m / resolution)

        # Check free space ratio in 5 directions
        cos_yaw, sin_yaw = float(np.cos(robot_yaw)), float(np.sin(robot_yaw))
        rays_deg = [0, -45, 45, -90, 90]
        free_counts: list[tuple[str, float]] = []
        for angle_deg in rays_deg:
            angle = robot_yaw + float(np.deg2rad(angle_deg))
            dx, dy = float(np.cos(angle)), float(np.sin(angle))
            free = 0
            for step in range(1, check_steps):
                x = robot_x + dx * step * resolution
                y = robot_y + dy * step * resolution
                # Check if occupied at (x, y) and nearby cells
                clear = True
                for ox, oy in ((0, 0), (resolution, 0), (-resolution, 0), (0, resolution), (0, -resolution)):
                    col = int(np.floor((x + ox - float(getattr(snapshot, "origin_x", 0.0))) / resolution))
                    row = int(np.floor((y + oy - float(getattr(snapshot, "origin_y", 0.0))) / resolution))
                    if hasattr(snapshot, "inflated_occupied"):
                        grid = getattr(snapshot, "inflated_occupied", None)
                    else:
                        grid = getattr(snapshot, "log_odds", None)
                    if grid is not None:
                        if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]:
                            if grid[row, col] > 0.5:
                                clear = False
                                break
                if clear:
                    free += 1
                else:
                    break
            ratio = free / check_steps
            if angle_deg <= -45:
                name = "left"
            elif angle_deg >= 45:
                name = "right"
            else:
                name = "front"
            free_counts.append((name, ratio))

        front = max(r for n, r in free_counts if n == "front")
        left = max(r for n, r in free_counts if n == "left")
        right = max(r for n, r in free_counts if n == "right")

        if front > 0.6 and left < 0.3 and right < 0.3:
            return "layout: corridor ahead"
        if front < 0.2 and left > 0.4 and right > 0.4:
            return "layout: T-junction or crossroad"
        if front < 0.1 and left > 0.3 and right < 0.1:
            return "layout: doorway on left"
        if front < 0.1 and right > 0.3 and left < 0.1:
            return "layout: doorway on right"
        if front < 0.15:
            return "layout: near wall/dead end"
        if front > 0.6 and left > 0.4 and right > 0.4:
            return "layout: open room"
        return "layout: open"
    except Exception:
        return ""


def _build_semantic_map_summary(
    *,
    current_target: str,
    landmark_entries,
    robot_pose,
    obstacle_summary: str,
    completed_anchors: dict[str, list[float]] | None = None,
    direction_clause: str = "",
    action_history: str = "",
    layout_description: str = "",
) -> str:
    """Build a clean semantic map description for VLM."""
    lines = [f"Current Goal: {current_target}"]
    object_lines = []
    for entry in landmark_entries:
        label = str(entry.label).strip().lower()
        if not label or label in {"none", "null", "unknown", "n/a"}:
            continue
        is_goal = label == current_target.strip().lower()
        status = str(entry.evidence_level).strip()
        fwd_m = entry.robot_forward_m
        left_m = entry.robot_left_m
        if fwd_m is not None and left_m is not None:
            if abs(left_m) < 0.3:
                direction = "ahead"
            elif left_m > 0:
                direction = "left-front" if fwd_m > 1.0 else "left"
            else:
                direction = "right-front" if fwd_m > 1.0 else "right"
            dist_str = f"{entry.anchor_distance_m or float(np.hypot(fwd_m, left_m)):.1f}m"
        else:
            direction = "mapped"
            dist_str = "?"
        marker = ">> " if is_goal else ""
        confidence = f"conf={float(entry.confidence):.2f}" if entry.confidence else ""
        object_lines.append(
            f"  {marker}{label}: {direction} {dist_str} [{status}] {confidence}"
        )
    if object_lines:
        lines.append("Objects:")
        lines.extend(object_lines)
    else:
        lines.append("Objects: none mapped yet")
    lines.append(f"Obstacles: {obstacle_summary}")
    # Direction check
    if completed_anchors and direction_clause:
        direction_words = _extract_direction_words(direction_clause)
        prev_keys = list(completed_anchors.keys())
        if prev_keys and direction_words:
            prev_label = prev_keys[-1]
            prev_anchor = completed_anchors.get(prev_label)
            if prev_anchor is not None:
                robot_x, robot_y = float(robot_pose.x), float(robot_pose.y)
                dx = robot_x - prev_anchor[0]
                dy = robot_y - prev_anchor[1]
                dist = float(np.hypot(dx, dy))
                lateral = dy
                if abs(lateral) < 0.5:
                    actual_side = "ahead"
                elif lateral > 0:
                    actual_side = "left"
                else:
                    actual_side = "right"
                expected_str = ", ".join(sorted(direction_words))
                match = "OK" if actual_side in direction_words else "CHECK"
                lines.append(
                    f"Direction: should go {expected_str} of {prev_label} | "
                    f"you are {actual_side} of {prev_label} (dist={dist:.1f}m) | {match}"
                )
                for entry in landmark_entries:
                    if (entry.label == current_target
                            and entry.evidence_level in ("potential", "confirmed_candidate")
                            and entry.world_xy and prev_anchor):
                        ex = entry.world_xy[0] - prev_anchor[0]
                        ey = entry.world_xy[1] - prev_anchor[1]
                        if abs(ey) < 0.5:
                            entry_side = "ahead"
                        elif ey > 0:
                            entry_side = "left"
                        else:
                            entry_side = "right"
                        if entry_side in direction_words:
                            lines.append(f"  [CONSISTENT] potential {entry.label} at {entry_side} matches instruction -> prefer this direction")
                        else:
                            lines.append(f"  [IGNORE] potential {entry.label} at {entry_side} contradicts instruction (expected {expected_str}) -> likely false positive")
                        break
    if layout_description:
        lines.append(layout_description)
    if action_history:
        lines.append(action_history)
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
