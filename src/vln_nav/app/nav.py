from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..control.locomotion import create_locomotion_controller
from ..navigation.candidate_waypoint_predictor import CandidateWaypointPredictor, CandidateWaypointPredictorConfig
from ..navigation.depth_safety import DepthSafetyConfig, DepthSafetyMonitor
from ..navigation.instruction_manager import InstructionContext, InstructionManager
from ..navigation.landmark_map import LandmarkMap, LandmarkMapConfig, LandmarkMapEntry
from ..navigation.landmark_tracker import LandmarkState, LandmarkTracker, LandmarkTrackerConfig
from ..navigation.map_candidate_generator import MapCandidateGenerator, MapCandidateGeneratorConfig
from ..navigation.occupancy_grid import GlobalOccupancyMap, OccupancyGridConfig
from ..navigation.observation_adapter import ObservationAdapter, Pose2D
from ..navigation.odometry import SimOdometrySensor
from ..navigation.primitive_collision_checker import (
    PrimitiveCollisionChecker,
    PrimitiveCollisionCheckerConfig,
)
from ..navigation.progress_manager import ProgressManager
from ..navigation.rgbd_geometry_map import (
    RGBDGeometryMapConfig,
    RGBDVirtualRayProjector,
)
from ..navigation.spatial_transform import (
    log_spatial_transform_self_check,
    transform_world_xy_to_odom,
)
from ..navigation.stuck_detector import StuckDetector, StuckDetectorConfig
from ..navigation.target_grounding import TargetGrounder, TargetGroundingConfig, TargetObservation
from ..navigation.topological_memory import TopologicalMemory, TopologicalMemoryConfig
from ..navigation.vlm_client import VLMClient
from ..navigation.vln_policy import ExecutionFeedback, PolicyStepResult, VLNPolicy, VLNPolicyConfig
from ..navigation.vln_reasoner import VLNReasoner
from ..runtime.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, AppConfig, load_config, setup_logging
from ..runtime.debug_video import DebugVideoRecorder
from ..runtime.scene_lighting import add_indoor_lighting

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAPTAINav Isaac Sim VLN runtime")
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
        config.simulation.headless = True

    logger = setup_logging(config.logging.level)
    if args.list_tasks:
        _print_available_tasks(tasks_file=Path(args.tasks_file), logger=logger)
        return 0
    applied_task_id = _apply_task_overrides(
        config=config,
        task_id=str(args.task).strip(),
        tasks_file=Path(args.tasks_file),
        logger=logger,
    )
    evaluation_result_path = _resolve_evaluation_result_path(task_id=applied_task_id)
    logger.info("Loaded config from %s", Path(args.config).resolve())
    if applied_task_id:
        logger.info("Loaded task from %s: %s", Path(args.tasks_file).resolve(), applied_task_id)
    logger.info("Robot USD path: %s", config.robot.asset.usd_path)
    logger.info(
        "LLM model=%s api_base=%s key_present=%s",
        config.llm.model_name,
        config.llm.api_base_url,
        bool(config.llm.api_key),
    )
    if not config.llm.api_key:
        logger.error(
            "LLM API key is missing. Set SILICONFLOW_API_KEY or LLM_API_KEY, "
            "or put api_key under the llm section in configs/default.yaml. "
            "This project has no local navigation fallback, so the robot will not move without the VLM."
        )
        return 2

    try:
        instruction_context = _parse_instruction_preflight(config=config, logger=logger)
    except Exception as exc:
        _write_evaluation_result(
            output_path=evaluation_result_path,
            status="FAILED",
            task_id=applied_task_id,
            config=config,
            instruction_context=None,
            step_result=None,
            sim_time_sec=0.0,
            failure_stage="instruction_preflight",
            reason=str(exc),
        )
        logger.exception("[ERROR] Instruction preflight failed before Isaac Sim startup.")
        return 1
    _write_evaluation_result(
        output_path=evaluation_result_path,
        status="RUNNING",
        task_id=applied_task_id,
        config=config,
        instruction_context=instruction_context,
        step_result=None,
        sim_time_sec=0.0,
        failure_stage="",
        reason="instruction plan validated",
    )

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    simulation_settings = _build_simulation_app_settings(config=config, logger=logger)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(simulation_settings)

    try:
        run_navigation(
            simulation_app=simulation_app,
            config=config,
            instruction_context=instruction_context,
            task_id=applied_task_id,
            evaluation_result_path=evaluation_result_path,
        )
    except Exception as exc:
        _write_evaluation_result(
            output_path=evaluation_result_path,
            status="FAILED",
            task_id=applied_task_id,
            config=config,
            instruction_context=instruction_context,
            step_result=None,
            sim_time_sec=0.0,
            failure_stage="runtime_exception",
            reason=str(exc),
        )
        logger.exception("Navigation loop crashed.")
        return 1
    finally:
        simulation_app.close()

    return 0


def _parse_instruction_preflight(*, config: AppConfig, logger) -> InstructionContext:
    parse_client = VLMClient(
        api_key=config.llm.api_key,
        model_name=config.llm.model_name,
        base_url=config.llm.api_base_url,
        timeout_sec=max(1, int(config.instruction.parse_timeout_sec)),
        retry_cooldown_sec=0.0,
        upload_max_edge_px=0,
        image_detail=config.llm.camera_image_detail,
        logger=logger,
    )
    instruction_manager = InstructionManager(
        llm_client=parse_client,
        logger=logger,
        max_tokens=int(config.instruction.parse_max_tokens),
        cache_enabled=bool(config.instruction.cache_enabled),
        cache_path=str(config.instruction.cache_path),
    )
    return instruction_manager.parse(config.instruction.text)


def _apply_task_overrides(*, config: AppConfig, task_id: str, tasks_file: Path, logger) -> str:
    tasks_path = tasks_file if tasks_file.is_absolute() else PROJECT_ROOT / tasks_file
    if not tasks_path.exists():
        if task_id:
            raise FileNotFoundError(f"Task file does not exist: {tasks_path}")
        if not str(config.instruction.text).strip():
            raise RuntimeError(
                "No task selected and config has no instruction.text. "
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
    elif not str(config.instruction.text).strip() and tasks:
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
    config.instruction.text = instruction

    scene_usd_path = str(selected.get("scene_usd_path", "")).strip()
    if scene_usd_path:
        config.scene.usd_path = scene_usd_path

    start_position = selected.get("robot_start_position", None)
    if isinstance(start_position, list) and len(start_position) >= 3:
        config.scene.robot_start_xy_m = (
            float(start_position[0]),
            float(start_position[1]),
        )

    start_orientation = selected.get("robot_start_orientation_deg", None)
    if isinstance(start_orientation, list) and len(start_orientation) >= 3:
        config.scene.robot_start_orientation_deg = (
            float(start_orientation[0]),
            float(start_orientation[1]),
            float(start_orientation[2]),
        )

    logger.info(
        "[Task] selected=%s scene=%s start=%s rpy_deg=%s instruction=%s",
        selected_id,
        config.scene.usd_path,
        config.robot_start_position,
        config.scene.robot_start_orientation_deg,
        config.instruction.text,
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
    logger.info("[PLAN] available_tasks file=%s count=%s", tasks_path, len(tasks))
    for task in tasks:
        logger.info(
            "[PLAN] task=%s scene=%s start=%s instruction=%s",
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
        "headless": bool(config.simulation.headless),
        "width": int(config.simulation.width),
        "height": int(config.simulation.height),
        "multi_gpu": bool(config.simulation.multi_gpu),
    }
    if active_gpu is not None:
        settings["active_gpu"] = int(active_gpu)
    if config.simulation.physics_gpu is not None:
        settings["physics_gpu"] = int(config.simulation.physics_gpu)
    logger.info(
        "[SimulationApp] headless=%s active_gpu=%s physics_gpu=%s multi_gpu=%s DISPLAY=%s WAYLAND_DISPLAY=%s CUDA_VISIBLE_DEVICES=%s",
        int(bool(config.simulation.headless)),
        "auto" if active_gpu is None else int(active_gpu),
        "auto" if config.simulation.physics_gpu is None else int(config.simulation.physics_gpu),
        int(bool(config.simulation.multi_gpu)),
        os.environ.get("DISPLAY", "<unset>"),
        os.environ.get("WAYLAND_DISPLAY", "<unset>"),
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    )
    return settings


def _resolve_active_gpu_for_display(*, config: AppConfig, logger) -> int | None:
    configured_gpu = config.simulation.active_gpu
    if bool(config.simulation.headless):
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
        if state in {"enabled", "active", "yes", "on", "1", "True"}:
            try:
                return int(parts[0])
            except ValueError:
                continue
    return None


def run_navigation(
    *,
    simulation_app,
    config: AppConfig,
    instruction_context: InstructionContext,
    task_id: str = "",
    evaluation_result_path: Path | None = None,
) -> None:
    logger = setup_logging(config.logging.level)

    llm_client = VLMClient(
        api_key=config.llm.api_key,
        model_name=config.llm.model_name,
        base_url=config.llm.api_base_url,
        timeout_sec=config.llm.timeout_sec,
        retry_cooldown_sec=config.llm.retry_cooldown_sec,
        upload_max_edge_px=config.llm.upload_max_edge_px,
        image_detail=config.llm.camera_image_detail,
        logger=logger,
    )
    target_verification_llm_client = VLMClient(
        api_key=config.llm.api_key,
        model_name=config.llm.model_name,
        base_url=config.llm.api_base_url,
        timeout_sec=config.llm.timeout_sec,
        retry_cooldown_sec=config.llm.retry_cooldown_sec,
        upload_max_edge_px=config.perception.grounding.upload_max_edge_px,
        image_detail=config.llm.camera_image_detail,
        logger=logger,
    )
    logger.info("[Goal] instruction=%s", instruction_context.raw_text)
    logger.info(
        "[Goal] parse_source=%s parse_reason=%s task_type=%s route_steps=%s",
        instruction_context.parsing_source,
        instruction_context.parsing_reason,
        instruction_context.task_type,
        [
            {
                "step_id": step.step_id,
                "step_type": step.step_type,
                "target": step.target,
                "from_region": step.from_region,
                "to_region": step.to_region,
                "direction": step.direction,
                "completion": step.completion,
                "cues": [
                    {
                        "label": cue.label,
                        "role": cue.role,
                        "side": cue.expected_side,
                    }
                    for cue in step.cues
                ],
                "relations": [
                    {
                        "target": relation.target,
                        "relation": relation.relation,
                        "reference": relation.reference,
                    }
                    for relation in step.relations
                ],
            }
            for step in instruction_context.route_steps
        ],
    )
    logger.info(
        "[Route] context_region=%s context_objects=%s navigation_targets=%s arrival_distance_m=%.2f success_distance_m=%.2f tracker_min_visible=%s",
        "unknown" if instruction_context.scene_context.current_region is None else instruction_context.scene_context.current_region.canonical_label,
        [item.canonical_label for item in instruction_context.scene_context.context_objects],
        instruction_context.navigation_target_labels,
        float(config.navigation.progress.landmark_clearance_m),
        float(config.navigation.progress.final_success_clearance_m),
        int(config.perception.landmark_tracking.min_visible_count),
    )
    logger.info(
        "[TargetPerceptionConfig] backend=grounding_dino call_period_sec=%.2f reuse_sec=%.2f pause_active=1 min_conf=%.2f depth_range=[%.2f, %.2f] depth_region=%s min_depth_valid_ratio=%.2f semantic_verify=1 verify_min=%.2f checkpoint=%s",
        float(config.perception.grounding.call_period_sec),
        float(config.perception.grounding.reuse_last_observation_sec),
        float(config.perception.grounding.min_confidence),
        float(config.perception.grounding.min_depth_m),
        float(config.perception.grounding.max_depth_m),
        str(config.perception.grounding.depth_region),
        float(config.perception.grounding.min_depth_valid_ratio),
        float(config.perception.grounding.semantic_verification.min_confidence),
        str(config.perception.grounding.detector.checkpoint_path),
    )
    logger.info(
        "[LandmarkTrackerConfig] arrival_distance=%.2f success_distance=%.2f min_visible=%s confirmations=%s landmark_map=%s map_ttl=%.1f",
        float(config.navigation.progress.landmark_clearance_m),
        float(config.navigation.progress.final_success_clearance_m),
        int(config.perception.landmark_tracking.min_visible_count),
        int(config.perception.landmark_tracking.arrival_confirmations),
        1,
        float(config.perception.landmark_map.anchor_ttl_sec),
    )

    observation_adapter = ObservationAdapter(
        camera_width=int(config.robot.sensors.camera.width),
        camera_height=int(config.robot.sensors.camera.height),
        horizontal_fov_deg=float(config.robot.sensors.camera.horizontal_fov_deg),
    )
    log_spatial_transform_self_check(
        logger=logger,
        image_width=int(config.robot.sensors.camera.width),
        image_height=int(config.robot.sensors.camera.height),
        horizontal_fov_deg=float(config.robot.sensors.camera.horizontal_fov_deg),
    )
    candidate_predictor = CandidateWaypointPredictor(
        CandidateWaypointPredictorConfig(
            image_width=int(config.robot.sensors.camera.width),
            image_height=int(config.robot.sensors.camera.height),
            horizontal_fov_deg=float(config.robot.sensors.camera.horizontal_fov_deg),
            roi_vertical_range=(float(config.robot.sensors.camera.depth_roi_vertical_range[0]), float(config.robot.sensors.camera.depth_roi_vertical_range[1])),
            valid_depth_range=(float(config.robot.sensors.camera.valid_depth_range_m[0]), float(config.robot.sensors.camera.valid_depth_range_m[1])),
            clearance_threshold_m=float(config.navigation.candidates.max_distance_m),
            slot_bearings_deg=tuple(float(value) for value in config.navigation.candidates.slot_bearings_deg),
            min_travel_distance_m=float(config.navigation.candidates.min_distance_m),
            preferred_distance_m=float(
                0.5
                * (
                    config.navigation.candidates.min_distance_m
                    + config.navigation.candidates.max_distance_m
                )
            ),
            max_travel_distance_m=float(config.navigation.candidates.max_distance_m),
            min_free_ratio=float(config.navigation.candidates.min_free_ratio),
            column_half_width_px=int(config.navigation.candidates.column_half_width_px),
            adaptive_count=int(config.navigation.candidates.adaptive_count),
            adaptive_min_spacing_deg=float(config.navigation.candidates.adaptive_min_spacing_deg),
            adaptive_smoothing_px=int(config.navigation.candidates.adaptive_smoothing_px),
        ),
        logger=logger,
    )
    depth_safety_monitor = DepthSafetyMonitor(
        config=DepthSafetyConfig(
            enabled=bool(config.navigation.safety.obstacle_avoidance_enabled),
            stop_distance_m=float(config.robot.geometry.obstacle_stop_distance_m),
            roi_vertical_range=(
                float(config.robot.sensors.camera.obstacle_roi_vertical_range[0]),
                float(config.robot.sensors.camera.obstacle_roi_vertical_range[1]),
            ),
            center_width_ratio=float(
                config.robot.sensors.camera.obstacle_center_width_ratio
            ),
            side_width_ratio=float(
                config.robot.sensors.camera.obstacle_side_width_ratio
            ),
            near_field_memory_enabled=bool(
                config.navigation.safety.near_field_memory_enabled
            ),
            near_field_memory_sec=float(
                config.navigation.safety.near_field_memory_sec
            ),
        )
    )
    stuck_detector = StuckDetector(
        config=StuckDetectorConfig(
            window_sec=float(config.navigation.recovery.stuck_window_sec),
            displacement_threshold_m=float(config.navigation.recovery.stuck_displacement_threshold_m),
            yaw_threshold_deg=float(config.navigation.recovery.stuck_yaw_threshold_deg),
            recovery_pause_sec=float(config.navigation.recovery.pause_sec),
            recovery_reverse_sec=float(config.navigation.recovery.reverse_sec),
            recovery_turn_sec=float(config.navigation.recovery.turn_sec),
        ),
        logger=logger,
    )
    progress_manager = ProgressManager(
        history_length=int(config.navigation.progress.history_length),
        stall_distance_m=float(config.navigation.progress.stall_distance_m),
        spin_yaw_deg=float(config.navigation.progress.spin_yaw_deg),
    )
    target_perception = TargetGrounder(
        config=TargetGroundingConfig(
            call_period_sec=float(config.perception.grounding.call_period_sec),
            reuse_last_observation_sec=float(config.perception.grounding.reuse_last_observation_sec),
            upload_max_edge_px=int(config.perception.grounding.upload_max_edge_px),
            min_bbox_area_ratio=float(config.perception.grounding.min_bbox_area_ratio),
            max_bbox_area_ratio=float(config.perception.grounding.max_bbox_area_ratio),
            min_bbox_width_px=int(config.perception.grounding.min_bbox_width_px),
            min_bbox_height_px=int(config.perception.grounding.min_bbox_height_px),
            max_bbox_aspect_ratio=float(config.perception.grounding.max_bbox_aspect_ratio),
            min_depth_valid_ratio=float(config.perception.grounding.min_depth_valid_ratio),
            depth_statistic=str(config.perception.grounding.depth_statistic),
            center_patch_ratio=float(config.perception.grounding.center_patch_ratio),
            min_depth_m=float(config.perception.grounding.min_depth_m),
            max_depth_m=float(config.perception.grounding.max_depth_m),
            min_confidence=float(config.perception.grounding.min_confidence),
            horizontal_fov_deg=float(config.robot.sensors.camera.horizontal_fov_deg),
            coordinate_mode=str(config.perception.grounding.coordinate_mode),
            allow_normalized_bbox=bool(config.perception.grounding.allow_normalized_bbox),
            normalized_bbox_scale=float(config.perception.grounding.normalized_bbox_scale),
            depth_region=str(config.perception.grounding.depth_region),
            depth_cluster_tolerance_ratio=float(
                config.perception.grounding.depth_cluster_tolerance_ratio
            ),
            depth_cluster_min_pixels=int(
                config.perception.grounding.depth_cluster_min_pixels
            ),
            grounding_dino_repo_path=str(config.perception.grounding.detector.repo_path),
            grounding_dino_config_path=str(config.perception.grounding.detector.config_path),
            grounding_dino_checkpoint_path=str(config.perception.grounding.detector.checkpoint_path),
            grounding_dino_device=str(config.perception.grounding.detector.device),
            grounding_dino_box_threshold=float(config.perception.grounding.detector.box_threshold),
            grounding_dino_text_threshold=float(config.perception.grounding.detector.text_threshold),
            grounding_dino_text_encoder_path=str(config.perception.grounding.detector.text_encoder_path),
            grounding_dino_max_detections=int(config.perception.grounding.detector.max_detections),
            grounding_dino_nms_iou_threshold=float(config.perception.grounding.detector.nms_iou_threshold),
            max_result_age_sec=float(config.perception.grounding.max_result_age_sec),
            max_result_pose_delta_m=float(config.perception.grounding.max_result_pose_delta_m),
            max_result_yaw_delta_deg=float(config.perception.grounding.max_result_yaw_delta_deg),
            semantic_verification_min_confidence=float(config.perception.grounding.semantic_verification.min_confidence),
            semantic_verification_max_edge_px=int(config.perception.grounding.semantic_verification.max_edge_px),
            semantic_verification_auto_accept_confidence=float(config.perception.grounding.semantic_verification.auto_accept_confidence),
        ),
        llm_client=target_verification_llm_client,
        logger=logger,
    )
    target_perception.require_ready()
    landmark_tracker = LandmarkTracker(
        config=LandmarkTrackerConfig(
            arrival_visible_depth_m=float(config.navigation.progress.landmark_clearance_m),
            arrival_anchor_radius_m=float(config.navigation.progress.landmark_clearance_m),
            min_visible_count=int(config.perception.landmark_tracking.min_visible_count),
            arrival_confirmations=int(config.perception.landmark_tracking.arrival_confirmations),
            arrival_window_size=int(config.perception.landmark_tracking.arrival_window_size),
            arrival_recent_visible_sec=float(config.perception.landmark_tracking.arrival_recent_visible_sec),
            max_lost_count=int(config.perception.landmark_tracking.max_lost_count),
            anchor_update_min_consistent_observations=int(
                config.perception.landmark_tracking.anchor_update_min_consistent_observations
            ),
            anchor_update_max_jump_m=float(config.perception.landmark_tracking.anchor_update_max_jump_m),
            confirm_confidence_threshold=float(config.perception.landmark_tracking.confirm_confidence_threshold),
            confirm_min_observations=int(config.perception.landmark_tracking.confirm_min_observations),
            confirm_max_anchor_jump_m=float(config.perception.landmark_tracking.confirm_max_anchor_jump_m),
            confirm_max_depth_jump_m=float(config.perception.landmark_tracking.confirm_max_depth_jump_m),
            potential_confidence_threshold=float(config.perception.landmark_tracking.potential_confidence_threshold),
            low_confidence_min_translation_m=float(
                config.perception.landmark_tracking.low_confidence_min_translation_m
            ),
            low_confidence_min_yaw_deg=float(
                config.perception.landmark_tracking.low_confidence_min_yaw_deg
            ),
        ),
        logger=logger,
    )
    landmark_map = LandmarkMap(
        config=LandmarkMapConfig(
            anchor_ttl_sec=float(config.perception.landmark_map.anchor_ttl_sec),
            min_confidence=float(config.perception.landmark_map.min_confidence),
            log_period_sec=float(config.perception.landmark_map.log_period_sec),
        ),
        logger=logger,
    )
    topological_memory = TopologicalMemory(
        config=TopologicalMemoryConfig(
            node_spacing_m=float(config.mapping.topology.node_spacing_m),
            turn_threshold_deg=float(config.mapping.topology.turn_threshold_deg),
            merge_radius_m=float(config.mapping.topology.merge_radius_m),
            landmark_bind_radius_m=float(config.mapping.topology.landmark_bind_radius_m),
            max_nodes=int(config.mapping.topology.max_nodes),
            recent_node_count=int(config.mapping.topology.recent_node_count),
        ),
        logger=logger,
    )
    odom_filter_alpha = 1.0
    if config.robot.g1_controller is not None:
        odom_filter_alpha = float(config.robot.g1_controller.odom_filter_alpha)
    odom_sensor = SimOdometrySensor(
        logger=logger,
        source="isaac_world_pose",
        pose_filter_alpha=odom_filter_alpha,
    )
    occupancy_grid = GlobalOccupancyMap(
        config=OccupancyGridConfig(
            resolution_m=float(config.mapping.occupancy.resolution_m),
            size_m=float(config.mapping.occupancy.size_m),
            update_period_sec=float(config.mapping.rgbd.update_period_sec),
            min_range_m=float(config.mapping.rgbd.min_depth_m),
            max_range_m=float(config.mapping.rgbd.max_depth_m),
            occupied_log_odds=float(config.mapping.occupancy.occupied_log_odds),
            free_log_odds=float(config.mapping.occupancy.free_log_odds),
            occupied_threshold=float(config.mapping.occupancy.occupied_threshold),
            free_threshold=float(config.mapping.occupancy.free_threshold),
            inflation_radius_m=float(config.robot.geometry.map_inflation_radius_m),
            occupied_min_neighbors=int(config.mapping.occupancy.occupied_min_neighbors),
            unknown_is_traversable=bool(config.mapping.occupancy.unknown_is_traversable),
            unknown_cell_cost=float(config.mapping.occupancy.unknown_cell_cost),
            semantic_goal_step_m=float(config.mapping.occupancy.semantic_goal_step_m),
            semantic_goal_min_step_m=float(config.mapping.occupancy.semantic_goal_min_step_m),
            semantic_goal_lateral_search_m=float(config.mapping.occupancy.semantic_goal_lateral_search_m),
            semantic_anchor_snap_max_correction_m=float(
                config.mapping.occupancy.semantic_anchor_snap_max_correction_m
            ),
            semantic_anchor_snap_angular_tolerance_deg=float(
                config.mapping.occupancy.semantic_anchor_snap_angular_tolerance_deg
            ),
            semantic_anchor_snap_min_cells=int(
                config.mapping.occupancy.semantic_anchor_snap_min_cells
            ),
            semantic_anchor_snap_front_band_m=float(
                config.mapping.occupancy.semantic_anchor_snap_front_band_m
            ),
        ),
        logger=logger,
    )
    rgbd_ray_projector = RGBDVirtualRayProjector(
        config=RGBDGeometryMapConfig(
            resolution_m=float(config.mapping.occupancy.resolution_m),
            size_m=float(config.mapping.occupancy.size_m),
            update_period_sec=float(config.mapping.rgbd.update_period_sec),
            min_depth_m=float(config.mapping.rgbd.min_depth_m),
            max_depth_m=float(config.mapping.rgbd.max_depth_m),
            pixel_stride=int(config.mapping.rgbd.pixel_stride),
            vertical_min_ratio=float(config.mapping.rgbd.vertical_min_ratio),
            vertical_max_ratio=float(config.mapping.rgbd.vertical_max_ratio),
            obstacle_min_height_m=float(config.mapping.rgbd.obstacle_min_height_m),
            obstacle_max_height_m=float(config.mapping.rgbd.obstacle_max_height_m),
            floor_tolerance_below_m=float(config.mapping.rgbd.floor_tolerance_below_m),
        ),
        floor_z_m=float(config.robot_start_position[2]),
    )
    logger.info(
        "[GeometryMapConfig] source=rgbd depth_range=[%.2f, %.2f]m stride=%s "
        "height_band=[%.2f, %.2f]m",
        float(config.mapping.rgbd.min_depth_m),
        float(config.mapping.rgbd.max_depth_m),
        int(config.mapping.rgbd.pixel_stride),
        float(config.mapping.rgbd.obstacle_min_height_m),
        float(config.mapping.rgbd.obstacle_max_height_m),
    )
    primitive_collision_checker = PrimitiveCollisionChecker(
        config=PrimitiveCollisionCheckerConfig(
            forward_distance_m=float(config.robot.primitives.forward_distance_m),
            turn_angle_deg=float(config.robot.primitives.turn_angle_deg),
        )
    )
    map_candidate_generator = MapCandidateGenerator(
        config=MapCandidateGeneratorConfig(
            max_candidates=int(config.navigation.candidates.max_candidates),
            min_distance_m=float(config.navigation.candidates.min_distance_m),
            max_distance_m=float(config.navigation.candidates.max_distance_m),
            min_clearance_m=float(config.robot.geometry.path_corridor_radius_m),
            unknown_gain_weight=float(config.navigation.candidates.unknown_gain_weight),
            semantic_weight=float(config.navigation.candidates.semantic_weight),
        ),
        collision_checker=primitive_collision_checker,
        logger=logger,
    )
    reasoner = VLNReasoner(
        llm_client=llm_client,
        robot_capability_summary=config.robot.capability_summary(),
        logger=logger,
    )
    policy = VLNPolicy(
        instruction_context=instruction_context,
        candidate_predictor=candidate_predictor,
        occupancy_grid=occupancy_grid,
        map_candidate_generator=map_candidate_generator,
        progress_manager=progress_manager,
        reasoner=reasoner,
        config=VLNPolicyConfig(
            turn_speed_scale=1.0,
            forward_speed_scale=1.0,
            replanning_cooldown_sec=float(config.navigation.policy.replanning_cooldown_sec),
            active_forward_duration_sec=float(config.robot.primitives.forward_timeout_sec),
            active_turn_duration_sec=float(config.robot.primitives.turn_timeout_sec),
            active_turn_angle_deg=float(config.robot.primitives.turn_angle_deg),
            active_forward_distance_m=float(config.robot.primitives.forward_distance_m),
            active_distance_tolerance_m=float(config.robot.primitives.distance_tolerance_m),
            active_turn_tolerance_deg=float(config.robot.primitives.turn_tolerance_deg),
            region_transition_min_displacement_m=float(config.navigation.policy.region_transition.min_displacement_m),
            region_transition_confirmations=int(config.navigation.policy.region_transition.confirmations),
            region_transition_min_confidence=float(config.navigation.policy.region_transition.min_confidence),
            semantic_turn_min_yaw_deg=float(config.navigation.policy.semantic_turn.min_yaw_deg),
            semantic_turn_max_yaw_deg=float(config.navigation.policy.semantic_turn.max_yaw_deg),
            semantic_turn_max_primitives=int(config.navigation.policy.semantic_turn.max_primitives),
            semantic_turn_min_forward_clearance_m=float(config.navigation.policy.semantic_turn.min_forward_clearance_m),
            semantic_turn_alignment_confirmations=int(
                config.navigation.policy.semantic_turn.alignment_confirmations
            ),
            semantic_turn_corridor_center_tolerance_deg=float(
                config.navigation.policy.semantic_turn.corridor_center_tolerance_deg
            ),
            semantic_turn_same_corridor_tolerance_deg=float(
                config.navigation.policy.semantic_turn.same_corridor_tolerance_deg
            ),
            semantic_turn_target_high_confidence=float(
                config.navigation.policy.semantic_turn.target_high_confidence
            ),
            semantic_turn_target_medium_confidence=float(
                config.navigation.policy.semantic_turn.target_medium_confidence
            ),
            semantic_turn_direction_tolerance_deg=float(
                config.navigation.policy.semantic_turn.direction_tolerance_deg
            ),
            semantic_turn_corridor_half_fov_deg=float(
                config.navigation.policy.semantic_turn.corridor_half_fov_deg
            ),
            semantic_turn_corridor_step_deg=float(
                config.navigation.policy.semantic_turn.corridor_step_deg
            ),
            semantic_turn_min_corridor_width_deg=float(
                config.navigation.policy.semantic_turn.min_corridor_width_deg
            ),
            semantic_turn_corridor_max_clearance_m=float(
                config.navigation.policy.semantic_turn.corridor_max_clearance_m
            ),
            semantic_turn_max_recovery_attempts=int(
                config.navigation.policy.semantic_turn.max_recovery_attempts
            ),
            confirm_min_observations=int(config.perception.landmark_tracking.confirm_min_observations),
            confirm_confidence_threshold=float(config.perception.landmark_tracking.confirm_confidence_threshold),
            visual_target_center_deadband_deg=float(config.navigation.policy.visual_target_center_deadband_deg),
            landmark_arrival_confirmations=int(config.navigation.progress.landmark_arrival_confirmations),
            landmark_arrival_window_size=int(config.perception.landmark_tracking.arrival_window_size),
            arrival_distance_m=float(config.navigation.progress.landmark_clearance_m),
            success_distance_m=float(config.navigation.progress.final_success_clearance_m),
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
        physics_dt=config.robot.simulation.physics_dt,
        rendering_dt=config.robot.simulation.rendering_dt,
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
            timeout_sec=float(config.scene.stage_load_timeout_sec),
            logger=logger,
            label="environment",
        )
    if config.scene.add_default_ground_plane:
        world.scene.add_default_ground_plane()
        logger.info("Added default ground plane.")

    if config.scene.add_dome_light:
        try:
            from pxr import UsdLux

            dome = UsdLux.DomeLight.Define(stage, "/World/defaultDomeLight")
            dome.CreateIntensityAttr().Set(3000.0)
            dome.CreateExposureAttr().Set(2.0)
            # InteriorNav scenes are closed meshes. Let the Dome act as ambient
            # fill instead of being fully occluded by ceilings and exterior walls.
            shadow_api = UsdLux.ShadowAPI.Apply(dome.GetPrim())
            shadow_api.CreateShadowEnableAttr().Set(False)
            logger.info(
                "Added Dome Light: path=/World/defaultDomeLight "
                "intensity=%s exposure=%s shadow_enabled=%s",
                dome.GetIntensityAttr().Get(),
                dome.GetExposureAttr().Get(),
                shadow_api.GetShadowEnableAttr().Get(),
            )
        except Exception as exc:
            logger.warning("Failed to add Dome Light: %s", exc)

    if has_environment:
        try:
            add_indoor_lighting(
                stage=stage,
                environment_prim_path=config.scene.prim_path,
                logger=logger,
            )
        except Exception:
            logger.exception("[InteriorLighting] failed to add scene lights")

    add_reference_to_stage(str(config.robot.asset.usd_path), config.robot.asset.prim_path)
    _wait_for_stage_assets(
        simulation_app=simulation_app,
        is_stage_loading=is_stage_loading,
        timeout_sec=float(config.scene.stage_load_timeout_sec),
        logger=logger,
        label="robot",
    )
    _set_local_xform(
        stage=stage,
        prim_path=config.robot.asset.prim_path,
        translation=config.robot_start_position,
        rotation_deg=config.scene.robot_start_orientation_deg,
    )

    articulation_path = (
        _resolve_robot_relative_prim_path(
            robot_root_path=config.robot.asset.prim_path,
            prim_path=config.robot.asset.articulation_path,
        )
        if str(config.robot.asset.articulation_path).strip()
        else _find_articulation_root_path(stage=stage, root_path=config.robot.asset.prim_path)
    )
    logger.info("Using articulation prim: %s", articulation_path)
    if config.robot is not None and config.robot.is_g1:
        articulation_prim = stage.GetPrimAtPath(articulation_path)
        if not articulation_prim or not articulation_prim.IsValid():
            raise RuntimeError(f"Required G1 articulation prim does not exist: {articulation_path}")

    robot = world.scene.add(
        RobotArticulation(
            prim_path=articulation_path,
            name="custom_robot",
            reset_xform_properties=not config.robot.is_g1,
        )
    )
    front_camera_prim_path = _resolve_robot_relative_prim_path(
        robot_root_path=config.robot.asset.prim_path,
        prim_path=config.robot.sensors.camera.front_prim_path,
    )
    observer_camera_prim_path = _resolve_robot_relative_prim_path(
        robot_root_path=config.robot.asset.prim_path,
        prim_path=config.robot.sensors.camera.observer_prim_path,
    )
    front_camera_prim_path = _resolve_existing_prim_path(
        stage=stage,
        prim_path=front_camera_prim_path,
        kind="front camera",
        strict=bool(config.robot is not None and config.robot.is_g1),
    )
    if config.robot.sensors.camera.observer_enabled:
        observer_camera_prim_path = _resolve_existing_prim_path(
            stage=stage,
            prim_path=observer_camera_prim_path,
            kind="observer camera",
            strict=bool(config.robot is not None and config.robot.is_g1),
        )
    logger.info("Using front camera prim: %s", front_camera_prim_path)
    if config.robot.sensors.camera.observer_enabled:
        logger.info("Using observer camera prim: %s", observer_camera_prim_path)

    camera = Camera(
        prim_path=front_camera_prim_path,
        frequency=max(1, int(config.robot.sensors.camera.frequency_hz)),
        resolution=(config.robot.sensors.camera.width, config.robot.sensors.camera.height),
    )
    observer_camera = None
    if config.robot.sensors.camera.observer_enabled:
        observer_camera = Camera(
            prim_path=observer_camera_prim_path,
            frequency=max(1, int(config.robot.sensors.camera.frequency_hz)),
            resolution=(config.robot.sensors.camera.width, config.robot.sensors.camera.height),
        )

    _set_viewport_camera(config=config, logger=logger)

    logger.info("Calling world.reset()")
    world.reset()
    logger.info("world.reset() complete")
    robot.initialize()
    logger.info("robot.initialize() complete")
    camera.initialize()
    logger.info("front camera initialize() complete")
    try:
        actual_camera_hfov_deg = float(np.degrees(float(camera.get_horizontal_fov())))
    except Exception as exc:
        actual_camera_hfov_deg = float(config.robot.sensors.camera.horizontal_fov_deg)
        logger.warning(
            "[CameraCalibration] unable to read USD horizontal FOV; using configured %.2fdeg error=%s",
            actual_camera_hfov_deg,
            exc,
        )
    if not np.isfinite(actual_camera_hfov_deg) or not 20.0 <= actual_camera_hfov_deg <= 160.0:
        logger.warning(
            "[CameraCalibration] invalid USD horizontal FOV %.2fdeg; using configured %.2fdeg",
            actual_camera_hfov_deg,
            float(config.robot.sensors.camera.horizontal_fov_deg),
        )
        actual_camera_hfov_deg = float(config.robot.sensors.camera.horizontal_fov_deg)
    configured_camera_hfov_deg = float(config.robot.sensors.camera.horizontal_fov_deg)
    try:
        front_camera_intrinsics = np.asarray(camera.get_intrinsics_matrix(), dtype=np.float64).reshape(3, 3)
    except Exception as exc:
        raise RuntimeError(
            "Front camera must expose pinhole intrinsics for calibrated landmark projection"
        ) from exc
    observation_adapter.set_horizontal_fov_deg(actual_camera_hfov_deg)
    candidate_predictor.set_horizontal_fov_deg(actual_camera_hfov_deg)
    target_perception.set_horizontal_fov_deg(actual_camera_hfov_deg)
    logger.info(
        "[CameraCalibration] configured_hfov_deg=%.2f usd_hfov_deg=%.2f source=front_camera_prim",
        configured_camera_hfov_deg,
        actual_camera_hfov_deg,
    )
    log_spatial_transform_self_check(
        logger=logger,
        image_width=int(config.robot.sensors.camera.width),
        image_height=int(config.robot.sensors.camera.height),
        horizontal_fov_deg=actual_camera_hfov_deg,
    )
    _attach_camera_annotators(camera=camera, logger=logger, enable_depth=True)
    if observer_camera is not None:
        observer_camera.initialize()
        logger.info("observer camera initialize() complete")
        _attach_camera_annotators(camera=observer_camera, logger=logger, enable_depth=False)
    _bind_viewport_to_camera(
        camera_prim_path=front_camera_prim_path,
        logger=logger,
        enabled=config.robot.sensors.camera.follow_front_in_gui,
    )
    if observer_camera is not None:
        _bind_viewport_to_camera(
            camera_prim_path=observer_camera_prim_path,
            logger=logger,
            enabled=config.robot.sensors.camera.follow_observer_in_gui,
        )

    available_joint_names = _get_joint_names(robot)
    logger.info("Detected articulation DOFs: %s", available_joint_names)
    if config.robot is None:
        raise RuntimeError("Robot profile was not loaded")
    drive_controller = create_locomotion_controller(
        profile=config.robot,
        robot=robot,
        stage=stage,
        articulation_path=articulation_path,
        available_joint_names=available_joint_names,
        logger=logger,
    )
    logger.info(
        "[RobotProfile] name=%s type=%s usd=%s physics_dt=%.4f render_dt=%.4f decimation=%s spawn_z=%.2f",
        config.robot.name,
        config.robot.robot_type,
        config.robot.asset.usd_path,
        float(config.robot.simulation.physics_dt),
        float(config.robot.simulation.rendering_dt),
        int(config.robot.simulation.policy_decimation),
        float(config.robot.spawn_z_m),
    )
    logger.info(
        "[OdomConfig] source=isaac_world_pose primitive_completion=odom_or_timeout forward_distance=%.2fm turn_angle=%.1fdeg tolerance=%.1fdeg",
        float(config.robot.primitives.forward_distance_m),
        float(config.robot.primitives.turn_angle_deg),
        float(config.robot.primitives.turn_tolerance_deg),
    )
    depth_debug_output_path = _resolve_output_path(config.debug.image.depth_path)
    target_debug_output_path = _resolve_output_path(config.debug.image.path)
    target_debug_video_path = _resolve_output_path(config.debug.video.path)
    debug_video_recorder = DebugVideoRecorder(
        enabled=bool(config.debug.video.enabled) and bool(config.debug.image.enabled),
        output_path=target_debug_video_path,
        fps=float(config.debug.video.fps),
        logger=logger,
    )
    observer_video_recorder = DebugVideoRecorder(
        enabled=bool(config.debug.observer_video.enabled) and observer_camera is not None,
        output_path=_resolve_output_path(config.debug.observer_video.path),
        fps=float(config.debug.observer_video.fps),
        logger=logger,
    )
    if config.debug.image.enabled:
        logger.info("Runtime debug image output: %s", target_debug_output_path)
    logger.info("Occupancy map is rendered in runtime debug only; VLM receives structured map text.")
    if config.debug.video.enabled and config.debug.image.enabled:
        logger.info("Runtime debug video output: %s", debug_video_recorder.output_path)
    if config.debug.observer_video.enabled and observer_camera is not None:
        logger.info(
            "Observer camera video output: %s fps=%.1f",
            observer_video_recorder.output_path,
            float(config.debug.observer_video.fps),
        )
    if config.debug.image.depth_enabled:
        logger.info("Depth debug output: %s", depth_debug_output_path)
    logger.info(
        "[TopologyConfig] node_spacing=%.2fm turn_threshold=%.1fdeg merge_radius=%.2fm bind_radius=%.2fm max_nodes=%s recent_nodes=%s",
        float(config.mapping.topology.node_spacing_m),
        float(config.mapping.topology.turn_threshold_deg),
        float(config.mapping.topology.merge_radius_m),
        float(config.mapping.topology.landmark_bind_radius_m),
        int(config.mapping.topology.max_nodes),
        int(config.mapping.topology.recent_node_count),
    )
    logger.info(
        "[OccupancyGridConfig] source=rgbd resolution=%.2fm size=%.1fm min_neighbors=%s debug=%s",
        float(config.mapping.occupancy.resolution_m),
        float(config.mapping.occupancy.size_m),
        int(config.mapping.occupancy.occupied_min_neighbors),
        occupancy_grid.output_path,
    )
    last_depth_image: np.ndarray | None = None
    last_depth_eval_time = 0.0
    frame_index = 0
    last_action_name = "STOP"
    last_applied_wheel_velocity = 0.0
    last_applied_steering_angle = 0.0
    last_motion_log_time = 0.0
    last_logged_position = np.asarray(config.robot_start_position, dtype=np.float32)
    last_logged_policy_signature: tuple | None = None
    last_policy_feedback: ExecutionFeedback | None = None
    active_step_result = None
    active_command_token = 0
    initial_robot_orientation_xyzw: np.ndarray | None = None
    depth_warning_emitted = False
    depth_debug_save_pending = False
    last_safety_log_signature: tuple[str, str] | None = None
    runtime_safety_hold_streak = 0
    runtime_approach_hold_streak = 0
    last_nav_debug_save_time = 0.0
    last_depth_debug_save_time = 0.0
    sim_time_sec = 0.0
    route_complete_handled = False
    navigation_finished = False
    navigation_failed = False
    fall_condition_started_sec: float | None = None
    primitive_settle_started_sec: float | None = None
    primitive_settle_stable_started_sec: float | None = None
    landmark_map_reached_synced: set[str] = set()
    global_pose_trace: list[tuple[float, float]] = []
    latest_global_map_image: Image.Image | None = None
    skip_target_detection_once = False
    target_keyframe_wait_reason: str | None = None
    last_observer_video_frame_sim_time = float("-inf")

    warmup_steps = int(config.simulation.warmup_steps)
    g1_render_stride = 1
    if config.robot.is_g1 and config.robot.g1_controller is not None:
        warmup_steps = max(
            warmup_steps,
            int(np.ceil(config.robot.g1_controller.standup_duration_sec / config.robot.simulation.physics_dt)),
        )
        render_ratio = float(config.robot.simulation.rendering_dt) / float(config.robot.simulation.physics_dt)
        g1_render_stride = max(1, int(round(render_ratio)))
        if not np.isclose(render_ratio, float(g1_render_stride), rtol=0.0, atol=1.0e-6):
            raise RuntimeError(
                "G1 rendering_dt must be an integer multiple of physics_dt: "
                f"rendering_dt={config.robot.simulation.rendering_dt} physics_dt={config.robot.simulation.physics_dt}"
            )
        logger.info(
            "[G1Startup] zero-command stabilization duration=%.2fs steps=%s "
            "policy_decimation=%s render_stride=%s",
            float(config.robot.g1_controller.standup_duration_sec),
            warmup_steps,
            int(config.robot.simulation.policy_decimation),
            g1_render_stride,
        )
        drive_controller.apply("STOP", speed_scale=0.0)
    for warmup_index in range(warmup_steps):
        if config.robot.is_g1:
            drive_controller.advance_physics()
            world.step(render=False)
            if (warmup_index + 1) % g1_render_stride == 0:
                world.render()
        else:
            drive_controller.apply("STOP", speed_scale=0.0)
            world.step(render=True)
        sim_time_sec += float(config.robot.simulation.physics_dt)
    logger.info("Warmup complete, controller is armed.")
    occupancy_bootstrap_end_sec = sim_time_sec
    occupancy_bootstrap_deadline_sec = sim_time_sec
    occupancy_bootstrap_logged = False
    occupancy_bootstrap_done_logged = False
    occupancy_bootstrap_last_ray_bucket = -1
    rgbd_mapping_ready = False
    latest_rgbd_ray_count = 0
    rgbd_frame_check_logged = False
    occupancy_bootstrap_end_sec = sim_time_sec + max(
        0.0,
        float(config.mapping.rgbd.bootstrap_sec),
    )
    occupancy_bootstrap_deadline_sec = occupancy_bootstrap_end_sec + max(
        0.0,
        float(config.mapping.rgbd.bootstrap_max_wait_sec),
    )

    while simulation_app.is_running():
        if config.robot.is_g1:
            for _ in range(g1_render_stride):
                drive_controller.advance_physics()
                world.step(render=False)
                sim_time_sec += float(config.robot.simulation.physics_dt)
            world.render()
        else:
            world.step(render=True)
            sim_time_sec += float(config.robot.simulation.physics_dt)
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

        if config.robot.safety.fall_detection_enabled:
            roll_deg, pitch_deg = _relative_roll_pitch_deg(
                initial_quaternion_xyzw=initial_robot_orientation_xyzw,
                current_quaternion_xyzw=orientation,
            )
            fall_condition = bool(
                abs(float(roll_deg)) >= float(config.robot.safety.max_roll_pitch_deg)
                or abs(float(pitch_deg)) >= float(config.robot.safety.max_roll_pitch_deg)
                or float(position[2]) < float(config.robot.safety.min_base_height_m)
            )
            if fall_condition:
                if fall_condition_started_sec is None:
                    fall_condition_started_sec = sim_time_sec
                fall_duration_sec = sim_time_sec - float(fall_condition_started_sec)
                if (
                    not navigation_failed
                    and fall_duration_sec >= float(config.robot.safety.confirmation_sec)
                ):
                    navigation_failed = True
                    navigation_finished = True
                    drive_controller.apply("STOP", speed_scale=0.0)
                    target_perception.close()
                    debug_video_recorder.close()
                    observer_video_recorder.close()
                    _write_evaluation_result(
                        output_path=evaluation_result_path,
                        status="FAILED",
                        task_id=task_id,
                        config=config,
                        instruction_context=instruction_context,
                        step_result=active_step_result,
                        sim_time_sec=sim_time_sec,
                        failure_stage="robot_fall",
                        reason=(
                            f"roll_deg={roll_deg:.1f} pitch_deg={pitch_deg:.1f} "
                            f"base_z_m={float(position[2]):.3f}"
                        ),
                    )
                    logger.error(
                        "[NAVIGATION_FAILED] reason=robot_fall roll_deg=%.1f pitch_deg=%.1f "
                        "base_z_m=%.3f confirmation_sec=%.2f; simulation remains open with active standing",
                        float(roll_deg),
                        float(pitch_deg),
                        float(position[2]),
                        float(fall_duration_sec),
                    )
            else:
                fall_condition_started_sec = None

        if navigation_finished:
            drive_controller.apply("STOP", speed_scale=0.0)
            last_action_name = "STOP"
            continue

        observer_video_period_sec = 1.0 / max(0.5, float(config.debug.observer_video.fps))
        if (
            bool(config.debug.observer_video.enabled)
            and observer_camera is not None
            and (sim_time_sec - last_observer_video_frame_sim_time) >= observer_video_period_sec
        ):
            observer_rgb = _get_rgb_image(camera=observer_camera)
            if observer_rgb is not None:
                observer_video_recorder.append(observer_rgb)
                last_observer_video_frame_sim_time = sim_time_sec

        depth_frame_updated = False
        if last_depth_image is None or (loop_now - last_depth_eval_time) >= float(config.robot.sensors.camera.depth_eval_period_sec):
            last_depth_image = _get_depth_image(camera=camera, logger=logger)
            last_depth_eval_time = loop_now
            depth_debug_save_pending = last_depth_image is not None
            depth_frame_updated = last_depth_image is not None
            if last_depth_image is None and not depth_warning_emitted:
                logger.warning(
                    "Depth frame unavailable; CAPTAINav candidate generation is waiting "
                    "for the first valid frame."
                )
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
            odom_pose=odom_state.pose,
            odom_meta={
                "enabled": True,
                "source": odom_state.source,
                "linear_velocity_mps": float(odom_state.linear_velocity_mps),
                "angular_velocity_radps": float(odom_state.angular_velocity_radps),
                "delta_distance_m": float(odom_state.delta_distance_m),
                "delta_yaw_rad": float(odom_state.delta_yaw_rad),
                "total_distance_m": float(odom_state.total_distance_m),
            },
        )
        mapping_pose = odom_state.raw_pose if config.robot.is_g1 else observation.pose
        if depth_frame_updated and last_depth_image is not None:
            try:
                mapping_view_matrix_ros = np.asarray(
                    camera.get_view_matrix_ros(),
                    dtype=np.float64,
                ).reshape(4, 4)
                rgbd_ray_frame = rgbd_ray_projector.project(
                    depth_image=last_depth_image,
                    camera_intrinsics=front_camera_intrinsics,
                    camera_view_matrix_ros=mapping_view_matrix_ros,
                )
                latest_rgbd_ray_count = int(rgbd_ray_frame.ray_count)
                if rgbd_ray_frame.ray_count > 0:
                    sensor_odom_xy = transform_world_xy_to_odom(
                        rgbd_ray_frame.sensor_world_xy,
                        origin_world_xy=odom_state.origin_world_xy,
                        origin_world_yaw_rad=odom_state.origin_world_yaw_rad,
                    )
                    endpoints_odom_xy = transform_world_xy_to_odom(
                        rgbd_ray_frame.endpoints_world_xy,
                        origin_world_xy=odom_state.origin_world_xy,
                        origin_world_yaw_rad=odom_state.origin_world_yaw_rad,
                    )
                    if not rgbd_frame_check_logged:
                        camera_offset_m = float(
                            np.linalg.norm(
                                sensor_odom_xy
                                - np.asarray(
                                    [float(mapping_pose.x), float(mapping_pose.y)],
                                    dtype=np.float64,
                                )
                            )
                        )
                        log_fn = logger.info if camera_offset_m <= 2.0 else logger.warning
                        log_fn(
                            "[RGBDFrameCheck] frame=odom origin_world=[%.3f,%.3f] "
                            "origin_yaw_deg=%.1f robot_odom=[%.3f,%.3f] "
                            "camera_odom=[%.3f,%.3f] camera_offset_m=%.3f rays=%s",
                            float(odom_state.origin_world_xy[0]),
                            float(odom_state.origin_world_xy[1]),
                            float(np.degrees(odom_state.origin_world_yaw_rad)),
                            float(mapping_pose.x),
                            float(mapping_pose.y),
                            float(sensor_odom_xy[0]),
                            float(sensor_odom_xy[1]),
                            camera_offset_m,
                            int(rgbd_ray_frame.ray_count),
                        )
                        rgbd_frame_check_logged = True
                    occupancy_grid.update_from_world_rays(
                        robot_pose=mapping_pose,
                        sensor_world_xy=sensor_odom_xy,
                        beam_endpoints_world_xy=endpoints_odom_xy,
                        beam_hit_mask=rgbd_ray_frame.hit_mask,
                        timestamp_sec=sim_time_sec,
                    )
                    rgbd_mapping_ready = True
            except Exception as exc:
                logger.warning("[RGBDMap] frame_rejected error=%s", exc)
        bootstrap_time_ready = sim_time_sec >= float(occupancy_bootstrap_end_sec)
        bootstrap_deadline_reached = sim_time_sec >= float(occupancy_bootstrap_deadline_sec)
        if not (
            bootstrap_time_ready
            and (bool(rgbd_mapping_ready) or bootstrap_deadline_reached)
        ):
            if not occupancy_bootstrap_logged:
                logger.info(
                    "[OccupancyBootstrap] accumulating initial RGB-D map before first policy "
                    "step duration_sec=%.2f max_wait_sec=%.2f",
                    max(0.0, float(config.mapping.rgbd.bootstrap_sec)),
                    float(config.mapping.rgbd.bootstrap_max_wait_sec),
                )
                occupancy_bootstrap_logged = True
            ray_bucket = int(latest_rgbd_ray_count // 10)
            if ray_bucket != occupancy_bootstrap_last_ray_bucket:
                logger.info(
                    "[OccupancyBootstrap] waiting time_ready=%s rgbd_ready=%s "
                    "deadline=%s rays=%s",
                    int(bool(bootstrap_time_ready)),
                    int(bool(rgbd_mapping_ready)),
                    int(bool(bootstrap_deadline_reached)),
                    int(latest_rgbd_ray_count),
                )
                occupancy_bootstrap_last_ray_bucket = ray_bucket
            drive_controller.apply("STOP", speed_scale=0.0)
            last_action_name = "STOP"
            continue
        if occupancy_bootstrap_logged and not occupancy_bootstrap_done_logged:
            logger.info(
                "[OccupancyBootstrap] complete; starting VLN policy. source=rgbd "
                "rays=%s deadline_reached=%s",
                int(latest_rgbd_ray_count),
                int(bool(bootstrap_deadline_reached)),
            )
            occupancy_bootstrap_done_logged = True

        runtime_recovery_command = stuck_detector.recovery_command(
            now_sec=sim_time_sec,
            observation=observation,
        )
        if runtime_recovery_command.active:
            if runtime_recovery_command.action_name in {
                "FORWARD",
                "BACKWARD",
                "TURN_LEFT",
                "TURN_RIGHT",
            }:
                recovery_collision = primitive_collision_checker.check(
                    action_name=runtime_recovery_command.action_name,
                    robot_pose=observation.pose,
                    snapshot=occupancy_grid.snapshot(),
                    travel_distance_m=float(runtime_recovery_command.remaining_distance_m),
                    turn_angle_deg=float(runtime_recovery_command.remaining_yaw_deg),
                )
                if not recovery_collision.valid:
                    runtime_recovery_command = stuck_detector.abort_recovery(
                        now_sec=sim_time_sec,
                        reason=f"rollback_collision:{recovery_collision.reason}",
                    )
                else:
                    recovery_safety = depth_safety_monitor.apply(
                        observation=observation,
                        action_name=runtime_recovery_command.action_name,
                        speed_scale=runtime_recovery_command.speed_scale,
                    )
                    if recovery_safety.overridden:
                        runtime_recovery_command = stuck_detector.abort_recovery(
                            now_sec=sim_time_sec,
                            reason=f"rollback_emergency:{recovery_safety.reason}",
                        )
            policy.cancel_active_goal()
            active_step_result = _runtime_recovery_step_result(
                recovery_command=runtime_recovery_command,
                previous_step_result=active_step_result,
            )
            last_applied_wheel_velocity, last_applied_steering_angle = drive_controller.apply(
                runtime_recovery_command.action_name,
                speed_scale=runtime_recovery_command.speed_scale,
            )
            last_action_name = runtime_recovery_command.action_name
            last_policy_feedback = ExecutionFeedback(
                applied_action=runtime_recovery_command.action_name,
                wheel_velocity=last_applied_wheel_velocity,
                steering_angle=last_applied_steering_angle,
            )
            if runtime_recovery_command.completed:
                policy.request_immediate_replan()
                policy.set_safety_context(
                    "RECOVERY_COMPLETE map refreshed; discard the blocked local action "
                    "and replan the current instruction step."
                )
                logger.warning(
                    "[RecoveryReplan] reason=%s result=checkpoint_exit next=fresh_map_replan",
                    runtime_recovery_command.reason,
                )
            continue

        settle_required = bool(
            active_step_result is not None
            and str(getattr(active_step_result, "decision_source", "")).strip().upper()
            == "PRIMITIVE_COMPLETE"
            and float(config.robot.primitives.settle_max_wait_sec) > 0.0
        )
        if settle_required:
            if primitive_settle_started_sec is None:
                primitive_settle_started_sec = sim_time_sec
                primitive_settle_stable_started_sec = None
            linear_speed = abs(float(odom_state.linear_velocity_mps))
            angular_speed = abs(float(odom_state.angular_velocity_radps))
            stable_now = bool(
                linear_speed <= float(config.robot.primitives.settle_linear_speed_threshold_mps)
                and angular_speed <= float(config.robot.primitives.settle_angular_speed_threshold_radps)
            )
            if stable_now:
                if primitive_settle_stable_started_sec is None:
                    primitive_settle_stable_started_sec = sim_time_sec
            else:
                primitive_settle_stable_started_sec = None
            settle_elapsed = sim_time_sec - float(primitive_settle_started_sec)
            stable_elapsed = (
                0.0
                if primitive_settle_stable_started_sec is None
                else sim_time_sec - float(primitive_settle_stable_started_sec)
            )
            settle_complete = bool(
                stable_elapsed >= float(config.robot.primitives.settle_stable_duration_sec)
                or settle_elapsed >= float(config.robot.primitives.settle_max_wait_sec)
            )
            if not settle_complete:
                last_applied_wheel_velocity, last_applied_steering_angle = drive_controller.apply(
                    "STOP", speed_scale=0.0
                )
                last_action_name = "STOP"
                last_policy_feedback = ExecutionFeedback(
                    applied_action="STOP",
                    wheel_velocity=last_applied_wheel_velocity,
                    steering_angle=last_applied_steering_angle,
                )
                continue
            logger.info(
                "[PrimitiveSettle] complete elapsed_sec=%.2f stable_sec=%.2f linear_mps=%.3f angular_radps=%.3f",
                float(settle_elapsed),
                float(stable_elapsed),
                float(linear_speed),
                float(angular_speed),
            )
            active_step_result = None
            primitive_settle_started_sec = None
            primitive_settle_stable_started_sec = None
        elif _is_executing_motion_step(active_step_result):
            primitive_settle_started_sec = None
            primitive_settle_stable_started_sec = None

        # --- Fast path during active primitive execution ---
        is_motion_active = _is_executing_motion_step(active_step_result)
        if is_motion_active:
            interrupted_primitive_context = policy.active_primitive_context()
            proposed_step_result = policy.step(observation, execution_feedback=last_policy_feedback)
            active_step_result = proposed_step_result
            step_result = active_step_result
            _record_completed_primitive(
                step_result=step_result,
                stuck_detector=stuck_detector,
            )
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
            if safety_decision.overridden:
                runtime_safety_hold_streak += 1
                policy.cancel_active_goal()
                active_step_result = None
                policy.set_safety_context(
                    f"SAFETY_STOP previous_desired={step_result.action_name}; "
                    f"reason={safety_decision.reason}; "
                    f"depth_m front={safety_decision.front_depth_m:.2f}, "
                    f"left={safety_decision.left_depth_m:.2f}, "
                    f"right={safety_decision.right_depth_m:.2f}. "
                    "Use the fresh sensor frame and choose another primitive."
                )
                logger.warning(
                    "[SAFETY_HOLD] desired=%s reason=%s depth_m(front=%.2f,left=%.2f,right=%.2f) next=fresh_replan",
                    step_result.action_name,
                    safety_decision.reason,
                    float(safety_decision.front_depth_m),
                    float(safety_decision.left_depth_m),
                    float(safety_decision.right_depth_m),
                )
                if runtime_safety_hold_streak >= 2:
                    interrupted_action = ""
                    interrupted_start_pose = None
                    if interrupted_primitive_context is not None:
                        interrupted_action, interrupted_start_pose = interrupted_primitive_context
                    stuck_detector.trigger_recovery(
                        now_sec=sim_time_sec,
                        reason="repeated_safety_hold",
                        current_pose=observation.pose,
                        interrupted_action=interrupted_action,
                        interrupted_start_pose=interrupted_start_pose,
                    )
                    runtime_safety_hold_streak = 0
            else:
                runtime_safety_hold_streak = 0
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
                target_perception.close()
                debug_video_recorder.close()
                observer_video_recorder.close()
                _write_evaluation_result(
                    output_path=evaluation_result_path,
                    status="SUCCESS",
                    task_id=task_id,
                    config=config,
                    instruction_context=instruction_context,
                    step_result=step_result,
                    sim_time_sec=sim_time_sec,
                    failure_stage="",
                    reason="all typed instruction steps completed",
                )
            continue

        if policy.current_step_type == "TURN":
            policy.update_turn_heading_context(
                robot_pose=mapping_pose,
                occupancy_snapshot=occupancy_grid.snapshot(),
            )

        current_landmark = policy.active_perception_target
        target_observation: TargetObservation | None = None
        landmark_map_entry = None
        target_detection_observation: TargetObservation | None = None
        target_keyframe_ready = False
        if current_landmark:
            try:
                front_camera_view_matrix_ros = np.asarray(
                    camera.get_view_matrix_ros(), dtype=np.float64
                ).reshape(4, 4)
            except Exception as exc:
                raise RuntimeError(
                    "Front camera ROS view matrix is unavailable for landmark projection"
                ) from exc
            remembered_landmark_entry = landmark_map.get(current_landmark)
            allow_new_target_detection = not skip_target_detection_once
            target_detection_observation = target_perception.observe(
                rgb=front_rgb,
                depth=last_depth_image,
                target_label=current_landmark,
                robot_pose=observation.pose,
                timestamp_sec=sim_time_sec,
                frame_index=frame_index,
                allow_new_request=allow_new_target_detection,
                preferred_world_xy=(
                    None
                    if remembered_landmark_entry is None
                    else list(remembered_landmark_entry.world_xy)
                ),
                camera_intrinsics=front_camera_intrinsics,
                camera_view_matrix_ros=front_camera_view_matrix_ros,
                robot_world_position_xyz=position,
                robot_world_orientation_wxyz=orientation,
                force_semantic_verification=(
                    policy.current_grounding_behavior_type
                    in {"turn_cue", "turn_lookahead"}
                ),
                expected_turn_direction=(
                    policy.current_turn_direction
                    if policy.current_grounding_behavior_type
                    in {"turn_cue", "turn_lookahead"}
                    else ""
                ),
                turn_start_yaw_rad=(
                    policy.current_turn_start_yaw_rad
                    if policy.current_grounding_behavior_type
                    in {"turn_cue", "turn_lookahead"}
                    else None
                ),
                turn_direction_tolerance_deg=float(
                    policy.current_turn_direction_tolerance_deg
                ),
                semantic_relation_text=policy.current_semantic_relation_text,
            )
            detection_reason = str(target_detection_observation.reason).strip().lower()
            detection_pending = bool(target_perception.request_in_flight) or detection_reason in {
                "grounding_request_started",
                "grounding_request_in_flight",
            }
            if detection_pending:
                if target_keyframe_wait_reason != detection_reason:
                    logger.info(
                        "[PerceptionKeyframe] target=%s state=waiting reason=%s "
                        "action=hold_current_pose",
                        current_landmark,
                        detection_reason,
                    )
                    target_keyframe_wait_reason = detection_reason
                last_applied_wheel_velocity, last_applied_steering_angle = (
                    drive_controller.apply("STOP", speed_scale=0.0)
                )
                last_action_name = "STOP"
                last_policy_feedback = ExecutionFeedback(
                    applied_action="STOP",
                    wheel_velocity=last_applied_wheel_velocity,
                    steering_angle=last_applied_steering_angle,
                )
                continue
            if target_keyframe_wait_reason is not None:
                target_keyframe_ready = True
                logger.info(
                    "[PerceptionKeyframe] target=%s state=ready visible=%s bbox=%s "
                    "current=%s reason=%s",
                    current_landmark,
                    int(bool(target_detection_observation.visible)),
                    list(target_detection_observation.bbox or ()),
                    int(bool(target_detection_observation.is_current_frame)),
                    detection_reason,
                )
                target_keyframe_wait_reason = None
            if skip_target_detection_once:
                skip_target_detection_once = False
            target_observation = _match_target_observation_to_occupancy(
                observation=target_detection_observation,
                occupancy_grid=occupancy_grid,
                behavior_type=policy.current_grounding_behavior_type,
                logger=logger,
            )
            target_detection_observation = target_observation
            if target_keyframe_ready:
                logger.info(
                    "[TargetFrame] target=%s bbox_source=%s semantic_source=%s "
                    "request_valid=%s grounding_bbox_current=%s",
                    current_landmark,
                    str(target_detection_observation.source),
                    str(target_detection_observation.semantic_source),
                    int(bool(getattr(target_detection_observation, "request_evidence_valid", False))),
                    int(bool(target_detection_observation.is_current_frame)),
                )
            landmark_state = landmark_tracker.update(
                target_label=current_landmark,
                behavior_type=policy.current_grounding_behavior_type,
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
        policy.set_topology_node_id(
            topological_memory.current_node_id(robot_pose=observation.pose)
        )

        obstacle_summary = _extract_obstacle_summary(
            depth=last_depth_image,
            occupancy_grid=occupancy_grid,
            robot_pose=observation.pose,
            stop_distance_m=float(config.robot.geometry.obstacle_stop_distance_m),
        )
        # Layout classification
        policy.set_layout_description(
            _classify_occupancy_layout(
                occupancy_grid=occupancy_grid,
                robot_pose=observation.pose,
            )
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
        target_verification_vlm_was_used = bool(
            current_landmark
            and target_detection_observation is not None
            and str(
                getattr(target_detection_observation, "vlm_verification_reason", "")
            )
            not in {"", "not_run", "auto_accept_high_detector_confidence"}
        )
        if target_verification_vlm_was_used and not bool(
            getattr(target_detection_observation, "can_confirm", False)
        ):
            skip_target_detection_once = True
            logger.info(
                "[TargetVerify] target=%s result=rejected navigation=continue_search",
                current_landmark,
            )
        should_refresh_policy = True
        proposed_step_result = policy.step(observation, execution_feedback=last_policy_feedback)
        proposed_command_token = active_command_token + int(bool(proposed_step_result.replanned))
        active_step_result = proposed_step_result
        active_command_token = proposed_command_token
        step_result = active_step_result
        if str(step_result.decision_source) == "TURN_RECOVERY_REQUEST":
            stuck_detector.trigger_recovery(
                now_sec=sim_time_sec,
                reason=str(step_result.last_error or step_result.reasoning_summary),
                current_pose=observation.pose,
            )
        if str(step_result.decision_source) in {
            "APPROACH_SAFETY_HOLD",
            "VISUAL_TARGET_SAFETY_HOLD",
            "TURN_SCAN_SAFETY_HOLD",
            "TURN_ALIGNMENT_SAFETY_HOLD",
        }:
            runtime_approach_hold_streak += 1
            if runtime_approach_hold_streak >= 3:
                stuck_detector.trigger_recovery(
                    now_sec=sim_time_sec,
                    reason="approach_path_unreachable",
                    current_pose=observation.pose,
                )
                runtime_approach_hold_streak = 0
        elif str(step_result.decision_source) != "REPLAN_COOLDOWN":
            runtime_approach_hold_streak = 0
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
            runtime_safety_hold_streak += 1
            policy.cancel_active_goal()
            safety_context = (
                f"SAFETY_STOP previous_desired={step_result.action_name}; "
                f"runtime_applied={safety_decision.action_name}; "
                f"reason={safety_decision.reason}; "
                f"depth_m front={safety_decision.front_depth_m:.2f}, "
                f"left={safety_decision.left_depth_m:.2f}, "
                f"right={safety_decision.right_depth_m:.2f}; "
                f"stop_threshold_m={config.robot.geometry.obstacle_stop_distance_m:.2f}. "
                "The runtime stopped the unsafe action; choose another primitive."
            )
            policy.set_safety_context(safety_context)
            safety_signature = (str(step_result.action_name), str(safety_decision.reason))
            if safety_signature != last_safety_log_signature:
                logger.warning(
                    "[SAFETY_HOLD] desired=%s applied=%s reason=%s depth_m(front=%.2f,left=%.2f,right=%.2f) next=fresh_replan",
                    step_result.action_name,
                    safety_decision.action_name,
                    safety_decision.reason,
                    float(safety_decision.front_depth_m),
                    float(safety_decision.left_depth_m),
                    float(safety_decision.right_depth_m),
                )
                last_safety_log_signature = safety_signature
            if runtime_safety_hold_streak >= 2:
                stuck_detector.trigger_recovery(
                    now_sec=sim_time_sec,
                    reason="repeated_safety_hold",
                    current_pose=observation.pose,
                )
                runtime_safety_hold_streak = 0
        else:
            runtime_safety_hold_streak = 0
            last_safety_log_signature = None
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
                "Runtime will retrace the latest verified-safe primitive and then replan."
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
                "[NAV_STATE] step=%s/%s type=%s target=%s landmark=%s completed=%s route_complete=%s action=%s speed_scale=%.2f source=%s token=%s candidate=%s remaining=%s progress=%s replanned=%s local_goal_xy=%s goal_xy=%s",
                int(step_result.current_step_index) + 1,
                int(step_result.route_step_count),
                step_result.current_step_type or "none",
                step_result.current_target or "none",
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
                "discrete_primitive",
            )
            if step_result.debug_image_path is not None:
                logger.info("[Debug] candidate_image=%s", step_result.debug_image_path)
            last_logged_policy_signature = policy_signature

        global_pose_trace.append((float(mapping_pose.x), float(mapping_pose.y)))
        if len(global_pose_trace) > 8000:
            del global_pose_trace[: len(global_pose_trace) - 8000]

        debug_update_period_sec = max(0.10, float(config.debug.image.update_period_sec))
        force_final_debug_frame = bool(step_result.route_complete and not route_complete_handled)
        should_save_target_debug = (
            bool(config.debug.image.enabled)
            and (
                force_final_debug_frame
                or target_keyframe_ready
                or (loop_now - last_nav_debug_save_time) >= debug_update_period_sec
            )
        )
        if should_save_target_debug:
            latest_global_map_image = _save_global_occupancy_debug(
                occupancy_snapshot=occupancy_grid.snapshot(),
                robot_pose=mapping_pose,
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
                map_image_path=None,
                horizontal_fov_deg=actual_camera_hfov_deg,
            )
            debug_video_recorder.append(debug_image)
            last_nav_debug_save_time = loop_now

        if step_result.route_complete and not route_complete_handled:
            route_complete_handled = True
            navigation_finished = True
            drive_controller.apply("STOP", speed_scale=0.0)
            last_action_name = "STOP"
            target_perception.close()
            debug_video_recorder.close()
            observer_video_recorder.close()
            _write_evaluation_result(
                output_path=evaluation_result_path,
                status="SUCCESS",
                task_id=task_id,
                config=config,
                instruction_context=instruction_context,
                step_result=step_result,
                sim_time_sec=sim_time_sec,
                failure_stage="",
                reason="all typed instruction steps completed",
            )
            logger.info("Route complete — navigation stopped; Isaac Sim scene remains open. Debug video saved: %s", debug_video_recorder.output_path)
            continue

        if (
            config.debug.image.depth_enabled
            and last_depth_image is not None
            and depth_debug_save_pending
            and (loop_now - last_depth_debug_save_time) >= debug_update_period_sec
        ):
            depth_debug = _render_depth_debug(depth_image=last_depth_image)
            _save_rgb_debug_image(depth_debug, depth_debug_output_path)
            depth_debug_save_pending = False
            last_depth_debug_save_time = loop_now

        if loop_now - last_motion_log_time >= float(config.logging.motion_log_period_sec):
            delta = position - last_logged_position
            joint_readback = drive_controller.readback()
            if config.robot.is_g1:
                command_velocity = joint_readback["command_velocity"]
                policy_action = joint_readback["policy_action"]
                joint_tracking = joint_readback["joint_tracking"]
                logger.info(
                    "[Motion:G1] policy_action=%s exec_action=%s token=%s "
                    "world_pos=[%.3f, %.3f, %.3f] odom=[%.3f, %.3f, %.1fdeg] "
                    "delta=[%.3f, %.3f, %.3f] command=[%.3f, %.3f, %.3f] "
                    "policy_abs_max=%.3f joint_error_max=%.3frad joint_error_mean=%.3frad",
                    step_result.action_name,
                    safety_decision.action_name,
                    active_command_token,
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                    float(observation.pose.x),
                    float(observation.pose.y),
                    float(np.degrees(observation.pose.yaw_rad)),
                    float(delta[0]),
                    float(delta[1]),
                    float(delta[2]),
                    float(command_velocity["linear_x_mps"]),
                    float(command_velocity["linear_y_mps"]),
                    float(command_velocity["angular_z_radps"]),
                    max((abs(float(value)) for value in policy_action.values()), default=0.0),
                    float(joint_tracking["max_error_rad"]),
                    float(joint_tracking["mean_error_rad"]),
                )
            else:
                logger.info(
                    "[Motion:AGX] policy_action=%s exec_action=%s token=%s mode=%s "
                    "world_pos=[%.3f, %.3f, %.3f] odom=[%.3f, %.3f, %.1fdeg] "
                    "delta=[%.3f, %.3f, %.3f] wheel_joint_target_rad_s=%.3f "
                    "steering_joint_target_rad=%.3f steering_actual=%s wheel_actual=%s",
                    step_result.action_name,
                    safety_decision.action_name,
                    active_command_token,
                    "discrete_primitive",
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

    target_perception.close()
    debug_video_recorder.close()
    observer_video_recorder.close()
    if not navigation_finished:
        _write_evaluation_result(
            output_path=evaluation_result_path,
            status="INTERRUPTED",
            task_id=task_id,
            config=config,
            instruction_context=instruction_context,
            step_result=active_step_result,
            sim_time_sec=sim_time_sec,
            failure_stage="simulation_closed",
            reason="simulation closed before route completion",
        )


def _resolve_evaluation_result_path(*, task_id: str) -> Path | None:
    configured_dir = str(os.getenv("VLN_EVAL_RESULTS_DIR", "")).strip()
    if not configured_dir:
        return None
    output_dir = Path(configured_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id).strip()) or "episode"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{safe_task_id}_{timestamp}.json"


def _write_evaluation_result(
    *,
    output_path: Path | None,
    status: str,
    task_id: str,
    config: AppConfig,
    instruction_context: InstructionContext | None,
    step_result: PolicyStepResult | None,
    sim_time_sec: float,
    failure_stage: str,
    reason: str,
) -> None:
    if output_path is None:
        return
    completed_count = (
        0
        if step_result is None
        else min(
            int(step_result.current_step_index)
            + int(bool(step_result.route_complete)),
            int(step_result.route_step_count),
        )
    )
    route_steps: list[dict] = []
    if instruction_context is not None:
        for index, step in enumerate(instruction_context.route_steps):
            if str(status).upper() == "SUCCESS" or index < completed_count:
                step_status = "COMPLETED"
            elif step_result is not None and index == int(step_result.current_step_index):
                step_status = "CURRENT"
            else:
                step_status = "PENDING"
            route_steps.append(
                {
                    "step_id": step.step_id,
                    "step_type": step.step_type,
                    "target": step.target,
                    "direction": step.direction,
                    "completion": step.completion,
                    "cues": [
                        {
                            "label": cue.label,
                            "role": cue.role,
                            "expected_side": cue.expected_side,
                        }
                        for cue in step.cues
                    ],
                    "status": step_status,
                }
            )
    payload = {
        "schema_version": "captainav-eval-v1",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": str(status).upper(),
        "failure_stage": str(failure_stage),
        "reason": str(reason),
        "task_id": str(task_id),
        "instruction_id": (
            "" if instruction_context is None else instruction_context.instruction_id
        ),
        "instruction": (
            str(config.instruction.text)
            if instruction_context is None
            else instruction_context.raw_text
        ),
        "scene_usd_path": str(config.scene.usd_path),
        "robot_profile": (
            "" if config.robot is None else str(config.robot.name)
        ),
        "sim_time_sec": round(float(sim_time_sec), 3),
        "current_step_index": (
            None if step_result is None else int(step_result.current_step_index)
        ),
        "current_step_id": (
            "" if step_result is None else str(step_result.current_step_id)
        ),
        "current_step_type": (
            "" if step_result is None else str(step_result.current_step_type)
        ),
        "completed_landmarks": (
            [] if step_result is None else list(step_result.completed_landmarks)
        ),
        "route_steps": route_steps,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


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
        semantic_role=(
            "PASS_CUE"
            if str(state.behavior_type).strip().lower() == "pass_cue"
            else "TURN_CUE"
            if str(state.behavior_type).strip().lower()
            in {"turn_cue", "turn_lookahead", "turn_at"}
            else "TRANSITION_CUE"
            if str(state.behavior_type).strip().lower() == "transition_cue"
            else "GOAL"
        ),
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
    add_reference_to_stage(usd_path, config.scene.prim_path)
    logger.info("Loaded environment USD: %s", usd_path)
    return True


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
        current_step_index=0 if previous_step_result is None else int(previous_step_result.current_step_index),
        route_step_count=0 if previous_step_result is None else int(previous_step_result.route_step_count),
        current_step_id="" if previous_step_result is None else str(previous_step_result.current_step_id),
        current_step_type="" if previous_step_result is None else str(previous_step_result.current_step_type),
        current_target="" if previous_step_result is None else str(previous_step_result.current_target),
        navigation_status="RECOVERY",
    )


def _record_completed_primitive(*, step_result: PolicyStepResult, stuck_detector: StuckDetector) -> None:
    if str(step_result.decision_source).strip().upper() != "PRIMITIVE_COMPLETE":
        return
    start = step_result.completed_primitive_start_pose
    end = step_result.completed_primitive_end_pose
    action = str(step_result.completed_primitive_action).strip().upper()
    if start is None or end is None or not action:
        return
    stuck_detector.record_completed_primitive(
        action_name=action,
        start_pose=Pose2D(x=float(start[0]), y=float(start[1]), yaw_rad=float(start[2])),
        end_pose=Pose2D(x=float(end[0]), y=float(end[1]), yaw_rad=float(end[2])),
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
    if config.simulation.headless:
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
            eye=np.asarray(config.scene.viewport_eye, dtype=np.float32),
            target=np.asarray(config.scene.viewport_target, dtype=np.float32),
            camera_prim_path="/OmniverseKit_Persp",
        )
        logger.info("Set GUI viewport camera using fallback viewport utils.")
        return

    ViewportManager.set_camera_view(
        "/OmniverseKit_Persp",
        eye=np.asarray(config.scene.viewport_eye, dtype=np.float32),
        target=np.asarray(config.scene.viewport_target, dtype=np.float32),
    )
    logger.info("Set GUI viewport camera to eye=%s target=%s", config.scene.viewport_eye, config.scene.viewport_target)


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
    if action not in {"FORWARD", "TURN_LEFT", "TURN_RIGHT"}:
        return False
    remaining = int(getattr(step_result, "execution_steps_remaining", 0) or 0)
    return remaining > 0


def _match_target_observation_to_occupancy(
    *,
    observation: TargetObservation,
    occupancy_grid: GlobalOccupancyMap,
    behavior_type: str,
    logger,
) -> TargetObservation:
    """Snap go-to geometry to a supported physical surface in the RGB-D map."""

    if str(behavior_type).strip().lower() != "go_to":
        return observation
    if (
        not bool(observation.visible)
        or str(observation.reason).strip().lower() != "ok"
        or not bool(getattr(observation, "request_evidence_valid", False))
        or observation.estimated_world_xy is None
        or observation.request_robot_pose is None
    ):
        return observation

    raw_world_xy = (
        list(observation.raw_estimated_world_xy)
        if observation.raw_estimated_world_xy is not None
        else list(observation.estimated_world_xy)
    )
    surface_match = occupancy_grid.match_landmark_surface(
        observer_pose=observation.request_robot_pose,
        raw_world_xy=raw_world_xy,
    )
    if not surface_match.matched or surface_match.world_xy is None:
        logger.info(
            "[LandmarkGeometry] target=%s matched=0 raw=%s reason=%s support=%s",
            observation.label,
            [round(float(value), 2) for value in raw_world_xy],
            surface_match.reason,
            int(surface_match.support_cells),
        )
        return replace(
            observation,
            raw_estimated_world_xy=raw_world_xy,
            surface_supported=False,
            surface_support_cells=int(surface_match.support_cells),
            surface_correction_m=float(surface_match.correction_m),
            surface_depth_m=None,
        )

    surface_world_xy = [
        float(surface_match.world_xy[0]),
        float(surface_match.world_xy[1]),
    ]
    logger.info(
        "[LandmarkGeometry] target=%s matched=1 raw=%s surface=%s "
        "raw_depth=%.2f surface_depth=%.2f correction=%.2f support=%s",
        observation.label,
        [round(float(value), 2) for value in raw_world_xy],
        [round(float(value), 2) for value in surface_world_xy],
        float(observation.depth_m or 0.0),
        float(surface_match.surface_distance_m),
        float(surface_match.correction_m),
        int(surface_match.support_cells),
    )
    return replace(
        observation,
        depth_m=float(surface_match.surface_distance_m),
        estimated_world_xy=surface_world_xy,
        source=f"{observation.source}:surface_matched",
        raw_estimated_world_xy=raw_world_xy,
        surface_supported=True,
        surface_support_cells=int(surface_match.support_cells),
        surface_correction_m=float(surface_match.correction_m),
        surface_depth_m=float(surface_match.surface_distance_m),
    )


def _is_transient_target_detection_observation(
    observation: TargetObservation | None,
) -> bool:
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

    fp_pixels = image_rgb[..., :3].copy()
    fp_image = Image.fromarray(fp_pixels)
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

    occ_width = max(540, int(fp_width))
    occ_image = _load_occupancy_debug_panel(
        occupancy_image=map_image,
        occupancy_image_path=map_image_path,
        width=occ_width,
        height=fp_height,
    )
    heading_deg = float(np.degrees(heading_rad))
    status_height = 92
    canvas = Image.new(
        "RGB",
        (occ_width + fp_width, status_height + fp_height),
        (16, 20, 25),
    )
    canvas.paste(occ_image, (0, status_height))
    canvas.paste(fp_image, (occ_width, status_height))
    status_draw = ImageDraw.Draw(canvas)
    status_font = _debug_font(16)
    status_lines = [
        (
            f"STEP {step_result.current_step_index + 1}/{max(1, step_result.route_step_count)}  "
            f"{step_result.current_step_type or 'COMPLETE'}  target={step_result.current_target or 'none'}"
        ),
        (
            f"ACTION policy={step_result.action_name} executed={executed_action}  "
            f"source={step_result.decision_source} candidate={step_result.selected_candidate_id}"
        ),
        (
            f"LANDMARK current={step_result.current_landmark or 'none'} "
            f"completed={step_result.completed_landmarks or ['none']}  "
            f"goal=({local_forward:.2f}m forward, {local_left:.2f}m left, {heading_deg:+.1f}deg)"
        ),
    ]
    for index, line in enumerate(status_lines):
        status_draw.text((14, 8 + index * 27), line, fill=(235, 240, 245), font=status_font)

    _save_image_atomic(canvas, output_path)
    return canvas


def _draw_grounded_target_overlay(
    *,
    draw: ImageDraw.ImageDraw,
    observation: TargetObservation | None,
    image_size: tuple[int, int],
) -> None:
    width, height = [int(value) for value in image_size]
    label_font = _debug_font(15)
    if observation is None:
        draw.text((12, height - 30), "perception: no current landmark", fill=(255, 210, 90), font=label_font)
        return
    visible = bool(observation.visible)
    reason = str(observation.reason)
    source = str(observation.source)
    depth_region_source = str(getattr(observation, "depth_region_source", "none"))
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
                f"can_confirm={int(can_confirm)} {verify_text} depth_region={depth_region_source}"
            )
            draw.rectangle((xmin, max(0, ymin - 28), min(width, xmin + 620), ymin), fill=(0, 0, 0))
            draw.text((xmin + 4, max(0, ymin - 24)), label[:110], fill=color, font=label_font)
        else:
            stale_color = (255, 160, 60)
            age_text = f"{float(observation.result_age_sec):.1f}s" if observation.result_age_sec > 0.0 else "?"
            stale_text = (
                f"STALE perception: {observation.label} age={age_text} "
                f"depth={depth_text} conf={observation.confidence:.2f} "
                f"request_valid={int(bool(getattr(observation, 'request_evidence_valid', False)))} "
                f"frame={observation.request_frame_index}"
            )
            draw.rectangle((0, height - 44, width, height), fill=(0, 0, 0))
            draw.text((12, height - 32), stale_text[:120], fill=stale_color, font=label_font)
        return
    text = (
        f"perception: lost/hidden label={observation.label} reason={reason} "
        f"source={source} depth_region={depth_region_source} depth={depth_text}"
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
    if config.scene.usd_path:
        candidate = Path(str(config.scene.usd_path))
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


def _resolve_existing_prim_path(*, stage, prim_path: str, kind: str, strict: bool = False) -> str:
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        if "camera" in str(kind).lower() and prim.GetTypeName() != "Camera":
            raise RuntimeError(
                f"Configured {kind} prim has type {prim.GetTypeName()!r}, expected 'Camera': {prim_path}"
            )
        return prim_path
    if strict:
        raise RuntimeError(f"Required {kind} prim does not exist at the configured path: {prim_path}")

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


def _get_joint_names(robot) -> list[str]:
    try:
        names = robot.dof_names
    except Exception as exc:
        raise RuntimeError(f"Unable to query articulation DOF names: {exc}") from exc
    return [str(name) for name in names]


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

    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    font_title = _debug_font(22)
    font_label = _debug_font(15)
    font_small = _debug_font(12)

    def world_to_px(x_m: float, y_m: float) -> tuple[int, int]:
        px = int(round(map_x0 + (float(x_m) - min_x) * scale))
        py = int(round(map_y0 + (max_y - float(y_m)) * scale))
        return px, py

    draw.text(
        (12, 10),
        "NORTH-UP RGB-D GLOBAL OCCUPANCY MAP",
        fill=(245, 245, 245),
        font=font_title,
    )
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
        passed = bool(getattr(entry, "passed", False))
        evidence = str(getattr(entry, "evidence_level", "")).strip().lower()
        confirmed = bool(getattr(entry, "confirmed", False)) or evidence == "confirmed"
        label = str(getattr(entry, "label", "landmark"))
        if reached or passed:
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
        role = str(getattr(entry, "semantic_role", "GOAL"))
        lifecycle = (
            "reached"
            if bool(getattr(entry, "reached", False))
            else "passed"
            if bool(getattr(entry, "passed", False))
            else status
        )
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
            f"  {marker}{label}: role={role} {direction} {dist_str} "
            f"[{lifecycle}] {confidence}"
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
