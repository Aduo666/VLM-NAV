from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..control.controller import (
    SteeringDriveController,
    resolve_front_steering_joint_groups,
    resolve_hold_joint_names,
    resolve_wheel_joint_groups,
)
from ..navigation.candidate_waypoint_predictor import CandidateWaypointPredictor, CandidateWaypointPredictorConfig
from ..navigation.depth_safety import DepthSafetyConfig, DepthSafetyMonitor
from ..navigation.instruction_manager import InstructionManager
from ..navigation.landmark_map import LandmarkMap, LandmarkMapConfig
from ..navigation.landmark_tracker import LandmarkTracker, LandmarkTrackerConfig
from ..navigation.navigation_map_visualizer import NavigationMapConfig, NavigationMapVisualizer
from ..navigation.occupancy_grid import OccupancyGridConfig, OccupancyGridMapper
from ..navigation.observation_adapter import ObservationAdapter
from ..navigation.odometry import SimOdometrySensor
from ..navigation.open_nav_client import OpenNavLLMClient
from ..navigation.progress_manager import ProgressManager
from ..navigation.spatial_transform import log_spatial_transform_self_check
from ..navigation.stuck_detector import StuckDetector, StuckDetectorConfig
from ..navigation.target_grounding import TargetGrounder, TargetGroundingConfig, TargetObservation
from ..navigation.vln_policy import ExecutionFeedback, OpenNavPolicyConfig, OpenNavVLNPolicy, PolicyStepResult
from ..navigation.vln_reasoner import OpenNavReasoner
from ..runtime.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, AppConfig, load_config, setup_logging
from ..runtime.debug_video import DebugVideoRecorder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isaac Sim Open-Nav style VLN runtime")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path")
    parser.add_argument("--headless", action="store_true", help="Force headless simulation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.headless:
        config.headless = True

    logger = setup_logging(config.log_level)
    logger.info("Loaded config from %s", Path(args.config).resolve())
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

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": config.headless,
            "width": config.sim_width,
            "height": config.sim_height,
            "active_gpu": config.active_gpu,
            "physics_gpu": config.physics_gpu,
            "multi_gpu": config.multi_gpu,
        }
    )

    try:
        run_navigation(simulation_app=simulation_app, config=config)
    except Exception:
        logger.exception("Navigation loop crashed.")
        return 1
    finally:
        simulation_app.close()

    return 0


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
    instruction_manager = InstructionManager(llm_client=llm_client, logger=logger)
    instruction_context = instruction_manager.parse(config.instruction)
    logger.info("[Goal] instruction=%s", instruction_context.raw_text)
    logger.info("[Goal] actions=%s", instruction_context.actions)
    logger.info("[Goal] landmarks=%s", instruction_context.landmarks)
    logger.info(
        "[Route] current_landmark=%s ordered_landmarks=%s arrival_distance_m=%.2f tracker_min_visible=%s recovery_enabled=%s recovery_stalled_steps=%s",
        instruction_context.goal_phrase or "none",
        instruction_context.landmarks,
        float(config.landmark_arrival_distance_m),
        int(config.landmark_tracker_min_visible_count),
        int(config.recovery_enabled),
        int(config.recovery_trigger_stalled_steps),
    )
    logger.info(
        "[GroundingConfig] enabled=%s backend=%s call_period_sec=%.2f reuse_sec=%.2f min_conf=%.2f depth_range=[%.2f, %.2f] coord=%s depth_region=%s min_depth_valid_ratio=%.2f dino_checkpoint=%s sam_checkpoint=%s sam_model=%s",
        int(bool(config.target_grounding_enabled)),
        str(config.target_grounding_backend),
        float(config.target_grounding_call_period_sec),
        float(config.target_grounding_reuse_last_observation_sec),
        float(config.target_grounding_min_confidence),
        float(config.target_grounding_min_depth_m),
        float(config.target_grounding_max_depth_m),
        str(config.target_grounding_coordinate_mode),
        str(config.target_grounding_depth_region),
        float(config.target_grounding_min_depth_valid_ratio),
        str(config.target_grounding_dino_checkpoint_path),
        str(config.target_grounding_sam_checkpoint_path),
        str(config.target_grounding_sam_model_type),
    )
    logger.info(
        "[LandmarkTrackerConfig] visible_depth=%.2f anchor_radius=%.2f min_visible=%s confirmations=%s landmark_map=%s map_ttl=%.1f reacquire_guard=%s recent_visible_sec=%.1f override=%s",
        float(config.landmark_tracker_arrival_visible_depth_m),
        float(config.landmark_tracker_arrival_anchor_radius_m),
        int(config.landmark_tracker_min_visible_count),
        int(config.landmark_tracker_arrival_confirmations),
        int(bool(config.landmark_map_enabled)),
        float(config.landmark_map_anchor_ttl_sec),
        int(bool(config.reacquire_guard_enabled)),
        float(config.reacquire_guard_recent_visible_sec),
        str(config.reacquire_guard_override_action),
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
            recovery_backward_sec=float(config.stuck_recovery_backward_sec),
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
        ),
        llm_client=target_grounding_llm_client,
        logger=logger,
    )
    landmark_tracker = LandmarkTracker(
        config=LandmarkTrackerConfig(
            arrival_visible_depth_m=float(config.landmark_tracker_arrival_visible_depth_m),
            arrival_anchor_radius_m=float(config.landmark_tracker_arrival_anchor_radius_m),
            min_visible_count=int(config.landmark_tracker_min_visible_count),
            arrival_confirmations=int(config.landmark_tracker_arrival_confirmations),
            max_lost_count=int(config.landmark_tracker_max_lost_count),
            anchor_update_min_consistent_observations=int(
                config.landmark_tracker_anchor_update_min_consistent_observations
            ),
            anchor_update_max_jump_m=float(config.landmark_tracker_anchor_update_max_jump_m),
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
            active_backward_duration_sec=float(config.action_backward_duration_sec),
            active_turn_duration_sec=float(config.action_turn_duration_sec),
            active_turn_angle_deg=float(config.action_turn_angle_deg),
            active_forward_distance_m=float(config.action_forward_distance_m),
            active_backward_distance_m=float(config.action_backward_distance_m),
            active_distance_tolerance_m=float(config.action_distance_tolerance_m),
            active_turn_tolerance_deg=float(config.action_turn_tolerance_deg),
            active_odom_completion_enabled=bool(config.action_odom_completion_enabled),
            active_track_max_yaw_deg=float(config.active_track_max_yaw_deg),
            active_track_min_duration_sec=float(config.active_track_min_duration_sec),
            use_discrete_action_primitives=bool(config.use_discrete_action_primitives),
            landmark_arrival_confirmations=int(config.landmark_arrival_confirmations),
            reacquire_guard_enabled=bool(config.reacquire_guard_enabled),
            reacquire_guard_recent_visible_sec=float(config.reacquire_guard_recent_visible_sec),
            reacquire_guard_override_action=str(config.reacquire_guard_override_action),
            landmark_arrival_distance_m=float(config.landmark_arrival_distance_m),
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
        "[OdomConfig] enabled=%s source=%s primitive_completion=%s forward_distance=%.2fm backward_distance=%.2fm turn_angle=%.1fdeg tolerance=%.1fdeg",
        int(bool(config.odom_enabled)),
        str(config.odom_source),
        int(bool(config.action_odom_completion_enabled)),
        float(config.action_forward_distance_m),
        float(config.action_backward_distance_m),
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
    if config.debug_video_enabled and config.target_debug_enabled:
        logger.info("Runtime debug video output: %s", target_debug_video_path)
    if config.depth_debug_enabled:
        logger.info("Depth debug output: %s", depth_debug_output_path)
    if config.navigation_map_enabled:
        logger.info("Navigation map output: %s", navigation_map_visualizer.output_path)
    if config.occupancy_grid_enabled:
        logger.info(
            "[OccupancyGridConfig] enabled=1 resolution=%.2fm size=%.1fm inflation=%.2fm min_neighbors=%s path_corridor=%.2fm roi=%s debug=%s",
            float(config.occupancy_grid_resolution_m),
            float(config.occupancy_grid_size_m),
            float(config.occupancy_grid_inflation_radius_m),
            int(config.occupancy_grid_occupied_min_neighbors),
            float(config.occupancy_grid_path_corridor_radius_m),
            config.occupancy_grid_depth_roi_vertical_range,
            occupancy_grid.output_path,
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

    for _ in range(config.warmup_steps):
        world.step(render=True)
        sim_time_sec += float(config.physics_dt)
        drive_controller.apply("STOP", speed_scale=0.0)
    logger.info("Warmup complete, controller is armed.")

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
        occupancy_grid.update(observation=observation)

        current_landmark = policy.current_landmark
        target_observation: TargetObservation | None = None
        if current_landmark:
            target_observation = target_grounder.observe(
                rgb=front_rgb,
                depth=last_depth_image,
                target_label=current_landmark,
                robot_pose=observation.pose,
                timestamp_sec=sim_time_sec,
            )
            landmark_state = landmark_tracker.update(
                target_label=current_landmark,
                behavior_type="go_to",
                observation=target_observation,
                robot_pose=observation.pose,
                timestamp_sec=sim_time_sec,
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
        safety_decision = depth_safety_monitor.apply(
            observation=observation,
            action_name=step_result.action_name,
            speed_scale=step_result.speed_scale,
            local_goal_xy=step_result.local_goal_xy,
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

        debug_update_period_sec = max(0.10, float(config.debug_image_update_period_sec))
        should_save_target_debug = (
            bool(config.target_debug_enabled)
            and not bool(step_result.replanned)
            and (loop_now - last_nav_debug_save_time) >= debug_update_period_sec
        )
        if should_save_target_debug:
            debug_image = _save_current_target_debug_image(
                rgb=front_rgb,
                step_result=step_result,
                target_observation=target_observation,
                executed_action=safety_decision.action_name,
                output_path=target_debug_output_path,
                map_image_path=(
                    navigation_map_visualizer.output_path
                    if bool(config.navigation_map_enabled)
                    else occupancy_grid.output_path if bool(config.occupancy_grid_enabled) else None
                ),
                horizontal_fov_deg=float(config.camera_horizontal_fov_deg),
            )
            debug_video_recorder.append(debug_image)
            last_nav_debug_save_time = loop_now

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
        fp_draw.text((target_u + 18, max(0, target_v - 12)), "current target", fill=color)
    else:
        color = (255, 80, 80)
        fp_draw.line((origin[0] - 35, origin[1], origin[0] + 35, origin[1]), fill=color, width=4)
        fp_draw.text((origin[0] + 42, origin[1] - 10), "no active target", fill=color)

    heading_deg = float(np.degrees(heading_rad))
    overlay_lines = [
        f"candidate={step_result.selected_candidate_id} source={step_result.decision_source}",
        f"landmark={step_result.current_landmark or 'none'} completed={step_result.completed_landmarks}",
        f"policy={step_result.action_name} executed={executed_action}",
        f"local_goal forward={local_forward:.2f}m left={local_left:.2f}m",
        f"distance={distance_m:.2f}m heading={heading_deg:+.1f}deg",
    ]
    margin = 10
    line_height = 18
    box_height = margin * 2 + line_height * len(overlay_lines)
    fp_draw.rectangle((0, 0, fp_width, box_height), fill=(0, 0, 0))
    for index, text in enumerate(overlay_lines):
        fp_draw.text((margin, margin + index * line_height), text, fill=(255, 255, 255))

    occ_width = max(360, int(fp_height * 0.95))
    occ_image = _load_occupancy_debug_panel(
        occupancy_image_path=map_image_path,
        width=occ_width,
        height=fp_height,
    )
    canvas = Image.new("RGB", (fp_width + occ_width, fp_height), (0, 0, 0))
    canvas.paste(fp_image, (0, 0))
    canvas.paste(occ_image, (fp_width, 0))

    _save_image_atomic(canvas, output_path)
    return canvas


def _overlay_target_mask(*, image: Image.Image, observation: TargetObservation | None) -> None:
    if observation is None or observation.mask is None:
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
    if observation is None:
        draw.text((12, height - 28), "grounding: no current landmark", fill=(255, 210, 90))
        return
    visible = bool(observation.visible)
    reason = str(observation.reason)
    source = str(observation.source)
    depth_text = "none" if observation.depth_m is None else f"{float(observation.depth_m):.2f}m"
    if visible and observation.bbox is not None:
        xmin, ymin, xmax, ymax = [int(value) for value in observation.bbox]
        xmin = max(0, min(width - 1, xmin))
        ymin = max(0, min(height - 1, ymin))
        xmax = max(0, min(width, xmax))
        ymax = max(0, min(height, ymax))
        color = (70, 255, 120)
        draw.rectangle((xmin, ymin, xmax, ymax), outline=color, width=4)
        if observation.center_uv is not None:
            center_u, center_v = [int(round(float(value))) for value in observation.center_uv]
            draw.line((center_u - 10, center_v, center_u + 10, center_v), fill=color, width=3)
            draw.line((center_u, center_v - 10, center_u, center_v + 10), fill=color, width=3)
        label = f"grounded {observation.label} depth={depth_text} conf={observation.confidence:.2f}"
        draw.rectangle((xmin, max(0, ymin - 24), min(width, xmin + 420), ymin), fill=(0, 0, 0))
        draw.text((xmin + 4, max(0, ymin - 20)), label, fill=color)
        return
    text = f"grounding: lost/hidden label={observation.label} reason={reason} source={source} depth={depth_text}"
    draw.rectangle((0, height - 38, width, height), fill=(0, 0, 0))
    draw.text((12, height - 26), text[:120], fill=(255, 110, 90))


def _load_occupancy_debug_panel(*, occupancy_image_path: Path | None, width: int, height: int) -> Image.Image:
    if occupancy_image_path is not None and Path(occupancy_image_path).exists():
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
    draw.text((12, 10), "NAV MAP / OCCUPANCY / LANDMARKS", fill=(245, 245, 245))
    return panel


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

    raise RuntimeError(f"Configured {kind} prim does not exist: {prim_path}")


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


def _format_joint_map(values: dict[str, float]) -> str:
    ordered_items = sorted(values.items(), key=lambda item: item[0])
    return ",".join(f"{name}={float(value):+.3f}" for name, value in ordered_items)


if __name__ == "__main__":
    raise SystemExit(main())
