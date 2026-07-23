from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from vln_nav.runtime.robot_profile import RobotProfile, load_robot_profile


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
@dataclass
class AppConfig:
    llm_api_key: str = ""
    model_name: str = "Qwen/Qwen3-VL-235B-A22B-Instruct"
    api_base_url: str = "https://api.siliconflow.cn/v1"
    llm_timeout_sec: int = 30
    llm_retry_cooldown_sec: float = 0.0
    llm_upload_max_edge_px: int = 512
    camera_image_detail: str = "low"
    instruction: str = ""
    instruction_parse_max_tokens: int = 2048
    instruction_parse_timeout_sec: int = 90
    instruction_cache_enabled: bool = True
    instruction_cache_path: str = str(PROJECT_ROOT / ".cache" / "instruction_plans.json")
    region_transition_min_displacement_m: float = 1.0
    region_transition_confirmations: int = 2
    region_transition_min_confidence: float = 0.55
    semantic_turn_min_yaw_deg: float = 45.0
    semantic_turn_max_yaw_deg: float = 105.0
    semantic_turn_max_primitives: int = 7
    semantic_turn_min_forward_clearance_m: float = 0.60
    semantic_turn_alignment_confirmations: int = 2
    semantic_turn_corridor_center_tolerance_deg: float = 10.0
    semantic_turn_same_corridor_tolerance_deg: float = 15.0
    semantic_turn_target_high_confidence: float = 0.75
    semantic_turn_target_medium_confidence: float = 0.50
    semantic_turn_direction_tolerance_deg: float = 10.0
    semantic_turn_corridor_half_fov_deg: float = 60.0
    semantic_turn_corridor_step_deg: float = 5.0
    semantic_turn_lidar_bin_half_width_deg: float = 3.0
    semantic_turn_min_corridor_width_deg: float = 15.0
    semantic_turn_corridor_max_clearance_m: float = 4.0
    semantic_turn_max_recovery_attempts: int = 2
    local_detour_max_turn_primitives: int = 2
    traverse_heading_correction_limit_deg: float = 30.0
    traverse_candidate_heading_tolerance_deg: float = 22.0
    traverse_min_progress_m: float = 0.50
    traverse_pass_margin_m: float = 0.20
    traverse_review_min_progress_m: float = 1.00
    traverse_review_min_translation_m: float = 0.20
    traverse_max_recovery_attempts: int = 2
    visual_target_center_deadband_deg: float = 15.0
    candidate_slot_bearings_deg: list[float] = field(default_factory=lambda: [75.0, 35.0, 0.0, -35.0, -75.0])
    candidate_adaptive_enabled: bool = True
    candidate_adaptive_count: int = 5
    candidate_adaptive_min_spacing_deg: float = 18.0
    candidate_adaptive_smoothing_px: int = 15
    candidate_min_travel_distance_m: float = 0.55
    candidate_max_travel_distance_m: float = 1.35
    candidate_min_free_ratio: float = 0.28
    candidate_column_half_width_px: int = 26
    target_debug_enabled: bool = True
    target_debug_image_path: str = "debug/vln_debug.png"
    debug_video_enabled: bool = True
    debug_video_path: str = "debug/videos/vln_debug.mp4"
    debug_video_fps: float = 4.0
    debug_image_update_period_sec: float = 1.0
    progress_history_length: int = 6
    progress_stall_distance_m: float = 0.08
    progress_spin_yaw_deg: float = 85.0
    policy_execution_steps: int = 24
    arrival_distance_m: float = 1.4
    success_distance_m: float = 2.0
    landmark_arrival_confirmations: int = 2
    target_grounding_enabled: bool = True
    target_grounding_backend: str = "grounding_dino"
    target_grounding_call_period_sec: float = 1.5
    target_grounding_reuse_last_observation_sec: float = 2.5
    target_grounding_upload_max_edge_px: int = 512
    target_grounding_min_bbox_area_ratio: float = 0.002
    target_grounding_max_bbox_area_ratio: float = 0.60
    target_grounding_min_bbox_width_px: int = 4
    target_grounding_min_bbox_height_px: int = 4
    target_grounding_max_bbox_aspect_ratio: float = 8.0
    target_grounding_min_depth_valid_ratio: float = 0.12
    target_grounding_depth_statistic: str = "median"
    target_grounding_center_patch_ratio: float = 0.25
    target_grounding_min_depth_m: float = 0.20
    target_grounding_max_depth_m: float = 10.0
    target_grounding_min_confidence: float = 0.50
    target_grounding_coordinate_mode: str = "auto"
    target_grounding_allow_normalized_bbox: bool = True
    target_grounding_normalized_bbox_scale: float = 1000.0
    target_grounding_depth_region: str = "bbox_depth_mask"
    target_grounding_depth_mask_tolerance_ratio: float = 0.35
    target_grounding_depth_mask_min_pixels: int = 16
    target_grounding_dino_repo_path: str = str(PROJECT_ROOT / "third_party" / "GroundingDINO")
    target_grounding_dino_config_path: str = str(
        PROJECT_ROOT / "weights" / "grounded_sam" / "GroundingDINO_SwinT_OGC.py"
    )
    target_grounding_dino_checkpoint_path: str = str(
        PROJECT_ROOT / "weights" / "grounded_sam" / "groundingdino_swint_ogc.pth"
    )
    target_grounding_dino_device: str = "cuda"
    target_grounding_dino_box_threshold: float = 0.45
    target_grounding_dino_text_threshold: float = 0.25
    target_grounding_dino_max_detections: int = 5
    target_grounding_dino_nms_iou_threshold: float = 0.50
    target_grounding_dino_text_encoder_path: str = str(
        PROJECT_ROOT / "weights" / "grounded_sam" / "bert-base-uncased"
    )
    target_grounding_sam_checkpoint_path: str = str(PROJECT_ROOT / "weights" / "grounded_sam" / "sam_vit_h_4b8939.pth")
    target_grounding_sam_model_type: str = "vit_h"
    target_grounding_sam_device: str = "cuda"
    target_grounding_sam_min_mask_area_ratio: float = 0.0005
    target_grounding_sam_enabled: bool = True
    target_grounding_max_result_age_sec: float = 30.0
    target_grounding_max_result_pose_delta_m: float = 0.05
    target_grounding_max_result_yaw_delta_deg: float = 3.0
    target_grounding_semantic_verification_enabled: bool = True
    target_grounding_semantic_verification_min_confidence: float = 0.70
    target_grounding_semantic_verification_max_edge_px: int = 768
    target_grounding_semantic_verification_auto_accept_confidence: float = 0.75
    target_grounding_pause_during_active_primitive: bool = True
    target_tracker_enabled: bool = True
    target_tracker_type: str = "CSRT"
    target_tracker_max_lost_frames: int = 8
    target_tracker_lost_confirmed_sec: float = 2.0
    target_tracker_min_depth_valid_ratio: float = 0.08
    target_tracker_center_patch_ratio: float = 0.35
    target_tracker_confidence_decay_per_lost_frame: float = 0.06
    target_tracker_min_tracking_confidence: float = 0.35
    landmark_tracker_arrival_visible_depth_m: float = 1.00
    landmark_tracker_arrival_anchor_radius_m: float = 1.00
    landmark_tracker_min_visible_count: int = 2
    landmark_tracker_arrival_confirmations: int = 2
    landmark_tracker_arrival_window_size: int = 5
    landmark_tracker_arrival_recent_visible_sec: float = 3.0
    landmark_tracker_max_lost_count: int = 6
    landmark_tracker_anchor_update_min_consistent_observations: int = 2
    landmark_tracker_anchor_update_max_jump_m: float = 1.5
    landmark_tracker_confirm_confidence_threshold: float = 0.50
    landmark_tracker_potential_confidence_threshold: float = 0.50
    landmark_tracker_confirm_min_observations: int = 3
    landmark_tracker_confirm_max_anchor_jump_m: float = 0.50
    landmark_tracker_confirm_max_depth_jump_m: float = 0.80
    landmark_tracker_low_confidence_min_translation_m: float = 0.20
    landmark_tracker_low_confidence_min_yaw_deg: float = 10.0
    landmark_map_enabled: bool = True
    landmark_map_anchor_ttl_sec: float = 120.0
    landmark_map_min_confidence: float = 0.20
    landmark_map_log_period_sec: float = 2.0
    depth_debug_enabled: bool = True
    depth_debug_image_path: str = "debug/depth_debug.png"

    headless: bool = False
    sim_width: int = 1280
    sim_height: int = 720
    physics_dt: float = 1.0 / 60.0
    rendering_dt: float = 1.0 / 60.0
    warmup_steps: int = 30
    active_gpu: int | None = None
    physics_gpu: int | None = 0
    multi_gpu: bool = False

    robot_profile_name: str = "agx"
    robot_type: str = "agx_4ws"
    robot_profile: RobotProfile | None = None
    robot_policy_decimation: int = 1
    robot_usd_path: Path = Path()
    robot_prim_path: str = "/World/Robot"
    robot_articulation_path: str = ""
    robot_init_position: list[float] = field(default_factory=lambda: [1.5, 4.0, 0.22])
    robot_init_orientation_deg: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    camera_width: int = 640
    camera_height: int = 480
    front_camera_prim_path: str = "link6/Camera"
    observer_camera_enabled: bool = True
    observer_camera_prim_path: str = "dummy_base_link/observecamera"
    observer_video_enabled: bool = True
    observer_video_path: str = "debug/local/observevideos/observer.mp4"
    observer_video_fps: float = 10.0
    follow_front_camera_in_gui: bool = False
    follow_observer_camera_in_gui: bool = True
    sensor_frequency_hz: int = 15
    camera_horizontal_fov_deg: float = 90.0
    interactive_confirmation_enabled: bool = False
    roi_vertical_range: list[float] = field(default_factory=lambda: [0.45, 0.85])
    clearance_threshold_m: float = 1.2
    valid_depth_range_m: list[float] = field(default_factory=lambda: [0.2, 8.0])
    obstacle_avoidance_enabled: bool = True
    obstacle_stop_distance_m: float = 0.70
    obstacle_roi_vertical_range: list[float] = field(default_factory=lambda: [0.55, 0.92])
    obstacle_center_width_ratio: float = 0.28
    obstacle_side_width_ratio: float = 0.24
    near_field_memory_enabled: bool = True
    near_field_memory_sec: float = 2.0
    candidate_preferred_distance_m: float = 1.0
    waypoint_heading_threshold_rad: float = 0.3
    waypoint_stop_distance_m: float = 0.45
    waypoint_turn_speed_scale: float = 1.0
    waypoint_forward_speed_scale: float = 1.0
    replanning_cooldown_sec: float = 0.5
    depth_eval_period_sec: float = 0.12
    wheel_drive_velocity: float = 14.0
    turn_drive_velocity: float = 6.0
    steering_angle: float = 0.65
    turn_mode: str = "spin_4ws"
    turn_steering_angle: float = 0.50
    rear_steering_ratio: float = 1.0
    track_min_speed_scale: float = 0.45
    track_max_speed_scale: float = 1.0
    track_heading_gain: float = 1.15
    track_slow_heading_rad: float = 1.35
    track_smoothing_alpha: float = 0.92
    track_steering_sign: float = -1.0
    action_forward_duration_sec: float = 0.8
    action_turn_duration_sec: float = 0.30
    action_turn_angle_deg: float = 15.0
    action_forward_distance_m: float = 0.55
    action_distance_tolerance_m: float = 0.04
    action_turn_tolerance_deg: float = 1.5
    action_settle_max_wait_sec: float = 0.0
    action_settle_linear_speed_threshold_mps: float = 0.0
    action_settle_angular_speed_threshold_radps: float = 0.0
    action_settle_stable_duration_sec: float = 0.0
    action_odom_completion_enabled: bool = True
    active_track_max_yaw_deg: float = 10.0
    active_track_min_duration_sec: float = 0.45
    use_discrete_action_primitives: bool = True
    odom_enabled: bool = True
    odom_source: str = "isaac_world_pose"
    navigation_map_enabled: bool = False
    navigation_map_image_path: str = "assets/navigation_map.png"
    navigation_map_update_period_sec: float = 0.5
    navigation_map_width: int = 900
    navigation_map_height: int = 900
    navigation_map_depth_sample_stride_px: int = 16
    navigation_map_max_depth_obstacle_points: int = 2500
    topology_enabled: bool = True
    topology_node_spacing_m: float = 1.0
    topology_turn_threshold_deg: float = 40.0
    topology_merge_radius_m: float = 0.75
    topology_landmark_bind_radius_m: float = 1.50
    topology_max_nodes: int = 128
    topology_recent_node_count: int = 8
    occupancy_grid_enabled: bool = True
    occupancy_grid_resolution_m: float = 0.08
    occupancy_grid_size_m: float = 80.0
    occupancy_grid_update_period_sec: float = 0.12
    occupancy_grid_depth_sample_stride_px: int = 10
    occupancy_grid_depth_roi_vertical_range: list[float] = field(default_factory=lambda: [0.38, 0.72])
    occupancy_grid_min_depth_m: float = 0.20
    occupancy_grid_max_depth_m: float = 10.0
    occupancy_grid_inflation_radius_m: float = 0.26
    occupancy_grid_occupied_min_neighbors: int = 3
    occupancy_grid_robot_radius_m: float = 0.32
    occupancy_grid_path_corridor_radius_m: float = 0.30
    occupancy_grid_turn_clearance_radius_m: float = 0.36
    occupancy_grid_unknown_is_traversable: bool = False
    occupancy_grid_unknown_cell_cost: float = 0.80
    occupancy_grid_lidar_bootstrap_sec: float = 0.8
    occupancy_grid_lidar_bootstrap_min_fov_deg: float = 165.0
    occupancy_grid_lidar_bootstrap_max_wait_sec: float = 3.0
    occupancy_grid_semantic_planner_enabled: bool = True
    occupancy_grid_semantic_goal_step_m: float = 0.90
    occupancy_grid_semantic_goal_min_step_m: float = 0.45
    occupancy_grid_semantic_goal_lateral_search_m: float = 0.80
    map_candidates_enabled: bool = True
    map_candidates_max_candidates: int = 3
    map_candidates_min_distance_m: float = 0.55
    map_candidates_max_distance_m: float = 1.35
    map_candidates_semantic_enabled: bool = True
    map_candidates_frontier_enabled: bool = True
    map_candidates_escape_enabled: bool = True
    map_candidates_min_clearance_m: float = 0.34
    map_candidates_unknown_gain_weight: float = 0.35
    map_candidates_semantic_weight: float = 0.45
    lidar_enabled: bool = True
    lidar_prim_path: str = "dummy_base_link/base_link/Lidar_link"
    lidar_child_name: str = "lidar_sensor"
    lidar_horizontal_fov_deg: float = 360.0
    lidar_vertical_fov_deg: float = 10.0
    lidar_horizontal_resolution_deg: float = 1.0
    lidar_vertical_resolution_deg: float = 5.0
    lidar_rotation_rate_hz: float = 10.0
    lidar_min_range_m: float = 0.20
    lidar_max_range_m: float = 6.0
    lidar_yaw_offset_deg: float = 0.0
    lidar_sector_merge_keep_sec: float = 1.5
    lidar_draw_points: bool = False
    lidar_draw_lines: bool = False
    lidar_mode: str = "single"
    lidar_multi_sector_count: int = 3
    lidar_multi_sector_fov_deg: float = 60.0
    lidar_multi_sector_yaw_offsets_deg: list[float] = field(default_factory=lambda: [-60.0, 0.0, 60.0])
    lidar_multi_child_names: list[str] = field(default_factory=lambda: ["LidarLeft", "LidarCenter", "LidarRight"])
    stuck_detection_enabled: bool = True
    stuck_window_sec: float = 2.0
    stuck_displacement_threshold_m: float = 0.05
    stuck_yaw_threshold_deg: float = 5.0
    stuck_recovery_pause_sec: float = 0.5
    stuck_recovery_reverse_sec: float = 0.65
    stuck_recovery_turn_sec: float = 0.35
    stuck_recovery_turn_action: str = "TURN_RIGHT"
    wheel_drive_damping: float = 10000.0
    steering_stiffness: float = 10000.0
    steering_damping: float = 1000.0
    hold_joint_stiffness: float = 20000.0
    hold_joint_damping: float = 2000.0

    left_wheel_joint_names: list[str] = field(default_factory=list)
    right_wheel_joint_names: list[str] = field(default_factory=list)
    steering_joint_names: list[str] = field(default_factory=list)

    scene_usd_path: str = "usd/hospital.usd"
    scene_prim_path: str = "/World/Environment"
    add_default_ground_plane: bool = False
    add_dome_light: bool = False
    stage_load_timeout_sec: float = 120.0
    viewport_eye: list[float] = field(default_factory=lambda: [7.0, -2.0, 4.0])
    viewport_target: list[float] = field(default_factory=lambda: [1.5, 4.0, 1.0])
    tip_stop_enabled: bool = False
    tip_stop_roll_pitch_deg: float = 28.0
    tip_stop_min_base_height_m: float = 0.0
    tip_stop_confirmation_sec: float = 0.0

    log_level: str = "INFO"
    motion_log_period_sec: float = 5.0


def setup_logging(level: str) -> logging.Logger:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    if not getattr(root_logger, "_vln_root_initialized", False):
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            stream=sys.stdout,
        )
        root_logger._vln_root_initialized = True  # type: ignore[attr-defined]

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("vln")
    logger.setLevel(numeric_level)
    logger.propagate = False
    if getattr(logger, "_vln_handlers_initialized", False):
        for handler in logger.handlers:
            handler.setLevel(numeric_level)
        return logger
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_NavigationConsoleFilter())
    logger.addHandler(console_handler)

    log_path = PROJECT_ROOT / "logs" / "vln.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger._vln_handlers_initialized = True  # type: ignore[attr-defined]

    for noisy_logger in (
        "asyncio",
        "PIL",
        "matplotlib",
        "matplotlib.font_manager",
        "fontTools",
        "transformers",
        "torch",
        "timm",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    return logger


class _NavigationConsoleFilter(logging.Filter):
    _TAGS = (
        "[PLAN]",
        "[NAV_STATE]",
        "[LANDMARK_CONFIRMED]",
        "[STEP_COMPLETE]",
        "[SAFETY_HOLD]",
        "[RECOVERY]",
        "[ERROR]",
        "[ROUTE_COMPLETE]",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if any(tag in message for tag in self._TAGS):
            return True
        if record.levelno >= logging.ERROR:
            record.msg = f"[ERROR] {message}"
            record.args = ()
            return True
        return False


def load_config(config_path: str | None = None) -> AppConfig:
    selected_path = Path(config_path or os.getenv("CONFIG_PATH") or DEFAULT_CONFIG_PATH)
    selected_path = selected_path if selected_path.is_absolute() else PROJECT_ROOT / selected_path

    file_data: dict[str, Any] = {}
    if selected_path.exists():
        with selected_path.open("r", encoding="utf-8") as handle:
            file_data = yaml.safe_load(handle) or {}

    sim = _section(file_data, "sim")
    robot = _section(file_data, "robot")
    robot_extra_keys = sorted(set(robot) - {"profile"})
    if robot_extra_keys:
        raise ValueError(
            "The robot section only accepts 'profile'; move robot-specific values to "
            f"configs/robots/<profile>.yaml: {robot_extra_keys}"
        )
    robot_profile_name = str(robot.get("profile", "agx")).strip().lower()
    robot_profile = load_robot_profile(profile_name=robot_profile_name, project_root=PROJECT_ROOT)
    robot_primitives = robot_profile.primitives
    robot_geometry = robot_profile.geometry
    agx_controller = robot_profile.agx_controller
    camera = _section(file_data, "camera")
    llm = _section(file_data, "llm")
    legacy_deepseek = _section(file_data, "deepseek")
    opennav = _section(file_data, "opennav")
    target_grounding = _section(file_data, "target_grounding")
    landmark_tracker = _section(file_data, "landmark_tracker")
    control = _section(file_data, "control")
    odom = _section(file_data, "odom")
    navigation_map = _section(file_data, "navigation_map")
    occupancy_grid = _section(file_data, "occupancy_grid")
    map_candidates = _section(file_data, "map_candidates")
    lidar = _section(file_data, "lidar")
    topology = _section(file_data, "topology")
    scene = _section(file_data, "scene")
    logging_section = _section(file_data, "logging")

    config = AppConfig(
        llm_api_key=_env_or_value(
            "SILICONFLOW_API_KEY",
            _env_or_value("LLM_API_KEY", _section_value(llm, legacy_deepseek, "api_key", "")),
        ),
        model_name=_env_or_value(
            "MODEL_NAME",
            _section_value(llm, legacy_deepseek, "model_name", "Qwen/Qwen3-VL-235B-A22B-Instruct"),
        ),
        api_base_url=_env_or_value(
            "API_BASE_URL",
            _section_value(llm, legacy_deepseek, "api_base_url", "https://api.siliconflow.cn/v1"),
        ),
        llm_timeout_sec=_int_env_or_value("LLM_TIMEOUT_SEC", _section_value(llm, legacy_deepseek, "timeout_sec", 30)),
        llm_retry_cooldown_sec=_float_env_or_value(
            "LLM_RETRY_COOLDOWN_SEC",
            _section_value(llm, legacy_deepseek, "retry_cooldown_sec", 0.0),
        ),
        llm_upload_max_edge_px=_int_env_or_value(
            "LLM_UPLOAD_MAX_EDGE_PX",
            _section_value(llm, legacy_deepseek, "upload_max_edge_px", 512),
        ),
        camera_image_detail=_env_or_value(
            "CAMERA_IMAGE_DETAIL",
            _section_value(llm, legacy_deepseek, "camera_image_detail", "low"),
        ),
        instruction=_env_or_value("INSTRUCTION", opennav.get("instruction", "")),
        instruction_parse_max_tokens=_int_env_or_value(
            "INSTRUCTION_PARSE_MAX_TOKENS",
            opennav.get("instruction_parse_max_tokens", 2048),
        ),
        instruction_parse_timeout_sec=_int_env_or_value(
            "INSTRUCTION_PARSE_TIMEOUT_SEC",
            opennav.get("instruction_parse_timeout_sec", 90),
        ),
        instruction_cache_enabled=_bool_env_or_value(
            "INSTRUCTION_CACHE_ENABLED",
            opennav.get("instruction_cache_enabled", True),
        ),
        instruction_cache_path=_env_or_value(
            "INSTRUCTION_CACHE_PATH",
            opennav.get("instruction_cache_path", str(PROJECT_ROOT / ".cache" / "instruction_plans.json")),
        ),
        region_transition_min_displacement_m=_float_env_or_value(
            "REGION_TRANSITION_MIN_DISPLACEMENT_M",
            opennav.get("region_transition_min_displacement_m", 1.0),
        ),
        region_transition_confirmations=_int_env_or_value(
            "REGION_TRANSITION_CONFIRMATIONS",
            opennav.get("region_transition_confirmations", 2),
        ),
        region_transition_min_confidence=_float_env_or_value(
            "REGION_TRANSITION_MIN_CONFIDENCE",
            opennav.get("region_transition_min_confidence", 0.55),
        ),
        semantic_turn_min_yaw_deg=_float_env_or_value(
            "SEMANTIC_TURN_MIN_YAW_DEG",
            opennav.get("semantic_turn_min_yaw_deg", 45.0),
        ),
        semantic_turn_max_yaw_deg=_float_env_or_value(
            "SEMANTIC_TURN_MAX_YAW_DEG",
            opennav.get("semantic_turn_max_yaw_deg", 105.0),
        ),
        semantic_turn_max_primitives=_int_env_or_value(
            "SEMANTIC_TURN_MAX_PRIMITIVES",
            opennav.get("semantic_turn_max_primitives", 7),
        ),
        semantic_turn_min_forward_clearance_m=_float_env_or_value(
            "SEMANTIC_TURN_MIN_FORWARD_CLEARANCE_M",
            opennav.get("semantic_turn_min_forward_clearance_m", 0.60),
        ),
        semantic_turn_alignment_confirmations=_int_env_or_value(
            "SEMANTIC_TURN_ALIGNMENT_CONFIRMATIONS",
            opennav.get("semantic_turn_alignment_confirmations", 2),
        ),
        semantic_turn_corridor_center_tolerance_deg=_float_env_or_value(
            "SEMANTIC_TURN_CORRIDOR_CENTER_TOLERANCE_DEG",
            opennav.get("semantic_turn_corridor_center_tolerance_deg", 10.0),
        ),
        semantic_turn_same_corridor_tolerance_deg=_float_env_or_value(
            "SEMANTIC_TURN_SAME_CORRIDOR_TOLERANCE_DEG",
            opennav.get("semantic_turn_same_corridor_tolerance_deg", 15.0),
        ),
        semantic_turn_target_high_confidence=_float_env_or_value(
            "SEMANTIC_TURN_TARGET_HIGH_CONFIDENCE",
            opennav.get("semantic_turn_target_high_confidence", 0.75),
        ),
        semantic_turn_target_medium_confidence=_float_env_or_value(
            "SEMANTIC_TURN_TARGET_MEDIUM_CONFIDENCE",
            opennav.get("semantic_turn_target_medium_confidence", 0.50),
        ),
        semantic_turn_direction_tolerance_deg=_float_env_or_value(
            "SEMANTIC_TURN_DIRECTION_TOLERANCE_DEG",
            opennav.get("semantic_turn_direction_tolerance_deg", 10.0),
        ),
        semantic_turn_corridor_half_fov_deg=_float_env_or_value(
            "SEMANTIC_TURN_CORRIDOR_HALF_FOV_DEG",
            opennav.get("semantic_turn_corridor_half_fov_deg", 60.0),
        ),
        semantic_turn_corridor_step_deg=_float_env_or_value(
            "SEMANTIC_TURN_CORRIDOR_STEP_DEG",
            opennav.get("semantic_turn_corridor_step_deg", 5.0),
        ),
        semantic_turn_lidar_bin_half_width_deg=_float_env_or_value(
            "SEMANTIC_TURN_LIDAR_BIN_HALF_WIDTH_DEG",
            opennav.get("semantic_turn_lidar_bin_half_width_deg", 3.0),
        ),
        semantic_turn_min_corridor_width_deg=_float_env_or_value(
            "SEMANTIC_TURN_MIN_CORRIDOR_WIDTH_DEG",
            opennav.get("semantic_turn_min_corridor_width_deg", 15.0),
        ),
        semantic_turn_corridor_max_clearance_m=_float_env_or_value(
            "SEMANTIC_TURN_CORRIDOR_MAX_CLEARANCE_M",
            opennav.get("semantic_turn_corridor_max_clearance_m", 4.0),
        ),
        semantic_turn_max_recovery_attempts=_int_env_or_value(
            "SEMANTIC_TURN_MAX_RECOVERY_ATTEMPTS",
            opennav.get("semantic_turn_max_recovery_attempts", 2),
        ),
        local_detour_max_turn_primitives=_int_env_or_value(
            "LOCAL_DETOUR_MAX_TURN_PRIMITIVES",
            opennav.get("local_detour_max_turn_primitives", 2),
        ),
        traverse_heading_correction_limit_deg=_float_env_or_value(
            "TRAVERSE_HEADING_CORRECTION_LIMIT_DEG",
            opennav.get("traverse_heading_correction_limit_deg", 30.0),
        ),
        traverse_candidate_heading_tolerance_deg=_float_env_or_value(
            "TRAVERSE_CANDIDATE_HEADING_TOLERANCE_DEG",
            opennav.get("traverse_candidate_heading_tolerance_deg", 22.0),
        ),
        traverse_min_progress_m=_float_env_or_value(
            "TRAVERSE_MIN_PROGRESS_M",
            opennav.get("traverse_min_progress_m", 0.50),
        ),
        traverse_pass_margin_m=_float_env_or_value(
            "TRAVERSE_PASS_MARGIN_M",
            opennav.get("traverse_pass_margin_m", 0.20),
        ),
        traverse_review_min_progress_m=_float_env_or_value(
            "TRAVERSE_REVIEW_MIN_PROGRESS_M",
            opennav.get("traverse_review_min_progress_m", 1.00),
        ),
        traverse_review_min_translation_m=_float_env_or_value(
            "TRAVERSE_REVIEW_MIN_TRANSLATION_M",
            opennav.get("traverse_review_min_translation_m", 0.20),
        ),
        traverse_max_recovery_attempts=_int_env_or_value(
            "TRAVERSE_MAX_RECOVERY_ATTEMPTS",
            opennav.get("traverse_max_recovery_attempts", 2),
        ),
        visual_target_center_deadband_deg=_float_env_or_value(
            "VISUAL_TARGET_CENTER_DEADBAND_DEG",
            opennav.get("visual_target_center_deadband_deg", 15.0),
        ),
        candidate_slot_bearings_deg=_list_env_or_value(
            "CANDIDATE_SLOT_BEARINGS_DEG",
            opennav.get("candidate_slot_bearings_deg", [75.0, 35.0, 0.0, -35.0, -75.0]),
        ),
        candidate_adaptive_enabled=_bool_env_or_value(
            "CANDIDATE_ADAPTIVE_ENABLED",
            opennav.get("candidate_adaptive_enabled", True),
        ),
        candidate_adaptive_count=_int_env_or_value(
            "CANDIDATE_ADAPTIVE_COUNT",
            opennav.get("candidate_adaptive_count", 5),
        ),
        candidate_adaptive_min_spacing_deg=_float_env_or_value(
            "CANDIDATE_ADAPTIVE_MIN_SPACING_DEG",
            opennav.get("candidate_adaptive_min_spacing_deg", 18.0),
        ),
        candidate_adaptive_smoothing_px=_int_env_or_value(
            "CANDIDATE_ADAPTIVE_SMOOTHING_PX",
            opennav.get("candidate_adaptive_smoothing_px", 15),
        ),
        candidate_min_travel_distance_m=_float_env_or_value(
            "CANDIDATE_MIN_TRAVEL_DISTANCE_M",
            opennav.get("candidate_min_travel_distance_m", 0.55),
        ),
        candidate_max_travel_distance_m=_float_env_or_value(
            "CANDIDATE_MAX_TRAVEL_DISTANCE_M",
            opennav.get("candidate_max_travel_distance_m", 1.35),
        ),
        candidate_min_free_ratio=_float_env_or_value(
            "CANDIDATE_MIN_FREE_RATIO",
            opennav.get("candidate_min_free_ratio", 0.28),
        ),
        candidate_column_half_width_px=_int_env_or_value(
            "CANDIDATE_COLUMN_HALF_WIDTH_PX",
            opennav.get("candidate_column_half_width_px", 26),
        ),
        target_debug_enabled=_bool_env_or_value(
            "TARGET_DEBUG_ENABLED",
            opennav.get("target_debug_enabled", True),
        ),
        target_debug_image_path=_env_or_value(
            "TARGET_DEBUG_IMAGE_PATH",
            opennav.get("target_debug_image_path", "debug/vln_debug.png"),
        ),
        debug_video_enabled=_bool_env_or_value(
            "DEBUG_VIDEO_ENABLED",
            opennav.get("debug_video_enabled", True),
        ),
        debug_video_path=_env_or_value(
            "DEBUG_VIDEO_PATH",
            opennav.get("debug_video_path", "debug/videos/vln_debug.mp4"),
        ),
        debug_video_fps=_float_env_or_value(
            "DEBUG_VIDEO_FPS",
            opennav.get("debug_video_fps", 4.0),
        ),
        debug_image_update_period_sec=_float_env_or_value(
            "DEBUG_IMAGE_UPDATE_PERIOD_SEC",
            opennav.get("debug_image_update_period_sec", 1.0),
        ),
        progress_history_length=_int_env_or_value(
            "PROGRESS_HISTORY_LENGTH",
            opennav.get("progress_history_length", 6),
        ),
        progress_stall_distance_m=_float_env_or_value(
            "PROGRESS_STALL_DISTANCE_M",
            opennav.get("progress_stall_distance_m", 0.08),
        ),
        progress_spin_yaw_deg=_float_env_or_value(
            "PROGRESS_SPIN_YAW_DEG",
            opennav.get("progress_spin_yaw_deg", 85.0),
        ),
        policy_execution_steps=_int_env_or_value(
            "POLICY_EXECUTION_STEPS",
            opennav.get("policy_execution_steps", 24),
        ),
        arrival_distance_m=_float_env_or_value(
            "ARRIVAL_DISTANCE_M",
            opennav.get("arrival_distance_m", opennav.get("landmark_arrival_distance_m", 1.4)),
        ),
        success_distance_m=_float_env_or_value(
            "SUCCESS_DISTANCE_M",
            opennav.get("success_distance_m", 2.0),
        ),
        landmark_arrival_confirmations=_int_env_or_value(
            "LANDMARK_ARRIVAL_CONFIRMATIONS",
            opennav.get("landmark_arrival_confirmations", 2),
        ),
        target_grounding_enabled=_bool_env_or_value(
            "TARGET_GROUNDING_ENABLED",
            target_grounding.get("enabled", True),
        ),
        target_grounding_backend=_env_or_value(
            "TARGET_GROUNDING_BACKEND",
            target_grounding.get("backend", "grounding_dino"),
        ),
        target_grounding_call_period_sec=_float_env_or_value(
            "TARGET_GROUNDING_CALL_PERIOD_SEC",
            target_grounding.get("call_period_sec", 1.5),
        ),
        target_grounding_reuse_last_observation_sec=_float_env_or_value(
            "TARGET_GROUNDING_REUSE_LAST_OBSERVATION_SEC",
            target_grounding.get("reuse_last_observation_sec", 2.5),
        ),
        target_grounding_upload_max_edge_px=_int_env_or_value(
            "TARGET_GROUNDING_UPLOAD_MAX_EDGE_PX",
            target_grounding.get("upload_max_edge_px", 512),
        ),
        target_grounding_min_bbox_area_ratio=_float_env_or_value(
            "TARGET_GROUNDING_MIN_BBOX_AREA_RATIO",
            target_grounding.get("min_bbox_area_ratio", 0.002),
        ),
        target_grounding_max_bbox_area_ratio=_float_env_or_value(
            "TARGET_GROUNDING_MAX_BBOX_AREA_RATIO",
            target_grounding.get("max_bbox_area_ratio", 0.60),
        ),
        target_grounding_min_bbox_width_px=_int_env_or_value(
            "TARGET_GROUNDING_MIN_BBOX_WIDTH_PX",
            target_grounding.get("min_bbox_width_px", target_grounding.get("min_bbox_size_px", 4)),
        ),
        target_grounding_min_bbox_height_px=_int_env_or_value(
            "TARGET_GROUNDING_MIN_BBOX_HEIGHT_PX",
            target_grounding.get("min_bbox_height_px", target_grounding.get("min_bbox_size_px", 4)),
        ),
        target_grounding_max_bbox_aspect_ratio=_float_env_or_value(
            "TARGET_GROUNDING_MAX_BBOX_ASPECT_RATIO",
            target_grounding.get("max_bbox_aspect_ratio", 8.0),
        ),
        target_grounding_min_depth_valid_ratio=_float_env_or_value(
            "TARGET_GROUNDING_MIN_DEPTH_VALID_RATIO",
            target_grounding.get("min_depth_valid_ratio", 0.12),
        ),
        target_grounding_depth_statistic=_env_or_value(
            "TARGET_GROUNDING_DEPTH_STATISTIC",
            target_grounding.get("depth_statistic", "median"),
        ),
        target_grounding_center_patch_ratio=_float_env_or_value(
            "TARGET_GROUNDING_CENTER_PATCH_RATIO",
            target_grounding.get("center_patch_ratio", 0.25),
        ),
        target_grounding_min_depth_m=_float_env_or_value(
            "TARGET_GROUNDING_MIN_DEPTH_M",
            target_grounding.get("min_depth_m", 0.20),
        ),
        target_grounding_max_depth_m=_float_env_or_value(
            "TARGET_GROUNDING_MAX_DEPTH_M",
            target_grounding.get("max_depth_m", 6.0),
        ),
        target_grounding_min_confidence=_float_env_or_value(
            "TARGET_GROUNDING_MIN_CONFIDENCE",
            target_grounding.get("min_confidence", 0.50),
        ),
        target_grounding_coordinate_mode=_env_or_value(
            "TARGET_GROUNDING_COORDINATE_MODE",
            target_grounding.get("coordinate_mode", "auto"),
        ),
        target_grounding_allow_normalized_bbox=_bool_env_or_value(
            "TARGET_GROUNDING_ALLOW_NORMALIZED_BBOX",
            target_grounding.get("allow_normalized_bbox", True),
        ),
        target_grounding_normalized_bbox_scale=_float_env_or_value(
            "TARGET_GROUNDING_NORMALIZED_BBOX_SCALE",
            target_grounding.get("normalized_bbox_scale", 1000.0),
        ),
        target_grounding_depth_region=_env_or_value(
            "TARGET_GROUNDING_DEPTH_REGION",
            target_grounding.get("depth_region", "bbox_depth_mask"),
        ),
        target_grounding_depth_mask_tolerance_ratio=_float_env_or_value(
            "TARGET_GROUNDING_DEPTH_MASK_TOLERANCE_RATIO",
            target_grounding.get("depth_mask_tolerance_ratio", 0.35),
        ),
        target_grounding_depth_mask_min_pixels=_int_env_or_value(
            "TARGET_GROUNDING_DEPTH_MASK_MIN_PIXELS",
            target_grounding.get("depth_mask_min_pixels", 16),
        ),
        target_grounding_dino_repo_path=str(
            _resolve_path(
                _env_or_value(
                    "GROUNDING_DINO_REPO_PATH",
                    target_grounding.get("grounding_dino_repo_path", "third_party/GroundingDINO"),
                )
            )
        ),
        target_grounding_dino_config_path=str(
            _resolve_path(
                _env_or_value(
                    "GROUNDING_DINO_CONFIG_PATH",
                    target_grounding.get(
                        "grounding_dino_config_path",
                        "weights/grounded_sam/GroundingDINO_SwinT_OGC.py",
                    ),
                )
            )
        ),
        target_grounding_dino_checkpoint_path=str(
            _resolve_path(
                _env_or_value(
                    "GROUNDING_DINO_CHECKPOINT_PATH",
                    target_grounding.get(
                        "grounding_dino_checkpoint_path",
                        "weights/grounded_sam/groundingdino_swint_ogc.pth",
                    ),
                )
            )
        ),
        target_grounding_dino_device=_env_or_value(
            "GROUNDING_DINO_DEVICE",
            target_grounding.get("grounding_dino_device", "cuda"),
        ),
        target_grounding_dino_box_threshold=_float_env_or_value(
            "GROUNDING_DINO_BOX_THRESHOLD",
            target_grounding.get("grounding_dino_box_threshold", 0.45),
        ),
        target_grounding_dino_text_threshold=_float_env_or_value(
            "GROUNDING_DINO_TEXT_THRESHOLD",
            target_grounding.get("grounding_dino_text_threshold", 0.25),
        ),
        target_grounding_dino_max_detections=_int_env_or_value(
            "GROUNDING_DINO_MAX_DETECTIONS",
            target_grounding.get("grounding_dino_max_detections", 5),
        ),
        target_grounding_dino_nms_iou_threshold=_float_env_or_value(
            "GROUNDING_DINO_NMS_IOU_THRESHOLD",
            target_grounding.get("grounding_dino_nms_iou_threshold", 0.50),
        ),
        target_grounding_dino_text_encoder_path=str(
            _resolve_path(
                _env_or_value(
                    "GROUNDING_DINO_TEXT_ENCODER_PATH",
                    target_grounding.get(
                        "grounding_dino_text_encoder_path",
                        "weights/grounded_sam/bert-base-uncased",
                    ),
                )
            )
        ),
        target_grounding_sam_checkpoint_path=str(
            _resolve_path(
                _env_or_value(
                    "SAM_CHECKPOINT_PATH",
                    target_grounding.get(
                        "sam_checkpoint_path",
                        "weights/grounded_sam/sam_vit_h_4b8939.pth",
                    ),
                )
            )
        ),
        target_grounding_sam_model_type=_env_or_value(
            "SAM_MODEL_TYPE",
            target_grounding.get("sam_model_type", "vit_h"),
        ),
        target_grounding_sam_device=_env_or_value(
            "SAM_DEVICE",
            target_grounding.get("sam_device", "cuda"),
        ),
        target_grounding_sam_min_mask_area_ratio=_float_env_or_value(
            "SAM_MIN_MASK_AREA_RATIO",
            target_grounding.get("sam_min_mask_area_ratio", 0.0005),
        ),
        target_grounding_sam_enabled=_bool_env_or_value(
            "SAM_ENABLED",
            target_grounding.get("sam_enabled", True),
        ),
        target_grounding_max_result_age_sec=_float_env_or_value(
            "TARGET_GROUNDING_MAX_RESULT_AGE_SEC",
            target_grounding.get("max_result_age_sec", 30.0),
        ),
        target_grounding_max_result_pose_delta_m=_float_env_or_value(
            "TARGET_GROUNDING_MAX_RESULT_POSE_DELTA_M",
            target_grounding.get("max_result_pose_delta_m", 0.05),
        ),
        target_grounding_max_result_yaw_delta_deg=_float_env_or_value(
            "TARGET_GROUNDING_MAX_RESULT_YAW_DELTA_DEG",
            target_grounding.get("max_result_yaw_delta_deg", 3.0),
        ),
        target_grounding_semantic_verification_enabled=_bool_env_or_value(
            "TARGET_GROUNDING_SEMANTIC_VERIFICATION_ENABLED",
            target_grounding.get("semantic_verification_enabled", True),
        ),
        target_grounding_semantic_verification_min_confidence=_float_env_or_value(
            "TARGET_GROUNDING_SEMANTIC_VERIFICATION_MIN_CONFIDENCE",
            target_grounding.get("semantic_verification_min_confidence", 0.70),
        ),
        target_grounding_semantic_verification_max_edge_px=_int_env_or_value(
            "TARGET_GROUNDING_SEMANTIC_VERIFICATION_MAX_EDGE_PX",
            target_grounding.get("semantic_verification_max_edge_px", 768),
        ),
        target_grounding_semantic_verification_auto_accept_confidence=_float_env_or_value(
            "TARGET_GROUNDING_SEMANTIC_VERIFICATION_AUTO_ACCEPT_CONFIDENCE",
            target_grounding.get("semantic_verification_auto_accept_confidence", 0.75),
        ),
        target_grounding_pause_during_active_primitive=_bool_env_or_value(
            "TARGET_GROUNDING_PAUSE_DURING_ACTIVE_PRIMITIVE",
            target_grounding.get("pause_during_active_primitive", True),
        ),
        target_tracker_enabled=_bool_env_or_value(
            "TARGET_TRACKER_ENABLED",
            target_grounding.get("tracker_enabled", target_grounding.get("target_tracker_enabled", True)),
        ),
        target_tracker_type=_env_or_value(
            "TARGET_TRACKER_TYPE",
            target_grounding.get("tracker_type", "CSRT"),
        ),
        target_tracker_max_lost_frames=_int_env_or_value(
            "TARGET_TRACKER_MAX_LOST_FRAMES",
            target_grounding.get("tracker_max_lost_frames", 8),
        ),
        target_tracker_lost_confirmed_sec=_float_env_or_value(
            "TARGET_TRACKER_LOST_CONFIRMED_SEC",
            target_grounding.get("tracker_lost_confirmed_sec", 2.0),
        ),
        target_tracker_min_depth_valid_ratio=_float_env_or_value(
            "TARGET_TRACKER_MIN_DEPTH_VALID_RATIO",
            target_grounding.get("tracker_min_depth_valid_ratio", target_grounding.get("min_depth_valid_ratio", 0.08)),
        ),
        target_tracker_center_patch_ratio=_float_env_or_value(
            "TARGET_TRACKER_CENTER_PATCH_RATIO",
            target_grounding.get("tracker_center_patch_ratio", 0.35),
        ),
        target_tracker_confidence_decay_per_lost_frame=_float_env_or_value(
            "TARGET_TRACKER_CONFIDENCE_DECAY_PER_LOST_FRAME",
            target_grounding.get("tracker_confidence_decay_per_lost_frame", 0.06),
        ),
        target_tracker_min_tracking_confidence=_float_env_or_value(
            "TARGET_TRACKER_MIN_TRACKING_CONFIDENCE",
            target_grounding.get("tracker_min_tracking_confidence", 0.35),
        ),
        landmark_tracker_arrival_visible_depth_m=_float_env_or_value(
            "LANDMARK_TRACKER_ARRIVAL_VISIBLE_DEPTH_M",
            landmark_tracker.get(
                "arrival_visible_depth_m",
                opennav.get("arrival_distance_m", opennav.get("landmark_arrival_distance_m", 1.4)),
            ),
        ),
        landmark_tracker_arrival_anchor_radius_m=_float_env_or_value(
            "LANDMARK_TRACKER_ARRIVAL_ANCHOR_RADIUS_M",
            landmark_tracker.get(
                "arrival_anchor_radius_m",
                opennav.get("arrival_distance_m", opennav.get("landmark_arrival_distance_m", 1.4)),
            ),
        ),
        landmark_tracker_min_visible_count=_int_env_or_value(
            "LANDMARK_TRACKER_MIN_VISIBLE_COUNT",
            landmark_tracker.get("min_visible_count", 2),
        ),
        landmark_tracker_arrival_confirmations=_int_env_or_value(
            "LANDMARK_TRACKER_ARRIVAL_CONFIRMATIONS",
            landmark_tracker.get("arrival_confirmations", 2),
        ),
        landmark_tracker_arrival_window_size=_int_env_or_value(
            "LANDMARK_TRACKER_ARRIVAL_WINDOW_SIZE",
            landmark_tracker.get("arrival_window_size", 5),
        ),
        landmark_tracker_arrival_recent_visible_sec=_float_env_or_value(
            "LANDMARK_TRACKER_ARRIVAL_RECENT_VISIBLE_SEC",
            landmark_tracker.get("arrival_recent_visible_sec", 3.0),
        ),
        landmark_tracker_max_lost_count=_int_env_or_value(
            "LANDMARK_TRACKER_MAX_LOST_COUNT",
            landmark_tracker.get("max_lost_count", 6),
        ),
        landmark_tracker_anchor_update_min_consistent_observations=_int_env_or_value(
            "LANDMARK_TRACKER_ANCHOR_UPDATE_MIN_CONSISTENT_OBSERVATIONS",
            landmark_tracker.get("anchor_update_min_consistent_observations", 2),
        ),
        landmark_tracker_anchor_update_max_jump_m=_float_env_or_value(
            "LANDMARK_TRACKER_ANCHOR_UPDATE_MAX_JUMP_M",
            landmark_tracker.get("anchor_update_max_jump_m", 1.5),
        ),
        landmark_tracker_confirm_confidence_threshold=_float_env_or_value(
            "LANDMARK_TRACKER_CONFIRM_CONFIDENCE_THRESHOLD",
            landmark_tracker.get("confirm_confidence_threshold", 0.50),
        ),
        landmark_tracker_potential_confidence_threshold=_float_env_or_value(
            "LANDMARK_TRACKER_POTENTIAL_CONFIDENCE_THRESHOLD",
            landmark_tracker.get("potential_confidence_threshold", 0.50),
        ),
        landmark_tracker_confirm_min_observations=_int_env_or_value(
            "LANDMARK_TRACKER_CONFIRM_MIN_OBSERVATIONS",
            landmark_tracker.get("confirm_min_observations", 3),
        ),
        landmark_tracker_confirm_max_anchor_jump_m=_float_env_or_value(
            "LANDMARK_TRACKER_CONFIRM_MAX_ANCHOR_JUMP_M",
            landmark_tracker.get("confirm_max_anchor_jump_m", 0.50),
        ),
        landmark_tracker_confirm_max_depth_jump_m=_float_env_or_value(
            "LANDMARK_TRACKER_CONFIRM_MAX_DEPTH_JUMP_M",
            landmark_tracker.get("confirm_max_depth_jump_m", 0.80),
        ),
        landmark_tracker_low_confidence_min_translation_m=_float_env_or_value(
            "LANDMARK_TRACKER_LOW_CONFIDENCE_MIN_TRANSLATION_M",
            landmark_tracker.get("low_confidence_min_translation_m", 0.20),
        ),
        landmark_tracker_low_confidence_min_yaw_deg=_float_env_or_value(
            "LANDMARK_TRACKER_LOW_CONFIDENCE_MIN_YAW_DEG",
            landmark_tracker.get("low_confidence_min_yaw_deg", 10.0),
        ),
        landmark_map_enabled=_bool_env_or_value(
            "LANDMARK_MAP_ENABLED",
            landmark_tracker.get("landmark_map_enabled", True),
        ),
        landmark_map_anchor_ttl_sec=_float_env_or_value(
            "LANDMARK_MAP_ANCHOR_TTL_SEC",
            landmark_tracker.get("landmark_map_anchor_ttl_sec", 120.0),
        ),
        landmark_map_min_confidence=_float_env_or_value(
            "LANDMARK_MAP_MIN_CONFIDENCE",
            landmark_tracker.get("landmark_map_min_confidence", 0.20),
        ),
        landmark_map_log_period_sec=_float_env_or_value(
            "LANDMARK_MAP_LOG_PERIOD_SEC",
            landmark_tracker.get("landmark_map_log_period_sec", 2.0),
        ),
        depth_debug_enabled=_bool_env_or_value(
            "DEPTH_DEBUG_ENABLED",
            camera.get("depth_debug_enabled", True),
        ),
        depth_debug_image_path=_env_or_value(
            "DEPTH_DEBUG_IMAGE_PATH",
            camera.get("depth_debug_image_path", "debug/depth_debug.png"),
        ),
        headless=_bool_env_or_value("HEADLESS", sim.get("headless", False)),
        sim_width=_int_env_or_value("SIM_WIDTH", sim.get("width", 1280)),
        sim_height=_int_env_or_value("SIM_HEIGHT", sim.get("height", 720)),
        active_gpu=_optional_int_env_or_value("ACTIVE_GPU", sim.get("active_gpu", None)),
        physics_gpu=_optional_int_env_or_value("PHYSICS_GPU", sim.get("physics_gpu", 0)),
        multi_gpu=_bool_env_or_value("MULTI_GPU", sim.get("multi_gpu", False)),
        physics_dt=float(robot_profile.simulation.physics_dt),
        rendering_dt=float(robot_profile.simulation.rendering_dt),
        warmup_steps=_int_env_or_value("WARMUP_STEPS", sim.get("warmup_steps", 30)),
        robot_profile_name=robot_profile.name,
        robot_type=robot_profile.robot_type,
        robot_profile=robot_profile,
        robot_policy_decimation=int(robot_profile.simulation.policy_decimation),
        robot_usd_path=robot_profile.asset.usd_path,
        robot_prim_path=robot_profile.asset.prim_path,
        robot_articulation_path=robot_profile.asset.articulation_path,
        robot_init_position=[1.5, 4.0, float(robot_profile.spawn_z_m)],
        robot_init_orientation_deg=[0.0, 0.0, 0.0],
        camera_width=_int_env_or_value("CAMERA_WIDTH", camera.get("width", 640)),
        camera_height=_int_env_or_value("CAMERA_HEIGHT", camera.get("height", 480)),
        front_camera_prim_path=robot_profile.sensors.front_camera_prim_path,
        observer_camera_enabled=bool(robot_profile.sensors.observer_camera_enabled),
        observer_camera_prim_path=robot_profile.sensors.observer_camera_prim_path,
        observer_video_enabled=_bool_env_or_value(
            "OBSERVER_VIDEO_ENABLED",
            camera.get("observer_video_enabled", True),
        ),
        observer_video_path=_env_or_value(
            "OBSERVER_VIDEO_PATH",
            camera.get("observer_video_path", "debug/local/observevideos/observer.mp4"),
        ),
        observer_video_fps=_float_env_or_value(
            "OBSERVER_VIDEO_FPS",
            camera.get("observer_video_fps", 10.0),
        ),
        follow_front_camera_in_gui=_bool_env_or_value(
            "FOLLOW_FRONT_CAMERA_IN_GUI",
            camera.get("follow_front_camera_in_gui", False),
        ),
        follow_observer_camera_in_gui=_bool_env_or_value(
            "FOLLOW_OBSERVER_CAMERA_IN_GUI",
            camera.get("follow_observer_camera_in_gui", True),
        ),
        sensor_frequency_hz=int(robot_profile.sensors.camera_frequency_hz),
        camera_horizontal_fov_deg=_float_env_or_value(
            "CAMERA_HORIZONTAL_FOV_DEG",
            camera.get("camera_horizontal_fov_deg", 90.0),
        ),
        interactive_confirmation_enabled=_bool_env_or_value(
            "INTERACTIVE_CONFIRMATION_ENABLED",
            camera.get("interactive_confirmation_enabled", False),
        ),
        roi_vertical_range=_list_env_or_value(
            "ROI_VERTICAL_RANGE",
            camera.get("roi_vertical_range", [0.45, 0.85]),
        ),
        clearance_threshold_m=_float_env_or_value(
            "CLEARANCE_THRESHOLD_M",
            camera.get("clearance_threshold_m", 1.2),
        ),
        valid_depth_range_m=_list_env_or_value(
            "VALID_DEPTH_RANGE_M",
            camera.get("valid_depth_range_m", [0.2, 8.0]),
        ),
        obstacle_avoidance_enabled=_bool_env_or_value(
            "OBSTACLE_AVOIDANCE_ENABLED",
            camera.get("obstacle_avoidance_enabled", True),
        ),
        obstacle_stop_distance_m=float(robot_geometry.obstacle_stop_distance_m),
        obstacle_roi_vertical_range=_list_env_or_value(
            "OBSTACLE_ROI_VERTICAL_RANGE",
            camera.get("obstacle_roi_vertical_range", [0.55, 0.92]),
        ),
        obstacle_center_width_ratio=_float_env_or_value(
            "OBSTACLE_CENTER_WIDTH_RATIO",
            camera.get("obstacle_center_width_ratio", 0.28),
        ),
        obstacle_side_width_ratio=_float_env_or_value(
            "OBSTACLE_SIDE_WIDTH_RATIO",
            camera.get("obstacle_side_width_ratio", 0.24),
        ),
        near_field_memory_enabled=_bool_env_or_value(
            "NEAR_FIELD_MEMORY_ENABLED",
            camera.get("near_field_memory_enabled", True),
        ),
        near_field_memory_sec=_float_env_or_value(
            "NEAR_FIELD_MEMORY_SEC",
            camera.get("near_field_memory_sec", 2.0),
        ),
        candidate_preferred_distance_m=_float_env_or_value(
            "CANDIDATE_PREFERRED_DISTANCE_M",
            camera.get("candidate_preferred_distance_m", 1.0),
        ),
        waypoint_heading_threshold_rad=_float_env_or_value(
            "WAYPOINT_HEADING_THRESHOLD_RAD",
            camera.get("waypoint_heading_threshold_rad", 0.3),
        ),
        waypoint_stop_distance_m=_float_env_or_value(
            "WAYPOINT_STOP_DISTANCE_M",
            camera.get("waypoint_stop_distance_m", 0.45),
        ),
        waypoint_turn_speed_scale=_float_env_or_value(
            "WAYPOINT_TURN_SPEED_SCALE",
            camera.get("waypoint_turn_speed_scale", 1.0),
        ),
        waypoint_forward_speed_scale=_float_env_or_value(
            "WAYPOINT_FORWARD_SPEED_SCALE",
            camera.get("waypoint_forward_speed_scale", 1.0),
        ),
        replanning_cooldown_sec=_float_env_or_value(
            "REPLANNING_COOLDOWN_SEC",
            camera.get("replanning_cooldown_sec", 0.5),
        ),
        depth_eval_period_sec=_float_env_or_value(
            "DEPTH_EVAL_PERIOD_SEC",
            camera.get("depth_eval_period_sec", 0.12),
        ),
        wheel_drive_velocity=0.0 if agx_controller is None else float(agx_controller.wheel_drive_velocity),
        turn_drive_velocity=0.0 if agx_controller is None else float(agx_controller.turn_drive_velocity),
        steering_angle=0.0 if agx_controller is None else float(agx_controller.steering_angle),
        turn_mode="policy" if agx_controller is None else agx_controller.turn_mode,
        turn_steering_angle=0.0 if agx_controller is None else float(agx_controller.turn_steering_angle),
        rear_steering_ratio=0.0 if agx_controller is None else float(agx_controller.rear_steering_ratio),
        track_min_speed_scale=0.0 if agx_controller is None else float(agx_controller.track_min_speed_scale),
        track_max_speed_scale=0.0 if agx_controller is None else float(agx_controller.track_max_speed_scale),
        track_heading_gain=0.0 if agx_controller is None else float(agx_controller.track_heading_gain),
        track_slow_heading_rad=0.0 if agx_controller is None else float(agx_controller.track_slow_heading_rad),
        track_smoothing_alpha=0.0 if agx_controller is None else float(agx_controller.track_smoothing_alpha),
        track_steering_sign=0.0 if agx_controller is None else float(agx_controller.track_steering_sign),
        action_forward_duration_sec=float(robot_primitives.forward_timeout_sec),
        action_turn_duration_sec=float(robot_primitives.turn_timeout_sec),
        action_turn_angle_deg=float(robot_primitives.turn_angle_deg),
        action_forward_distance_m=float(robot_primitives.forward_distance_m),
        action_distance_tolerance_m=float(robot_primitives.distance_tolerance_m),
        action_turn_tolerance_deg=float(robot_primitives.turn_tolerance_deg),
        action_settle_max_wait_sec=float(robot_primitives.settle_max_wait_sec),
        action_settle_linear_speed_threshold_mps=float(robot_primitives.settle_linear_speed_threshold_mps),
        action_settle_angular_speed_threshold_radps=float(robot_primitives.settle_angular_speed_threshold_radps),
        action_settle_stable_duration_sec=float(robot_primitives.settle_stable_duration_sec),
        action_odom_completion_enabled=_bool_env_or_value(
            "ACTION_ODOM_COMPLETION_ENABLED",
            control.get("action_odom_completion_enabled", True),
        ),
        active_track_max_yaw_deg=_float_env_or_value(
            "ACTIVE_TRACK_MAX_YAW_DEG",
            control.get("active_track_max_yaw_deg", 10.0),
        ),
        active_track_min_duration_sec=_float_env_or_value(
            "ACTIVE_TRACK_MIN_DURATION_SEC",
            control.get("active_track_min_duration_sec", 0.45),
        ),
        use_discrete_action_primitives=_bool_env_or_value(
            "USE_DISCRETE_ACTION_PRIMITIVES",
            control.get("use_discrete_action_primitives", True),
        ),
        odom_enabled=_bool_env_or_value("ODOM_ENABLED", odom.get("enabled", True)),
        odom_source=_env_or_value("ODOM_SOURCE", odom.get("source", "isaac_world_pose")),
        navigation_map_enabled=_bool_env_or_value(
            "NAVIGATION_MAP_ENABLED",
            navigation_map.get("enabled", False),
        ),
        navigation_map_image_path=_env_or_value(
            "NAVIGATION_MAP_IMAGE_PATH",
            navigation_map.get("image_path", "assets/navigation_map.png"),
        ),
        navigation_map_update_period_sec=_float_env_or_value(
            "NAVIGATION_MAP_UPDATE_PERIOD_SEC",
            navigation_map.get("update_period_sec", 0.5),
        ),
        navigation_map_width=_int_env_or_value(
            "NAVIGATION_MAP_WIDTH",
            navigation_map.get("width", 900),
        ),
        navigation_map_height=_int_env_or_value(
            "NAVIGATION_MAP_HEIGHT",
            navigation_map.get("height", 900),
        ),
        navigation_map_depth_sample_stride_px=_int_env_or_value(
            "NAVIGATION_MAP_DEPTH_SAMPLE_STRIDE_PX",
            navigation_map.get("depth_sample_stride_px", 16),
        ),
        navigation_map_max_depth_obstacle_points=_int_env_or_value(
            "NAVIGATION_MAP_MAX_DEPTH_OBSTACLE_POINTS",
            navigation_map.get("max_depth_obstacle_points", 2500),
        ),
        topology_enabled=_bool_env_or_value(
            "TOPOLOGY_ENABLED",
            topology.get("enabled", True),
        ),
        topology_node_spacing_m=_float_env_or_value(
            "TOPOLOGY_NODE_SPACING_M",
            topology.get("node_spacing_m", 1.0),
        ),
        topology_turn_threshold_deg=_float_env_or_value(
            "TOPOLOGY_TURN_THRESHOLD_DEG",
            topology.get("turn_threshold_deg", 40.0),
        ),
        topology_merge_radius_m=_float_env_or_value(
            "TOPOLOGY_MERGE_RADIUS_M",
            topology.get("merge_radius_m", 0.75),
        ),
        topology_landmark_bind_radius_m=_float_env_or_value(
            "TOPOLOGY_LANDMARK_BIND_RADIUS_M",
            topology.get("landmark_bind_radius_m", 1.50),
        ),
        topology_max_nodes=_int_env_or_value(
            "TOPOLOGY_MAX_NODES",
            topology.get("max_nodes", 128),
        ),
        topology_recent_node_count=_int_env_or_value(
            "TOPOLOGY_RECENT_NODE_COUNT",
            topology.get("recent_node_count", 8),
        ),
        occupancy_grid_enabled=_bool_env_or_value(
            "OCCUPANCY_GRID_ENABLED",
            occupancy_grid.get("enabled", True),
        ),
        occupancy_grid_resolution_m=_float_env_or_value(
            "OCCUPANCY_GRID_RESOLUTION_M",
            occupancy_grid.get("resolution_m", 0.08),
        ),
        occupancy_grid_size_m=_float_env_or_value(
            "OCCUPANCY_GRID_SIZE_M",
            occupancy_grid.get("size_m", 24.0),
        ),
        occupancy_grid_update_period_sec=_float_env_or_value(
            "OCCUPANCY_GRID_UPDATE_PERIOD_SEC",
            occupancy_grid.get("update_period_sec", 0.12),
        ),
        occupancy_grid_depth_sample_stride_px=_int_env_or_value(
            "OCCUPANCY_GRID_DEPTH_SAMPLE_STRIDE_PX",
            occupancy_grid.get("depth_sample_stride_px", 10),
        ),
        occupancy_grid_depth_roi_vertical_range=_list_env_or_value(
            "OCCUPANCY_GRID_DEPTH_ROI_VERTICAL_RANGE",
            occupancy_grid.get("depth_roi_vertical_range", [0.38, 0.72]),
        ),
        occupancy_grid_min_depth_m=_float_env_or_value(
            "OCCUPANCY_GRID_MIN_DEPTH_M",
            occupancy_grid.get("min_depth_m", 0.20),
        ),
        occupancy_grid_max_depth_m=_float_env_or_value(
            "OCCUPANCY_GRID_MAX_DEPTH_M",
            occupancy_grid.get("max_depth_m", 10.0),
        ),
        occupancy_grid_inflation_radius_m=float(robot_geometry.map_inflation_radius_m),
        occupancy_grid_occupied_min_neighbors=_int_env_or_value(
            "OCCUPANCY_GRID_OCCUPIED_MIN_NEIGHBORS",
            occupancy_grid.get("occupied_min_neighbors", 3),
        ),
        occupancy_grid_robot_radius_m=float(robot_geometry.footprint_radius_m),
        occupancy_grid_path_corridor_radius_m=float(robot_geometry.path_corridor_radius_m),
        occupancy_grid_turn_clearance_radius_m=float(robot_geometry.turn_clearance_radius_m),
        occupancy_grid_unknown_is_traversable=_bool_env_or_value(
            "OCCUPANCY_GRID_UNKNOWN_IS_TRAVERSABLE",
            occupancy_grid.get("unknown_is_traversable", False),
        ),
        occupancy_grid_unknown_cell_cost=_float_env_or_value(
            "OCCUPANCY_GRID_UNKNOWN_CELL_COST",
            occupancy_grid.get("unknown_cell_cost", 0.80),
        ),
        occupancy_grid_lidar_bootstrap_sec=_float_env_or_value(
            "OCCUPANCY_GRID_LIDAR_BOOTSTRAP_SEC",
            occupancy_grid.get("lidar_bootstrap_sec", 0.8),
        ),
        occupancy_grid_lidar_bootstrap_min_fov_deg=_float_env_or_value(
            "OCCUPANCY_GRID_LIDAR_BOOTSTRAP_MIN_FOV_DEG",
            occupancy_grid.get("lidar_bootstrap_min_fov_deg", 165.0),
        ),
        occupancy_grid_lidar_bootstrap_max_wait_sec=_float_env_or_value(
            "OCCUPANCY_GRID_LIDAR_BOOTSTRAP_MAX_WAIT_SEC",
            occupancy_grid.get("lidar_bootstrap_max_wait_sec", 3.0),
        ),
        occupancy_grid_semantic_planner_enabled=_bool_env_or_value(
            "OCCUPANCY_GRID_SEMANTIC_PLANNER_ENABLED",
            occupancy_grid.get("semantic_planner_enabled", True),
        ),
        occupancy_grid_semantic_goal_step_m=_float_env_or_value(
            "OCCUPANCY_GRID_SEMANTIC_GOAL_STEP_M",
            occupancy_grid.get("semantic_goal_step_m", 0.90),
        ),
        occupancy_grid_semantic_goal_min_step_m=_float_env_or_value(
            "OCCUPANCY_GRID_SEMANTIC_GOAL_MIN_STEP_M",
            occupancy_grid.get("semantic_goal_min_step_m", 0.45),
        ),
        occupancy_grid_semantic_goal_lateral_search_m=_float_env_or_value(
            "OCCUPANCY_GRID_SEMANTIC_GOAL_LATERAL_SEARCH_M",
            occupancy_grid.get("semantic_goal_lateral_search_m", 0.80),
        ),
        map_candidates_enabled=_bool_env_or_value(
            "MAP_CANDIDATES_ENABLED",
            map_candidates.get("enabled", True),
        ),
        map_candidates_max_candidates=_int_env_or_value(
            "MAP_CANDIDATES_MAX_CANDIDATES",
            map_candidates.get("max_candidates", 3),
        ),
        map_candidates_min_distance_m=_float_env_or_value(
            "MAP_CANDIDATES_MIN_DISTANCE_M",
            map_candidates.get("min_distance_m", 0.55),
        ),
        map_candidates_max_distance_m=_float_env_or_value(
            "MAP_CANDIDATES_MAX_DISTANCE_M",
            map_candidates.get("max_distance_m", 1.35),
        ),
        map_candidates_semantic_enabled=_bool_env_or_value(
            "MAP_CANDIDATES_SEMANTIC_ENABLED",
            map_candidates.get("semantic_enabled", True),
        ),
        map_candidates_frontier_enabled=_bool_env_or_value(
            "MAP_CANDIDATES_FRONTIER_ENABLED",
            map_candidates.get("frontier_enabled", True),
        ),
        map_candidates_escape_enabled=_bool_env_or_value(
            "MAP_CANDIDATES_ESCAPE_ENABLED",
            map_candidates.get("escape_enabled", True),
        ),
        map_candidates_min_clearance_m=_float_env_or_value(
            "MAP_CANDIDATES_MIN_CLEARANCE_M",
            map_candidates.get("min_clearance_m", 0.34),
        ),
        map_candidates_unknown_gain_weight=_float_env_or_value(
            "MAP_CANDIDATES_UNKNOWN_GAIN_WEIGHT",
            map_candidates.get("unknown_gain_weight", 0.35),
        ),
        map_candidates_semantic_weight=_float_env_or_value(
            "MAP_CANDIDATES_SEMANTIC_WEIGHT",
            map_candidates.get("semantic_weight", 0.45),
        ),
        lidar_enabled=_bool_env_or_value("LIDAR_ENABLED", lidar.get("enabled", True)),
        lidar_prim_path=robot_profile.sensors.lidar_prim_path,
        lidar_child_name=_env_or_value(
            "LIDAR_CHILD_NAME",
            lidar.get("child_name", "lidar_sensor"),
        ),
        lidar_horizontal_fov_deg=_float_env_or_value(
            "LIDAR_HORIZONTAL_FOV_DEG",
            lidar.get("horizontal_fov_deg", 360.0),
        ),
        lidar_vertical_fov_deg=_float_env_or_value(
            "LIDAR_VERTICAL_FOV_DEG",
            lidar.get("vertical_fov_deg", 10.0),
        ),
        lidar_horizontal_resolution_deg=_float_env_or_value(
            "LIDAR_HORIZONTAL_RESOLUTION_DEG",
            lidar.get("horizontal_resolution_deg", 1.0),
        ),
        lidar_vertical_resolution_deg=_float_env_or_value(
            "LIDAR_VERTICAL_RESOLUTION_DEG",
            lidar.get("vertical_resolution_deg", 5.0),
        ),
        lidar_rotation_rate_hz=_float_env_or_value(
            "LIDAR_ROTATION_RATE_HZ",
            lidar.get("rotation_rate_hz", 10.0),
        ),
        lidar_min_range_m=_float_env_or_value(
            "LIDAR_MIN_RANGE_M",
            lidar.get("min_range_m", 0.20),
        ),
        lidar_max_range_m=_float_env_or_value(
            "LIDAR_MAX_RANGE_M",
            lidar.get("max_range_m", 6.0),
        ),
        lidar_yaw_offset_deg=_float_env_or_value(
            "LIDAR_YAW_OFFSET_DEG",
            lidar.get("yaw_offset_deg", 0.0),
        ),
        lidar_sector_merge_keep_sec=_float_env_or_value(
            "LIDAR_SECTOR_MERGE_KEEP_SEC",
            lidar.get("sector_merge_keep_sec", 1.5),
        ),
        lidar_draw_points=_bool_env_or_value("LIDAR_DRAW_POINTS", lidar.get("draw_points", False)),
        lidar_draw_lines=_bool_env_or_value("LIDAR_DRAW_LINES", lidar.get("draw_lines", False)),
        lidar_mode=_env_or_value("LIDAR_MODE", lidar.get("mode", "single")),
        lidar_multi_sector_count=_int_env_or_value(
            "LIDAR_MULTI_SECTOR_COUNT", lidar.get("multi_sector_count", 3),
        ),
        lidar_multi_sector_fov_deg=_float_env_or_value(
            "LIDAR_MULTI_SECTOR_FOV_DEG", lidar.get("multi_sector_fov_deg", 60.0),
        ),
        lidar_multi_sector_yaw_offsets_deg=_list_env_or_value(
            "LIDAR_MULTI_SECTOR_YAW_OFFSETS_DEG",
            lidar.get("multi_sector_yaw_offsets_deg", [-60.0, 0.0, 60.0]),
        ),
        lidar_multi_child_names=_list_env_or_value(
            "LIDAR_MULTI_CHILD_NAMES",
            lidar.get("multi_child_names", ["LidarLeft", "LidarCenter", "LidarRight"]),
        ),
        stuck_detection_enabled=_bool_env_or_value(
            "STUCK_DETECTION_ENABLED",
            control.get("stuck_detection_enabled", True),
        ),
        stuck_window_sec=_float_env_or_value(
            "STUCK_WINDOW_SEC",
            control.get("stuck_window_sec", 2.0),
        ),
        stuck_displacement_threshold_m=_float_env_or_value(
            "STUCK_DISPLACEMENT_THRESHOLD_M",
            control.get("stuck_displacement_threshold_m", 0.05),
        ),
        stuck_yaw_threshold_deg=_float_env_or_value(
            "STUCK_YAW_THRESHOLD_DEG",
            control.get("stuck_yaw_threshold_deg", 5.0),
        ),
        stuck_recovery_pause_sec=_float_env_or_value(
            "STUCK_RECOVERY_PAUSE_SEC",
            control.get("stuck_recovery_pause_sec", 0.5),
        ),
        stuck_recovery_reverse_sec=_float_env_or_value(
            "STUCK_RECOVERY_REVERSE_SEC",
            control.get("stuck_recovery_reverse_sec", 0.65),
        ),
        stuck_recovery_turn_sec=_float_env_or_value(
            "STUCK_RECOVERY_TURN_SEC",
            control.get("stuck_recovery_turn_sec", 0.35),
        ),
        stuck_recovery_turn_action=_env_or_value(
            "STUCK_RECOVERY_TURN_ACTION",
            control.get("stuck_recovery_turn_action", "TURN_RIGHT"),
        ),
        wheel_drive_damping=0.0 if agx_controller is None else float(agx_controller.wheel_drive_damping),
        steering_stiffness=0.0 if agx_controller is None else float(agx_controller.steering_stiffness),
        steering_damping=0.0 if agx_controller is None else float(agx_controller.steering_damping),
        hold_joint_stiffness=0.0 if agx_controller is None else float(agx_controller.hold_joint_stiffness),
        hold_joint_damping=0.0 if agx_controller is None else float(agx_controller.hold_joint_damping),
        left_wheel_joint_names=[] if agx_controller is None else list(agx_controller.left_wheel_joint_names),
        right_wheel_joint_names=[] if agx_controller is None else list(agx_controller.right_wheel_joint_names),
        steering_joint_names=[] if agx_controller is None else list(agx_controller.steering_joint_names),
        scene_usd_path=_env_or_value("SCENE_USD_PATH", scene.get("usd_path", "usd/hospital.usd")),
        scene_prim_path=_env_or_value("SCENE_PRIM_PATH", scene.get("prim_path", "/World/Environment")),
        add_default_ground_plane=_bool_env_or_value(
            "ADD_DEFAULT_GROUND_PLANE",
            scene.get("add_default_ground_plane", False),
        ),
        add_dome_light=_bool_env_or_value(
            "ADD_DOME_LIGHT",
            scene.get("add_dome_light", False),
        ),
        stage_load_timeout_sec=_float_env_or_value(
            "STAGE_LOAD_TIMEOUT_SEC",
            scene.get("stage_load_timeout_sec", 120.0),
        ),
        viewport_eye=_list_env_or_value(
            "VIEWPORT_EYE",
            scene.get("viewport_eye", [7.0, -2.0, 4.0]),
        ),
        viewport_target=_list_env_or_value(
            "VIEWPORT_TARGET",
            scene.get("viewport_target", [1.5, 4.0, 1.0]),
        ),
        tip_stop_enabled=bool(robot_profile.safety.fall_detection_enabled),
        tip_stop_roll_pitch_deg=float(robot_profile.safety.max_roll_pitch_deg),
        tip_stop_min_base_height_m=float(robot_profile.safety.min_base_height_m),
        tip_stop_confirmation_sec=float(robot_profile.safety.confirmation_sec),
        log_level=_env_or_value("LOG_LEVEL", logging_section.get("level", "INFO")),
        motion_log_period_sec=_float_env_or_value(
            "MOTION_LOG_PERIOD_SEC",
            logging_section.get("motion_log_period_sec", 5.0),
        ),
    )

    if robot_profile.is_g1 and not bool(config.use_discrete_action_primitives):
        raise ValueError(
            "unitree_g1_29dof requires control.use_discrete_action_primitives=true; "
            "TRACK_TARGET is not supported by the G1 locomotion backend"
        )

    return config


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    return value if isinstance(value, dict) else {}


def _section_value(primary: dict[str, Any], secondary: dict[str, Any], key: str, default: Any) -> Any:
    if key in primary:
        return primary.get(key, default)
    return secondary.get(key, default)


def _resolve_path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _env_or_value(name: str, default: Any) -> Any:
    return os.getenv(name, default)


def _bool_env_or_value(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int_env_or_value(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw is not None else int(default)


def _optional_int_env_or_value(name: str, default: Any) -> int | None:
    raw = os.getenv(name)
    candidate = raw if raw is not None else default
    if candidate is None:
        return None
    if isinstance(candidate, str):
        text = candidate.strip().lower()
        if text in {"", "none", "null", "auto", "display"}:
            return None
        return int(text)
    return int(candidate)


def _float_env_or_value(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw is not None else float(default)


def _list_env_or_value(name: str, default: list[Any]) -> list[Any]:
    raw = os.getenv(name)
    if raw is None:
        return list(default)

    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        return []

    parsed: list[Any] = []
    for item in items:
        try:
            parsed.append(float(item))
        except ValueError:
            parsed.append(item)
    return parsed
