from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
DEFAULT_ROBOT_USD_PATH = (
    PROJECT_ROOT
    / "src"
    / "ros2_car_model_ws"
    / "src"
    / "1557"
    / "urdf"
    / "1557_all"
    / "1557_all.usd"
)


@dataclass
class AppConfig:
    llm_api_key: str = ""
    model_name: str = "Qwen/Qwen3-VL-235B-A22B-Instruct"
    api_base_url: str = "https://api.siliconflow.cn/v1"
    llm_timeout_sec: int = 30
    llm_retry_cooldown_sec: float = 0.0
    llm_upload_max_edge_px: int = 512
    camera_image_detail: str = "low"
    instruction: str = "find the wheelchair"
    candidate_slot_bearings_deg: list[float] = field(default_factory=lambda: [75.0, 35.0, 0.0, -35.0, -75.0])
    candidate_min_travel_distance_m: float = 0.55
    candidate_max_travel_distance_m: float = 1.35
    candidate_min_free_ratio: float = 0.28
    candidate_column_half_width_px: int = 26
    candidate_debug_image_path: str = "assets/opennav_request.png"
    target_debug_enabled: bool = True
    target_debug_image_path: str = "assets/opennav_debug.png"
    debug_video_enabled: bool = True
    debug_video_path: str = "assets/opennav_debug.mp4"
    debug_video_fps: float = 4.0
    debug_image_update_period_sec: float = 1.0
    progress_history_length: int = 6
    progress_stall_distance_m: float = 0.08
    progress_spin_yaw_deg: float = 85.0
    policy_execution_steps: int = 24
    landmark_arrival_distance_m: float = 1.0
    landmark_arrival_confirmations: int = 2
    recovery_enabled: bool = True
    recovery_trigger_stalled_steps: int = 3
    recovery_execution_steps: int = 18
    target_grounding_enabled: bool = True
    target_grounding_backend: str = "auto"
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
    target_grounding_max_depth_m: float = 6.0
    target_grounding_min_confidence: float = 0.20
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
    target_grounding_dino_box_threshold: float = 0.30
    target_grounding_dino_text_threshold: float = 0.25
    target_grounding_sam_checkpoint_path: str = str(PROJECT_ROOT / "weights" / "grounded_sam" / "sam_vit_h_4b8939.pth")
    target_grounding_sam_model_type: str = "vit_h"
    target_grounding_sam_device: str = "cuda"
    target_grounding_sam_min_mask_area_ratio: float = 0.0005
    landmark_tracker_arrival_visible_depth_m: float = 1.00
    landmark_tracker_arrival_anchor_radius_m: float = 1.00
    landmark_tracker_min_visible_count: int = 2
    landmark_tracker_arrival_confirmations: int = 2
    landmark_tracker_max_lost_count: int = 6
    landmark_tracker_anchor_update_min_consistent_observations: int = 2
    landmark_tracker_anchor_update_max_jump_m: float = 1.5
    landmark_map_enabled: bool = True
    landmark_map_anchor_ttl_sec: float = 120.0
    landmark_map_min_confidence: float = 0.20
    landmark_map_log_period_sec: float = 2.0
    reacquire_guard_enabled: bool = True
    reacquire_guard_recent_visible_sec: float = 20.0
    reacquire_guard_override_action: str = "STOP"
    depth_debug_enabled: bool = True
    depth_debug_image_path: str = "assets/depth_debug.png"

    headless: bool = False
    sim_width: int = 1280
    sim_height: int = 720
    physics_dt: float = 1.0 / 60.0
    rendering_dt: float = 1.0 / 60.0
    warmup_steps: int = 30
    active_gpu: int | None = None
    physics_gpu: int | None = 0
    multi_gpu: bool = False

    robot_usd_path: Path = DEFAULT_ROBOT_USD_PATH
    robot_prim_path: str = "/World/Robot"
    robot_articulation_path: str = ""
    robot_init_position: list[float] = field(default_factory=lambda: [1.5, 4.0, 0.22])
    robot_init_orientation_deg: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    camera_width: int = 640
    camera_height: int = 480
    front_camera_prim_path: str = "link6/Camera"
    observer_camera_enabled: bool = True
    observer_camera_prim_path: str = "dummy_base_link/observecamera"
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
    action_backward_duration_sec: float = 1.4
    action_turn_duration_sec: float = 0.30
    action_turn_angle_deg: float = 15.0
    action_forward_distance_m: float = 0.55
    action_backward_distance_m: float = 0.35
    action_distance_tolerance_m: float = 0.04
    action_turn_tolerance_deg: float = 1.5
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
    occupancy_grid_enabled: bool = True
    occupancy_grid_resolution_m: float = 0.08
    occupancy_grid_size_m: float = 24.0
    occupancy_grid_update_period_sec: float = 0.12
    occupancy_grid_depth_sample_stride_px: int = 10
    occupancy_grid_depth_roi_vertical_range: list[float] = field(default_factory=lambda: [0.38, 0.72])
    occupancy_grid_min_depth_m: float = 0.20
    occupancy_grid_max_depth_m: float = 5.0
    occupancy_grid_inflation_radius_m: float = 0.26
    occupancy_grid_occupied_min_neighbors: int = 3
    occupancy_grid_robot_radius_m: float = 0.32
    occupancy_grid_path_corridor_radius_m: float = 0.30
    occupancy_grid_turn_clearance_radius_m: float = 0.36
    occupancy_grid_unknown_is_traversable: bool = True
    occupancy_grid_unknown_cell_cost: float = 0.10
    occupancy_grid_debug_enabled: bool = True
    occupancy_grid_debug_image_path: str = "assets/occupancy_grid.png"
    occupancy_grid_debug_update_period_sec: float = 0.5
    occupancy_grid_debug_width: int = 900
    occupancy_grid_debug_height: int = 900
    occupancy_grid_debug_view_size_m: float = 8.0
    occupancy_grid_debug_show_free_space: bool = False
    stuck_detection_enabled: bool = True
    stuck_window_sec: float = 2.0
    stuck_displacement_threshold_m: float = 0.05
    stuck_yaw_threshold_deg: float = 5.0
    stuck_recovery_backward_sec: float = 0.75
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
    stage_load_timeout_sec: float = 120.0
    viewport_eye: list[float] = field(default_factory=lambda: [7.0, -2.0, 4.0])
    viewport_target: list[float] = field(default_factory=lambda: [1.5, 4.0, 1.0])
    tip_stop_enabled: bool = False
    tip_stop_roll_pitch_deg: float = 28.0

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
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    return logger


def load_config(config_path: str | None = None) -> AppConfig:
    selected_path = Path(config_path or os.getenv("CONFIG_PATH") or DEFAULT_CONFIG_PATH)
    selected_path = selected_path if selected_path.is_absolute() else PROJECT_ROOT / selected_path

    file_data: dict[str, Any] = {}
    if selected_path.exists():
        with selected_path.open("r", encoding="utf-8") as handle:
            file_data = yaml.safe_load(handle) or {}

    sim = _section(file_data, "sim")
    robot = _section(file_data, "robot")
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
        instruction=_env_or_value(
            "INSTRUCTION",
            opennav.get("instruction", "find the wheelchair"),
        ),
        candidate_slot_bearings_deg=_list_env_or_value(
            "CANDIDATE_SLOT_BEARINGS_DEG",
            opennav.get("candidate_slot_bearings_deg", [75.0, 35.0, 0.0, -35.0, -75.0]),
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
        candidate_debug_image_path=_env_or_value(
            "CANDIDATE_DEBUG_IMAGE_PATH",
            opennav.get("candidate_debug_image_path", "assets/opennav_request.png"),
        ),
        target_debug_enabled=_bool_env_or_value(
            "TARGET_DEBUG_ENABLED",
            opennav.get("target_debug_enabled", True),
        ),
        target_debug_image_path=_env_or_value(
            "TARGET_DEBUG_IMAGE_PATH",
            opennav.get("target_debug_image_path", "assets/opennav_debug.png"),
        ),
        debug_video_enabled=_bool_env_or_value(
            "DEBUG_VIDEO_ENABLED",
            opennav.get("debug_video_enabled", True),
        ),
        debug_video_path=_env_or_value(
            "DEBUG_VIDEO_PATH",
            opennav.get("debug_video_path", "assets/opennav_debug.mp4"),
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
        landmark_arrival_distance_m=_float_env_or_value(
            "LANDMARK_ARRIVAL_DISTANCE_M",
            opennav.get("landmark_arrival_distance_m", 1.0),
        ),
        landmark_arrival_confirmations=_int_env_or_value(
            "LANDMARK_ARRIVAL_CONFIRMATIONS",
            opennav.get("landmark_arrival_confirmations", 2),
        ),
        recovery_enabled=_bool_env_or_value(
            "RECOVERY_ENABLED",
            opennav.get("recovery_enabled", True),
        ),
        recovery_trigger_stalled_steps=_int_env_or_value(
            "RECOVERY_TRIGGER_STALLED_STEPS",
            opennav.get("recovery_trigger_stalled_steps", 3),
        ),
        recovery_execution_steps=_int_env_or_value(
            "RECOVERY_EXECUTION_STEPS",
            opennav.get("recovery_execution_steps", 18),
        ),
        target_grounding_enabled=_bool_env_or_value(
            "TARGET_GROUNDING_ENABLED",
            target_grounding.get("enabled", True),
        ),
        target_grounding_backend=_env_or_value(
            "TARGET_GROUNDING_BACKEND",
            target_grounding.get("backend", "auto"),
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
            target_grounding.get("min_confidence", 0.20),
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
            target_grounding.get("grounding_dino_box_threshold", 0.30),
        ),
        target_grounding_dino_text_threshold=_float_env_or_value(
            "GROUNDING_DINO_TEXT_THRESHOLD",
            target_grounding.get("grounding_dino_text_threshold", 0.25),
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
        landmark_tracker_arrival_visible_depth_m=_float_env_or_value(
            "LANDMARK_TRACKER_ARRIVAL_VISIBLE_DEPTH_M",
            landmark_tracker.get("arrival_visible_depth_m", 1.00),
        ),
        landmark_tracker_arrival_anchor_radius_m=_float_env_or_value(
            "LANDMARK_TRACKER_ARRIVAL_ANCHOR_RADIUS_M",
            landmark_tracker.get("arrival_anchor_radius_m", 1.00),
        ),
        landmark_tracker_min_visible_count=_int_env_or_value(
            "LANDMARK_TRACKER_MIN_VISIBLE_COUNT",
            landmark_tracker.get("min_visible_count", 2),
        ),
        landmark_tracker_arrival_confirmations=_int_env_or_value(
            "LANDMARK_TRACKER_ARRIVAL_CONFIRMATIONS",
            landmark_tracker.get("arrival_confirmations", 2),
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
        reacquire_guard_enabled=_bool_env_or_value(
            "REACQUIRE_GUARD_ENABLED",
            landmark_tracker.get("reacquire_guard_enabled", True),
        ),
        reacquire_guard_recent_visible_sec=_float_env_or_value(
            "REACQUIRE_GUARD_RECENT_VISIBLE_SEC",
            landmark_tracker.get("reacquire_guard_recent_visible_sec", 20.0),
        ),
        reacquire_guard_override_action=_env_or_value(
            "REACQUIRE_GUARD_OVERRIDE_ACTION",
            landmark_tracker.get("reacquire_guard_override_action", "STOP"),
        ),
        depth_debug_enabled=_bool_env_or_value(
            "DEPTH_DEBUG_ENABLED",
            camera.get("depth_debug_enabled", True),
        ),
        depth_debug_image_path=_env_or_value(
            "DEPTH_DEBUG_IMAGE_PATH",
            camera.get("depth_debug_image_path", "assets/depth_debug.png"),
        ),
        headless=_bool_env_or_value("HEADLESS", sim.get("headless", False)),
        sim_width=_int_env_or_value("SIM_WIDTH", sim.get("width", 1280)),
        sim_height=_int_env_or_value("SIM_HEIGHT", sim.get("height", 720)),
        active_gpu=_optional_int_env_or_value("ACTIVE_GPU", sim.get("active_gpu", None)),
        physics_gpu=_optional_int_env_or_value("PHYSICS_GPU", sim.get("physics_gpu", 0)),
        multi_gpu=_bool_env_or_value("MULTI_GPU", sim.get("multi_gpu", False)),
        physics_dt=_float_env_or_value("PHYSICS_DT", sim.get("physics_dt", 1.0 / 60.0)),
        rendering_dt=_float_env_or_value("RENDERING_DT", sim.get("rendering_dt", 1.0 / 60.0)),
        warmup_steps=_int_env_or_value("WARMUP_STEPS", sim.get("warmup_steps", 30)),
        robot_usd_path=_resolve_path(
            _env_or_value("ROBOT_USD_PATH", robot.get("usd_path", str(DEFAULT_ROBOT_USD_PATH)))
        ),
        robot_prim_path=_env_or_value("ROBOT_PRIM_PATH", robot.get("prim_path", "/World/Robot")),
        robot_articulation_path=_env_or_value(
            "ROBOT_ARTICULATION_PATH", robot.get("articulation_path", "")
        ),
        robot_init_position=_list_env_or_value(
            "ROBOT_INIT_POSITION", robot.get("init_position", [1.5, 4.0, 0.22])
        ),
        robot_init_orientation_deg=_list_env_or_value(
            "ROBOT_INIT_ORIENTATION_DEG",
            robot.get("init_orientation_deg", [0.0, 0.0, 0.0]),
        ),
        camera_width=_int_env_or_value("CAMERA_WIDTH", camera.get("width", 640)),
        camera_height=_int_env_or_value("CAMERA_HEIGHT", camera.get("height", 480)),
        front_camera_prim_path=_env_or_value(
            "FRONT_CAMERA_PRIM_PATH",
            camera.get("front_camera_prim_path", "link6/Camera"),
        ),
        observer_camera_enabled=_bool_env_or_value(
            "OBSERVER_CAMERA_ENABLED",
            camera.get("observer_enabled", True),
        ),
        observer_camera_prim_path=_env_or_value(
            "OBSERVER_CAMERA_PRIM_PATH",
            camera.get("observer_camera_prim_path", "dummy_base_link/observecamera"),
        ),
        follow_front_camera_in_gui=_bool_env_or_value(
            "FOLLOW_FRONT_CAMERA_IN_GUI",
            camera.get("follow_front_camera_in_gui", False),
        ),
        follow_observer_camera_in_gui=_bool_env_or_value(
            "FOLLOW_OBSERVER_CAMERA_IN_GUI",
            camera.get("follow_observer_camera_in_gui", True),
        ),
        sensor_frequency_hz=_int_env_or_value(
            "SENSOR_FREQUENCY_HZ",
            camera.get("sensor_frequency_hz", 15),
        ),
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
        obstacle_stop_distance_m=_float_env_or_value(
            "OBSTACLE_STOP_DISTANCE_M",
            camera.get("obstacle_stop_distance_m", 0.70),
        ),
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
        wheel_drive_velocity=_float_env_or_value(
            "WHEEL_DRIVE_VELOCITY", control.get("wheel_drive_velocity", 14.0)
        ),
        turn_drive_velocity=_float_env_or_value(
            "TURN_DRIVE_VELOCITY", control.get("turn_drive_velocity", 6.0)
        ),
        steering_angle=_float_env_or_value("STEERING_ANGLE", control.get("steering_angle", 0.65)),
        turn_mode=_env_or_value("TURN_MODE", control.get("turn_mode", "spin_4ws")),
        turn_steering_angle=_float_env_or_value(
            "TURN_STEERING_ANGLE",
            control.get("turn_steering_angle", 0.50),
        ),
        rear_steering_ratio=_float_env_or_value(
            "REAR_STEERING_RATIO",
            control.get("rear_steering_ratio", 1.0),
        ),
        track_min_speed_scale=_float_env_or_value(
            "TRACK_MIN_SPEED_SCALE",
            control.get("track_min_speed_scale", 0.45),
        ),
        track_max_speed_scale=_float_env_or_value(
            "TRACK_MAX_SPEED_SCALE",
            control.get("track_max_speed_scale", 1.0),
        ),
        track_heading_gain=_float_env_or_value(
            "TRACK_HEADING_GAIN",
            control.get("track_heading_gain", 1.15),
        ),
        track_slow_heading_rad=_float_env_or_value(
            "TRACK_SLOW_HEADING_RAD",
            control.get("track_slow_heading_rad", 1.35),
        ),
        track_smoothing_alpha=_float_env_or_value(
            "TRACK_SMOOTHING_ALPHA",
            control.get("track_smoothing_alpha", 0.92),
        ),
        track_steering_sign=_float_env_or_value(
            "TRACK_STEERING_SIGN",
            control.get("track_steering_sign", -1.0),
        ),
        action_forward_duration_sec=_float_env_or_value(
            "ACTION_FORWARD_DURATION_SEC",
            control.get("action_forward_duration_sec", 0.8),
        ),
        action_backward_duration_sec=_float_env_or_value(
            "ACTION_BACKWARD_DURATION_SEC",
            control.get("action_backward_duration_sec", 1.4),
        ),
        action_turn_duration_sec=_float_env_or_value(
            "ACTION_TURN_DURATION_SEC",
            control.get("action_turn_duration_sec", 0.30),
        ),
        action_turn_angle_deg=_float_env_or_value(
            "ACTION_TURN_ANGLE_DEG",
            control.get("action_turn_angle_deg", 15.0),
        ),
        action_forward_distance_m=_float_env_or_value(
            "ACTION_FORWARD_DISTANCE_M",
            control.get("action_forward_distance_m", 0.55),
        ),
        action_backward_distance_m=_float_env_or_value(
            "ACTION_BACKWARD_DISTANCE_M",
            control.get("action_backward_distance_m", 0.35),
        ),
        action_distance_tolerance_m=_float_env_or_value(
            "ACTION_DISTANCE_TOLERANCE_M",
            control.get("action_distance_tolerance_m", 0.04),
        ),
        action_turn_tolerance_deg=_float_env_or_value(
            "ACTION_TURN_TOLERANCE_DEG",
            control.get("action_turn_tolerance_deg", 1.5),
        ),
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
            occupancy_grid.get("max_depth_m", 5.0),
        ),
        occupancy_grid_inflation_radius_m=_float_env_or_value(
            "OCCUPANCY_GRID_INFLATION_RADIUS_M",
            occupancy_grid.get("inflation_radius_m", 0.26),
        ),
        occupancy_grid_occupied_min_neighbors=_int_env_or_value(
            "OCCUPANCY_GRID_OCCUPIED_MIN_NEIGHBORS",
            occupancy_grid.get("occupied_min_neighbors", 3),
        ),
        occupancy_grid_robot_radius_m=_float_env_or_value(
            "OCCUPANCY_GRID_ROBOT_RADIUS_M",
            occupancy_grid.get("robot_radius_m", 0.32),
        ),
        occupancy_grid_path_corridor_radius_m=_float_env_or_value(
            "OCCUPANCY_GRID_PATH_CORRIDOR_RADIUS_M",
            occupancy_grid.get("path_corridor_radius_m", 0.30),
        ),
        occupancy_grid_turn_clearance_radius_m=_float_env_or_value(
            "OCCUPANCY_GRID_TURN_CLEARANCE_RADIUS_M",
            occupancy_grid.get("turn_clearance_radius_m", 0.36),
        ),
        occupancy_grid_unknown_is_traversable=_bool_env_or_value(
            "OCCUPANCY_GRID_UNKNOWN_IS_TRAVERSABLE",
            occupancy_grid.get("unknown_is_traversable", True),
        ),
        occupancy_grid_unknown_cell_cost=_float_env_or_value(
            "OCCUPANCY_GRID_UNKNOWN_CELL_COST",
            occupancy_grid.get("unknown_cell_cost", 0.10),
        ),
        occupancy_grid_debug_enabled=_bool_env_or_value(
            "OCCUPANCY_GRID_DEBUG_ENABLED",
            occupancy_grid.get("debug_enabled", True),
        ),
        occupancy_grid_debug_image_path=_env_or_value(
            "OCCUPANCY_GRID_DEBUG_IMAGE_PATH",
            occupancy_grid.get("debug_image_path", "assets/occupancy_grid.png"),
        ),
        occupancy_grid_debug_update_period_sec=_float_env_or_value(
            "OCCUPANCY_GRID_DEBUG_UPDATE_PERIOD_SEC",
            occupancy_grid.get("debug_update_period_sec", 0.5),
        ),
        occupancy_grid_debug_width=_int_env_or_value(
            "OCCUPANCY_GRID_DEBUG_WIDTH",
            occupancy_grid.get("debug_width", 900),
        ),
        occupancy_grid_debug_height=_int_env_or_value(
            "OCCUPANCY_GRID_DEBUG_HEIGHT",
            occupancy_grid.get("debug_height", 900),
        ),
        occupancy_grid_debug_view_size_m=_float_env_or_value(
            "OCCUPANCY_GRID_DEBUG_VIEW_SIZE_M",
            occupancy_grid.get("debug_view_size_m", 8.0),
        ),
        occupancy_grid_debug_show_free_space=_bool_env_or_value(
            "OCCUPANCY_GRID_DEBUG_SHOW_FREE_SPACE",
            occupancy_grid.get("debug_show_free_space", False),
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
        stuck_recovery_backward_sec=_float_env_or_value(
            "STUCK_RECOVERY_BACKWARD_SEC",
            control.get("stuck_recovery_backward_sec", 0.75),
        ),
        stuck_recovery_turn_sec=_float_env_or_value(
            "STUCK_RECOVERY_TURN_SEC",
            control.get("stuck_recovery_turn_sec", 0.35),
        ),
        stuck_recovery_turn_action=_env_or_value(
            "STUCK_RECOVERY_TURN_ACTION",
            control.get("stuck_recovery_turn_action", "TURN_RIGHT"),
        ),
        wheel_drive_damping=_float_env_or_value(
            "WHEEL_DRIVE_DAMPING", control.get("wheel_drive_damping", 10000.0)
        ),
        steering_stiffness=_float_env_or_value(
            "STEERING_STIFFNESS", control.get("steering_stiffness", 10000.0)
        ),
        steering_damping=_float_env_or_value(
            "STEERING_DAMPING", control.get("steering_damping", 1000.0)
        ),
        hold_joint_stiffness=_float_env_or_value(
            "HOLD_JOINT_STIFFNESS", control.get("hold_joint_stiffness", 20000.0)
        ),
        hold_joint_damping=_float_env_or_value(
            "HOLD_JOINT_DAMPING", control.get("hold_joint_damping", 2000.0)
        ),
        left_wheel_joint_names=_list_env_or_value(
            "LEFT_WHEEL_JOINTS", robot.get("left_wheel_joint_names", [])
        ),
        right_wheel_joint_names=_list_env_or_value(
            "RIGHT_WHEEL_JOINTS", robot.get("right_wheel_joint_names", [])
        ),
        steering_joint_names=_list_env_or_value(
            "STEERING_JOINTS", robot.get("steering_joint_names", [])
        ),
        scene_usd_path=_env_or_value("SCENE_USD_PATH", scene.get("usd_path", "usd/hospital.usd")),
        scene_prim_path=_env_or_value("SCENE_PRIM_PATH", scene.get("prim_path", "/World/Environment")),
        add_default_ground_plane=_bool_env_or_value(
            "ADD_DEFAULT_GROUND_PLANE",
            scene.get("add_default_ground_plane", False),
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
        tip_stop_enabled=_bool_env_or_value(
            "TIP_STOP_ENABLED",
            scene.get("tip_stop_enabled", False),
        ),
        tip_stop_roll_pitch_deg=_float_env_or_value(
            "TIP_STOP_ROLL_PITCH_DEG",
            scene.get("tip_stop_roll_pitch_deg", 28.0),
        ),
        log_level=_env_or_value("LOG_LEVEL", logging_section.get("level", "INFO")),
        motion_log_period_sec=_float_env_or_value(
            "MOTION_LOG_PERIOD_SEC",
            logging_section.get("motion_log_period_sec", 5.0),
        ),
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


def _optional_int_env_or_value(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    if raw is not None:
        candidate = raw.strip()
        if candidate == "" or candidate.lower() == "none":
            return None
        return int(candidate)
    if default is None:
        return None
    return int(default)


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
