from __future__ import annotations

from dataclasses import MISSING, dataclass, fields, is_dataclass
import logging
import os
from pathlib import Path
import sys
import types
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

from vln_nav.runtime.robot_profile import RobotProfile, load_robot_profile


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
T = TypeVar("T")


@dataclass
class SimulationConfig:
    headless: bool
    width: int
    height: int
    active_gpu: int | None
    physics_gpu: int | None
    multi_gpu: bool
    warmup_steps: int


@dataclass
class SceneConfig:
    usd_path: str
    prim_path: str
    add_default_ground_plane: bool
    add_dome_light: bool
    stage_load_timeout_sec: float
    viewport_eye: tuple[float, float, float]
    viewport_target: tuple[float, float, float]
    robot_start_xy_m: tuple[float, float]
    robot_start_orientation_deg: tuple[float, float, float]


@dataclass
class LLMConfig:
    api_key: str
    api_base_url: str
    model_name: str
    timeout_sec: int
    retry_cooldown_sec: float
    upload_max_edge_px: int
    camera_image_detail: str


@dataclass
class InstructionConfig:
    text: str
    parse_max_tokens: int
    parse_timeout_sec: int
    cache_enabled: bool
    cache_path: str


@dataclass
class GroundingDetectorConfig:
    repo_path: str
    config_path: str
    checkpoint_path: str
    device: str
    box_threshold: float
    text_threshold: float
    max_detections: int
    nms_iou_threshold: float
    text_encoder_path: str


@dataclass
class SemanticVerificationConfig:
    min_confidence: float
    max_edge_px: int
    auto_accept_confidence: float


@dataclass
class TargetTrackerRuntimeConfig:
    tracker_type: str
    max_lost_frames: int
    lost_confirmed_sec: float
    min_depth_valid_ratio: float
    center_patch_ratio: float
    confidence_decay_per_lost_frame: float
    min_tracking_confidence: float


@dataclass
class GroundingRuntimeConfig:
    call_period_sec: float
    reuse_last_observation_sec: float
    upload_max_edge_px: int
    min_bbox_area_ratio: float
    max_bbox_area_ratio: float
    min_bbox_width_px: int
    min_bbox_height_px: int
    max_bbox_aspect_ratio: float
    min_depth_valid_ratio: float
    depth_statistic: str
    center_patch_ratio: float
    min_depth_m: float
    max_depth_m: float
    min_confidence: float
    coordinate_mode: str
    allow_normalized_bbox: bool
    normalized_bbox_scale: float
    depth_region: str
    depth_cluster_tolerance_ratio: float
    depth_cluster_min_pixels: int
    max_result_age_sec: float
    max_result_pose_delta_m: float
    max_result_yaw_delta_deg: float
    detector: GroundingDetectorConfig
    semantic_verification: SemanticVerificationConfig
    tracker: TargetTrackerRuntimeConfig


@dataclass
class LandmarkTrackingRuntimeConfig:
    min_visible_count: int
    arrival_confirmations: int
    arrival_window_size: int
    arrival_recent_visible_sec: float
    max_lost_count: int
    anchor_update_min_consistent_observations: int
    anchor_update_max_jump_m: float
    confirm_confidence_threshold: float
    potential_confidence_threshold: float
    confirm_min_observations: int
    confirm_max_anchor_jump_m: float
    confirm_max_depth_jump_m: float
    low_confidence_min_translation_m: float
    low_confidence_min_yaw_deg: float


@dataclass
class LandmarkMapRuntimeConfig:
    anchor_ttl_sec: float
    min_confidence: float
    log_period_sec: float


@dataclass
class PerceptionConfig:
    grounding: GroundingRuntimeConfig
    landmark_tracking: LandmarkTrackingRuntimeConfig
    landmark_map: LandmarkMapRuntimeConfig


@dataclass
class OccupancyRuntimeConfig:
    resolution_m: float
    size_m: float
    update_period_sec: float
    occupied_min_neighbors: int
    unknown_is_traversable: bool
    unknown_cell_cost: float
    lidar_bootstrap_sec: float
    lidar_bootstrap_min_fov_deg: float
    lidar_bootstrap_max_wait_sec: float
    semantic_goal_step_m: float
    semantic_goal_min_step_m: float
    semantic_goal_lateral_search_m: float


@dataclass
class TopologyRuntimeConfig:
    node_spacing_m: float
    turn_threshold_deg: float
    merge_radius_m: float
    landmark_bind_radius_m: float
    max_nodes: int
    recent_node_count: int


@dataclass
class MappingConfig:
    occupancy: OccupancyRuntimeConfig
    topology: TopologyRuntimeConfig


@dataclass
class CandidateGenerationConfig:
    max_candidates: int
    min_distance_m: float
    max_distance_m: float
    slot_bearings_deg: tuple[float, ...]
    adaptive_count: int
    adaptive_min_spacing_deg: float
    adaptive_smoothing_px: int
    min_free_ratio: float
    column_half_width_px: int
    unknown_gain_weight: float
    semantic_weight: float


@dataclass
class RegionTransitionRuntimeConfig:
    min_displacement_m: float
    confirmations: int
    min_confidence: float


@dataclass
class SemanticTurnRuntimeConfig:
    min_yaw_deg: float
    max_yaw_deg: float
    max_primitives: int
    min_forward_clearance_m: float
    alignment_confirmations: int
    corridor_center_tolerance_deg: float
    same_corridor_tolerance_deg: float
    target_high_confidence: float
    target_medium_confidence: float
    direction_tolerance_deg: float
    corridor_half_fov_deg: float
    corridor_step_deg: float
    lidar_bin_half_width_deg: float
    min_corridor_width_deg: float
    corridor_max_clearance_m: float
    max_recovery_attempts: int


@dataclass
class NavigationPolicyRuntimeConfig:
    replanning_cooldown_sec: float
    visual_target_center_deadband_deg: float
    local_detour_max_turn_primitives: int
    region_transition: RegionTransitionRuntimeConfig
    semantic_turn: SemanticTurnRuntimeConfig


@dataclass
class ProgressRuntimeConfig:
    history_length: int
    stall_distance_m: float
    spin_yaw_deg: float
    landmark_clearance_m: float
    final_success_clearance_m: float
    landmark_arrival_confirmations: int


@dataclass
class RecoveryRuntimeConfig:
    stuck_window_sec: float
    stuck_displacement_threshold_m: float
    stuck_yaw_threshold_deg: float
    pause_sec: float
    reverse_sec: float
    turn_sec: float
    turn_action: str


@dataclass
class NavigationConfig:
    candidates: CandidateGenerationConfig
    policy: NavigationPolicyRuntimeConfig
    progress: ProgressRuntimeConfig
    recovery: RecoveryRuntimeConfig


@dataclass
class DebugImageConfig:
    enabled: bool
    path: str
    update_period_sec: float
    depth_enabled: bool
    depth_path: str


@dataclass
class DebugVideoConfig:
    enabled: bool
    path: str
    fps: float


@dataclass
class ObserverVideoConfig:
    enabled: bool
    path: str
    fps: float


@dataclass
class DebugConfig:
    image: DebugImageConfig
    video: DebugVideoConfig
    observer_video: ObserverVideoConfig


@dataclass
class LoggingConfig:
    level: str
    motion_log_period_sec: float


@dataclass
class AppConfig:
    simulation: SimulationConfig
    robot: RobotProfile
    scene: SceneConfig
    llm: LLMConfig
    instruction: InstructionConfig
    perception: PerceptionConfig
    mapping: MappingConfig
    navigation: NavigationConfig
    debug: DebugConfig
    logging: LoggingConfig

    @property
    def robot_start_position(self) -> list[float]:
        return [
            float(self.scene.robot_start_xy_m[0]),
            float(self.scene.robot_start_xy_m[1]),
            float(self.robot.spawn_z_m),
        ]


@dataclass
class _RobotSelection:
    profile: str


@dataclass
class _FileConfig:
    simulation: SimulationConfig
    robot: _RobotSelection
    scene: SceneConfig
    llm: LLMConfig
    instruction: InstructionConfig
    perception: PerceptionConfig
    mapping: MappingConfig
    navigation: NavigationConfig
    debug: DebugConfig
    logging: LoggingConfig


def load_config(config_path: str | None = None) -> AppConfig:
    selected_path = Path(config_path or os.getenv("CONFIG_PATH") or DEFAULT_CONFIG_PATH)
    selected_path = selected_path if selected_path.is_absolute() else PROJECT_ROOT / selected_path
    if not selected_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {selected_path}")

    base_data = _read_yaml(DEFAULT_CONFIG_PATH)
    selected_data = base_data if selected_path.resolve() == DEFAULT_CONFIG_PATH.resolve() else _deep_merge(
        base_data,
        _read_yaml(selected_path),
    )
    parsed = _decode_dataclass(_FileConfig, selected_data, path="config")
    profile = load_robot_profile(
        profile_name=parsed.robot.profile,
        project_root=PROJECT_ROOT,
    )
    config = AppConfig(
        simulation=parsed.simulation,
        robot=profile,
        scene=parsed.scene,
        llm=parsed.llm,
        instruction=parsed.instruction,
        perception=parsed.perception,
        mapping=parsed.mapping,
        navigation=parsed.navigation,
        debug=parsed.debug,
        logging=parsed.logging,
    )
    _apply_environment_overrides(config)
    _normalize_paths(config)
    _validate_config(config)
    return config


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return payload


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key, value in base.items():
        merged[key] = _deep_copy(value)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = _deep_copy(value)
    return merged


def _deep_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _deep_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_copy(item) for item in value]
    return value


def _decode_dataclass(cls: type[T], payload: Any, *, path: str) -> T:
    if not is_dataclass(cls) or not isinstance(payload, dict):
        raise ValueError(f"{path} must be a mapping for {getattr(cls, '__name__', cls)}")
    field_map = {item.name: item for item in fields(cls)}
    unknown = sorted(set(payload) - set(field_map))
    if unknown:
        raise ValueError(f"Unknown config fields under {path}: {unknown}")
    type_hints = get_type_hints(cls)
    values: dict[str, Any] = {}
    for name, item in field_map.items():
        if name not in payload:
            if item.default is not MISSING or item.default_factory is not MISSING:
                continue
            raise ValueError(f"Missing required config field: {path}.{name}")
        values[name] = _decode_value(
            type_hints[name],
            payload[name],
            path=f"{path}.{name}",
        )
    return cls(**values)


def _decode_value(expected_type: Any, value: Any, *, path: str) -> Any:
    if is_dataclass(expected_type):
        return _decode_dataclass(expected_type, value, path=path)
    origin = get_origin(expected_type)
    args = get_args(expected_type)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{path} must be a list")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode_value(args[0], item, path=f"{path}[]") for item in value)
        if len(value) != len(args):
            raise ValueError(f"{path} must contain exactly {len(args)} values")
        return tuple(
            _decode_value(item_type, item, path=f"{path}[{index}]")
            for index, (item_type, item) in enumerate(zip(args, value))
        )
    if origin in {Union, types.UnionType}:
        if value is None and type(None) in args:
            return None
        for member in args:
            if member is type(None):
                continue
            try:
                return _decode_value(member, value, path=path)
            except (TypeError, ValueError):
                continue
        raise ValueError(f"{path} does not match {expected_type}")
    if expected_type is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{path} must be a boolean")
        return value
    if expected_type is int:
        if isinstance(value, bool):
            raise ValueError(f"{path} must be an integer")
        return int(value)
    if expected_type is float:
        if isinstance(value, bool):
            raise ValueError(f"{path} must be numeric")
        return float(value)
    if expected_type is str:
        if value is None:
            raise ValueError(f"{path} must be a string")
        return str(value)
    return value


def _apply_environment_overrides(config: AppConfig) -> None:
    config.llm.api_key = _first_env("SILICONFLOW_API_KEY", "LLM_API_KEY") or config.llm.api_key
    config.llm.model_name = os.getenv("MODEL_NAME", config.llm.model_name)
    config.llm.api_base_url = os.getenv("API_BASE_URL", config.llm.api_base_url)
    config.instruction.text = os.getenv("INSTRUCTION", config.instruction.text)
    config.scene.usd_path = os.getenv("SCENE_USD_PATH", config.scene.usd_path)
    config.debug.image.path = os.getenv("TARGET_DEBUG_IMAGE_PATH", config.debug.image.path)
    config.debug.image.depth_path = os.getenv("DEPTH_DEBUG_IMAGE_PATH", config.debug.image.depth_path)
    config.debug.video.path = os.getenv("DEBUG_VIDEO_PATH", config.debug.video.path)
    config.logging.level = os.getenv("LOG_LEVEL", config.logging.level)

    if "HEADLESS" in os.environ:
        config.simulation.headless = _parse_bool(os.environ["HEADLESS"], "HEADLESS")
    if "ADD_DOME_LIGHT" in os.environ:
        config.scene.add_dome_light = _parse_bool(os.environ["ADD_DOME_LIGHT"], "ADD_DOME_LIGHT")
    if "ACTIVE_GPU" in os.environ:
        config.simulation.active_gpu = _parse_optional_int(os.environ["ACTIVE_GPU"])
    if "PHYSICS_GPU" in os.environ:
        config.simulation.physics_gpu = _parse_optional_int(os.environ["PHYSICS_GPU"])


def _normalize_paths(config: AppConfig) -> None:
    config.instruction.cache_path = str(_project_path(config.instruction.cache_path))
    grounding = config.perception.grounding
    grounding.detector.repo_path = str(_project_path(grounding.detector.repo_path))
    grounding.detector.config_path = str(_project_path(grounding.detector.config_path))
    grounding.detector.checkpoint_path = str(_project_path(grounding.detector.checkpoint_path))
    grounding.detector.text_encoder_path = str(_project_path(grounding.detector.text_encoder_path))
    config.debug.image.path = str(_project_path(config.debug.image.path))
    config.debug.image.depth_path = str(_project_path(config.debug.image.depth_path))
    config.debug.video.path = str(_project_path(config.debug.video.path))
    config.debug.observer_video.path = str(_project_path(config.debug.observer_video.path))


def _validate_config(config: AppConfig) -> None:
    if config.navigation.candidates.min_distance_m > config.navigation.candidates.max_distance_m:
        raise ValueError("navigation.candidates.min_distance_m must not exceed max_distance_m")
    if config.mapping.occupancy.resolution_m <= 0.0:
        raise ValueError("mapping.occupancy.resolution_m must be > 0")
    if config.navigation.progress.landmark_clearance_m <= 0.0:
        raise ValueError("navigation.progress.landmark_clearance_m must be > 0")
    if config.navigation.progress.final_success_clearance_m <= 0.0:
        raise ValueError("navigation.progress.final_success_clearance_m must be > 0")
    if config.navigation.recovery.turn_action not in {"TURN_LEFT", "TURN_RIGHT"}:
        raise ValueError("navigation.recovery.turn_action must be TURN_LEFT or TURN_RIGHT")


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
        "[SAFETY_REJECT]",
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


def _project_path(value: str) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _parse_bool(value: str, name: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _parse_optional_int(value: str) -> int | None:
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "null", "auto"}:
        return None
    return int(normalized)
