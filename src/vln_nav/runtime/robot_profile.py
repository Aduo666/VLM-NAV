from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROFILE_SCHEMA_VERSION = "1.0"
SUPPORTED_PROFILES = {"agx", "unitree_g1_29dof"}
SUPPORTED_CONTROLLER_TYPES = {"agx_4ws", "unitree_g1_policy"}


@dataclass(frozen=True)
class RobotAssetProfile:
    usd_path: Path
    prim_path: str
    articulation_path: str


@dataclass(frozen=True)
class RobotSimulationProfile:
    physics_dt: float
    rendering_dt: float
    policy_decimation: int


@dataclass(frozen=True)
class RobotSensorProfile:
    front_camera_prim_path: str
    observer_camera_enabled: bool
    observer_camera_prim_path: str
    camera_frequency_hz: int
    lidar_prim_path: str


@dataclass(frozen=True)
class PrimitiveProfile:
    forward_distance_m: float
    turn_angle_deg: float
    distance_tolerance_m: float
    turn_tolerance_deg: float
    forward_timeout_sec: float
    turn_timeout_sec: float
    forward_command: float
    turn_command: float
    settle_max_wait_sec: float
    settle_linear_speed_threshold_mps: float
    settle_angular_speed_threshold_radps: float
    settle_stable_duration_sec: float


@dataclass(frozen=True)
class RobotGeometryProfile:
    footprint_radius_m: float
    obstacle_stop_distance_m: float
    map_inflation_radius_m: float
    path_corridor_radius_m: float
    turn_clearance_radius_m: float


@dataclass(frozen=True)
class RobotSafetyProfile:
    fall_detection_enabled: bool
    max_roll_pitch_deg: float
    min_base_height_m: float
    confirmation_sec: float


@dataclass(frozen=True)
class AgxControllerProfile:
    left_wheel_joint_names: tuple[str, ...]
    right_wheel_joint_names: tuple[str, ...]
    steering_joint_names: tuple[str, ...]
    wheel_drive_velocity: float
    turn_drive_velocity: float
    steering_angle: float
    turn_mode: str
    turn_steering_angle: float
    rear_steering_ratio: float
    track_min_speed_scale: float
    track_max_speed_scale: float
    track_heading_gain: float
    track_slow_heading_rad: float
    track_smoothing_alpha: float
    track_steering_sign: float
    wheel_drive_damping: float
    steering_stiffness: float
    steering_damping: float
    hold_joint_stiffness: float
    hold_joint_damping: float


@dataclass(frozen=True)
class G1ControllerProfile:
    policy_path: Path
    deploy_config_path: Path
    inference_device: str
    standup_duration_sec: float
    sdk_joint_names: tuple[str, ...]
    odom_filter_alpha: float


@dataclass(frozen=True)
class RobotProfile:
    name: str
    robot_type: str
    asset: RobotAssetProfile
    spawn_z_m: float
    simulation: RobotSimulationProfile
    sensors: RobotSensorProfile
    primitives: PrimitiveProfile
    geometry: RobotGeometryProfile
    safety: RobotSafetyProfile
    agx_controller: AgxControllerProfile | None
    g1_controller: G1ControllerProfile | None

    @property
    def is_g1(self) -> bool:
        return self.robot_type == "unitree_g1_policy"


def load_robot_profile(*, profile_name: str, project_root: Path) -> RobotProfile:
    normalized_name = str(profile_name).strip().lower()
    if normalized_name not in SUPPORTED_PROFILES:
        raise ValueError(
            f"Unsupported robot profile {profile_name!r}; expected one of {sorted(SUPPORTED_PROFILES)}"
        )
    profile_path = project_root / "configs" / "robots" / f"{normalized_name}.yaml"
    if not profile_path.is_file():
        raise FileNotFoundError(f"Robot profile does not exist: {profile_path}")
    with profile_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Robot profile must be a YAML mapping: {profile_path}")
    _require_exact_keys(
        payload,
        required={
            "schema_version",
            "identity",
            "asset",
            "spawn",
            "simulation",
            "sensors",
            "controller",
            "primitives",
            "geometry",
            "safety",
        },
        context=normalized_name,
    )
    if str(payload["schema_version"]) != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"Robot profile {normalized_name} schema_version={payload['schema_version']!r}; "
            f"expected {PROFILE_SCHEMA_VERSION!r}"
        )

    identity = _mapping(payload, "identity", normalized_name)
    _require_exact_keys(identity, required={"name", "type"}, context=f"{normalized_name}.identity")
    if str(identity["name"]).strip().lower() != normalized_name:
        raise ValueError(f"Robot profile identity.name must equal {normalized_name!r}")
    robot_type = str(identity["type"]).strip()
    if robot_type not in SUPPORTED_CONTROLLER_TYPES:
        raise ValueError(f"Unsupported controller type {robot_type!r} in {profile_path}")

    asset = _mapping(payload, "asset", normalized_name)
    _require_exact_keys(
        asset,
        required={"usd_path", "prim_path", "articulation_path"},
        context=f"{normalized_name}.asset",
    )
    spawn = _mapping(payload, "spawn", normalized_name)
    _require_exact_keys(spawn, required={"z_m"}, context=f"{normalized_name}.spawn")
    simulation = _mapping(payload, "simulation", normalized_name)
    _require_exact_keys(
        simulation,
        required={"physics_dt", "rendering_dt", "policy_decimation"},
        context=f"{normalized_name}.simulation",
    )
    sensors = _mapping(payload, "sensors", normalized_name)
    _require_exact_keys(
        sensors,
        required={
            "front_camera_prim_path",
            "observer_camera_enabled",
            "observer_camera_prim_path",
            "camera_frequency_hz",
            "lidar_prim_path",
        },
        context=f"{normalized_name}.sensors",
    )
    primitives = _mapping(payload, "primitives", normalized_name)
    _require_exact_keys(
        primitives,
        required={
            "forward_distance_m",
            "turn_angle_deg",
            "distance_tolerance_m",
            "turn_tolerance_deg",
            "forward_timeout_sec",
            "turn_timeout_sec",
            "forward_command",
            "turn_command",
            "settle_max_wait_sec",
            "settle_linear_speed_threshold_mps",
            "settle_angular_speed_threshold_radps",
            "settle_stable_duration_sec",
        },
        context=f"{normalized_name}.primitives",
    )
    geometry = _mapping(payload, "geometry", normalized_name)
    _require_exact_keys(
        geometry,
        required={
            "footprint_radius_m",
            "obstacle_stop_distance_m",
            "map_inflation_radius_m",
            "path_corridor_radius_m",
            "turn_clearance_radius_m",
        },
        context=f"{normalized_name}.geometry",
    )
    safety = _mapping(payload, "safety", normalized_name)
    _require_exact_keys(
        safety,
        required={
            "fall_detection_enabled",
            "max_roll_pitch_deg",
            "min_base_height_m",
            "confirmation_sec",
        },
        context=f"{normalized_name}.safety",
    )
    controller = _mapping(payload, "controller", normalized_name)

    agx_controller = None
    g1_controller = None
    if robot_type == "agx_4ws":
        required_controller_keys = {
            "left_wheel_joint_names",
            "right_wheel_joint_names",
            "steering_joint_names",
            "wheel_drive_velocity",
            "turn_drive_velocity",
            "steering_angle",
            "turn_mode",
            "turn_steering_angle",
            "rear_steering_ratio",
            "track_min_speed_scale",
            "track_max_speed_scale",
            "track_heading_gain",
            "track_slow_heading_rad",
            "track_smoothing_alpha",
            "track_steering_sign",
            "wheel_drive_damping",
            "steering_stiffness",
            "steering_damping",
            "hold_joint_stiffness",
            "hold_joint_damping",
        }
        _require_exact_keys(
            controller,
            required=required_controller_keys,
            context=f"{normalized_name}.controller",
        )
        agx_controller = AgxControllerProfile(
            left_wheel_joint_names=_string_tuple(controller["left_wheel_joint_names"], "left_wheel_joint_names"),
            right_wheel_joint_names=_string_tuple(controller["right_wheel_joint_names"], "right_wheel_joint_names"),
            steering_joint_names=_string_tuple(controller["steering_joint_names"], "steering_joint_names"),
            wheel_drive_velocity=_positive_float(controller["wheel_drive_velocity"], "wheel_drive_velocity"),
            turn_drive_velocity=_positive_float(controller["turn_drive_velocity"], "turn_drive_velocity"),
            steering_angle=_positive_float(controller["steering_angle"], "steering_angle"),
            turn_mode=str(controller["turn_mode"]),
            turn_steering_angle=_positive_float(controller["turn_steering_angle"], "turn_steering_angle"),
            rear_steering_ratio=float(controller["rear_steering_ratio"]),
            track_min_speed_scale=_positive_float(controller["track_min_speed_scale"], "track_min_speed_scale"),
            track_max_speed_scale=_positive_float(controller["track_max_speed_scale"], "track_max_speed_scale"),
            track_heading_gain=_positive_float(controller["track_heading_gain"], "track_heading_gain"),
            track_slow_heading_rad=_positive_float(controller["track_slow_heading_rad"], "track_slow_heading_rad"),
            track_smoothing_alpha=float(controller["track_smoothing_alpha"]),
            track_steering_sign=float(controller["track_steering_sign"]),
            wheel_drive_damping=_positive_float(controller["wheel_drive_damping"], "wheel_drive_damping"),
            steering_stiffness=_positive_float(controller["steering_stiffness"], "steering_stiffness"),
            steering_damping=_positive_float(controller["steering_damping"], "steering_damping"),
            hold_joint_stiffness=_positive_float(controller["hold_joint_stiffness"], "hold_joint_stiffness"),
            hold_joint_damping=_positive_float(controller["hold_joint_damping"], "hold_joint_damping"),
        )
    else:
        _require_exact_keys(
            controller,
            required={
                "policy_path",
                "deploy_config_path",
                "inference_device",
                "standup_duration_sec",
                "sdk_joint_names",
                "odom_filter_alpha",
            },
            context=f"{normalized_name}.controller",
        )
        inference_device = str(controller["inference_device"]).strip().lower()
        if inference_device != "cpu":
            raise ValueError("unitree_g1_29dof inference_device must be 'cpu'")
        sdk_joint_names = _string_tuple(controller["sdk_joint_names"], "sdk_joint_names")
        if len(sdk_joint_names) != 29 or len(set(sdk_joint_names)) != 29:
            raise ValueError("unitree_g1_29dof sdk_joint_names must contain 29 unique names")
        g1_controller = G1ControllerProfile(
            policy_path=_resolve_project_path(controller["policy_path"], project_root),
            deploy_config_path=_resolve_project_path(controller["deploy_config_path"], project_root),
            inference_device=inference_device,
            standup_duration_sec=_positive_float(controller["standup_duration_sec"], "standup_duration_sec"),
            sdk_joint_names=sdk_joint_names,
            odom_filter_alpha=_unit_float(controller["odom_filter_alpha"], "odom_filter_alpha"),
        )

    profile = RobotProfile(
        name=normalized_name,
        robot_type=robot_type,
        asset=RobotAssetProfile(
            usd_path=_resolve_project_path(asset["usd_path"], project_root),
            prim_path=_absolute_prim_path(asset["prim_path"], "asset.prim_path"),
            articulation_path=_relative_prim_path(asset["articulation_path"], "asset.articulation_path"),
        ),
        spawn_z_m=float(spawn["z_m"]),
        simulation=RobotSimulationProfile(
            physics_dt=_positive_float(simulation["physics_dt"], "physics_dt"),
            rendering_dt=_positive_float(simulation["rendering_dt"], "rendering_dt"),
            policy_decimation=_positive_int(simulation["policy_decimation"], "policy_decimation"),
        ),
        sensors=RobotSensorProfile(
            front_camera_prim_path=_relative_prim_path(
                sensors["front_camera_prim_path"], "front_camera_prim_path"
            ),
            observer_camera_enabled=bool(sensors["observer_camera_enabled"]),
            observer_camera_prim_path=_relative_prim_path(
                sensors["observer_camera_prim_path"], "observer_camera_prim_path"
            ),
            camera_frequency_hz=_positive_int(
                sensors["camera_frequency_hz"], "camera_frequency_hz"
            ),
            lidar_prim_path=_relative_prim_path(sensors["lidar_prim_path"], "lidar_prim_path"),
        ),
        primitives=PrimitiveProfile(
            forward_distance_m=_positive_float(primitives["forward_distance_m"], "forward_distance_m"),
            turn_angle_deg=_positive_float(primitives["turn_angle_deg"], "turn_angle_deg"),
            distance_tolerance_m=_positive_float(primitives["distance_tolerance_m"], "distance_tolerance_m"),
            turn_tolerance_deg=_positive_float(primitives["turn_tolerance_deg"], "turn_tolerance_deg"),
            forward_timeout_sec=_positive_float(primitives["forward_timeout_sec"], "forward_timeout_sec"),
            turn_timeout_sec=_positive_float(primitives["turn_timeout_sec"], "turn_timeout_sec"),
            forward_command=_positive_float(primitives["forward_command"], "forward_command"),
            turn_command=_positive_float(primitives["turn_command"], "turn_command"),
            settle_max_wait_sec=max(0.0, float(primitives["settle_max_wait_sec"])),
            settle_linear_speed_threshold_mps=max(
                0.0, float(primitives["settle_linear_speed_threshold_mps"])
            ),
            settle_angular_speed_threshold_radps=max(
                0.0, float(primitives["settle_angular_speed_threshold_radps"])
            ),
            settle_stable_duration_sec=max(0.0, float(primitives["settle_stable_duration_sec"])),
        ),
        geometry=RobotGeometryProfile(
            footprint_radius_m=_positive_float(geometry["footprint_radius_m"], "footprint_radius_m"),
            obstacle_stop_distance_m=_positive_float(
                geometry["obstacle_stop_distance_m"], "obstacle_stop_distance_m"
            ),
            map_inflation_radius_m=_positive_float(
                geometry["map_inflation_radius_m"], "map_inflation_radius_m"
            ),
            path_corridor_radius_m=_positive_float(
                geometry["path_corridor_radius_m"], "path_corridor_radius_m"
            ),
            turn_clearance_radius_m=_positive_float(
                geometry["turn_clearance_radius_m"], "turn_clearance_radius_m"
            ),
        ),
        safety=RobotSafetyProfile(
            fall_detection_enabled=bool(safety["fall_detection_enabled"]),
            max_roll_pitch_deg=_positive_float(safety["max_roll_pitch_deg"], "max_roll_pitch_deg"),
            min_base_height_m=max(0.0, float(safety["min_base_height_m"])),
            confirmation_sec=max(0.0, float(safety["confirmation_sec"])),
        ),
        agx_controller=agx_controller,
        g1_controller=g1_controller,
    )
    _validate_profile_files(profile)
    return profile


def _validate_profile_files(profile: RobotProfile) -> None:
    if not profile.asset.usd_path.is_file():
        raise FileNotFoundError(f"Robot USD does not exist: {profile.asset.usd_path}")
    if profile.g1_controller is None:
        return
    if not profile.g1_controller.policy_path.is_file():
        raise FileNotFoundError(f"G1 TorchScript policy does not exist: {profile.g1_controller.policy_path}")
    if not profile.g1_controller.deploy_config_path.is_file():
        raise FileNotFoundError(
            f"G1 deployment config does not exist: {profile.g1_controller.deploy_config_path}"
        )


def _mapping(payload: dict[str, Any], key: str, context: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{context}.{key} must be a YAML mapping")
    return value


def _require_exact_keys(payload: dict[str, Any], *, required: set[str], context: str) -> None:
    actual = set(payload)
    missing = sorted(required - actual)
    extra = sorted(actual - required)
    if missing or extra:
        raise ValueError(f"Invalid robot profile section {context}: missing={missing} extra={extra}")


def _resolve_project_path(value: Any, project_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else project_root / path


def _absolute_prim_path(value: Any, field_name: str) -> str:
    text = str(value).strip().rstrip("/")
    if not text.startswith("/"):
        raise ValueError(f"{field_name} must be an absolute USD prim path")
    return text


def _relative_prim_path(value: Any, field_name: str) -> str:
    text = str(value).strip().strip("/")
    if not text:
        raise ValueError(f"{field_name} must be a non-empty robot-relative USD prim path")
    return text


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty YAML list")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise ValueError(f"{field_name} cannot contain empty names")
    return result


def _positive_float(value: Any, field_name: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{field_name} must be > 0")
    return result


def _positive_int(value: Any, field_name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return result


def _unit_float(value: Any, field_name: str) -> float:
    result = float(value)
    if not 0.0 < result <= 1.0:
        raise ValueError(f"{field_name} must be in (0, 1]")
    return result
