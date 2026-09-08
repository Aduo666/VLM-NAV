from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from vln_nav.control.trajectory import TrajectoryConfig


PROFILE_SCHEMA_VERSION = "2.0"
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
class CameraProfile:
    front_prim_path: str
    observer_enabled: bool
    observer_prim_path: str
    width: int
    height: int
    frequency_hz: int
    horizontal_fov_deg: float
    follow_front_in_gui: bool
    follow_observer_in_gui: bool
    depth_roi_vertical_range: tuple[float, float]
    valid_depth_range_m: tuple[float, float]
    obstacle_roi_vertical_range: tuple[float, float]
    obstacle_center_width_ratio: float
    obstacle_side_width_ratio: float
    depth_eval_period_sec: float


@dataclass(frozen=True)
class RobotSensorProfile:
    camera: CameraProfile

    @property
    def front_camera_prim_path(self) -> str:
        return self.camera.front_prim_path

    @property
    def observer_camera_enabled(self) -> bool:
        return self.camera.observer_enabled

    @property
    def observer_camera_prim_path(self) -> str:
        return self.camera.observer_prim_path

    @property
    def camera_frequency_hz(self) -> int:
        return self.camera.frequency_hz

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
    obstacle_stop_distance_m: float
    map_inflation_radius_m: float
    path_corridor_radius_m: float


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
    wheel_drive_damping: float
    steering_stiffness: float
    steering_damping: float
    hold_joint_stiffness: float
    hold_joint_damping: float
    continuous: TrajectoryConfig


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

    def capability_summary(self) -> str:
        platform = "biped humanoid" if self.is_g1 else "four-wheel-steering wheeled robot"
        return (
            f"platform={platform}; "
            f"FORWARD={self.primitives.forward_distance_m:.2f}m; "
            f"TURN_LEFT/RIGHT={self.primitives.turn_angle_deg:.1f}deg"
        )


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

    asset_data = _mapping(payload, "asset", normalized_name)
    _require_exact_keys(
        asset_data,
        required={"usd_path", "prim_path", "articulation_path"},
        context=f"{normalized_name}.asset",
    )
    spawn_data = _mapping(payload, "spawn", normalized_name)
    _require_exact_keys(spawn_data, required={"z_m"}, context=f"{normalized_name}.spawn")
    simulation_data = _mapping(payload, "simulation", normalized_name)
    _require_exact_keys(
        simulation_data,
        required={"physics_dt", "rendering_dt", "policy_decimation"},
        context=f"{normalized_name}.simulation",
    )

    sensors_data = _mapping(payload, "sensors", normalized_name)
    _require_exact_keys(
        sensors_data,
        required={"camera"},
        context=f"{normalized_name}.sensors",
    )
    camera_data = _mapping(sensors_data, "camera", f"{normalized_name}.sensors")
    _require_exact_keys(
        camera_data,
        required={
            "front_prim_path",
            "observer_enabled",
            "observer_prim_path",
            "width",
            "height",
            "frequency_hz",
            "horizontal_fov_deg",
            "follow_front_in_gui",
            "follow_observer_in_gui",
            "depth_roi_vertical_range",
            "valid_depth_range_m",
            "obstacle_roi_vertical_range",
            "obstacle_center_width_ratio",
            "obstacle_side_width_ratio",
            "depth_eval_period_sec",
        },
        context=f"{normalized_name}.sensors.camera",
    )
    primitives_data = _mapping(payload, "primitives", normalized_name)
    primitive_keys = {
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
    }
    _require_exact_keys(
        primitives_data,
        required=primitive_keys,
        context=f"{normalized_name}.primitives",
    )

    geometry_data = _mapping(payload, "geometry", normalized_name)
    _require_exact_keys(
        geometry_data,
        required={
            "obstacle_stop_distance_m",
            "map_inflation_radius_m",
            "path_corridor_radius_m",
        },
        context=f"{normalized_name}.geometry",
    )

    safety_data = _mapping(payload, "safety", normalized_name)
    _require_exact_keys(
        safety_data,
        required={
            "fall_detection_enabled",
            "max_roll_pitch_deg",
            "min_base_height_m",
            "confirmation_sec",
        },
        context=f"{normalized_name}.safety",
    )

    controller_data = _mapping(payload, "controller", normalized_name)
    agx_controller: AgxControllerProfile | None = None
    g1_controller: G1ControllerProfile | None = None
    if robot_type == "agx_4ws":
        required = {
            "left_wheel_joint_names",
            "right_wheel_joint_names",
            "steering_joint_names",
            "wheel_drive_velocity",
            "turn_drive_velocity",
            "steering_angle",
            "turn_mode",
            "turn_steering_angle",
            "rear_steering_ratio",
            "wheel_drive_damping",
            "steering_stiffness",
            "steering_damping",
            "hold_joint_stiffness",
            "hold_joint_damping",
            "continuous",
        }
        _require_exact_keys(controller_data, required=required, context=f"{normalized_name}.controller")
        agx_controller = AgxControllerProfile(
            left_wheel_joint_names=_string_tuple(controller_data["left_wheel_joint_names"]),
            right_wheel_joint_names=_string_tuple(controller_data["right_wheel_joint_names"]),
            steering_joint_names=_string_tuple(controller_data["steering_joint_names"]),
            wheel_drive_velocity=_positive_float(controller_data["wheel_drive_velocity"], "wheel_drive_velocity"),
            turn_drive_velocity=_positive_float(controller_data["turn_drive_velocity"], "turn_drive_velocity"),
            steering_angle=float(controller_data["steering_angle"]),
            turn_mode=str(controller_data["turn_mode"]),
            turn_steering_angle=float(controller_data["turn_steering_angle"]),
            rear_steering_ratio=float(controller_data["rear_steering_ratio"]),
            wheel_drive_damping=_nonnegative_float(controller_data["wheel_drive_damping"], "wheel_drive_damping"),
            steering_stiffness=_nonnegative_float(controller_data["steering_stiffness"], "steering_stiffness"),
            steering_damping=_nonnegative_float(controller_data["steering_damping"], "steering_damping"),
            hold_joint_stiffness=_nonnegative_float(controller_data["hold_joint_stiffness"], "hold_joint_stiffness"),
            hold_joint_damping=_nonnegative_float(controller_data["hold_joint_damping"], "hold_joint_damping"),
            continuous=TrajectoryConfig.from_mapping(controller_data["continuous"]),
        )
        if {w.drive_joint for w in agx_controller.continuous.wheels} != set(
            agx_controller.left_wheel_joint_names + agx_controller.right_wheel_joint_names
        ) or {w.steering_joint for w in agx_controller.continuous.wheels} != set(
            agx_controller.steering_joint_names
        ):
            raise ValueError("AGX continuous wheel pairs must match controller joint names")
    else:
        required = {
            "policy_path",
            "deploy_config_path",
            "inference_device",
            "standup_duration_sec",
            "odom_filter_alpha",
            "sdk_joint_names",
        }
        _require_exact_keys(controller_data, required=required, context=f"{normalized_name}.controller")
        g1_controller = G1ControllerProfile(
            policy_path=_project_path(project_root, controller_data["policy_path"]),
            deploy_config_path=_project_path(project_root, controller_data["deploy_config_path"]),
            inference_device=str(controller_data["inference_device"]),
            standup_duration_sec=_nonnegative_float(controller_data["standup_duration_sec"], "standup_duration_sec"),
            odom_filter_alpha=float(controller_data["odom_filter_alpha"]),
            sdk_joint_names=_string_tuple(controller_data["sdk_joint_names"]),
        )

    profile = RobotProfile(
        name=normalized_name,
        robot_type=robot_type,
        asset=RobotAssetProfile(
            usd_path=_project_path(project_root, asset_data["usd_path"]),
            prim_path=_prim_path(asset_data["prim_path"], "asset.prim_path"),
            articulation_path=str(asset_data["articulation_path"]).strip(),
        ),
        spawn_z_m=float(spawn_data["z_m"]),
        simulation=RobotSimulationProfile(
            physics_dt=_positive_float(simulation_data["physics_dt"], "physics_dt"),
            rendering_dt=_positive_float(simulation_data["rendering_dt"], "rendering_dt"),
            policy_decimation=_positive_int(simulation_data["policy_decimation"], "policy_decimation"),
        ),
        sensors=RobotSensorProfile(
            camera=CameraProfile(
                front_prim_path=str(camera_data["front_prim_path"]).strip(),
                observer_enabled=bool(camera_data["observer_enabled"]),
                observer_prim_path=str(camera_data["observer_prim_path"]).strip(),
                width=_positive_int(camera_data["width"], "camera.width"),
                height=_positive_int(camera_data["height"], "camera.height"),
                frequency_hz=_positive_int(camera_data["frequency_hz"], "camera.frequency_hz"),
                horizontal_fov_deg=_positive_float(camera_data["horizontal_fov_deg"], "camera.horizontal_fov_deg"),
                follow_front_in_gui=bool(camera_data["follow_front_in_gui"]),
                follow_observer_in_gui=bool(camera_data["follow_observer_in_gui"]),
                depth_roi_vertical_range=_range_pair(
                    camera_data["depth_roi_vertical_range"], "camera.depth_roi_vertical_range"
                ),
                valid_depth_range_m=_range_pair(
                    camera_data["valid_depth_range_m"], "camera.valid_depth_range_m"
                ),
                obstacle_roi_vertical_range=_range_pair(
                    camera_data["obstacle_roi_vertical_range"], "camera.obstacle_roi_vertical_range"
                ),
                obstacle_center_width_ratio=float(camera_data["obstacle_center_width_ratio"]),
                obstacle_side_width_ratio=float(camera_data["obstacle_side_width_ratio"]),
                depth_eval_period_sec=_positive_float(camera_data["depth_eval_period_sec"], "camera.depth_eval_period_sec"),
            ),
        ),
        primitives=PrimitiveProfile(
            forward_distance_m=_positive_float(primitives_data["forward_distance_m"], "forward_distance_m"),
            turn_angle_deg=_positive_float(primitives_data["turn_angle_deg"], "turn_angle_deg"),
            distance_tolerance_m=_positive_float(primitives_data["distance_tolerance_m"], "distance_tolerance_m"),
            turn_tolerance_deg=_positive_float(primitives_data["turn_tolerance_deg"], "turn_tolerance_deg"),
            forward_timeout_sec=_positive_float(primitives_data["forward_timeout_sec"], "forward_timeout_sec"),
            turn_timeout_sec=_positive_float(primitives_data["turn_timeout_sec"], "turn_timeout_sec"),
            forward_command=_positive_float(primitives_data["forward_command"], "forward_command"),
            turn_command=_positive_float(primitives_data["turn_command"], "turn_command"),
            settle_max_wait_sec=_nonnegative_float(primitives_data["settle_max_wait_sec"], "settle_max_wait_sec"),
            settle_linear_speed_threshold_mps=_nonnegative_float(
                primitives_data["settle_linear_speed_threshold_mps"],
                "settle_linear_speed_threshold_mps",
            ),
            settle_angular_speed_threshold_radps=_nonnegative_float(
                primitives_data["settle_angular_speed_threshold_radps"],
                "settle_angular_speed_threshold_radps",
            ),
            settle_stable_duration_sec=_nonnegative_float(
                primitives_data["settle_stable_duration_sec"], "settle_stable_duration_sec"
            ),
        ),
        geometry=RobotGeometryProfile(
            obstacle_stop_distance_m=_positive_float(
                geometry_data["obstacle_stop_distance_m"], "geometry.obstacle_stop_distance_m"
            ),
            map_inflation_radius_m=_nonnegative_float(
                geometry_data["map_inflation_radius_m"],
                "geometry.map_inflation_radius_m",
            ),
            path_corridor_radius_m=_nonnegative_float(
                geometry_data["path_corridor_radius_m"],
                "geometry.path_corridor_radius_m",
            ),
        ),
        safety=RobotSafetyProfile(
            fall_detection_enabled=bool(safety_data["fall_detection_enabled"]),
            max_roll_pitch_deg=_positive_float(safety_data["max_roll_pitch_deg"], "max_roll_pitch_deg"),
            min_base_height_m=_nonnegative_float(safety_data["min_base_height_m"], "min_base_height_m"),
            confirmation_sec=_nonnegative_float(safety_data["confirmation_sec"], "confirmation_sec"),
        ),
        agx_controller=agx_controller,
        g1_controller=g1_controller,
    )
    _validate_profile_files(profile)
    return profile


def _validate_profile_files(profile: RobotProfile) -> None:
    if not profile.asset.usd_path.is_file():
        raise FileNotFoundError(f"Robot USD does not exist: {profile.asset.usd_path}")
    if profile.is_g1:
        controller = profile.g1_controller
        if controller is None:
            raise ValueError("G1 profile must define a G1 controller")
        if len(controller.sdk_joint_names) != 29:
            raise ValueError(
                f"G1 deploy contract requires 29 sdk_joint_names, got {len(controller.sdk_joint_names)}"
            )
        if profile.spawn_z_m < profile.safety.min_base_height_m:
            raise ValueError(
                "G1 spawn.z_m must not be below the configured pelvis/root height safety threshold"
            )
        if not controller.policy_path.is_file():
            raise FileNotFoundError(f"G1 policy does not exist: {controller.policy_path}")
        if not controller.deploy_config_path.is_file():
            raise FileNotFoundError(f"G1 deploy config does not exist: {controller.deploy_config_path}")


def _mapping(payload: dict[str, Any], key: str, context: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{context}.{key} must be a mapping")
    return value


def _require_exact_keys(mapping: dict[str, Any], *, required: set[str], context: str) -> None:
    missing = sorted(required - set(mapping))
    extra = sorted(set(mapping) - required)
    if missing or extra:
        raise ValueError(f"{context} keys invalid: missing={missing} extra={extra}")


def _project_path(project_root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else project_root / path


def _prim_path(value: Any, field_name: str) -> str:
    text = str(value).strip()
    if not text.startswith("/"):
        raise ValueError(f"{field_name} must be an absolute prim path")
    return text


def _positive_float(value: Any, field_name: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{field_name} must be > 0")
    return result


def _nonnegative_float(value: Any, field_name: str) -> float:
    result = float(value)
    if result < 0.0:
        raise ValueError(f"{field_name} must be >= 0")
    return result


def _positive_int(value: Any, field_name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return result


def _range_pair(value: Any, field_name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} must contain exactly two numbers")
    lower, upper = float(value[0]), float(value[1])
    if lower >= upper:
        raise ValueError(f"{field_name} lower bound must be < upper bound")
    return lower, upper


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("Expected a list of strings")
    return tuple(str(item) for item in value)
