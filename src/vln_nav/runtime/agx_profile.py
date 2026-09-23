from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from vln_nav.control.trajectory import TrajectoryConfig


@dataclass(frozen=True)
class AgxAsset:
    usd_path: Path
    prim_path: str
    articulation_path: str


@dataclass(frozen=True)
class AgxSimulation:
    physics_dt: float
    rendering_dt: float


@dataclass(frozen=True)
class AgxCamera:
    front_prim_path: str
    s2_down_prim_path: str
    mapping_down_prim_path: str


@dataclass(frozen=True)
class AgxSensors:
    camera: AgxCamera
    observer_camera_prim_path: str


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
class AgxProfile:
    name: str
    robot_type: str
    asset: AgxAsset
    spawn_z_m: float
    simulation: AgxSimulation
    sensors: AgxSensors
    agx_controller: AgxControllerProfile


def load_agx_profile(*, project_root: Path) -> AgxProfile:
    profile_path = project_root / "configs/robots/agx.yaml"
    with profile_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict) or set(data) != {
        "asset", "spawn", "simulation", "sensors", "controller"
    }:
        raise ValueError("Invalid AGX profile sections")

    asset = data["asset"]
    if set(asset) != {"usd_path", "prim_path", "articulation_path"}:
        raise ValueError("Invalid AGX asset settings")
    usd_path = Path(str(asset["usd_path"]))
    if not usd_path.is_absolute():
        usd_path = project_root / usd_path
    if not usd_path.is_file():
        raise FileNotFoundError(f"AGX USD does not exist: {usd_path}")
    prim_path = str(asset["prim_path"])
    if not prim_path.startswith("/"):
        raise ValueError("AGX asset prim_path must be absolute")

    spawn = data["spawn"]
    simulation = data["simulation"]
    sensors = data["sensors"]
    controller = data["controller"]
    if set(spawn) != {"z_m"} or float(spawn["z_m"]) < 0:
        raise ValueError("Invalid AGX spawn height")
    if set(simulation) != {"physics_dt", "rendering_dt"} or any(
        float(value) <= 0 for value in simulation.values()
    ):
        raise ValueError("Invalid AGX simulation timing")
    if set(sensors) != {
        "front_prim_path",
        "s2_down_prim_path",
        "mapping_down_prim_path",
        "observer_prim_path",
    }:
        raise ValueError("Invalid AGX sensor paths")
    controller_keys = {
        "left_wheel_joint_names", "right_wheel_joint_names", "steering_joint_names",
        "wheel_drive_velocity", "turn_drive_velocity", "steering_angle", "turn_mode",
        "turn_steering_angle", "rear_steering_ratio", "wheel_drive_damping",
        "steering_stiffness", "steering_damping", "hold_joint_stiffness",
        "hold_joint_damping", "continuous",
    }
    if set(controller) != controller_keys:
        raise ValueError("Invalid AGX controller settings")
    continuous = TrajectoryConfig.from_mapping(controller["continuous"])
    agx_controller = AgxControllerProfile(
        left_wheel_joint_names=tuple(controller["left_wheel_joint_names"]),
        right_wheel_joint_names=tuple(controller["right_wheel_joint_names"]),
        steering_joint_names=tuple(controller["steering_joint_names"]),
        wheel_drive_velocity=float(controller["wheel_drive_velocity"]),
        turn_drive_velocity=float(controller["turn_drive_velocity"]),
        steering_angle=float(controller["steering_angle"]),
        turn_mode=str(controller["turn_mode"]),
        turn_steering_angle=float(controller["turn_steering_angle"]),
        rear_steering_ratio=float(controller["rear_steering_ratio"]),
        wheel_drive_damping=float(controller["wheel_drive_damping"]),
        steering_stiffness=float(controller["steering_stiffness"]),
        steering_damping=float(controller["steering_damping"]),
        hold_joint_stiffness=float(controller["hold_joint_stiffness"]),
        hold_joint_damping=float(controller["hold_joint_damping"]),
        continuous=continuous,
    )
    if {wheel.drive_joint for wheel in continuous.wheels} != set(
        agx_controller.left_wheel_joint_names + agx_controller.right_wheel_joint_names
    ) or {wheel.steering_joint for wheel in continuous.wheels} != set(
        agx_controller.steering_joint_names
    ):
        raise ValueError("AGX continuous wheel joints do not match controller joints")
    return AgxProfile(
        name="agx",
        robot_type="agx_4ws",
        asset=AgxAsset(
            usd_path=usd_path,
            prim_path=prim_path,
            articulation_path=str(asset["articulation_path"]),
        ),
        spawn_z_m=float(spawn["z_m"]),
        simulation=AgxSimulation(
            physics_dt=float(simulation["physics_dt"]),
            rendering_dt=float(simulation["rendering_dt"]),
        ),
        sensors=AgxSensors(
            camera=AgxCamera(
                front_prim_path=str(sensors["front_prim_path"]),
                s2_down_prim_path=str(sensors["s2_down_prim_path"]),
                mapping_down_prim_path=str(sensors["mapping_down_prim_path"]),
            ),
            observer_camera_prim_path=str(sensors["observer_prim_path"]),
        ),
        agx_controller=agx_controller,
    )
