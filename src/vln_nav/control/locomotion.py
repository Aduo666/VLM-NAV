from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np

from vln_nav.control.controller import (
    SteeringDriveController,
    resolve_front_steering_joint_groups,
    resolve_hold_joint_names,
    resolve_wheel_joint_groups,
)
from vln_nav.control.g1_policy_controller import G1PolicyController
from vln_nav.runtime.robot_profile import RobotProfile


class LocomotionController(Protocol):
    def apply(self, action_name: str, *, speed_scale: float = 1.0) -> tuple[float, float]: ...

    def readback(self) -> dict[str, Any]: ...


def create_locomotion_controller(
    *,
    profile: RobotProfile,
    robot: Any,
    stage: Any,
    articulation_path: str,
    available_joint_names: list[str],
    logger: logging.Logger,
) -> LocomotionController:
    if profile.is_g1:
        if profile.g1_controller is None:
            raise RuntimeError("G1 robot profile is missing its controller section")
        return G1PolicyController(
            robot=robot,
            available_joint_names=available_joint_names,
            controller_profile=profile.g1_controller,
            primitive_profile=profile.primitives,
            policy_decimation=profile.simulation.policy_decimation,
            logger=logger,
        )

    agx = profile.agx_controller
    if agx is None:
        raise RuntimeError("AGX robot profile is missing its controller section")
    left_names, right_names, steering_names, error_message = resolve_wheel_joint_groups(
        available_joint_names=available_joint_names,
        configured_left=list(agx.left_wheel_joint_names),
        configured_right=list(agx.right_wheel_joint_names),
        configured_steering=list(agx.steering_joint_names),
    )
    if error_message:
        raise RuntimeError(f"{error_message}\nAvailable articulation DOFs: {available_joint_names}")
    front_steering_names, rear_steering_names = resolve_front_steering_joint_groups(steering_names)
    hold_names = resolve_hold_joint_names(
        available_joint_names=available_joint_names,
        wheel_joint_names=[*left_names, *right_names],
        steering_joint_names=steering_names,
    )
    hold_targets = _joint_position_targets(
        robot=robot,
        available_joint_names=available_joint_names,
        target_joint_names=hold_names,
    )
    controller = SteeringDriveController(
        stage=stage,
        articulation_path=articulation_path,
        robot_root_path=profile.asset.prim_path,
        left_wheel_joint_names=left_names,
        right_wheel_joint_names=right_names,
        steering_joint_names=front_steering_names,
        passive_steering_joint_names=rear_steering_names,
        hold_joint_targets=hold_targets,
        wheel_drive_velocity=agx.wheel_drive_velocity,
        turn_drive_velocity=agx.turn_drive_velocity,
        steering_angle=agx.steering_angle,
        turn_mode=agx.turn_mode,
        turn_steering_angle=agx.turn_steering_angle,
        rear_steering_ratio=agx.rear_steering_ratio,
        wheel_drive_damping=agx.wheel_drive_damping,
        steering_stiffness=agx.steering_stiffness,
        steering_damping=agx.steering_damping,
        hold_joint_stiffness=agx.hold_joint_stiffness,
        hold_joint_damping=agx.hold_joint_damping,
        logger=logger,
    )
    logger.info(
        "[DriveConfig] profile=%s type=%s left_wheels=%s right_wheels=%s "
        "front_steering=%s rear_steering=%s turn_mode=%s",
        profile.name,
        profile.robot_type,
        left_names,
        right_names,
        front_steering_names,
        rear_steering_names,
        agx.turn_mode,
    )
    return controller


def _joint_position_targets(
    *, robot: Any, available_joint_names: list[str], target_joint_names: list[str]
) -> dict[str, float]:
    positions = robot.get_joint_positions()
    if positions is None:
        raise RuntimeError("Unable to read articulation joint positions for AGX hold targets")
    values = np.asarray(positions, dtype=np.float32).reshape(-1)
    if values.size != len(available_joint_names):
        raise RuntimeError("Articulation joint position vector does not match its DOF names")
    index_by_name = {name: index for index, name in enumerate(available_joint_names)}
    return {name: float(values[index_by_name[name]]) for name in target_joint_names}
