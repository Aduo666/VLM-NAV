from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Pose2D:
    x: float
    y: float
    yaw_rad: float


@dataclass
class MotionFeedback:
    last_action: str
    wheel_velocity: float
    steering_angle: float


@dataclass
class VLNObservation:
    rgb: np.ndarray | None
    depth: np.ndarray | None
    pose: Pose2D
    camera: dict[str, Any]
    motion_feedback: MotionFeedback
    meta: dict[str, Any]


class ObservationAdapter:
    def __init__(self, *, camera_width: int, camera_height: int, horizontal_fov_deg: float) -> None:
        self._camera_width = int(camera_width)
        self._camera_height = int(camera_height)
        self._horizontal_fov_deg = float(horizontal_fov_deg)

    def set_horizontal_fov_deg(self, value: float) -> None:
        self._horizontal_fov_deg = float(value)

    def adapt(
        self,
        *,
        rgb: np.ndarray | None,
        depth: np.ndarray | None,
        robot_position_xyz: np.ndarray,
        robot_orientation_xyzw: np.ndarray,
        frame_index: int,
        timestamp_sec: float,
        last_action: str,
        wheel_velocity: float,
        steering_angle: float,
        odom_pose: Pose2D | None = None,
        odom_meta: dict[str, Any] | None = None,
    ) -> VLNObservation:
        position = np.asarray(robot_position_xyz, dtype=np.float32).reshape(-1)
        # Isaac Sim core APIs return quaternions in scalar-first wxyz order.
        orientation = np.asarray(robot_orientation_xyzw, dtype=np.float32).reshape(-1)
        pose = odom_pose
        if pose is None:
            pose = Pose2D(
                x=float(position[0]),
                y=float(position[1]),
                yaw_rad=float(_yaw_from_quaternion(orientation)),
            )
        return VLNObservation(
            rgb=None if rgb is None else np.asarray(rgb),
            depth=None if depth is None else np.asarray(depth, dtype=np.float32),
            pose=pose,
            camera={
                "width": self._camera_width,
                "height": self._camera_height,
                "hfov_deg": self._horizontal_fov_deg,
            },
            motion_feedback=MotionFeedback(
                last_action=str(last_action),
                wheel_velocity=float(wheel_velocity),
                steering_angle=float(steering_angle),
            ),
            meta={
                "frame_index": int(frame_index),
                "timestamp_sec": float(timestamp_sec),
                "world_position_xyz": [float(value) for value in position[:3]],
                "odom": dict(odom_meta or {}),
            },
        )


def _yaw_from_quaternion(quaternion_wxyz: np.ndarray) -> float:
    w, x, y, z = [float(value) for value in quaternion_wxyz[:4]]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))
