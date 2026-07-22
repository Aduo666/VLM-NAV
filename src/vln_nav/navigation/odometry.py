from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .observation_adapter import Pose2D


@dataclass
class OdomState:
    pose: Pose2D
    source: str
    timestamp_sec: float
    linear_velocity_mps: float
    angular_velocity_radps: float
    delta_distance_m: float
    delta_yaw_rad: float
    total_distance_m: float


class SimOdometrySensor:
    """Truth-backed odometry adapter for Isaac Sim.

    This exposes an odom-like interface while we are still in simulation. It is
    intentionally isolated so a future ROS2 `/odom` or `/tf` provider can replace
    it without touching the VLN policy, tracker, or visualizer.
    """

    def __init__(
        self,
        *,
        logger,
        source: str = "isaac_world_pose",
        pose_filter_alpha: float = 1.0,
    ) -> None:
        self._logger = logger.getChild("odom")
        self._source = str(source)
        self._pose_filter_alpha = float(np.clip(pose_filter_alpha, 0.0, 1.0))
        self._origin_position_xy: np.ndarray | None = None
        self._origin_yaw_rad: float | None = None
        self._last_pose: Pose2D | None = None
        self._last_timestamp_sec: float | None = None
        self._total_distance_m = 0.0
        self._logged_ready = False

    def update(
        self,
        *,
        world_position_xyz: np.ndarray,
        world_orientation_wxyz: np.ndarray,
        timestamp_sec: float,
    ) -> OdomState:
        position = np.asarray(world_position_xyz, dtype=np.float32).reshape(-1)
        orientation = np.asarray(world_orientation_wxyz, dtype=np.float32).reshape(-1)
        world_xy = np.asarray([float(position[0]), float(position[1])], dtype=np.float32)
        world_yaw_rad = _yaw_from_quaternion_wxyz(orientation)

        if self._origin_position_xy is None:
            self._origin_position_xy = world_xy.copy()
            self._origin_yaw_rad = float(world_yaw_rad)

        raw_pose = Pose2D(
            x=float(world_xy[0] - self._origin_position_xy[0]),
            y=float(world_xy[1] - self._origin_position_xy[1]),
            yaw_rad=float(_wrap_angle_rad(world_yaw_rad - float(self._origin_yaw_rad or 0.0))),
        )
        if self._last_pose is None or self._pose_filter_alpha >= 1.0:
            pose = raw_pose
        else:
            alpha = self._pose_filter_alpha
            pose = Pose2D(
                x=float((1.0 - alpha) * self._last_pose.x + alpha * raw_pose.x),
                y=float((1.0 - alpha) * self._last_pose.y + alpha * raw_pose.y),
                yaw_rad=float(
                    _wrap_angle_rad(
                        self._last_pose.yaw_rad
                        + alpha * _wrap_angle_rad(raw_pose.yaw_rad - self._last_pose.yaw_rad)
                    )
                ),
            )

        delta_distance_m = 0.0
        delta_yaw_rad = 0.0
        linear_velocity_mps = 0.0
        angular_velocity_radps = 0.0
        if self._last_pose is not None and self._last_timestamp_sec is not None:
            delta_distance_m = float(np.hypot(pose.x - self._last_pose.x, pose.y - self._last_pose.y))
            delta_yaw_rad = float(_wrap_angle_rad(pose.yaw_rad - self._last_pose.yaw_rad))
            dt = max(1e-6, float(timestamp_sec) - float(self._last_timestamp_sec))
            linear_velocity_mps = delta_distance_m / dt
            angular_velocity_radps = delta_yaw_rad / dt
            self._total_distance_m += delta_distance_m

        self._last_pose = pose
        self._last_timestamp_sec = float(timestamp_sec)
        if not self._logged_ready:
            self._logger.info(
                "[Odom] source=%s origin_world_xy=[%.3f,%.3f] origin_yaw_deg=%.1f filter_alpha=%.2f",
                self._source,
                float(self._origin_position_xy[0]),
                float(self._origin_position_xy[1]),
                float(np.degrees(float(self._origin_yaw_rad or 0.0))),
                self._pose_filter_alpha,
            )
            self._logged_ready = True

        return OdomState(
            pose=pose,
            source=self._source,
            timestamp_sec=float(timestamp_sec),
            linear_velocity_mps=float(linear_velocity_mps),
            angular_velocity_radps=float(angular_velocity_radps),
            delta_distance_m=float(delta_distance_m),
            delta_yaw_rad=float(delta_yaw_rad),
            total_distance_m=float(self._total_distance_m),
        )


def _yaw_from_quaternion_wxyz(quaternion_wxyz: np.ndarray) -> float:
    w, x, y, z = [float(value) for value in quaternion_wxyz[:4]]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _wrap_angle_rad(value: float) -> float:
    return float((float(value) + np.pi) % (2.0 * np.pi) - np.pi)
