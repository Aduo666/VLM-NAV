from __future__ import annotations

import math
import numpy as np


def wrap_angle(angle_rad: float | np.ndarray):
    return (np.asarray(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion_wxyz(quaternion_wxyz: np.ndarray) -> float:
    q = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(-1)
    if q.size < 4 or not np.isfinite(q[:4]).all():
        raise ValueError("Quaternion must contain four finite WXYZ values")
    w, x, y, z = map(float, q[:4])
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-9:
        raise ValueError("Quaternion has zero norm")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return float(math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    ))


def quaternion_wxyz_from_yaw(yaw_rad: float) -> np.ndarray:
    half = 0.5 * float(yaw_rad)
    return np.asarray(
        [math.cos(half), 0.0, 0.0, math.sin(half)],
        dtype=np.float32,
    )


def pose_xyyaw(position_xyz: np.ndarray, orientation_wxyz: np.ndarray) -> np.ndarray:
    position = np.asarray(position_xyz, dtype=np.float64).reshape(-1)
    if position.size < 2 or not np.isfinite(position[:2]).all():
        raise ValueError("Position must contain finite XY values")
    return np.asarray(
        [float(position[0]), float(position[1]), yaw_from_quaternion_wxyz(orientation_wxyz)],
        dtype=np.float64,
    )


def relative_goal_xyyaw(pose_world_xyyaw: np.ndarray, goal_world_xyyaw: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_world_xyyaw, dtype=np.float64).reshape(3)
    goal = np.asarray(goal_world_xyyaw, dtype=np.float64).reshape(3)
    dx = float(goal[0] - pose[0])
    dy = float(goal[1] - pose[1])
    c = math.cos(float(pose[2]))
    s = math.sin(float(pose[2]))
    forward = c * dx + s * dy
    left = -s * dx + c * dy
    yaw = float(wrap_angle(float(goal[2] - pose[2])))
    return np.asarray([forward, left, yaw], dtype=np.float32)


def world_xy_to_body(
    world_xy: np.ndarray,
    *,
    robot_world_xy: tuple[float, float],
    robot_yaw_rad: float,
) -> np.ndarray:
    points = np.asarray(world_xy, dtype=np.float64).reshape(-1, 2)
    delta = points - np.asarray(robot_world_xy, dtype=np.float64)
    c = math.cos(float(robot_yaw_rad))
    s = math.sin(float(robot_yaw_rad))
    return np.column_stack(
        (
            c * delta[:, 0] + s * delta[:, 1],
            -s * delta[:, 0] + c * delta[:, 1],
        )
    )


class BodyMotionEstimator:
    """Finite-difference signed body velocity estimator for collection metadata."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._pose: np.ndarray | None = None
        self._timestamp: float | None = None

    def update(self, pose: np.ndarray, timestamp_sec: float) -> tuple[float, float]:
        current = np.asarray(pose, dtype=np.float64).reshape(3)
        timestamp = float(timestamp_sec)
        if self._pose is None or self._timestamp is None:
            self._pose = current.copy()
            self._timestamp = timestamp
            return 0.0, 0.0
        dt = timestamp - self._timestamp
        if dt <= 1.0e-6:
            return 0.0, 0.0
        dx = float(current[0] - self._pose[0])
        dy = float(current[1] - self._pose[1])
        yaw_ref = float(self._pose[2])
        signed_linear = (math.cos(yaw_ref) * dx + math.sin(yaw_ref) * dy) / dt
        angular = float(wrap_angle(float(current[2] - self._pose[2]))) / dt
        self._pose = current.copy()
        self._timestamp = timestamp
        return float(signed_linear), float(angular)
