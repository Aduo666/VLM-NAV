from __future__ import annotations

import math

import numpy as np

from vln_nav.control.trajectory import (
    TrackingResult,
    TrajectoryConfig,
    VelocityCommand,
    wheel_targets,
)


class FinePathTracker:
    def __init__(
        self,
        path_xy: np.ndarray,
        path_yaw_rad: np.ndarray,
        motion_direction: np.ndarray,
        config: TrajectoryConfig,
        *,
        initial_speed_mps: float = 0.0,
        waypoint_tolerance_m: float = 0.04,
        yaw_tolerance_deg: float = 3.0,
    ) -> None:
        config.validate()
        points = np.asarray(path_xy, dtype=np.float64)
        yaw = np.asarray(path_yaw_rad, dtype=np.float64).reshape(-1)
        direction = np.asarray(motion_direction, dtype=np.int8).reshape(-1)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("细粒度轨迹至少需要两个 XY 位姿")
        if len(yaw) != len(points) or len(direction) != len(points):
            raise ValueError("细粒度轨迹的位置、航向和运动方向数量必须一致")
        if not np.isfinite(points).all() or not np.isfinite(yaw).all():
            raise ValueError("细粒度轨迹包含无效数值")
        if direction[0] != 0 or not np.isin(direction, (-1, 0, 1)).all():
            raise ValueError("细粒度轨迹运动方向必须属于 -1、0、1")
        if not math.isfinite(initial_speed_mps):
            raise ValueError("细粒度轨迹初始速度必须有效")
        self.path = points
        self.yaw = yaw
        self.direction = direction
        self.config = config
        self.waypoint_tolerance_m = float(waypoint_tolerance_m)
        self.yaw_tolerance_rad = math.radians(float(yaw_tolerance_deg))
        self.segment_index = 0
        self.elapsed = 0.0
        self.speed = min(abs(float(initial_speed_mps)), config.max_linear_mps)
        self.finished = False
        self.fault: str | None = None
        delta = np.diff(points, axis=0)
        translation = np.linalg.norm(delta, axis=1)
        yaw_delta = np.abs(np.asarray([
            _wrap_angle(yaw[index + 1] - yaw[index])
            for index in range(len(yaw) - 1)
        ]))
        turn_radius = max(config.lookahead_m, 0.10)
        self._segment_length = np.where(
            direction[1:] == 0,
            yaw_delta * turn_radius,
            translation,
        )
        self._remaining = np.r_[np.cumsum(self._segment_length[::-1])[::-1], 0.0]

    def feedback(self, applied: VelocityCommand) -> None:
        self.speed = abs(float(applied.linear_mps))

    def step(self, pose, dt: float) -> TrackingResult:
        x, y, yaw = map(float, pose)
        if not all(math.isfinite(value) for value in (x, y, yaw, dt)) or dt <= 0:
            raise ValueError("细粒度轨迹跟踪需要有效位姿和正时间步长")
        self.elapsed += dt
        position = np.asarray([x, y], dtype=np.float64)
        while self.segment_index < len(self.path) - 1:
            next_index = self.segment_index + 1
            direction = int(self.direction[next_index])
            if direction == 0:
                reached = abs(_wrap_angle(self.yaw[next_index] - yaw)) <= self.yaw_tolerance_rad
            else:
                segment = self.path[next_index] - self.path[self.segment_index]
                passed = float(np.dot(position - self.path[next_index], segment)) >= 0.0
                reached = (
                    float(np.linalg.norm(position - self.path[next_index]))
                    <= self.waypoint_tolerance_m
                    or passed
                )
            if not reached:
                break
            self.segment_index = next_index

        if self.segment_index >= len(self.path) - 1:
            self.finished = True
            return TrackingResult(
                VelocityCommand(),
                "ARRIVED",
                0.0,
                float(np.linalg.norm(position - self.path[-1])),
                tuple(map(float, self.path[-1])),
            )

        next_index = self.segment_index + 1
        direction = int(self.direction[next_index])
        target = self.path[next_index]
        remaining = float(
            np.linalg.norm(position - target) + self._remaining[next_index]
        )
        segment = target - self.path[self.segment_index]
        segment_norm = float(np.linalg.norm(segment))
        if segment_norm > 1.0e-9:
            cross_track = abs(float(np.cross(segment, position - self.path[self.segment_index])))
            cross_track /= segment_norm
        else:
            cross_track = 0.0
        if cross_track > self.config.max_path_error_m:
            self.fault = "PATH_ERROR"
        elif self.elapsed > self.config.timeout_sec:
            self.fault = "TIMEOUT"
        if self.fault is not None:
            return TrackingResult(
                VelocityCommand(),
                self.fault,
                remaining,
                cross_track,
                tuple(map(float, target)),
            )

        if direction == 0:
            yaw_error = _wrap_angle(self.yaw[next_index] - yaw)
            angular = float(np.clip(
                2.0 * yaw_error,
                -self.config.max_angular_radps,
                self.config.max_angular_radps,
            ))
            command = VelocityCommand(0.0, angular)
        else:
            segment_start = self.path[self.segment_index]
            segment_yaw_start = self.yaw[self.segment_index]
            segment_yaw_delta = _wrap_angle(
                self.yaw[next_index] - segment_yaw_start
            )
            progress = float(np.clip(
                np.dot(position - segment_start, segment)
                / max(segment_norm**2, 1.0e-9),
                0.0,
                1.0,
            ))
            reference_yaw = _wrap_angle(
                segment_yaw_start + progress * segment_yaw_delta
            )
            yaw_error = _wrap_angle(reference_yaw - yaw)
            speed_limit = min(
                self.config.max_linear_mps,
                math.sqrt(2.0 * self.config.max_acceleration_mps2 * max(remaining, 0.0)),
            )
            speed = min(
                speed_limit,
                self.speed + self.config.max_acceleration_mps2 * dt,
            )
            linear = float(direction) * speed
            planned_curvature = segment_yaw_delta / (
                float(direction) * max(segment_norm, 1.0e-9)
            )
            angular = float(np.clip(
                linear * planned_curvature + 1.5 * yaw_error,
                -self.config.max_angular_radps,
                self.config.max_angular_radps,
            ))
            command = _fit_agx_command(
                VelocityCommand(linear, angular),
                self.config,
            )
        return TrackingResult(
            command,
            "TRACKING",
            remaining,
            cross_track,
            tuple(map(float, target)),
            command,
        )


def _wrap_angle(angle_rad: float) -> float:
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def _fit_agx_command(
    command: VelocityCommand,
    config: TrajectoryConfig,
) -> VelocityCommand:
    try:
        return wheel_targets(command, config)[2]
    except ValueError as exc:
        if "steering limit" not in str(exc):
            raise
    lower = 0.0
    upper = 1.0
    feasible = wheel_targets(
        VelocityCommand(command.linear_mps, 0.0),
        config,
    )[2]
    for _ in range(16):
        scale = 0.5 * (lower + upper)
        candidate = VelocityCommand(
            command.linear_mps,
            command.angular_radps * scale,
        )
        try:
            feasible = wheel_targets(candidate, config)[2]
            lower = scale
        except ValueError as exc:
            if "steering limit" not in str(exc):
                raise
            upper = scale
    return feasible
