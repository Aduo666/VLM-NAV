from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Callable

import numpy as np

from vln_nav.navigation.trajectory_clearance import (
    FootprintCollisionChecker,
    GeometryStatus,
    PlanningGrid,
)


@dataclass(frozen=True)
class TrajectoryRefinerConfig:
    iterations: int
    smoothing_gain: float
    minimum_clearance_m: float
    maximum_curvature_per_m: float

    @classmethod
    def from_mapping(cls, data: dict) -> "TrajectoryRefinerConfig":
        expected = {item.name for item in fields(cls)}
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError(f"trajectory_refiner 必须包含这些字段：{sorted(expected)}")
        config = cls(
            iterations=int(data["iterations"]),
            smoothing_gain=float(data["smoothing_gain"]),
            minimum_clearance_m=float(data["minimum_clearance_m"]),
            maximum_curvature_per_m=float(data["maximum_curvature_per_m"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not 1 <= self.iterations <= 100:
            raise ValueError("trajectory_refiner.iterations 必须位于 1 至 100")
        if not 0.0 < self.smoothing_gain < 1.0:
            raise ValueError("trajectory_refiner.smoothing_gain 必须位于 0 至 1")
        if not math.isfinite(self.minimum_clearance_m) or self.minimum_clearance_m <= 0:
            raise ValueError("trajectory_refiner.minimum_clearance_m 必须为有限正数")
        if not math.isfinite(self.maximum_curvature_per_m) or self.maximum_curvature_per_m <= 0:
            raise ValueError("trajectory_refiner.maximum_curvature_per_m 必须为有限正数")


@dataclass(frozen=True)
class RefinedTrajectory:
    path_world_xy: np.ndarray
    path_world_yaw_rad: np.ndarray
    applied: bool
    accepted_iterations: int


class TrajectoryRefiner:
    def __init__(
        self,
        config: TrajectoryRefinerConfig,
        collision_checker: FootprintCollisionChecker,
    ) -> None:
        config.validate()
        self.config = config
        self.collision_checker = collision_checker

    def refine(
        self,
        *,
        grid: PlanningGrid,
        path_world_xy: np.ndarray,
        path_world_yaw_rad: np.ndarray,
        motion_direction: np.ndarray,
        pose_allowed: Callable[[float, float, float], bool] | None = None,
    ) -> RefinedTrajectory:
        points = np.asarray(path_world_xy, dtype=np.float64).copy()
        yaw = np.asarray(path_world_yaw_rad, dtype=np.float64).reshape(-1).copy()
        direction = np.asarray(motion_direction, dtype=np.int8).reshape(-1)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 3
            or len(yaw) != len(points)
            or len(direction) != len(points)
        ):
            return RefinedTrajectory(points, yaw, False, 0)
        objective = _smoothness_objective(points, yaw)
        accepted = 0
        for _ in range(self.config.iterations):
            candidate = points.copy()
            for index in range(1, len(points) - 1):
                if (
                    direction[index] == 0
                    or direction[index - 1] != direction[index]
                    or direction[index + 1] != direction[index]
                ):
                    continue
                midpoint = 0.5 * (points[index - 1] + points[index + 1])
                candidate[index] += self.config.smoothing_gain * (
                    midpoint - points[index]
                )
            candidate_yaw = _path_yaw(candidate, yaw, direction)
            candidate_objective = _smoothness_objective(candidate, candidate_yaw)
            if candidate_objective >= objective - 1.0e-9:
                break
            if not self._valid(
                grid=grid,
                points=candidate,
                yaw=candidate_yaw,
                direction=direction,
                pose_allowed=pose_allowed,
            ):
                break
            points = candidate
            yaw = candidate_yaw
            objective = candidate_objective
            accepted += 1
        return RefinedTrajectory(
            path_world_xy=np.ascontiguousarray(points),
            path_world_yaw_rad=np.ascontiguousarray(yaw),
            applied=accepted > 0,
            accepted_iterations=accepted,
        )

    def _valid(
        self,
        *,
        grid: PlanningGrid,
        points: np.ndarray,
        yaw: np.ndarray,
        direction: np.ndarray,
        pose_allowed: Callable[[float, float, float], bool] | None,
    ) -> bool:
        for index, (point, heading) in enumerate(zip(points, yaw)):
            assessment = self.collision_checker.classify_pose(
                grid,
                x_m=float(point[0]),
                y_m=float(point[1]),
                yaw_rad=float(heading),
            )
            if (
                assessment.status is not GeometryStatus.SAFE
                or assessment.min_clearance_m < self.config.minimum_clearance_m
            ):
                return False
            if pose_allowed is not None and not pose_allowed(
                float(point[0]), float(point[1]), float(heading)
            ):
                return False
            if index == 0 or direction[index] == 0:
                continue
            distance = float(np.linalg.norm(points[index] - points[index - 1]))
            if distance < 1.0e-5:
                continue
            curvature = abs(_wrap_angle(yaw[index] - yaw[index - 1])) / distance
            if curvature > self.config.maximum_curvature_per_m:
                return False
        assessment = self.collision_checker.classify_swept_poses(
            grid,
            np.column_stack((points, yaw)),
        )
        return bool(
            assessment.status is GeometryStatus.SAFE
            and assessment.min_clearance_m >= self.config.minimum_clearance_m
        )


def _path_yaw(
    points: np.ndarray,
    original_yaw: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    yaw = original_yaw.copy()
    for index in range(1, len(points)):
        if direction[index] == 0:
            continue
        delta = points[index] - points[index - 1]
        if np.linalg.norm(delta) < 1.0e-6:
            continue
        heading = math.atan2(float(delta[1]), float(delta[0]))
        if direction[index] < 0:
            heading = _wrap_angle(heading + math.pi)
        yaw[index] = heading
    if len(yaw) > 1 and direction[1] != 0:
        yaw[0] = yaw[1]
    return np.asarray([_wrap_angle(value) for value in yaw], dtype=np.float64)


def _smoothness_objective(points: np.ndarray, yaw: np.ndarray) -> float:
    length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    if len(points) < 3:
        return length
    second_difference = points[:-2] - 2.0 * points[1:-1] + points[2:]
    position_smoothness = float(np.sum(second_difference**2))
    yaw_delta = np.asarray(
        [_wrap_angle(yaw[index + 1] - yaw[index]) for index in range(len(yaw) - 1)]
    )
    yaw_smoothness = float(np.sum(np.diff(yaw_delta) ** 2)) if len(yaw_delta) > 1 else 0.0
    return length + 4.0 * position_smoothness + 0.2 * yaw_smoothness


def _wrap_angle(angle_rad: float) -> float:
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi
