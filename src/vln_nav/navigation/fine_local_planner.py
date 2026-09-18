from __future__ import annotations

from dataclasses import dataclass, fields
import heapq
import math

import numpy as np

from vln_nav.navigation.trajectory_clearance import (
    FootprintCollisionChecker,
    PlanningGrid,
)


@dataclass(frozen=True)
class FinePlannerConfig:
    yaw_resolution_deg: float
    step_length_m: float
    collision_sample_step_m: float
    curvature_samples_per_m: tuple[float, ...]
    goal_tolerance_m: float
    goal_yaw_tolerance_deg: float
    max_plan_distance_m: float
    max_expansions: int
    unknown_block_radius_m: float
    max_near_unknown_ratio: float
    min_near_unknown_area_m2: float
    clearance_weight: float
    unknown_weight: float
    curvature_weight: float
    curvature_change_weight: float
    reverse_penalty: float
    direction_change_penalty: float
    spin_step_deg: float
    spin_penalty: float
    heuristic_weight: float

    @classmethod
    def from_mapping(cls, data: dict) -> "FinePlannerConfig":
        expected = {item.name for item in fields(cls)}
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError(f"fine_planner 必须包含这些字段：{sorted(expected)}")
        values = dict(data)
        values["curvature_samples_per_m"] = tuple(
            float(value) for value in data["curvature_samples_per_m"]
        )
        values["max_expansions"] = int(data["max_expansions"])
        config = cls(**{
            name: values[name]
            if name in {"curvature_samples_per_m", "max_expansions"}
            else float(values[name])
            for name in expected
        })
        config.validate()
        return config

    def validate(self) -> None:
        positive = (
            self.yaw_resolution_deg,
            self.step_length_m,
            self.collision_sample_step_m,
            self.goal_tolerance_m,
            self.goal_yaw_tolerance_deg,
            self.max_plan_distance_m,
            self.unknown_block_radius_m,
            self.min_near_unknown_area_m2,
            self.clearance_weight,
            self.unknown_weight,
            self.curvature_weight,
            self.curvature_change_weight,
            self.reverse_penalty,
            self.direction_change_penalty,
            self.spin_step_deg,
            self.spin_penalty,
            self.heuristic_weight,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0 for value in positive):
            raise ValueError("细粒度规划器数值配置必须为有限正数")
        if not 1.0 <= self.yaw_resolution_deg <= 30.0:
            raise ValueError("细粒度规划器朝向分辨率必须位于 1 至 30 度范围内")
        if not 1.0 <= self.goal_yaw_tolerance_deg <= 90.0:
            raise ValueError("细粒度目标航向容差必须位于 1 至 90 度范围内")
        if self.collision_sample_step_m > self.step_length_m:
            raise ValueError("碰撞采样间隔不能超过运动段长度")
        if self.max_expansions < 100:
            raise ValueError("细粒度规划器 max_expansions 至少为 100")
        if not 0 <= self.max_near_unknown_ratio <= 1:
            raise ValueError("细粒度规划器未知比例必须位于 0 至 1 范围内")
        if not self.curvature_samples_per_m or 0.0 not in self.curvature_samples_per_m:
            raise ValueError("细粒度规划器曲率样本必须包含零")
        if not all(math.isfinite(value) for value in self.curvature_samples_per_m):
            raise ValueError("细粒度规划器曲率样本必须为有限数值")


@dataclass(frozen=True)
class FinePlan:
    path_world_xy: np.ndarray
    path_world_yaw_rad: np.ndarray
    path_motion_direction: np.ndarray
    expanded_node_count: int
    path_cost: float
    map_version: int
    local_goal_world_pose: tuple[float, float, float]
    rejected_collision_count: int
    rejected_unknown_count: int
    rejected_outside_count: int


@dataclass(frozen=True)
class _MotionPrimitive:
    name: str
    direction: int
    curvature_per_m: float
    spin_rad: float


class FineLocalPlanner:
    def __init__(
        self,
        config: FinePlannerConfig,
        collision_checker: FootprintCollisionChecker,
    ) -> None:
        config.validate()
        self.config = config
        self.collision_checker = collision_checker
        self.footprint = collision_checker.footprint
        self._yaw_step = math.radians(config.yaw_resolution_deg)
        self._yaw_bins = int(round(2.0 * math.pi / self._yaw_step))
        self._actions = tuple(
            [
                _MotionPrimitive(
                    name=f"forward_{index}",
                    direction=1,
                    curvature_per_m=float(curvature),
                    spin_rad=0.0,
                )
                for index, curvature in enumerate(config.curvature_samples_per_m)
            ]
            + [
                _MotionPrimitive(
                    name=f"reverse_{index}",
                    direction=-1,
                    curvature_per_m=float(curvature),
                    spin_rad=0.0,
                )
                for index, curvature in enumerate(config.curvature_samples_per_m)
            ]
            + [
                _MotionPrimitive("spin_left", 0, 0.0, math.radians(config.spin_step_deg)),
                _MotionPrimitive("spin_right", 0, 0.0, -math.radians(config.spin_step_deg)),
            ]
        )

    def plan(
        self,
        *,
        grid: PlanningGrid,
        start_pose_world: tuple[float, float, float],
        goal_world_pose: tuple[float, float, float],
    ) -> FinePlan:
        start = np.asarray(start_pose_world, dtype=np.float64)
        goal = np.asarray(goal_world_pose, dtype=np.float64)
        if start.shape != (3,) or goal.shape != (3,) or not (
            np.isfinite(start).all() and np.isfinite(goal).all()
        ):
            raise ValueError("细粒度规划器需要有限的起始位姿和目标坐标")
        goal_delta = goal[:2] - start[:2]
        goal_distance = float(np.linalg.norm(goal_delta))
        if goal_distance > self.config.max_plan_distance_m:
            goal[:2] = (
                start[:2]
                + goal_delta / goal_distance * self.config.max_plan_distance_m
            )
        start_key = self._state_key(grid, start[0], start[1], start[2], -1)
        if start_key is None:
            raise RuntimeError("细粒度规划器起始位姿位于局部地图外部")

        queue: list[tuple[float, int, tuple[int, int, int, int]]] = []
        counter = 0
        g_score = {start_key: 0.0}
        pose_by_key = {start_key: (float(start[0]), float(start[1]), float(start[2]))}
        parent: dict[tuple[int, int, int, int], tuple[int, int, int, int] | None] = {
            start_key: None
        }
        heapq.heappush(
            queue,
            (
                self.config.heuristic_weight * self._heuristic(start, goal),
                counter,
                start_key,
            ),
        )
        goal_key = None
        expanded = 0
        rejected_collision = 0
        rejected_unknown = 0
        rejected_outside = 0

        while queue and expanded < self.config.max_expansions:
            _, _, key = heapq.heappop(queue)
            pose = pose_by_key[key]
            cost = g_score[key]
            goal_position_error = math.hypot(pose[0] - goal[0], pose[1] - goal[1])
            goal_yaw_error = abs(self._wrap_angle(pose[2] - goal[2]))
            if (
                goal_position_error <= self.config.goal_tolerance_m
                and goal_yaw_error <= math.radians(self.config.goal_yaw_tolerance_deg)
            ):
                goal_key = key
                break
            expanded += 1
            previous_action = self._actions[key[3]] if key[3] >= 0 else None
            for action_index, action in enumerate(self._actions):
                primitive = self._sample_primitive(pose, action)
                metrics = [
                    self.collision_checker.metrics(
                        grid,
                        x_m=float(sample[0]),
                        y_m=float(sample[1]),
                        yaw_rad=float(sample[2]),
                    )
                    for sample in primitive[1:]
                ]
                if any(item.outside_ratio > 0 for item in metrics):
                    rejected_outside += 1
                    continue
                if any(item.collision for item in metrics):
                    rejected_collision += 1
                    continue
                unknown_metrics = self.collision_checker.incremental_unknown_metrics(
                    grid,
                    initial_pose=start,
                    swept_poses=primitive[1:],
                )
                unknown_ratio = unknown_metrics.ratio
                distance_from_start = math.hypot(
                    primitive[-1, 0] - start[0], primitive[-1, 1] - start[1]
                )
                if (
                    distance_from_start <= self.config.unknown_block_radius_m
                    and unknown_ratio > self.config.max_near_unknown_ratio
                    and unknown_metrics.area_m2
                    >= self.config.min_near_unknown_area_m2
                ):
                    rejected_unknown += 1
                    continue
                endpoint = primitive[-1]
                next_key = self._state_key(
                    grid,
                    endpoint[0],
                    endpoint[1],
                    endpoint[2],
                    action_index,
                )
                if next_key is None:
                    continue
                min_clearance = min(
                    (item.min_clearance_m for item in metrics), default=0.0
                )
                transition_length = (
                    self.config.step_length_m
                    if action.direction != 0
                    else self.collision_checker.bounding_radius_m * abs(action.spin_rad)
                )
                transition_cost = transition_length
                if action.direction < 0:
                    transition_cost *= self.config.reverse_penalty
                elif action.direction == 0:
                    transition_cost *= self.config.spin_penalty
                if (
                    previous_action is not None
                    and previous_action.direction != 0
                    and action.direction != 0
                    and previous_action.direction != action.direction
                ):
                    transition_cost *= self.config.direction_change_penalty
                transition_cost += (
                    self.config.clearance_weight
                    * transition_length
                    / max(min_clearance, 0.02)
                )
                transition_cost += (
                    self.config.unknown_weight
                    * unknown_ratio
                    * transition_length
                )
                transition_cost += (
                    self.config.curvature_weight
                    * abs(action.curvature_per_m)
                    * transition_length
                )
                if previous_action is not None:
                    transition_cost += (
                        self.config.curvature_change_weight
                        * abs(
                            action.curvature_per_m
                            - previous_action.curvature_per_m
                        )
                    )
                new_cost = cost + transition_cost
                if new_cost >= g_score.get(next_key, float("inf")):
                    continue
                g_score[next_key] = new_cost
                pose_by_key[next_key] = tuple(map(float, endpoint))
                parent[next_key] = key
                counter += 1
                priority = new_cost + self.config.heuristic_weight * self._heuristic(
                    endpoint, goal
                )
                heapq.heappush(queue, (priority, counter, next_key))

        if goal_key is None:
            raise RuntimeError(
                f"细粒度规划器扩展 {expanded} 个节点后仍未找到路径；"
                f"碰撞拒绝={rejected_collision}，未知拒绝={rejected_unknown}，"
                f"地图外拒绝={rejected_outside}"
            )
        keys = []
        current = goal_key
        while current is not None:
            keys.append(current)
            current = parent[current]
        keys.reverse()
        poses = np.asarray([pose_by_key[key] for key in keys], dtype=np.float64)
        motion_direction = np.asarray(
            [0] + [self._actions[key[3]].direction for key in keys[1:]],
            dtype=np.int8,
        )
        if len(poses) < 2:
            raise RuntimeError("细粒度规划器返回了零长度路径")
        return FinePlan(
            path_world_xy=np.ascontiguousarray(poses[:, :2]),
            path_world_yaw_rad=np.ascontiguousarray(poses[:, 2]),
            path_motion_direction=np.ascontiguousarray(motion_direction),
            expanded_node_count=int(expanded),
            path_cost=float(g_score[goal_key]),
            map_version=int(grid.map_version),
            local_goal_world_pose=(float(goal[0]), float(goal[1]), float(goal[2])),
            rejected_collision_count=int(rejected_collision),
            rejected_unknown_count=int(rejected_unknown),
            rejected_outside_count=int(rejected_outside),
        )

    def _state_key(
        self,
        grid: PlanningGrid,
        x_m: float,
        y_m: float,
        yaw_rad: float,
        curvature_index: int,
    ) -> tuple[int, int, int, int] | None:
        row, column, inside = grid.world_to_cell(x_m, y_m)
        if not inside:
            return None
        yaw = (float(yaw_rad) + math.pi) % (2.0 * math.pi)
        yaw_index = int(round(yaw / self._yaw_step)) % self._yaw_bins
        return row, column, yaw_index, int(curvature_index)

    def _sample_primitive(
        self,
        pose: tuple[float, float, float],
        action: _MotionPrimitive,
    ) -> np.ndarray:
        x, y, yaw = map(float, pose)
        if action.direction == 0:
            sample_count = max(
                2,
                int(math.ceil(abs(action.spin_rad) / self._yaw_step)) + 1,
            )
            sample_yaw = yaw + np.linspace(0.0, action.spin_rad, sample_count)
            sample_yaw = (sample_yaw + math.pi) % (2.0 * math.pi) - math.pi
            return np.column_stack(
                (
                    np.full_like(sample_yaw, x),
                    np.full_like(sample_yaw, y),
                    sample_yaw,
                )
            )
        sample_count = max(
            2,
            int(math.ceil(
                self.config.step_length_m / self.config.collision_sample_step_m
            )) + 1,
        )
        distances = np.linspace(
            0.0,
            float(action.direction) * self.config.step_length_m,
            sample_count,
        )
        curvature = action.curvature_per_m
        if abs(curvature) < 1.0e-9:
            sample_x = x + distances * math.cos(yaw)
            sample_y = y + distances * math.sin(yaw)
            sample_yaw = np.full_like(distances, yaw)
        else:
            sample_yaw = yaw + curvature * distances
            sample_x = x + (np.sin(sample_yaw) - math.sin(yaw)) / curvature
            sample_y = y + (-np.cos(sample_yaw) + math.cos(yaw)) / curvature
        sample_yaw = (sample_yaw + math.pi) % (2.0 * math.pi) - math.pi
        return np.column_stack((sample_x, sample_y, sample_yaw))

    @staticmethod
    def _heuristic(pose: np.ndarray, goal_pose: np.ndarray) -> float:
        position_cost = float(
            np.linalg.norm(np.asarray(pose[:2]) - np.asarray(goal_pose[:2]))
        )
        heading_cost = 0.10 * abs(
            FineLocalPlanner._wrap_angle(float(pose[2]) - float(goal_pose[2]))
        )
        return position_cost + heading_cost

    @staticmethod
    def _wrap_angle(angle_rad: float) -> float:
        return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi
