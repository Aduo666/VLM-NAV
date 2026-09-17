from __future__ import annotations

from dataclasses import dataclass, fields
import heapq
import math

import numpy as np

from vln_nav.navigation.trajectory_clearance import FootprintConfig, PlanningGrid


@dataclass(frozen=True)
class FinePlannerConfig:
    yaw_resolution_deg: float
    step_length_m: float
    collision_sample_step_m: float
    curvature_samples_per_m: tuple[float, ...]
    goal_tolerance_m: float
    max_plan_distance_m: float
    max_expansions: int
    unknown_block_radius_m: float
    max_near_unknown_ratio: float
    clearance_weight: float
    unknown_weight: float
    curvature_weight: float
    curvature_change_weight: float
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
            self.max_plan_distance_m,
            self.unknown_block_radius_m,
            self.clearance_weight,
            self.unknown_weight,
            self.curvature_weight,
            self.curvature_change_weight,
            self.heuristic_weight,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0 for value in positive):
            raise ValueError("细粒度规划器数值配置必须为有限正数")
        if not 1.0 <= self.yaw_resolution_deg <= 30.0:
            raise ValueError("细粒度规划器朝向分辨率必须位于 1 至 30 度范围内")
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
    expanded_node_count: int
    path_cost: float
    map_version: int
    local_goal_world_xy: tuple[float, float]
    rejected_collision_count: int
    rejected_unknown_count: int
    rejected_outside_count: int


class FineLocalPlanner:
    def __init__(self, config: FinePlannerConfig, footprint: FootprintConfig) -> None:
        config.validate()
        footprint.validate()
        self.config = config
        self.footprint = footprint
        self._yaw_step = math.radians(config.yaw_resolution_deg)
        self._yaw_bins = int(round(2.0 * math.pi / self._yaw_step))
        self._footprint_offset_cache: dict[tuple[float, int], np.ndarray] = {}

    def plan(
        self,
        *,
        grid: PlanningGrid,
        start_pose_world: tuple[float, float, float],
        goal_world_xy: tuple[float, float],
    ) -> FinePlan:
        start = np.asarray(start_pose_world, dtype=np.float64)
        goal = np.asarray(goal_world_xy, dtype=np.float64)
        if start.shape != (3,) or goal.shape != (2,) or not (
            np.isfinite(start).all() and np.isfinite(goal).all()
        ):
            raise ValueError("细粒度规划器需要有限的起始位姿和目标坐标")
        goal_delta = goal - start[:2]
        goal_distance = float(np.linalg.norm(goal_delta))
        if goal_distance > self.config.max_plan_distance_m:
            goal = start[:2] + goal_delta / goal_distance * self.config.max_plan_distance_m
        start_key = self._state_key(grid, start[0], start[1], start[2], 0)
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
                self.config.heuristic_weight * self._heuristic(start[:2], goal),
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
            if math.hypot(pose[0] - goal[0], pose[1] - goal[1]) <= self.config.goal_tolerance_m:
                goal_key = key
                break
            expanded += 1
            previous_curvature = self.config.curvature_samples_per_m[key[3]]
            for curvature_index, curvature in enumerate(self.config.curvature_samples_per_m):
                primitive = self._sample_primitive(pose, curvature)
                metrics = [self._footprint_metrics(grid, sample) for sample in primitive[1:]]
                if any(item[3] > 0 for item in metrics):
                    rejected_outside += 1
                    continue
                if any(item[0] for item in metrics):
                    rejected_collision += 1
                    continue
                distance_from_start = math.hypot(
                    primitive[-1, 0] - start[0], primitive[-1, 1] - start[1]
                )
                if (
                    distance_from_start <= self.config.unknown_block_radius_m
                    and max((item[2] for item in metrics), default=0.0)
                    > self.config.max_near_unknown_ratio
                ):
                    rejected_unknown += 1
                    continue
                endpoint = primitive[-1]
                next_key = self._state_key(
                    grid,
                    endpoint[0],
                    endpoint[1],
                    endpoint[2],
                    curvature_index,
                )
                if next_key is None:
                    continue
                min_clearance = min(
                    (item[1] for item in metrics), default=0.0
                )
                unknown_ratio = max(
                    (item[2] for item in metrics), default=0.0
                )
                transition_cost = self.config.step_length_m
                transition_cost += (
                    self.config.clearance_weight
                    * self.config.step_length_m
                    / max(min_clearance, 0.02)
                )
                transition_cost += (
                    self.config.unknown_weight
                    * unknown_ratio
                    * self.config.step_length_m
                )
                transition_cost += (
                    self.config.curvature_weight
                    * abs(curvature)
                    * self.config.step_length_m
                )
                transition_cost += (
                    self.config.curvature_change_weight
                    * abs(curvature - previous_curvature)
                )
                new_cost = cost + transition_cost
                if new_cost >= g_score.get(next_key, float("inf")):
                    continue
                g_score[next_key] = new_cost
                pose_by_key[next_key] = tuple(map(float, endpoint))
                parent[next_key] = key
                counter += 1
                priority = new_cost + self.config.heuristic_weight * self._heuristic(
                    endpoint[:2], goal
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
        if len(poses) < 2:
            raise RuntimeError("细粒度规划器返回了零长度路径")
        return FinePlan(
            path_world_xy=np.ascontiguousarray(poses[:, :2]),
            path_world_yaw_rad=np.ascontiguousarray(poses[:, 2]),
            expanded_node_count=int(expanded),
            path_cost=float(g_score[goal_key]),
            map_version=int(grid.map_version),
            local_goal_world_xy=(float(goal[0]), float(goal[1])),
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
        curvature: float,
    ) -> np.ndarray:
        sample_count = max(
            2,
            int(math.ceil(
                self.config.step_length_m / self.config.collision_sample_step_m
            )) + 1,
        )
        distances = np.linspace(0.0, self.config.step_length_m, sample_count)
        x, y, yaw = map(float, pose)
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

    def _footprint_metrics(
        self,
        grid: PlanningGrid,
        pose: np.ndarray,
    ) -> tuple[bool, float, float, float]:
        row, column, inside_center = grid.world_to_cell(
            float(pose[0]), float(pose[1])
        )
        if not inside_center:
            return False, 0.0, 1.0, 1.0
        yaw_index = int(
            round(((float(pose[2]) + math.pi) % (2.0 * math.pi)) / self._yaw_step)
        ) % self._yaw_bins
        cache_key = (float(grid.resolution_m), yaw_index)
        offsets = self._footprint_offset_cache.get(cache_key)
        if offsets is None:
            yaw = yaw_index * self._yaw_step - math.pi
            local_x = np.arange(
                -self.footprint.rear_m - self.footprint.margin_m,
                self.footprint.front_m + self.footprint.margin_m + 0.5 * grid.resolution_m,
                grid.resolution_m,
            )
            local_y = np.arange(
                -self.footprint.half_width_m - self.footprint.margin_m,
                self.footprint.half_width_m + self.footprint.margin_m + 0.5 * grid.resolution_m,
                grid.resolution_m,
            )
            xx, yy = np.meshgrid(local_x, local_y, indexing="xy")
            cosine, sine = math.cos(yaw), math.sin(yaw)
            column_offset = np.rint(
                (cosine * xx - sine * yy) / grid.resolution_m
            ).astype(np.int64)
            row_offset = np.rint(
                (sine * xx + cosine * yy) / grid.resolution_m
            ).astype(np.int64)
            offsets = np.unique(
                np.column_stack((row_offset.ravel(), column_offset.ravel())), axis=0
            )
            self._footprint_offset_cache[cache_key] = offsets
        rows = row + offsets[:, 0]
        columns = column + offsets[:, 1]
        inside = (
            (rows >= 0)
            & (rows < grid.shape[0])
            & (columns >= 0)
            & (columns < grid.shape[1])
        )
        outside_ratio = float(1.0 - np.mean(inside))
        if not np.any(inside):
            return False, 0.0, 1.0, 1.0
        valid_rows = rows[inside]
        valid_columns = columns[inside]
        collision = bool(np.any(grid.occupied_mask[valid_rows, valid_columns]))
        unknown_count = int(
            np.count_nonzero(grid.unknown_mask[valid_rows, valid_columns])
        ) + int(np.count_nonzero(~inside))
        unknown_ratio = float(unknown_count / len(inside))
        min_clearance = float(np.min(grid.clearance_m[valid_rows, valid_columns]))
        return collision, min_clearance, unknown_ratio, outside_ratio

    @staticmethod
    def _heuristic(position_xy: np.ndarray, goal_xy: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(position_xy) - np.asarray(goal_xy)))
