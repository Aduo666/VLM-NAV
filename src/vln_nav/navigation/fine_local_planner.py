from __future__ import annotations

from dataclasses import dataclass, fields
import heapq
import math

import numpy as np

from vln_nav.navigation.trajectory_clearance import (
    FootprintCollisionChecker,
    PlanningGrid,
    stopping_distance_m,
)


@dataclass(frozen=True)
class FinePlanningConstraint:
    robust_world_poses: np.ndarray
    robust_layer_indices: np.ndarray
    restricted_start_layer: int
    restricted_end_layer: int
    position_tolerance_m: float
    heading_tolerance_deg: float
    corridor: object

    def allows(self, x_m: float, y_m: float, yaw_rad: float) -> bool:
        layer, _, _ = self.corridor.coordinates(
            np.asarray([x_m, y_m], dtype=np.float64),
            float(yaw_rad),
        )
        if not self.restricted_start_layer <= layer <= self.restricted_end_layer:
            return True
        candidates = np.flatnonzero(
            np.abs(self.robust_layer_indices - layer) <= 1
        )
        if candidates.size == 0:
            return False
        poses = self.robust_world_poses[candidates]
        position_error = np.linalg.norm(
            poses[:, :2] - np.asarray([x_m, y_m], dtype=np.float64),
            axis=1,
        )
        yaw_error = np.abs(np.asarray([
            _wrap_angle(float(value) - float(yaw_rad)) for value in poses[:, 2]
        ]))
        return bool(np.any(
            (position_error <= self.position_tolerance_m)
            & (yaw_error <= math.radians(self.heading_tolerance_deg))
        ))


@dataclass(frozen=True)
class FinePlannerConfig:
    yaw_resolution_deg: float
    step_length_m: float
    partial_path_max_distance_m: float
    partial_search_max_expansions: int
    collision_sample_step_m: float
    curvature_samples_per_m: tuple[float, ...]
    goal_tolerance_m: float
    max_plan_distance_m: float
    max_expansions: int
    clearance_weight: float
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
        values["partial_search_max_expansions"] = int(
            data["partial_search_max_expansions"]
        )
        config = cls(**{
            name: values[name]
            if name in {
                "curvature_samples_per_m",
                "max_expansions",
                "partial_search_max_expansions",
            }
            else float(values[name])
            for name in expected
        })
        config.validate()
        return config

    def validate(self) -> None:
        positive = (
            self.yaw_resolution_deg,
            self.step_length_m,
            self.partial_path_max_distance_m,
            self.collision_sample_step_m,
            self.goal_tolerance_m,
            self.max_plan_distance_m,
            self.clearance_weight,
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
        if self.collision_sample_step_m > self.step_length_m:
            raise ValueError("碰撞采样间隔不能超过运动段长度")
        if self.partial_path_max_distance_m < self.step_length_m:
            raise ValueError("增量路径最大距离不能短于单个运动段")
        if self.max_expansions < 100:
            raise ValueError("细粒度规划器 max_expansions 至少为 100")
        if not 1 <= self.partial_search_max_expansions <= self.max_expansions:
            raise ValueError("增量路径搜索节点上限必须位于 1 至 max_expansions 之间")
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
    local_goal_world_xy: tuple[float, float]
    rejected_collision_count: int
    rejected_unknown_count: int
    rejected_outside_count: int
    rejected_constraint_count: int
    max_unknown_ratio: float
    max_unknown_area_m2: float
    partial: bool = False
    refinement_applied: bool = False
    refinement_iterations: int = 0

    @property
    def status(self) -> str:
        return "SAFE_PARTIAL_PATH" if self.partial else "FULL_PATH"


@dataclass(frozen=True)
class FineActionDiagnostic:
    action_name: str
    endpoint_world_xyyaw: tuple[float, float, float]
    result: str
    unknown_ratio: float
    unknown_area_m2: float
    unknown_linear_indices: tuple[int, ...]
    part_unknown_counts: tuple[tuple[str, int], ...]
    part_unknown_linear_indices: tuple[tuple[str, tuple[int, ...]], ...]
    swept_linear_indices: tuple[int, ...]
    physical_unknown_count: int = 0
    safety_margin_unknown_count: int = 0
    physical_collision: bool = False
    safety_margin_collision: bool = False
    physical_unknown_linear_indices: tuple[int, ...] = ()
    safety_margin_unknown_linear_indices: tuple[int, ...] = ()


class FinePlanningFailure(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        reason: str,
        action_diagnostics: tuple[FineActionDiagnostic, ...],
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.action_diagnostics = action_diagnostics


class FineUnknownBlockedError(FinePlanningFailure):
    def __init__(
        self,
        message: str,
        unknown_linear_indices: np.ndarray,
        action_diagnostics: tuple[FineActionDiagnostic, ...],
        reason: str = "UNKNOWN_BLOCKED",
    ) -> None:
        super().__init__(
            message,
            reason=reason,
            action_diagnostics=action_diagnostics,
        )
        self.unknown_linear_indices = np.ascontiguousarray(
            unknown_linear_indices, dtype=np.int64
        )


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

    def inspect_initial_actions(
        self,
        *,
        grid: PlanningGrid,
        start_pose_world: tuple[float, float, float],
        constraint: FinePlanningConstraint | None = None,
    ) -> tuple[FineActionDiagnostic, ...]:
        initial = np.asarray(start_pose_world, dtype=np.float64).reshape(3)
        diagnostics = []
        for action in self._actions:
            primitive = self._sample_primitive(tuple(map(float, initial)), action)
            metrics = [
                self.collision_checker.metrics(
                    grid,
                    x_m=float(pose[0]),
                    y_m=float(pose[1]),
                    yaw_rad=float(pose[2]),
                )
                for pose in primitive[1:]
            ]
            unknown = self.collision_checker.incremental_unknown_metrics(
                grid,
                initial_pose=initial,
                swept_poses=primitive[1:],
            )
            part_counts = []
            part_unknown_cells = []
            all_swept_parts = []
            for part in self.footprint.parts:
                swept_parts = [
                    self.collision_checker.covered_linear_indices(
                        grid,
                        x_m=float(pose[0]),
                        y_m=float(pose[1]),
                        yaw_rad=float(pose[2]),
                        part_name=part.name,
                    )[0]
                    for pose in primitive[1:]
                ]
                part_cells = np.unique(np.concatenate(swept_parts))
                all_swept_parts.append(part_cells)
                unknown_part_cells = np.intersect1d(
                    part_cells,
                    unknown.unknown_linear_indices,
                    assume_unique=True,
                )
                part_counts.append((
                    part.name,
                    int(unknown_part_cells.size),
                ))
                part_unknown_cells.append((
                    part.name,
                    tuple(map(int, unknown_part_cells)),
                ))
            if constraint is not None and any(
                not constraint.allows(
                    float(pose[0]), float(pose[1]), float(pose[2])
                )
                for pose in primitive[1:]
            ):
                result = "CONSTRAINT"
            elif any(item.outside_ratio > 0 for item in metrics):
                result = "OUTSIDE"
            elif any(item.collision for item in metrics):
                result = "COLLISION"
            elif unknown.cell_count > 0:
                result = "UNKNOWN"
            else:
                result = "SAFE"
            diagnostics.append(FineActionDiagnostic(
                action_name=action.name,
                endpoint_world_xyyaw=tuple(map(float, primitive[-1])),
                result=result,
                unknown_ratio=unknown.ratio,
                unknown_area_m2=unknown.area_m2,
                unknown_linear_indices=tuple(
                    map(int, unknown.unknown_linear_indices)
                ),
                part_unknown_counts=tuple(part_counts),
                part_unknown_linear_indices=tuple(part_unknown_cells),
                swept_linear_indices=tuple(map(
                    int, np.unique(np.concatenate(all_swept_parts))
                )),
                physical_unknown_count=int(unknown.physical_unknown_linear_indices.size),
                safety_margin_unknown_count=int(unknown.safety_margin_unknown_linear_indices.size),
                physical_collision=any(item.physical_collision for item in metrics),
                safety_margin_collision=any(item.safety_margin_collision for item in metrics),
                physical_unknown_linear_indices=tuple(map(int, unknown.physical_unknown_linear_indices)),
                safety_margin_unknown_linear_indices=tuple(map(int, unknown.safety_margin_unknown_linear_indices)),
            ))
        return tuple(diagnostics)

    def _has_safe_partial_continuation(
        self,
        *,
        grid: PlanningGrid,
        initial_pose: np.ndarray,
        endpoint_pose: tuple[float, float, float],
        reference_path_world_xy: np.ndarray,
        travel_distance_m: float,
        constraint: FinePlanningConstraint | None,
    ) -> bool:
        endpoint = np.asarray(endpoint_pose, dtype=np.float64)
        route_position, lateral_error, tangent_yaw = _route_projection(
            endpoint[:2],
            reference_path_world_xy,
            max_progress_m=travel_distance_m + grid.resolution_m,
        )
        current_score = (
            route_position
            - 0.5 * lateral_error
            - 0.15 * abs(_wrap_angle(endpoint[2] - tangent_yaw))
        )
        for action in self._actions:
            if action.direction == 0:
                continue
            primitive = self._sample_primitive(endpoint_pose, action)
            if constraint is not None and any(
                not constraint.allows(
                    float(pose[0]), float(pose[1]), float(pose[2])
                )
                for pose in primitive[1:]
            ):
                continue
            metrics = [
                self.collision_checker.metrics(
                    grid,
                    x_m=float(pose[0]),
                    y_m=float(pose[1]),
                    yaw_rad=float(pose[2]),
                )
                for pose in primitive[1:]
            ]
            if any(item.collision or item.outside_ratio > 0 for item in metrics):
                continue
            unknown = self.collision_checker.incremental_unknown_metrics(
                grid,
                initial_pose=initial_pose,
                swept_poses=primitive[1:],
            )
            if unknown.cell_count > 0:
                continue
            next_pose = primitive[-1]
            next_position, next_lateral, next_tangent = _route_projection(
                next_pose[:2],
                reference_path_world_xy,
                max_progress_m=(
                    travel_distance_m
                    + self.config.step_length_m
                    + grid.resolution_m
                ),
            )
            next_score = (
                next_position
                - 0.5 * next_lateral
                - 0.15 * abs(_wrap_angle(next_pose[2] - next_tangent))
            )
            if next_score >= current_score + 0.5 * grid.resolution_m:
                return True
        return False

    def select_local_goal(
        self,
        *,
        grid: PlanningGrid,
        start_pose_world: tuple[float, float, float],
        target_world_xy: tuple[float, float],
        lookahead_m: float,
    ) -> tuple[float, float]:
        goal, _ = self.select_local_route(
            grid=grid,
            start_pose_world=start_pose_world,
            target_world_xy=target_world_xy,
            lookahead_m=lookahead_m,
        )
        return goal

    def select_local_route(
        self,
        *,
        grid: PlanningGrid,
        start_pose_world: tuple[float, float, float],
        target_world_xy: tuple[float, float],
        lookahead_m: float,
    ) -> tuple[tuple[float, float], np.ndarray]:
        start = np.asarray(start_pose_world, dtype=np.float64)
        target = np.asarray(target_world_xy, dtype=np.float64)
        if start.shape != (3,) or target.shape != (2,) or not (
            np.isfinite(start).all() and np.isfinite(target).all()
        ):
            raise ValueError("细粒度目标选择需要有效的机器人位姿和目标坐标")
        if not math.isfinite(lookahead_m) or lookahead_m <= 0:
            raise ValueError("细粒度局部目标前视距离必须为正数")
        start_row, start_column, start_inside = grid.world_to_cell(
            float(start[0]), float(start[1])
        )
        if not start_inside:
            raise RuntimeError("机器人位于局部地图范围外")

        minimum_clearance = (
            self.footprint.half_width_m + self.footprint.margin_m
        )
        traversable = (
            grid.free_mask
            & ~grid.occupied_mask
            & (grid.clearance_m >= minimum_clearance)
        ).copy()
        traversable[start_row, start_column] = True
        start_cell = (start_row, start_column)
        goal_row, goal_column, goal_inside = grid.world_to_cell(
            float(target[0]), float(target[1])
        )
        goal_cell = (goal_row, goal_column) if goal_inside else None
        goal_traversable = bool(
            goal_cell is not None and traversable[goal_row, goal_column]
        )

        queue = [(0.0, start_row, start_column)]
        distance = {start_cell: 0.0}
        parent: dict[tuple[int, int], tuple[int, int] | None] = {
            start_cell: None
        }
        best_cell = None
        best_score = float("inf")
        start_target_distance = float(np.linalg.norm(start[:2] - target))
        search_radius = self.config.max_plan_distance_m
        neighbors = (
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        )
        while queue:
            cost, row, column = heapq.heappop(queue)
            cell = (row, column)
            if cost > distance[cell] + 1.0e-9:
                continue
            cell_xy = np.asarray(
                [
                    grid.origin_x_m + (column + 0.5) * grid.resolution_m,
                    grid.origin_y_m + (row + 0.5) * grid.resolution_m,
                ],
                dtype=np.float64,
            )
            goal_distance = float(np.linalg.norm(cell_xy - target))
            if goal_cell == cell and goal_traversable:
                best_cell = cell
                break
            if (
                cell != start_cell
                and start_target_distance - goal_distance
                >= self.config.goal_tolerance_m
            ):
                score = goal_distance + 0.15 * cost
                if score < best_score:
                    best_score = score
                    best_cell = cell
            for delta_row, delta_column in neighbors:
                next_row = row + delta_row
                next_column = column + delta_column
                if not (
                    0 <= next_row < grid.shape[0]
                    and 0 <= next_column < grid.shape[1]
                    and traversable[next_row, next_column]
                ):
                    continue
                if delta_row and delta_column and not (
                    traversable[row + delta_row, column]
                    and traversable[row, column + delta_column]
                ):
                    continue
                next_x = grid.origin_x_m + (next_column + 0.5) * grid.resolution_m
                next_y = grid.origin_y_m + (next_row + 0.5) * grid.resolution_m
                if math.hypot(next_x - start[0], next_y - start[1]) > search_radius:
                    continue
                step = grid.resolution_m * math.hypot(delta_row, delta_column)
                next_cost = cost + step * (
                    1.0 + 0.05 / max(float(grid.clearance_m[next_row, next_column]), 0.05)
                )
                next_cell = (next_row, next_column)
                if next_cost >= distance.get(next_cell, float("inf")):
                    continue
                distance[next_cell] = next_cost
                parent[next_cell] = cell
                heapq.heappush(queue, (next_cost, next_row, next_column))

        if best_cell is None:
            raise RuntimeError("局部地图中没有朝目标可达的空闲区域")
        cells = []
        current = best_cell
        while current is not None:
            cells.append(current)
            current = parent[current]
        cells.reverse()
        path = np.asarray(
            [
                [
                    grid.origin_x_m + (column + 0.5) * grid.resolution_m,
                    grid.origin_y_m + (row + 0.5) * grid.resolution_m,
                ]
                for row, column in cells
            ],
            dtype=np.float64,
        )
        path[0] = start[:2]
        lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        limit = min(float(lookahead_m), self.config.max_plan_distance_m)
        for index in range(len(path) - 1, 0, -1):
            if lengths[index] > limit:
                continue
            if lengths[index] < self.config.goal_tolerance_m:
                break
            candidate = path[index]
            heading = math.atan2(
                float(path[index, 1] - path[index - 1, 1]),
                float(path[index, 0] - path[index - 1, 0]),
            )
            headings = (
                heading,
                float(start[2]),
                heading + 0.5 * math.pi,
                heading - 0.5 * math.pi,
                heading + math.pi,
            )
            if any(
                not metrics.collision and metrics.outside_ratio == 0.0
                for yaw in headings
                for metrics in [self.collision_checker.metrics(
                    grid,
                    x_m=float(candidate[0]),
                    y_m=float(candidate[1]),
                    yaw_rad=yaw,
                )]
            ):
                return (
                    (float(candidate[0]), float(candidate[1])),
                    np.ascontiguousarray(path[:index + 1]),
                )
        raise RuntimeError("局部路径内没有满足车体足迹的短程目标")

    def plan(
        self,
        *,
        grid: PlanningGrid,
        start_pose_world: tuple[float, float, float],
        goal_world_xy: tuple[float, float],
        goal_world_yaw_rad: float | None = None,
        constraint: FinePlanningConstraint | None = None,
        reference_path_world_xy: np.ndarray | None = None,
        recent_world_poses: np.ndarray | None = None,
        initial_speed_mps: float = 0.0,
        reaction_time_sec: float = 0.25,
        braking_deceleration_mps2: float = 0.60,
    ) -> FinePlan:
        start = np.asarray(start_pose_world, dtype=np.float64)
        goal = np.asarray(goal_world_xy, dtype=np.float64)
        if start.shape != (3,) or goal.shape != (2,) or not (
            np.isfinite(start).all() and np.isfinite(goal).all()
        ):
            raise ValueError("细粒度规划器需要有限的起始位姿和目标坐标")
        if float(np.linalg.norm(goal - start[:2])) > self.config.max_plan_distance_m:
            raise ValueError("细粒度局部目标超出规划距离")
        reference = (
            np.vstack((start[:2], goal))
            if reference_path_world_xy is None
            else np.asarray(reference_path_world_xy, dtype=np.float64)
        )
        if (
            reference.ndim != 2
            or reference.shape[1] != 2
            or len(reference) < 2
            or not np.isfinite(reference).all()
        ):
            raise ValueError("细粒度参考路线必须包含至少两个有效位置")
        recent = (
            np.empty((0, 3), dtype=np.float64)
            if recent_world_poses is None
            else np.asarray(recent_world_poses, dtype=np.float64).reshape(-1, 3)
        )
        if not np.isfinite(recent).all():
            raise ValueError("近期实际位姿必须有效")
        start_key = self._state_key(grid, start[0], start[1], start[2], -1)
        if start_key is None:
            raise RuntimeError("细粒度规划器起始位姿位于局部地图外部")

        queue: list[tuple[float, int, int]] = []
        counter = 0
        best_cost_by_key = {start_key: 0.0}
        node_pose = {0: (float(start[0]), float(start[1]), float(start[2]))}
        node_parent: dict[int, int | None] = {0: None}
        node_action: dict[int, int | None] = {0: None}
        node_cost = {0: 0.0}
        node_travel_distance = {0: 0.0}
        node_translation_distance = {0: 0.0}
        node_key = {0: start_key}
        node_max_unknown_ratio = {0: 0.0}
        node_max_unknown_area_m2 = {0: 0.0}
        heapq.heappush(
            queue,
            (
                self.config.heuristic_weight * self._heuristic(start[:2], goal),
                counter,
                0,
            ),
        )
        goal_node = None
        partial_candidates: list[tuple[tuple[float, float], int]] = []
        _, start_lateral, start_tangent = _route_projection(
            start[:2], reference, max_progress_m=grid.resolution_m
        )
        start_route_score = (
            -0.5 * start_lateral
            -0.15 * abs(_wrap_angle(start[2] - start_tangent))
        )
        minimum_stop_distance = stopping_distance_m(
            initial_speed_mps,
            reaction_time_sec=reaction_time_sec,
            braking_deceleration_mps2=braking_deceleration_mps2,
            reserve_m=grid.resolution_m,
        )
        expanded = 0
        rejected_collision = 0
        rejected_unknown = 0
        rejected_outside = 0
        rejected_constraint = 0
        observed_max_unknown_ratio = 0.0
        observed_max_unknown_area_m2 = 0.0
        rejected_unknown_mask = np.zeros(grid.shape, dtype=bool)

        while queue and expanded < self.config.max_expansions:
            if (
                partial_candidates
                and expanded >= self.config.partial_search_max_expansions
            ):
                break
            _, _, current_node = heapq.heappop(queue)
            key = node_key[current_node]
            cost = node_cost[current_node]
            if cost > best_cost_by_key[key] + 1.0e-9:
                continue
            pose = node_pose[current_node]
            goal_position_error = math.hypot(pose[0] - goal[0], pose[1] - goal[1])
            goal_yaw_error = (
                0.0
                if goal_world_yaw_rad is None
                else abs(_wrap_angle(pose[2] - float(goal_world_yaw_rad)))
            )
            if (
                goal_position_error <= self.config.goal_tolerance_m
                and goal_yaw_error <= 2.0 * self._yaw_step
            ):
                goal_node = current_node
                break
            expanded += 1
            previous_action = self._actions[key[3]] if key[3] >= 0 else None
            for action_index, action in enumerate(self._actions):
                primitive = self._sample_primitive(pose, action)
                if constraint is not None and any(
                    not constraint.allows(
                        float(sample[0]), float(sample[1]), float(sample[2])
                    )
                    for sample in primitive[1:]
                ):
                    rejected_constraint += 1
                    continue
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
                observed_max_unknown_ratio = max(
                    observed_max_unknown_ratio, unknown_ratio
                )
                observed_max_unknown_area_m2 = max(
                    observed_max_unknown_area_m2, unknown_metrics.area_m2
                )
                if unknown_metrics.cell_count > 0:
                    rejected_unknown += 1
                    rejected_unknown_mask.reshape(-1)[
                        unknown_metrics.unknown_linear_indices
                    ] = True
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
                if new_cost >= best_cost_by_key.get(next_key, float("inf")):
                    continue
                best_cost_by_key[next_key] = new_cost
                counter += 1
                node_pose[counter] = tuple(map(float, endpoint))
                node_parent[counter] = current_node
                node_action[counter] = action_index
                node_cost[counter] = new_cost
                node_travel_distance[counter] = (
                    node_travel_distance[current_node] + transition_length
                )
                node_translation_distance[counter] = (
                    node_translation_distance[current_node]
                    + (self.config.step_length_m if action.direction != 0 else 0.0)
                )
                node_key[counter] = next_key
                node_max_unknown_ratio[counter] = max(
                    node_max_unknown_ratio[current_node], unknown_ratio
                )
                node_max_unknown_area_m2[counter] = max(
                    node_max_unknown_area_m2[current_node], unknown_metrics.area_m2
                )
                route_progress, lateral_error, tangent_yaw = _route_projection(
                    endpoint[:2],
                    reference,
                    max_progress_m=(
                        node_travel_distance[counter] + grid.resolution_m
                    ),
                )
                progress = (
                    route_progress
                    - 0.5 * lateral_error
                    - 0.15 * abs(_wrap_angle(endpoint[2] - tangent_yaw))
                    - start_route_score
                )
                repeated_pose = any(
                    math.hypot(
                        endpoint[0] - old_pose[0],
                        endpoint[1] - old_pose[1],
                    ) < grid.resolution_m
                    and abs(_wrap_angle(endpoint[2] - old_pose[2]))
                    < self._yaw_step
                    for old_pose in recent
                )
                if (
                    node_travel_distance[counter]
                    <= self.config.partial_path_max_distance_m + 1.0e-9
                    and node_translation_distance[counter]
                    >= minimum_stop_distance
                    and node_max_unknown_area_m2[counter] == 0.0
                    and not repeated_pose
                    and progress >= max(
                        self.config.goal_tolerance_m,
                        0.5 * self.config.step_length_m,
                    )
                ):
                    candidate_rank = (
                        progress,
                        -new_cost,
                    )
                    partial_candidates.append((candidate_rank, counter))
                priority = new_cost + self.config.heuristic_weight * self._heuristic(
                    endpoint[:2], goal
                )
                heapq.heappush(queue, (priority, counter, counter))

        partial = False
        if goal_node is None:
            for _, candidate_node in sorted(
                partial_candidates, reverse=True
            ):
                if self._has_safe_partial_continuation(
                    grid=grid,
                    initial_pose=start,
                    endpoint_pose=node_pose[candidate_node],
                    reference_path_world_xy=reference,
                    travel_distance_m=node_travel_distance[candidate_node],
                    constraint=constraint,
                ):
                    goal_node = candidate_node
                    partial = True
                    break
        if goal_node is None:
            diagnostics = self.inspect_initial_actions(
                grid=grid,
                start_pose_world=start_pose_world,
                constraint=constraint,
            )
            message = (
                f"细粒度规划器扩展 {expanded} 个节点后仍未找到路径；"
                f"碰撞拒绝={rejected_collision}，未知拒绝={rejected_unknown}，"
                f"地图外拒绝={rejected_outside}，管道拒绝={rejected_constraint}，"
                f"未知最大比例={observed_max_unknown_ratio:.3f}，"
                f"未知最大面积={observed_max_unknown_area_m2:.3f}m2"
            )
            budget_exhausted = (
                bool(queue)
                and (
                    expanded >= self.config.max_expansions
                    or (
                        bool(partial_candidates)
                        and expanded >= self.config.partial_search_max_expansions
                    )
                )
            )
            reason = (
                "SEARCH_BUDGET_EXHAUSTED"
                if budget_exhausted
                else "NO_SAFE_CONTINUATION"
                if partial_candidates
                else "UNKNOWN_BLOCKED"
                if rejected_unknown
                else "NO_SAFE_PROGRESS"
            )
            if partial_candidates and not budget_exhausted:
                message += "；安全短路径终点没有朝参考路线继续移动的空间"
            if rejected_unknown and not budget_exhausted:
                raise FineUnknownBlockedError(
                    message,
                    np.flatnonzero(rejected_unknown_mask),
                    diagnostics,
                    reason=reason,
                )
            raise FinePlanningFailure(
                message,
                reason=reason,
                action_diagnostics=diagnostics,
            )
        node_ids = []
        current = goal_node
        while current is not None:
            node_ids.append(current)
            current = node_parent[current]
        node_ids.reverse()
        samples = [np.asarray(node_pose[0], dtype=np.float64)]
        motion_direction = [0]
        for node_id in node_ids[1:]:
            parent_id = node_parent[node_id]
            action_index = node_action[node_id]
            if parent_id is None or action_index is None:
                raise RuntimeError("细粒度路径父节点信息无效")
            primitive = self._sample_primitive(
                node_pose[parent_id], self._actions[action_index]
            )
            samples.extend(primitive[1:])
            motion_direction.extend(
                [self._actions[action_index].direction] * (len(primitive) - 1)
            )
        poses = np.asarray(samples, dtype=np.float64)
        if len(poses) < 2:
            raise RuntimeError("细粒度规划器返回了零长度路径")
        return FinePlan(
            path_world_xy=np.ascontiguousarray(poses[:, :2]),
            path_world_yaw_rad=np.ascontiguousarray(poses[:, 2]),
            path_motion_direction=np.asarray(motion_direction, dtype=np.int8),
            expanded_node_count=int(expanded),
            path_cost=float(node_cost[goal_node]),
            map_version=int(grid.map_version),
            local_goal_world_xy=(float(goal[0]), float(goal[1])),
            rejected_collision_count=int(rejected_collision),
            rejected_unknown_count=int(rejected_unknown),
            rejected_outside_count=int(rejected_outside),
            rejected_constraint_count=int(rejected_constraint),
            max_unknown_ratio=float(node_max_unknown_ratio[goal_node]),
            max_unknown_area_m2=float(node_max_unknown_area_m2[goal_node]),
            partial=bool(partial),
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
    def _heuristic(position_xy: np.ndarray, goal_xy: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(position_xy) - np.asarray(goal_xy)))


def _wrap_angle(angle_rad: float) -> float:
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def _route_projection(
    position_xy: np.ndarray,
    reference_path_xy: np.ndarray,
    *,
    max_progress_m: float,
) -> tuple[float, float, float]:
    point = np.asarray(position_xy, dtype=np.float64).reshape(2)
    path = np.asarray(reference_path_xy, dtype=np.float64)
    cumulative = 0.0
    best = None
    for start, end in zip(path[:-1], path[1:]):
        segment = end - start
        length = float(np.linalg.norm(segment))
        if length < 1.0e-9:
            continue
        allowed = min(1.0, max(0.0, (max_progress_m - cumulative) / length))
        fraction = float(np.clip(
            np.dot(point - start, segment) / (length * length),
            0.0,
            allowed,
        ))
        projected = start + fraction * segment
        lateral = float(np.linalg.norm(point - projected))
        progress = cumulative + fraction * length
        tangent = math.atan2(float(segment[1]), float(segment[0]))
        candidate = (lateral, progress, tangent)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
        cumulative += length
        if cumulative >= max_progress_m:
            break
    if best is None:
        raise ValueError("参考路线没有有效的运动段")
    return best[1], best[0], best[2]
