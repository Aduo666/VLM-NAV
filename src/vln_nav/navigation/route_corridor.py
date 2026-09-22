from __future__ import annotations

from dataclasses import dataclass, fields
import math

import numpy as np

from vln_nav.navigation.trajectory_clearance import PlanningGrid


@dataclass(frozen=True)
class RouteCorridorConfig:
    horizon_m: float
    cross_section_step_m: float
    half_width_m: float
    center_search_step_m: float
    center_deviation_weight: float
    smoothing_iterations: int

    @classmethod
    def from_mapping(cls, data: dict) -> "RouteCorridorConfig":
        expected = {item.name for item in fields(cls)}
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError(f"route_corridor 必须包含这些字段：{sorted(expected)}")
        config = cls(
            horizon_m=float(data["horizon_m"]),
            cross_section_step_m=float(data["cross_section_step_m"]),
            half_width_m=float(data["half_width_m"]),
            center_search_step_m=float(data["center_search_step_m"]),
            center_deviation_weight=float(data["center_deviation_weight"]),
            smoothing_iterations=int(data["smoothing_iterations"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        positive = (
            self.horizon_m,
            self.cross_section_step_m,
            self.half_width_m,
            self.center_search_step_m,
            self.center_deviation_weight,
        )
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise ValueError("route_corridor 数值配置必须为有限正数")
        if self.cross_section_step_m > self.horizon_m:
            raise ValueError("cross_section_step_m 不能超过 horizon_m")
        if self.center_search_step_m > self.half_width_m:
            raise ValueError("center_search_step_m 不能超过 half_width_m")
        if self.smoothing_iterations < 0 or self.smoothing_iterations > 20:
            raise ValueError("smoothing_iterations 必须位于 0 至 20")


@dataclass(frozen=True)
class RouteCorridor:
    s_m: np.ndarray
    center_world_xy: np.ndarray
    tangent_world_xy: np.ndarray
    normal_world_xy: np.ndarray
    yaw_world_rad: np.ndarray
    half_width_m: float

    @property
    def layer_count(self) -> int:
        return int(len(self.s_m))

    def world_pose(
        self,
        layer_index: int,
        lateral_m: float,
        heading_delta_rad: float,
    ) -> tuple[float, float, float]:
        center = self.center_world_xy[int(layer_index)]
        normal = self.normal_world_xy[int(layer_index)]
        position = center + float(lateral_m) * normal
        yaw = _wrap_angle(
            float(self.yaw_world_rad[int(layer_index)]) + float(heading_delta_rad)
        )
        return float(position[0]), float(position[1]), yaw

    def nearest_layer(self, world_xy: np.ndarray) -> int:
        point = np.asarray(world_xy, dtype=np.float64).reshape(2)
        return int(np.argmin(np.linalg.norm(self.center_world_xy - point, axis=1)))

    def coordinates(self, world_xy: np.ndarray, yaw_world_rad: float) -> tuple[int, float, float]:
        layer = self.nearest_layer(world_xy)
        delta = np.asarray(world_xy, dtype=np.float64).reshape(2) - self.center_world_xy[layer]
        lateral = float(np.dot(delta, self.normal_world_xy[layer]))
        heading = _wrap_angle(float(yaw_world_rad) - float(self.yaw_world_rad[layer]))
        return layer, lateral, heading


def build_route_corridor(
    *,
    grid: PlanningGrid,
    start_pose_world: tuple[float, float, float],
    target_world_xy: tuple[float, float],
    route_hint_world_xy: np.ndarray | None,
    config: RouteCorridorConfig,
) -> RouteCorridor:
    config.validate()
    start = np.asarray(start_pose_world[:2], dtype=np.float64)
    target = np.asarray(target_world_xy, dtype=np.float64).reshape(2)
    if not np.isfinite(start).all() or not np.isfinite(target).all():
        raise ValueError("route corridor 需要有效的起点和目标")
    seed = _prepare_seed(start, target, route_hint_world_xy, config.horizon_m)
    centers = _resample_polyline(seed, config.cross_section_step_m, config.horizon_m)
    if len(centers) < 2:
        raise RuntimeError("局部通路长度不足，无法建立 corridor")
    centers = _move_toward_clearance_ridge(centers, grid, config)
    centers = _smooth_polyline(centers, config.smoothing_iterations)
    centers[0] = start
    centers = _resample_polyline(centers, config.cross_section_step_m, config.horizon_m)
    if len(centers) < 2:
        raise RuntimeError("平滑后的局部通路长度不足")
    delta = np.gradient(centers, axis=0)
    norm = np.linalg.norm(delta, axis=1)
    if np.any(norm < 1.0e-6):
        raise RuntimeError("局部通路包含重复截面")
    tangent = delta / norm[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    yaw = np.arctan2(tangent[:, 1], tangent[:, 0])
    segment = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    s_m = np.r_[0.0, np.cumsum(segment)]
    return RouteCorridor(
        s_m=np.ascontiguousarray(s_m),
        center_world_xy=np.ascontiguousarray(centers),
        tangent_world_xy=np.ascontiguousarray(tangent),
        normal_world_xy=np.ascontiguousarray(normal),
        yaw_world_rad=np.ascontiguousarray(yaw),
        half_width_m=float(config.half_width_m),
    )


def _prepare_seed(
    start: np.ndarray,
    target: np.ndarray,
    route_hint_world_xy: np.ndarray | None,
    horizon_m: float,
) -> np.ndarray:
    if route_hint_world_xy is None:
        hint = np.empty((0, 2), dtype=np.float64)
    else:
        hint = np.asarray(route_hint_world_xy, dtype=np.float64)
        if hint.ndim != 2 or hint.shape[1] != 2 or not np.isfinite(hint).all():
            raise ValueError("route hint 必须是有限的 N×2 坐标")
    if len(hint):
        nearest = int(np.argmin(np.linalg.norm(hint - start, axis=1)))
        hint = hint[nearest:]
        points = np.vstack((start, hint))
    else:
        points = start[None, :]
    points = _remove_duplicate_points(points)
    current_length = _polyline_length(points)
    if current_length < horizon_m and np.linalg.norm(target - points[-1]) > 1.0e-4:
        points = np.vstack((points, target))
    if len(points) < 2:
        yaw_target = start + np.asarray([horizon_m, 0.0], dtype=np.float64)
        points = np.vstack((start, yaw_target))
    return points


def _move_toward_clearance_ridge(
    centers: np.ndarray,
    grid: PlanningGrid,
    config: RouteCorridorConfig,
) -> np.ndarray:
    adjusted = centers.copy()
    if len(centers) < 3:
        return adjusted
    delta = np.gradient(centers, axis=0)
    norms = np.maximum(np.linalg.norm(delta, axis=1), 1.0e-9)
    normals = np.column_stack((-delta[:, 1] / norms, delta[:, 0] / norms))
    offsets = np.arange(
        -config.half_width_m,
        config.half_width_m + 0.5 * config.center_search_step_m,
        config.center_search_step_m,
    )
    for index in range(1, len(centers)):
        best = centers[index]
        best_score = -float("inf")
        for offset in offsets:
            candidate = centers[index] + offset * normals[index]
            row, column, inside = grid.world_to_cell(float(candidate[0]), float(candidate[1]))
            if not inside or not grid.free_mask[row, column] or grid.occupied_mask[row, column]:
                continue
            score = float(grid.clearance_m[row, column]) - (
                config.center_deviation_weight * abs(float(offset))
            )
            if score > best_score:
                best_score = score
                best = candidate
        adjusted[index] = best
    return adjusted


def _smooth_polyline(points: np.ndarray, iterations: int) -> np.ndarray:
    smoothed = np.asarray(points, dtype=np.float64).copy()
    for _ in range(int(iterations)):
        if len(smoothed) < 3:
            break
        updated = smoothed.copy()
        updated[1:-1] = (
            0.25 * smoothed[:-2]
            + 0.50 * smoothed[1:-1]
            + 0.25 * smoothed[2:]
        )
        smoothed = updated
    return smoothed


def _resample_polyline(points: np.ndarray, step_m: float, horizon_m: float) -> np.ndarray:
    points = _remove_duplicate_points(np.asarray(points, dtype=np.float64))
    if len(points) < 2:
        return points
    lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    total = min(float(lengths[-1]), float(horizon_m))
    if total < 1.0e-5:
        return points[:1]
    samples = np.arange(0.0, total + 0.5 * step_m, step_m)
    samples[-1] = min(samples[-1], total)
    samples = np.unique(np.r_[samples, total])
    x = np.interp(samples, lengths, points[:, 0])
    y = np.interp(samples, lengths, points[:, 1])
    return np.column_stack((x, y))


def _remove_duplicate_points(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.asarray(points, dtype=np.float64)
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1.0e-4]
    return np.asarray(points, dtype=np.float64)[keep]


def _polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _wrap_angle(angle_rad: float) -> float:
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi
