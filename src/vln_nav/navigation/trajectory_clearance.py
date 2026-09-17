from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import distance_transform_edt

from vln_nav.active_perception.local_geometry import FREE, OCCUPIED, LocalVoxelSnapshot


@dataclass(frozen=True)
class FootprintConfig:
    front_m: float
    rear_m: float
    half_width_m: float
    margin_m: float

    def validate(self) -> None:
        if not all(
            math.isfinite(float(value)) and float(value) > 0
            for value in (self.front_m, self.rear_m, self.half_width_m)
        ):
            raise ValueError("车体轮廓尺寸必须为有限正数")
        if not math.isfinite(float(self.margin_m)) or float(self.margin_m) < 0:
            raise ValueError("车体轮廓余量必须为有限非负数")


@dataclass(frozen=True)
class FootprintMetrics:
    collision: bool
    min_clearance_m: float
    unknown_ratio: float
    outside_ratio: float


@dataclass(frozen=True)
class PathClearance:
    collision: bool
    min_clearance_m: float
    unknown_ratio: float
    outside_ratio: float
    checked_pose_count: int


@dataclass(frozen=True)
class PlanningGrid:
    occupied_mask: np.ndarray
    free_mask: np.ndarray
    unknown_mask: np.ndarray
    clearance_m: np.ndarray
    origin_x_m: float
    origin_y_m: float
    resolution_m: float
    map_version: int

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(map(int, self.occupied_mask.shape))

    def world_to_cell(self, x_m: float, y_m: float) -> tuple[int, int, bool]:
        column = int(math.floor((float(x_m) - self.origin_x_m) / self.resolution_m))
        row = int(math.floor((float(y_m) - self.origin_y_m) / self.resolution_m))
        inside = 0 <= row < self.shape[0] and 0 <= column < self.shape[1]
        return row, column, inside

    def footprint_metrics(
        self,
        *,
        x_m: float,
        y_m: float,
        yaw_rad: float,
        footprint: FootprintConfig,
    ) -> FootprintMetrics:
        footprint.validate()
        sample_step = max(0.5 * self.resolution_m, 0.01)
        local_x = np.arange(
            -footprint.rear_m - footprint.margin_m,
            footprint.front_m + footprint.margin_m + 0.5 * sample_step,
            sample_step,
        )
        local_y = np.arange(
            -footprint.half_width_m - footprint.margin_m,
            footprint.half_width_m + footprint.margin_m + 0.5 * sample_step,
            sample_step,
        )
        xx, yy = np.meshgrid(local_x, local_y, indexing="xy")
        cosine, sine = math.cos(float(yaw_rad)), math.sin(float(yaw_rad))
        world_x = float(x_m) + cosine * xx - sine * yy
        world_y = float(y_m) + sine * xx + cosine * yy
        columns = np.floor((world_x - self.origin_x_m) / self.resolution_m).astype(np.int64)
        rows = np.floor((world_y - self.origin_y_m) / self.resolution_m).astype(np.int64)
        inside = (
            (rows >= 0)
            & (rows < self.shape[0])
            & (columns >= 0)
            & (columns < self.shape[1])
        )
        outside_ratio = float(1.0 - np.mean(inside))
        if not np.any(inside):
            return FootprintMetrics(False, 0.0, 1.0, 1.0)
        valid_rows = rows[inside]
        valid_columns = columns[inside]
        occupied = self.occupied_mask[valid_rows, valid_columns]
        unknown_count = int(np.count_nonzero(self.unknown_mask[valid_rows, valid_columns]))
        unknown_count += int(np.count_nonzero(~inside))
        total_count = int(inside.size)
        clearances = self.clearance_m[valid_rows, valid_columns]
        min_clearance = float(np.min(clearances)) if clearances.size else 0.0
        return FootprintMetrics(
            collision=bool(np.any(occupied)),
            min_clearance_m=min_clearance,
            unknown_ratio=float(unknown_count / max(total_count, 1)),
            outside_ratio=outside_ratio,
        )


def planning_grid_from_snapshot(
    snapshot: LocalVoxelSnapshot,
    *,
    floor_z_m: float,
    obstacle_min_height_m: float,
    obstacle_max_height_m: float,
) -> PlanningGrid:
    if not 0 <= float(obstacle_min_height_m) < float(obstacle_max_height_m):
        raise ValueError("障碍物高度范围必须有序且非负")
    state = np.asarray(snapshot.state, dtype=np.int8)
    z_centers = snapshot.origin_world_xyz[2] + (
        np.arange(state.shape[0], dtype=np.float64) + 0.5
    ) * snapshot.resolution_m
    z_band = (
        (z_centers >= float(floor_z_m) + float(obstacle_min_height_m))
        & (z_centers <= float(floor_z_m) + float(obstacle_max_height_m))
    )
    if not np.any(z_band):
        raise ValueError("障碍物高度范围没有覆盖局部体素地图")
    band = state[z_band]
    occupied = np.any(band == OCCUPIED, axis=0)
    free = np.any(band == FREE, axis=0) & ~occupied
    unknown = ~(occupied | free)
    clearance = distance_transform_edt(~occupied).astype(np.float32)
    clearance *= float(snapshot.resolution_m)
    return PlanningGrid(
        occupied_mask=np.ascontiguousarray(occupied),
        free_mask=np.ascontiguousarray(free),
        unknown_mask=np.ascontiguousarray(unknown),
        clearance_m=np.ascontiguousarray(clearance),
        origin_x_m=float(snapshot.origin_world_xyz[0]),
        origin_y_m=float(snapshot.origin_world_xyz[1]),
        resolution_m=float(snapshot.resolution_m),
        map_version=int(snapshot.version),
    )


def clear_current_footprint(
    grid: PlanningGrid,
    *,
    x_m: float,
    y_m: float,
    yaw_rad: float,
    footprint: FootprintConfig,
) -> PlanningGrid:
    sample_step = max(0.5 * grid.resolution_m, 0.01)
    local_x = np.arange(-footprint.rear_m, footprint.front_m + 0.5 * sample_step, sample_step)
    local_y = np.arange(
        -footprint.half_width_m,
        footprint.half_width_m + 0.5 * sample_step,
        sample_step,
    )
    xx, yy = np.meshgrid(local_x, local_y, indexing="xy")
    cosine, sine = math.cos(float(yaw_rad)), math.sin(float(yaw_rad))
    world_x = float(x_m) + cosine * xx - sine * yy
    world_y = float(y_m) + sine * xx + cosine * yy
    columns = np.floor((world_x - grid.origin_x_m) / grid.resolution_m).astype(np.int64)
    rows = np.floor((world_y - grid.origin_y_m) / grid.resolution_m).astype(np.int64)
    inside = (
        (rows >= 0)
        & (rows < grid.shape[0])
        & (columns >= 0)
        & (columns < grid.shape[1])
    )
    occupied = grid.occupied_mask.copy()
    free = grid.free_mask.copy()
    unknown = grid.unknown_mask.copy()
    valid_rows = rows[inside]
    valid_columns = columns[inside]
    occupied[valid_rows, valid_columns] = False
    free[valid_rows, valid_columns] = True
    unknown[valid_rows, valid_columns] = False
    clearance = distance_transform_edt(~occupied).astype(np.float32)
    clearance *= grid.resolution_m
    return PlanningGrid(
        occupied_mask=np.ascontiguousarray(occupied),
        free_mask=np.ascontiguousarray(free),
        unknown_mask=np.ascontiguousarray(unknown),
        clearance_m=np.ascontiguousarray(clearance),
        origin_x_m=grid.origin_x_m,
        origin_y_m=grid.origin_y_m,
        resolution_m=grid.resolution_m,
        map_version=grid.map_version,
    )


def assess_path_clearance(
    grid: PlanningGrid,
    path_xy: np.ndarray,
    *,
    footprint: FootprintConfig,
) -> PathClearance:
    points = np.asarray(path_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("轨迹余量检查至少需要两个 XY 点")
    if not np.isfinite(points).all():
        raise ValueError("轨迹余量检查收到非有限坐标")
    delta = np.gradient(points, axis=0)
    yaw = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    collision = False
    min_clearance = float("inf")
    unknown_sum = 0.0
    outside_sum = 0.0
    for point, heading in zip(points, yaw):
        metrics = grid.footprint_metrics(
            x_m=float(point[0]),
            y_m=float(point[1]),
            yaw_rad=float(heading),
            footprint=footprint,
        )
        collision = collision or metrics.collision
        min_clearance = min(min_clearance, metrics.min_clearance_m)
        unknown_sum += metrics.unknown_ratio
        outside_sum += metrics.outside_ratio
    return PathClearance(
        collision=collision,
        min_clearance_m=0.0 if not math.isfinite(min_clearance) else min_clearance,
        unknown_ratio=float(unknown_sum / len(points)),
        outside_ratio=float(outside_sum / len(points)),
        checked_pose_count=int(len(points)),
    )
