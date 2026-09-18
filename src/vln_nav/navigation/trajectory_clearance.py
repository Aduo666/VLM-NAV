from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import distance_transform_edt

from vln_nav.active_perception.local_geometry import FREE, OCCUPIED, LocalVoxelSnapshot


@dataclass(frozen=True)
class FootprintPartConfig:
    name: str
    min_x_m: float
    max_x_m: float
    min_y_m: float
    max_y_m: float

    @classmethod
    def from_mapping(cls, data: dict) -> "FootprintPartConfig":
        expected = {"name", "min_x_m", "max_x_m", "min_y_m", "max_y_m"}
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError(f"车体部件必须包含这些字段：{sorted(expected)}")
        part = cls(
            name=str(data["name"]),
            min_x_m=float(data["min_x_m"]),
            max_x_m=float(data["max_x_m"]),
            min_y_m=float(data["min_y_m"]),
            max_y_m=float(data["max_y_m"]),
        )
        part.validate()
        return part

    def validate(self) -> None:
        values = (self.min_x_m, self.max_x_m, self.min_y_m, self.max_y_m)
        if not self.name or not all(math.isfinite(value) for value in values):
            raise ValueError("车体部件名称和尺寸必须有效")
        if self.min_x_m >= self.max_x_m or self.min_y_m >= self.max_y_m:
            raise ValueError(f"车体部件 {self.name} 的尺寸范围无效")


@dataclass(frozen=True)
class FootprintConfig:
    parts: tuple[FootprintPartConfig, ...]
    margin_m: float

    @classmethod
    def from_mapping(cls, data: dict) -> "FootprintConfig":
        if not isinstance(data, dict) or set(data) != {"parts", "margin_m"}:
            raise ValueError("footprint 必须包含 parts 和 margin_m")
        if not isinstance(data["parts"], list):
            raise ValueError("footprint.parts 必须是列表")
        config = cls(
            parts=tuple(FootprintPartConfig.from_mapping(row) for row in data["parts"]),
            margin_m=float(data["margin_m"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.parts:
            raise ValueError("车体轮廓至少需要一个部件")
        if len({part.name for part in self.parts}) != len(self.parts):
            raise ValueError("车体部件名称不能重复")
        for part in self.parts:
            part.validate()
        if not math.isfinite(float(self.margin_m)) or float(self.margin_m) < 0:
            raise ValueError("车体轮廓余量必须为有限非负数")

    @property
    def front_m(self) -> float:
        return max(part.max_x_m for part in self.parts)

    @property
    def rear_m(self) -> float:
        return -min(part.min_x_m for part in self.parts)

    @property
    def half_width_m(self) -> float:
        return max(
            max(abs(part.min_y_m), abs(part.max_y_m)) for part in self.parts
        )


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
class IncrementalUnknownMetrics:
    ratio: float
    cell_count: int
    area_m2: float


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

class FootprintCollisionChecker:
    def __init__(
        self,
        footprint: FootprintConfig,
        *,
        yaw_resolution_deg: float,
    ) -> None:
        footprint.validate()
        if not math.isfinite(yaw_resolution_deg) or not 1.0 <= yaw_resolution_deg <= 30.0:
            raise ValueError("足迹航向分辨率必须位于 1 至 30 度范围内")
        self.footprint = footprint
        self._yaw_step = math.radians(float(yaw_resolution_deg))
        self._yaw_bins = int(round(2.0 * math.pi / self._yaw_step))
        self._offset_cache: dict[tuple[float, int, float], np.ndarray] = {}

    @property
    def bounding_radius_m(self) -> float:
        return math.hypot(
            max(self.footprint.front_m, self.footprint.rear_m),
            self.footprint.half_width_m,
        ) + self.footprint.margin_m

    def metrics(
        self,
        grid: PlanningGrid,
        *,
        x_m: float,
        y_m: float,
        yaw_rad: float,
    ) -> FootprintMetrics:
        rows, columns, inside = self._indices(
            grid,
            x_m=float(x_m),
            y_m=float(y_m),
            yaw_rad=float(yaw_rad),
            margin_m=self.footprint.margin_m,
        )
        outside_ratio = float(1.0 - np.mean(inside))
        if not np.any(inside):
            return FootprintMetrics(False, 0.0, 1.0, 1.0)
        valid_rows = rows[inside]
        valid_columns = columns[inside]
        occupied = grid.occupied_mask[valid_rows, valid_columns]
        unknown_count = int(np.count_nonzero(grid.unknown_mask[valid_rows, valid_columns]))
        unknown_count += int(np.count_nonzero(~inside))
        clearances = grid.clearance_m[valid_rows, valid_columns]
        return FootprintMetrics(
            collision=bool(np.any(occupied)),
            min_clearance_m=float(np.min(clearances)) if clearances.size else 0.0,
            unknown_ratio=float(unknown_count / max(len(inside), 1)),
            outside_ratio=outside_ratio,
        )

    def clear_pose(
        self,
        grid: PlanningGrid,
        *,
        x_m: float,
        y_m: float,
        yaw_rad: float,
    ) -> PlanningGrid:
        rows, columns, inside = self._indices(
            grid,
            x_m=float(x_m),
            y_m=float(y_m),
            yaw_rad=float(yaw_rad),
            margin_m=0.0,
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

    def covered_linear_indices(
        self,
        grid: PlanningGrid,
        *,
        x_m: float,
        y_m: float,
        yaw_rad: float,
    ) -> tuple[np.ndarray, float]:
        rows, columns, inside = self._indices(
            grid,
            x_m=float(x_m),
            y_m=float(y_m),
            yaw_rad=float(yaw_rad),
            margin_m=self.footprint.margin_m,
        )
        outside_ratio = float(1.0 - np.mean(inside))
        if not np.any(inside):
            return np.empty((0,), dtype=np.int64), outside_ratio
        linear = rows[inside] * grid.shape[1] + columns[inside]
        return np.unique(linear), outside_ratio

    def incremental_unknown_metrics(
        self,
        grid: PlanningGrid,
        *,
        initial_pose: np.ndarray,
        swept_poses: np.ndarray,
    ) -> IncrementalUnknownMetrics:
        initial = np.asarray(initial_pose, dtype=np.float64).reshape(3)
        sweep = np.asarray(swept_poses, dtype=np.float64).reshape(-1, 3)
        initial_cells, _ = self.covered_linear_indices(
            grid,
            x_m=float(initial[0]),
            y_m=float(initial[1]),
            yaw_rad=float(initial[2]),
        )
        if len(sweep) == 0:
            return IncrementalUnknownMetrics(0.0, 0, 0.0)
        swept_cells = []
        for pose in sweep:
            cells, _ = self.covered_linear_indices(
                grid,
                x_m=float(pose[0]),
                y_m=float(pose[1]),
                yaw_rad=float(pose[2]),
            )
            if cells.size:
                swept_cells.append(cells)
        if not swept_cells:
            return IncrementalUnknownMetrics(0.0, 0, 0.0)
        all_swept = np.unique(np.concatenate(swept_cells))
        newly_covered = np.setdiff1d(all_swept, initial_cells, assume_unique=True)
        if newly_covered.size == 0:
            return IncrementalUnknownMetrics(0.0, 0, 0.0)
        unknown = grid.unknown_mask.reshape(-1)[newly_covered]
        unknown_count = int(np.count_nonzero(unknown))
        return IncrementalUnknownMetrics(
            ratio=float(unknown_count / newly_covered.size),
            cell_count=unknown_count,
            area_m2=float(unknown_count * grid.resolution_m**2),
        )

    def _indices(
        self,
        grid: PlanningGrid,
        *,
        x_m: float,
        y_m: float,
        yaw_rad: float,
        margin_m: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        row, column, _ = grid.world_to_cell(x_m, y_m)
        offsets = self._offsets(grid.resolution_m, yaw_rad, margin_m)
        rows = row + offsets[:, 0]
        columns = column + offsets[:, 1]
        inside = (
            (rows >= 0)
            & (rows < grid.shape[0])
            & (columns >= 0)
            & (columns < grid.shape[1])
        )
        return rows, columns, inside

    def _offsets(
        self,
        resolution_m: float,
        yaw_rad: float,
        margin_m: float,
    ) -> np.ndarray:
        yaw_index = int(
            round(((float(yaw_rad) + math.pi) % (2.0 * math.pi)) / self._yaw_step)
        ) % self._yaw_bins
        key = (float(resolution_m), yaw_index, float(margin_m))
        cached = self._offset_cache.get(key)
        if cached is not None:
            return cached
        sample_step = max(0.5 * float(resolution_m), 0.01)
        local_chunks = []
        margin = float(margin_m)
        for part in self.footprint.parts:
            local_x = np.arange(
                part.min_x_m - margin,
                part.max_x_m + margin + 0.5 * sample_step,
                sample_step,
            )
            local_y = np.arange(
                part.min_y_m - margin,
                part.max_y_m + margin + 0.5 * sample_step,
                sample_step,
            )
            xx, yy = np.meshgrid(local_x, local_y, indexing="xy")
            local_chunks.append(np.column_stack((xx.ravel(), yy.ravel())))
        local = np.concatenate(local_chunks, axis=0)
        yaw = yaw_index * self._yaw_step - math.pi
        cosine, sine = math.cos(yaw), math.sin(yaw)
        column_offset = np.rint(
            (cosine * local[:, 0] - sine * local[:, 1]) / resolution_m
        ).astype(np.int64)
        row_offset = np.rint(
            (sine * local[:, 0] + cosine * local[:, 1]) / resolution_m
        ).astype(np.int64)
        offsets = np.unique(np.column_stack((row_offset, column_offset)), axis=0)
        self._offset_cache[key] = offsets
        return offsets


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
    collision_checker: FootprintCollisionChecker,
) -> PlanningGrid:
    return collision_checker.clear_pose(
        grid,
        x_m=float(x_m),
        y_m=float(y_m),
        yaw_rad=float(yaw_rad),
    )


def assess_path_clearance(
    grid: PlanningGrid,
    path_xy: np.ndarray,
    path_yaw_rad: np.ndarray,
    *,
    collision_checker: FootprintCollisionChecker,
) -> PathClearance:
    points = np.asarray(path_xy, dtype=np.float64)
    yaw = np.asarray(path_yaw_rad, dtype=np.float64).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("轨迹余量检查至少需要两个 XY 点")
    if len(yaw) != len(points):
        raise ValueError("轨迹位置与航向数量必须一致")
    if not np.isfinite(points).all() or not np.isfinite(yaw).all():
        raise ValueError("轨迹余量检查收到非有限坐标")
    collision = False
    min_clearance = float("inf")
    unknown_sum = 0.0
    outside_sum = 0.0
    for point, heading in zip(points, yaw):
        metrics = collision_checker.metrics(
            grid,
            x_m=float(point[0]),
            y_m=float(point[1]),
            yaw_rad=float(heading),
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
