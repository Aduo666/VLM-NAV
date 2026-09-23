from __future__ import annotations

from dataclasses import dataclass, fields
import math

import numpy as np

from vln_nav.active_perception.local_geometry import (
    FREE,
    OCCUPIED,
    LocalVoxelMapConfig,
    LocalVoxelSnapshot,
)


@dataclass(frozen=True)
class BevConfig:
    forward_m: float = 3.0
    backward_m: float = 1.0
    left_m: float = 2.0
    right_m: float = 2.0
    resolution_m: float = 0.05
    obstacle_min_height_m: float = 0.05
    obstacle_max_height_m: float = 1.50

    @classmethod
    def from_mapping(cls, data: dict) -> "BevConfig":
        expected = {item.name for item in fields(cls)}
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError(f"BEV config requires exactly: {sorted(expected)}")
        config = cls(**{name: float(data[name]) for name in expected})
        config.validate()
        return config

    def validate(self) -> None:
        values = (
            self.forward_m,
            self.backward_m,
            self.left_m,
            self.right_m,
            self.resolution_m,
            self.obstacle_max_height_m - self.obstacle_min_height_m,
        )
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("BEV dimensions and resolution must be finite positive values")

    @property
    def height(self) -> int:
        return int(round((self.forward_m + self.backward_m) / self.resolution_m))

    @property
    def width(self) -> int:
        return int(round((self.left_m + self.right_m) / self.resolution_m))


def local_voxel_config_from_mapping(data: dict) -> LocalVoxelMapConfig:
    expected = {item.name for item in fields(LocalVoxelMapConfig)}
    if not isinstance(data, dict) or set(data) != expected:
        raise ValueError(f"voxel_map requires exactly: {sorted(expected)}")
    kwargs = {}
    for item in fields(LocalVoxelMapConfig):
        value = data[item.name]
        kwargs[item.name] = int(value) if item.name == "pixel_stride" else float(value)
    config = LocalVoxelMapConfig(**kwargs)
    config.validate()
    return config


def snapshot_to_robot_bev(
    snapshot: LocalVoxelSnapshot,
    *,
    robot_world_xyyaw: np.ndarray,
    floor_z_m: float,
    config: BevConfig,
) -> np.ndarray:
    """Return uint8 [3,H,W] = occupied, free, unknown in robot coordinates."""
    config.validate()
    pose = np.asarray(robot_world_xyyaw, dtype=np.float64).reshape(3)
    state = np.asarray(snapshot.state)
    z_centers = (
        snapshot.origin_world_xyz[2]
        + (np.arange(state.shape[0], dtype=np.float64) + 0.5) * snapshot.resolution_m
    )
    z_mask = (
        (z_centers >= float(floor_z_m) + config.obstacle_min_height_m)
        & (z_centers <= float(floor_z_m) + config.obstacle_max_height_m)
    )
    if not np.any(z_mask):
        return _unknown_bev(config)

    vertical = state[z_mask]
    occupied_xy = np.any(vertical == OCCUPIED, axis=0)
    free_xy = np.any(vertical == FREE, axis=0) & ~occupied_xy

    h, w = config.height, config.width
    rows = np.arange(h, dtype=np.float64)
    cols = np.arange(w, dtype=np.float64)
    forward = config.forward_m - (rows + 0.5) * config.resolution_m
    left = config.left_m - (cols + 0.5) * config.resolution_m
    body_x, body_y = np.meshgrid(forward, left, indexing="ij")

    c = math.cos(float(pose[2]))
    s = math.sin(float(pose[2]))
    world_x = pose[0] + c * body_x - s * body_y
    world_y = pose[1] + s * body_x + c * body_y

    grid_x = np.floor(
        (world_x - snapshot.origin_world_xyz[0]) / snapshot.resolution_m
    ).astype(np.int64)
    grid_y = np.floor(
        (world_y - snapshot.origin_world_xyz[1]) / snapshot.resolution_m
    ).astype(np.int64)
    inside = (
        (grid_x >= 0)
        & (grid_x < state.shape[2])
        & (grid_y >= 0)
        & (grid_y < state.shape[1])
    )

    occupied = np.zeros((h, w), dtype=np.uint8)
    free = np.zeros((h, w), dtype=np.uint8)
    occupied[inside] = occupied_xy[grid_y[inside], grid_x[inside]].astype(np.uint8) * 255
    free_mask = np.zeros((h, w), dtype=bool)
    free_mask[inside] = free_xy[grid_y[inside], grid_x[inside]]
    free[free_mask & (occupied == 0)] = 255
    unknown = np.where((occupied == 0) & (free == 0), 255, 0).astype(np.uint8)
    return np.ascontiguousarray(np.stack((occupied, free, unknown), axis=0))


def body_xy_to_bev_pixel(body_xy: np.ndarray, config: BevConfig) -> tuple[int, int] | None:
    point = np.asarray(body_xy, dtype=np.float64).reshape(2)
    row = int(math.floor((config.forward_m - float(point[0])) / config.resolution_m))
    col = int(math.floor((config.left_m - float(point[1])) / config.resolution_m))
    if not (0 <= row < config.height and 0 <= col < config.width):
        return None
    return col, row


def _unknown_bev(config: BevConfig) -> np.ndarray:
    result = np.zeros((3, config.height, config.width), dtype=np.uint8)
    result[2] = 255
    return result
