"""Minimal robot-centric BEV for selecting a local navigation goal."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw

from vln_nav.active_perception.local_geometry import (
    FREE,
    OCCUPIED,
    LocalVoxelSnapshot,
)


@dataclass(frozen=True)
class BevGoalMap:
    rgba: np.ndarray
    selectable_mask: np.ndarray
    view_size_m: float
    rear_view_m: float

    def pixel_to_local(self, column: int, row: int) -> tuple[float, float]:
        height, width = self.selectable_mask.shape
        if not (0 <= column < width and 0 <= row < height):
            raise ValueError("BEV pixel is outside the map")
        forward = (
            1.0 - (float(row) + 0.5) / height
        ) * self.view_size_m - self.rear_view_m
        left = (0.5 - (float(column) + 0.5) / width) * self.view_size_m
        return forward, left


def render_bev_goal_map(
    snapshot: LocalVoxelSnapshot,
    *,
    robot_world_xyz: tuple[float, float, float] | np.ndarray,
    robot_yaw_rad: float,
    view_size_m: float,
    rear_view_m: float,
    image_size_px: int,
    obstacle_min_relative_z_m: float,
    obstacle_max_relative_z_m: float,
    goal_local_xy: tuple[float, float] | None = None,
) -> BevGoalMap:
    """Project a 3-D voxel snapshot into a clean robot-centric occupancy image."""
    robot = np.asarray(robot_world_xyz, dtype=np.float64).reshape(-1)
    if robot.size != 3 or not np.isfinite(robot).all():
        raise ValueError("robot_world_xyz must contain finite XYZ")
    if not math.isfinite(robot_yaw_rad):
        raise ValueError("robot_yaw_rad must be finite")
    if not math.isfinite(view_size_m) or view_size_m <= 0 or image_size_px < 32:
        raise ValueError("BEV dimensions must be positive")
    if not math.isfinite(rear_view_m) or not 0 <= rear_view_m < view_size_m:
        raise ValueError("BEV rear view must be within [0, view_size)")
    if obstacle_min_relative_z_m >= obstacle_max_relative_z_m:
        raise ValueError("BEV obstacle height interval is invalid")

    state = np.asarray(snapshot.state, dtype=np.int8)
    z_centers = snapshot.origin_world_xyz[2] + (
        np.arange(state.shape[0], dtype=np.float64) + 0.5
    ) * snapshot.resolution_m
    z_band = (
        (z_centers >= robot[2] + obstacle_min_relative_z_m)
        & (z_centers <= robot[2] + obstacle_max_relative_z_m)
    )
    if not np.any(z_band):
        raise ValueError("BEV obstacle height interval does not overlap the voxel map")

    band = state[z_band]
    occupied_world = np.any(band == OCCUPIED, axis=0)
    free_world = np.any(band == FREE, axis=0) & ~occupied_world

    size = int(image_size_px)
    rows, columns = np.indices((size, size), dtype=np.float64)
    forward = (1.0 - (rows + 0.5) / size) * view_size_m - rear_view_m
    left = (0.5 - (columns + 0.5) / size) * view_size_m
    cosine, sine = math.cos(robot_yaw_rad), math.sin(robot_yaw_rad)
    world_x = robot[0] + cosine * forward - sine * left
    world_y = robot[1] + sine * forward + cosine * left
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
    safe_x = np.clip(grid_x, 0, state.shape[2] - 1)
    safe_y = np.clip(grid_y, 0, state.shape[1] - 1)
    occupied = inside & occupied_world[safe_y, safe_x]
    free = inside & free_world[safe_y, safe_x] & ~occupied

    rgb = np.full((size, size, 3), (108, 110, 112), dtype=np.uint8)
    rgb[free] = (238, 237, 232)
    rgb[occupied] = (28, 29, 29)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    center_x = size / 2.0
    center_y = (1.0 - rear_view_m / view_size_m) * size
    arrow_half_width = max(5, round(size * 0.015))
    arrow_length = max(12, round(size * 0.038))
    draw.polygon(
        [
            (center_x, center_y - arrow_length),
            (center_x - arrow_half_width, center_y + arrow_half_width),
            (center_x + arrow_half_width, center_y + arrow_half_width),
        ],
        fill=(55, 92, 168),
    )
    if goal_local_xy is not None:
        goal_forward, goal_left = map(float, goal_local_xy)
        goal_column = (0.5 - goal_left / view_size_m) * size
        goal_row = (1.0 - (goal_forward + rear_view_m) / view_size_m) * size
        if 0 <= goal_column < size and 0 <= goal_row < size:
            radius = max(5, round(size * 0.014))
            draw.ellipse(
                (
                    goal_column - radius,
                    goal_row - radius,
                    goal_column + radius,
                    goal_row + radius,
                ),
                fill=(224, 82, 74),
            )

    rgba = np.empty((size, size, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.asarray(image, dtype=np.uint8)
    rgba[:, :, 3] = 255
    return BevGoalMap(
        rgba=np.ascontiguousarray(rgba),
        selectable_mask=np.ascontiguousarray(free),
        view_size_m=float(view_size_m),
        rear_view_m=float(rear_view_m),
    )
