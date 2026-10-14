from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw

from vln_nav.active_perception.local_geometry import (
    FREE,
    OCCUPIED,
    SOURCE_BITS,
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
    projection_reference_z_m: float | None = None,
    show_source_coverage: bool = False,
    rejected_unknown_world_xy: np.ndarray | None = None,
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
    reference_z = (
        robot[2] if projection_reference_z_m is None else float(projection_reference_z_m)
    )
    if not math.isfinite(reference_z):
        raise ValueError("BEV projection reference height must be finite")

    state = np.asarray(snapshot.state, dtype=np.int8)
    z_centers = snapshot.origin_world_xyz[2] + (
        np.arange(state.shape[0], dtype=np.float64) + 0.5
    ) * snapshot.resolution_m
    z_band = (
        (z_centers >= reference_z + obstacle_min_relative_z_m)
        & (z_centers <= reference_z + obstacle_max_relative_z_m)
    )
    if not np.any(z_band):
        raise ValueError("BEV obstacle height interval does not overlap the voxel map")

    band = state[z_band]
    occupied_world = np.any(band == OCCUPIED, axis=0)
    free_world = np.any(band == FREE, axis=0) & ~occupied_world
    source_world = np.bitwise_or.reduce(snapshot.source_mask[z_band], axis=0)
    return _render_goal_map(
        occupied_world, free_world, source_world,
        origin_world_xy=snapshot.origin_world_xyz[:2], resolution_m=snapshot.resolution_m,
        robot_world_xyz=robot, robot_yaw_rad=robot_yaw_rad, view_size_m=view_size_m,
        rear_view_m=rear_view_m, image_size_px=image_size_px, goal_local_xy=goal_local_xy,
        show_source_coverage=show_source_coverage,
        rejected_unknown_world_xy=rejected_unknown_world_xy,
    )


def render_planning_goal_map(
    grid, *, source_mask, robot_world_xyz, robot_yaw_rad, view_size_m,
    rear_view_m, image_size_px, goal_local_xy=None,
    show_source_coverage=False, rejected_unknown_world_xy=None,
) -> BevGoalMap:
    robot = np.asarray(robot_world_xyz, dtype=np.float64).reshape(3)
    if not np.isfinite(robot).all() or not math.isfinite(robot_yaw_rad):
        raise ValueError("地图显示需要有效的机器人位姿")
    if not math.isfinite(view_size_m) or view_size_m <= 0 or image_size_px < 32:
        raise ValueError("地图显示尺寸无效")
    if not math.isfinite(rear_view_m) or not 0 <= rear_view_m < view_size_m:
        raise ValueError("地图后方显示范围无效")
    if np.asarray(source_mask).shape != grid.shape:
        raise ValueError("二维地图来源标记尺寸无效")
    return _render_goal_map(
        grid.occupied_mask, grid.free_mask, source_mask,
        origin_world_xy=(grid.origin_x_m, grid.origin_y_m), resolution_m=grid.resolution_m,
        robot_world_xyz=robot, robot_yaw_rad=robot_yaw_rad, view_size_m=view_size_m,
        rear_view_m=rear_view_m, image_size_px=image_size_px, goal_local_xy=goal_local_xy,
        show_source_coverage=show_source_coverage,
        rejected_unknown_world_xy=rejected_unknown_world_xy,
    )


def _render_goal_map(
    occupied_world, free_world, source_world, *, origin_world_xy, resolution_m,
    robot_world_xyz, robot_yaw_rad, view_size_m, rear_view_m, image_size_px,
    goal_local_xy, show_source_coverage, rejected_unknown_world_xy,
):
    robot = np.asarray(robot_world_xyz)
    height, width = occupied_world.shape

    size = int(image_size_px)
    rows, columns = np.indices((size, size), dtype=np.float64)
    forward = (1.0 - (rows + 0.5) / size) * view_size_m - rear_view_m
    left = (0.5 - (columns + 0.5) / size) * view_size_m
    cosine, sine = math.cos(robot_yaw_rad), math.sin(robot_yaw_rad)
    world_x = robot[0] + cosine * forward - sine * left
    world_y = robot[1] + sine * forward + cosine * left
    grid_x = np.floor(
        (world_x - origin_world_xy[0]) / resolution_m
    ).astype(np.int64)
    grid_y = np.floor(
        (world_y - origin_world_xy[1]) / resolution_m
    ).astype(np.int64)
    inside = (
        (grid_x >= 0)
        & (grid_x < width)
        & (grid_y >= 0)
        & (grid_y < height)
    )
    safe_x = np.clip(grid_x, 0, width - 1)
    safe_y = np.clip(grid_y, 0, height - 1)
    occupied = inside & occupied_world[safe_y, safe_x]
    free = inside & free_world[safe_y, safe_x] & ~occupied

    rgb = np.full((size, size, 3), (108, 110, 112), dtype=np.uint8)
    rgb[free] = (238, 237, 232)
    rgb[occupied] = (28, 29, 29)
    if show_source_coverage:
        sources = source_world[safe_y, safe_x]
        front_seen = inside & ((sources & SOURCE_BITS["front"]) != 0)
        down_seen = inside & ((sources & SOURCE_BITS["down"]) != 0)
        coverage = (
            (front_seen & ~down_seen, (242, 224, 187), (119, 111, 97)),
            (down_seen & ~front_seen, (195, 225, 247), (102, 113, 124)),
            (front_seen & down_seen, (202, 232, 203), (101, 123, 103)),
        )
        for seen, free_color, unknown_color in coverage:
            rgb[seen & free] = free_color
            rgb[seen & ~free & ~occupied] = unknown_color
    if rejected_unknown_world_xy is not None:
        rejected_points = np.asarray(rejected_unknown_world_xy, dtype=np.float64)
        if rejected_points.ndim != 2 or rejected_points.shape[1] != 2:
            raise ValueError("rejected_unknown_world_xy must have shape (N, 2)")
        if not np.isfinite(rejected_points).all():
            raise ValueError("rejected_unknown_world_xy must be finite")
        rejected_world = np.zeros((height, width), dtype=bool)
        rejected_x = np.floor(
            (rejected_points[:, 0] - origin_world_xy[0])
            / resolution_m
        ).astype(np.int64)
        rejected_y = np.floor(
            (rejected_points[:, 1] - origin_world_xy[1])
            / resolution_m
        ).astype(np.int64)
        rejected_inside = (
            (rejected_x >= 0)
            & (rejected_x < width)
            & (rejected_y >= 0)
            & (rejected_y < height)
        )
        rejected_world[rejected_y[rejected_inside], rejected_x[rejected_inside]] = True
        rgb[inside & rejected_world[safe_y, safe_x] & ~free & ~occupied] = (
            235, 68, 125
        )
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    if show_source_coverage and size >= 160:
        draw.rectangle((6, 6, 126, 73), fill=(37, 40, 44))
        legend = (
            ("Front", (242, 224, 187)),
            ("Base down", (195, 225, 247)),
            ("Both", (202, 232, 203)),
            ("Blocked unknown", (235, 68, 125)),
        )
        for index, (label, color) in enumerate(legend):
            top = 11 + 15 * index
            draw.rectangle((12, top, 22, top + 9), fill=color)
            draw.text((29, top - 1), label, fill=(245, 245, 242))
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
    selectable = free.copy()
    if show_source_coverage and size >= 160:
        selectable[6:74, 6:127] = False
    return BevGoalMap(
        rgba=np.ascontiguousarray(rgba),
        selectable_mask=np.ascontiguousarray(selectable),
        view_size_m=float(view_size_m),
        rear_view_m=float(rear_view_m),
    )
