from __future__ import annotations

import numpy as np


def world_to_robot_xy(
    world_xy: np.ndarray,
    *,
    robot_x: float,
    robot_y: float,
    robot_yaw_rad: float,
) -> np.ndarray:
    """Transform world XY into robot coordinates (+X forward, +Y left)."""

    points = np.asarray(world_xy, dtype=np.float32)
    delta = points[..., :2] - np.asarray([robot_x, robot_y], dtype=np.float32)
    cosine = float(np.cos(robot_yaw_rad))
    sine = float(np.sin(robot_yaw_rad))
    forward = cosine * delta[..., 0] + sine * delta[..., 1]
    left = -sine * delta[..., 0] + cosine * delta[..., 1]
    return np.stack([forward, left], axis=-1).astype(np.float32)


def robot_to_world_xy(
    local_xy: np.ndarray,
    *,
    robot_x: float,
    robot_y: float,
    robot_yaw_rad: float,
) -> np.ndarray:
    points = np.asarray(local_xy, dtype=np.float32)
    cosine = float(np.cos(robot_yaw_rad))
    sine = float(np.sin(robot_yaw_rad))
    world_x = robot_x + cosine * points[..., 0] - sine * points[..., 1]
    world_y = robot_y + sine * points[..., 0] + cosine * points[..., 1]
    return np.stack([world_x, world_y], axis=-1).astype(np.float32)


def crop_egocentric_occupancy(
    *,
    log_odds: np.ndarray,
    origin_x: float,
    origin_y: float,
    source_resolution_m: float,
    robot_x: float,
    robot_y: float,
    robot_yaw_rad: float,
    output_size_m: float,
    output_cells: int,
    occupied_threshold: float = 1.2,
    free_threshold: float = -0.5,
) -> np.ndarray:
    """Return [occupied, free, unknown] channels in a robot-centred map.

    Image top is robot-forward and image left is robot-left. Values outside the
    current global map are unknown, which preserves partial-map semantics.
    """

    grid = np.asarray(log_odds, dtype=np.float32)
    if grid.ndim != 2:
        raise ValueError("log_odds must have shape [H, W]")
    cells = max(16, int(output_cells))
    size_m = max(float(source_resolution_m) * 4.0, float(output_size_m))
    pixel_m = size_m / float(cells)
    coordinates = (np.arange(cells, dtype=np.float32) + 0.5) * pixel_m
    forward = 0.5 * size_m - coordinates[:, None]
    left = 0.5 * size_m - coordinates[None, :]
    forward = np.broadcast_to(forward, (cells, cells))
    left = np.broadcast_to(left, (cells, cells))
    world = robot_to_world_xy(
        np.stack([forward, left], axis=-1),
        robot_x=robot_x,
        robot_y=robot_y,
        robot_yaw_rad=robot_yaw_rad,
    )
    source_col = np.floor((world[..., 0] - float(origin_x)) / float(source_resolution_m)).astype(np.intp)
    source_row = np.floor((world[..., 1] - float(origin_y)) / float(source_resolution_m)).astype(np.intp)
    inside = (
        (source_row >= 0)
        & (source_row < grid.shape[0])
        & (source_col >= 0)
        & (source_col < grid.shape[1])
    )
    sampled = np.zeros((cells, cells), dtype=np.float32)
    sampled[inside] = grid[source_row[inside], source_col[inside]]
    occupied = inside & (sampled >= float(occupied_threshold))
    free = inside & (sampled <= float(free_threshold))
    unknown = ~(occupied | free)
    return np.stack([occupied, free, unknown], axis=0).astype(np.float32)
