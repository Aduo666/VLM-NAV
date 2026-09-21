from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from vln_nav.active_perception.local_geometry import OCCUPIED, LocalVoxelSnapshot


def project_local_points(
    points_world: np.ndarray,
    *,
    robot_pose: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    width, height = image_size
    robot_x, robot_y, robot_z, robot_yaw = robot_pose
    delta = points_world - np.asarray((robot_x, robot_y, robot_z), dtype=np.float64)
    cosine, sine = math.cos(robot_yaw), math.sin(robot_yaw)
    points_robot = np.column_stack(
        (
            cosine * delta[:, 0] + sine * delta[:, 1],
            -sine * delta[:, 0] + cosine * delta[:, 1],
            delta[:, 2],
        )
    )
    camera_position = np.asarray((-2.4, 0.0, 2.7), dtype=np.float64)
    look_at = np.asarray((2.2, 0.0, 0.65), dtype=np.float64)
    camera_forward = look_at - camera_position
    camera_forward /= np.linalg.norm(camera_forward)
    camera_right = np.cross(camera_forward, np.asarray((0.0, 0.0, 1.0)))
    camera_right /= np.linalg.norm(camera_right)
    camera_up = np.cross(camera_right, camera_forward)
    relative = points_robot - camera_position
    depth = relative @ camera_forward
    visible = depth > 0.10
    focal_px = 0.78 * min(width, height)
    u = width * 0.5 + focal_px * (relative @ camera_right) / np.maximum(depth, 0.10)
    v = height * 0.52 - focal_px * (relative @ camera_up) / np.maximum(depth, 0.10)
    visible &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u[visible], v[visible], depth[visible], points_robot[visible, 2]


def _height_colors(relative_height_m: np.ndarray) -> np.ndarray:
    stops = np.asarray(
        [
            (0.00, 230, 0, 220),
            (0.30, 55, 45, 245),
            (0.62, 0, 225, 235),
            (1.00, 0, 240, 105),
        ],
        dtype=np.float64,
    )
    normalized = np.clip(relative_height_m / 2.4, 0.0, 1.0)
    colors = np.empty((len(normalized), 3), dtype=np.float64)
    for index in range(len(stops) - 1):
        low, high = stops[index], stops[index + 1]
        mask = (normalized >= low[0]) & (normalized <= high[0])
        alpha = (normalized[mask] - low[0]) / (high[0] - low[0])
        colors[mask] = low[1:] + alpha[:, None] * (high[1:] - low[1:])
    return colors.astype(np.uint8)


def render_local_occupancy(
    snapshot: LocalVoxelSnapshot,
    *,
    robot_pose: tuple[float, float, float, float],
    image_size: tuple[int, int] = (1280, 900),
) -> Image.Image:
    occupied_zyx = np.argwhere(snapshot.state == OCCUPIED)
    width, height = image_size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if occupied_zyx.size:
        points_world = snapshot.origin_world_xyz + (
            occupied_zyx[:, [2, 1, 0]].astype(np.float64) + 0.5
        ) * snapshot.resolution_m
        u, v, depth, relative_z = project_local_points(
            points_world, robot_pose=robot_pose, image_size=image_size
        )
        colors = _height_colors(relative_z)
        order = np.argsort(depth)[::-1]
        u_int = u[order].astype(np.int32)
        v_int = v[order].astype(np.int32)
        colors = colors[order]
        for offset_y in range(-2, 3):
            for offset_x in range(-2, 3):
                px = u_int + offset_x
                py = v_int + offset_y
                valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
                canvas[py[valid], px[valid]] = colors[valid]

    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    robot_x, robot_y, robot_z, robot_yaw = robot_pose
    marker_world = np.asarray(
        [
            (robot_x, robot_y, robot_z),
            (
                robot_x + 0.8 * math.cos(robot_yaw),
                robot_y + 0.8 * math.sin(robot_yaw),
                robot_z,
            ),
        ],
        dtype=np.float64,
    )
    marker_u, marker_v, _, _ = project_local_points(
        marker_world, robot_pose=robot_pose, image_size=image_size
    )
    if len(marker_u) == 2:
        robot_pixel = (int(marker_u[0]), int(marker_v[0]))
        heading_pixel = (int(marker_u[1]), int(marker_v[1]))
        draw.ellipse(
            (robot_pixel[0] - 7, robot_pixel[1] - 7, robot_pixel[0] + 7, robot_pixel[1] + 7),
            fill=(255, 205, 35),
        )
        draw.line((*robot_pixel, *heading_pixel), fill=(255, 85, 35), width=4)
    return image
