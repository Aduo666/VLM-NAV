from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .observation_adapter import Pose2D


@dataclass
class TargetGeometry:
    bearing_rad: float
    robot_frame_xy: tuple[float, float]
    world_xy: list[float]


def estimate_target_world_xy(
    *,
    center_uv: tuple[float, float],
    depth_m: float,
    image_width: int,
    image_height: int,
    horizontal_fov_deg: float,
    robot_pose: Pose2D,
) -> TargetGeometry:
    """Estimate a 2D target anchor in robot/world coordinates.

    Coordinate assumptions for this first implementation:
    - The front camera optical axis is approximately aligned with robot forward.
    - Only horizontal bearing is used; vertical angle and camera height are ignored.
    - depth_m is treated as forward range along the image ray in the horizontal plane.
    - Image u increases to the right; robot local +Y is left. Therefore targets on
      the image left must produce robot_left_m > 0, and image right robot_left_m < 0.
    - The result is a navigation memory anchor, not a precise metric object pose.
    """
    width = max(2, int(image_width))
    _ = max(2, int(image_height))
    u = float(center_uv[0])
    depth = max(0.0, float(depth_m))
    cx = (float(width) - 1.0) * 0.5
    half_width = max(cx, 1.0)
    half_fov_rad = float(np.deg2rad(float(horizontal_fov_deg)) * 0.5)

    normalized_x = (u - cx) / half_width
    bearing_rad = float(np.arctan(normalized_x * np.tan(half_fov_rad)))

    robot_forward_m = float(depth * np.cos(bearing_rad))
    robot_left_m = float(-depth * np.sin(bearing_rad))

    cos_yaw = float(np.cos(robot_pose.yaw_rad))
    sin_yaw = float(np.sin(robot_pose.yaw_rad))
    world_x = float(robot_pose.x) + cos_yaw * robot_forward_m - sin_yaw * robot_left_m
    world_y = float(robot_pose.y) + sin_yaw * robot_forward_m + cos_yaw * robot_left_m
    return TargetGeometry(
        bearing_rad=bearing_rad,
        robot_frame_xy=(robot_forward_m, robot_left_m),
        world_xy=[float(world_x), float(world_y)],
    )


def log_spatial_transform_self_check(
    *,
    logger,
    image_width: int,
    image_height: int,
    horizontal_fov_deg: float,
) -> None:
    pose = Pose2D(x=0.0, y=0.0, yaw_rad=0.0)
    left = estimate_target_world_xy(
        center_uv=(float(image_width) * 0.25, float(image_height) * 0.5),
        depth_m=1.0,
        image_width=int(image_width),
        image_height=int(image_height),
        horizontal_fov_deg=float(horizontal_fov_deg),
        robot_pose=pose,
    )
    right = estimate_target_world_xy(
        center_uv=(float(image_width) * 0.75, float(image_height) * 0.5),
        depth_m=1.0,
        image_width=int(image_width),
        image_height=int(image_height),
        horizontal_fov_deg=float(horizontal_fov_deg),
        robot_pose=pose,
    )
    left_ok = bool(left.robot_frame_xy[1] > 0.0)
    right_ok = bool(right.robot_frame_xy[1] < 0.0)
    log_fn = logger.info if left_ok and right_ok else logger.warning
    log_fn(
        "[SpatialTransformCheck] image_left_robot_left_m=%.3f image_right_robot_left_m=%.3f expected_left_positive=%s expected_right_negative=%s",
        float(left.robot_frame_xy[1]),
        float(right.robot_frame_xy[1]),
        int(left_ok),
        int(right_ok),
    )
