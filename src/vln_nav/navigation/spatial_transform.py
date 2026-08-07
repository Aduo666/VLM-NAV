from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .observation_adapter import Pose2D


@dataclass
class TargetGeometry:
    bearing_rad: float
    robot_frame_xy: tuple[float, float]
    world_xy: list[float]


def transform_world_xy_to_odom(
    world_xy: np.ndarray,
    *,
    origin_world_xy: np.ndarray | tuple[float, float] | list[float],
    origin_world_yaw_rad: float,
) -> np.ndarray:
    """Transform Isaac world XY points into the navigation odom frame.

    The odom origin is the robot world pose captured on the first simulation
    frame. Odom +X follows the robot's initial heading and odom +Y points left.
    The input may be a single ``[x, y]`` point or an array ending in dimension 2.
    """
    points = np.asarray(world_xy, dtype=np.float64)
    if points.ndim == 0 or points.shape[-1] != 2:
        raise ValueError(f"world_xy must end in dimension 2, got shape={points.shape}")
    origin = np.asarray(origin_world_xy, dtype=np.float64).reshape(-1)
    if origin.size < 2 or not np.all(np.isfinite(origin[:2])):
        raise ValueError("origin_world_xy must contain two finite values")
    if not np.all(np.isfinite(points)) or not np.isfinite(float(origin_world_yaw_rad)):
        raise ValueError("world-to-odom transform contains non-finite values")

    delta = points - origin[:2]
    cos_yaw = float(np.cos(float(origin_world_yaw_rad)))
    sin_yaw = float(np.sin(float(origin_world_yaw_rad)))
    return np.stack(
        [
            cos_yaw * delta[..., 0] + sin_yaw * delta[..., 1],
            -sin_yaw * delta[..., 0] + cos_yaw * delta[..., 1],
        ],
        axis=-1,
    )


def transform_camera_view_world_to_odom(
    camera_view_matrix_ros: np.ndarray,
    *,
    origin_world_xy: np.ndarray | tuple[float, float] | list[float],
    origin_world_yaw_rad: float,
) -> np.ndarray:
    """Express an Isaac world-to-camera ROS view matrix in the odom frame."""
    view_world = np.asarray(camera_view_matrix_ros, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(view_world)):
        raise ValueError("camera_view_matrix_ros contains non-finite values")
    origin = np.asarray(origin_world_xy, dtype=np.float64).reshape(-1)
    if origin.size < 2 or not np.all(np.isfinite(origin[:2])):
        raise ValueError("origin_world_xy must contain two finite values")

    yaw = float(origin_world_yaw_rad)
    if not np.isfinite(yaw):
        raise ValueError("origin_world_yaw_rad must be finite")
    cos_yaw = float(np.cos(yaw))
    sin_yaw = float(np.sin(yaw))
    world_to_odom = np.asarray(
        [
            [cos_yaw, sin_yaw, 0.0, -(cos_yaw * origin[0] + sin_yaw * origin[1])],
            [-sin_yaw, cos_yaw, 0.0, sin_yaw * origin[0] - cos_yaw * origin[1]],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    camera_to_world = np.linalg.inv(view_world)
    camera_to_odom = world_to_odom @ camera_to_world
    return np.linalg.inv(camera_to_odom)


def estimate_target_world_xy(
    *,
    center_uv: tuple[float, float],
    depth_m: float,
    image_width: int,
    image_height: int,
    horizontal_fov_deg: float,
    robot_pose: Pose2D,
    camera_intrinsics: np.ndarray | None = None,
    camera_view_matrix_ros: np.ndarray | None = None,
    robot_world_position_xyz: np.ndarray | None = None,
    robot_world_orientation_wxyz: np.ndarray | None = None,
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
    if (
        camera_intrinsics is not None
        and camera_view_matrix_ros is not None
        and robot_world_position_xyz is not None
        and robot_world_orientation_wxyz is not None
    ):
        return _estimate_with_camera_calibration(
            center_uv=center_uv,
            depth_m=depth_m,
            robot_pose=robot_pose,
            camera_intrinsics=camera_intrinsics,
            camera_view_matrix_ros=camera_view_matrix_ros,
            robot_world_position_xyz=robot_world_position_xyz,
            robot_world_orientation_wxyz=robot_world_orientation_wxyz,
        )

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


def _estimate_with_camera_calibration(
    *,
    center_uv: tuple[float, float],
    depth_m: float,
    robot_pose: Pose2D,
    camera_intrinsics: np.ndarray,
    camera_view_matrix_ros: np.ndarray,
    robot_world_position_xyz: np.ndarray,
    robot_world_orientation_wxyz: np.ndarray,
) -> TargetGeometry:
    """Back-project one depth pixel using Isaac Camera's ROS view matrix.

    Frames and signs:
    - camera frame is Isaac Camera's ROS projection frame used by
      ``get_view_matrix_ros`` and ``get_intrinsics_matrix``;
    - simulator world is +Z up;
    - robot frame is +X forward, +Y left;
    - navigation world is the odometry frame represented by ``robot_pose``.
    """
    intrinsics = np.asarray(camera_intrinsics, dtype=np.float64).reshape(3, 3)
    view = np.asarray(camera_view_matrix_ros, dtype=np.float64).reshape(4, 4)
    robot_world_position = np.asarray(robot_world_position_xyz, dtype=np.float64).reshape(-1)
    robot_world_orientation = np.asarray(robot_world_orientation_wxyz, dtype=np.float64).reshape(-1)
    if robot_world_position.size < 3 or robot_world_orientation.size < 4:
        raise ValueError("Robot world pose is incomplete for calibrated target projection")
    if not np.all(np.isfinite(intrinsics)) or not np.all(np.isfinite(view)):
        raise ValueError("Camera calibration contains non-finite values")

    pixel_h = np.asarray([float(center_uv[0]), float(center_uv[1]), 1.0], dtype=np.float64)
    point_camera = np.linalg.solve(intrinsics, pixel_h) * max(0.0, float(depth_m))
    point_world_h = np.linalg.inv(view) @ np.asarray(
        [float(point_camera[0]), float(point_camera[1]), float(point_camera[2]), 1.0],
        dtype=np.float64,
    )
    point_world = point_world_h[:3] / max(1.0e-9, float(point_world_h[3]))
    delta_world_xy = point_world[:2] - robot_world_position[:2]

    robot_world_yaw = _yaw_from_quaternion_wxyz(robot_world_orientation)
    cos_world = float(np.cos(robot_world_yaw))
    sin_world = float(np.sin(robot_world_yaw))
    robot_forward_m = float(cos_world * delta_world_xy[0] + sin_world * delta_world_xy[1])
    robot_left_m = float(-sin_world * delta_world_xy[0] + cos_world * delta_world_xy[1])
    # TargetObservation uses image/right-positive bearing throughout the policy.
    # Robot-frame lateral position remains +Y/left, so the signs are opposite.
    bearing_rad = float(np.arctan2(-robot_left_m, max(1.0e-9, robot_forward_m)))

    cos_odom = float(np.cos(robot_pose.yaw_rad))
    sin_odom = float(np.sin(robot_pose.yaw_rad))
    world_x = float(robot_pose.x) + cos_odom * robot_forward_m - sin_odom * robot_left_m
    world_y = float(robot_pose.y) + sin_odom * robot_forward_m + cos_odom * robot_left_m
    return TargetGeometry(
        bearing_rad=bearing_rad,
        robot_frame_xy=(robot_forward_m, robot_left_m),
        world_xy=[world_x, world_y],
    )


def _yaw_from_quaternion_wxyz(quaternion_wxyz: np.ndarray) -> float:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(quaternion[:4]))
    if norm <= 1.0e-9:
        return 0.0
    w, x, y, z = quaternion[:4] / norm
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


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
