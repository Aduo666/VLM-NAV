from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def backproject_depth_pixels_to_world(
    *,
    pixel_u: np.ndarray,
    pixel_v: np.ndarray,
    depth_m: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> np.ndarray:
    u_values, v_values, depth_values = np.broadcast_arrays(
        np.asarray(pixel_u, dtype=np.float64),
        np.asarray(pixel_v, dtype=np.float64),
        np.asarray(depth_m, dtype=np.float64),
    )
    intrinsics = np.asarray(camera_intrinsics, dtype=np.float64).reshape(3, 3)
    transform = np.asarray(camera_to_world, dtype=np.float64).reshape(4, 4)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    if abs(fx) < 1.0e-9 or abs(fy) < 1.0e-9:
        raise ValueError("Camera intrinsics contain a zero focal length")
    points_camera = np.stack(
        [
            (u_values - cx) * depth_values / fx,
            (v_values - cy) * depth_values / fy,
            depth_values,
            np.ones_like(depth_values),
        ],
        axis=-1,
    )
    points_world_h = points_camera @ transform.T
    scale = np.maximum(np.abs(points_world_h[..., 3]), 1.0e-9)
    return points_world_h[..., :3] / scale[..., None]


@dataclass(frozen=True)
class CameraFrameGeometry:
    source: str
    frame_index: int
    timestamp_sec: float
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    base_from_camera: np.ndarray
    world_from_camera: np.ndarray
    world_from_base: np.ndarray

    def validate(self) -> None:
        if not self.source.strip() or int(self.frame_index) < 0:
            raise ValueError("Camera frame requires a source and nonnegative frame index")
        if not math.isfinite(float(self.timestamp_sec)):
            raise ValueError("Camera frame timestamp must be finite")
        rgb = np.asarray(self.rgb)
        depth = np.asarray(self.depth_m)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError(f"{self.source}: RGB must be HxWx3+, got {rgb.shape}")
        if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
            raise ValueError(
                f"{self.source}: RGB/depth dimensions differ rgb={rgb.shape} depth={depth.shape}"
            )
        intrinsics = np.asarray(self.intrinsics, dtype=np.float64)
        if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
            raise ValueError(f"{self.source}: intrinsics must be a finite 3x3 matrix")
        height, width = depth.shape
        fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
        cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
        if fx <= 0 or fy <= 0 or not (0 <= cx < width) or not (0 <= cy < height):
            raise ValueError(
                f"{self.source}: invalid pinhole intrinsics fx={fx:.3f} fy={fy:.3f} "
                f"cx={cx:.3f} cy={cy:.3f} image={width}x{height}"
            )
        for name, transform in (
            ("base_from_camera", self.base_from_camera),
            ("world_from_camera", self.world_from_camera),
            ("world_from_base", self.world_from_base),
        ):
            _validate_transform(transform, f"{self.source}.{name}")


def camera_rgb_to_uint8(value) -> np.ndarray | None:
    """Normalize an Isaac camera RGB/RGBA result without importing Isaac modules."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    image = np.asarray(value)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[-1] < 3:
        return None
    image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
        if image.size and float(np.max(image)) <= 1.0:
            image = image * 255.0
    return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))


def capture_camera_frame(
    *,
    source: str,
    camera,
    frame_index: int,
    world_from_base: np.ndarray,
) -> CameraFrameGeometry | None:
    """Capture one timestamped RGB-D frame with its current optical transform."""
    frame = camera.get_current_frame()
    if not isinstance(frame, dict):
        return None
    depth_value = frame.get("distance_to_image_plane")
    rgb = camera_rgb_to_uint8(camera.get_rgba())
    timestamp = float(frame.get("rendering_time", float("nan")))
    if depth_value is None or rgb is None or not math.isfinite(timestamp):
        return None
    depth = np.asarray(depth_value, dtype=np.float32).copy()
    intrinsics = np.asarray(camera.get_intrinsics_matrix(), dtype=np.float64)
    base_from_camera, world_from_camera = transforms_from_ros_view(
        np.asarray(camera.get_view_matrix_ros(), dtype=np.float64),
        world_from_base,
    )
    return CameraFrameGeometry(
        source=source,
        frame_index=int(frame_index),
        timestamp_sec=timestamp,
        rgb=rgb,
        depth_m=depth,
        intrinsics=intrinsics,
        base_from_camera=base_from_camera,
        world_from_camera=world_from_camera,
        world_from_base=np.asarray(world_from_base, dtype=np.float64).copy(),
    )


def world_from_pose_wxyz(position_xyz: np.ndarray, orientation_wxyz: np.ndarray) -> np.ndarray:
    """Build ``world_from_base`` from Isaac's scalar-first quaternion."""
    position = np.asarray(position_xyz, dtype=np.float64).reshape(-1)
    quaternion = np.asarray(orientation_wxyz, dtype=np.float64).reshape(-1)
    if position.size < 3 or quaternion.size < 4 or not (
        np.isfinite(position[:3]).all() and np.isfinite(quaternion[:4]).all()
    ):
        raise ValueError("Robot pose must contain finite XYZ and WXYZ values")
    w, x, y, z = quaternion[:4]
    norm = math.sqrt(float(w * w + x * x + y * y + z * z))
    if norm <= 1.0e-9:
        raise ValueError("Robot orientation quaternion has zero norm")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = position[:3]
    return result


def transforms_from_ros_view(
    camera_view_matrix_ros: np.ndarray,
    world_from_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(base_from_camera, world_from_camera)`` for an Isaac camera."""
    world_to_camera = np.asarray(camera_view_matrix_ros, dtype=np.float64).reshape(4, 4)
    _validate_transform(world_to_camera, "camera_from_world")
    world_from_base = np.asarray(world_from_base, dtype=np.float64).reshape(4, 4)
    _validate_transform(world_from_base, "world_from_base")
    world_from_camera = np.linalg.inv(world_to_camera)
    base_from_camera = np.linalg.inv(world_from_base) @ world_from_camera
    _validate_transform(world_from_camera, "world_from_camera")
    _validate_transform(base_from_camera, "base_from_camera")
    return base_from_camera, world_from_camera


def _validate_transform(transform: np.ndarray, name: str) -> None:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-5):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-3):
        raise ValueError(f"{name} rotation is not orthonormal")
