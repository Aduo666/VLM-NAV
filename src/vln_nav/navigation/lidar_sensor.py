from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .observation_adapter import Pose2D


@dataclass
class PhysXLidarConfig:
    enabled: bool
    prim_path: str
    child_name: str
    horizontal_fov_deg: float
    vertical_fov_deg: float
    horizontal_resolution_deg: float
    vertical_resolution_deg: float
    rotation_rate_hz: float
    min_range_m: float
    max_range_m: float
    yaw_offset_deg: float
    draw_points: bool
    draw_lines: bool


@dataclass
class LidarScan:
    points_local_xy: np.ndarray
    point_count: int
    timestamp_sec: float
    sensor_path: str
    source: str


class PhysXLidarReader:
    """Read a PhysX RangeSensor Lidar as robot-local 2D scan points."""

    def __init__(self, *, config: PhysXLidarConfig, robot_root_path: str, logger) -> None:
        self._config = config
        self._robot_root_path = str(robot_root_path).rstrip("/")
        self._logger = logger.getChild("lidar")
        self._sensor_path = ""
        self._interface = None
        self._initialized = False
        self._last_log_time_sec = -float("inf")
        self._diagnostic_logged = False
        self._sensor_origin_robot_xy = np.zeros((2,), dtype=np.float32)
        self._sensor_x_axis_robot_xy = np.asarray([1.0, 0.0], dtype=np.float32)
        self._sensor_y_axis_robot_xy = np.asarray([0.0, 1.0], dtype=np.float32)

    @property
    def enabled(self) -> bool:
        return bool(self._config.enabled)

    @property
    def sensor_path(self) -> str:
        return self._sensor_path

    def initialize(self, *, stage) -> bool:
        if not bool(self._config.enabled):
            return False
        try:
            import omni.kit.commands
            from isaacsim.sensors.physx import _range_sensor
            from pxr import Gf
        except Exception as exc:
            self._logger.warning("[Lidar] unavailable reason=import_failed error=%s", exc)
            return False

        requested_path = _resolve_robot_relative_prim_path(
            robot_root_path=self._robot_root_path,
            prim_path=str(self._config.prim_path),
        )
        requested_prim = stage.GetPrimAtPath(requested_path)
        if requested_prim.IsValid() and requested_prim.GetTypeName() == "Lidar":
            self._sensor_path = requested_path
            self._logger.info("[Lidar] using existing PhysX Lidar prim: %s", self._sensor_path)
        else:
            existing_child_path = _find_child_lidar_path(stage=stage, root_path=requested_path)
            if existing_child_path:
                self._sensor_path = existing_child_path
                self._logger.info(
                    "[Lidar] using existing mounted PhysX Lidar prim=%s mount=%s",
                    self._sensor_path,
                    requested_path,
                )
                self._configure_sensor_to_robot_transform(stage=stage)
                self._interface = _range_sensor.acquire_lidar_sensor_interface()
                self._initialized = True
                return True
            parent_path = requested_path if requested_prim.IsValid() else str(requested_path).rsplit("/", 1)[0]
            child_name = str(self._config.child_name).strip() or "lidar_sensor"
            managed_path = f"{str(parent_path).rstrip('/')}/{child_name}"
            managed_prim = stage.GetPrimAtPath(managed_path)
            if managed_prim.IsValid() and managed_prim.GetTypeName() == "Lidar":
                self._sensor_path = managed_path
                self._logger.info(
                    "[Lidar] using managed PhysX Lidar prim=%s mount=%s",
                    self._sensor_path,
                    parent_path,
                )
                self._configure_sensor_to_robot_transform(stage=stage)
                self._interface = _range_sensor.acquire_lidar_sensor_interface()
                self._initialized = True
                return True
            result, lidar_prim = omni.kit.commands.execute(
                "RangeSensorCreateLidar",
                path=f"/{child_name}",
                parent=parent_path,
                min_range=float(self._config.min_range_m),
                max_range=float(self._config.max_range_m),
                draw_points=bool(self._config.draw_points),
                draw_lines=bool(self._config.draw_lines),
                horizontal_fov=float(self._config.horizontal_fov_deg),
                vertical_fov=float(self._config.vertical_fov_deg),
                horizontal_resolution=float(self._config.horizontal_resolution_deg),
                vertical_resolution=float(self._config.vertical_resolution_deg),
                rotation_rate=float(self._config.rotation_rate_hz),
                high_lod=False,
                yaw_offset=float(self._config.yaw_offset_deg),
            )
            if not result or lidar_prim is None:
                self._logger.warning("[Lidar] unavailable reason=create_failed requested=%s parent=%s", requested_path, parent_path)
                return False
            self._sensor_path = str(lidar_prim.GetPath())
            try:
                lidar_prim.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.0, 0.0, 0.0))
            except Exception:
                pass
            self._logger.info(
                "[Lidar] created fallback PhysX Lidar prim=%s parent=%s requested_mount=%s",
                self._sensor_path,
                parent_path,
                requested_path,
            )

        self._configure_sensor_to_robot_transform(stage=stage)
        self._interface = _range_sensor.acquire_lidar_sensor_interface()
        self._initialized = True
        return True

    def read(self, *, pose: Pose2D, timestamp_sec: float) -> LidarScan | None:
        _ = pose
        if not self._initialized or self._interface is None or not self._sensor_path:
            return None
        try:
            depth_raw = np.asarray(self._interface.get_depth_data(self._sensor_path))
            azimuth = np.asarray(self._interface.get_azimuth_data(self._sensor_path), dtype=np.float32).reshape(-1)
        except Exception as exc:
            if float(timestamp_sec) - self._last_log_time_sec >= 2.0:
                self._logger.warning("[Lidar] read_failed path=%s error=%s", self._sensor_path, exc)
                self._last_log_time_sec = float(timestamp_sec)
            return None

        sensor_points, projection_debug = _scan_to_local_points(
            depth_raw=depth_raw,
            azimuth=azimuth,
            min_range_m=float(self._config.min_range_m),
            max_range_m=float(self._config.max_range_m),
            yaw_offset_deg=float(self._config.yaw_offset_deg),
        )
        points = _transform_sensor_points_to_robot(
            sensor_points=sensor_points,
            sensor_origin_robot_xy=self._sensor_origin_robot_xy,
            sensor_x_axis_robot_xy=self._sensor_x_axis_robot_xy,
            sensor_y_axis_robot_xy=self._sensor_y_axis_robot_xy,
        )
        projection_debug = _with_point_extents(projection_debug, points)
        if not self._diagnostic_logged:
            self._logger.info(
                "[LidarDebug] path=%s depth_shape=%s azimuth_count=%s raw_range=[%.3f,%.3f] "
                "unit=%s centered=%s normalized_deg=[%.1f,%.1f] local_forward=[%.2f,%.2f] "
                "local_left=[%.2f,%.2f] points=%s",
                self._sensor_path,
                projection_debug["depth_shape"],
                projection_debug["azimuth_count"],
                projection_debug["raw_min"],
                projection_debug["raw_max"],
                projection_debug["unit"],
                int(bool(projection_debug["centered"])),
                projection_debug["normalized_min_deg"],
                projection_debug["normalized_max_deg"],
                projection_debug["forward_min"],
                projection_debug["forward_max"],
                projection_debug["left_min"],
                projection_debug["left_max"],
                int(points.shape[0]),
            )
            self._diagnostic_logged = True
        if points.size == 0:
            if float(timestamp_sec) - self._last_log_time_sec >= 2.0:
                self._logger.info("[Lidar] no_valid_points path=%s raw_shape=%s", self._sensor_path, depth_raw.shape)
                self._last_log_time_sec = float(timestamp_sec)
            return None
        if float(timestamp_sec) - self._last_log_time_sec >= 2.0:
            self._logger.info("[Lidar] scan path=%s points=%s", self._sensor_path, int(points.shape[0]))
            self._last_log_time_sec = float(timestamp_sec)
        return LidarScan(
            points_local_xy=points,
            point_count=int(points.shape[0]),
            timestamp_sec=float(timestamp_sec),
            sensor_path=self._sensor_path,
            source="physx_lidar",
        )

    def _configure_sensor_to_robot_transform(self, *, stage) -> None:
        try:
            from pxr import Gf, Usd, UsdGeom

            robot_prim = stage.GetPrimAtPath(self._robot_root_path)
            sensor_prim = stage.GetPrimAtPath(self._sensor_path)
            if not robot_prim.IsValid() or not sensor_prim.IsValid():
                self._logger.warning(
                    "[Lidar] sensor_to_robot_transform unavailable robot=%s sensor=%s",
                    self._robot_root_path,
                    self._sensor_path,
                )
                return
            time_code = Usd.TimeCode.Default()
            robot_world = UsdGeom.Xformable(robot_prim).ComputeLocalToWorldTransform(time_code)
            sensor_world = UsdGeom.Xformable(sensor_prim).ComputeLocalToWorldTransform(time_code)
            robot_inv = robot_world.GetInverse()

            origin_robot = robot_inv.Transform(sensor_world.Transform(Gf.Vec3d(0.0, 0.0, 0.0)))
            x_robot = robot_inv.Transform(sensor_world.Transform(Gf.Vec3d(1.0, 0.0, 0.0))) - origin_robot
            y_robot = robot_inv.Transform(sensor_world.Transform(Gf.Vec3d(0.0, 1.0, 0.0))) - origin_robot

            x_axis = np.asarray([float(x_robot[0]), float(x_robot[1])], dtype=np.float32)
            y_axis = np.asarray([float(y_robot[0]), float(y_robot[1])], dtype=np.float32)
            x_norm = float(np.linalg.norm(x_axis))
            y_norm = float(np.linalg.norm(y_axis))
            if x_norm > 1.0e-6:
                x_axis = x_axis / x_norm
            if y_norm > 1.0e-6:
                y_axis = y_axis / y_norm

            self._sensor_origin_robot_xy = np.asarray(
                [float(origin_robot[0]), float(origin_robot[1])],
                dtype=np.float32,
            )
            self._sensor_x_axis_robot_xy = x_axis
            self._sensor_y_axis_robot_xy = y_axis
            yaw_deg = float(np.rad2deg(np.arctan2(float(x_axis[1]), float(x_axis[0]))))
            self._logger.info(
                "[Lidar] sensor_to_robot origin_xy=[%.3f,%.3f] x_axis_xy=[%.3f,%.3f] "
                "y_axis_xy=[%.3f,%.3f] sensor_forward_yaw_deg=%.1f",
                float(self._sensor_origin_robot_xy[0]),
                float(self._sensor_origin_robot_xy[1]),
                float(self._sensor_x_axis_robot_xy[0]),
                float(self._sensor_x_axis_robot_xy[1]),
                float(self._sensor_y_axis_robot_xy[0]),
                float(self._sensor_y_axis_robot_xy[1]),
                yaw_deg,
            )
        except Exception as exc:
            self._logger.warning("[Lidar] sensor_to_robot_transform_failed error=%s", exc)


def _scan_to_local_points(
    *,
    depth_raw: np.ndarray,
    azimuth: np.ndarray,
    min_range_m: float,
    max_range_m: float,
    yaw_offset_deg: float,
) -> tuple[np.ndarray, dict[str, object]]:
    depth = np.asarray(depth_raw)
    empty_debug = _projection_debug(
        depth_shape=tuple(depth.shape),
        azimuth_count=int(np.asarray(azimuth).size),
    )
    if depth.size == 0 or azimuth.size == 0:
        return np.zeros((0, 2), dtype=np.float32), empty_debug
    if depth.ndim == 1:
        depth = depth.reshape((-1, 1))
    if depth.ndim != 2:
        return np.zeros((0, 2), dtype=np.float32), empty_debug

    rows, cols = depth.shape
    if azimuth.size != rows:
        if azimuth.size == depth.size:
            angles = azimuth.reshape(depth.shape)
        else:
            return np.zeros((0, 2), dtype=np.float32), empty_debug
    else:
        angles = np.repeat(azimuth[:, None], cols, axis=1)

    depth_f = depth.astype(np.float32)
    invalid_raw = depth_f >= 65534.0
    finite_depth = depth_f[np.isfinite(depth_f)]
    max_depth_value = float(np.max(finite_depth)) if finite_depth.size else 0.0
    if np.issubdtype(depth.dtype, np.integer) or max_depth_value > float(max_range_m) * 2.0:
        ranges = depth_f * float(max_range_m) / 65535.0
    else:
        ranges = depth_f
    angle, angle_debug = _normalize_lidar_azimuth(angles=angles, yaw_offset_deg=yaw_offset_deg)
    valid = (
        np.isfinite(ranges)
        & ~invalid_raw
        & np.isfinite(angle)
        & (ranges >= max(0.01, float(min_range_m)))
        & (ranges <= max(float(min_range_m) + 0.1, float(max_range_m)))
    )
    if not np.any(valid):
        debug = _projection_debug(
            depth_shape=tuple(depth.shape),
            azimuth_count=int(azimuth.size),
            **angle_debug,
        )
        return np.zeros((0, 2), dtype=np.float32), debug

    selected_ranges = ranges[valid]
    selected_angles = angle[valid]
    forward = selected_ranges * np.cos(selected_angles)
    left = selected_ranges * np.sin(selected_angles)
    points = np.stack([forward, left], axis=1).astype(np.float32)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if points.size == 0:
        debug = _projection_debug(
            depth_shape=tuple(depth.shape),
            azimuth_count=int(azimuth.size),
            **angle_debug,
        )
        return points, debug
    debug = _projection_debug(
        depth_shape=tuple(depth.shape),
        azimuth_count=int(azimuth.size),
        forward_min=float(np.min(points[:, 0])),
        forward_max=float(np.max(points[:, 0])),
        left_min=float(np.min(points[:, 1])),
        left_max=float(np.max(points[:, 1])),
        **angle_debug,
    )
    return points, debug


def _normalize_lidar_azimuth(*, angles: np.ndarray, yaw_offset_deg: float) -> tuple[np.ndarray, dict[str, object]]:
    raw = np.asarray(angles, dtype=np.float32)
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        return raw, {
            "raw_min": 0.0,
            "raw_max": 0.0,
            "unit": "unknown",
            "centered": False,
            "normalized_min_deg": 0.0,
            "normalized_max_deg": 0.0,
        }

    raw_min = float(np.min(finite))
    raw_max = float(np.max(finite))
    raw_abs_max = max(abs(raw_min), abs(raw_max))
    if raw_abs_max > float(2.0 * np.pi + 0.25):
        angle = np.deg2rad(raw).astype(np.float32)
        unit = "deg"
    else:
        angle = raw.astype(np.float32)
        unit = "rad"

    finite_angle = angle[np.isfinite(angle)]
    centered = False
    if finite_angle.size:
        min_angle = float(np.min(finite_angle))
        max_angle = float(np.max(finite_angle))
        span = max_angle - min_angle
        # PhysX front-only lidar can report a 180-degree scan as [0, pi].
        # For robot-local occupancy mapping we need that sector centered on +X:
        # [-pi/2, +pi/2], where +Y is robot-left.
        if span < float(1.95 * np.pi) and min_angle >= -0.05 and max_angle > 0.20:
            center = 0.5 * (min_angle + max_angle)
            angle = angle - np.float32(center)
            centered = True

    angle = angle + np.float32(np.deg2rad(float(yaw_offset_deg)))
    finite_normalized = angle[np.isfinite(angle)]
    if finite_normalized.size:
        normalized_min_deg = float(np.rad2deg(np.min(finite_normalized)))
        normalized_max_deg = float(np.rad2deg(np.max(finite_normalized)))
    else:
        normalized_min_deg = 0.0
        normalized_max_deg = 0.0
    return angle, {
        "raw_min": raw_min,
        "raw_max": raw_max,
        "unit": unit,
        "centered": centered,
        "normalized_min_deg": normalized_min_deg,
        "normalized_max_deg": normalized_max_deg,
    }


def _transform_sensor_points_to_robot(
    *,
    sensor_points: np.ndarray,
    sensor_origin_robot_xy: np.ndarray,
    sensor_x_axis_robot_xy: np.ndarray,
    sensor_y_axis_robot_xy: np.ndarray,
) -> np.ndarray:
    points = np.asarray(sensor_points, dtype=np.float32)
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    forward = points[:, 0:1]
    left = points[:, 1:2]
    transformed = (
        np.asarray(sensor_origin_robot_xy, dtype=np.float32)[None, :]
        + forward * np.asarray(sensor_x_axis_robot_xy, dtype=np.float32)[None, :]
        + left * np.asarray(sensor_y_axis_robot_xy, dtype=np.float32)[None, :]
    )
    finite = np.isfinite(transformed).all(axis=1)
    return transformed[finite].astype(np.float32)


def _with_point_extents(debug: dict[str, object], points: np.ndarray) -> dict[str, object]:
    updated = dict(debug)
    if points.size == 0:
        updated.update(
            {
                "forward_min": 0.0,
                "forward_max": 0.0,
                "left_min": 0.0,
                "left_max": 0.0,
            }
        )
        return updated
    updated.update(
        {
            "forward_min": float(np.min(points[:, 0])),
            "forward_max": float(np.max(points[:, 0])),
            "left_min": float(np.min(points[:, 1])),
            "left_max": float(np.max(points[:, 1])),
        }
    )
    return updated


def _projection_debug(
    *,
    depth_shape: tuple[int, ...],
    azimuth_count: int,
    raw_min: float = 0.0,
    raw_max: float = 0.0,
    unit: str = "unknown",
    centered: bool = False,
    normalized_min_deg: float = 0.0,
    normalized_max_deg: float = 0.0,
    forward_min: float = 0.0,
    forward_max: float = 0.0,
    left_min: float = 0.0,
    left_max: float = 0.0,
) -> dict[str, object]:
    return {
        "depth_shape": depth_shape,
        "azimuth_count": int(azimuth_count),
        "raw_min": float(raw_min),
        "raw_max": float(raw_max),
        "unit": str(unit),
        "centered": bool(centered),
        "normalized_min_deg": float(normalized_min_deg),
        "normalized_max_deg": float(normalized_max_deg),
        "forward_min": float(forward_min),
        "forward_max": float(forward_max),
        "left_min": float(left_min),
        "left_max": float(left_max),
    }


def _resolve_robot_relative_prim_path(*, robot_root_path: str, prim_path: str) -> str:
    text = str(prim_path).strip()
    if text.startswith("/"):
        return text
    return f"{str(robot_root_path).rstrip('/')}/{text.lstrip('/')}"


def _find_child_lidar_path(*, stage, root_path: str) -> str:
    root = stage.GetPrimAtPath(str(root_path))
    if not root.IsValid():
        return ""
    direct_children = list(root.GetChildren())
    for prim in direct_children:
        if prim.GetTypeName() == "Lidar" and prim.GetName() == "Lidar":
            return str(prim.GetPath())
    for prim in direct_children:
        if prim.GetTypeName() == "Lidar":
            return str(prim.GetPath())
    for prim in direct_children:
        nested = _find_child_lidar_path(stage=stage, root_path=str(prim.GetPath()))
        if nested:
            return nested
    return ""
