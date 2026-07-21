from __future__ import annotations

from dataclasses import dataclass
from collections import deque

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
    sector_merge_keep_sec: float
    draw_points: bool
    draw_lines: bool


@dataclass
class LidarScan:
    points_local_xy: np.ndarray
    beam_endpoints_local_xy: np.ndarray
    beam_hit_mask: np.ndarray
    current_points_local_xy: np.ndarray
    current_beam_endpoints_local_xy: np.ndarray
    current_beam_hit_mask: np.ndarray
    point_count: int
    current_point_count: int
    raw_beam_count: int
    valid_beam_count: int
    angle_min_deg: float
    angle_max_deg: float
    effective_fov_deg: float
    timestamp_sec: float
    sensor_path: str
    source: str
    sensor_origin_local_xy: np.ndarray


@dataclass
class MultiLidarSectorConfig:
    """Configuration for one sector of a multi-lidar setup."""
    child_name: str
    yaw_offset_deg: float
    horizontal_fov_deg: float


@dataclass
class MultiLidarConfig:
    """Configuration for multi-lidar setup (e.g. 3x60deg = 180deg forward)."""
    enabled: bool
    prim_path: str
    vertical_fov_deg: float
    horizontal_resolution_deg: float
    vertical_resolution_deg: float
    rotation_rate_hz: float
    min_range_m: float
    max_range_m: float
    draw_points: bool
    draw_lines: bool
    sectors: list[MultiLidarSectorConfig]


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
        self._recent_sector_history: deque[dict[str, object]] = deque(maxlen=96)
        self._last_coverage_log_time_sec = -float("inf")

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
                self._log_sensor_parameters(stage=stage)
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
                self._log_sensor_parameters(stage=stage)
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
        self._log_sensor_parameters(stage=stage)
        self._interface = _range_sensor.acquire_lidar_sensor_interface()
        self._initialized = True
        return True

    def read(self, *, pose: Pose2D, timestamp_sec: float) -> LidarScan | None:
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

        sensor_beam_endpoints, sensor_beam_hit_mask, projection_debug = _scan_to_local_beams(
            depth_raw=depth_raw,
            azimuth=azimuth,
            min_range_m=float(self._config.min_range_m),
            max_range_m=float(self._config.max_range_m),
            yaw_offset_deg=float(self._config.yaw_offset_deg),
        )
        beam_endpoints = _transform_sensor_points_to_robot(
            sensor_points=sensor_beam_endpoints,
            sensor_origin_robot_xy=self._sensor_origin_robot_xy,
            sensor_x_axis_robot_xy=self._sensor_x_axis_robot_xy,
            sensor_y_axis_robot_xy=self._sensor_y_axis_robot_xy,
        )
        beam_hit_mask = np.asarray(sensor_beam_hit_mask, dtype=bool)
        if beam_endpoints.shape[0] != beam_hit_mask.shape[0]:
            beam_hit_mask = np.zeros((beam_endpoints.shape[0],), dtype=bool)
        points = beam_endpoints[beam_hit_mask] if beam_endpoints.size else np.zeros((0, 2), dtype=np.float32)
        projection_debug = _with_point_extents(projection_debug, points)
        self._update_sector_history(
            angle_min_deg=float(projection_debug["normalized_min_deg"]),
            angle_max_deg=float(projection_debug["normalized_max_deg"]),
            beam_endpoints_local_xy=beam_endpoints,
            beam_hit_mask=beam_hit_mask,
            pose=pose,
            timestamp_sec=float(timestamp_sec),
        )
        merged_beam_endpoints, merged_beam_hit_mask, coverage_debug = self._merged_sector_scan(latest_pose=pose)
        merged_points = (
            merged_beam_endpoints[merged_beam_hit_mask]
            if merged_beam_endpoints.size and merged_beam_hit_mask.size
            else np.zeros((0, 2), dtype=np.float32)
        )
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
        if float(timestamp_sec) - self._last_coverage_log_time_sec >= 2.0:
            self._logger.info(
                "[LidarCoverage] raw_beams=%s valid_beams=%s sector_deg=[%.1f,%.1f] effective_fov_deg=%.1f merged_beams=%s merged_hits=%s",
                int(projection_debug["raw_beam_count"]),
                int(projection_debug["valid_beam_count"]),
                float(projection_debug["normalized_min_deg"]),
                float(projection_debug["normalized_max_deg"]),
                float(coverage_debug["effective_fov_deg"]),
                int(merged_beam_endpoints.shape[0]),
                int(np.count_nonzero(merged_beam_hit_mask)),
            )
            self._last_coverage_log_time_sec = float(timestamp_sec)
        if merged_beam_endpoints.size == 0:
            if float(timestamp_sec) - self._last_log_time_sec >= 2.0:
                self._logger.info("[Lidar] no_valid_points path=%s raw_shape=%s", self._sensor_path, depth_raw.shape)
                self._last_log_time_sec = float(timestamp_sec)
            return None
        if float(timestamp_sec) - self._last_log_time_sec >= 2.0:
            self._logger.info(
                "[Lidar] scan path=%s points=%s beams=%s fov_deg=%.1f",
                self._sensor_path,
                int(merged_points.shape[0]),
                int(merged_beam_endpoints.shape[0]),
                float(coverage_debug["effective_fov_deg"]),
            )
            self._last_log_time_sec = float(timestamp_sec)
        return LidarScan(
            points_local_xy=merged_points,
            beam_endpoints_local_xy=merged_beam_endpoints,
            beam_hit_mask=merged_beam_hit_mask,
            current_points_local_xy=points,
            current_beam_endpoints_local_xy=beam_endpoints,
            current_beam_hit_mask=beam_hit_mask,
            point_count=int(merged_points.shape[0]),
            current_point_count=int(points.shape[0]),
            raw_beam_count=int(projection_debug["raw_beam_count"]),
            valid_beam_count=int(projection_debug["valid_beam_count"]),
            angle_min_deg=float(coverage_debug["angle_min_deg"]),
            angle_max_deg=float(coverage_debug["angle_max_deg"]),
            effective_fov_deg=float(coverage_debug["effective_fov_deg"]),
            timestamp_sec=float(timestamp_sec),
            sensor_path=self._sensor_path,
            source="physx_lidar_sector_merged_binned",
            sensor_origin_local_xy=self._sensor_origin_robot_xy.copy(),
        )

    def _update_sector_history(
        self,
        *,
        angle_min_deg: float,
        angle_max_deg: float,
        beam_endpoints_local_xy: np.ndarray,
        beam_hit_mask: np.ndarray,
        pose: Pose2D,
        timestamp_sec: float,
    ) -> None:
        if beam_endpoints_local_xy.ndim != 2 or beam_endpoints_local_xy.shape[0] == 0:
            return
        self._recent_sector_history.append(
            {
                "angle_min_deg": float(angle_min_deg),
                "angle_max_deg": float(angle_max_deg),
                "beam_endpoints_local_xy": np.asarray(beam_endpoints_local_xy, dtype=np.float32).copy(),
                "beam_hit_mask": np.asarray(beam_hit_mask, dtype=bool).copy(),
                "pose": Pose2D(float(pose.x), float(pose.y), float(pose.yaw_rad)),
                "timestamp_sec": float(timestamp_sec),
            }
        )

    def _merged_sector_scan(self, *, latest_pose: Pose2D) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        if not self._recent_sector_history:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=bool),
                {
                    "angle_min_deg": 0.0,
                    "angle_max_deg": 0.0,
                    "effective_fov_deg": 0.0,
                },
            )
        latest_time = float(self._recent_sector_history[-1]["timestamp_sec"])
        keep_sec = max(
            0.10,
            float(self._config.sector_merge_keep_sec),
            2.5 / max(1.0, float(self._config.rotation_rate_hz)),
        )
        # The PhysX lidar can return a rotating sector instead of a full 180 deg
        # scan in one frame.  We merge recent sectors, but every stored sector is
        # first motion-compensated from its capture pose into the latest robot
        # frame.  Without this, straight walls become skewed whenever the robot
        # moves during sector accumulation.
        max_pose_delta_m = 0.45
        max_yaw_delta_deg = 30.0
        selected = [
            sector
            for sector in self._recent_sector_history
            if latest_time - float(sector["timestamp_sec"]) <= keep_sec
            and _pose_distance_m(latest_pose, sector["pose"]) <= max_pose_delta_m
            and _pose_yaw_delta_deg(latest_pose, sector["pose"]) <= max_yaw_delta_deg
        ]
        if not selected:
            selected = [self._recent_sector_history[-1]]
        beam_arrays = [
            _transform_points_between_robot_frames(
                points_local_xy=np.asarray(sector["beam_endpoints_local_xy"], dtype=np.float32),
                from_pose=sector["pose"],
                to_pose=latest_pose,
            )
            for sector in selected
        ]
        hit_arrays = [np.asarray(sector["beam_hit_mask"], dtype=bool) for sector in selected]
        merged_beams = np.concatenate(beam_arrays, axis=0) if beam_arrays else np.zeros((0, 2), dtype=np.float32)
        merged_hits = np.concatenate(hit_arrays, axis=0) if hit_arrays else np.zeros((0,), dtype=bool)
        merged_beams, merged_hits = _bin_scan_by_angle(
            beam_endpoints_local_xy=merged_beams,
            beam_hit_mask=merged_hits,
            bin_deg=1.0,
        )
        if merged_beams.size:
            merged_angles = np.rad2deg(np.arctan2(merged_beams[:, 1], merged_beams[:, 0]))
            angle_min_deg = float(np.min(merged_angles))
            angle_max_deg = float(np.max(merged_angles))
        else:
            angle_min_deg = min(float(sector["angle_min_deg"]) for sector in selected)
            angle_max_deg = max(float(sector["angle_max_deg"]) for sector in selected)
        return merged_beams, merged_hits, {
            "angle_min_deg": float(angle_min_deg),
            "angle_max_deg": float(angle_max_deg),
            "effective_fov_deg": float(max(0.0, angle_max_deg - angle_min_deg)),
        }

    def _log_sensor_parameters(self, *, stage) -> None:
        try:
            prim = stage.GetPrimAtPath(self._sensor_path)
            if not prim.IsValid():
                return
            def _attr_value(name: str):
                attr = prim.GetAttribute(name)
                return None if attr is None or not attr.IsValid() else attr.Get()

            horizontal_fov = _attr_value("horizontalFov")
            vertical_fov = _attr_value("verticalFov")
            horizontal_resolution = _attr_value("horizontalResolution")
            vertical_resolution = _attr_value("verticalResolution")
            rotation_rate = _attr_value("rotationRate")
            yaw_offset = _attr_value("yawOffset")
            min_range = _attr_value("minRange")
            max_range = _attr_value("maxRange")
            self._logger.info(
                "[LidarConfig] path=%s horizontal_fov=%s vertical_fov=%s horizontal_resolution=%s vertical_resolution=%s rotation_rate=%s min_range=%s max_range=%s yaw_offset=%s merge_keep_sec=%.2f",
                self._sensor_path,
                horizontal_fov,
                vertical_fov,
                horizontal_resolution,
                vertical_resolution,
                rotation_rate,
                min_range,
                max_range,
                yaw_offset,
                float(self._config.sector_merge_keep_sec),
            )
        except Exception as exc:
            self._logger.warning("[LidarConfig] read_failed path=%s error=%s", self._sensor_path, exc)

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


class MultiLidarReader:
    """Read multiple fixed-FOV PhysX Lidars and merge into one wide-FOV forward scan.

    Creates N PhysX Lidar prims at the same mount position, each covering a
    fixed angular sector.  The default 3-sector layout (left -60, center 0,
    right +60 deg yaw offsets, each 60 deg FOV) yields a full 180 deg forward
    scan every frame without relying on temporal sector merging.
    """

    def __init__(self, *, config: MultiLidarConfig, robot_root_path: str, logger) -> None:
        self._config = config
        self._robot_root_path = str(robot_root_path).rstrip("/")
        self._logger = logger.getChild("multi_lidar")
        self._initialized = False
        self._sector_states: list[dict[str, object]] = []
        self._sensor_origin_robot_xy = np.zeros((2,), dtype=np.float32)
        self._sensor_x_axis_robot_xy = np.asarray([1.0, 0.0], dtype=np.float32)
        self._sensor_y_axis_robot_xy = np.asarray([0.0, 1.0], dtype=np.float32)
        self._last_log_time_sec = -float("inf")
        self._last_coverage_log_time_sec = -float("inf")

    @property
    def enabled(self) -> bool:
        return bool(self._config.enabled)

    @property
    def sensor_path(self) -> str:
        if self._sector_states:
            return str(self._sector_states[0].get("sensor_path", ""))
        return ""

    def initialize(self, *, stage) -> bool:
        if not bool(self._config.enabled):
            return False
        try:
            import omni.kit.commands
            from isaacsim.sensors.physx import _range_sensor
            from pxr import Gf
        except Exception as exc:
            self._logger.warning("[MultiLidar] unavailable reason=import_failed error=%s", exc)
            return False

        mount_path = _resolve_robot_relative_prim_path(
            robot_root_path=self._robot_root_path,
            prim_path=str(self._config.prim_path),
        )
        mount_prim = stage.GetPrimAtPath(mount_path)
        if not mount_prim.IsValid():
            self._logger.warning("[MultiLidar] mount prim invalid: %s", mount_path)
            return False

        interface = _range_sensor.acquire_lidar_sensor_interface()
        self._sector_states = []

        for idx, sector_cfg in enumerate(self._config.sectors):
            child_name = str(sector_cfg.child_name).strip() or f"LidarSector{idx}"
            sector_path = f"{str(mount_path).rstrip('/')}/{child_name}"
            sector_prim = stage.GetPrimAtPath(sector_path)

            if sector_prim.IsValid() and sector_prim.GetTypeName() == "Lidar":
                self._logger.info("[MultiLidar] sector[%s] using existing prim: %s", idx, sector_path)
            else:
                result, lidar_prim = omni.kit.commands.execute(
                    "RangeSensorCreateLidar",
                    path=f"/{child_name}",
                    parent=str(mount_path),
                    min_range=float(self._config.min_range_m),
                    max_range=float(self._config.max_range_m),
                    draw_points=bool(self._config.draw_points),
                    draw_lines=bool(self._config.draw_lines),
                    horizontal_fov=float(sector_cfg.horizontal_fov_deg),
                    vertical_fov=float(self._config.vertical_fov_deg),
                    horizontal_resolution=float(self._config.horizontal_resolution_deg),
                    vertical_resolution=float(self._config.vertical_resolution_deg),
                    rotation_rate=float(self._config.rotation_rate_hz),
                    high_lod=False,
                    yaw_offset=0.0,
                )
                if not result or lidar_prim is None:
                    self._logger.warning(
                        "[MultiLidar] sector[%s] create_failed child=%s mount=%s",
                        idx, child_name, mount_path,
                    )
                    continue
                sector_path = str(lidar_prim.GetPath())
                try:
                    lidar_prim.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.0, 0.0, 0.0))
                except Exception:
                    pass
                self._logger.info(
                    "[MultiLidar] sector[%s] created prim=%s fov=%.1f yaw_offset=%.1f",
                    idx, sector_path, sector_cfg.horizontal_fov_deg, sector_cfg.yaw_offset_deg,
                )

            self._sector_states.append({
                "sensor_path": sector_path,
                "interface": interface,
                "yaw_offset_deg": float(sector_cfg.yaw_offset_deg),
                "horizontal_fov_deg": float(sector_cfg.horizontal_fov_deg),
                "min_range_m": float(self._config.min_range_m),
                "max_range_m": float(self._config.max_range_m),
            })

        if not self._sector_states:
            self._logger.warning("[MultiLidar] no sectors initialized")
            return False

        self._configure_sensor_to_robot_transform(stage=stage)
        self._log_all_sensor_parameters(stage=stage)
        self._initialized = True
        self._logger.info("[MultiLidar] initialized %d sectors", len(self._sector_states))
        return True

    def read(self, *, pose: Pose2D, timestamp_sec: float) -> LidarScan | None:
        if not self._initialized or not self._sector_states:
            return None

        all_endpoints: list[np.ndarray] = []
        all_hits: list[np.ndarray] = []
        total_raw_beams = 0
        total_valid_beams = 0
        any_read = False

        for idx, sector in enumerate(self._sector_states):
            sensor_path = str(sector["sensor_path"])
            try:
                depth_raw = np.asarray(sector["interface"].get_depth_data(sensor_path))
                azimuth = np.asarray(
                    sector["interface"].get_azimuth_data(sensor_path), dtype=np.float32
                ).reshape(-1)
            except Exception as exc:
                if float(timestamp_sec) - self._last_log_time_sec >= 2.0:
                    self._logger.warning(
                        "[MultiLidar] sector[%s] read_failed path=%s error=%s",
                        idx, sensor_path, exc,
                    )
                continue

            sensor_beams, sensor_hits, debug = _scan_to_local_beams(
                depth_raw=depth_raw,
                azimuth=azimuth,
                min_range_m=float(sector["min_range_m"]),
                max_range_m=float(sector["max_range_m"]),
                yaw_offset_deg=float(sector["yaw_offset_deg"]),
            )

            robot_beams = _transform_sensor_points_to_robot(
                sensor_points=sensor_beams,
                sensor_origin_robot_xy=self._sensor_origin_robot_xy,
                sensor_x_axis_robot_xy=self._sensor_x_axis_robot_xy,
                sensor_y_axis_robot_xy=self._sensor_y_axis_robot_xy,
            )
            hit_mask = np.asarray(sensor_hits, dtype=bool)
            if robot_beams.shape[0] != hit_mask.shape[0]:
                hit_mask = np.zeros((robot_beams.shape[0],), dtype=bool)

            if robot_beams.size > 0:
                all_endpoints.append(robot_beams)
                all_hits.append(hit_mask)
                any_read = True
            total_raw_beams += int(debug.get("raw_beam_count", 0))
            total_valid_beams += int(debug.get("valid_beam_count", 0))

        if not any_read or not all_endpoints:
            if float(timestamp_sec) - self._last_log_time_sec >= 2.0:
                self._logger.info("[MultiLidar] no_valid_points sectors=%d", len(self._sector_states))
                self._last_log_time_sec = float(timestamp_sec)
            return None

        combined_endpoints = np.concatenate(all_endpoints, axis=0)
        combined_hits = np.concatenate(all_hits, axis=0)

        if combined_endpoints.size == 0:
            return None

        # Sort by angle for consistent ordering
        angles = np.arctan2(combined_endpoints[:, 1], combined_endpoints[:, 0])
        order = np.argsort(angles)
        combined_endpoints = combined_endpoints[order]
        combined_hits = combined_hits[order]

        points = (
            combined_endpoints[combined_hits]
            if combined_hits.any()
            else np.zeros((0, 2), dtype=np.float32)
        )

        angle_min_deg = float(np.rad2deg(np.min(angles)))
        angle_max_deg = float(np.rad2deg(np.max(angles)))
        effective_fov_deg = max(0.0, angle_max_deg - angle_min_deg)

        if float(timestamp_sec) - self._last_coverage_log_time_sec >= 2.0:
            self._logger.info(
                "[MultiLidarCoverage] sectors=%d total_raw_beams=%d total_valid_beams=%d "
                "angle_range=[%.1f,%.1f] effective_fov_deg=%.1f combined_beams=%d hits=%d",
                len(self._sector_states),
                total_raw_beams,
                total_valid_beams,
                angle_min_deg,
                angle_max_deg,
                effective_fov_deg,
                int(combined_endpoints.shape[0]),
                int(np.count_nonzero(combined_hits)),
            )
            self._last_coverage_log_time_sec = float(timestamp_sec)

        if float(timestamp_sec) - self._last_log_time_sec >= 2.0:
            self._logger.info(
                "[MultiLidar] scan sectors=%d points=%d beams=%d fov_deg=%.1f",
                len(self._sector_states),
                int(points.shape[0]),
                int(combined_endpoints.shape[0]),
                effective_fov_deg,
            )
            self._last_log_time_sec = float(timestamp_sec)

        return LidarScan(
            points_local_xy=points,
            beam_endpoints_local_xy=combined_endpoints,
            beam_hit_mask=combined_hits,
            current_points_local_xy=points,
            current_beam_endpoints_local_xy=combined_endpoints,
            current_beam_hit_mask=combined_hits,
            point_count=int(points.shape[0]),
            current_point_count=int(points.shape[0]),
            raw_beam_count=total_raw_beams,
            valid_beam_count=total_valid_beams,
            angle_min_deg=angle_min_deg,
            angle_max_deg=angle_max_deg,
            effective_fov_deg=effective_fov_deg,
            timestamp_sec=float(timestamp_sec),
            sensor_path=str(self._sector_states[0].get("sensor_path", "")),
            source="multi_lidar",
            sensor_origin_local_xy=self._sensor_origin_robot_xy.copy(),
        )

    def _configure_sensor_to_robot_transform(self, *, stage) -> None:
        try:
            from pxr import Gf, Usd, UsdGeom

            robot_prim = stage.GetPrimAtPath(self._robot_root_path)
            sensor_path = str(self._sector_states[0]["sensor_path"])
            sensor_prim = stage.GetPrimAtPath(sensor_path)
            if not robot_prim.IsValid() or not sensor_prim.IsValid():
                self._logger.warning(
                    "[MultiLidar] sensor_to_robot_transform unavailable robot=%s sensor=%s",
                    self._robot_root_path, sensor_path,
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
                "[MultiLidar] sensor_to_robot origin_xy=[%.3f,%.3f] x_axis_xy=[%.3f,%.3f] "
                "sensor_forward_yaw_deg=%.1f",
                float(self._sensor_origin_robot_xy[0]),
                float(self._sensor_origin_robot_xy[1]),
                float(self._sensor_x_axis_robot_xy[0]),
                float(self._sensor_x_axis_robot_xy[1]),
                yaw_deg,
            )
        except Exception as exc:
            self._logger.warning("[MultiLidar] sensor_to_robot_transform_failed error=%s", exc)

    def _log_all_sensor_parameters(self, *, stage) -> None:
        for idx, sector in enumerate(self._sector_states):
            try:
                sensor_path = str(sector["sensor_path"])
                prim = stage.GetPrimAtPath(sensor_path)
                if not prim.IsValid():
                    continue

                def _attr_value(name: str, _p=prim):
                    attr = _p.GetAttribute(name)
                    return None if attr is None or not attr.IsValid() else attr.Get()

                self._logger.info(
                    "[MultiLidarConfig] sector[%d] path=%s horizontal_fov=%s vertical_fov=%s "
                    "horizontal_resolution=%s vertical_resolution=%s rotation_rate=%s yaw_offset=%s",
                    idx,
                    sensor_path,
                    _attr_value("horizontalFov"),
                    _attr_value("verticalFov"),
                    _attr_value("horizontalResolution"),
                    _attr_value("verticalResolution"),
                    _attr_value("rotationRate"),
                    _attr_value("yawOffset"),
                )
            except Exception as exc:
                self._logger.warning("[MultiLidarConfig] sector[%d] read_failed error=%s", idx, exc)


def _scan_to_local_beams(
    *,
    depth_raw: np.ndarray,
    azimuth: np.ndarray,
    min_range_m: float,
    max_range_m: float,
    yaw_offset_deg: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    depth = np.asarray(depth_raw)
    empty_debug = _projection_debug(
        depth_shape=tuple(depth.shape),
        azimuth_count=int(np.asarray(azimuth).size),
    )
    if depth.size == 0 or azimuth.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool), empty_debug
    if depth.ndim == 1:
        depth = depth.reshape((-1, 1))
    if depth.ndim != 2:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool), empty_debug

    rows, cols = depth.shape
    if azimuth.size != rows:
        if azimuth.size == depth.size:
            angles = azimuth.reshape(depth.shape)
        else:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool), empty_debug
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
    any_beam = np.isfinite(angle)
    hit = (
        np.isfinite(ranges)
        & ~invalid_raw
        & np.isfinite(angle)
        & (ranges >= max(0.01, float(min_range_m)))
        & (ranges <= max(float(min_range_m) + 0.1, float(max_range_m)))
    )
    if not np.any(any_beam):
        debug = _projection_debug(
            depth_shape=tuple(depth.shape),
            azimuth_count=int(azimuth.size),
            raw_beam_count=int(angle.size),
            valid_beam_count=0,
            **angle_debug,
        )
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool), debug

    endpoint_ranges = np.where(hit, ranges, float(max_range_m)).astype(np.float32)
    selected_ranges = endpoint_ranges[any_beam]
    selected_angles = angle[any_beam]
    selected_hits = hit[any_beam]
    forward = selected_ranges * np.cos(selected_angles)
    left = selected_ranges * np.sin(selected_angles)
    points = np.stack([forward, left], axis=1).astype(np.float32)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    selected_hits = np.asarray(selected_hits[finite], dtype=bool)
    if points.size == 0:
        debug = _projection_debug(
            depth_shape=tuple(depth.shape),
            azimuth_count=int(azimuth.size),
            raw_beam_count=int(angle.size),
            valid_beam_count=0,
            **angle_debug,
        )
        return points, np.zeros((0,), dtype=bool), debug
    debug = _projection_debug(
        depth_shape=tuple(depth.shape),
        azimuth_count=int(azimuth.size),
        raw_beam_count=int(angle.size),
        valid_beam_count=int(np.count_nonzero(selected_hits)),
        forward_min=float(np.min(points[:, 0])),
        forward_max=float(np.max(points[:, 0])),
        left_min=float(np.min(points[:, 1])),
        left_max=float(np.max(points[:, 1])),
        **angle_debug,
    )
    return points, selected_hits, debug


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
        # Only recenter the common static front-lidar encoding [0, pi] ->
        # [-pi/2, +pi/2].  Rotating PhysX sectors can also be positive-only
        # but have a much smaller span; recentering those collapses every
        # sector onto the robot front and destroys the map coverage.
        if (
            span >= float(0.75 * np.pi)
            and span < float(1.95 * np.pi)
            and min_angle >= -0.05
            and max_angle > 0.20
        ):
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


def _bin_scan_by_angle(
    *,
    beam_endpoints_local_xy: np.ndarray,
    beam_hit_mask: np.ndarray,
    bin_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate a motion-compensated scan by bearing.

    PhysX may expose a rotating sector.  The merged history is useful for
    visualization and bootstrap coverage, but feeding every old beam forward
    creates thousands of duplicate endpoints and makes walls look smeared.
    This keeps at most one beam per angle bin, with newer beams winning because
    callers concatenate history in chronological order.
    """
    beams = np.asarray(beam_endpoints_local_xy, dtype=np.float32)
    hits = np.asarray(beam_hit_mask, dtype=bool).reshape(-1)
    if beams.ndim != 2 or beams.shape[1] < 2 or beams.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)
    if hits.shape[0] != beams.shape[0]:
        hits = np.zeros((beams.shape[0],), dtype=bool)

    xy = beams[:, :2]
    finite = np.isfinite(xy).all(axis=1)
    if not np.any(finite):
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)

    xy = xy[finite]
    hits = hits[finite]
    bin_width = max(0.25, float(bin_deg))
    angles_deg = np.rad2deg(np.arctan2(xy[:, 1], xy[:, 0]))
    bin_ids = np.round(angles_deg / bin_width).astype(np.int32)

    latest_by_bin: dict[int, tuple[np.ndarray, bool]] = {}
    for idx, bin_id in enumerate(bin_ids):
        latest_by_bin[int(bin_id)] = (xy[idx], bool(hits[idx]))
    if not latest_by_bin:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)

    ordered_bins = sorted(latest_by_bin)
    binned_beams = np.asarray([latest_by_bin[item][0] for item in ordered_bins], dtype=np.float32)
    binned_hits = np.asarray([latest_by_bin[item][1] for item in ordered_bins], dtype=bool)
    return binned_beams, binned_hits


def _projection_debug(
    *,
    depth_shape: tuple[int, ...],
    azimuth_count: int,
    raw_beam_count: int = 0,
    valid_beam_count: int = 0,
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
        "raw_beam_count": int(raw_beam_count),
        "valid_beam_count": int(valid_beam_count),
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


def _pose_distance_m(first: Pose2D, second: Pose2D) -> float:
    return float(np.hypot(float(first.x) - float(second.x), float(first.y) - float(second.y)))


def _pose_yaw_delta_deg(first: Pose2D, second: Pose2D) -> float:
    delta = float(first.yaw_rad) - float(second.yaw_rad)
    delta = (delta + float(np.pi)) % float(2.0 * np.pi) - float(np.pi)
    return float(abs(np.degrees(delta)))


def _transform_points_between_robot_frames(
    *,
    points_local_xy: np.ndarray,
    from_pose: Pose2D,
    to_pose: Pose2D,
) -> np.ndarray:
    points = np.asarray(points_local_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)

    from_yaw = float(from_pose.yaw_rad)
    from_cos = float(np.cos(from_yaw))
    from_sin = float(np.sin(from_yaw))
    world_x = float(from_pose.x) + points[:, 0] * from_cos - points[:, 1] * from_sin
    world_y = float(from_pose.y) + points[:, 0] * from_sin + points[:, 1] * from_cos

    dx = world_x - float(to_pose.x)
    dy = world_y - float(to_pose.y)
    to_yaw = float(to_pose.yaw_rad)
    to_cos = float(np.cos(to_yaw))
    to_sin = float(np.sin(to_yaw))
    local_forward = dx * to_cos + dy * to_sin
    local_left = -dx * to_sin + dy * to_cos
    return np.stack([local_forward, local_left], axis=1).astype(np.float32)
