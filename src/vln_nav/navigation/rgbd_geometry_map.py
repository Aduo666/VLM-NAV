from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class RGBDGeometryMapConfig:
    resolution_m: float = 0.08
    size_m: float = 80.0
    update_period_sec: float = 0.20
    min_depth_m: float = 0.30
    max_depth_m: float = 8.0
    pixel_stride: int = 8
    vertical_min_ratio: float = 0.12
    vertical_max_ratio: float = 0.95
    obstacle_min_height_m: float = 0.10
    obstacle_max_height_m: float = 1.80
    floor_tolerance_below_m: float = 0.20
    occupied_log_odds: float = 0.90
    free_log_odds: float = -0.25
    occupied_threshold: float = 1.20
    free_threshold: float = -0.45
    log_odds_min: float = -4.0
    log_odds_max: float = 4.0
    debug_crop_margin_m: float = 2.0
    debug_min_view_size_m: float = 12.0
    debug_output_size_px: int = 720


@dataclass(frozen=True)
class RGBDMapUpdate:
    updated: bool
    ray_count: int
    hit_ray_count: int
    valid_depth_count: int
    occupied_cell_count: int
    free_cell_count: int


def backproject_depth_pixels_to_world(
    *,
    pixel_u: np.ndarray,
    pixel_v: np.ndarray,
    depth_m: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> np.ndarray:
    """Back-project image-plane depth into Isaac world coordinates.

    ``depth_m`` is Z-depth from Isaac's distance-to-image-plane annotator.
    Pixel coordinates use the original RGB/depth frame. The camera frame is the
    ROS optical frame expected by ``Camera.get_view_matrix_ros()``; the returned
    coordinates are in the simulator's +Z-up world frame.
    """
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


class RGBDGeometryMap:
    """North-up global geometry map built only from posed RGB-D observations.

    The map is the geometric layer of an HSGM-style representation. Each image
    column becomes one virtual planar ray. The nearest point inside the obstacle
    height band is treated as a hit; otherwise the farthest observed floor point
    clears free space. No LiDAR or semantic detector is used.
    """

    def __init__(
        self,
        *,
        config: RGBDGeometryMapConfig,
        center_world_xy: tuple[float, float],
        floor_z_m: float,
        logger,
    ) -> None:
        self._config = config
        self._logger = logger.getChild("rgbd_geometry_map")
        self._resolution_m = max(0.02, float(config.resolution_m))
        self._size_m = max(4.0, float(config.size_m))
        self._cell_count = max(2, int(round(self._size_m / self._resolution_m)))
        if self._cell_count % 2:
            self._cell_count += 1
        self._center_world_xy = np.asarray(center_world_xy, dtype=np.float64).reshape(2)
        half_extent = 0.5 * float(self._cell_count) * self._resolution_m
        self._origin_x = float(self._center_world_xy[0] - half_extent)
        self._origin_y = float(self._center_world_xy[1] - half_extent)
        self._floor_z_m = float(floor_z_m)
        self._log_odds = np.zeros((self._cell_count, self._cell_count), dtype=np.float32)
        self._trajectory_world_xy: list[tuple[float, float]] = []
        self._last_update_time_sec = -float("inf")
        self._last_update = RGBDMapUpdate(False, 0, 0, 0, 0, 0)

    @property
    def last_update(self) -> RGBDMapUpdate:
        return self._last_update

    def clear(self) -> None:
        self._log_odds.fill(0.0)
        self._trajectory_world_xy.clear()
        self._last_update_time_sec = -float("inf")
        self._last_update = RGBDMapUpdate(False, 0, 0, 0, 0, 0)

    def update(
        self,
        *,
        depth_image: np.ndarray,
        camera_intrinsics: np.ndarray,
        camera_view_matrix_ros: np.ndarray,
        robot_world_xy: tuple[float, float],
        robot_yaw_rad: float,
        timestamp_sec: float,
    ) -> RGBDMapUpdate:
        robot_xy = (float(robot_world_xy[0]), float(robot_world_xy[1]))
        self._append_trajectory(robot_xy)
        if float(timestamp_sec) - self._last_update_time_sec < max(
            0.02, float(self._config.update_period_sec)
        ):
            return self._last_update

        depth = np.asarray(depth_image, dtype=np.float32)
        intrinsics = np.asarray(camera_intrinsics, dtype=np.float64).reshape(3, 3)
        view_ros = np.asarray(camera_view_matrix_ros, dtype=np.float64).reshape(4, 4)
        if depth.ndim != 2:
            raise ValueError(f"RGB-D mapper expects HxW depth, got shape={depth.shape}")
        if not np.all(np.isfinite(intrinsics)) or not np.all(np.isfinite(view_ros)):
            raise ValueError("Camera calibration contains non-finite values")

        camera_to_world = np.linalg.inv(view_ros)
        camera_origin_world = camera_to_world[:3, 3].astype(np.float64)
        endpoints_world, hit_mask, valid_depth_count = self._extract_virtual_rays(
            depth=depth,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            camera_origin_world=camera_origin_world,
        )
        if endpoints_world.shape[0] > 0:
            self._integrate_rays(
                sensor_world_xy=camera_origin_world[:2],
                endpoints_world_xy=endpoints_world[:, :2],
                hit_mask=hit_mask,
            )

        self._last_update_time_sec = float(timestamp_sec)
        occupied_count = int(np.count_nonzero(self._log_odds >= self._config.occupied_threshold))
        free_count = int(np.count_nonzero(self._log_odds <= self._config.free_threshold))
        self._last_update = RGBDMapUpdate(
            updated=True,
            ray_count=int(endpoints_world.shape[0]),
            hit_ray_count=int(np.count_nonzero(hit_mask)),
            valid_depth_count=int(valid_depth_count),
            occupied_cell_count=occupied_count,
            free_cell_count=free_count,
        )
        return self._last_update

    def render_debug(
        self,
        *,
        robot_world_xy: tuple[float, float],
        robot_yaw_rad: float,
        semantic_instances: Iterable[object] = (),
    ) -> Image.Image:
        instances = tuple(semantic_instances)
        row_min, row_max, col_min, col_max = self._debug_crop_bounds(
            robot_world_xy=robot_world_xy,
            semantic_instances=instances,
        )
        occupied = self._log_odds >= float(self._config.occupied_threshold)
        free = self._log_odds <= float(self._config.free_threshold)
        canvas = np.full((self._cell_count, self._cell_count, 3), 150, dtype=np.uint8)
        canvas[free] = np.asarray([245, 245, 245], dtype=np.uint8)
        canvas[occupied] = np.asarray([20, 20, 20], dtype=np.uint8)
        cropped = np.flipud(canvas[row_min : row_max + 1, col_min : col_max + 1])
        output_size = max(320, int(self._config.debug_output_size_px))
        image = Image.fromarray(cropped, mode="RGB").resize(
            (output_size, output_size),
            resample=Image.Resampling.NEAREST,
        )
        draw = ImageDraw.Draw(image)
        small_font = _debug_font(16)
        medium_font = _debug_font(18)

        def world_to_pixel(x_m: float, y_m: float) -> tuple[int, int] | None:
            cell = self._world_to_cell(float(x_m), float(y_m))
            if cell is None:
                return None
            row, col = cell
            if row < row_min or row > row_max or col < col_min or col > col_max:
                return None
            crop_size = max(1, row_max - row_min + 1)
            x_px = (float(col - col_min) + 0.5) * output_size / crop_size
            y_px = (float(row_max - row) + 0.5) * output_size / crop_size
            return int(round(x_px)), int(round(y_px))

        if len(self._trajectory_world_xy) >= 2:
            path_pixels = [
                pixel
                for point in self._trajectory_world_xy
                if (pixel := world_to_pixel(*point)) is not None
            ]
            if len(path_pixels) >= 2:
                draw.line(path_pixels, fill=(30, 130, 230), width=3)

        for instance in instances:
            center = np.asarray(
                getattr(instance, "world_xyz", ()),
                dtype=np.float64,
            ).reshape(-1)
            if center.size < 2:
                continue
            pixel = world_to_pixel(float(center[0]), float(center[1]))
            if pixel is None:
                continue
            color = tuple(
                int(value) for value in getattr(instance, "color_rgb", (230, 110, 20))
            )
            radius = 8
            draw.ellipse(
                (pixel[0] - radius, pixel[1] - radius, pixel[0] + radius, pixel[1] + radius),
                fill=color,
                outline=(255, 255, 255),
                width=2,
            )
            instance_id = int(getattr(instance, "instance_id", 0))
            label = str(getattr(instance, "label", "object"))
            confidence = float(getattr(instance, "confidence", 0.0))
            observations = int(getattr(instance, "observation_count", 0))
            text = f"#{instance_id} {label} {confidence:.2f} x{observations}"
            text_position = (pixel[0] + 11, pixel[1] - 9)
            text_box = draw.textbbox(text_position, text, font=small_font)
            draw.rectangle(
                (text_box[0] - 3, text_box[1] - 2, text_box[2] + 3, text_box[3] + 2),
                fill=(255, 255, 255),
            )
            draw.text(text_position, text, fill=(0, 0, 0), font=small_font)

        robot_pixel = world_to_pixel(*robot_world_xy)
        if robot_pixel is not None:
            radius = 7
            draw.ellipse(
                (
                    robot_pixel[0] - radius,
                    robot_pixel[1] - radius,
                    robot_pixel[0] + radius,
                    robot_pixel[1] + radius,
                ),
                fill=(0, 95, 255),
                outline=(255, 255, 255),
                width=2,
            )
            arrow_length = 24
            arrow_tip = (
                int(round(robot_pixel[0] + arrow_length * np.cos(robot_yaw_rad))),
                int(round(robot_pixel[1] - arrow_length * np.sin(robot_yaw_rad))),
            )
            draw.line((robot_pixel, arrow_tip), fill=(0, 95, 255), width=5)

        visible_size_m = float(row_max - row_min + 1) * self._resolution_m
        scale_length_m = max(1.0, float(round(visible_size_m / 8.0)))
        scale_length_px = int(round(scale_length_m / visible_size_m * output_size))
        scale_y = output_size - 28
        draw.line((24, scale_y, 24 + scale_length_px, scale_y), fill=(20, 20, 20), width=4)
        draw.line((24, scale_y - 5, 24, scale_y + 5), fill=(20, 20, 20), width=2)
        draw.line(
            (24 + scale_length_px, scale_y - 5, 24 + scale_length_px, scale_y + 5),
            fill=(20, 20, 20),
            width=2,
        )
        draw.text(
            (24, scale_y - 26),
            f"{scale_length_m:g} m",
            fill=(20, 20, 20),
            font=small_font,
        )
        draw.polygon(
            [(output_size - 34, 18), (output_size - 43, 39), (output_size - 25, 39)],
            fill=(25, 25, 25),
        )
        draw.text((output_size - 41, 42), "N", fill=(25, 25, 25), font=small_font)
        draw.rectangle((8, 8, 440, 84), fill=(255, 255, 255), outline=(80, 80, 80))
        draw.text(
            (16, 13),
            "RGB-D GLOBAL MAP (NORTH UP)",
            fill=(0, 0, 0),
            font=medium_font,
        )
        draw.text(
            (16, 38),
            "black obstacle | white free | gray unknown",
            fill=(0, 0, 0),
            font=small_font,
        )
        draw.text(
            (16, 59),
            f"auto view: {visible_size_m:.1f} m | global: {self._size_m:.0f} m",
            fill=(0, 0, 0),
            font=small_font,
        )
        return image

    def _debug_crop_bounds(
        self,
        *,
        robot_world_xy: tuple[float, float],
        semantic_instances: tuple[object, ...],
    ) -> tuple[int, int, int, int]:
        observed_rows, observed_cols = np.nonzero(np.abs(self._log_odds) > 1.0e-6)
        rows = observed_rows.tolist()
        cols = observed_cols.tolist()

        points = [robot_world_xy, *self._trajectory_world_xy]
        for instance in semantic_instances:
            center = np.asarray(
                getattr(instance, "world_xyz", ()),
                dtype=np.float64,
            ).reshape(-1)
            if center.size >= 2:
                points.append((float(center[0]), float(center[1])))
        for point in points:
            cell = self._world_to_cell(float(point[0]), float(point[1]))
            if cell is not None:
                rows.append(cell[0])
                cols.append(cell[1])

        if not rows or not cols:
            center_cell = self._world_to_cell(*robot_world_xy)
            center_row, center_col = center_cell or (
                self._cell_count // 2,
                self._cell_count // 2,
            )
            rows = [center_row]
            cols = [center_col]

        margin_cells = max(
            1,
            int(np.ceil(float(self._config.debug_crop_margin_m) / self._resolution_m)),
        )
        min_side_cells = max(
            2,
            int(np.ceil(float(self._config.debug_min_view_size_m) / self._resolution_m)),
        )
        row_low = max(0, min(rows) - margin_cells)
        row_high = min(self._cell_count - 1, max(rows) + margin_cells)
        col_low = max(0, min(cols) - margin_cells)
        col_high = min(self._cell_count - 1, max(cols) + margin_cells)
        side = min(
            self._cell_count,
            max(min_side_cells, row_high - row_low + 1, col_high - col_low + 1),
        )
        center_row = 0.5 * (row_low + row_high)
        center_col = 0.5 * (col_low + col_high)
        row_min = int(round(center_row - 0.5 * (side - 1)))
        col_min = int(round(center_col - 0.5 * (side - 1)))
        row_min = int(np.clip(row_min, 0, self._cell_count - side))
        col_min = int(np.clip(col_min, 0, self._cell_count - side))
        return row_min, row_min + side - 1, col_min, col_min + side - 1

    def _extract_virtual_rays(
        self,
        *,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        camera_to_world: np.ndarray,
        camera_origin_world: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        height, width = depth.shape
        stride = max(1, int(self._config.pixel_stride))
        u_values = np.arange(stride // 2, width, stride, dtype=np.intp)
        v_start = int(np.clip(self._config.vertical_min_ratio, 0.0, 0.99) * height)
        v_stop = int(np.clip(self._config.vertical_max_ratio, 0.01, 1.0) * height)
        v_values = np.arange(v_start, max(v_start + 1, v_stop), stride, dtype=np.intp)
        if u_values.size == 0 or v_values.size == 0:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=bool), 0

        sampled_depth = depth[np.ix_(v_values, u_values)].astype(np.float64)
        valid_depth = (
            np.isfinite(sampled_depth)
            & (sampled_depth >= float(self._config.min_depth_m))
            & (sampled_depth <= float(self._config.max_depth_m))
        )
        if not np.any(valid_depth):
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=bool), 0

        uu, vv = np.meshgrid(u_values.astype(np.float64), v_values.astype(np.float64))
        points_world = backproject_depth_pixels_to_world(
            pixel_u=uu,
            pixel_v=vv,
            depth_m=sampled_depth,
            camera_intrinsics=intrinsics,
            camera_to_world=camera_to_world,
        )
        planar_distance = np.linalg.norm(points_world[..., :2] - camera_origin_world[None, None, :2], axis=-1)
        valid = (
            valid_depth
            & np.all(np.isfinite(points_world), axis=-1)
            & np.isfinite(planar_distance)
            & (planar_distance >= float(self._config.min_depth_m))
            & (planar_distance <= float(self._config.max_depth_m))
        )
        relative_height = points_world[..., 2] - self._floor_z_m
        obstacle = (
            valid
            & (relative_height >= float(self._config.obstacle_min_height_m))
            & (relative_height <= float(self._config.obstacle_max_height_m))
        )
        floor = (
            valid
            & (relative_height >= -float(self._config.floor_tolerance_below_m))
            & (relative_height < float(self._config.obstacle_min_height_m))
        )

        obstacle_distances = np.where(obstacle, planar_distance, np.inf)
        nearest_obstacle_row = np.argmin(obstacle_distances, axis=0)
        nearest_obstacle_distance = np.min(obstacle_distances, axis=0)
        has_obstacle = np.isfinite(nearest_obstacle_distance)

        floor_distances = np.where(floor, planar_distance, -np.inf)
        farthest_floor_row = np.argmax(floor_distances, axis=0)
        farthest_floor_distance = np.max(floor_distances, axis=0)
        has_floor = np.isfinite(farthest_floor_distance) & (farthest_floor_distance > 0.0)
        keep_column = has_obstacle | has_floor
        if not np.any(keep_column):
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=bool), int(np.count_nonzero(valid))

        column_indices = np.arange(u_values.size, dtype=np.intp)
        selected_rows = np.where(has_obstacle, nearest_obstacle_row, farthest_floor_row)
        selected_points = points_world[selected_rows, column_indices]
        return (
            selected_points[keep_column].astype(np.float64),
            has_obstacle[keep_column].astype(bool),
            int(np.count_nonzero(valid)),
        )

    def _integrate_rays(
        self,
        *,
        sensor_world_xy: np.ndarray,
        endpoints_world_xy: np.ndarray,
        hit_mask: np.ndarray,
    ) -> None:
        free_linear_indices: list[np.ndarray] = []
        occupied_linear_indices: list[int] = []
        sensor = np.asarray(sensor_world_xy, dtype=np.float64).reshape(2)
        endpoints = np.asarray(endpoints_world_xy, dtype=np.float64).reshape(-1, 2)
        hits = np.asarray(hit_mask, dtype=bool).reshape(-1)

        for endpoint, is_hit in zip(endpoints, hits):
            delta = endpoint - sensor
            distance = float(np.linalg.norm(delta))
            if not np.isfinite(distance) or distance < self._resolution_m:
                continue
            step_count = max(1, int(np.ceil(distance / self._resolution_m)))
            free_count = step_count if bool(is_hit) else step_count + 1
            t_values = np.arange(free_count, dtype=np.float64) / float(step_count)
            samples = sensor[None, :] + t_values[:, None] * delta[None, :]
            rows, cols, inside = self._world_to_cells(samples[:, 0], samples[:, 1])
            if np.any(inside):
                free_linear_indices.append(rows[inside] * self._cell_count + cols[inside])
            if bool(is_hit):
                endpoint_cell = self._world_to_cell(float(endpoint[0]), float(endpoint[1]))
                if endpoint_cell is not None:
                    occupied_linear_indices.append(
                        int(endpoint_cell[0] * self._cell_count + endpoint_cell[1])
                    )

        flat = self._log_odds.reshape(-1)
        if free_linear_indices:
            free_unique = np.unique(np.concatenate(free_linear_indices))
            flat[free_unique] += float(self._config.free_log_odds)
        if occupied_linear_indices:
            occupied_unique = np.unique(np.asarray(occupied_linear_indices, dtype=np.intp))
            flat[occupied_unique] += float(self._config.occupied_log_odds)
        np.clip(
            flat,
            float(self._config.log_odds_min),
            float(self._config.log_odds_max),
            out=flat,
        )

    def _append_trajectory(self, robot_world_xy: tuple[float, float]) -> None:
        point = (float(robot_world_xy[0]), float(robot_world_xy[1]))
        if not self._trajectory_world_xy:
            self._trajectory_world_xy.append(point)
            return
        previous = self._trajectory_world_xy[-1]
        if float(np.hypot(point[0] - previous[0], point[1] - previous[1])) >= 0.04:
            self._trajectory_world_xy.append(point)

    def _world_to_cells(
        self, x_values: np.ndarray, y_values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cols = np.floor((np.asarray(x_values) - self._origin_x) / self._resolution_m).astype(np.intp)
        rows = np.floor((np.asarray(y_values) - self._origin_y) / self._resolution_m).astype(np.intp)
        inside = (
            (rows >= 0)
            & (rows < self._cell_count)
            & (cols >= 0)
            & (cols < self._cell_count)
        )
        return rows, cols, inside

    def _world_to_cell(self, x_m: float, y_m: float) -> tuple[int, int] | None:
        rows, cols, inside = self._world_to_cells(np.asarray([x_m]), np.asarray([y_m]))
        if not bool(inside[0]):
            return None
        return int(rows[0]), int(cols[0])


def _debug_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=max(10, int(size)))
    except OSError:
        return ImageFont.load_default()
