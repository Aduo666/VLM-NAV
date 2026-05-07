from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .landmark_map import LandmarkMapEntry
from .observation_adapter import Pose2D, VLNObservation
from .occupancy_grid import OccupancyGridSnapshot
from ..runtime.config import PROJECT_ROOT


@dataclass
class NavigationMapConfig:
    enabled: bool = True
    output_path: str = "assets/navigation_map.png"
    update_period_sec: float = 0.5
    width: int = 900
    height: int = 900
    max_depth_obstacle_points: int = 2500
    depth_sample_stride_px: int = 16
    depth_max_range_m: float = 5.0
    depth_min_range_m: float = 0.15


class NavigationMapVisualizer:
    """Persistent top-down map for debugging the VLN loop.

    It is not SLAM. It visualizes odom-frame trajectory, currently remembered
    landmark anchors, active local/world goal, and occupancy. If no occupancy
    snapshot is supplied, it falls back to sparse obstacle points projected
    from the latest depth frame.
    """

    def __init__(self, *, config: NavigationMapConfig, logger) -> None:
        self._config = config
        self._logger = logger.getChild("navigation_map")
        self._output_path = Path(config.output_path)
        if not self._output_path.is_absolute():
            self._output_path = PROJECT_ROOT / self._output_path
        self._trajectory: list[Pose2D] = []
        self._obstacle_points: list[tuple[float, float]] = []
        self._last_save_time_sec = -float("inf")
        self._last_logged = False

    @property
    def output_path(self) -> Path:
        return self._output_path

    def update(
        self,
        *,
        observation: VLNObservation,
        step_result,
        executed_action: str,
        landmark_entries: list[LandmarkMapEntry],
        timestamp_sec: float,
        occupancy_snapshot: OccupancyGridSnapshot | None = None,
    ) -> None:
        if not bool(self._config.enabled):
            return
        if float(timestamp_sec) - float(self._last_save_time_sec) < max(0.05, float(self._config.update_period_sec)):
            return
        self._append_pose(observation.pose)
        if occupancy_snapshot is None:
            self._integrate_depth_obstacles(observation=observation)
        image = self._render(
            observation=observation,
            step_result=step_result,
            executed_action=executed_action,
            landmark_entries=landmark_entries,
            occupancy_snapshot=occupancy_snapshot,
        )
        _save_image_atomic(image, self._output_path)
        self._last_save_time_sec = float(timestamp_sec)
        if not self._last_logged:
            self._logger.info("[NavigationMap] output=%s", self._output_path)
            self._last_logged = True

    def _append_pose(self, pose: Pose2D) -> None:
        if not self._trajectory:
            self._trajectory.append(Pose2D(float(pose.x), float(pose.y), float(pose.yaw_rad)))
            return
        last = self._trajectory[-1]
        distance = float(np.hypot(float(pose.x) - float(last.x), float(pose.y) - float(last.y)))
        yaw_delta = abs(_wrap_angle_rad(float(pose.yaw_rad) - float(last.yaw_rad)))
        if distance >= 0.03 or yaw_delta >= np.deg2rad(2.0):
            self._trajectory.append(Pose2D(float(pose.x), float(pose.y), float(pose.yaw_rad)))
        if len(self._trajectory) > 3000:
            self._trajectory = self._trajectory[-3000:]

    def _integrate_depth_obstacles(self, *, observation: VLNObservation) -> None:
        depth = observation.depth
        if depth is None:
            return
        depth_np = np.asarray(depth, dtype=np.float32)
        if depth_np.ndim != 2 or depth_np.size == 0:
            return
        height, width = depth_np.shape[:2]
        stride = max(4, int(self._config.depth_sample_stride_px))
        v0 = int(round(height * 0.48))
        v1 = int(round(height * 0.92))
        half_fov_rad = float(np.deg2rad(float(observation.camera.get("hfov_deg", 90.0))) * 0.5)
        max_range = max(0.2, float(self._config.depth_max_range_m))
        min_range = max(0.01, float(self._config.depth_min_range_m))

        cos_yaw = float(np.cos(observation.pose.yaw_rad))
        sin_yaw = float(np.sin(observation.pose.yaw_rad))
        new_points: list[tuple[float, float]] = []
        for v in range(max(0, v0), min(height, v1), stride):
            for u in range(0, width, stride):
                distance = float(depth_np[v, u])
                if not np.isfinite(distance) or distance < min_range or distance > max_range:
                    continue
                normalized_u = (float(u) - float(width - 1) * 0.5) / max(float(width - 1) * 0.5, 1.0)
                bearing_rad = normalized_u * half_fov_rad
                forward_m = distance * float(np.cos(bearing_rad))
                left_m = -distance * float(np.sin(bearing_rad))
                world_x = float(observation.pose.x) + cos_yaw * forward_m - sin_yaw * left_m
                world_y = float(observation.pose.y) + sin_yaw * forward_m + cos_yaw * left_m
                new_points.append((float(world_x), float(world_y)))

        if not new_points:
            return
        self._obstacle_points.extend(new_points)
        max_points = max(100, int(self._config.max_depth_obstacle_points))
        if len(self._obstacle_points) > max_points:
            self._obstacle_points = self._obstacle_points[-max_points:]

    def _render(
        self,
        *,
        observation: VLNObservation,
        step_result,
        executed_action: str,
        landmark_entries: list[LandmarkMapEntry],
        occupancy_snapshot: OccupancyGridSnapshot | None,
    ) -> Image.Image:
        width = max(320, int(self._config.width))
        height = max(320, int(self._config.height))
        image = Image.new("RGB", (width, height), (12, 15, 18))
        draw = ImageDraw.Draw(image)

        transform = _MapTransform.from_center(
            center_x=float(observation.pose.x),
            center_y=float(observation.pose.y),
            size_m=8.0,
            width=width,
            height=height,
            padding_px=70,
        )
        self._draw_grid(draw=draw, transform=transform)
        self._draw_occupancy_grid(draw=draw, transform=transform, snapshot=occupancy_snapshot)
        if occupancy_snapshot is None:
            self._draw_obstacles(draw=draw, transform=transform)
        self._draw_trajectory(draw=draw, transform=transform)
        self._draw_landmarks(draw=draw, transform=transform, entries=landmark_entries)
        self._draw_goal(draw=draw, transform=transform, step_result=step_result, robot_pose=observation.pose)
        self._draw_robot(draw=draw, transform=transform, pose=observation.pose)
        self._draw_legend(
            draw=draw,
            step_result=step_result,
            executed_action=executed_action,
            pose=observation.pose,
            width=width,
        )
        return image

    def _draw_grid(self, *, draw: ImageDraw.ImageDraw, transform: "_MapTransform") -> None:
        for x_m in np.arange(np.floor(transform.min_x), np.ceil(transform.max_x) + 0.01, 1.0):
            x, _ = transform.to_px(float(x_m), transform.min_y)
            draw.line((x, 42, x, transform.height - 30), fill=(38, 46, 54), width=1)
        for y_m in np.arange(np.floor(transform.min_y), np.ceil(transform.max_y) + 0.01, 1.0):
            _, y = transform.to_px(transform.min_x, float(y_m))
            draw.line((30, y, transform.width - 30, y), fill=(38, 46, 54), width=1)
        draw.rectangle((0, 0, transform.width, 40), fill=(0, 0, 0))
        draw.text((12, 12), "LOCAL ODOM MAP / LIDAR OCCUPANCY / LANDMARKS", fill=(245, 245, 245))

    def _draw_obstacles(self, *, draw: ImageDraw.ImageDraw, transform: "_MapTransform") -> None:
        for world_x, world_y in self._obstacle_points[-int(self._config.max_depth_obstacle_points) :]:
            x, y = transform.to_px(world_x, world_y)
            draw.rectangle((x - 1, y - 1, x + 1, y + 1), fill=(95, 105, 112))

    def _draw_occupancy_grid(
        self,
        *,
        draw: ImageDraw.ImageDraw,
        transform: "_MapTransform",
        snapshot: OccupancyGridSnapshot | None,
    ) -> None:
        if snapshot is None:
            return
        log_odds = np.asarray(snapshot.log_odds, dtype=np.float32)
        inflated = np.asarray(snapshot.inflated_occupied, dtype=bool)
        free = np.asarray(snapshot.free_mask, dtype=bool)
        if log_odds.ndim != 2 or inflated.shape != log_odds.shape:
            return
        rows, cols = log_odds.shape
        resolution = max(1e-6, float(snapshot.resolution_m))
        origin_x = float(getattr(snapshot, "origin_x", 0.0))
        origin_y = float(getattr(snapshot, "origin_y", 0.0))
        occupied = log_odds >= 1.0
        inflated_only = inflated & ~occupied
        _ = free
        self._draw_mask_cells(
            draw=draw,
            transform=transform,
            mask=inflated_only,
            rows=rows,
            cols=cols,
            resolution=resolution,
            origin_x=origin_x,
            origin_y=origin_y,
            color=(85, 28, 28),
            max_cells=4500,
        )
        self._draw_mask_cells(
            draw=draw,
            transform=transform,
            mask=occupied,
            rows=rows,
            cols=cols,
            resolution=resolution,
            origin_x=origin_x,
            origin_y=origin_y,
            color=(205, 52, 42),
            max_cells=4500,
        )

    def _draw_mask_cells(
        self,
        *,
        draw: ImageDraw.ImageDraw,
        transform: "_MapTransform",
        mask: np.ndarray,
        rows: int,
        cols: int,
        resolution: float,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
        half_size: float = 0.0,
        color: tuple[int, int, int],
        max_cells: int,
    ) -> None:
        indices = np.argwhere(mask)
        if indices.size == 0:
            return
        if len(indices) > int(max_cells):
            stride = int(np.ceil(len(indices) / max(1, int(max_cells))))
            indices = indices[::stride]
        radius = max(1, int(round(transform.scale * float(resolution) * 0.45)))
        # Use origin-based conversion when origin is provided, otherwise
        # fall back to the legacy half_size centred-at-origin conversion.
        use_origin = origin_x != 0.0 or origin_y != 0.0
        for row, col in indices:
            if row < 0 or row >= rows or col < 0 or col >= cols:
                continue
            if use_origin:
                world_x = origin_x + (float(col) + 0.5) * float(resolution)
                world_y = origin_y + (float(row) + 0.5) * float(resolution)
            else:
                world_x = (float(col) + 0.5) * float(resolution) - float(half_size)
                world_y = (float(row) + 0.5) * float(resolution) - float(half_size)
            x, y = transform.to_px(float(world_x), float(world_y))
            draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color)

    def _draw_trajectory(self, *, draw: ImageDraw.ImageDraw, transform: "_MapTransform") -> None:
        if len(self._trajectory) < 2:
            return
        pts = [transform.to_px(pose.x, pose.y) for pose in self._trajectory]
        draw.line(pts, fill=(90, 185, 255), width=3)

    def _draw_landmarks(
        self,
        *,
        draw: ImageDraw.ImageDraw,
        transform: "_MapTransform",
        entries: list[LandmarkMapEntry],
    ) -> None:
        for entry in entries:
            x, y = transform.to_px(float(entry.world_xy[0]), float(entry.world_xy[1]))
            color = (70, 220, 120) if bool(entry.reached) else (255, 190, 55)
            draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=color, outline=(255, 255, 255), width=2)
            draw.text((x + 13, y - 10), f"{entry.label} {entry.status}", fill=color)

    def _draw_goal(self, *, draw: ImageDraw.ImageDraw, transform: "_MapTransform", step_result, robot_pose: Pose2D) -> None:
        if step_result.world_goal_xy is None:
            return
        gx, gy = transform.to_px(float(step_result.world_goal_xy[0]), float(step_result.world_goal_xy[1]))
        rx, ry = transform.to_px(float(robot_pose.x), float(robot_pose.y))
        draw.line((rx, ry, gx, gy), fill=(255, 230, 0), width=3)
        draw.ellipse((gx - 7, gy - 7, gx + 7, gy + 7), fill=(255, 230, 0))
        draw.text((gx + 10, gy - 8), "active goal", fill=(255, 230, 0))

    def _draw_robot(self, *, draw: ImageDraw.ImageDraw, transform: "_MapTransform", pose: Pose2D) -> None:
        x, y = transform.to_px(float(pose.x), float(pose.y))
        yaw = float(pose.yaw_rad)
        scale = 18.0
        tip = (x + int(round(np.cos(yaw) * scale)), y - int(round(np.sin(yaw) * scale)))
        left = (
            x + int(round(np.cos(yaw + 2.45) * scale * 0.75)),
            y - int(round(np.sin(yaw + 2.45) * scale * 0.75)),
        )
        right = (
            x + int(round(np.cos(yaw - 2.45) * scale * 0.75)),
            y - int(round(np.sin(yaw - 2.45) * scale * 0.75)),
        )
        draw.polygon([tip, left, right], fill=(80, 180, 255), outline=(220, 245, 255))
        draw.text((x + 14, y + 8), "robot", fill=(180, 230, 255))

    def _draw_legend(self, *, draw: ImageDraw.ImageDraw, step_result, executed_action: str, pose: Pose2D, width: int) -> None:
        lines = [
            f"landmark={step_result.current_landmark or 'none'} completed={step_result.completed_landmarks}",
            f"policy={step_result.action_name} executed={executed_action} source={step_result.decision_source}",
            f"odom=({pose.x:.2f},{pose.y:.2f}) yaw={np.degrees(pose.yaw_rad):+.1f}deg",
            "cyan=trajectory/robot yellow=active goal orange=landmark green=reached red=LiDAR occupancy gray=depth fallback",
        ]
        line_height = 18
        box_h = 12 + len(lines) * line_height
        draw.rectangle((0, 40, min(width, 760), 40 + box_h), fill=(0, 0, 0))
        for idx, line in enumerate(lines):
            draw.text((12, 48 + idx * line_height), line, fill=(235, 238, 240))


@dataclass
class _MapTransform:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    width: int
    height: int
    scale: float
    center_x: float
    center_y: float

    @classmethod
    def from_points(cls, *, points: list[tuple[float, float]], width: int, height: int, padding_px: int) -> "_MapTransform":
        if not points:
            points = [(0.0, 0.0)]
        xs = np.asarray([float(p[0]) for p in points], dtype=np.float32)
        ys = np.asarray([float(p[1]) for p in points], dtype=np.float32)
        min_x = float(np.nanmin(xs))
        max_x = float(np.nanmax(xs))
        min_y = float(np.nanmin(ys))
        max_y = float(np.nanmax(ys))
        if max_x - min_x < 4.0:
            cx = 0.5 * (min_x + max_x)
            min_x = cx - 2.0
            max_x = cx + 2.0
        if max_y - min_y < 4.0:
            cy = 0.5 * (min_y + max_y)
            min_y = cy - 2.0
            max_y = cy + 2.0
        min_x -= 0.8
        max_x += 0.8
        min_y -= 0.8
        max_y += 0.8
        usable_w = max(10, int(width) - 2 * int(padding_px))
        usable_h = max(10, int(height) - 2 * int(padding_px) - 40)
        scale = min(usable_w / max(max_x - min_x, 1e-6), usable_h / max(max_y - min_y, 1e-6))
        return cls(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            width=int(width),
            height=int(height),
            scale=float(scale),
            center_x=0.5 * (min_x + max_x),
            center_y=0.5 * (min_y + max_y),
        )

    @classmethod
    def from_center(
        cls,
        *,
        center_x: float,
        center_y: float,
        size_m: float,
        width: int,
        height: int,
        padding_px: int,
    ) -> "_MapTransform":
        size = max(2.0, float(size_m))
        usable_w = max(10, int(width) - 2 * int(padding_px))
        usable_h = max(10, int(height) - 2 * int(padding_px) - 40)
        scale = min(float(usable_w), float(usable_h)) / size
        half = 0.5 * size
        return cls(
            min_x=float(center_x) - half,
            max_x=float(center_x) + half,
            min_y=float(center_y) - half,
            max_y=float(center_y) + half,
            width=int(width),
            height=int(height),
            scale=float(scale),
            center_x=float(center_x),
            center_y=float(center_y),
        )

    def to_px(self, x_m: float, y_m: float) -> tuple[int, int]:
        x = int(round(self.width * 0.5 + (float(x_m) - self.center_x) * self.scale))
        y = int(round(self.height * 0.5 - (float(y_m) - self.center_y) * self.scale))
        return x, y


def _save_image_atomic(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=f"{output_path.name}.", suffix=".tmp", dir=str(output_path.parent))
    os.close(fd)
    try:
        image.save(temp_path, format="PNG")
        os.replace(temp_path, output_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _wrap_angle_rad(value: float) -> float:
    return float((float(value) + np.pi) % (2.0 * np.pi) - np.pi)
