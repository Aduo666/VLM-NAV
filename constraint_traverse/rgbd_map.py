from __future__ import annotations

from dataclasses import dataclass, fields
import math

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.draw import line

from vln_nav.active_perception.camera_geometry import backproject_depth_pixels_to_world
from vln_nav.active_perception.contracts import TimedDepthObservation
from vln_nav.navigation.trajectory_clearance import PlanningGrid


SOURCE_BITS = {"front": 1, "down": 2, "arm": 4}


def _logit(probability):
    return math.log(probability / (1.0 - probability))


@dataclass(frozen=True)
class RGBDMapConfig:
    size_x_m: float = 10.0
    size_y_m: float = 10.0
    resolution_m: float = 0.05
    slide_threshold_m: float = 0.20
    min_depth_m: float = 0.12
    max_depth_m: float = 5.0
    max_ray_length_m: float = 5.0
    pixel_stride: int = 10
    probability_hit: float = 0.85
    probability_miss: float = 0.30
    probability_min: float = 0.12
    probability_max: float = 0.98
    probability_occupied: float = 0.80
    probability_free: float = 0.45

    def validate(self):
        positive = (self.size_x_m, self.size_y_m, self.resolution_m,
                    self.slide_threshold_m, self.min_depth_m, self.max_depth_m,
                    self.max_ray_length_m)
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise ValueError("二维地图尺寸、分辨率和深度范围必须为正数")
        if self.min_depth_m >= self.max_depth_m or int(self.pixel_stride) != self.pixel_stride or self.pixel_stride < 1:
            raise ValueError("二维地图深度范围或采样间隔无效")
        probabilities = (self.probability_min, self.probability_miss,
                         self.probability_free, 0.5, self.probability_occupied,
                         self.probability_hit, self.probability_max)
        if not all(math.isfinite(value) and 0 < value < 1 for value in probabilities):
            raise ValueError("二维地图占用概率必须位于零与一之间")
        if any(a >= b for a, b in zip(probabilities, probabilities[1:])):
            raise ValueError("二维地图占用概率的大小关系无效")

    @classmethod
    def from_mapping(cls, values):
        names = {item.name for item in fields(cls)}
        if not names <= set(values) or set(values) - names - {"min_z_m", "max_z_m", "integration_period_sec"}:
            raise ValueError("二维地图配置字段无效")
        config = cls(**{name: values[name] for name in names})
        config.validate()
        return config


@dataclass(frozen=True)
class RGBDMapSnapshot:
    grid: PlanningGrid
    source_mask: np.ndarray


@dataclass(frozen=True)
class RGBDMapUpdate:
    version: int
    shifted: bool
    source_ray_counts: tuple[tuple[str, int], ...]
    source_cell_update_counts: tuple[tuple[str, int, int], ...]


class RGBDPlanarMap:
    def __init__(self, config: RGBDMapConfig, *, obstacle_min_height_m, obstacle_max_height_m):
        config.validate()
        self.config = config
        self.min_height = float(obstacle_min_height_m)
        self.max_height = float(obstacle_max_height_m)
        if not (math.isfinite(self.min_height) and math.isfinite(self.max_height)
                and 0 < self.min_height < self.max_height):
            raise ValueError("二维地图障碍高度范围无效")
        self.shape = (math.ceil(config.size_y_m / config.resolution_m),
                      math.ceil(config.size_x_m / config.resolution_m))
        self.reset(np.zeros(2))

    def reset(self, center_world_xy):
        center = np.asarray(center_world_xy, dtype=float).reshape(2)
        if not np.isfinite(center).all():
            raise ValueError("二维地图中心坐标必须有限")
        size = np.asarray(self.shape[::-1]) * self.config.resolution_m
        self.origin = np.floor((center - size / 2) / self.config.resolution_m) * self.config.resolution_m
        self.log_odds = np.zeros(self.shape, dtype=np.float32)
        self.source_mask = np.zeros(self.shape, dtype=np.uint8)
        self.version = 0

    def _slide(self, center):
        r = self.config.resolution_m
        current = self.origin + np.asarray(self.shape[::-1]) * r / 2
        if np.max(np.abs(center - current)) < self.config.slide_threshold_m:
            return False
        dx, dy = np.rint((center - current) / r).astype(int)
        if dx == 0 and dy == 0:
            return False
        height, width = self.shape
        for name in ("log_odds", "source_mask"):
            previous = getattr(self, name)
            shifted = np.zeros_like(previous)
            if abs(dx) < width and abs(dy) < height:
                x0, x1 = max(0, -dx), min(width, width - dx)
                y0, y1 = max(0, -dy), min(height, height - dy)
                shifted[y0:y1, x0:x1] = previous[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            setattr(self, name, shifted)
        self.origin += np.asarray([dx, dy]) * r
        return True

    def _rays(self, observation, floor_z_m):
        depth = np.asarray(observation.depth_m)
        stride = int(self.config.pixel_stride)
        us = np.arange(stride // 2, depth.shape[1], stride)
        vs = np.arange(stride // 2, depth.shape[0], stride)
        if not len(us) or not len(vs):
            return np.empty((0, 2)), np.empty(0, dtype=bool)
        u, v = np.meshgrid(us, vs)
        sampled = depth[v, u]
        valid = np.isfinite(sampled) & (sampled >= self.config.min_depth_m) & (sampled <= self.config.max_depth_m)
        points = backproject_depth_pixels_to_world(
            pixel_u=u, pixel_v=v, depth_m=np.where(valid, sampled, 0.0),
            camera_intrinsics=observation.intrinsics,
            camera_to_world=observation.camera_to_world,
        )
        distance = np.linalg.norm(points[..., :2] - observation.camera_to_world[:2, 3], axis=-1)
        height = points[..., 2] - floor_z_m
        valid &= np.isfinite(points).all(axis=-1) & (distance > 1.0e-6)
        obstacle = valid & (height >= self.min_height) & (height <= self.max_height)
        floor = valid & (height >= -0.20) & (height < self.min_height)
        keep = obstacle | floor
        return points[..., :2][keep], obstacle[keep]

    def integrate_batch(self, observations: list[TimedDepthObservation], *, center_world_xy, floor_z_m):
        center = np.asarray(center_world_xy, dtype=float).reshape(2)
        if not np.isfinite(center).all() or not math.isfinite(floor_z_m):
            raise ValueError("二维地图位姿和地面高度必须有限")
        for observation in observations:
            observation.validate()
            if observation.source not in SOURCE_BITS:
                raise ValueError(f"二维地图相机来源无效：{observation.source}")
        shifted = self._slide(center)
        free_all, hit_all = np.zeros(self.shape, bool), np.zeros(self.shape, bool)
        ray_counts, cell_counts = [], []
        r = self.config.resolution_m
        height, width = self.shape
        prepared = []
        for observation in observations:
            endpoints, hits = self._rays(observation, floor_z_m)
            origin = observation.camera_to_world[:2, 3]
            delta = endpoints - origin
            distance = np.linalg.norm(delta, axis=-1)
            clipped = distance > self.config.max_ray_length_m
            endpoints = origin + delta * np.minimum(1.0, self.config.max_ray_length_m / np.maximum(distance, 1.0e-9))[:, None]
            hits = hits & ~clipped
            start = tuple(np.floor((origin - self.origin) / r).astype(int))
            cells = np.floor((endpoints - self.origin) / r).astype(int)
            occupied = np.zeros(self.shape, bool)
            for (column, row), hit in zip(cells, hits):
                if hit and 0 <= column < width and 0 <= row < height:
                    occupied[row, column] = True
            hit_all |= occupied
            prepared.append((observation, start, np.unique(cells, axis=0), occupied, len(endpoints)))
        for observation, start, cells, occupied, ray_count in prepared:
            free = np.zeros(self.shape, bool)
            for column, row in cells:
                endpoint = (int(column), int(row))
                intersects, first, last = cv2.clipLine((0, 0, width, height), tuple(map(int, start)), endpoint)
                if intersects:
                    rr, cc = line(first[1], first[0], last[1], last[0])
                    blocked = np.flatnonzero(hit_all[rr, cc])
                    end = int(blocked[0]) if len(blocked) else len(rr)
                    free[rr[:end], cc[:end]] = True
            free_mask = free & ~hit_all
            observed = free_mask | occupied
            self.source_mask[observed] |= SOURCE_BITS[observation.source]
            free_all |= free_mask
            ray_counts.append((observation.source, ray_count))
            cell_counts.append((observation.source, int(free_mask.sum()), int(occupied.sum())))
        # 同一批相机的障碍证据优先，空闲射线不能清除本批障碍终点。
        free_all &= ~hit_all
        self.log_odds[free_all] += _logit(self.config.probability_miss)
        self.log_odds[hit_all] += _logit(self.config.probability_hit)
        np.clip(self.log_odds, _logit(self.config.probability_min),
                _logit(self.config.probability_max), out=self.log_odds)
        self.version += 1
        return RGBDMapUpdate(self.version, shifted, tuple(ray_counts), tuple(cell_counts))

    def snapshot(self):
        occupied = self.log_odds >= _logit(self.config.probability_occupied)
        free = (self.source_mask != 0) & (self.log_odds <= _logit(self.config.probability_free)) & ~occupied
        unknown = ~(occupied | free)
        clearance = distance_transform_edt(~occupied, sampling=self.config.resolution_m).astype(np.float32)
        grid = PlanningGrid(occupied, free, unknown, clearance, float(self.origin[0]),
                            float(self.origin[1]), self.config.resolution_m, self.version)
        return RGBDMapSnapshot(grid, self.source_mask.copy())
