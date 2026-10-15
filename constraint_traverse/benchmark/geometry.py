from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import shapely
import torch
from shapely import affinity
from shapely.geometry import box
from shapely.ops import unary_union


@dataclass(frozen=True)
class DoorScene:
    width_m: float
    thickness_m: float = 0.25
    half_size_m: float = 3.0
    height_m: float = 2.0

    def __post_init__(self):
        values = (self.width_m, self.thickness_m, self.half_size_m, self.height_m)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("窄门尺寸必须为有限正数")
        if self.width_m >= 2 * self.half_size_m:
            raise ValueError("门宽必须小于场景宽度")

    @property
    def obstacles(self) -> np.ndarray:
        t, w, h = self.thickness_m / 2, self.width_m / 2, self.half_size_m
        return np.asarray([[-t, t, w, h], [-t, t, -h, -w]], dtype=np.float64)

    @property
    def exit_x_m(self) -> float:
        return self.thickness_m / 2


@dataclass(frozen=True)
class SweepResult:
    safe: bool
    reason: str
    min_clearance_m: float


def sample_body_points(parts, spacing_m):
    points = []
    for xmin, xmax, ymin, ymax in np.asarray(parts):
        xs = np.linspace(xmin, xmax, max(2, math.ceil((xmax - xmin) / spacing_m) + 1))
        ys = np.linspace(ymin, ymax, max(2, math.ceil((ymax - ymin) / spacing_m) + 1))
        x, y = np.meshgrid(xs, ys)
        points.append(np.column_stack((x.ravel(), y.ravel())))
    return np.unique(np.round(np.concatenate(points), 7), axis=0).astype(np.float32)


def swept_part_polygons(parts, poses, *, radius_m, margin_m, step_m):
    parts = np.asarray(parts, dtype=float).reshape(-1, 4)
    poses = np.asarray(poses, dtype=float).reshape(-1, 3)
    previous = poses[0]
    samples, margins = [], []
    for endpoint in poses:
        delta = endpoint - previous
        delta[2] = (delta[2] + math.pi) % (2 * math.pi) - math.pi
        motion_bound = np.linalg.norm(delta[:2]) + radius_m * abs(delta[2])
        count = max(1, math.ceil(motion_bound / step_m))
        # 相邻采样间最大轮廓位移的一半覆盖连续运动空隙。
        fractions = np.linspace(0, 1, count + 1)
        samples.append(previous[None, :] + fractions[:, None] * delta)
        margins.extend([margin_m + motion_bound / (2 * count)] * len(fractions))
        previous = endpoint
    samples = np.concatenate(samples)
    corners = np.stack((parts[:, [0, 2]], parts[:, [1, 2]], parts[:, [1, 3]], parts[:, [0, 3]]), axis=1)
    c, s = np.cos(samples[:, 2, None, None]), np.sin(samples[:, 2, None, None])
    x = samples[:, 0, None, None] + c * corners[:, :, 0] - s * corners[:, :, 1]
    y = samples[:, 1, None, None] + s * corners[:, :, 0] + c * corners[:, :, 1]
    polygons = shapely.polygons(np.stack((x, y), axis=-1).reshape(-1, 4, 2))
    return polygons, np.asarray(margins), x, y


class PassageGeometry:
    def __init__(self, parts, scene: DoorScene, *, margin_m=0.03, sweep_step_m=0.01):
        self.parts = np.asarray(parts, dtype=np.float64).reshape(-1, 4)
        if len(self.parts) == 0 or not np.isfinite(self.parts).all():
            raise ValueError("车体轮廓不能为空且必须包含有效坐标")
        if np.any(self.parts[:, 0] >= self.parts[:, 1]) or np.any(self.parts[:, 2] >= self.parts[:, 3]):
            raise ValueError("车体部件的边界无效")
        if margin_m < 0 or sweep_step_m <= 0:
            raise ValueError("安全余量和扫掠步长无效")
        self.scene = scene
        self.margin_m = float(margin_m)
        self.sweep_step_m = float(sweep_step_m)
        self.body = unary_union([box(p[0], p[2], p[1], p[3]) for p in self.parts])
        self.obstacle_geometry = unary_union([box(p[0], p[2], p[1], p[3]) for p in scene.obstacles])
        self.room = box(-scene.half_size_m, -scene.half_size_m, scene.half_size_m, scene.half_size_m)
        self.radius_m = float(np.max(np.hypot(np.abs(self.parts[:, :2]).max(axis=1), np.abs(self.parts[:, 2:]).max(axis=1))))
        self.corners = np.stack((self.parts[:, [0, 2]], self.parts[:, [1, 2]], self.parts[:, [1, 3]], self.parts[:, [0, 3]]), axis=1)
        self._tensor_cache = {}
        self._point_cache = {}

    def torch_metrics(self, poses):
        key = (poses.device, poses.dtype)
        if key not in self._point_cache:
            self._point_cache[key] = (
                torch.as_tensor(sample_body_points(self.parts, 0.05), device=poses.device, dtype=poses.dtype),
                torch.as_tensor(self.scene.obstacles, device=poses.device, dtype=poses.dtype),
            )
        points, obstacles = self._point_cache[key]
        clearance = self.torch_clearance(poses)
        occupied_fraction = torch.zeros_like(clearance)
        outside_fraction = torch.zeros_like(clearance)
        # 精确分离轴距离为正时，车体覆盖的障碍面积为零。
        overlapping = clearance <= 0
        selected = poses[overlapping]
        if len(selected):
            c, s = torch.cos(selected[..., 2, None]), torch.sin(selected[..., 2, None])
            x = selected[..., 0, None] + c * points[:, 0] - s * points[:, 1]
            y = selected[..., 1, None] + s * points[:, 0] + c * points[:, 1]
            occupied = ((x[..., None] >= obstacles[:, 0]) & (x[..., None] <= obstacles[:, 1])
                        & (y[..., None] >= obstacles[:, 2]) & (y[..., None] <= obstacles[:, 3])).any(dim=-1)
            outside = (x.abs() > self.scene.half_size_m) | (y.abs() > self.scene.half_size_m)
            occupied_fraction[overlapping] = occupied.to(poses.dtype).mean(dim=-1)
            outside_fraction[overlapping] = outside.to(poses.dtype).mean(dim=-1)
        return {"clearance": clearance, "penetration": torch.relu(-clearance),
                "occupied_fraction": occupied_fraction,
                "unknown_fraction": torch.zeros_like(clearance),
                "outside_fraction": outside_fraction}

    @property
    def body_width_m(self):
        return float(self.parts[:, 3].max() - self.parts[:, 2].min())

    def body_at(self, pose):
        x, y, yaw = np.asarray(pose, dtype=float)[:3]
        rotated = affinity.rotate(self.body, float(yaw), origin=(0, 0), use_radians=True)
        return affinity.translate(rotated, float(x), float(y))

    def clearance(self, pose):
        return float(self.body_at(pose).distance(self.obstacle_geometry))

    def completely_passed(self, pose):
        return self.body_at(pose).bounds[0] > self.scene.exit_x_m + 1.0e-6

    def validate_poses(self, poses) -> SweepResult:
        poses = np.asarray(poses, dtype=float).reshape(-1, 3)
        if len(poses) == 0 or not np.isfinite(poses).all():
            return SweepResult(False, "INVALID_PREDICTION", 0.0)
        polygons, margins, x, y = swept_part_polygons(
            self.parts, poses, radius_m=self.radius_m, margin_m=self.margin_m, step_m=self.sweep_step_m,
        )
        distances = shapely.distance(polygons, self.obstacle_geometry).reshape(len(margins), -1).min(axis=1)
        room_gap = self.scene.half_size_m - np.maximum(np.abs(x).max(axis=(1, 2)), np.abs(y).max(axis=(1, 2)))
        outside = room_gap < margins
        blocked = distances <= margins + 1.0e-9
        bad = np.flatnonzero(outside | blocked)
        if len(bad):
            index = int(bad[0])
            return SweepResult(False, "OUTSIDE_SCENE" if outside[index] else "OBSTACLE_OR_MARGIN", float(distances[:index + 1].min()))
        return SweepResult(True, "SAFE", float(distances.min()))

    def torch_clearance(self, poses: torch.Tensor):
        key = (poses.device, poses.dtype)
        if key not in self._tensor_cache:
            parts = torch.as_tensor(self.parts, device=poses.device, dtype=poses.dtype)
            obstacles = torch.as_tensor(self.scene.obstacles, device=poses.device, dtype=poses.dtype)
            self._tensor_cache[key] = (
                torch.stack(((parts[:, 0] + parts[:, 1]) / 2, (parts[:, 2] + parts[:, 3]) / 2), dim=-1),
                torch.stack(((parts[:, 1] - parts[:, 0]) / 2, (parts[:, 3] - parts[:, 2]) / 2), dim=-1),
                torch.stack(((obstacles[:, 0] + obstacles[:, 1]) / 2, (obstacles[:, 2] + obstacles[:, 3]) / 2), dim=-1),
                torch.stack(((obstacles[:, 1] - obstacles[:, 0]) / 2, (obstacles[:, 3] - obstacles[:, 2]) / 2), dim=-1),
            )
        centers, half, oc, oh = self._tensor_cache[key]
        c, s = torch.cos(poses[..., 2]), torch.sin(poses[..., 2])
        wx = poses[..., 0, None] + c[..., None] * centers[:, 0] - s[..., None] * centers[:, 1]
        wy = poses[..., 1, None] + s[..., None] * centers[:, 0] + c[..., None] * centers[:, 1]
        dx = wx[..., :, None] - oc[:, 0]
        dy = wy[..., :, None] - oc[:, 1]
        c, s = c[..., None, None], s[..., None, None]
        hx, hy = half[:, 0, None], half[:, 1, None]
        ox, oy = oh[:, 0], oh[:, 1]
        separation = torch.stack((
            dx.abs() - (c.abs() * hx + s.abs() * hy + ox),
            dy.abs() - (s.abs() * hx + c.abs() * hy + oy),
            (c * dx + s * dy).abs() - (hx + c.abs() * ox + s.abs() * oy),
            (-s * dx + c * dy).abs() - (hy + s.abs() * ox + c.abs() * oy),
        ), dim=-1).amax(dim=-1)
        obstacle_gap = separation.amin(dim=(-1, -2))
        extent_x = c[..., 0].abs() * half[:, 0] + s[..., 0].abs() * half[:, 1]
        extent_y = s[..., 0].abs() * half[:, 0] + c[..., 0].abs() * half[:, 1]
        room_gap = torch.minimum(
            self.scene.half_size_m - wx.abs() - extent_x,
            self.scene.half_size_m - wy.abs() - extent_y,
        ).amin(dim=-1)
        return torch.minimum(obstacle_gap, room_gap)
