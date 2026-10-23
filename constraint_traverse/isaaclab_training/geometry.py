from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull


def footprint_points(records):
    return np.concatenate([np.array([[x, y] for x in (r["envelope_min_base_xyz_m"][0], r["envelope_max_base_xyz_m"][0])
        for y in (r["envelope_min_base_xyz_m"][1], r["envelope_max_base_xyz_m"][1])])
        for r in records if r["enabled"]])


@dataclass
class DoorTracker:
    center: np.ndarray
    normal: np.ndarray
    width: float
    thickness: float
    body_points: np.ndarray
    crossed: bool = False
    passed: bool = False
    previous: np.ndarray | None = None
    armed: bool = False

    @classmethod
    def create(cls, data, points, pose):
        center = np.asarray(data["center_world_xy"], dtype=float).reshape(2)
        normal = np.asarray(data["normal_world_xy"], dtype=float).reshape(2)
        width, thickness = float(data["width_m"]), float(data["thickness_m"])
        if not np.isfinite([*center, *normal, width, thickness]).all() or np.linalg.norm(normal) < 1e-8 or width <= 0 or thickness <= 0:
            raise ValueError("门洞标注参数无效")
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError("车体轮廓参数无效")
        if np.linalg.matrix_rank(points - points[0]) < 2:
            raise ValueError("车体轮廓必须具有面积")
        polygon = points[ConvexHull(points).vertices]
        tracker = cls(center, normal / np.linalg.norm(normal), width, thickness, polygon)
        tracker.previous = np.asarray(pose, dtype=float).reshape(3).copy()
        if not np.isfinite(tracker.previous).all():
            raise ValueError("机器人位姿包含非有限值")
        tracker._advance(tracker.previous)
        return tracker

    def update(self, pose):
        pose = np.asarray(pose, dtype=float).reshape(3)
        if not np.isfinite(pose).all():
            raise ValueError("机器人位姿包含非有限值")
        delta = pose - self.previous
        delta[2] = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
        travel = np.linalg.norm(delta[:2]) + np.linalg.norm(self.body_points, axis=1).max() * abs(delta[2])
        # 检查控制周期内的位姿变化，车体边缘每次最多前进约一厘米。
        count = max(1, int(np.ceil(travel / 0.01)))
        for fraction in np.linspace(0, 1, count + 1)[1:]:
            self._advance(self.previous + fraction * delta)
        self.previous = pose.copy()

    def _advance(self, pose):
        if self.passed:
            return
        c, s = np.cos(pose[2]), np.sin(pose[2])
        points = self.body_points @ np.array([[c, s], [-s, c]]) + pose[:2]
        normal = (points - self.center) @ self.normal
        half = self.thickness / 2
        if normal.max() <= -half:
            self.armed = True
            self.crossed = False
            return
        if not self.armed:
            return
        lateral = (points - self.center) @ np.array([-self.normal[1], self.normal[0]])
        section = list(lateral[(normal >= -half) & (normal <= half)])
        following_normal = np.roll(normal, -1)
        following_lateral = np.roll(lateral, -1)
        for plane in (-half, half):
            intersects = (normal - plane) * (following_normal - plane) < 0
            fraction = (plane - normal[intersects]) / (following_normal[intersects] - normal[intersects])
            section.extend(lateral[intersects] + fraction * (following_lateral[intersects] - lateral[intersects]))
        if section and np.max(np.abs(section)) > self.width / 2 + 1e-9:
            self.armed = False
            self.crossed = False
            return
        self.crossed |= np.dot(pose[:2] - self.center, self.normal) >= 0
        self.passed = self.crossed and bool(np.all(normal > half))
