from __future__ import annotations

from dataclasses import dataclass, fields
import math

import numpy as np

from vln_nav.active_perception.camera_geometry import (
    CameraFrameGeometry,
    backproject_depth_pixels_to_world,
)
@dataclass(frozen=True)
class S2GoalAdapterConfig:
    depth_patch_radius_px: int
    min_depth_m: float
    max_depth_m: float
    min_goal_distance_m: float
    max_goal_distance_m: float

    @classmethod
    def from_mapping(cls, data: dict) -> "S2GoalAdapterConfig":
        expected = {item.name for item in fields(cls)}
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError(f"internnav_s2.goal_adapter 必须包含这些字段：{sorted(expected)}")
        config = cls(
            depth_patch_radius_px=int(data["depth_patch_radius_px"]),
            min_depth_m=float(data["min_depth_m"]),
            max_depth_m=float(data["max_depth_m"]),
            min_goal_distance_m=float(data["min_goal_distance_m"]),
            max_goal_distance_m=float(data["max_goal_distance_m"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.depth_patch_radius_px < 0:
            raise ValueError("depth_patch_radius_px 不能为负数")
        values = (
            self.min_depth_m,
            self.max_depth_m,
            self.min_goal_distance_m,
            self.max_goal_distance_m,
        )
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("S2 goal adapter 数值配置必须为有限正数")
        if self.min_depth_m >= self.max_depth_m:
            raise ValueError("S2 depth 范围无效")
        if self.min_goal_distance_m >= self.max_goal_distance_m:
            raise ValueError("S2 goal 距离范围无效")


@dataclass(frozen=True)
class S2MetricGoal:
    request_id: int
    view: str
    pixel_original_vu: tuple[float, float]
    depth_m: float
    body_xy: tuple[float, float]
    world_xy: tuple[float, float]


class S2GoalAdapter:
    def __init__(self, config: S2GoalAdapterConfig) -> None:
        config.validate()
        self.config = config

    def convert(
        self,
        *,
        request_id: int,
        frame: CameraFrameGeometry,
        pixel_vu: tuple[float, float],
        pixel_space_hw: tuple[int, int],
    ) -> S2MetricGoal:
        frame.validate()
        model_height, model_width = map(int, pixel_space_hw)
        if model_height <= 0 or model_width <= 0:
            raise ValueError("S2 pixel space 无效")
        image_height, image_width = frame.depth_m.shape
        model_v, model_u = map(float, pixel_vu)
        original_v = (model_v + 0.5) * image_height / model_height - 0.5
        original_u = (model_u + 0.5) * image_width / model_width - 0.5
        center_v = int(np.clip(round(original_v), 0, image_height - 1))
        center_u = int(np.clip(round(original_u), 0, image_width - 1))
        radius = self.config.depth_patch_radius_px
        row_slice = slice(max(0, center_v - radius), min(image_height, center_v + radius + 1))
        column_slice = slice(max(0, center_u - radius), min(image_width, center_u + radius + 1))
        patch = np.asarray(frame.depth_m[row_slice, column_slice], dtype=np.float64)
        valid = (
            np.isfinite(patch)
            & (patch >= self.config.min_depth_m)
            & (patch <= self.config.max_depth_m)
        )
        if not np.any(valid):
            raise RuntimeError(
                f"S2 pixel 周围没有有效深度 view={frame.source} pixel={[center_v, center_u]}"
            )
        depth_m = float(np.median(patch[valid]))
        world_point = backproject_depth_pixels_to_world(
            pixel_u=np.asarray(original_u),
            pixel_v=np.asarray(original_v),
            depth_m=np.asarray(depth_m),
            camera_intrinsics=frame.intrinsics,
            camera_to_world=frame.world_from_camera,
        ).reshape(3)
        base_from_world = np.linalg.inv(np.asarray(frame.world_from_base, dtype=np.float64))
        point_body_h = base_from_world @ np.asarray(
            [world_point[0], world_point[1], world_point[2], 1.0],
            dtype=np.float64,
        )
        body_xy = point_body_h[:2]
        distance = float(np.linalg.norm(body_xy))
        if distance < self.config.min_goal_distance_m:
            raise RuntimeError(f"S2 metric goal 距离过近：{distance:.3f}m")
        if distance > self.config.max_goal_distance_m:
            scale = self.config.max_goal_distance_m / distance
            body_xy = body_xy * scale
            world_xy = _body_xy_to_world(body_xy, frame.world_from_base)
        else:
            world_xy = np.asarray(world_point[:2], dtype=np.float64)
        final_distance = float(np.linalg.norm(body_xy))
        if not self.config.min_goal_distance_m <= final_distance <= self.config.max_goal_distance_m:
            raise RuntimeError(f"S2 安全目标距离无效：{final_distance:.3f}m")
        return S2MetricGoal(
            request_id=int(request_id),
            view=str(frame.source),
            pixel_original_vu=(float(original_v), float(original_u)),
            depth_m=depth_m,
            body_xy=(float(body_xy[0]), float(body_xy[1])),
            world_xy=(float(world_xy[0]), float(world_xy[1])),
        )


def _body_xy_to_world(body_xy: np.ndarray, world_from_base: np.ndarray) -> np.ndarray:
    point = np.asarray([body_xy[0], body_xy[1], 0.0, 1.0], dtype=np.float64)
    return (np.asarray(world_from_base, dtype=np.float64) @ point)[:2]
