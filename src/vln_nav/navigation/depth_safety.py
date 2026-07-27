from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .observation_adapter import VLNObservation


@dataclass
class DepthSafetyConfig:
    enabled: bool
    stop_distance_m: float
    roi_vertical_range: tuple[float, float]
    center_width_ratio: float
    side_width_ratio: float
    near_field_memory_enabled: bool = True
    near_field_memory_sec: float = 2.0


@dataclass
class DepthSafetyDecision:
    action_name: str
    speed_scale: float
    overridden: bool
    reason: str
    front_depth_m: float
    left_depth_m: float
    right_depth_m: float


class DepthSafetyMonitor:
    def __init__(self, *, config: DepthSafetyConfig) -> None:
        self._config = config
        self._recent_close_depths: dict[str, tuple[float, float]] = {
            "front": (float("inf"), -float("inf")),
            "left": (float("inf"), -float("inf")),
            "right": (float("inf"), -float("inf")),
        }

    def apply(
        self,
        *,
        observation: VLNObservation,
        action_name: str,
        speed_scale: float,
        local_goal_xy: list[float] | tuple[float, float] | None = None,
    ) -> DepthSafetyDecision:
        del local_goal_xy
        action = str(action_name).strip().upper()
        speed = max(0.0, float(speed_scale))
        if not self._config.enabled:
            return DepthSafetyDecision(
                action_name=action,
                speed_scale=speed,
                overridden=False,
                reason="disabled",
                front_depth_m=float("nan"),
                left_depth_m=float("nan"),
                right_depth_m=float("nan"),
            )

        depth = None if observation.depth is None else np.asarray(observation.depth, dtype=np.float32)
        if depth is None or depth.ndim != 2 or depth.size == 0:
            front_depth_m = left_depth_m = right_depth_m = float("inf")
        else:
            front_depth_m, left_depth_m, right_depth_m = self._region_depths(depth)

        lidar_meta = observation.meta.get("lidar", {})
        if isinstance(lidar_meta, dict) and bool(lidar_meta.get("enabled", False)):
            front_depth_m = _minimum_finite(front_depth_m, lidar_meta.get("front_depth_m"))
            left_depth_m = _minimum_finite(left_depth_m, lidar_meta.get("left_depth_m"))
            right_depth_m = _minimum_finite(right_depth_m, lidar_meta.get("right_depth_m"))

        stop_distance_m = float(self._config.stop_distance_m)
        now_sec = float(observation.meta.get("timestamp_sec", 0.0))
        front_depth_m, left_depth_m, right_depth_m = self._apply_near_field_memory(
            now_sec=now_sec,
            stop_distance_m=stop_distance_m,
            front_depth_m=front_depth_m,
            left_depth_m=left_depth_m,
            right_depth_m=right_depth_m,
        )

        if action in {"STOP", "SAFETY_HOLD"}:
            return self._decision(
                action_name=action,
                speed_scale=0.0,
                overridden=False,
                reason="stop_requested",
                front_depth_m=front_depth_m,
                left_depth_m=left_depth_m,
                right_depth_m=right_depth_m,
            )

        if action == "FORWARD":
            if front_depth_m <= stop_distance_m:
                return self._decision(
                    action_name="STOP",
                    speed_scale=0.0,
                    overridden=True,
                    reason="front_blocked_stop",
                    front_depth_m=front_depth_m,
                    left_depth_m=left_depth_m,
                    right_depth_m=right_depth_m,
                )
            if front_depth_m <= stop_distance_m + 0.25:
                return self._decision(
                    action_name="FORWARD",
                    speed_scale=min(speed, 0.55),
                    overridden=False,
                    reason="front_close_slow",
                    front_depth_m=front_depth_m,
                    left_depth_m=left_depth_m,
                    right_depth_m=right_depth_m,
                )

        if action == "TURN_LEFT":
            turn_block_distance_m = stop_distance_m + 0.08
            if (
                front_depth_m <= turn_block_distance_m
                or left_depth_m <= turn_block_distance_m
            ):
                return self._decision(
                    action_name="STOP",
                    speed_scale=0.0,
                    overridden=True,
                    reason="turn_left_blocked_safety_hold",
                    front_depth_m=front_depth_m,
                    left_depth_m=left_depth_m,
                    right_depth_m=right_depth_m,
                )

        if action == "TURN_RIGHT":
            turn_block_distance_m = stop_distance_m + 0.08
            if (
                front_depth_m <= turn_block_distance_m
                or right_depth_m <= turn_block_distance_m
            ):
                return self._decision(
                    action_name="STOP",
                    speed_scale=0.0,
                    overridden=True,
                    reason="turn_right_blocked_safety_hold",
                    front_depth_m=front_depth_m,
                    left_depth_m=left_depth_m,
                    right_depth_m=right_depth_m,
                )

        return self._decision(
            action_name=action,
            speed_scale=speed,
            overridden=False,
            reason="clear",
            front_depth_m=front_depth_m,
            left_depth_m=left_depth_m,
            right_depth_m=right_depth_m,
        )

    def _region_depths(self, depth: np.ndarray) -> tuple[float, float, float]:
        height, width = depth.shape[:2]
        top_ratio, bottom_ratio = self._config.roi_vertical_range
        v0 = max(0, min(height - 1, int(round(float(height) * float(top_ratio)))))
        v1 = max(v0 + 1, min(height, int(round(float(height) * float(bottom_ratio)))))
        center_half_width = max(
            4,
            int(round(width * float(self._config.center_width_ratio) * 0.5)),
        )
        side_width = max(4, int(round(width * float(self._config.side_width_ratio))))
        center_x = width // 2
        front_patch = depth[
            v0:v1,
            max(0, center_x - center_half_width) : min(width, center_x + center_half_width),
        ]
        left_patch = depth[v0:v1, :side_width]
        right_patch = depth[v0:v1, max(0, width - side_width) : width]
        return (
            _robust_patch_depth(front_patch),
            _robust_patch_depth(left_patch),
            _robust_patch_depth(right_patch),
        )

    def _apply_near_field_memory(
        self,
        *,
        now_sec: float,
        stop_distance_m: float,
        front_depth_m: float,
        left_depth_m: float,
        right_depth_m: float,
    ) -> tuple[float, float, float]:
        if not self._config.near_field_memory_enabled:
            return front_depth_m, left_depth_m, right_depth_m

        remember_distance_m = float(stop_distance_m) + 0.45
        current = {
            "front": float(front_depth_m),
            "left": float(left_depth_m),
            "right": float(right_depth_m),
        }
        for region, depth_m in current.items():
            if np.isfinite(depth_m) and depth_m <= remember_distance_m:
                self._recent_close_depths[region] = (depth_m, float(now_sec))

        ttl_sec = max(0.0, float(self._config.near_field_memory_sec))
        effective: dict[str, float] = {}
        for region, depth_m in current.items():
            remembered_depth_m, remembered_time_sec = self._recent_close_depths[region]
            if ttl_sec > 0.0 and now_sec - remembered_time_sec <= ttl_sec:
                effective[region] = min(depth_m, remembered_depth_m)
            else:
                effective[region] = depth_m
        return effective["front"], effective["left"], effective["right"]

    @staticmethod
    def _decision(
        *,
        action_name: str,
        speed_scale: float,
        overridden: bool,
        reason: str,
        front_depth_m: float,
        left_depth_m: float,
        right_depth_m: float,
    ) -> DepthSafetyDecision:
        return DepthSafetyDecision(
            action_name=action_name,
            speed_scale=float(speed_scale),
            overridden=bool(overridden),
            reason=reason,
            front_depth_m=float(front_depth_m),
            left_depth_m=float(left_depth_m),
            right_depth_m=float(right_depth_m),
        )


def _robust_patch_depth(patch: np.ndarray) -> float:
    patch = np.asarray(patch, dtype=np.float32)
    valid = patch[np.isfinite(patch) & (patch > 0.05)]
    if valid.size == 0:
        return float("inf")
    return float(np.percentile(valid, 20.0))


def _minimum_finite(first, second) -> float:
    values = []
    for value in (first, second):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            values.append(numeric)
    return min(values) if values else float("inf")
