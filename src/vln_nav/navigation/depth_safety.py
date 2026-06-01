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
        sanitized_action = str(action_name).strip().upper()
        safety_action = "FORWARD" if sanitized_action == "TRACK_TARGET" else sanitized_action
        base_speed_scale = max(0.0, float(speed_scale))
        if not self._config.enabled or observation.depth is None:
            return DepthSafetyDecision(
                action_name=sanitized_action,
                speed_scale=base_speed_scale,
                overridden=False,
                reason="disabled_or_no_depth",
                front_depth_m=float("nan"),
                left_depth_m=float("nan"),
                right_depth_m=float("nan"),
            )

        depth = np.asarray(observation.depth, dtype=np.float32)
        if depth.ndim != 2 or depth.size == 0:
            return DepthSafetyDecision(
                action_name=sanitized_action,
                speed_scale=base_speed_scale,
                overridden=False,
                reason="invalid_depth",
                front_depth_m=float("nan"),
                left_depth_m=float("nan"),
                right_depth_m=float("nan"),
            )

        front_depth_m, left_depth_m, right_depth_m = self._region_depths(depth)
        stop_distance_m = float(self._config.stop_distance_m)
        now_sec = float(observation.meta.get("timestamp_sec", 0.0))
        front_depth_m, left_depth_m, right_depth_m = self._apply_near_field_memory(
            now_sec=now_sec,
            stop_distance_m=stop_distance_m,
            front_depth_m=front_depth_m,
            left_depth_m=left_depth_m,
            right_depth_m=right_depth_m,
        )

        if safety_action == "STOP":
            return DepthSafetyDecision(
                action_name="STOP",
                speed_scale=0.0,
                overridden=False,
                reason="stop_requested",
                front_depth_m=front_depth_m,
                left_depth_m=left_depth_m,
                right_depth_m=right_depth_m,
            )

        if sanitized_action == "TRACK_TARGET":
            heading_rad = _local_goal_heading_rad(local_goal_xy)
            if front_depth_m <= stop_distance_m:
                side_depth_m = left_depth_m if heading_rad > 0.0 else right_depth_m
                side_name = "left" if heading_rad > 0.0 else "right"
                if abs(heading_rad) >= 0.25 and side_depth_m >= stop_distance_m + 0.05:
                    return DepthSafetyDecision(
                        action_name=sanitized_action,
                        speed_scale=min(base_speed_scale, 0.45),
                        overridden=False,
                        reason=f"front_close_{side_name}_escape",
                        front_depth_m=front_depth_m,
                        left_depth_m=left_depth_m,
                        right_depth_m=right_depth_m,
                    )
                return DepthSafetyDecision(
                    action_name="STOP",
                    speed_scale=0.0,
                    overridden=True,
                    reason="front_blocked_stop",
                    front_depth_m=front_depth_m,
                    left_depth_m=left_depth_m,
                    right_depth_m=right_depth_m,
                )

            if front_depth_m <= stop_distance_m + 0.35:
                return DepthSafetyDecision(
                    action_name=sanitized_action,
                    speed_scale=min(base_speed_scale, 0.60),
                    overridden=False,
                    reason="front_close_slow",
                    front_depth_m=front_depth_m,
                    left_depth_m=left_depth_m,
                    right_depth_m=right_depth_m,
                )

        if safety_action == "FORWARD":
            if front_depth_m <= stop_distance_m:
                return DepthSafetyDecision(
                    action_name="STOP",
                    speed_scale=0.0,
                    overridden=True,
                    reason="front_blocked_stop",
                    front_depth_m=front_depth_m,
                    left_depth_m=left_depth_m,
                    right_depth_m=right_depth_m,
                )
            if front_depth_m <= stop_distance_m + 0.25:
                return DepthSafetyDecision(
                    action_name="FORWARD",
                    speed_scale=min(base_speed_scale, 0.55),
                    overridden=False,
                    reason="front_close_slow",
                    front_depth_m=front_depth_m,
                    left_depth_m=left_depth_m,
                    right_depth_m=right_depth_m,
                )

        if safety_action == "TURN_LEFT":
            turn_block_distance_m = stop_distance_m + 0.08
            if front_depth_m <= turn_block_distance_m or left_depth_m <= turn_block_distance_m:
                if right_depth_m > turn_block_distance_m:
                    return DepthSafetyDecision(
                        action_name="TURN_RIGHT",
                        speed_scale=min(base_speed_scale, 0.45),
                        overridden=True,
                        reason="turn_left_blocked_redirect_right",
                        front_depth_m=front_depth_m,
                        left_depth_m=left_depth_m,
                        right_depth_m=right_depth_m,
                    )
                return DepthSafetyDecision(
                    action_name="TURN_RIGHT",
                    speed_scale=min(base_speed_scale, 0.30),
                    overridden=True,
                    reason="turn_left_blocked_both_sides",
                    front_depth_m=front_depth_m,
                    left_depth_m=left_depth_m,
                    right_depth_m=right_depth_m,
                )

        if safety_action == "TURN_RIGHT":
            turn_block_distance_m = stop_distance_m + 0.08
            if front_depth_m <= turn_block_distance_m or right_depth_m <= turn_block_distance_m:
                if left_depth_m > turn_block_distance_m:
                    return DepthSafetyDecision(
                        action_name="TURN_LEFT",
                        speed_scale=min(base_speed_scale, 0.45),
                        overridden=True,
                        reason="turn_right_blocked_redirect_left",
                        front_depth_m=front_depth_m,
                        left_depth_m=left_depth_m,
                        right_depth_m=right_depth_m,
                    )
                return DepthSafetyDecision(
                    action_name="TURN_LEFT",
                    speed_scale=min(base_speed_scale, 0.30),
                    overridden=True,
                    reason="turn_right_blocked_both_sides",
                    front_depth_m=front_depth_m,
                    left_depth_m=left_depth_m,
                    right_depth_m=right_depth_m,
                )

        return DepthSafetyDecision(
            action_name=sanitized_action,
            speed_scale=base_speed_scale,
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

        center_half_width = max(4, int(round(width * float(self._config.center_width_ratio) * 0.5)))
        side_width = max(4, int(round(width * float(self._config.side_width_ratio))))
        cx = width // 2

        front_patch = depth[v0:v1, max(0, cx - center_half_width) : min(width, cx + center_half_width)]
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
        if not bool(self._config.near_field_memory_enabled):
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
            remembered_depth_m, remembered_time_sec = self._recent_close_depths.get(
                region,
                (float("inf"), -float("inf")),
            )
            if ttl_sec > 0.0 and float(now_sec) - remembered_time_sec <= ttl_sec:
                effective[region] = min(float(depth_m), float(remembered_depth_m))
            else:
                effective[region] = float(depth_m)
        return effective["front"], effective["left"], effective["right"]


def _robust_patch_depth(patch: np.ndarray) -> float:
    patch = np.asarray(patch, dtype=np.float32)
    valid = patch[np.isfinite(patch) & (patch > 0.05)]
    if valid.size == 0:
        return float("inf")
    return float(np.percentile(valid, 20.0))


def _local_goal_heading_rad(local_goal_xy: list[float] | tuple[float, float] | None) -> float:
    if local_goal_xy is None or len(local_goal_xy) < 2:
        return 0.0
    forward_m = float(local_goal_xy[0])
    left_m = float(local_goal_xy[1])
    return float(np.arctan2(left_m, max(forward_m, 1e-6)))
