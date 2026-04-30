from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TargetMemoryState:
    seen_count: int = 0
    lost_count: int = 0
    last_seen_world_xy: list[float] | None = None
    last_seen_bearing_hint: str = "unknown"
    last_status: str = "target has not been observed yet"


class TargetMemory:
    def __init__(self, *, target_text: str) -> None:
        self._target_text = str(target_text).strip() or "navigation target"
        self._state = TargetMemoryState()

    @property
    def state(self) -> TargetMemoryState:
        return self._state

    @property
    def target_text(self) -> str:
        return self._target_text

    def reset_for_target(self, target_text: str) -> None:
        self._target_text = str(target_text).strip() or "navigation target"
        self._state = TargetMemoryState()

    def summary(self, *, robot_pose) -> str:
        if self._state.last_seen_world_xy is None:
            return (
                f"target={self._target_text}; status=not_seen; "
                "no spatial anchor yet; keep searching according to the instruction."
            )

        local_forward_m, local_left_m = _world_to_local_xy(
            robot_pose=robot_pose,
            world_xy=self._state.last_seen_world_xy,
        )
        distance_m = float(np.hypot(local_forward_m, local_left_m))
        bearing_deg = float(np.degrees(np.arctan2(local_left_m, max(local_forward_m, 1e-6))))
        side = _bearing_to_side(local_forward_m=local_forward_m, local_left_m=local_left_m)
        return (
            f"target={self._target_text}; status=memory_only_last_seen_not_current_visibility; "
            f"seen_count={self._state.seen_count}; lost_count={self._state.lost_count}; "
            f"last_seen_image_bearing={self._state.last_seen_bearing_hint}; "
            f"estimated_relative_to_robot_now={side}, distance={distance_m:.2f}m, bearing={bearing_deg:+.1f}deg; "
            "Do not treat this memory as current image visibility. "
            f"last_status={self._state.last_status}"
        )

    def distance_to_anchor(self, *, robot_pose) -> float | None:
        if self._state.last_seen_world_xy is None:
            return None
        local_forward_m, local_left_m = _world_to_local_xy(
            robot_pose=robot_pose,
            world_xy=self._state.last_seen_world_xy,
        )
        return float(np.hypot(local_forward_m, local_left_m))

    def update_from_decision(
        self,
        *,
        target_visible: bool | None,
        bearing_hint: str,
        status: str,
        selected_world_goal_xy: list[float] | None,
    ) -> None:
        visible = bool(target_visible)
        if visible and selected_world_goal_xy is not None:
            self._state.seen_count += 1
            self._state.lost_count = 0
            self._state.last_seen_world_xy = [float(selected_world_goal_xy[0]), float(selected_world_goal_xy[1])]
            self._state.last_seen_bearing_hint = _sanitize_bearing_hint(bearing_hint)
            self._state.last_status = str(status).strip() or "target visible in current image"
            return

        if self._state.last_seen_world_xy is not None:
            self._state.lost_count += 1
            if status:
                self._state.last_status = str(status).strip()


def _world_to_local_xy(*, robot_pose, world_xy: list[float]) -> tuple[float, float]:
    delta_world = np.asarray(world_xy, dtype=np.float32) - np.asarray(
        [float(robot_pose.x), float(robot_pose.y)],
        dtype=np.float32,
    )
    cos_yaw = float(np.cos(robot_pose.yaw_rad))
    sin_yaw = float(np.sin(robot_pose.yaw_rad))
    forward_m = cos_yaw * float(delta_world[0]) + sin_yaw * float(delta_world[1])
    left_m = -sin_yaw * float(delta_world[0]) + cos_yaw * float(delta_world[1])
    return float(forward_m), float(left_m)


def _bearing_to_side(*, local_forward_m: float, local_left_m: float) -> str:
    if local_forward_m < -0.2:
        return "behind-left" if local_left_m > 0.2 else "behind-right" if local_left_m < -0.2 else "behind"
    if abs(local_left_m) <= 0.25:
        return "front"
    return "left" if local_left_m > 0.0 else "right"


def _sanitize_bearing_hint(value: str) -> str:
    candidate = str(value).strip().lower().replace("-", "_")
    if candidate in {"left", "right", "center", "front", "behind", "unknown"}:
        return candidate
    return "unknown"
