from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from vln_nav.active_perception.camera_geometry import (
    CameraFrameGeometry,
    camera_rgb_to_uint8,
    world_from_pose_wxyz,
)

from .recorder import quantize_depth_m


@dataclass(frozen=True)
class PhysicsState:
    timestamp_sec: float
    physics_step: int
    position: np.ndarray
    orientation: np.ndarray
    pose: np.ndarray
    velocity: np.ndarray
    steering_position: np.ndarray
    steering_velocity: np.ndarray
    wheel_velocity: np.ndarray
    front_world_from_camera: np.ndarray
    down_world_from_camera: np.ndarray


class PhysicsHistory:
    def __init__(self, max_steps: int = 600) -> None:
        self._states: deque[PhysicsState] = deque(maxlen=max_steps)

    def clear(self) -> None:
        self._states.clear()

    def append(self, state: PhysicsState) -> None:
        if self._states and state.timestamp_sec < self._states[-1].timestamp_sec - 1.0e-7:
            self.clear()
        if self._states and abs(state.timestamp_sec - self._states[-1].timestamp_sec) <= 1.0e-7:
            self._states.pop()
        self._states.append(state)

    def at(self, timestamp_sec: float) -> PhysicsState | None:
        return next((
            state for state in reversed(self._states)
            if abs(state.timestamp_sec - timestamp_sec) <= 1.0e-6
        ), None)


@dataclass(frozen=True)
class AlignedFramePair:
    front: CameraFrameGeometry
    down: CameraFrameGeometry
    state: PhysicsState


def pair_aligned_frames(
    front: tuple[CameraFrameGeometry, PhysicsState] | None,
    down: tuple[CameraFrameGeometry, PhysicsState] | None,
    *, max_skew_sec: float, not_before_sec: float = -float("inf"),
) -> AlignedFramePair | None:
    if front is None or down is None:
        return None
    front_frame, front_state = front
    down_frame, down_state = down
    if abs(front_frame.timestamp_sec - down_frame.timestamp_sec) > max_skew_sec + 1.0e-9:
        return None
    if min(front_frame.timestamp_sec, down_frame.timestamp_sec) < not_before_sec - 1.0e-6:
        return None
    state = front_state if front_frame.timestamp_sec >= down_frame.timestamp_sec else down_state
    return AlignedFramePair(front_frame, down_frame, state)


def capture_aligned_frame(
    *, source: str, camera, history: PhysicsHistory, after_sec: float,
) -> tuple[CameraFrameGeometry, PhysicsState] | None:
    frame = camera.get_current_frame()
    timestamp = float(frame.get("rendering_time", np.nan))
    if not np.isfinite(timestamp) or timestamp <= after_sec + 1.0e-9:
        return None
    state = history.at(timestamp)
    if state is None:
        return None
    # RGB 与深度取自同一次相机缓存更新。
    frame = camera.get_current_frame(clone=True)
    if abs(float(frame["rendering_time"]) - timestamp) > 1.0e-9:
        return None
    rgb = camera_rgb_to_uint8(frame.get("rgb"))
    depth = frame.get("distance_to_image_plane")
    if rgb is None or depth is None:
        return None
    world_from_base = world_from_pose_wxyz(state.position, state.orientation)
    world_from_camera = (
        state.front_world_from_camera if source == "front" else state.down_world_from_camera
    )
    result = CameraFrameGeometry(
        source=source,
        frame_index=state.physics_step,
        timestamp_sec=timestamp,
        rgb=rgb.copy(),
        depth_m=quantize_depth_m(depth),
        intrinsics=np.asarray(camera.get_intrinsics_matrix(), dtype=np.float64).copy(),
        base_from_camera=np.linalg.inv(world_from_base) @ world_from_camera,
        world_from_camera=world_from_camera.copy(),
        world_from_base=world_from_base,
    )
    result.validate()
    return result, state
