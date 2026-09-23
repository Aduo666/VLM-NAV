from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class EpisodeMode(str, Enum):
    IDLE = "IDLE"
    RECORDING = "RECORDING"


@dataclass
class EpisodeSession:
    mode: EpisodeMode = EpisodeMode.IDLE
    start_pose_world: np.ndarray | None = None
    start_position_world_xyz: np.ndarray | None = None
    last_start_pose_world: np.ndarray | None = None
    last_start_position_world_xyz: np.ndarray | None = None
    last_goal_pose_world: np.ndarray | None = None
    started_at_sec: float | None = None
    sample_count: int = 0
    warmup_pairs_remaining: int = 0
    trajectory_world_xy: list[tuple[float, float]] = field(default_factory=list)

    def begin(
        self,
        *,
        start_pose_world: np.ndarray,
        start_position_world_xyz: np.ndarray,
        timestamp_sec: float,
        warmup_pairs: int,
    ) -> None:
        if self.mode is EpisodeMode.RECORDING:
            raise RuntimeError("An episode is already recording")
        pose = np.asarray(start_pose_world, dtype=np.float64).reshape(3).copy()
        position = np.asarray(start_position_world_xyz, dtype=np.float64).reshape(3).copy()
        self.mode = EpisodeMode.RECORDING
        self.start_pose_world = pose
        self.start_position_world_xyz = position
        self.last_start_pose_world = pose.copy()
        self.last_start_position_world_xyz = position.copy()
        self.last_goal_pose_world = None
        self.started_at_sec = float(timestamp_sec)
        self.sample_count = 0
        self.warmup_pairs_remaining = max(0, int(warmup_pairs))
        self.trajectory_world_xy = [(float(pose[0]), float(pose[1]))]

    def note_sample(self, pose_world_xyyaw: np.ndarray) -> None:
        if self.mode is not EpisodeMode.RECORDING:
            return
        pose = np.asarray(pose_world_xyyaw, dtype=np.float64).reshape(3)
        self.sample_count += 1
        self.trajectory_world_xy.append((float(pose[0]), float(pose[1])))

    def consume_warmup_pair(self) -> bool:
        if self.warmup_pairs_remaining <= 0:
            return False
        self.warmup_pairs_remaining -= 1
        return True

    def complete(self, *, goal_pose_world: np.ndarray) -> None:
        if self.mode is not EpisodeMode.RECORDING:
            raise RuntimeError("No recording episode to complete")
        self.last_goal_pose_world = (
            np.asarray(goal_pose_world, dtype=np.float64).reshape(3).copy()
        )
        self.mode = EpisodeMode.IDLE

    def abort(self) -> None:
        self.mode = EpisodeMode.IDLE
        self.start_pose_world = None
        self.start_position_world_xyz = None
        self.started_at_sec = None
        self.sample_count = 0
        self.warmup_pairs_remaining = 0
        self.trajectory_world_xy = []

    @property
    def recording(self) -> bool:
        return self.mode is EpisodeMode.RECORDING

    def elapsed(self, now_sec: float) -> float:
        if self.started_at_sec is None:
            return 0.0
        return max(0.0, float(now_sec) - self.started_at_sec)
