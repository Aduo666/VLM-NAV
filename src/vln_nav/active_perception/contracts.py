"""Stable contracts between navigation, geometry, and active observation."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np


class ExecutionAssessment(str, Enum):
    """A three-state result; uncertainty must never be treated as collision."""

    EXECUTABLE = "EXECUTABLE"
    BLOCKED = "BLOCKED"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class TimedDepthObservation:
    source: str
    timestamp_sec: float
    depth_m: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray

    def validate(self) -> None:
        depth = np.asarray(self.depth_m)
        if not self.source.strip() or not math.isfinite(float(self.timestamp_sec)):
            raise ValueError("Depth observation requires a source and finite timestamp")
        if depth.ndim != 2 or depth.size == 0:
            raise ValueError(f"Depth observation must be HxW, got {depth.shape}")
        if np.asarray(self.intrinsics).shape != (3, 3):
            raise ValueError("Camera intrinsics must be 3x3")
        transform = np.asarray(self.camera_to_world)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("camera_to_world must be a finite 4x4 transform")


@dataclass(frozen=True)
class TrajectorySample:
    """One nominal base state in a fixed, Z-up world frame."""

    time_sec: float
    x_m: float
    y_m: float
    yaw_rad: float
    linear_mps: float = 0.0
    angular_radps: float = 0.0

    def validate(self) -> None:
        if not all(math.isfinite(float(value)) for value in (
            self.time_sec,
            self.x_m,
            self.y_m,
            self.yaw_rad,
            self.linear_mps,
            self.angular_radps,
        )):
            raise ValueError("Trajectory samples must contain finite values")


@dataclass(frozen=True)
class TrajectoryVerificationRequest:
    request_id: str
    frame_id: str
    samples: tuple[TrajectorySample, ...]
    map_version: int

    def validate(self) -> None:
        if not self.request_id.strip() or not self.frame_id.strip():
            raise ValueError("Trajectory request requires request_id and frame_id")
        if len(self.samples) < 1:
            raise ValueError("Trajectory request must contain at least one sample")
        for sample in self.samples:
            sample.validate()
        times = [float(sample.time_sec) for sample in self.samples]
        if any(next_time < time for time, next_time in zip(times, times[1:])):
            raise ValueError("Trajectory sample times must be monotonic")
        if int(self.map_version) < 0:
            raise ValueError("map_version must be nonnegative")


@dataclass(frozen=True)
class ExecutionVerification:
    request_id: str
    assessment: ExecutionAssessment
    reason: str
    map_version: int
    checked_voxel_count: int
    unknown_voxel_count: int
    occupied_voxel_count: int
    critical_unknown_world_xyz: np.ndarray
    blocker_world_xyz: tuple[float, float, float] | None = None

    @property
    def needs_active_observation(self) -> bool:
        return self.assessment is ExecutionAssessment.UNCERTAIN


@dataclass(frozen=True)
class ArmMotionVerification:
    assessment: ExecutionAssessment
    reason: str
    checked_state_count: int
    unknown_sample_count: int
    occupied_sample_count: int
    blocker_world_xyz: tuple[float, float, float] | None = None
