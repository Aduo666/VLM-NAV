"""Conservative execution verification against explicit 3-D occupancy."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from vln_nav.active_perception.contracts import (
    ExecutionAssessment,
    ExecutionVerification,
    TrajectoryVerificationRequest,
)
from vln_nav.active_perception.local_geometry import LocalVoxelSnapshot, OCCUPIED, UNKNOWN


@dataclass(frozen=True)
class BaseEnvelope:
    front_m: float
    rear_m: float
    half_width_m: float
    min_z_m: float
    max_z_m: float
    margin_m: float = 0.04

    def validate(self) -> None:
        values = (self.front_m, self.rear_m, self.half_width_m, self.max_z_m - self.min_z_m)
        if not all(math.isfinite(float(value)) and float(value) > 0 for value in values):
            raise ValueError("Base envelope dimensions must be positive")
        if not math.isfinite(self.margin_m) or self.margin_m < 0:
            raise ValueError("Base envelope margin must be nonnegative")


class TrajectoryExecutionVerifier:
    def __init__(self, envelope: BaseEnvelope) -> None:
        envelope.validate()
        self.envelope = envelope

    def verify(
        self,
        request: TrajectoryVerificationRequest,
        snapshot: LocalVoxelSnapshot,
        *,
        trusted_current_pose: tuple[float, float, float] | None = None,
    ) -> ExecutionVerification:
        request.validate()
        checked: list[np.ndarray] = []
        states: list[np.ndarray] = []
        for index, sample in enumerate(request.samples):
            points = self._envelope_points(sample.x_m, sample.y_m, sample.yaw_rad, snapshot.resolution_m)
            state = snapshot.query(points)
            # The accepted initial placement may exempt unknown at the first
            # pose only. A measured obstacle is never erased or ignored.
            if index == 0 and trusted_current_pose is not None:
                tx, ty, tyaw = map(float, trusted_current_pose)
                if math.hypot(sample.x_m - tx, sample.y_m - ty) <= snapshot.resolution_m and abs(
                    _wrap(sample.yaw_rad - tyaw)
                ) <= math.radians(2):
                    state = np.where(state == UNKNOWN, 0, state)
            checked.append(points)
            states.append(state)
        points = np.concatenate(checked, axis=0)
        state = np.concatenate(states, axis=0)
        occupied_indices = np.flatnonzero(state == OCCUPIED)
        unknown_indices = np.flatnonzero(state == UNKNOWN)
        if len(occupied_indices):
            assessment = ExecutionAssessment.BLOCKED
            reason = "occupied_voxel_in_swept_volume"
            blocker = tuple(map(float, points[occupied_indices[0]]))
        elif len(unknown_indices):
            assessment = ExecutionAssessment.UNCERTAIN
            reason = "missing_evidence_in_swept_volume"
            blocker = None
        else:
            assessment = ExecutionAssessment.EXECUTABLE
            reason = "swept_volume_observed_free"
            blocker = None
        critical = points[unknown_indices]
        if len(critical) > 5000:
            critical = critical[np.linspace(0, len(critical) - 1, 5000).astype(int)]
        return ExecutionVerification(
            request_id=request.request_id,
            assessment=assessment,
            reason=reason,
            map_version=snapshot.version,
            checked_voxel_count=int(len(state)),
            unknown_voxel_count=int(len(unknown_indices)),
            occupied_voxel_count=int(len(occupied_indices)),
            critical_unknown_world_xyz=critical.copy(),
            blocker_world_xyz=blocker,
        )

    def _envelope_points(self, x_m: float, y_m: float, yaw_rad: float, resolution_m: float) -> np.ndarray:
        e = self.envelope
        step = float(resolution_m)
        local_x = np.arange(-e.rear_m - e.margin_m, e.front_m + e.margin_m + step * 0.5, step)
        local_y = np.arange(-e.half_width_m - e.margin_m, e.half_width_m + e.margin_m + step * 0.5, step)
        local_z = np.arange(e.min_z_m, e.max_z_m + step * 0.5, step)
        xx, yy, zz = np.meshgrid(local_x, local_y, local_z, indexing="ij")
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        world_x = x_m + c * xx - s * yy
        world_y = y_m + s * xx + c * yy
        return np.column_stack((world_x.ravel(), world_y.ravel(), zz.ravel()))


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
