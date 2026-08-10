from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .observation_adapter import Pose2D
from .occupancy_grid import OccupancyGridSnapshot


@dataclass(frozen=True)
class PrimitiveCollisionCheckerConfig:
    forward_distance_m: float
    turn_angle_deg: float
    max_clearance_search_m: float = 2.0


@dataclass(frozen=True)
class PrimitiveCollisionResult:
    action_name: str
    valid: bool
    reason: str
    raw_clearance_m: float
    known_free_ratio: float
    sampled_pose_count: int
    travel_distance_m: float
    turn_angle_deg: float


class PrimitiveCollisionChecker:
    """Check a complete primitive center trajectory against an inflated map.

    ``inflated_occupied`` already encodes robot size and safety margin, so the
    sampled poses are checked as points. Raw occupancy is used only to report
    physical clearance and is never inflated a second time.
    """

    _SUPPORTED_ACTIONS = {"FORWARD", "BACKWARD", "TURN_LEFT", "TURN_RIGHT"}

    def __init__(self, *, config: PrimitiveCollisionCheckerConfig) -> None:
        self._config = config

    @property
    def forward_distance_m(self) -> float:
        return max(0.0, float(self._config.forward_distance_m))

    @property
    def turn_angle_deg(self) -> float:
        return max(0.0, float(self._config.turn_angle_deg))

    def check(
        self,
        *,
        action_name: str,
        robot_pose: Pose2D,
        snapshot: OccupancyGridSnapshot | None,
        travel_distance_m: float | None = None,
        turn_angle_deg: float | None = None,
    ) -> PrimitiveCollisionResult:
        action = str(action_name).strip().upper()
        distance_m = (
            self.forward_distance_m
            if travel_distance_m is None
            else max(0.0, float(travel_distance_m))
        )
        angle_deg = (
            self.turn_angle_deg
            if turn_angle_deg is None
            else max(0.0, float(turn_angle_deg))
        )
        if action not in self._SUPPORTED_ACTIONS:
            return self._result(
                action=action,
                valid=False,
                reason="unsupported_action",
                clearance_m=0.0,
                known_free_ratio=0.0,
                sample_count=0,
                distance_m=distance_m,
                angle_deg=angle_deg,
            )
        if snapshot is None:
            return self._result(
                action=action,
                valid=True,
                reason="map_unavailable",
                clearance_m=float("inf"),
                known_free_ratio=0.0,
                sample_count=0,
                distance_m=distance_m,
                angle_deg=angle_deg,
            )

        samples = self._sample_center_trajectory(
            action=action,
            robot_pose=robot_pose,
            resolution_m=float(snapshot.resolution_m),
            distance_m=distance_m,
            angle_deg=angle_deg,
        )
        cells: list[tuple[int, int]] = []
        raw_clearance_m = float(self._config.max_clearance_search_m)
        free_hits = 0
        for sample_index, (world_x, world_y, _yaw_rad) in enumerate(samples):
            cell = _world_to_cell(snapshot=snapshot, world_xy=(world_x, world_y))
            if cell is None:
                return self._result(
                    action=action,
                    valid=False,
                    reason="trajectory_outside_map",
                    clearance_m=0.0,
                    known_free_ratio=float(free_hits) / float(max(1, len(cells))),
                    sample_count=len(samples),
                    distance_m=distance_m,
                    angle_deg=angle_deg,
                )
            cells.append(cell)
            row, col = cell
            raw_clearance_m = min(
                raw_clearance_m,
                _raw_clearance_m(
                    snapshot=snapshot,
                    cell=cell,
                    max_radius_m=float(self._config.max_clearance_search_m),
                ),
            )
            if bool(snapshot.free_mask[row, col]):
                free_hits += 1

            # The current physical pose is known to be occupied safely by the
            # robot. Ignore only that exact starting cell; all newly entered
            # cells must be clear in the inflated map.
            if sample_index == 0:
                if bool(snapshot.occupied_mask[row, col]):
                    return self._result(
                        action=action,
                        valid=False,
                        reason="current_pose_raw_occupied",
                        clearance_m=raw_clearance_m,
                        known_free_ratio=float(free_hits) / float(max(1, len(cells))),
                        sample_count=len(samples),
                        distance_m=distance_m,
                        angle_deg=angle_deg,
                    )
                continue
            if cell == cells[0]:
                continue
            if bool(snapshot.inflated_occupied[row, col]):
                return self._result(
                    action=action,
                    valid=False,
                    reason="swept_path_collision",
                    clearance_m=raw_clearance_m,
                    known_free_ratio=float(free_hits) / float(max(1, len(cells))),
                    sample_count=len(samples),
                    distance_m=distance_m,
                    angle_deg=angle_deg,
                )

        return self._result(
            action=action,
            valid=True,
            reason="clear",
            clearance_m=raw_clearance_m,
            known_free_ratio=float(free_hits) / float(max(1, len(cells))),
            sample_count=len(samples),
            distance_m=distance_m,
            angle_deg=angle_deg,
        )

    @staticmethod
    def _sample_center_trajectory(
        *,
        action: str,
        robot_pose: Pose2D,
        resolution_m: float,
        distance_m: float,
        angle_deg: float,
    ) -> list[tuple[float, float, float]]:
        if action in {"FORWARD", "BACKWARD"}:
            signed_distance = float(distance_m) * (1.0 if action == "FORWARD" else -1.0)
            sample_step_m = max(0.02, min(0.08, float(resolution_m) * 0.5))
            count = max(1, int(np.ceil(abs(signed_distance) / sample_step_m)))
            offsets = np.linspace(0.0, signed_distance, count + 1)
            cos_yaw = float(np.cos(robot_pose.yaw_rad))
            sin_yaw = float(np.sin(robot_pose.yaw_rad))
            return [
                (
                    float(robot_pose.x) + cos_yaw * float(offset),
                    float(robot_pose.y) + sin_yaw * float(offset),
                    float(robot_pose.yaw_rad),
                )
                for offset in offsets
            ]

        signed_angle_rad = float(np.deg2rad(angle_deg))
        if action == "TURN_RIGHT":
            signed_angle_rad = -signed_angle_rad
        count = max(1, int(np.ceil(abs(float(angle_deg)) / 3.0)))
        return [
            (
                float(robot_pose.x),
                float(robot_pose.y),
                float(robot_pose.yaw_rad) + float(delta),
            )
            for delta in np.linspace(0.0, signed_angle_rad, count + 1)
        ]

    @staticmethod
    def _result(
        *,
        action: str,
        valid: bool,
        reason: str,
        clearance_m: float,
        known_free_ratio: float,
        sample_count: int,
        distance_m: float,
        angle_deg: float,
    ) -> PrimitiveCollisionResult:
        return PrimitiveCollisionResult(
            action_name=action,
            valid=bool(valid),
            reason=str(reason),
            raw_clearance_m=float(clearance_m),
            known_free_ratio=float(np.clip(known_free_ratio, 0.0, 1.0)),
            sampled_pose_count=int(sample_count),
            travel_distance_m=float(distance_m if action in {"FORWARD", "BACKWARD"} else 0.0),
            turn_angle_deg=float(angle_deg if action in {"TURN_LEFT", "TURN_RIGHT"} else 0.0),
        )


def _world_to_cell(
    *,
    snapshot: OccupancyGridSnapshot,
    world_xy: tuple[float, float],
) -> tuple[int, int] | None:
    col = int(np.floor((float(world_xy[0]) - float(snapshot.origin_x)) / float(snapshot.resolution_m)))
    row = int(np.floor((float(world_xy[1]) - float(snapshot.origin_y)) / float(snapshot.resolution_m)))
    if row < 0 or row >= snapshot.log_odds.shape[0] or col < 0 or col >= snapshot.log_odds.shape[1]:
        return None
    return row, col


def _raw_clearance_m(
    *,
    snapshot: OccupancyGridSnapshot,
    cell: tuple[int, int],
    max_radius_m: float,
) -> float:
    resolution_m = max(1.0e-6, float(snapshot.resolution_m))
    max_cells = max(1, int(np.ceil(float(max_radius_m) / resolution_m)))
    row, col = cell
    row0 = max(0, row - max_cells)
    row1 = min(snapshot.occupied_mask.shape[0], row + max_cells + 1)
    col0 = max(0, col - max_cells)
    col1 = min(snapshot.occupied_mask.shape[1], col + max_cells + 1)
    occupied_rows, occupied_cols = np.nonzero(snapshot.occupied_mask[row0:row1, col0:col1])
    if occupied_rows.size == 0:
        return float(max_radius_m)
    delta_rows = occupied_rows.astype(np.float64) + float(row0 - row)
    delta_cols = occupied_cols.astype(np.float64) + float(col0 - col)
    return float(
        min(
            float(max_radius_m),
            float(np.min(np.hypot(delta_rows, delta_cols))) * resolution_m,
        )
    )
