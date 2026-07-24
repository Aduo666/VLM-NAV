from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import Any

import numpy as np

from .candidate_waypoint_predictor import CandidateSet
from .observation_adapter import Pose2D, VLNObservation
from .occupancy_grid import OccupancyGridSnapshot
from .policy_helpers import candidate_primitive_action_name
from ..runtime.robot_profile import RobotProfile


@dataclass(frozen=True)
class PrimitiveSafetyDecision:
    action_name: str
    valid: bool
    risk_score: float
    reason: str
    source: str
    collision_points_local_xy: tuple[tuple[float, float], ...] = ()
    swept_polygons_local_xy: tuple[tuple[tuple[float, float], ...], ...] = ()


class PrimitiveSafetyEvaluator:
    """Evaluate semantic motion primitives against the configured robot body.

    Frame convention:
    - robot-local +X is forward and +Y is left;
    - current LiDAR hit points are already expressed in that robot frame;
    - occupancy cells are stored in the world frame and transformed back into
      the current robot frame before collision checks;
    - unknown occupancy cells are not obstacles.
    """

    def __init__(self, *, profile: RobotProfile, logger: logging.Logger) -> None:
        self._profile = profile
        self._logger = logger.getChild("primitive_safety")
        self._current_lidar_points = np.zeros((0, 2), dtype=np.float32)
        self._occupancy_snapshot: OccupancyGridSnapshot | None = None
        self._robot_pose: Pose2D | None = None
        self._last_decisions: dict[str, PrimitiveSafetyDecision] = {}

    @property
    def last_decisions(self) -> dict[str, PrimitiveSafetyDecision]:
        return dict(self._last_decisions)

    def update_environment(
        self,
        *,
        robot_pose: Pose2D,
        current_lidar_points_local_xy,
        occupancy_snapshot: OccupancyGridSnapshot | None,
    ) -> None:
        points = np.asarray(current_lidar_points_local_xy, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 2:
            points = np.zeros((0, 2), dtype=np.float32)
        else:
            points = points[:, :2]
            finite = np.all(np.isfinite(points), axis=1)
            points = points[finite]
        self._current_lidar_points = points
        self._occupancy_snapshot = occupancy_snapshot
        self._robot_pose = robot_pose

    def filter_candidates(
        self,
        *,
        observation: VLNObservation,
        candidate_set: CandidateSet,
    ) -> CandidateSet:
        del observation
        decisions: dict[str, PrimitiveSafetyDecision] = {}
        candidates = []
        for candidate in candidate_set.candidates:
            if not candidate.valid:
                candidates.append(candidate)
                continue
            action_name = candidate_primitive_action_name(candidate)
            decision = decisions.get(action_name)
            if decision is None:
                decision = self.evaluate(action_name=action_name)
                decisions[action_name] = decision
            support = dict(candidate.depth_support)
            support.update(
                {
                    "primitive_safety_valid": 1.0 if decision.valid else 0.0,
                    "primitive_safety_risk": float(decision.risk_score),
                    "primitive_safety_reason": decision.reason,
                    "primitive_safety_source": decision.source,
                    "primitive_action": action_name,
                }
            )
            candidates.append(
                replace(
                    candidate,
                    valid=bool(candidate.valid and decision.valid),
                    clearance_score=(
                        float(candidate.clearance_score)
                        if decision.valid
                        else 0.0
                    ),
                    depth_support=support,
                )
            )

        self._last_decisions = decisions
        valid_count = sum(1 for candidate in candidates if candidate.valid)
        debug = dict(candidate_set.predictor_debug)
        debug["primitive_safety"] = {
            action: {
                "valid": int(decision.valid),
                "risk": round(float(decision.risk_score), 3),
                "reason": decision.reason,
                "source": decision.source,
            }
            for action, decision in decisions.items()
        }
        return CandidateSet(
            candidates=candidates,
            valid_count=valid_count,
            predictor_debug=debug,
        )

    def evaluate(
        self,
        *,
        action_name: str,
        robot_pose: Pose2D | None = None,
    ) -> PrimitiveSafetyDecision:
        action = str(action_name).strip().upper()
        if action not in {"FORWARD", "TURN_LEFT", "TURN_RIGHT", "BACKWARD"}:
            return PrimitiveSafetyDecision(
                action_name=action,
                valid=action in {"STOP", "SAFETY_HOLD"},
                risk_score=0.0,
                reason="non_motion_action",
                source="none",
            )

        pose = robot_pose or self._robot_pose
        polygons = self._swept_footprint(action)
        lidar_collision = _collision_points(
            points=self._current_lidar_points,
            polygons=polygons,
            margin_m=self._effective_margin(action),
        )
        map_points = self._occupied_map_points_local(robot_pose=pose, polygons=polygons)
        map_collision = _collision_points(
            points=map_points,
            polygons=polygons,
            margin_m=self._effective_margin(action),
        )
        collisions = _unique_points(lidar_collision, map_collision)
        source = _collision_source(lidar_collision, map_collision)
        valid = len(collisions) == 0
        risk_score = 0.0 if valid else min(1.0, 0.35 + 0.08 * len(collisions))
        reason = "swept_footprint_clear" if valid else f"swept_footprint_collision:{source}"
        decision = PrimitiveSafetyDecision(
            action_name=action,
            valid=valid,
            risk_score=risk_score,
            reason=reason,
            source=source,
            collision_points_local_xy=tuple(
                (float(point[0]), float(point[1])) for point in collisions[:32]
            ),
            swept_polygons_local_xy=tuple(
                tuple((float(point[0]), float(point[1])) for point in polygon)
                for polygon in polygons
            ),
        )
        self._last_decisions[action] = decision
        if not valid:
            self._logger.warning(
                "[SAFETY_REJECT] robot=%s action=%s source=%s collisions=%s risk=%.2f",
                self._profile.name,
                action,
                source,
                len(collisions),
                risk_score,
            )
        return decision

    def _effective_margin(self, action_name: str) -> float:
        margin = float(self._profile.geometry.safety_margin_m)
        if action_name in {"TURN_LEFT", "TURN_RIGHT"}:
            margin += float(self._profile.motion_model.turn_drift_margin_m)
        return max(0.0, margin)

    def _swept_footprint(self, action_name: str) -> list[np.ndarray]:
        footprint = np.asarray(
            self._profile.geometry.footprint_polygon_xy_m,
            dtype=np.float32,
        )
        if action_name in {"FORWARD", "BACKWARD"}:
            distance = float(self._profile.primitives.forward_distance_m)
            if action_name == "BACKWARD":
                distance = -distance
            step_m = max(0.02, min(0.08, float(self._profile.geometry.safety_margin_m)))
            count = max(2, int(np.ceil(abs(distance) / step_m)) + 1)
            offsets = np.linspace(0.0, distance, count, dtype=np.float32)
            return [
                footprint + np.asarray([offset, 0.0], dtype=np.float32)
                for offset in offsets
            ]

        angle_deg = float(self._profile.primitives.turn_angle_deg)
        if action_name == "TURN_RIGHT":
            angle_deg = -angle_deg
        step_deg = max(0.5, float(self._profile.motion_model.sweep_resolution_deg))
        count = max(2, int(np.ceil(abs(angle_deg) / step_deg)) + 1)
        angles = np.deg2rad(np.linspace(0.0, angle_deg, count, dtype=np.float32))
        return [_rotate_points(footprint, float(angle)) for angle in angles]

    def _occupied_map_points_local(
        self,
        *,
        robot_pose: Pose2D | None,
        polygons: list[np.ndarray],
    ) -> np.ndarray:
        snapshot = self._occupancy_snapshot
        if snapshot is None or robot_pose is None:
            return np.zeros((0, 2), dtype=np.float32)
        occupied = np.asarray(snapshot.occupied_mask, dtype=bool)
        rows, cols = np.nonzero(occupied)
        if rows.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        world_x = float(snapshot.origin_x) + (
            cols.astype(np.float32) + 0.5
        ) * float(snapshot.resolution_m)
        world_y = float(snapshot.origin_y) + (
            rows.astype(np.float32) + 0.5
        ) * float(snapshot.resolution_m)
        dx = world_x - float(robot_pose.x)
        dy = world_y - float(robot_pose.y)
        cosine = float(np.cos(robot_pose.yaw_rad))
        sine = float(np.sin(robot_pose.yaw_rad))
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        points = np.stack([local_x, local_y], axis=1).astype(np.float32)

        all_vertices = np.concatenate(polygons, axis=0)
        radius = float(np.max(np.linalg.norm(all_vertices, axis=1)))
        radius += self._effective_margin("TURN_LEFT") + float(snapshot.resolution_m)
        nearby = np.linalg.norm(points, axis=1) <= radius
        return points[nearby]

def _rotate_points(points: np.ndarray, angle_rad: float) -> np.ndarray:
    cosine = float(np.cos(angle_rad))
    sine = float(np.sin(angle_rad))
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    return np.asarray(points, dtype=np.float32) @ rotation.T


def _collision_points(
    *,
    points: np.ndarray,
    polygons: list[np.ndarray],
    margin_m: float,
) -> np.ndarray:
    if points.size == 0 or not polygons:
        return np.zeros((0, 2), dtype=np.float32)
    result: list[np.ndarray] = []
    for polygon in polygons:
        mask = _points_inside_or_near_polygon(
            points=points,
            polygon=polygon,
            margin_m=margin_m,
        )
        if np.any(mask):
            result.append(points[mask])
    if not result:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(result, axis=0)


def _points_inside_or_near_polygon(
    *,
    points: np.ndarray,
    polygon: np.ndarray,
    margin_m: float,
) -> np.ndarray:
    point_count = points.shape[0]
    inside = np.zeros((point_count,), dtype=bool)
    x = points[:, 0]
    y = points[:, 1]
    j = polygon.shape[0] - 1
    for i in range(polygon.shape[0]):
        xi, yi = float(polygon[i, 0]), float(polygon[i, 1])
        xj, yj = float(polygon[j, 0]), float(polygon[j, 1])
        intersects = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi + 1.0e-12) + xi
        )
        inside ^= intersects
        j = i
    if margin_m <= 0.0:
        return inside

    near = np.zeros((point_count,), dtype=bool)
    for index in range(polygon.shape[0]):
        start = polygon[index]
        end = polygon[(index + 1) % polygon.shape[0]]
        edge = end - start
        edge_norm_sq = max(1.0e-12, float(np.dot(edge, edge)))
        relative = points - start
        projection = np.clip((relative @ edge) / edge_norm_sq, 0.0, 1.0)
        closest = start + projection[:, None] * edge
        distance_sq = np.sum((points - closest) ** 2, axis=1)
        near |= distance_sq <= float(margin_m) ** 2
    return inside | near


def _unique_points(*groups: np.ndarray) -> np.ndarray:
    valid = [group for group in groups if group.size]
    if not valid:
        return np.zeros((0, 2), dtype=np.float32)
    points = np.concatenate(valid, axis=0)
    quantized = np.round(points, decimals=3)
    _, indices = np.unique(quantized, axis=0, return_index=True)
    return points[np.sort(indices)]


def _collision_source(lidar_points: np.ndarray, map_points: np.ndarray) -> str:
    if lidar_points.size and map_points.size:
        return "current_lidar+occupancy"
    if lidar_points.size:
        return "current_lidar"
    if map_points.size:
        return "occupancy"
    return "none"
