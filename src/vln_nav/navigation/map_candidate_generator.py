from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .candidate_waypoint_predictor import CandidateSet, WaypointCandidate
from .landmark_map import LandmarkMapEntry
from .landmark_tracker import LandmarkState
from .observation_adapter import Pose2D, VLNObservation
from .occupancy_grid import OccupancyGridSnapshot
from .primitive_collision_checker import PrimitiveCollisionChecker


@dataclass
class MapCandidateGeneratorConfig:
    max_candidates: int = 6
    min_distance_m: float = 0.55
    max_distance_m: float = 1.35
    frontier_bearings_deg: tuple[float, ...] = (35.0, 18.0, 0.0, -18.0, -35.0)
    frontier_distances_m: tuple[float, ...] = (0.75, 1.10, 1.35)
    min_clearance_m: float = 0.34
    unknown_gain_weight: float = 0.35
    semantic_weight: float = 0.45


class MapCandidateGenerator:
    """Generate map-aware candidates while preserving WaypointCandidate output.

    Depth-derived candidates provide geometric evidence for three executable
    primitive outcomes. The policy and reasoner consume the same candidate
    contract without depending on the internal map representation.
    """

    def __init__(
        self,
        *,
        config: MapCandidateGeneratorConfig,
        collision_checker: PrimitiveCollisionChecker,
        logger,
    ) -> None:
        self._config = config
        self._collision_checker = collision_checker
        self._logger = logger.getChild("map_candidate_generator")
        self._last_signature: tuple | None = None

    def augment(
        self,
        *,
        observation: VLNObservation,
        base_candidate_set: CandidateSet,
        occupancy_snapshot: OccupancyGridSnapshot | None,
        current_target: str,
        landmark_state: LandmarkState | None,
        landmark_entry: LandmarkMapEntry | None,
        progress_signal: str,
        safety_context_text: str,
    ) -> CandidateSet:
        ranked = self._primitive_candidates(
            observation=observation,
            base_candidate_set=base_candidate_set,
            occupancy_snapshot=occupancy_snapshot,
            current_target=current_target,
            landmark_state=landmark_state,
            landmark_entry=landmark_entry,
        )
        valid_count = sum(1 for candidate in ranked if candidate.valid)
        debug = dict(base_candidate_set.predictor_debug)
        debug["candidate_representation"] = "atomic_primitive_outcome"
        debug["map_candidate_count"] = int(len(ranked))
        debug["map_candidate_valid_count"] = int(valid_count)
        self._log_if_changed(
            current_target=current_target,
            ranked=ranked,
            valid_count=valid_count,
        )
        return CandidateSet(candidates=ranked, valid_count=int(valid_count), predictor_debug=debug)

    def _primitive_candidates(
        self,
        *,
        observation: VLNObservation,
        base_candidate_set: CandidateSet,
        occupancy_snapshot: OccupancyGridSnapshot | None,
        current_target: str,
        landmark_state: LandmarkState | None,
        landmark_entry: LandmarkMapEntry | None,
    ) -> list[WaypointCandidate]:
        """Represent the three executable outcomes, not arbitrary depth points."""

        primitive_specs = (
            (0, "TURN_LEFT", float(self._collision_checker.turn_angle_deg)),
            (1, "FORWARD", 0.0),
            (2, "TURN_RIGHT", -float(self._collision_checker.turn_angle_deg)),
        )
        distance_m = max(0.10, float(self._collision_checker.forward_distance_m))
        anchor_world_xy = _confirmed_anchor_world_xy(
            landmark_state=landmark_state,
            landmark_entry=landmark_entry,
        )
        candidates: list[WaypointCandidate] = []
        for slot_id, action, heading_deg in primitive_specs:
            heading_rad = float(np.deg2rad(heading_deg))
            local_goal = [
                float(distance_m * np.cos(heading_rad)),
                float(distance_m * np.sin(heading_rad)),
            ]
            depth_candidate = _nearest_depth_candidate(
                base_candidate_set=base_candidate_set,
                heading_deg=heading_deg,
            )
            depth_support = {} if depth_candidate is None else dict(depth_candidate.depth_support)
            depth_valid = bool(depth_candidate is not None and depth_candidate.valid)
            free_ratio = float(depth_support.get("free_ratio", 0.0))
            min_depth_m = float(depth_support.get("min_depth", 0.0))
            unknown_gain = 0.0
            min_clearance_m = min_depth_m
            occupancy_reason = "not_available"
            occupancy_valid = True

            collision_result = self._collision_checker.check(
                action_name=action,
                robot_pose=observation.pose,
                snapshot=occupancy_snapshot,
                travel_distance_m=distance_m,
                turn_angle_deg=abs(float(heading_deg)),
            )
            free_ratio = float(collision_result.known_free_ratio)
            unknown_gain = max(0.0, 1.0 - free_ratio)
            min_clearance_m = float(collision_result.raw_clearance_m)
            occupancy_valid = bool(collision_result.valid)
            occupancy_reason = str(collision_result.reason)
            if (
                action == "FORWARD"
                and occupancy_snapshot is not None
                and occupancy_valid
                and free_ratio < 0.50
            ):
                occupancy_valid = False
                occupancy_reason = "corridor_not_known_free"

            # Depth candidates remain useful evidence for scoring and prompts,
            # but collision validity has a single source: the map checker.
            valid = bool(occupancy_valid)
            predicted_world_xy = _local_to_world_xy(robot_pose=observation.pose, local_xy=local_goal)
            anchor_delta_m = None
            anchor_bearing_after_deg = None
            if anchor_world_xy is not None:
                current_distance = float(
                    np.hypot(
                        float(anchor_world_xy[0]) - float(observation.pose.x),
                        float(anchor_world_xy[1]) - float(observation.pose.y),
                    )
                )
                predicted_distance = float(
                    np.hypot(
                        float(anchor_world_xy[0]) - float(predicted_world_xy[0]),
                        float(anchor_world_xy[1]) - float(predicted_world_xy[1]),
                    )
                )
                anchor_delta_m = predicted_distance - current_distance
                anchor_local = _world_to_local_xy(robot_pose=observation.pose, world_xy=anchor_world_xy)
                anchor_bearing_after_deg = float(
                    np.degrees(np.arctan2(float(anchor_local[1]), max(1.0e-6, float(anchor_local[0]))))
                    - heading_deg
                )

            support = {
                **depth_support,
                "map_candidate_type": "primitive_outcome",
                "primitive_action": action,
                "predicted_endpoint_world_xy": [float(predicted_world_xy[0]), float(predicted_world_xy[1])],
                "free_ratio": float(free_ratio),
                "min_depth": float(min_depth_m),
                "min_clearance_m": float(min_clearance_m),
                "unknown_gain": float(unknown_gain),
                "occupancy_reason": occupancy_reason,
                "occupancy_valid": float(bool(occupancy_valid)),
                "depth_valid": float(bool(depth_valid)),
                "collision_sample_count": int(collision_result.sampled_pose_count),
                "collision_truth": "inflated_map_center_trajectory",
                "anchor_distance_delta_m": "none" if anchor_delta_m is None else round(float(anchor_delta_m), 3),
                "anchor_bearing_after_deg": "none" if anchor_bearing_after_deg is None else round(float(anchor_bearing_after_deg), 1),
                "route_alignment": 0.0,
                "repetition_penalty": 0.0,
                "target_label": str(current_target),
            }
            clearance_score = float(
                np.clip(
                    0.55 * free_ratio
                    + 0.30 * min(1.0, min_clearance_m / max(0.1, float(self._config.min_clearance_m) * 2.0))
                    + 0.15 * (1.0 - min(1.0, unknown_gain)),
                    0.0,
                    1.0,
                )
            )
            candidates.append(
                WaypointCandidate(
                    slot_id=int(slot_id),
                    label=action.lower(),
                    valid=valid,
                    local_goal_xy=local_goal,
                    heading_delta_deg=float(heading_deg),
                    travel_distance_m=float(distance_m if action == "FORWARD" else 0.0),
                    clearance_score=clearance_score,
                    depth_support=support,
                    image_uv=_local_goal_to_image_uv(local_goal_xy=local_goal, observation=observation),
                )
            )
        return candidates

    def _semantic_candidate(
        self,
        *,
        observation: VLNObservation,
        snapshot: OccupancyGridSnapshot,
        slot_id: int,
        current_target: str,
        landmark_state: LandmarkState | None,
        landmark_entry: LandmarkMapEntry | None,
    ) -> WaypointCandidate | None:
        anchor_world_xy: list[float] | None = None
        anchor_confidence = 0.0
        if landmark_entry is not None and bool(landmark_entry.world_xy):
            anchor_world_xy = [float(landmark_entry.world_xy[0]), float(landmark_entry.world_xy[1])]
            anchor_confidence = float(landmark_entry.confidence)
        elif landmark_state is not None and landmark_state.estimated_world_xy is not None:
            anchor_world_xy = [float(landmark_state.estimated_world_xy[0]), float(landmark_state.estimated_world_xy[1])]
            anchor_confidence = float(landmark_state.confirmed_confidence or landmark_state.confidence)
        if anchor_world_xy is None:
            return None

        local = _world_to_local_xy(robot_pose=observation.pose, world_xy=anchor_world_xy)
        forward_m, left_m = float(local[0]), float(local[1])
        distance_m = float(np.hypot(forward_m, left_m))
        if distance_m < max(0.20, float(self._config.min_distance_m) * 0.60):
            return None
        heading_rad = float(np.arctan2(left_m, max(1.0e-6, forward_m)))
        heading_deg = float(np.degrees(heading_rad))
        travel_m = float(np.clip(distance_m * 0.70, self._config.min_distance_m, self._config.max_distance_m))
        local_goal = [
            float(travel_m * np.cos(heading_rad)),
            float(travel_m * np.sin(heading_rad)),
        ]
        evaluation = _evaluate_local_goal(
            snapshot=snapshot,
            robot_pose=observation.pose,
            local_goal_xy=local_goal,
            min_clearance_m=float(self._config.min_clearance_m),
        )
        if not evaluation["valid"]:
            return None
        support = {
            "map_candidate_type": "semantic_approach",
            "target_label": str(current_target),
            "anchor_distance_m": float(distance_m),
            "anchor_confidence": float(anchor_confidence),
            "free_ratio": float(evaluation["free_ratio"]),
            "min_depth": float(evaluation["clearance_m"]),
            "median_depth": float(travel_m),
            "occupancy_reason": "semantic_anchor_reachable",
            "occupancy_path_length_m": float(evaluation["path_length_m"]),
            "unknown_gain": float(evaluation["unknown_gain"]),
        }
        return WaypointCandidate(
            slot_id=int(slot_id),
            label=f"semantic_approach:{current_target}",
            valid=True,
            local_goal_xy=local_goal,
            heading_delta_deg=float(heading_deg),
            travel_distance_m=float(travel_m),
            clearance_score=float(np.clip(0.65 + 0.35 * evaluation["free_ratio"], 0.0, 1.0)),
            depth_support=support,
            image_uv=_local_goal_to_image_uv(local_goal_xy=local_goal, observation=observation),
        )

    def _frontier_candidates(
        self,
        *,
        observation: VLNObservation,
        snapshot: OccupancyGridSnapshot,
        start_slot_id: int,
        current_target: str,
        landmark_entry: LandmarkMapEntry | None,
    ) -> list[WaypointCandidate]:
        candidates: list[WaypointCandidate] = []
        slot_id = int(start_slot_id)
        anchor_local = None
        if landmark_entry is not None and bool(landmark_entry.world_xy):
            anchor_local = _world_to_local_xy(robot_pose=observation.pose, world_xy=landmark_entry.world_xy)
        for bearing_deg in self._config.frontier_bearings_deg:
            bearing_candidates: list[WaypointCandidate] = []
            for distance_m in self._config.frontier_distances_m:
                heading_rad = float(np.deg2rad(float(bearing_deg)))
                local_goal = [
                    float(distance_m * np.cos(heading_rad)),
                    float(distance_m * np.sin(heading_rad)),
                ]
                evaluation = _evaluate_local_goal(
                    snapshot=snapshot,
                    robot_pose=observation.pose,
                    local_goal_xy=local_goal,
                    min_clearance_m=float(self._config.min_clearance_m),
                )
                if not evaluation["valid"]:
                    continue
                semantic_alignment = _semantic_alignment_score(
                    local_goal_xy=local_goal,
                    anchor_local_xy=anchor_local,
                )
                semantic_weight = float(self._config.semantic_weight) if anchor_local is not None else 0.0
                free_weight = 0.40 + (0.40 * float(self._config.semantic_weight) if anchor_local is None else 0.0)
                unknown_weight = (
                    float(self._config.unknown_gain_weight)
                    + (0.60 * float(self._config.semantic_weight) if anchor_local is None else 0.0)
                )
                score = float(
                    free_weight * evaluation["free_ratio"]
                    + unknown_weight * evaluation["unknown_gain"]
                    + semantic_weight * semantic_alignment
                    + 0.15 * np.clip(float(distance_m) / max(0.1, float(self._config.max_distance_m)), 0.0, 1.0)
                )
                support = {
                    "map_candidate_type": "frontier",
                    "target_label": str(current_target),
                    "free_ratio": float(evaluation["free_ratio"]),
                    "min_depth": float(evaluation["clearance_m"]),
                    "median_depth": float(distance_m),
                    "occupancy_reason": "frontier_reachable",
                    "occupancy_path_length_m": float(evaluation["path_length_m"]),
                    "unknown_gain": float(evaluation["unknown_gain"]),
                    "semantic_alignment": float(semantic_alignment),
                }
                candidate = WaypointCandidate(
                    slot_id=int(slot_id),
                    label="frontier",
                    valid=True,
                    local_goal_xy=local_goal,
                    heading_delta_deg=float(bearing_deg),
                    travel_distance_m=float(distance_m),
                    clearance_score=float(np.clip(score, 0.0, 1.0)),
                    depth_support=support,
                    image_uv=_local_goal_to_image_uv(local_goal_xy=local_goal, observation=observation),
                )
                bearing_candidates.append(candidate)
            for candidate in sorted(
                bearing_candidates,
                key=lambda item: (
                    float(item.clearance_score),
                    float(item.depth_support.get("unknown_gain", 0.0)),
                    float(item.travel_distance_m),
                ),
                reverse=True,
            )[:2]:
                candidate.slot_id = slot_id
                candidates.append(candidate)
                slot_id += 1
        return candidates

    def _escape_candidates(
        self,
        *,
        observation: VLNObservation,
        snapshot: OccupancyGridSnapshot,
        start_slot_id: int,
    ) -> list[WaypointCandidate]:
        candidates: list[WaypointCandidate] = []
        for offset, bearing_deg in enumerate((40.0, -40.0)):
            local_goal = [0.45, float(np.sin(np.deg2rad(bearing_deg)) * 0.75)]
            evaluation = _evaluate_local_goal(
                snapshot=snapshot,
                robot_pose=observation.pose,
                local_goal_xy=local_goal,
                min_clearance_m=float(self._config.min_clearance_m),
            )
            support = {
                "map_candidate_type": "escape_turn",
                "free_ratio": float(evaluation["free_ratio"]),
                "min_depth": float(evaluation["clearance_m"]),
                "median_depth": float(self._config.min_distance_m),
                "occupancy_reason": "escape_recovery" if evaluation["valid"] else "escape_blocked",
                "occupancy_path_length_m": float(evaluation["path_length_m"]),
                "unknown_gain": float(evaluation["unknown_gain"]),
            }
            candidates.append(
                WaypointCandidate(
                    slot_id=int(start_slot_id + offset),
                    label="escape_turn",
                    valid=bool(evaluation["valid"]),
                    local_goal_xy=local_goal if evaluation["valid"] else [0.0, 0.0],
                    heading_delta_deg=float(bearing_deg),
                    travel_distance_m=float(self._config.min_distance_m if evaluation["valid"] else 0.0),
                    clearance_score=float(0.55 if evaluation["valid"] else 0.0),
                    depth_support=support,
                    image_uv=_local_goal_to_image_uv(local_goal_xy=local_goal, observation=observation),
                )
            )
        return candidates

    def _rank_candidates(
        self,
        *,
        base_candidates: list[WaypointCandidate],
        generated_candidates: list[WaypointCandidate],
        progress_signal: str,
    ) -> list[WaypointCandidate]:
        max_candidates = max(3, int(self._config.max_candidates))
        valid_generated = [item for item in generated_candidates if item.valid]
        valid_base = [item for item in base_candidates if item.valid]
        invalid_base = [item for item in base_candidates if not item.valid]
        ordered = sorted(
            valid_generated,
            key=lambda item: (
                _candidate_priority(item, progress_signal=progress_signal),
                float(item.clearance_score),
                float(item.depth_support.get("unknown_gain", 0.0)),
            ),
            reverse=True,
        )
        for candidate in sorted(valid_base, key=lambda item: float(item.clearance_score), reverse=True):
            if len(ordered) >= max_candidates:
                break
            if _too_close_to_existing(candidate, ordered):
                continue
            ordered.append(candidate)
        if len(ordered) < max_candidates:
            ordered.extend(invalid_base[: max_candidates - len(ordered)])
        return _renumber_candidates(ordered[:max_candidates])

    def _log_if_changed(
        self,
        *,
        current_target: str,
        ranked: list[WaypointCandidate],
        valid_count: int,
    ) -> None:
        signature = (
            str(current_target),
            tuple(str(item.depth_support.get("map_candidate_type", "depth")) for item in ranked),
            tuple(int(item.valid) for item in ranked),
            tuple(str(item.depth_support.get("occupancy_reason", "unknown")) for item in ranked),
            int(valid_count),
        )
        if signature == self._last_signature:
            return
        self._last_signature = signature
        self._logger.info(
            "[PrimitiveCheck] target=%s valid=%s checks=%s",
            current_target,
            int(valid_count),
            [
                {
                    "action": str(item.depth_support.get("primitive_action", item.label)).upper(),
                    "valid": int(item.valid),
                    "clearance": round(float(item.depth_support.get("min_clearance_m", 0.0)), 2),
                    "free_ratio": round(float(item.depth_support.get("free_ratio", 0.0)), 2),
                    "reason": str(item.depth_support.get("occupancy_reason", "unknown")),
                }
                for item in ranked
            ],
        )


def _evaluate_local_goal(
    *,
    snapshot: OccupancyGridSnapshot,
    robot_pose: Pose2D,
    local_goal_xy: list[float],
    min_clearance_m: float,
    path_start_world_xy: tuple[float, float] | None = None,
) -> dict[str, float | bool]:
    path_start = (
        (float(robot_pose.x), float(robot_pose.y))
        if path_start_world_xy is None
        else (float(path_start_world_xy[0]), float(path_start_world_xy[1]))
    )
    world_goal = _offset_world_xy(
        start_world_xy=path_start,
        yaw_rad=float(robot_pose.yaw_rad),
        local_offset_xy=local_goal_xy,
    )
    goal_cell = _world_to_cell(snapshot=snapshot, world_xy=world_goal)
    if goal_cell is None:
        return _eval_result(False, "outside", 0.0, 0.0, 0.0, 0.0)
    row, col = goal_cell
    if bool(snapshot.inflated_occupied[row, col]):
        return _eval_result(False, "occupied", 0.0, 0.0, 0.0, 0.0)
    path_cells = _line_cells(
        _world_to_cell(snapshot=snapshot, world_xy=path_start),
        goal_cell,
    )
    if not path_cells:
        return _eval_result(False, "no_path", 0.0, 0.0, 0.0, 0.0)
    blocked = 0
    free_hits = 0
    unknown_hits = 0
    for rr, cc in path_cells:
        if rr < 0 or rr >= snapshot.log_odds.shape[0] or cc < 0 or cc >= snapshot.log_odds.shape[1]:
            blocked += 1
            continue
        if bool(snapshot.inflated_occupied[rr, cc]):
            blocked += 1
        elif bool(snapshot.free_mask[rr, cc]):
            free_hits += 1
        else:
            unknown_hits += 1
    allowed_blocked = max(1, int(round(0.08 * float(len(path_cells)))))
    if blocked > allowed_blocked:
        return _eval_result(False, "path_blocked", 0.0, 0.0, 0.0, 0.0)
    local_distance = float(np.hypot(float(local_goal_xy[0]), float(local_goal_xy[1])))
    free_ratio = float(free_hits) / float(max(1, len(path_cells)))
    unknown_gain = float(unknown_hits) / float(max(1, len(path_cells)))
    clearance_m = _clearance_m(snapshot=snapshot, cell=goal_cell, max_radius_m=1.0)
    valid = bool(clearance_m >= float(min_clearance_m))
    return _eval_result(valid, "ok", free_ratio, unknown_gain, clearance_m, local_distance)


def _eval_result(
    valid: bool,
    reason: str,
    free_ratio: float,
    unknown_gain: float,
    clearance_m: float,
    path_length_m: float,
) -> dict[str, float | bool | str]:
    return {
        "valid": bool(valid),
        "reason": str(reason),
        "free_ratio": float(free_ratio),
        "unknown_gain": float(unknown_gain),
        "clearance_m": float(clearance_m),
        "path_length_m": float(path_length_m),
    }


def _clearance_m(
    *,
    snapshot: OccupancyGridSnapshot,
    cell: tuple[int, int],
    max_radius_m: float,
) -> float:
    max_cells = max(1, int(round(float(max_radius_m) / max(1.0e-6, float(snapshot.resolution_m)))))
    row, col = cell
    occupied = snapshot.inflated_occupied
    best = float(max_radius_m)
    for radius in range(1, max_cells + 1):
        r0 = max(0, row - radius)
        r1 = min(occupied.shape[0], row + radius + 1)
        c0 = max(0, col - radius)
        c1 = min(occupied.shape[1], col + radius + 1)
        if np.any(occupied[r0:r1, c0:c1]):
            best = float(radius) * float(snapshot.resolution_m)
            break
    return float(best)


def _line_cells(start: tuple[int, int] | None, goal: tuple[int, int] | None) -> list[tuple[int, int]]:
    if start is None or goal is None:
        return []
    r0, c0 = start
    r1, c1 = goal
    steps = max(abs(int(r1) - int(r0)), abs(int(c1) - int(c0)), 1)
    rows = np.linspace(r0, r1, steps + 1).round().astype(int)
    cols = np.linspace(c0, c1, steps + 1).round().astype(int)
    return list(dict.fromkeys((int(rr), int(cc)) for rr, cc in zip(rows, cols)))


def _world_to_cell(*, snapshot: OccupancyGridSnapshot, world_xy: tuple[float, float] | list[float]) -> tuple[int, int] | None:
    col = int(np.floor((float(world_xy[0]) - float(snapshot.origin_x)) / float(snapshot.resolution_m)))
    row = int(np.floor((float(world_xy[1]) - float(snapshot.origin_y)) / float(snapshot.resolution_m)))
    if row < 0 or row >= snapshot.log_odds.shape[0] or col < 0 or col >= snapshot.log_odds.shape[1]:
        return None
    return row, col


def _local_to_world_xy(*, robot_pose: Pose2D, local_xy: list[float] | tuple[float, float]) -> tuple[float, float]:
    forward_m, left_m = [float(value) for value in local_xy[:2]]
    cos_yaw = float(np.cos(robot_pose.yaw_rad))
    sin_yaw = float(np.sin(robot_pose.yaw_rad))
    return (
        float(robot_pose.x) + cos_yaw * forward_m - sin_yaw * left_m,
        float(robot_pose.y) + sin_yaw * forward_m + cos_yaw * left_m,
    )


def _offset_world_xy(
    *,
    start_world_xy: tuple[float, float],
    yaw_rad: float,
    local_offset_xy: list[float] | tuple[float, float],
) -> tuple[float, float]:
    forward_m, left_m = [float(value) for value in local_offset_xy[:2]]
    cos_yaw = float(np.cos(yaw_rad))
    sin_yaw = float(np.sin(yaw_rad))
    return (
        float(start_world_xy[0]) + cos_yaw * forward_m - sin_yaw * left_m,
        float(start_world_xy[1]) + sin_yaw * forward_m + cos_yaw * left_m,
    )


def _world_to_local_xy(*, robot_pose: Pose2D, world_xy: list[float] | tuple[float, float]) -> tuple[float, float]:
    delta_x = float(world_xy[0]) - float(robot_pose.x)
    delta_y = float(world_xy[1]) - float(robot_pose.y)
    cos_yaw = float(np.cos(robot_pose.yaw_rad))
    sin_yaw = float(np.sin(robot_pose.yaw_rad))
    return (
        cos_yaw * delta_x + sin_yaw * delta_y,
        -sin_yaw * delta_x + cos_yaw * delta_y,
    )


def _local_goal_to_image_uv(*, local_goal_xy: list[float], observation: VLNObservation) -> tuple[int, int]:
    width = int(observation.rgb.shape[1]) if observation.rgb is not None and observation.rgb.ndim == 3 else 640
    height = int(observation.rgb.shape[0]) if observation.rgb is not None and observation.rgb.ndim == 3 else 480
    forward_m = max(1.0e-6, float(local_goal_xy[0]))
    left_m = float(local_goal_xy[1])
    heading_rad = float(np.arctan2(left_m, forward_m))
    hfov_rad = np.deg2rad(float(observation.meta.get("horizontal_fov_deg", 90.0)))
    normalized = np.tan(heading_rad) / max(np.tan(hfov_rad * 0.5), 1.0e-6)
    cx = (float(width) - 1.0) * 0.5
    u = int(round(cx - normalized * cx))
    return int(np.clip(u, 0, width - 1)), int(round(float(height) * 0.58))


def _semantic_alignment_score(
    *,
    local_goal_xy: list[float],
    anchor_local_xy: tuple[float, float] | None,
) -> float:
    if anchor_local_xy is None:
        return 0.0
    goal = np.asarray(local_goal_xy, dtype=np.float32)
    anchor = np.asarray(anchor_local_xy, dtype=np.float32)
    goal_norm = float(np.linalg.norm(goal))
    anchor_norm = float(np.linalg.norm(anchor))
    if goal_norm <= 1.0e-6 or anchor_norm <= 1.0e-6:
        return 0.0
    return float(np.clip(float(np.dot(goal, anchor) / (goal_norm * anchor_norm)), -1.0, 1.0) * 0.5 + 0.5)


def _candidate_priority(candidate: WaypointCandidate, *, progress_signal: str) -> float:
    kind = str(candidate.depth_support.get("map_candidate_type", "depth"))
    progress = str(progress_signal).strip().upper()
    if progress in {"STALLED", "SPINNING"} or progress.startswith("SAFETY_BLOCKED"):
        if kind == "escape_turn":
            return 5.0
        if kind == "frontier":
            return 3.0
        if kind == "semantic_approach":
            return 2.5
    if kind == "semantic_approach":
        return 4.0
    if kind == "escape_turn":
        return 3.0
    if kind == "frontier":
        return 2.0
    return 1.0


def _too_close_to_existing(candidate: WaypointCandidate, existing: list[WaypointCandidate]) -> bool:
    current = np.asarray(candidate.local_goal_xy, dtype=np.float32)
    for item in existing:
        other = np.asarray(item.local_goal_xy, dtype=np.float32)
        if float(np.linalg.norm(current - other)) < 0.25:
            return True
    return False


def _renumber_candidates(candidates: list[WaypointCandidate]) -> list[WaypointCandidate]:
    result: list[WaypointCandidate] = []
    for slot_id, candidate in enumerate(candidates):
        result.append(
            WaypointCandidate(
                slot_id=int(slot_id),
                label=str(candidate.label),
                valid=bool(candidate.valid),
                local_goal_xy=list(candidate.local_goal_xy),
                heading_delta_deg=float(candidate.heading_delta_deg),
                travel_distance_m=float(candidate.travel_distance_m),
                clearance_score=float(candidate.clearance_score),
                depth_support=dict(candidate.depth_support),
                image_uv=tuple(candidate.image_uv),
            )
        )
    return result


def _needs_escape(*, progress_signal: str, safety_context_text: str) -> bool:
    if str(progress_signal).strip().upper() in {"STALLED", "SPINNING"}:
        return True
    text = str(safety_context_text).upper()
    return "SAFETY_STOP" in text or "STUCK_DETECTED" in text


def _nearest_depth_candidate(
    *,
    base_candidate_set: CandidateSet,
    heading_deg: float,
) -> WaypointCandidate | None:
    if not base_candidate_set.candidates:
        return None
    return min(
        base_candidate_set.candidates,
        key=lambda item: abs(float(item.heading_delta_deg) - float(heading_deg)),
    )


def _confirmed_anchor_world_xy(
    *,
    landmark_state: LandmarkState | None,
    landmark_entry: LandmarkMapEntry | None,
) -> list[float] | None:
    if landmark_entry is not None and bool(landmark_entry.confirmed) and bool(landmark_entry.world_xy):
        return [float(landmark_entry.world_xy[0]), float(landmark_entry.world_xy[1])]
    if (
        landmark_state is not None
        and str(landmark_state.evidence_level).strip().lower() == "confirmed"
        and landmark_state.estimated_world_xy is not None
    ):
        return [float(landmark_state.estimated_world_xy[0]), float(landmark_state.estimated_world_xy[1])]
    return None
