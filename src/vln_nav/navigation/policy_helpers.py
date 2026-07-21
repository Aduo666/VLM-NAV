"""Module-level helper functions for vln_policy."""

from __future__ import annotations

import numpy as np

from .candidate_waypoint_predictor import CandidateSet, WaypointCandidate
from .instruction_manager import InstructionContext, RouteStep

def _find_candidate(candidate_set: CandidateSet, slot_id: int | None) -> WaypointCandidate | None:
    if slot_id is None:
        return None
    for candidate in candidate_set.candidates:
        if int(candidate.slot_id) == int(slot_id):
            return candidate
    return None


def _route_steps(instruction_context: InstructionContext) -> list[RouteStep]:
    steps = list(instruction_context.route_steps)
    if not steps:
        raise ValueError("validated instruction graph has no route steps")
    return steps


def _relative_xy_from_anchor(
    *,
    anchor_world_xy: list[float],
    anchor_yaw_rad: float,
    target_world_xy: list[float],
) -> tuple[float, float]:
    delta_world = np.asarray(target_world_xy, dtype=np.float32) - np.asarray(anchor_world_xy, dtype=np.float32)
    cos_yaw = float(np.cos(anchor_yaw_rad))
    sin_yaw = float(np.sin(anchor_yaw_rad))
    forward_m = cos_yaw * float(delta_world[0]) + sin_yaw * float(delta_world[1])
    left_m = -sin_yaw * float(delta_world[0]) + cos_yaw * float(delta_world[1])
    return float(forward_m), float(left_m)


def _relative_side(*, local_forward_m: float, local_left_m: float) -> str:
    if local_forward_m < -0.30:
        if local_left_m > 0.35:
            return "behind_left"
        if local_left_m < -0.35:
            return "behind_right"
        return "behind"
    if abs(local_left_m) <= 0.35:
        return "ahead"
    return "left" if local_left_m > 0.0 else "right"


def _direction_consistency(*, expected_words: set[str], observed_side: str) -> str:
    observed = str(observed_side)
    if "left" in expected_words and "right" in observed:
        return "contradiction"
    if "right" in expected_words and "left" in observed:
        return "contradiction"
    if ("forward" in expected_words or "straight" in expected_words or "ahead" in expected_words) and observed.startswith("behind"):
        return "contradiction"
    if expected_words:
        return "consistent_or_unknown"
    return "no_direction_constraint"


def _local_waypoint_to_world_xy(*, robot_pose, waypoint_local_xy: list[float]) -> np.ndarray:
    forward_m, left_m = [float(value) for value in waypoint_local_xy]
    cos_yaw = float(np.cos(robot_pose.yaw_rad))
    sin_yaw = float(np.sin(robot_pose.yaw_rad))
    delta_world = np.asarray(
        [
            cos_yaw * forward_m - sin_yaw * left_m,
            sin_yaw * forward_m + cos_yaw * left_m,
        ],
        dtype=np.float32,
    )
    return np.asarray([float(robot_pose.x), float(robot_pose.y)], dtype=np.float32) + delta_world


def _world_waypoint_to_local_xy(*, robot_pose, world_goal_xy: list[float]) -> tuple[float, float]:
    delta_world = np.asarray(world_goal_xy, dtype=np.float32) - np.asarray(
        [float(robot_pose.x), float(robot_pose.y)],
        dtype=np.float32,
    )
    cos_yaw = float(np.cos(robot_pose.yaw_rad))
    sin_yaw = float(np.sin(robot_pose.yaw_rad))
    forward_m = cos_yaw * float(delta_world[0]) + sin_yaw * float(delta_world[1])
    left_m = -sin_yaw * float(delta_world[0]) + cos_yaw * float(delta_world[1])
    return float(forward_m), float(left_m)


def _pose_distance_m(start_pose, current_pose) -> float:
    if start_pose is None or current_pose is None:
        return 0.0
    return float(
        np.hypot(
            float(current_pose.x) - float(start_pose.x),
            float(current_pose.y) - float(start_pose.y),
        )
    )


def _track_target_command(
    *,
    local_goal_xy: tuple[float, float] | list[float],
    stop_distance_m: float,
) -> tuple[str, float]:
    forward_m = float(local_goal_xy[0])
    left_m = float(local_goal_xy[1])
    distance_m = float(np.hypot(forward_m, left_m))
    if distance_m <= float(stop_distance_m) or forward_m < -0.10:
        return "STOP", 0.0
    return "TRACK_TARGET", 1.0


def _candidate_to_primitive_action(
    candidate: WaypointCandidate,
    *,
    turn_speed_scale: float,
    forward_speed_scale: float,
) -> tuple[str, float]:
    explicit = str(candidate.depth_support.get("primitive_action", "")).strip().upper()
    if explicit == "TURN_LEFT":
        return explicit, float(turn_speed_scale)
    if explicit == "TURN_RIGHT":
        return explicit, float(turn_speed_scale)
    if explicit == "FORWARD":
        return explicit, float(forward_speed_scale)
    heading_deg = float(candidate.heading_delta_deg)
    if heading_deg > 12.0:
        return "TURN_LEFT", float(turn_speed_scale)
    if heading_deg < -12.0:
        return "TURN_RIGHT", float(turn_speed_scale)
    return "FORWARD", float(forward_speed_scale)


def _local_goal_to_primitive_action(*, local_goal_xy: tuple[float, float] | list[float]) -> str:
    forward_m = float(local_goal_xy[0])
    left_m = float(local_goal_xy[1])
    heading_deg = float(np.degrees(np.arctan2(left_m, max(1.0e-6, forward_m))))
    if forward_m < -0.10:
        return "TURN_LEFT" if left_m >= 0.0 else "TURN_RIGHT"
    if heading_deg > 12.0:
        return "TURN_LEFT"
    if heading_deg < -12.0:
        return "TURN_RIGHT"
    return "FORWARD"


def _nearest_candidate_for_local_goal(
    *,
    candidate_set: CandidateSet,
    local_goal_xy: tuple[float, float] | list[float],
) -> WaypointCandidate | None:
    candidates = [candidate for candidate in candidate_set.candidates if bool(candidate.valid)]
    if not candidates:
        candidates = list(candidate_set.candidates)
    if not candidates:
        return None
    goal_forward = float(local_goal_xy[0])
    goal_left = float(local_goal_xy[1])
    goal_heading = float(np.degrees(np.arctan2(goal_left, max(1.0e-6, goal_forward))))
    return min(
        candidates,
        key=lambda candidate: (
            abs(float(candidate.heading_delta_deg) - goal_heading),
            abs(float(candidate.travel_distance_m) - float(np.hypot(goal_forward, goal_left))),
            0 if candidate.valid else 1,
        ),
    )


def _primitive_speed_scale(action_name: str, config) -> float:  # OpenNavPolicyConfig
    action = str(action_name).strip().upper()
    if action in {"TURN_LEFT", "TURN_RIGHT"}:
        return float(config.turn_speed_scale)
    if action == "FORWARD":
        return float(config.forward_speed_scale)
    return 0.0


def _format_optional_float(value: float | None) -> str:
    if value is None or not np.isfinite(float(value)):
        return "none"
    return f"{float(value):.2f}"


def _format_xy(value) -> str:
    if value is None:
        return "none"
    try:
        if len(value) < 2:
            return "none"
        return f"[{float(value[0]):.2f},{float(value[1]):.2f}]"
    except Exception:
        return "none"


def _wrap_angle_rad(angle_rad: float) -> float:
    return float((float(angle_rad) + np.pi) % (2.0 * np.pi) - np.pi)
