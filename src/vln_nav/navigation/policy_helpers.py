"""Module-level helper functions for vln_policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .candidate_waypoint_predictor import CandidateSet, WaypointCandidate
from .instruction_manager import InstructionContext, RouteStep

# TODO: resolve circular imports when OpenNavPolicyConfig and ReacquireGuardContext
# are extracted. Currently they are imported at call sites.


_INVALID_LANDMARKS: set[str] = {
    "none", "null", "unknown", "n/a", "na", "target", "object", "thing",
    "place", "location", "area", "destination", "goal", "",
}


def _find_candidate(candidate_set: CandidateSet, slot_id: int | None) -> WaypointCandidate | None:
    if slot_id is None:
        return None
    for candidate in candidate_set.candidates:
        if int(candidate.slot_id) == int(slot_id):
            return candidate
    return None


def _is_valid_landmark(label: str) -> bool:
    text = str(label).strip().lower()
    return bool(text) and text not in _INVALID_LANDMARKS and len(text) >= 2


def _route_steps(instruction_context: InstructionContext) -> list[RouteStep]:
    steps: list[RouteStep] = []
    for route_step in getattr(instruction_context, "route_steps", []) or []:
        landmark = str(getattr(route_step, "landmark", "")).strip().lower()
        if not _is_valid_landmark(landmark):
            continue
        directive = str(getattr(route_step, "directive", "")).strip()
        behavior_type = str(getattr(route_step, "behavior_type", "go_to")).strip().lower() or "go_to"
        steps.append(
            RouteStep(
                landmark=landmark,
                directive=directive,
                behavior_type=behavior_type,
            )
        )
    if steps:
        return steps

    fallback_steps: list[RouteStep] = []
    for raw_landmark in instruction_context.landmarks:
        landmark = str(raw_landmark).strip().lower()
        if _is_valid_landmark(landmark):
            fallback_steps.append(
                RouteStep(
                    landmark=landmark,
                    directive=f"go to {landmark}",
                    behavior_type="go_to",
                )
            )
    if fallback_steps:
        return fallback_steps

    fallback = str(instruction_context.goal_phrase or instruction_context.normalized_text).strip().lower()
    if _is_valid_landmark(fallback):
        return [RouteStep(landmark=fallback, directive=f"go to {fallback}", behavior_type="go_to")]
    return []


def _ordered_landmarks_from_steps(
    route_steps: list[RouteStep],
    instruction_context: InstructionContext,
) -> list[str]:
    ordered: list[str] = []
    for route_step in route_steps:
        landmark = str(getattr(route_step, "landmark", "")).strip().lower()
        if _is_valid_landmark(landmark) and landmark not in ordered:
            ordered.append(landmark)
    for raw_landmark in instruction_context.landmarks:
        landmark = str(raw_landmark).strip().lower()
        if _is_valid_landmark(landmark) and landmark not in ordered:
            ordered.append(landmark)
    if not ordered:
        fallback = str(instruction_context.goal_phrase or instruction_context.normalized_text).strip()
        if fallback and _is_valid_landmark(fallback):
            ordered.append(fallback)
    return ordered


def _pass_action_sequence(directive: str) -> list[str]:
    text = str(directive).strip().lower()
    if "right" in text:
        return ["TURN_RIGHT", "FORWARD", "TURN_LEFT", "FORWARD"]
    if "left" in text:
        return ["TURN_LEFT", "FORWARD", "TURN_RIGHT", "FORWARD"]
    return ["FORWARD"]


def _direction_tokens_from_text(text: str) -> list[str]:
    lowered = str(text).strip().lower()
    tokens: list[str] = []
    checks = [
        ("left", ("left", "左")),
        ("right", ("right", "右")),
        ("forward", ("forward", "straight", "ahead", "front", "前", "直")),
        ("backward", ("back", "backward", "后", "退")),
        ("through", ("through", "go through", "穿过", "通过")),
        ("pass", ("pass", "around", "绕过", "经过")),
    ]
    for token, aliases in checks:
        if any(alias in lowered for alias in aliases):
            tokens.append(token)
    return tokens


def _is_fresh_grounded_observation(observation) -> bool:
    return (
        bool(getattr(observation, "visible", False))
        and str(getattr(observation, "reason", "")).strip().lower() == "ok"
        and "reused" not in str(getattr(observation, "source", "")).strip().lower()
        and bool(getattr(observation, "is_current_frame", True))
        and getattr(observation, "depth_m", None) is not None
        and getattr(observation, "estimated_world_xy", None) is not None
    )


def _format_reacquire_prompt_context(guard) -> str:
    if not bool(guard.target_recently_visible) and not bool(guard.lost_after_visible):
        return "none"
    if bool(guard.lost_after_visible):
        return (
            "LOST_AFTER_VISIBLE active. "
            f"previous_semantic_visible={int(guard.previous_semantic_visible)}; "
            f"previous_grounded_visible={int(guard.previous_grounded_visible)}; "
            f"current_grounded_visible={int(guard.current_grounded_visible)}; "
            f"last_action={guard.last_action}; "
            f"last_seen_bearing={guard.last_seen_bearing}; "
            f"Forbidden action this step: {guard.forbidden_action}. "
            "Do not choose the forbidden action. If the target is still not visible in the current image, "
            "prefer a candidate that stops repeating the previous turn direction; if the target is visible again "
            "in the current image, choose based on that current visual evidence. The runtime will force STOP if "
            "the forbidden action is selected while the target is still not visible."
        )
    return (
        "target_recently_visible=1 but LOST_AFTER_VISIBLE inactive. "
        f"previous_semantic_visible={int(guard.previous_semantic_visible)}; "
        f"previous_grounded_visible={int(guard.previous_grounded_visible)}; "
        f"current_grounded_visible={int(guard.current_grounded_visible)}; "
        f"last_action={guard.last_action}; last_seen_bearing={guard.last_seen_bearing}."
    )


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


def _memory_anchor_local_xy(*, candidate: WaypointCandidate, bearing_hint: str) -> list[float]:
    distance_m = max(0.8, float(candidate.travel_distance_m))
    hint = str(bearing_hint).strip().lower()
    if hint == "left":
        return [float(distance_m * 0.75), float(distance_m * 0.65)]
    if hint == "right":
        return [float(distance_m * 0.75), float(-distance_m * 0.65)]
    return [float(distance_m), 0.0]


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


def _has_non_forbidden_valid_candidate(
    *,
    candidate_set: CandidateSet,
    forbidden_action: str,
    config,  # OpenNavPolicyConfig
) -> bool:
    forbidden = str(forbidden_action).strip().upper()
    if forbidden not in {"TURN_LEFT", "TURN_RIGHT", "FORWARD", "STOP"}:
        return True
    for candidate in candidate_set.candidates:
        if not bool(candidate.valid):
            continue
        action_name, _ = _candidate_to_primitive_action(
            candidate,
            turn_speed_scale=float(config.turn_speed_scale),
            forward_speed_scale=float(config.forward_speed_scale),
        )
        if str(action_name).strip().upper() != forbidden:
            return True
    return False


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
