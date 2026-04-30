from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .observation_adapter import Pose2D
from .target_grounding import TargetObservation


@dataclass
class LandmarkState:
    label: str
    behavior_type: str
    status: str
    visible_count: int
    lost_count: int
    reached_count: int
    last_observation: TargetObservation | None
    candidate_world_xy: list[float] | None
    estimated_world_xy: list[float] | None
    confidence: float
    last_seen_time_sec: float | None
    reached: bool
    reach_reason: str
    anchor_distance_m: float | None


@dataclass
class LandmarkTrackerConfig:
    arrival_visible_depth_m: float
    arrival_anchor_radius_m: float
    min_visible_count: int
    arrival_confirmations: int
    max_lost_count: int
    anchor_update_min_consistent_observations: int
    anchor_update_max_jump_m: float


@dataclass
class _PendingAnchor:
    xy_sum: np.ndarray
    count: int
    last_xy: np.ndarray


class LandmarkTracker:
    """Track one language landmark with candidate/confirmed anchor separation.

    The first implementation only implements go_to arrival logic. The
    behavior_type field is kept explicit so pass / turn_at / go_through can be
    added without changing the policy interface.
    """

    def __init__(self, *, config: LandmarkTrackerConfig, logger) -> None:
        self._config = config
        self._logger = logger.getChild("landmark_tracker")
        self._states: dict[str, LandmarkState] = {}
        self._pending_anchors: dict[str, _PendingAnchor] = {}
        self._last_log_time_by_label: dict[str, float] = {}
        self._last_log_signature_by_label: dict[str, tuple] = {}

    def reset(self, label: str | None = None) -> None:
        if label is None:
            self._states.clear()
            self._pending_anchors.clear()
            return
        key = _normalize_label(label)
        self._states.pop(key, None)
        self._pending_anchors.pop(key, None)

    def update(
        self,
        *,
        target_label: str,
        behavior_type: str,
        observation: TargetObservation,
        robot_pose: Pose2D,
        timestamp_sec: float,
    ) -> LandmarkState:
        label = _normalize_label(target_label)
        behavior = str(behavior_type).strip().lower() or "go_to"
        state = self._states.get(label)
        if state is None or state.behavior_type != behavior:
            state = LandmarkState(
                label=label,
                behavior_type=behavior,
                status="searching",
                visible_count=0,
                lost_count=0,
                reached_count=0,
                last_observation=None,
                candidate_world_xy=None,
                estimated_world_xy=None,
                confidence=0.0,
                last_seen_time_sec=None,
                reached=False,
                reach_reason="not_reached",
                anchor_distance_m=None,
            )

        fresh_visible = (
            bool(observation.visible)
            and str(observation.reason).strip().lower() == "ok"
            and "reused" not in str(observation.source).strip().lower()
        )
        transient_observation = _is_observation_transient(observation)
        if bool(observation.visible):
            if fresh_visible:
                state.visible_count += 1
            state.lost_count = 0
            state.last_observation = observation
            state.last_seen_time_sec = float(timestamp_sec)
            state.confidence = max(float(state.confidence), float(observation.confidence))
            state.candidate_world_xy = (
                None if observation.estimated_world_xy is None else list(observation.estimated_world_xy)
            )
            state.status = "visible"
            if fresh_visible and observation.estimated_world_xy is not None:
                confirmed = self._update_anchor_candidate(
                    label=label,
                    candidate_world_xy=observation.estimated_world_xy,
                    current_confirmed_world_xy=state.estimated_world_xy,
                )
                if confirmed is not None:
                    state.estimated_world_xy = confirmed
                    state.status = "confirmed"
        elif transient_observation:
            state.last_observation = observation
            if state.estimated_world_xy is not None:
                state.status = "confirmed"
            elif int(state.visible_count) > 0:
                state.status = "visible"
            else:
                state.status = "searching"
        else:
            state.visible_count = 0
            state.lost_count += 1
            state.status = "lost" if state.estimated_world_xy is not None else "searching"

        state.anchor_distance_m = _distance_to_anchor(
            robot_pose=robot_pose,
            anchor_world_xy=state.estimated_world_xy,
        )
        state.reached, state.reach_reason, state.reached_count = self._update_reach_state(
            state=state,
            observation=observation,
        )
        if state.reached:
            state.status = "reached"

        self._states[label] = state
        self._log_state_if_needed(state=state, observation=observation, timestamp_sec=timestamp_sec)
        return state

    def _log_state_if_needed(
        self,
        *,
        state: LandmarkState,
        observation: TargetObservation,
        timestamp_sec: float,
    ) -> None:
        label = state.label
        lost_bucket = 0 if int(state.lost_count) == 0 else 1
        if int(state.lost_count) > int(self._config.max_lost_count):
            lost_bucket = 2
        signature = (
            state.status,
            bool(observation.visible),
            int(state.visible_count),
            lost_bucket,
            _format_xy(state.estimated_world_xy),
            bool(state.reached),
            state.reach_reason,
            observation.reason,
        )
        last_time = self._last_log_time_by_label.get(label, -float("inf"))
        last_signature = self._last_log_signature_by_label.get(label)
        if signature == last_signature and float(timestamp_sec) - float(last_time) < 1.0:
            return
        self._last_log_time_by_label[label] = float(timestamp_sec)
        self._last_log_signature_by_label[label] = signature
        self._logger.info(
            "[LandmarkTracker] target=%s behavior=%s status=%s visible=%s visible_count=%s lost_count=%s "
            "depth=%s candidate_anchor=%s confirmed_anchor=%s anchor_dist=%s reached=%s confirm=%s obs_source=%s reason=%s",
            state.label,
            state.behavior_type,
            state.status,
            int(bool(observation.visible)),
            int(state.visible_count),
            int(state.lost_count),
            _format_optional_float(observation.depth_m),
            _format_xy(state.candidate_world_xy),
            _format_xy(state.estimated_world_xy),
            _format_optional_float(state.anchor_distance_m),
            int(bool(state.reached)),
            int(state.reached_count),
            observation.source,
            state.reach_reason,
        )

    def _update_anchor_candidate(
        self,
        *,
        label: str,
        candidate_world_xy: list[float],
        current_confirmed_world_xy: list[float] | None,
    ) -> list[float] | None:
        candidate = np.asarray(candidate_world_xy, dtype=np.float32).reshape(2)
        max_jump_m = max(0.05, float(self._config.anchor_update_max_jump_m))
        pending = self._pending_anchors.get(label)
        if pending is None or float(np.linalg.norm(candidate - pending.last_xy)) > max_jump_m:
            pending = _PendingAnchor(xy_sum=candidate.copy(), count=1, last_xy=candidate.copy())
            self._pending_anchors[label] = pending
        else:
            pending.xy_sum = pending.xy_sum + candidate
            pending.count += 1
            pending.last_xy = candidate.copy()

        required = max(1, int(self._config.anchor_update_min_consistent_observations))
        if pending.count < required:
            self._logger.info(
                "[LandmarkAnchor] target=%s candidate=%s confirm_wait=%s/%s",
                label,
                _format_xy(candidate.tolist()),
                int(pending.count),
                int(required),
            )
            return None

        averaged = pending.xy_sum / float(max(1, pending.count))
        if current_confirmed_world_xy is not None:
            current = np.asarray(current_confirmed_world_xy, dtype=np.float32).reshape(2)
            jump_m = float(np.linalg.norm(averaged - current))
            if jump_m > max_jump_m:
                self._logger.warning(
                    "[LandmarkAnchorReject] target=%s reason=confirmed_anchor_jump jump_m=%.2f max_jump_m=%.2f candidate=%s confirmed=%s",
                    label,
                    jump_m,
                    max_jump_m,
                    _format_xy(averaged.tolist()),
                    _format_xy(current_confirmed_world_xy),
                )
                return None

        confirmed = [float(averaged[0]), float(averaged[1])]
        self._logger.info(
            "[LandmarkAnchor] target=%s confirmed=%s observations=%s",
            label,
            _format_xy(confirmed),
            int(pending.count),
        )
        return confirmed

    def _update_reach_state(
        self,
        *,
        state: LandmarkState,
        observation: TargetObservation,
    ) -> tuple[bool, str, int]:
        if state.behavior_type != "go_to":
            return False, f"behavior_not_implemented:{state.behavior_type}", 0

        min_visible = max(1, int(self._config.min_visible_count))
        depth_threshold_m = max(0.05, float(self._config.arrival_visible_depth_m))
        anchor_threshold_m = max(0.05, float(self._config.arrival_anchor_radius_m))
        if _is_observation_transient(observation):
            if bool(state.reached):
                return True, state.reach_reason, int(state.reached_count)
            if int(state.reached_count) > 0:
                base_reason = str(state.reach_reason).split("; waiting_fresh_confirmation", 1)[0]
                return False, f"{base_reason}; waiting_fresh_confirmation", int(state.reached_count)
            return False, "not_reached", 0

        visible_depth_reached = (
            bool(observation.visible)
            and str(observation.reason).strip().lower() == "ok"
            and "reused" not in str(observation.source).strip().lower()
            and int(state.visible_count) >= min_visible
            and observation.depth_m is not None
            and float(observation.depth_m) <= depth_threshold_m
        )
        anchor_reached = (
            state.estimated_world_xy is not None
            and state.anchor_distance_m is not None
            and float(state.anchor_distance_m) <= anchor_threshold_m
        )

        if visible_depth_reached:
            reason = f"visible_depth<={depth_threshold_m:.2f}m"
            reached_count = int(state.reached_count) + 1
        elif anchor_reached:
            reason = f"confirmed_anchor_distance<={anchor_threshold_m:.2f}m"
            reached_count = int(state.reached_count) + 1
        else:
            reason = "not_reached"
            reached_count = 0

        required = max(1, int(self._config.arrival_confirmations))
        return reached_count >= required, reason, reached_count


def _normalize_label(label: str) -> str:
    return str(label).strip().lower() or "target"


def _distance_to_anchor(*, robot_pose: Pose2D, anchor_world_xy: list[float] | None) -> float | None:
    if anchor_world_xy is None:
        return None
    delta = np.asarray(anchor_world_xy, dtype=np.float32).reshape(2) - np.asarray(
        [float(robot_pose.x), float(robot_pose.y)],
        dtype=np.float32,
    )
    return float(np.linalg.norm(delta))


def _is_observation_transient(observation: TargetObservation) -> bool:
    reason = str(observation.reason).strip().lower()
    source = str(observation.source).strip().lower()
    return (
        "reused" in source
        or reason
        in {
            "reused_last_observation",
            "grounding_request_started",
            "grounding_request_in_flight",
            "grounding_call_period",
            "rgb_unavailable",
            "image_prepare_failed",
        }
    )


def _format_optional_float(value: float | None) -> str:
    if value is None or not np.isfinite(float(value)):
        return "none"
    return f"{float(value):.2f}"


def _format_xy(value: list[float] | tuple[float, float] | None) -> str:
    if value is None:
        return "none"
    return f"[{float(value[0]):.2f},{float(value[1]):.2f}]"
