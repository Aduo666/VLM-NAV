from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

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
    reach_window_hits: int = 0
    reach_window_size: int = 0
    recent_reach_history: list[int] = field(default_factory=list)
    potential_world_xy: list[float] | None = None
    potential_confidence: float = 0.0
    potential_bbox: tuple[int, int, int, int] | None = None
    potential_center_uv: tuple[float, float] | None = None
    potential_bearing_rad: float | None = None
    potential_last_seen_time_sec: float | None = None
    potential_source: str = "unknown"
    confirmed_confidence: float = 0.0
    confirmed_observation_count: int = 0
    last_confirmed_depth_m: float | None = None
    evidence_level: str = "none"  # none / potential / confirmed
    can_complete: bool = False


@dataclass
class LandmarkTrackerConfig:
    arrival_visible_depth_m: float
    arrival_anchor_radius_m: float
    min_visible_count: int
    arrival_confirmations: int
    max_lost_count: int
    anchor_update_min_consistent_observations: int
    anchor_update_max_jump_m: float
    arrival_window_size: int
    arrival_recent_visible_sec: float = 3.0
    confirm_confidence_threshold: float = 0.50
    confirm_min_observations: int = 3
    confirm_max_anchor_jump_m: float = 0.50
    confirm_max_depth_jump_m: float = 0.80
    potential_confidence_threshold: float = 0.50


@dataclass
class _PendingAnchor:
    xy_sum: np.ndarray
    count: int
    last_xy: np.ndarray


@dataclass
class _PendingConfirmedCandidate:
    xy_sum: np.ndarray
    count: int
    last_xy: np.ndarray
    last_depth_m: float | None = None


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
        self._pending_confirmed: dict[str, _PendingConfirmedCandidate] = {}
        self._last_log_time_by_label: dict[str, float] = {}
        self._last_log_signature_by_label: dict[str, tuple] = {}
        self._last_anchor_log_time_by_label: dict[str, float] = {}
        self._last_anchor_log_xy_by_label: dict[str, list[float]] = {}
        self._reach_windows: dict[str, deque[bool]] = {}

    def reset(self, label: str | None = None) -> None:
        if label is None:
            self._states.clear()
            self._pending_anchors.clear()
            self._pending_confirmed.clear()
            self._reach_windows.clear()
            return
        key = _normalize_label(label)
        self._states.pop(key, None)
        self._pending_anchors.pop(key, None)
        self._pending_confirmed.pop(key, None)
        self._reach_windows.pop(key, None)

    def update(
        self,
        *,
        target_label: str,
        behavior_type: str,
        observation: TargetObservation,
        robot_pose: Pose2D,
        timestamp_sec: float,
        arrival_distance_m: float | None = None,
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
                reach_window_hits=0,
                reach_window_size=max(1, int(self._config.arrival_window_size)),
                recent_reach_history=[],
                potential_world_xy=None,
                potential_confidence=0.0,
                potential_bbox=None,
                potential_center_uv=None,
                potential_bearing_rad=None,
                potential_last_seen_time_sec=None,
                potential_source="unknown",
                confirmed_confidence=0.0,
                confirmed_observation_count=0,
                last_confirmed_depth_m=None,
                evidence_level="none",
                can_complete=False,
            )

        is_ok = str(observation.reason).strip().lower() == "ok"
        is_depth_invalid = str(observation.reason).strip().lower() == "depth_invalid"
        is_reused = "reused" in str(observation.source).strip().lower()
        fresh_visible = bool(observation.visible) and (is_ok or is_depth_invalid) and not is_reused
        has_valid_depth = fresh_visible and is_ok and observation.depth_m is not None
        current_fresh = has_valid_depth and bool(getattr(observation, "is_current_frame", True))
        transient_observation = _is_observation_transient(observation)

        if bool(observation.visible):
            if fresh_visible:
                state.visible_count += 1
            state.lost_count = 0
            state.last_observation = observation
            state.last_seen_time_sec = float(timestamp_sec)

            # --- Image-level potential (depth_invalid ok, no world_xy) ---
            has_bbox = observation.bbox is not None
            meets_potential_conf = float(observation.confidence) >= float(self._config.potential_confidence_threshold)
            if has_bbox and meets_potential_conf and not is_reused:
                state.potential_bbox = tuple(observation.bbox) if observation.bbox is not None else None
                state.potential_center_uv = (
                    None if observation.center_uv is None
                    else (float(observation.center_uv[0]), float(observation.center_uv[1]))
                )
                state.potential_bearing_rad = (
                    None if observation.bearing_rad is None else float(observation.bearing_rad)
                )
                state.potential_confidence = max(float(state.potential_confidence), float(observation.confidence))
                state.potential_last_seen_time_sec = float(timestamp_sec)
                state.potential_source = str(getattr(observation, "semantic_source", "unknown"))
                if is_depth_invalid:
                    self._logger.info(
                        "[LandmarkPotentialImageOnly] target=%s reason=depth_invalid bbox=%s center=%s confidence=%.2f",
                        label,
                        list(observation.bbox),
                        _format_xy(observation.center_uv) if observation.center_uv is not None else "none",
                        float(observation.confidence),
                    )

            # --- World-level potential (only ok + depth + world_xy + threshold) ---
            can_update_world_potential = (
                is_ok
                and bool(getattr(observation, "is_current_frame", True))
                and observation.depth_m is not None
                and observation.estimated_world_xy is not None
                and meets_potential_conf
            )
            if can_update_world_potential:
                state.potential_world_xy = list(observation.estimated_world_xy)
                if state.evidence_level == "none":
                    state.evidence_level = "potential"
            elif fresh_visible and not is_depth_invalid and not is_reused:
                # Log why world potential was not updated
                reasons = []
                if not is_ok:
                    reasons.append("reason_not_ok")
                if not bool(getattr(observation, "is_current_frame", True)):
                    reasons.append("stale")
                if observation.depth_m is None:
                    reasons.append("no_depth")
                if observation.estimated_world_xy is None:
                    reasons.append("no_world_xy")
                if not meets_potential_conf:
                    reasons.append("low_confidence")
                if reasons:
                    self._logger.info(
                        "[LandmarkPotentialReject] target=%s reason=%s confidence=%.2f",
                        label,
                        "_".join(reasons),
                        float(observation.confidence),
                    )

            state.candidate_world_xy = (
                None if observation.estimated_world_xy is None else list(observation.estimated_world_xy)
            )

            # Only high-confidence fresh current-frame from confirmable source enters confirmed pipeline
            high_conf = float(observation.confidence) >= float(self._config.confirm_confidence_threshold)
            semantic_source = str(getattr(observation, "semantic_source", "")).strip().lower()
            can_confirm = bool(
                getattr(observation, "can_confirm", False)
                and semantic_source.startswith("grounding_dino")
            )
            has_world = observation.estimated_world_xy is not None

            if current_fresh and has_world and high_conf and can_confirm:
                self._try_confirm_candidate(
                    label=label,
                    state=state,
                    observation=observation,
                )
            elif current_fresh and has_world and semantic_source != "opencv_tracker":
                reasons = []
                if not high_conf:
                    reasons.append("low_confidence")
                if not can_confirm:
                    reasons.append("semantic_source_not_confirmable")
                if reasons:
                    self._logger.info(
                        "[LandmarkConfirmReject] target=%s reason=%s confidence=%.2f can_confirm=%s source=%s",
                        label,
                        "_".join(reasons),
                        float(observation.confidence),
                        int(can_confirm),
                        getattr(observation, "semantic_source", "unknown"),
                    )

            state.status = "visible"
            if state.evidence_level == "confirmed":
                state.status = "confirmed"

        elif transient_observation:
            state.last_observation = observation
            if state.evidence_level == "confirmed" and state.estimated_world_xy is not None:
                state.status = "confirmed"
            elif int(state.visible_count) > 0:
                state.status = "visible"
            else:
                state.status = "searching"
        else:
            state.visible_count = 0
            state.lost_count += 1
            if state.evidence_level == "confirmed" and state.estimated_world_xy is not None:
                state.status = "lost"
            else:
                state.status = "searching"

        # Update can_complete
        state.can_complete = (
            state.estimated_world_xy is not None
            and state.confirmed_observation_count >= int(self._config.confirm_min_observations)
            and state.evidence_level == "confirmed"
        )

        state.anchor_distance_m = _distance_to_anchor(
            robot_pose=robot_pose,
            anchor_world_xy=state.estimated_world_xy,
        )
        was_reached = bool(state.reached)
        reach_result = self._update_reach_state(
            label=label,
            state=state,
            observation=observation,
            arrival_distance_m=arrival_distance_m,
            timestamp_sec=float(timestamp_sec),
        )
        state.reached = bool(was_reached or reach_result[0])
        state.reach_reason = state.reach_reason if was_reached else reach_result[1]
        state.reached_count = max(int(state.reached_count), int(reach_result[2]))
        state.reach_window_hits = int(reach_result[3])
        state.reach_window_size = int(reach_result[4])
        state.recent_reach_history = list(reach_result[5])
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
            lost_bucket,
            _format_xy_bucket(state.estimated_world_xy, bucket_m=0.10),
            bool(state.reached),
            state.reach_reason,
            observation.reason,
            state.evidence_level,
        )
        last_time = self._last_log_time_by_label.get(label, -float("inf"))
        last_signature = self._last_log_signature_by_label.get(label)
        if signature == last_signature and float(timestamp_sec) - float(last_time) < 3.0:
            return
        self._last_log_time_by_label[label] = float(timestamp_sec)
        self._last_log_signature_by_label[label] = signature
        can_confirm = bool(getattr(observation, "can_confirm", False))
        self._logger.info(
            "[LandmarkTracker] target=%s behavior=%s status=%s visible=%s visible_count=%s lost_count=%s "
            "depth=%s candidate_anchor=%s confirmed_anchor=%s anchor_dist=%s reached=%s confirm=%s obs_source=%s reason=%s "
            "evidence=%s can_confirm=%s can_complete=%s potential_anchor=%s potential_image_bbox=%s "
            "potential_bearing=%s confirmed_obs=%s confidence=%.2f",
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
            f"{int(state.reach_window_hits)}/{int(state.reach_window_size)}",
            observation.source,
            state.reach_reason,
            state.evidence_level,
            int(can_confirm),
            int(state.can_complete),
            _format_xy(state.potential_world_xy),
            list(state.potential_bbox) if state.potential_bbox is not None else "none",
            _format_optional_float(None if state.potential_bearing_rad is None else float(np.degrees(state.potential_bearing_rad))),
            int(state.confirmed_observation_count),
            float(state.confidence),
        )

    def _try_confirm_candidate(
        self,
        *,
        label: str,
        state: LandmarkState,
        observation: TargetObservation,
    ) -> None:
        candidate = np.asarray(observation.estimated_world_xy, dtype=np.float32).reshape(2)
        confirm_max_anchor_jump = max(0.05, float(self._config.confirm_max_anchor_jump_m))
        confirm_max_depth_jump = max(0.05, float(self._config.confirm_max_depth_jump_m))
        required = max(1, int(self._config.confirm_min_observations))

        pending = self._pending_confirmed.get(label)
        if pending is None:
            pending = _PendingConfirmedCandidate(
                xy_sum=candidate.copy(), count=1, last_xy=candidate.copy(),
                last_depth_m=observation.depth_m,
            )
            self._pending_confirmed[label] = pending
            if required <= 1:
                state.estimated_world_xy = [float(candidate[0]), float(candidate[1])]
                state.confirmed_observation_count = 1
                state.confirmed_confidence = float(observation.confidence)
                state.last_confirmed_depth_m = observation.depth_m
                state.evidence_level = "confirmed"
                state.confidence = float(observation.confidence)
                self._pending_confirmed.pop(label, None)
                self._logger.info(
                    "[LandmarkConfirmed] target=%s anchor=%s obs=1 confidence=%.2f depth=%.2f",
                    label,
                    _format_xy(state.estimated_world_xy),
                    float(state.confirmed_confidence),
                    float(observation.depth_m) if observation.depth_m is not None else 0.0,
                )
                return
            self._logger.info(
                "[LandmarkConfirm] target=%s candidate=%s confirm_wait=1/%s",
                label, _format_xy(candidate.tolist()), int(required),
            )
            return

        # Check anchor jump from last pending candidate
        jump = float(np.linalg.norm(candidate - pending.last_xy))
        if jump > confirm_max_anchor_jump:
            self._logger.info(
                "[LandmarkConfirmReject] target=%s reason=anchor_jump jump_m=%.2f max=%.2f",
                label, jump, confirm_max_anchor_jump,
            )
            # Reset pending
            self._pending_confirmed[label] = _PendingConfirmedCandidate(
                xy_sum=candidate.copy(), count=1, last_xy=candidate.copy(),
                last_depth_m=observation.depth_m,
            )
            return

        # Check depth jump
        if pending.last_depth_m is not None and observation.depth_m is not None:
            depth_jump = abs(float(observation.depth_m) - float(pending.last_depth_m))
            if depth_jump > confirm_max_depth_jump:
                self._logger.info(
                    "[LandmarkConfirmReject] target=%s reason=depth_jump jump_m=%.2f max=%.2f",
                    label, depth_jump, confirm_max_depth_jump,
                )
                self._pending_confirmed[label] = _PendingConfirmedCandidate(
                    xy_sum=candidate.copy(), count=1, last_xy=candidate.copy(),
                    last_depth_m=observation.depth_m,
                )
                return

        # Accumulate
        pending.xy_sum = pending.xy_sum + candidate
        pending.count += 1
        pending.last_xy = candidate.copy()
        pending.last_depth_m = observation.depth_m

        if pending.count < required:
            self._logger.info(
                "[LandmarkConfirm] target=%s candidate=%s confirm_wait=%s/%s",
                label, _format_xy(candidate.tolist()), int(pending.count), int(required),
            )
            return

        # Confirmed!
        averaged = pending.xy_sum / float(max(1, pending.count))

        # Check jump from existing confirmed anchor
        if state.estimated_world_xy is not None:
            current = np.asarray(state.estimated_world_xy, dtype=np.float32).reshape(2)
            jump_from_confirmed = float(np.linalg.norm(averaged - current))
            if jump_from_confirmed > confirm_max_anchor_jump:
                reanchor_required = max(required + 2, 5)
                if int(pending.count) < int(reanchor_required):
                    self._logger.warning(
                        "[LandmarkReanchorWait] target=%s reason=confirmed_anchor_jump jump_m=%.2f max=%.2f candidate=%s confirmed=%s confirm_wait=%s/%s",
                        label,
                        jump_from_confirmed,
                        confirm_max_anchor_jump,
                        _format_xy(averaged.tolist()),
                        _format_xy(state.estimated_world_xy),
                        int(pending.count),
                        int(reanchor_required),
                    )
                    return
                self._logger.warning(
                    "[LandmarkReanchor] target=%s reason=stable_new_anchor jump_m=%.2f old=%s new=%s observations=%s",
                    label,
                    jump_from_confirmed,
                    _format_xy(state.estimated_world_xy),
                    _format_xy(averaged.tolist()),
                    int(pending.count),
                )

        state.estimated_world_xy = [float(averaged[0]), float(averaged[1])]
        state.confirmed_observation_count = int(pending.count)
        state.confirmed_confidence = float(observation.confidence)
        state.last_confirmed_depth_m = observation.depth_m
        state.evidence_level = "confirmed"
        state.confidence = float(observation.confidence)
        self._pending_confirmed.pop(label, None)

        self._logger.info(
            "[LandmarkConfirmed] target=%s anchor=%s obs=%s confidence=%.2f depth=%.2f",
            label,
            _format_xy(state.estimated_world_xy),
            int(state.confirmed_observation_count),
            float(state.confirmed_confidence),
            float(observation.depth_m) if observation.depth_m is not None else 0.0,
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
        last_anchor_log_time = self._last_anchor_log_time_by_label.get(label, -float("inf"))
        last_anchor_xy = self._last_anchor_log_xy_by_label.get(label)
        moved_m = float("inf") if last_anchor_xy is None else float(
            np.linalg.norm(np.asarray(confirmed, dtype=np.float32) - np.asarray(last_anchor_xy, dtype=np.float32))
        )
        if moved_m >= 0.10 or float(pending.count) <= float(required) or float(pending.count) % 10 == 0 or float(last_anchor_log_time) < -1.0e20:
            self._logger.info(
                "[LandmarkAnchor] target=%s confirmed=%s observations=%s",
                label,
                _format_xy(confirmed),
                int(pending.count),
            )
            self._last_anchor_log_time_by_label[label] = 0.0
            self._last_anchor_log_xy_by_label[label] = list(confirmed)
        return confirmed

    def _update_reach_state(
        self,
        *,
        label: str,
        state: LandmarkState,
        observation: TargetObservation,
        arrival_distance_m: float | None,
        timestamp_sec: float,
    ) -> tuple[bool, str, int, int, int, list[int]]:
        if state.behavior_type != "go_to":
            return False, f"behavior_not_implemented:{state.behavior_type}", 0, 0, 0, []

        if not state.can_complete:
            window_size = max(1, int(self._config.arrival_window_size))
            return False, "not_confirmed_for_completion", 0, 0, window_size, []

        min_visible = max(1, int(self._config.min_visible_count))
        if arrival_distance_m is None:
            depth_threshold_m = max(0.05, float(self._config.arrival_visible_depth_m))
            anchor_threshold_m = max(0.05, float(self._config.arrival_anchor_radius_m))
        else:
            depth_threshold_m = max(0.05, float(arrival_distance_m))
            anchor_threshold_m = max(0.05, float(arrival_distance_m))
        required = max(1, int(self._config.arrival_confirmations))
        window_size = max(required, int(self._config.arrival_window_size))
        reach_window = self._reach_windows.get(label)
        if reach_window is None or reach_window.maxlen != window_size:
            reach_window = deque(maxlen=window_size)
            self._reach_windows[label] = reach_window
        if _is_observation_transient(observation):
            if bool(state.reached):
                hits = int(sum(1 for item in reach_window if item))
                return True, state.reach_reason, int(state.reached_count), hits, int(window_size), [int(item) for item in reach_window]
            if int(sum(1 for item in reach_window if item)) > 0:
                base_reason = str(state.reach_reason).split("; waiting_fresh_confirmation", 1)[0]
                hits = int(sum(1 for item in reach_window if item))
                return False, f"{base_reason}; waiting_fresh_confirmation", hits, hits, int(window_size), [int(item) for item in reach_window]
            return False, "not_reached", 0, 0, int(window_size), []

        semantic_arrival_observation = str(
            getattr(observation, "semantic_source", "")
        ).strip().lower().startswith("grounding_dino")
        visible_depth_reached = (
            bool(observation.visible)
            and semantic_arrival_observation
            and str(observation.reason).strip().lower() == "ok"
            and "reused" not in str(observation.source).strip().lower()
            and int(state.visible_count) >= min_visible
            and observation.depth_m is not None
            and float(observation.depth_m) <= depth_threshold_m
            and state.can_complete
        )
        anchor_reached = (
            state.estimated_world_xy is not None
            and state.anchor_distance_m is not None
            and float(state.anchor_distance_m) <= anchor_threshold_m
            and state.can_complete
            and state.last_seen_time_sec is not None
            and float(timestamp_sec) - float(state.last_seen_time_sec) <= max(0.0, float(self._config.arrival_recent_visible_sec))
        )
        recent_visible_age = None
        if state.last_seen_time_sec is not None:
            recent_visible_age = max(0.0, float(timestamp_sec) - float(state.last_seen_time_sec))

        evidence = bool(visible_depth_reached or anchor_reached)
        reach_window.append(bool(evidence))
        hits = int(sum(1 for item in reach_window if item))

        if visible_depth_reached:
            reason = f"confirmed_visible_depth<={depth_threshold_m:.2f}m"
        elif anchor_reached:
            reason = f"confirmed_anchor_distance<={anchor_threshold_m:.2f}m"
        else:
            reason = "not_reached"
        reached = bool(hits >= required)
        reached_count = hits
        self._logger.info(
            "[ReachCheck] target=%s behavior=%s reached=%s can_complete=%s distance=%s depth=%s "
            "recent_visible_age=%s reason=%s confirm=%s/%s evidence_level=%s",
            label,
            state.behavior_type,
            int(reached),
            int(bool(state.can_complete)),
            _format_optional_float(state.anchor_distance_m),
            _format_optional_float(observation.depth_m),
            _format_optional_float(recent_visible_age),
            reason,
            int(hits),
            int(window_size),
            state.evidence_level,
        )
        return reached, reason, reached_count, hits, int(window_size), [int(item) for item in reach_window]


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
            "grounding_paused",
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


def _format_xy_bucket(value: list[float] | tuple[float, float] | None, *, bucket_m: float) -> str:
    if value is None:
        return "none"
    bucket = max(0.01, float(bucket_m))
    x = round(float(value[0]) / bucket) * bucket
    y = round(float(value[1]) / bucket) * bucket
    return f"[{x:.2f},{y:.2f}]"
