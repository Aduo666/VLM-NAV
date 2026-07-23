from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .candidate_waypoint_predictor import CandidateSet
from .observation_adapter import Pose2D, VLNObservation


@dataclass
class TraverseProgressState:
    step_id: str = ""
    active: bool = False
    start_x: float = 0.0
    start_y: float = 0.0
    start_yaw_rad: float = 0.0
    locked_heading_rad: float = 0.0
    corrected_heading_rad: float = 0.0
    measured_progress_m: float = 0.0
    cue_label: str = ""
    cue_expected_side: str = "none"
    cue_anchor_world_xy: list[float] | None = None
    cue_confirmed: bool = False
    cue_passed: bool = False
    previous_anchor_projection_m: float | None = None
    review_stage: str = "NONE"
    review_confirmations: int = 0
    last_review_progress_m: float | None = None
    recovery_attempts: int = 0
    last_pose_x: float = 0.0
    last_pose_y: float = 0.0
    last_pose_yaw_rad: float = 0.0


@dataclass
class ProgressState:
    step_index: int
    recent_observation_summaries: list[str] = field(default_factory=list)
    recent_candidate_history: list[str] = field(default_factory=list)
    recent_action_history: list[str] = field(default_factory=list)
    recent_motion_history: list[str] = field(default_factory=list)
    stalled_steps: int = 0
    net_displacement_m: float = 0.0
    progress_signal: str = "NO_PROGRESS"
    history_text: str = "none"
    last_reasoning_summary: str = "none"
    traverse: TraverseProgressState = field(default_factory=TraverseProgressState)


class ProgressManager:
    def __init__(
        self,
        *,
        history_length: int,
        stall_distance_m: float,
        spin_yaw_deg: float,
    ) -> None:
        self._history_length = max(3, int(history_length))
        self._stall_distance_m = float(stall_distance_m)
        self._spin_yaw_rad = float(np.deg2rad(spin_yaw_deg))
        self._observation_history: deque[str] = deque(maxlen=self._history_length)
        self._candidate_history: deque[str] = deque(maxlen=self._history_length)
        self._action_history: deque[str] = deque(maxlen=self._history_length)
        self._motion_history: deque[str] = deque(maxlen=self._history_length)
        self._state = ProgressState(step_index=0)
        self._reference_pose: np.ndarray | None = None
        self._last_pose: np.ndarray | None = None
        self._last_yaw: float | None = None
        self._last_update_time_sec: float | None = None
        self._min_update_period_sec = 0.75
        self._traverse = TraverseProgressState()

    @property
    def state(self) -> ProgressState:
        return self._state

    def update(
        self,
        *,
        observation: VLNObservation,
        candidate_set: CandidateSet | None,
        selected_candidate_label: str,
        applied_action: str,
        reasoning_summary: str,
    ) -> ProgressState:
        timestamp_sec = float(observation.meta.get("timestamp_sec", 0.0))
        if (
            self._last_update_time_sec is not None
            and timestamp_sec - self._last_update_time_sec < self._min_update_period_sec
        ):
            return self._state

        pose = np.asarray([observation.pose.x, observation.pose.y], dtype=np.float32)
        yaw = float(observation.pose.yaw_rad)

        if self._reference_pose is None:
            self._reference_pose = pose.copy()
        if self._last_pose is None:
            self._last_pose = pose.copy()
        if self._last_yaw is None:
            self._last_yaw = yaw

        delta_xy = pose - self._last_pose
        delta_distance = float(np.linalg.norm(delta_xy))
        delta_yaw = float(_wrap_angle_rad(yaw - self._last_yaw))
        net_displacement = float(np.linalg.norm(pose - self._reference_pose))

        valid_count = int(candidate_set.valid_count) if candidate_set is not None else 0
        self._observation_history.append(
            f"step={self._state.step_index} pose=({observation.pose.x:.2f},{observation.pose.y:.2f}) "
            f"yaw_deg={np.degrees(observation.pose.yaw_rad):.1f} valid_candidates={valid_count}"
        )
        self._candidate_history.append(f"selected={selected_candidate_label} valid_count={valid_count}")
        self._action_history.append(
            f"action={applied_action} wheel={observation.motion_feedback.wheel_velocity:.2f} "
            f"steer={observation.motion_feedback.steering_angle:.2f}"
        )
        self._motion_history.append(
            f"delta_m={delta_distance:.3f} delta_yaw_deg={np.degrees(delta_yaw):.1f}"
        )

        recent_motion = list(self._motion_history)
        recent_delta_distance = delta_distance
        accumulated_abs_yaw = abs(delta_yaw)
        if len(recent_motion) >= 2:
            accumulated_abs_yaw = 0.0
            recent_delta_distance = 0.0
            for item in recent_motion[-min(3, len(recent_motion)) :]:
                parts = item.split()
                recent_delta_distance += float(parts[0].split("=")[1])
                accumulated_abs_yaw += abs(np.deg2rad(float(parts[1].split("=")[1])))

        if recent_delta_distance < self._stall_distance_m and accumulated_abs_yaw >= self._spin_yaw_rad:
            progress_signal = "SPINNING"
        elif recent_delta_distance < self._stall_distance_m:
            progress_signal = "STALLED"
        elif net_displacement > self._stall_distance_m:
            progress_signal = "ADVANCING"
        else:
            progress_signal = "NO_PROGRESS"

        stalled_steps = int(self._state.stalled_steps)
        stalled_steps = stalled_steps + 1 if progress_signal in {"STALLED", "SPINNING"} else 0

        self._state = ProgressState(
            step_index=self._state.step_index + 1,
            recent_observation_summaries=list(self._observation_history),
            recent_candidate_history=list(self._candidate_history),
            recent_action_history=list(self._action_history),
            recent_motion_history=list(self._motion_history),
            stalled_steps=stalled_steps,
            net_displacement_m=net_displacement,
            progress_signal=progress_signal,
            history_text=self._compose_history_text(),
            last_reasoning_summary=str(reasoning_summary).strip() or "none",
            traverse=self._traverse,
        )

        self._last_pose = pose.copy()
        self._last_yaw = yaw
        self._last_update_time_sec = timestamp_sec
        return self._state

    def bootstrap(self, observation: VLNObservation) -> ProgressState:
        pose = np.asarray([observation.pose.x, observation.pose.y], dtype=np.float32)
        self._reference_pose = pose.copy()
        self._last_pose = pose.copy()
        self._last_yaw = float(observation.pose.yaw_rad)
        self._last_update_time_sec = float(observation.meta.get("timestamp_sec", 0.0))
        self._state = ProgressState(step_index=0, traverse=self._traverse)
        return self._state

    def begin_traverse(
        self,
        *,
        step_id: str,
        robot_pose: Pose2D,
        cue_label: str,
        cue_expected_side: str,
    ) -> TraverseProgressState:
        if self._traverse.active and self._traverse.step_id == str(step_id):
            return self._traverse
        self._traverse = TraverseProgressState(
            step_id=str(step_id),
            active=True,
            start_x=float(robot_pose.x),
            start_y=float(robot_pose.y),
            start_yaw_rad=float(robot_pose.yaw_rad),
            locked_heading_rad=float(robot_pose.yaw_rad),
            corrected_heading_rad=float(robot_pose.yaw_rad),
            cue_label=str(cue_label).strip().lower(),
            cue_expected_side=str(cue_expected_side).strip().lower() or "none",
            last_pose_x=float(robot_pose.x),
            last_pose_y=float(robot_pose.y),
            last_pose_yaw_rad=float(robot_pose.yaw_rad),
        )
        self._state.traverse = self._traverse
        return self._traverse

    def update_traverse_motion(self, *, robot_pose: Pose2D) -> TraverseProgressState:
        if not self._traverse.active:
            return self._traverse
        dx = float(robot_pose.x) - float(self._traverse.start_x)
        dy = float(robot_pose.y) - float(self._traverse.start_y)
        heading = float(self._traverse.locked_heading_rad)
        self._traverse.measured_progress_m = float(
            dx * np.cos(heading) + dy * np.sin(heading)
        )
        self._traverse.last_pose_x = float(robot_pose.x)
        self._traverse.last_pose_y = float(robot_pose.y)
        self._traverse.last_pose_yaw_rad = float(robot_pose.yaw_rad)
        self._state.traverse = self._traverse
        return self._traverse

    def update_traverse_heading(self, *, corrected_heading_rad: float) -> None:
        if not self._traverse.active:
            return
        self._traverse.corrected_heading_rad = float(corrected_heading_rad)

    def update_traverse_cue(
        self,
        *,
        robot_pose: Pose2D,
        anchor_world_xy: list[float] | None,
        confirmed: bool,
        pass_margin_m: float,
        min_progress_m: float,
    ) -> TraverseProgressState:
        if not self._traverse.active:
            return self._traverse
        self._traverse.cue_confirmed = bool(confirmed and anchor_world_xy is not None)
        if not self._traverse.cue_confirmed:
            return self._traverse
        anchor = np.asarray(anchor_world_xy, dtype=np.float64).reshape(2)
        robot = np.asarray([float(robot_pose.x), float(robot_pose.y)], dtype=np.float64)
        heading = float(self._traverse.locked_heading_rad)
        forward_axis = np.asarray([np.cos(heading), np.sin(heading)], dtype=np.float64)
        projection_m = float(np.dot(robot - anchor, forward_axis))
        previous = self._traverse.previous_anchor_projection_m
        self._traverse.cue_anchor_world_xy = [float(anchor[0]), float(anchor[1])]
        crossed_plane = bool(
            previous is not None
            and float(previous) < 0.0
            and projection_m >= max(0.05, float(pass_margin_m))
        )
        first_confirmed_after_crossing = bool(
            previous is None
            and projection_m >= max(0.05, float(pass_margin_m))
        )
        if (
            (crossed_plane or first_confirmed_after_crossing)
            and float(self._traverse.measured_progress_m) >= max(0.0, float(min_progress_m))
        ):
            self._traverse.cue_passed = True
        self._traverse.previous_anchor_projection_m = projection_m
        self._state.traverse = self._traverse
        return self._traverse

    def record_traverse_review(
        self,
        *,
        likely_complete: bool,
        min_translation_m: float,
    ) -> TraverseProgressState:
        if not self._traverse.active:
            return self._traverse
        if bool(likely_complete):
            if (
                self._traverse.last_review_progress_m is not None
                and abs(
                    float(self._traverse.measured_progress_m)
                    - float(self._traverse.last_review_progress_m)
                )
                < max(0.05, float(min_translation_m))
            ):
                return self._traverse
            self._traverse.review_confirmations += 1
            self._traverse.last_review_progress_m = float(
                self._traverse.measured_progress_m
            )
            self._traverse.review_stage = (
                "CONFIRMED"
                if self._traverse.review_confirmations >= 2
                else "LIKELY_COMPLETE"
            )
        else:
            self._traverse.review_confirmations = 0
            self._traverse.last_review_progress_m = None
            self._traverse.review_stage = "CONTINUE"
        return self._traverse

    def increment_traverse_recovery(self) -> int:
        if not self._traverse.active:
            return 0
        self._traverse.recovery_attempts += 1
        return int(self._traverse.recovery_attempts)

    def finish_traverse(self) -> None:
        self._traverse.active = False
        self._state.traverse = self._traverse

    def _compose_history_text(self) -> str:
        lines = []
        for obs, cand, action, motion in zip(
            self._observation_history,
            self._candidate_history,
            self._action_history,
            self._motion_history,
        ):
            lines.append(f"{obs}; {cand}; {action}; {motion}")
        return " | ".join(lines) if lines else "none"


def _wrap_angle_rad(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)
