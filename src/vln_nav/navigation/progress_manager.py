from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .candidate_waypoint_predictor import CandidateSet
from .observation_adapter import VLNObservation


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

        delta_distance = float(np.linalg.norm(pose - self._last_pose))
        delta_yaw = float(_wrap_angle_rad(yaw - self._last_yaw))
        net_displacement = float(np.linalg.norm(pose - self._reference_pose))
        valid_count = int(candidate_set.valid_count) if candidate_set is not None else 0

        self._observation_history.append(
            f"step={self._state.step_index} pose=({observation.pose.x:.2f},{observation.pose.y:.2f}) "
            f"yaw_deg={np.degrees(observation.pose.yaw_rad):.1f} valid_candidates={valid_count}"
        )
        self._candidate_history.append(
            f"selected={selected_candidate_label} valid_count={valid_count}"
        )
        self._action_history.append(
            f"action={applied_action} wheel={observation.motion_feedback.wheel_velocity:.2f} "
            f"steer={observation.motion_feedback.steering_angle:.2f}"
        )
        self._motion_history.append(
            f"delta_m={delta_distance:.3f} delta_yaw_deg={np.degrees(delta_yaw):.1f}"
        )

        recent_delta_distance = 0.0
        accumulated_abs_yaw = 0.0
        for item in list(self._motion_history)[-3:]:
            parts = item.split()
            recent_delta_distance += float(parts[0].split("=")[1])
            accumulated_abs_yaw += abs(
                np.deg2rad(float(parts[1].split("=")[1]))
            )

        if (
            recent_delta_distance < self._stall_distance_m
            and accumulated_abs_yaw >= self._spin_yaw_rad
        ):
            progress_signal = "SPINNING"
        elif recent_delta_distance < self._stall_distance_m:
            progress_signal = "STALLED"
        elif net_displacement > self._stall_distance_m:
            progress_signal = "ADVANCING"
        else:
            progress_signal = "NO_PROGRESS"

        stalled_steps = int(self._state.stalled_steps)
        stalled_steps = (
            stalled_steps + 1
            if progress_signal in {"STALLED", "SPINNING"}
            else 0
        )
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
        self._last_update_time_sec = float(
            observation.meta.get("timestamp_sec", 0.0)
        )
        self._state = ProgressState(step_index=0)
        return self._state

    def _compose_history_text(self) -> str:
        lines = []
        for observation, candidate, action, motion in zip(
            self._observation_history,
            self._candidate_history,
            self._action_history,
            self._motion_history,
        ):
            lines.append(f"{observation}; {candidate}; {action}; {motion}")
        return " | ".join(lines) if lines else "none"


def _wrap_angle_rad(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)
