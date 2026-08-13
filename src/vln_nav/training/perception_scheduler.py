from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PerceptionSchedulerConfig:
    min_period_sec: float = 0.50
    max_period_sec: float = 2.00
    translation_trigger_m: float = 0.35
    yaw_trigger_deg: float = 12.0
    tracker_refresh_sec: float = 1.25


@dataclass(frozen=True)
class PerceptionScheduleDecision:
    should_run: bool
    reason: str


class PerceptionScheduler:
    """Trigger offline perception from motion and freshness, never policy replans."""

    def __init__(self, config: PerceptionSchedulerConfig | None = None) -> None:
        self.config = config or PerceptionSchedulerConfig()
        self._last_request_time_sec = -float("inf")
        self._last_pose_xyyaw: tuple[float, float, float] | None = None
        self._last_query_signature: tuple[str, ...] = ()
        self._last_positive_time_sec = -float("inf")

    def decide(
        self,
        *,
        timestamp_sec: float,
        pose_xyyaw: tuple[float, float, float],
        query_labels: tuple[str, ...],
    ) -> PerceptionScheduleDecision:
        now = float(timestamp_sec)
        signature = tuple(str(label).strip().lower() for label in query_labels)
        elapsed = now - self._last_request_time_sec
        if self._last_pose_xyyaw is None:
            return PerceptionScheduleDecision(True, "first_frame")
        if signature != self._last_query_signature:
            return PerceptionScheduleDecision(True, "query_changed")
        if elapsed < max(0.0, float(self.config.min_period_sec)):
            return PerceptionScheduleDecision(False, "min_period")
        previous = np.asarray(self._last_pose_xyyaw, dtype=np.float64)
        current = np.asarray(pose_xyyaw, dtype=np.float64)
        translation = float(np.linalg.norm(current[:2] - previous[:2]))
        yaw_delta = abs(float(_wrap_angle(current[2] - previous[2])))
        if translation >= max(0.0, float(self.config.translation_trigger_m)):
            return PerceptionScheduleDecision(True, "translation")
        if yaw_delta >= float(np.deg2rad(max(0.0, float(self.config.yaw_trigger_deg)))):
            return PerceptionScheduleDecision(True, "rotation")
        if (
            np.isfinite(self._last_positive_time_sec)
            and now - self._last_positive_time_sec
            >= max(0.0, float(self.config.tracker_refresh_sec))
        ):
            return PerceptionScheduleDecision(True, "tracker_refresh")
        if elapsed >= max(float(self.config.min_period_sec), float(self.config.max_period_sec)):
            return PerceptionScheduleDecision(True, "max_period")
        return PerceptionScheduleDecision(False, "motion_below_threshold")

    def record_request(
        self,
        *,
        timestamp_sec: float,
        pose_xyyaw: tuple[float, float, float],
        query_labels: tuple[str, ...],
        had_detection: bool,
    ) -> None:
        self._last_request_time_sec = float(timestamp_sec)
        self._last_pose_xyyaw = tuple(float(value) for value in pose_xyyaw)
        self._last_query_signature = tuple(str(label).strip().lower() for label in query_labels)
        if had_detection:
            self._last_positive_time_sec = float(timestamp_sec)


def _wrap_angle(angle_rad: float) -> float:
    return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))
