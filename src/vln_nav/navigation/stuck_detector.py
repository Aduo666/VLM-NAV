from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .observation_adapter import VLNObservation


@dataclass
class StuckDetectorConfig:
    window_sec: float
    displacement_threshold_m: float
    yaw_threshold_deg: float
    recovery_pause_sec: float
    recovery_reverse_sec: float
    recovery_turn_sec: float
    recovery_turn_action: str


@dataclass
class StuckEvent:
    triggered: bool
    reason: str = ""
    displacement_m: float = 0.0
    yaw_delta_deg: float = 0.0


@dataclass
class RecoveryCommand:
    active: bool
    action_name: str = "STOP"
    speed_scale: float = 0.0
    reason: str = ""


class StuckDetector:
    def __init__(self, *, config: StuckDetectorConfig, logger) -> None:
        self._config = config
        self._logger = logger.getChild("stuck_detector")
        self._samples: deque[tuple[float, float, float, float, str, float]] = deque()
        self._recovery_started_sec: float | None = None
        self._recovery_turn_action = _sanitize_turn_action(config.recovery_turn_action)
        self._cooldown_until_sec = 0.0

    def update(
        self,
        *,
        observation: VLNObservation,
        executed_action: str,
        speed_scale: float,
    ) -> StuckEvent:
        if self.recovery_active:
            self._samples.clear()
            return StuckEvent(triggered=False)

        now_sec = float(observation.meta.get("timestamp_sec", 0.0))
        if now_sec < self._cooldown_until_sec:
            return StuckEvent(triggered=False)

        action = str(executed_action).strip().upper()
        if action not in {"FORWARD", "TURN_LEFT", "TURN_RIGHT"} or float(speed_scale) <= 0.05:
            self._samples.clear()
            return StuckEvent(triggered=False)

        self._samples.append(
            (
                now_sec,
                float(observation.pose.x),
                float(observation.pose.y),
                float(observation.pose.yaw_rad),
                action,
                float(speed_scale),
            )
        )
        window_sec = max(0.2, float(self._config.window_sec))
        while self._samples and now_sec - self._samples[0][0] > window_sec:
            self._samples.popleft()
        if len(self._samples) < 2 or now_sec - self._samples[0][0] < window_sec:
            return StuckEvent(triggered=False)

        first = self._samples[0]
        last = self._samples[-1]
        displacement_m = float(np.hypot(float(last[1]) - float(first[1]), float(last[2]) - float(first[2])))
        yaw_delta_deg = abs(float(np.degrees(_wrap_angle_rad(float(last[3]) - float(first[3])))))

        linear_commands = [sample for sample in self._samples if sample[4] == "FORWARD"]
        turn_commands = [sample for sample in self._samples if sample[4] in {"TURN_LEFT", "TURN_RIGHT"}]
        if linear_commands and displacement_m < max(0.0, float(self._config.displacement_threshold_m)):
            reason = "linear_command_no_displacement"
        elif turn_commands and not linear_commands and yaw_delta_deg < max(0.0, float(self._config.yaw_threshold_deg)):
            reason = "turn_command_no_yaw"
        else:
            return StuckEvent(triggered=False)

        self._samples.clear()
        self._recovery_started_sec = now_sec
        self._cooldown_until_sec = (
            now_sec
            + window_sec
            + float(self._config.recovery_pause_sec)
            + float(self._config.recovery_reverse_sec)
            + float(self._config.recovery_turn_sec)
        )
        self._logger.warning(
            "[RECOVERY] reason=%s displacement=%.3fm yaw_delta=%.1fdeg sequence=PAUSE+BACKWARD+%s",
            reason,
            displacement_m,
            yaw_delta_deg,
            self._recovery_turn_action,
        )
        return StuckEvent(
            triggered=True,
            reason=reason,
            displacement_m=displacement_m,
            yaw_delta_deg=yaw_delta_deg,
        )

    @property
    def recovery_active(self) -> bool:
        return self._recovery_started_sec is not None

    def trigger_recovery(self, *, now_sec: float, reason: str) -> None:
        if self._recovery_started_sec is not None:
            return
        self._samples.clear()
        self._recovery_started_sec = float(now_sec)
        self._logger.warning(
            "[RECOVERY] reason=%s sequence=PAUSE+BACKWARD+%s",
            str(reason),
            self._recovery_turn_action,
        )

    def recovery_command(self, *, now_sec: float) -> RecoveryCommand:
        if self._recovery_started_sec is None:
            return RecoveryCommand(active=False)

        elapsed_sec = max(0.0, float(now_sec) - float(self._recovery_started_sec))
        pause_sec = max(0.0, float(self._config.recovery_pause_sec))
        reverse_sec = max(0.0, float(self._config.recovery_reverse_sec))
        turn_sec = max(0.0, float(self._config.recovery_turn_sec))
        if elapsed_sec < pause_sec:
            return RecoveryCommand(
                active=True,
                action_name="STOP",
                speed_scale=0.0,
                reason="stuck_recovery_pause",
            )
        if elapsed_sec < pause_sec + reverse_sec:
            return RecoveryCommand(
                active=True,
                action_name="BACKWARD",
                speed_scale=0.65,
                reason="stuck_recovery_reverse",
            )
        if elapsed_sec < pause_sec + reverse_sec + turn_sec:
            return RecoveryCommand(
                active=True,
                action_name=self._recovery_turn_action,
                speed_scale=1.0,
                reason="stuck_recovery_turn",
            )

        self._recovery_started_sec = None
        return RecoveryCommand(
            active=True,
            action_name="STOP",
            speed_scale=0.0,
            reason="stuck_recovery_complete",
        )


def _sanitize_turn_action(value: str) -> str:
    action = str(value).strip().upper()
    if action in {"TURN_LEFT", "LEFT"}:
        return "TURN_LEFT"
    return "TURN_RIGHT"


def _wrap_angle_rad(angle_rad: float) -> float:
    return float((float(angle_rad) + np.pi) % (2.0 * np.pi) - np.pi)
