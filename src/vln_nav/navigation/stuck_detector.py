from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .observation_adapter import Pose2D, VLNObservation


@dataclass
class StuckDetectorConfig:
    window_sec: float
    displacement_threshold_m: float
    yaw_threshold_deg: float
    recovery_pause_sec: float
    recovery_reverse_sec: float
    recovery_turn_sec: float


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
    completed: bool = False
    remaining_distance_m: float = 0.0
    remaining_yaw_deg: float = 0.0


@dataclass(frozen=True)
class SafePrimitiveRecord:
    action_name: str
    start_pose: Pose2D
    end_pose: Pose2D


class StuckDetector:
    """Detect no-progress motion and retrace the latest verified-safe primitive."""

    _MOTION_ACTIONS = {"FORWARD", "BACKWARD", "TURN_LEFT", "TURN_RIGHT"}

    def __init__(self, *, config: StuckDetectorConfig, logger) -> None:
        self._config = config
        self._logger = logger.getChild("stuck_detector")
        self._samples: deque[tuple[float, float, float, float, str, float]] = deque()
        self._safe_history: deque[SafePrimitiveRecord] = deque(maxlen=16)
        self._recovery_started_sec: float | None = None
        self._recovery_action = "STOP"
        self._recovery_target_pose: Pose2D | None = None
        self._cooldown_until_sec = 0.0

    def record_completed_primitive(
        self,
        *,
        action_name: str,
        start_pose: Pose2D,
        end_pose: Pose2D,
    ) -> None:
        action = str(action_name).strip().upper()
        if action not in self._MOTION_ACTIONS:
            return
        record = SafePrimitiveRecord(
            action_name=action,
            start_pose=_copy_pose(start_pose),
            end_pose=_copy_pose(end_pose),
        )
        displacement_m = _pose_distance_m(record.start_pose, record.end_pose)
        yaw_delta_deg = abs(
            float(
                np.degrees(
                    _wrap_angle_rad(record.end_pose.yaw_rad - record.start_pose.yaw_rad)
                )
            )
        )
        if displacement_m < 0.01 and yaw_delta_deg < 1.0:
            return
        self._safe_history.append(record)
        self._logger.info(
            "[PrimitiveHistory] recorded=%s displacement_m=%.2f yaw_delta_deg=%.1f depth=%s",
            action,
            displacement_m,
            yaw_delta_deg,
            len(self._safe_history),
        )

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
        if self._samples and str(self._samples[-1][4]) != action:
            self._samples.clear()

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

        if action == "FORWARD" and displacement_m < max(0.0, float(self._config.displacement_threshold_m)):
            reason = "linear_command_no_displacement"
        elif action in {"TURN_LEFT", "TURN_RIGHT"} and yaw_delta_deg < max(0.0, float(self._config.yaw_threshold_deg)):
            reason = "turn_command_no_yaw"
        else:
            return StuckEvent(triggered=False)

        interrupted_start = Pose2D(x=float(first[1]), y=float(first[2]), yaw_rad=float(first[3]))
        self._samples.clear()
        started = self.trigger_recovery(
            now_sec=now_sec,
            reason=reason,
            current_pose=observation.pose,
            interrupted_action=action,
            interrupted_start_pose=interrupted_start,
        )
        return StuckEvent(
            triggered=bool(started),
            reason=reason,
            displacement_m=displacement_m,
            yaw_delta_deg=yaw_delta_deg,
        )

    @property
    def recovery_active(self) -> bool:
        return self._recovery_started_sec is not None

    def trigger_recovery(
        self,
        *,
        now_sec: float,
        reason: str,
        current_pose: Pose2D | None = None,
        interrupted_action: str = "",
        interrupted_start_pose: Pose2D | None = None,
    ) -> bool:
        if self.recovery_active:
            return False

        rollback_record = self._interrupted_record(
            action_name=interrupted_action,
            start_pose=interrupted_start_pose,
            current_pose=current_pose,
        )
        source = "interrupted_primitive"
        if rollback_record is None and self._safe_history:
            rollback_record = self._safe_history.pop()
            source = "completed_history"
        if rollback_record is None:
            self._samples.clear()
            self._cooldown_until_sec = float(now_sec) + max(0.2, float(self._config.window_sec))
            self._logger.warning(
                "[RECOVERY] reason=%s result=no_safe_history action=STOP next=replan",
                str(reason),
            )
            return False

        self._samples.clear()
        self._recovery_started_sec = float(now_sec)
        self._recovery_action = _inverse_action(rollback_record.action_name)
        self._recovery_target_pose = _copy_pose(rollback_record.start_pose)
        self._logger.warning(
            "[RECOVERY] reason=%s source=%s undo=%s inverse=%s target=[%.2f,%.2f,%.1fdeg] sequence=PAUSE+RETRACE",
            str(reason),
            source,
            rollback_record.action_name,
            self._recovery_action,
            float(self._recovery_target_pose.x),
            float(self._recovery_target_pose.y),
            float(np.degrees(self._recovery_target_pose.yaw_rad)),
        )
        return True

    def recovery_command(
        self,
        *,
        now_sec: float,
        observation: VLNObservation | None = None,
    ) -> RecoveryCommand:
        if not self.recovery_active or self._recovery_target_pose is None:
            return RecoveryCommand(active=False)

        elapsed_sec = max(0.0, float(now_sec) - float(self._recovery_started_sec))
        pause_sec = max(0.0, float(self._config.recovery_pause_sec))
        if elapsed_sec < pause_sec:
            return RecoveryCommand(
                active=True,
                action_name="STOP",
                speed_scale=0.0,
                reason="history_retrace_pause",
            )

        current_pose = None if observation is None else observation.pose
        remaining_distance_m, remaining_yaw_deg = self._remaining_to_target(current_pose)
        if self._target_reached(
            remaining_distance_m=remaining_distance_m,
            remaining_yaw_deg=remaining_yaw_deg,
        ):
            return self._finish_recovery(
                now_sec=now_sec,
                reason="history_checkpoint_reached",
                remaining_distance_m=remaining_distance_m,
                remaining_yaw_deg=remaining_yaw_deg,
            )

        motion_elapsed_sec = elapsed_sec - pause_sec
        timeout_sec = (
            max(0.0, float(self._config.recovery_turn_sec))
            if self._recovery_action in {"TURN_LEFT", "TURN_RIGHT"}
            else max(0.0, float(self._config.recovery_reverse_sec))
        )
        if motion_elapsed_sec >= timeout_sec:
            return self._finish_recovery(
                now_sec=now_sec,
                reason="history_retrace_timeout",
                remaining_distance_m=remaining_distance_m,
                remaining_yaw_deg=remaining_yaw_deg,
            )

        return RecoveryCommand(
            active=True,
            action_name=self._recovery_action,
            speed_scale=0.65 if self._recovery_action in {"FORWARD", "BACKWARD"} else 1.0,
            reason=f"history_retrace_{self._recovery_action.lower()}",
            remaining_distance_m=remaining_distance_m,
            remaining_yaw_deg=remaining_yaw_deg,
        )

    def abort_recovery(self, *, now_sec: float, reason: str) -> RecoveryCommand:
        remaining_distance_m, remaining_yaw_deg = self._remaining_to_target(None)
        self._logger.warning(
            "[RECOVERY] aborted reason=%s inverse=%s next=replan",
            str(reason),
            self._recovery_action,
        )
        self._clear_recovery(now_sec=now_sec)
        return RecoveryCommand(
            active=True,
            action_name="STOP",
            speed_scale=0.0,
            reason=f"history_retrace_aborted:{reason}",
            completed=True,
            remaining_distance_m=remaining_distance_m,
            remaining_yaw_deg=remaining_yaw_deg,
        )

    def _interrupted_record(
        self,
        *,
        action_name: str,
        start_pose: Pose2D | None,
        current_pose: Pose2D | None,
    ) -> SafePrimitiveRecord | None:
        action = str(action_name).strip().upper()
        if action not in self._MOTION_ACTIONS or start_pose is None or current_pose is None:
            return None
        displacement_m = _pose_distance_m(start_pose, current_pose)
        yaw_delta_deg = abs(
            float(np.degrees(_wrap_angle_rad(current_pose.yaw_rad - start_pose.yaw_rad)))
        )
        moved = (
            displacement_m >= max(0.02, 0.5 * float(self._config.displacement_threshold_m))
            or yaw_delta_deg >= max(1.0, 0.5 * float(self._config.yaw_threshold_deg))
        )
        if not moved:
            return None
        return SafePrimitiveRecord(
            action_name=action,
            start_pose=_copy_pose(start_pose),
            end_pose=_copy_pose(current_pose),
        )

    def _remaining_to_target(self, current_pose: Pose2D | None) -> tuple[float, float]:
        if current_pose is None or self._recovery_target_pose is None:
            return 0.0, 0.0
        return (
            _pose_distance_m(current_pose, self._recovery_target_pose),
            abs(
                float(
                    np.degrees(
                        _wrap_angle_rad(
                            self._recovery_target_pose.yaw_rad - current_pose.yaw_rad
                        )
                    )
                )
            ),
        )

    def _target_reached(self, *, remaining_distance_m: float, remaining_yaw_deg: float) -> bool:
        if self._recovery_action in {"TURN_LEFT", "TURN_RIGHT"}:
            return remaining_yaw_deg <= max(2.0, float(self._config.yaw_threshold_deg))
        return remaining_distance_m <= max(0.06, float(self._config.displacement_threshold_m))

    def _finish_recovery(
        self,
        *,
        now_sec: float,
        reason: str,
        remaining_distance_m: float,
        remaining_yaw_deg: float,
    ) -> RecoveryCommand:
        action = self._recovery_action
        self._logger.warning(
            "[RECOVERY] complete reason=%s inverse=%s remaining_distance_m=%.2f remaining_yaw_deg=%.1f next=refresh_map_and_replan",
            reason,
            action,
            remaining_distance_m,
            remaining_yaw_deg,
        )
        self._clear_recovery(now_sec=now_sec)
        return RecoveryCommand(
            active=True,
            action_name="STOP",
            speed_scale=0.0,
            reason=reason,
            completed=True,
            remaining_distance_m=remaining_distance_m,
            remaining_yaw_deg=remaining_yaw_deg,
        )

    def _clear_recovery(self, *, now_sec: float) -> None:
        self._recovery_started_sec = None
        self._recovery_action = "STOP"
        self._recovery_target_pose = None
        self._cooldown_until_sec = float(now_sec) + max(0.2, float(self._config.window_sec))


def _inverse_action(action_name: str) -> str:
    action = str(action_name).strip().upper()
    return {
        "FORWARD": "BACKWARD",
        "BACKWARD": "FORWARD",
        "TURN_LEFT": "TURN_RIGHT",
        "TURN_RIGHT": "TURN_LEFT",
    }.get(action, "STOP")


def _copy_pose(pose: Pose2D) -> Pose2D:
    return Pose2D(x=float(pose.x), y=float(pose.y), yaw_rad=float(pose.yaw_rad))


def _pose_distance_m(first: Pose2D, second: Pose2D) -> float:
    return float(np.hypot(float(second.x) - float(first.x), float(second.y) - float(first.y)))


def _wrap_angle_rad(angle_rad: float) -> float:
    return float((float(angle_rad) + np.pi) % (2.0 * np.pi) - np.pi)
