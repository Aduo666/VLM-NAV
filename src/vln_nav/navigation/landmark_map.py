from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .landmark_tracker import LandmarkState
from .observation_adapter import Pose2D


@dataclass
class LandmarkMapConfig:
    enabled: bool = True
    anchor_ttl_sec: float = 120.0
    min_confidence: float = 0.20
    log_period_sec: float = 2.0


@dataclass
class LandmarkMapEntry:
    label: str
    behavior_type: str
    world_xy: list[float]
    confidence: float
    first_seen_time_sec: float
    last_seen_time_sec: float
    update_count: int
    status: str
    source: str
    depth_m: float | None
    bbox: tuple[int, int, int, int] | None
    anchor_distance_m: float | None
    robot_forward_m: float | None
    robot_left_m: float | None
    reached: bool = False
    confirmed_observation_count: int = 0
    evidence_level: str = "confirmed"

    @property
    def confirmed(self) -> bool:
        return bool(self.world_xy) and str(self.evidence_level).strip().lower() == "confirmed"


class LandmarkMap:
    """Minimal persistent landmark memory.

    This is not a dense semantic map. It stores confirmed landmark anchors from
    LandmarkTracker so the policy can reason about objects after they leave the
    current camera view. Full semantic occupancy can be layered on top later.
    """

    def __init__(self, *, config: LandmarkMapConfig, logger) -> None:
        self._config = config
        self._logger = logger.getChild("landmark_map")
        self._entries: dict[str, LandmarkMapEntry] = {}
        self._last_log_time_by_label: dict[str, float] = {}
        self._last_log_signature_by_label: dict[str, tuple] = {}

    def reset(self, label: str | None = None) -> None:
        if label is None:
            self._entries.clear()
            self._last_log_time_by_label.clear()
            self._last_log_signature_by_label.clear()
            return
        key = _normalize_label(label)
        self._entries.pop(key, None)
        self._last_log_time_by_label.pop(key, None)
        self._last_log_signature_by_label.pop(key, None)

    def update_from_tracker(
        self,
        *,
        state: LandmarkState | None,
        robot_pose: Pose2D,
        timestamp_sec: float,
    ) -> LandmarkMapEntry | None:
        if not bool(self._config.enabled) or state is None:
            return None

        label = _normalize_label(state.label)
        self._drop_expired(timestamp_sec=float(timestamp_sec))
        existing = self._entries.get(label)

        behavior = str(state.behavior_type).strip().lower()
        map_anchor_allowed = behavior in {"go_to", "turn_at", "turn_lookahead"}
        has_confirmed_anchor = (
            state.estimated_world_xy is not None
            and bool(state.can_complete)
            and str(state.evidence_level) == "confirmed"
            and bool(map_anchor_allowed)
        )

        if has_confirmed_anchor:
            cross_label_block = self._cross_label_guard(
                label=label,
                world_xy=[float(state.estimated_world_xy[0]), float(state.estimated_world_xy[1])],
                confidence=float(state.confidence),
            )
            if cross_label_block:
                self._logger.info(
                    "[LandmarkMap] cross_label_guard target=%s block=%s confidence=%.2f",
                    label, cross_label_block, float(state.confidence),
                )
                has_confirmed_anchor = False
        if has_confirmed_anchor:
            world_xy = [float(state.estimated_world_xy[0]), float(state.estimated_world_xy[1])]
            forward_m, left_m, distance_m = _relative_to_robot(
                robot_pose=robot_pose,
                world_xy=world_xy,
            )
            observation = state.last_observation
            reached = bool(state.reached or (existing is not None and existing.reached))
            entry = LandmarkMapEntry(
                label=label,
                behavior_type=str(state.behavior_type),
                world_xy=world_xy,
                confidence=float(state.confirmed_confidence) if state.confirmed_confidence > 0.0 else float(state.confidence),
                first_seen_time_sec=(
                    float(existing.first_seen_time_sec) if existing is not None else float(timestamp_sec)
                ),
                last_seen_time_sec=float(state.last_seen_time_sec or timestamp_sec),
                update_count=1 if existing is None else int(existing.update_count) + 1,
                status="reached" if reached else str(state.status),
                source="LandmarkTracker.confirmed_anchor",
                depth_m=None if observation is None else observation.depth_m,
                bbox=None if observation is None else observation.bbox,
                anchor_distance_m=float(distance_m),
                robot_forward_m=float(forward_m),
                robot_left_m=float(left_m),
                reached=reached,
                confirmed_observation_count=int(state.confirmed_observation_count),
                evidence_level=str(state.evidence_level),
            )
            self._entries[label] = entry
            self._log_entry_if_needed(entry=entry, timestamp_sec=float(timestamp_sec))
            return entry

        if existing is None:
            return None

        forward_m, left_m, distance_m = _relative_to_robot(
            robot_pose=robot_pose,
            world_xy=existing.world_xy,
        )
        if not bool(existing.reached):
            existing.status = str(state.status)
        existing.anchor_distance_m = float(distance_m)
        existing.robot_forward_m = float(forward_m)
        existing.robot_left_m = float(left_m)
        existing.reached = bool(existing.reached or state.reached)
        if existing.reached:
            existing.status = "reached"
        self._log_entry_if_needed(entry=existing, timestamp_sec=float(timestamp_sec))
        return existing

    def get(self, label: str) -> LandmarkMapEntry | None:
        return self._entries.get(_normalize_label(label))

    def entries(self) -> list[LandmarkMapEntry]:
        return list(self._entries.values())

    def mark_reached(self, label: str, *, timestamp_sec: float) -> None:
        key = _normalize_label(label)
        entry = self._entries.get(key)
        if entry is None:
            return
        entry.reached = True
        entry.status = "reached"
        entry.last_seen_time_sec = float(timestamp_sec)
        self._log_entry_if_needed(entry=entry, timestamp_sec=float(timestamp_sec))

    def summary(
        self,
        *,
        target_label: str,
        robot_pose: Pose2D,
        timestamp_sec: float,
    ) -> str:
        if not bool(self._config.enabled):
            return "landmark_map=disabled"
        label = _normalize_label(target_label)
        entry = self._entries.get(label)
        if entry is None:
            return f"landmark_map target={label} confirmed=0"
        forward_m, left_m, distance_m = _relative_to_robot(
            robot_pose=robot_pose,
            world_xy=entry.world_xy,
        )
        bearing = _bearing_hint(left_m=left_m)
        age_sec = max(0.0, float(timestamp_sec) - float(entry.last_seen_time_sec))
        confirmed = int(str(getattr(entry, "evidence_level", "")).strip().lower() == "confirmed")
        return (
            f"landmark_map target={label} confirmed={confirmed} evidence={entry.evidence_level} status={entry.status} "
            f"world_xy={_format_xy(entry.world_xy)} distance_m={distance_m:.2f} "
            f"robot_forward_m={forward_m:.2f} robot_left_m={left_m:.2f} "
            f"bearing_hint={bearing} age_sec={age_sec:.1f} updates={entry.update_count} "
            f"confidence={entry.confidence:.2f} reached={int(entry.reached)}"
        )

    def _cross_label_guard(
        self,
        *,
        label: str,
        world_xy: list[float],
        confidence: float,
    ) -> str:
        guard_radius_m = 1.0
        for existing_label, entry in self._entries.items():
            if existing_label == label:
                continue
            if not bool(entry.reached):
                continue
            if not bool(entry.world_xy):
                continue
            dx = float(world_xy[0]) - float(entry.world_xy[0])
            dy = float(world_xy[1]) - float(entry.world_xy[1])
            dist = float(np.hypot(dx, dy))
            if dist < guard_radius_m:
                return f"near_completed_{existing_label}_dist={dist:.2f}m_conf={confidence:.2f}"
        return ""

    def _drop_expired(self, *, timestamp_sec: float) -> None:
        ttl = float(self._config.anchor_ttl_sec)
        if ttl <= 0.0:
            return
        expired = [
            label
            for label, entry in self._entries.items()
            if not bool(entry.reached)
            and float(timestamp_sec) - float(entry.last_seen_time_sec) > ttl
        ]
        for label in expired:
            self._logger.info("[LandmarkMap] expired target=%s ttl_sec=%.1f", label, ttl)
            self._entries.pop(label, None)

    def _log_entry_if_needed(self, *, entry: LandmarkMapEntry, timestamp_sec: float) -> None:
        signature = (
            entry.status,
            _format_xy_bucket(entry.world_xy, bucket_m=0.10),
            None if entry.anchor_distance_m is None else round(float(entry.anchor_distance_m), 1),
            bool(entry.reached),
        )
        last_signature = self._last_log_signature_by_label.get(entry.label)
        last_time = self._last_log_time_by_label.get(entry.label, -float("inf"))
        if (
            signature == last_signature
            and float(timestamp_sec) - float(last_time) < max(3.0, float(self._config.log_period_sec))
        ):
            return
        self._last_log_signature_by_label[entry.label] = signature
        self._last_log_time_by_label[entry.label] = float(timestamp_sec)
        self._logger.info(
            "[LandmarkMap] target=%s status=%s evidence=%s confidence=%.2f world=%s distance=%s forward=%s left=%s updates=%s reached=%s source=%s",
            entry.label,
            entry.status,
            entry.evidence_level,
            float(entry.confidence),
            _format_xy(entry.world_xy),
            _format_optional_float(entry.anchor_distance_m),
            _format_optional_float(entry.robot_forward_m),
            _format_optional_float(entry.robot_left_m),
            int(entry.update_count),
            int(bool(entry.reached)),
            entry.source,
        )


def _relative_to_robot(*, robot_pose: Pose2D, world_xy: list[float]) -> tuple[float, float, float]:
    dx = float(world_xy[0]) - float(robot_pose.x)
    dy = float(world_xy[1]) - float(robot_pose.y)
    cos_yaw = float(np.cos(robot_pose.yaw_rad))
    sin_yaw = float(np.sin(robot_pose.yaw_rad))
    forward_m = cos_yaw * dx + sin_yaw * dy
    left_m = -sin_yaw * dx + cos_yaw * dy
    distance_m = float(np.hypot(forward_m, left_m))
    return float(forward_m), float(left_m), distance_m


def _bearing_hint(*, left_m: float) -> str:
    if float(left_m) > 0.25:
        return "left"
    if float(left_m) < -0.25:
        return "right"
    return "center"


def _normalize_label(label: str) -> str:
    return str(label).strip().lower() or "target"


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
