from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .instruction_manager import RouteStep
from .observation_adapter import Pose2D


@dataclass(frozen=True)
class RegionTransitionConfig:
    min_displacement_m: float = 1.0
    stable_confirmations: int = 2
    min_region_confidence: float = 0.55


@dataclass(frozen=True)
class RegionTransitionState:
    step_id: str
    start_region_label: str
    current_region_label: str
    start_node_id: int | None
    current_node_id: int | None
    displacement_m: float
    stable_confirmation_count: int
    completed: bool
    completion_reason: str


class RegionTransitionManager:
    """Validate a room transition using odometry, topology, and VLM region evidence."""

    def __init__(self, *, config: RegionTransitionConfig, logger) -> None:
        self._config = config
        self._logger = logger.getChild("region_transition")
        self._step_id = ""
        self._start_pose: Pose2D | None = None
        self._start_node_id: int | None = None
        self._start_region_label = "unknown"
        self._last_region_label = "unknown"
        self._stable_confirmation_count = 0
        self._last_log_signature: tuple | None = None

    def reset(self) -> None:
        self._step_id = ""
        self._start_pose = None
        self._start_node_id = None
        self._start_region_label = "unknown"
        self._last_region_label = "unknown"
        self._stable_confirmation_count = 0
        self._last_log_signature = None

    def begin(
        self,
        *,
        step: RouteStep,
        robot_pose: Pose2D,
        topology_node_id: int | None,
        observed_region_label: str = "",
    ) -> None:
        if self._step_id == step.step_id:
            return
        self.reset()
        self._step_id = step.step_id
        self._start_pose = Pose2D(
            x=float(robot_pose.x),
            y=float(robot_pose.y),
            yaw_rad=float(robot_pose.yaw_rad),
        )
        self._start_node_id = topology_node_id
        self._start_region_label = _normalize_region_label(
            observed_region_label or step.from_region or "unknown"
        )
        self._last_region_label = self._start_region_label
        self._logger.info(
            "[NAV_STATE] region_transition=start step=%s from=%s to=%s node=%s",
            step.step_id,
            self._start_region_label,
            step.to_region or "unspecified",
            topology_node_id,
        )

    def update(
        self,
        *,
        step: RouteStep,
        robot_pose: Pose2D,
        topology_node_id: int | None,
        current_region_label: str,
        region_confidence: float,
        region_changed: bool,
    ) -> RegionTransitionState:
        self.begin(
            step=step,
            robot_pose=robot_pose,
            topology_node_id=topology_node_id,
        )
        current_label = _normalize_region_label(current_region_label)
        displacement_m = _pose_distance_m(self._start_pose, robot_pose)
        topology_changed = (
            self._start_node_id is not None
            and topology_node_id is not None
            and int(topology_node_id) != int(self._start_node_id)
        )
        named_target = step.to_region not in {
            "",
            "unknown",
            "adjacent_region",
            "outside_current_region",
        }
        label_matches = _region_labels_match(current_label, step.to_region) if named_target else True
        label_changed = (
            current_label not in {"", "unknown"}
            and self._start_region_label not in {"", "unknown"}
            and current_label != self._start_region_label
        )
        semantic_evidence = (
            bool(region_changed)
            and float(region_confidence) >= float(self._config.min_region_confidence)
            and label_matches
            and (label_changed or named_target)
        )
        geometric_evidence = (
            displacement_m >= max(0.0, float(self._config.min_displacement_m))
            and topology_changed
        )

        if semantic_evidence and geometric_evidence:
            if current_label == self._last_region_label:
                self._stable_confirmation_count += 1
            else:
                self._last_region_label = current_label
                self._stable_confirmation_count = 1
        else:
            self._stable_confirmation_count = 0
            if current_label not in {"", "unknown"}:
                self._last_region_label = current_label

        required = max(1, int(self._config.stable_confirmations))
        completed = self._stable_confirmation_count >= required
        reason = "pending"
        if completed:
            reason = "semantic_region_change+odometry+topology"
        elif not geometric_evidence:
            reason = "waiting_for_displacement_and_topology_change"
        elif not semantic_evidence:
            reason = "waiting_for_stable_region_semantics"

        state = RegionTransitionState(
            step_id=step.step_id,
            start_region_label=self._start_region_label,
            current_region_label=current_label,
            start_node_id=self._start_node_id,
            current_node_id=topology_node_id,
            displacement_m=displacement_m,
            stable_confirmation_count=self._stable_confirmation_count,
            completed=completed,
            completion_reason=reason,
        )
        signature = (
            state.step_id,
            state.current_region_label,
            round(state.displacement_m, 1),
            state.current_node_id,
            state.stable_confirmation_count,
            state.completed,
            state.completion_reason,
        )
        if signature != self._last_log_signature:
            self._last_log_signature = signature
            self._logger.info(
                "[NAV_STATE] region_transition=%s step=%s region=%s confidence=%.2f "
                "changed=%s displacement_m=%.2f topology=%s->%s stable=%s/%s reason=%s",
                "complete" if completed else "pending",
                step.step_id,
                current_label,
                float(region_confidence),
                int(bool(region_changed)),
                displacement_m,
                self._start_node_id,
                topology_node_id,
                self._stable_confirmation_count,
                required,
                reason,
            )
        return state


def _normalize_region_label(value: str) -> str:
    text = " ".join(str(value or "").strip().lower().split())
    return text or "unknown"


def _region_labels_match(observed: str, expected: str) -> bool:
    observed_tokens = set(_normalize_region_label(observed).replace("_", " ").split())
    expected_tokens = set(_normalize_region_label(expected).replace("_", " ").split())
    if not observed_tokens or not expected_tokens:
        return False
    return bool(observed_tokens == expected_tokens or observed_tokens.issubset(expected_tokens) or expected_tokens.issubset(observed_tokens))


def _pose_distance_m(start_pose: Pose2D | None, current_pose: Pose2D) -> float:
    if start_pose is None:
        return 0.0
    return float(
        np.hypot(
            float(current_pose.x) - float(start_pose.x),
            float(current_pose.y) - float(start_pose.y),
        )
    )
