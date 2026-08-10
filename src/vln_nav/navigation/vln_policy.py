from __future__ import annotations

import io
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .candidate_waypoint_predictor import CandidateSet, CandidateWaypointPredictor, WaypointCandidate
from .instruction_manager import InstructionContext, RouteStep
from .landmark_map import LandmarkMapEntry
from .landmark_tracker import LandmarkState
from .map_candidate_generator import MapCandidateGenerator
from .observation_adapter import Pose2D, VLNObservation
from .progress_manager import ProgressManager
from .region_transition import RegionTransitionConfig, RegionTransitionManager
from .target_memory import TargetMemory
from .turn_heading_estimator import (
    TurnHeadingEstimate,
    TurnHeadingEstimator,
    TurnHeadingEstimatorConfig,
)
from .vln_reasoner import VLNReasoner
from .policy_helpers import (
    _candidate_to_primitive_action,
    _direction_consistency,
    _find_candidate,
    _format_optional_float,
    _format_xy,
    _local_goal_to_primitive_action,
    _local_waypoint_to_world_xy,
    _nearest_candidate_for_local_goal,
    _pose_distance_m,
    _primitive_speed_scale,
    _relative_side,
    _relative_xy_from_anchor,
    _route_steps,
    _world_waypoint_to_local_xy,
    _wrap_angle_rad,
)


@dataclass
class ExecutionFeedback:
    applied_action: str
    wheel_velocity: float
    steering_angle: float


@dataclass
class VLNPolicyConfig:
    turn_speed_scale: float
    forward_speed_scale: float
    replanning_cooldown_sec: float
    active_forward_duration_sec: float
    active_turn_duration_sec: float
    active_turn_angle_deg: float
    active_forward_distance_m: float
    active_distance_tolerance_m: float
    active_turn_tolerance_deg: float
    arrival_distance_m: float
    success_distance_m: float
    landmark_arrival_confirmations: int
    landmark_arrival_window_size: int
    region_transition_min_displacement_m: float = 1.0
    region_transition_confirmations: int = 2
    region_transition_min_confidence: float = 0.55
    semantic_turn_min_yaw_deg: float = 45.0
    semantic_turn_max_yaw_deg: float = 105.0
    semantic_turn_max_primitives: int = 7
    semantic_turn_min_forward_clearance_m: float = 0.60
    semantic_turn_alignment_confirmations: int = 2
    semantic_turn_corridor_center_tolerance_deg: float = 10.0
    semantic_turn_same_corridor_tolerance_deg: float = 15.0
    semantic_turn_target_high_confidence: float = 0.75
    semantic_turn_target_medium_confidence: float = 0.50
    semantic_turn_direction_tolerance_deg: float = 10.0
    semantic_turn_corridor_half_fov_deg: float = 60.0
    semantic_turn_corridor_step_deg: float = 5.0
    semantic_turn_min_corridor_width_deg: float = 15.0
    semantic_turn_corridor_max_clearance_m: float = 4.0
    semantic_turn_max_recovery_attempts: int = 2
    confirm_min_observations: int = 3
    confirm_confidence_threshold: float = 0.50
    visual_target_center_deadband_deg: float = 15.0


@dataclass
class PolicyStepResult:
    action_name: str
    speed_scale: float
    decision_source: str
    selected_candidate_id: int | None
    local_goal_xy: list[float]
    world_goal_xy: list[float] | None
    execution_steps_remaining: int
    reasoning_summary: str
    progress_signal: str
    replanned: bool
    debug_image_path: str | None
    current_landmark: str = ""
    completed_landmarks: list[str] = field(default_factory=list)
    route_complete: bool = False
    current_step_index: int = 0
    route_step_count: int = 0
    current_step_id: str = ""
    current_step_type: str = ""
    current_target: str = ""
    navigation_status: str = "RUNNING"
    last_error: str = ""
    completed_primitive_action: str = ""
    completed_primitive_reason: str = ""
    completed_primitive_start_pose: tuple[float, float, float] | None = None
    completed_primitive_end_pose: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class _TurnCorridorSample:
    robot_yaw_rad: float
    directed_yaw_deg: float
    corridor_world_yaw_rad: float
    local_heading_deg: float
    forward_clearance_m: float
    score: float


@dataclass
class _TurnTargetHypothesis:
    label: str = ""
    last_request_frame_index: int = -1
    confirmation_count: int = 0
    required_confirmations: int = 0
    last_world_xy: list[float] | None = None
    last_world_bearing_rad: float | None = None
    confidence: float = 0.0
    ready: bool = False
    has_world_anchor: bool = False


def _turn_target_observation_consistent(
    *,
    previous_world_xy: list[float] | None,
    current_world_xy,
    previous_world_bearing_rad: float | None,
    current_world_bearing_rad: float | None,
) -> bool:
    if previous_world_xy is not None and current_world_xy is not None:
        previous = np.asarray(previous_world_xy, dtype=np.float32).reshape(-1)
        current = np.asarray(current_world_xy, dtype=np.float32).reshape(-1)
        if previous.size >= 2 and current.size >= 2:
            return bool(float(np.linalg.norm(previous[:2] - current[:2])) <= 1.0)
    if (
        previous_world_bearing_rad is not None
        and current_world_bearing_rad is not None
    ):
        delta_deg = abs(
            float(
                np.degrees(
                    _wrap_angle_rad(
                        float(current_world_bearing_rad)
                        - float(previous_world_bearing_rad)
                    )
                )
            )
        )
        return bool(delta_deg <= 25.0)
    return True


def _clean_rgb_png_bytes(rgb) -> bytes | None:
    if rgb is None:
        return None
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] < 3:
        return None
    buffer = io.BytesIO()
    Image.fromarray(image[..., :3].copy(), mode="RGB").save(
        buffer,
        format="PNG",
        optimize=True,
    )
    return buffer.getvalue()


class VLNPolicy:
    def __init__(
        self,
        *,
        instruction_context: InstructionContext,
        candidate_predictor: CandidateWaypointPredictor,
        occupancy_grid: Any | None = None,
        map_candidate_generator: MapCandidateGenerator | None = None,
        progress_manager: ProgressManager,
        reasoner: VLNReasoner,
        config: VLNPolicyConfig,
        logger,
    ) -> None:
        self._instruction_context = instruction_context
        self._candidate_predictor = candidate_predictor
        self._occupancy_grid = occupancy_grid
        self._map_candidate_generator = map_candidate_generator
        self._progress_manager = progress_manager
        self._reasoner = reasoner
        self._config = config
        self._logger = logger.getChild("vln_policy")

        self._bootstrapped = False
        self._active_world_goal_xy: list[float] | None = None
        self._active_candidate_id: int | None = None
        self._active_candidate_label = "none"
        self._execution_steps_remaining = 0
        self._active_goal_started_sec: float | None = None
        self._active_action_name = "STOP"
        self._active_route_step_id = ""
        self._active_start_yaw_rad: float | None = None
        self._active_start_pose = None
        self._last_decision_summary = "startup"
        self._last_candidate_set: CandidateSet | None = None
        self._next_replan_time = 0.0
        self._route_steps = _route_steps(instruction_context)
        self._current_step_index = 0
        self._completed_landmarks: list[str] = []
        self._route_complete = False
        self._completed_step_ids: list[str] = []
        self._landmark_anchors: dict[str, list[float]] = {}
        self._landmark_anchor_yaws: dict[str, float] = {}
        self._target_memory = TargetMemory(target_text=self._current_target_text())
        self._landmark_tracker_state: LandmarkState | None = None
        self._landmark_map_entry: LandmarkMapEntry | None = None
        self._all_landmark_map_entries: list[LandmarkMapEntry] = []
        self._landmark_map_summary = ""
        self._obstacle_summary_text = ""
        self._semantic_map_summary = ""
        self._layout_description = ""
        self._action_history: deque[tuple[float, str, str]] = deque(maxlen=8)
        self._arrival_confirmation_window: deque[bool] = deque(
            maxlen=max(1, int(self._config.landmark_arrival_window_size))
        )
        self._safety_context_text = ""
        self._last_applied_action = "STOP"
        self._safety_override_streak = 0
        self._last_safety_override_signature: tuple[str, str, str] | None = None
        self._last_semantic_plan_signature: tuple | None = None
        self._last_semantic_skip_signature: tuple | None = None
        self._topology_summary = ""
        self._semantic_turn_step_id = ""
        self._semantic_turn_start_yaw_rad: float | None = None
        self._semantic_turn_primitives_taken = 0
        self._semantic_turn_samples: list[_TurnCorridorSample] = []
        self._semantic_turn_return_yaw_rad: float | None = None
        self._semantic_turn_latest_heading = TurnHeadingEstimate(
            valid=False,
            reason="not_updated",
        )
        self._semantic_turn_target_hypothesis = _TurnTargetHypothesis()
        self._semantic_turn_recovery_attempts = 0
        self._semantic_turn_restart_after_recovery = False
        self._semantic_turn_active_counts_as_scan = False
        self._turn_heading_estimator = TurnHeadingEstimator(
            TurnHeadingEstimatorConfig(
                heading_half_fov_deg=float(
                    self._config.semantic_turn_corridor_half_fov_deg
                ),
                heading_step_deg=float(self._config.semantic_turn_corridor_step_deg),
                min_corridor_width_deg=float(
                    self._config.semantic_turn_min_corridor_width_deg
                ),
                min_clearance_m=float(
                    self._config.semantic_turn_min_forward_clearance_m
                ),
                max_clearance_m=float(
                    self._config.semantic_turn_corridor_max_clearance_m
                ),
            )
        )
        self._satisfied_turn_triggers: set[str] = set()
        self._topology_node_id: int | None = None
        self._region_transition = RegionTransitionManager(
            config=RegionTransitionConfig(
                min_displacement_m=float(self._config.region_transition_min_displacement_m),
                stable_confirmations=int(self._config.region_transition_confirmations),
                min_region_confidence=float(self._config.region_transition_min_confidence),
            ),
            logger=logger,
        )
        self._recent_primitive_actions: deque[str] = deque(maxlen=4)
        self._last_turn_oscillation_signature: tuple | None = None
        self._last_route_relation_reject_key: tuple[str, str] | None = None

    def step(
        self,
        observation: VLNObservation,
        execution_feedback: ExecutionFeedback | None = None,
    ) -> PolicyStepResult:
        now = float(observation.meta["timestamp_sec"])
        if not self._bootstrapped:
            self._progress_manager.bootstrap(observation)
            self._bootstrapped = True
        elif execution_feedback is not None:
            self._progress_manager.update(
                observation=observation,
                candidate_set=self._last_candidate_set,
                selected_candidate_label=self._active_candidate_label,
                applied_action=execution_feedback.applied_action,
                reasoning_summary=self._last_decision_summary,
            )
            self._last_applied_action = str(execution_feedback.applied_action).strip().upper()

        progress_state = self._progress_manager.state
        if self.current_step_type == "GO_TO" and self._maybe_advance_landmark(observation=observation):
            self._clear_active_goal()
        if self._route_complete:
            return PolicyStepResult(
                action_name="STOP",
                speed_scale=0.0,
                decision_source="ROUTE_COMPLETE",
                selected_candidate_id=None,
                local_goal_xy=[0.0, 0.0],
                world_goal_xy=None,
                execution_steps_remaining=0,
                reasoning_summary="all typed instruction steps completed",
                progress_signal=progress_state.progress_signal,
                replanned=False,
                debug_image_path=None,
                **self._route_result_kwargs(),
            )

        if self.current_step_type == "TURN":
            return self._semantic_turn_step(
                observation=observation,
                progress_state=progress_state,
                now=now,
            )
        if self.current_step_type == "REGION_TRANSITION":
            return self._region_transition_step(
                observation=observation,
                progress_state=progress_state,
                now=now,
            )

        if self._active_world_goal_xy is not None and self._execution_steps_remaining > 0:
            local_goal = _world_waypoint_to_local_xy(
                robot_pose=observation.pose,
                world_goal_xy=self._active_world_goal_xy,
            )
            if self._active_action_name in {
                "FORWARD",
                "TURN_LEFT",
                "TURN_RIGHT",
            }:
                return self._motion_primitive_step(
                    observation=observation,
                    local_goal=local_goal,
                    progress_signal=progress_state.progress_signal,
                    now=now,
                )
            self._clear_active_goal()

        if now < self._next_replan_time:
            return PolicyStepResult(
                action_name="STOP",
                speed_scale=0.0,
                decision_source="REPLAN_COOLDOWN",
                selected_candidate_id=None,
                local_goal_xy=[0.0, 0.0],
                world_goal_xy=None,
                execution_steps_remaining=0,
                reasoning_summary="waiting_for_replan_cooldown",
                progress_signal=progress_state.progress_signal,
                replanned=False,
                debug_image_path=None,
                **self._route_result_kwargs(),
            )

        return self._clean_decision_step(
            observation=observation,
            progress_state=progress_state,
            now=now,
        )

    def _clean_decision_step(
        self,
        *,
        observation: VLNObservation,
        progress_state,
        now: float,
    ) -> PolicyStepResult:
        """Single decision authority for search and confirmed-anchor approach."""

        candidate_set = self._candidate_predictor.predict(observation)
        semantic_landmark_entry = self._semantic_landmark_entry()
        if self._map_candidate_generator is not None:
            candidate_set = self._map_candidate_generator.augment(
                observation=observation,
                base_candidate_set=candidate_set,
                occupancy_snapshot=(
                    None if self._occupancy_grid is None else self._occupancy_grid.snapshot()
                ),
                current_target=self._current_target_text(),
                landmark_state=self._landmark_tracker_state,
                landmark_entry=semantic_landmark_entry,
                progress_signal=progress_state.progress_signal,
                safety_context_text=self._safety_context_text,
            )
        self._annotate_candidate_context(candidate_set=candidate_set)
        self._last_candidate_set = candidate_set

        semantic_plan = None
        if (
            self._occupancy_grid is not None
            and semantic_landmark_entry is not None
            and not bool(semantic_landmark_entry.reached)
        ):
            try:
                semantic_plan = self._occupancy_grid.plan_toward_landmark(
                    observation=observation,
                    landmark_entry=semantic_landmark_entry,
                    stop_distance_m=float(self._current_arrival_distance_m()),
                )
            except Exception as exc:
                self._logger.warning(
                    "[SemanticPlanner] plan_failed landmark=%s error=%s",
                    self._current_target_text(),
                    exc,
                )

        debug_image_path = None
        # Fresh verified image evidence outranks a projected world anchor.
        visual_candidate = None
        visual_desired_action = None
        visual_bearing_deg = 0.0
        if self.current_step_type == "GO_TO":
            visual_candidate, visual_desired_action, visual_bearing_deg = (
                self._verified_visual_target_candidate(candidate_set=candidate_set)
            )
        if visual_desired_action is not None and visual_candidate is not None:
            action_name, speed_scale = _candidate_to_primitive_action(
                visual_candidate,
                turn_speed_scale=float(self._config.turn_speed_scale),
                forward_speed_scale=float(self._config.forward_speed_scale),
            )
            self._safety_context_text = ""
            return self._start_candidate_primitive(
                observation=observation,
                candidate=visual_candidate,
                action_name=action_name,
                speed_scale=speed_scale,
                now=now,
                progress_signal=progress_state.progress_signal,
                decision_source="VERIFIED_VISUAL_TARGET",
                reasoning_summary=(
                    f"fresh verified target bearing={visual_bearing_deg:+.1f}deg; "
                    f"execute {action_name} before map-anchor navigation"
                ),
                debug_image_path=debug_image_path,
            )
        if visual_desired_action is not None:
            detour_result = self._start_visual_local_detour(
                observation=observation,
                candidate_set=candidate_set,
                desired_action=visual_desired_action,
                target_bearing_deg=visual_bearing_deg,
                progress_signal=progress_state.progress_signal,
                debug_image_path=debug_image_path,
                now=now,
            )
            if detour_result is not None:
                self._safety_context_text = ""
                return detour_result
            self._safety_context_text = ""
            self._next_replan_time = now + float(self._config.replanning_cooldown_sec)
            return PolicyStepResult(
                action_name="STOP",
                speed_scale=0.0,
                decision_source="VISUAL_TARGET_SAFETY_HOLD",
                selected_candidate_id=None,
                local_goal_xy=[0.0, 0.0],
                world_goal_xy=None,
                execution_steps_remaining=0,
                reasoning_summary=(
                    f"verified target bearing={visual_bearing_deg:+.1f}deg requires "
                    f"{visual_desired_action}, but that primitive is not safe"
                ),
                progress_signal=progress_state.progress_signal,
                replanned=True,
                debug_image_path=None if debug_image_path is None else str(debug_image_path),
                **self._route_result_kwargs(),
            )

        # APPROACH: a confirmed semantic anchor has exclusive control through A*.
        semantic_result = self._semantic_planner_step(
            observation=observation,
            candidate_set=candidate_set,
            progress_signal=progress_state.progress_signal,
            debug_image_path=debug_image_path,
            semantic_plan=semantic_plan,
            now=now,
        )
        if semantic_result is not None:
            self._safety_context_text = ""
            return semantic_result

        if semantic_landmark_entry is not None:
            detour_candidate = self._best_anchor_detour_candidate(
                observation=observation,
                candidate_set=candidate_set,
                anchor_world_xy=semantic_landmark_entry.world_xy,
            )
            if detour_candidate is not None:
                action_name, speed_scale = _candidate_to_primitive_action(
                    detour_candidate,
                    turn_speed_scale=float(self._config.turn_speed_scale),
                    forward_speed_scale=float(self._config.forward_speed_scale),
                )
                self._logger.warning(
                    "[LocalDetour] target=%s reason=confirmed_anchor_no_astar_path action=%s candidate=%s",
                    self._current_target_text(),
                    action_name,
                    detour_candidate.slot_id,
                )
                self._safety_context_text = ""
                return self._start_candidate_primitive(
                    observation=observation,
                    candidate=detour_candidate,
                    action_name=action_name,
                    speed_scale=speed_scale,
                    now=now,
                    progress_signal=progress_state.progress_signal,
                    decision_source="ANCHOR_LOCAL_DETOUR",
                    reasoning_summary="confirmed target anchor retained; move through safest progress-making local corridor and replan",
                    debug_image_path=debug_image_path,
                )
            self._safety_context_text = ""
            self._next_replan_time = now + float(self._config.replanning_cooldown_sec)
            self._logger.error(
                "[SAFETY_HOLD] step=%s target=%s reason=confirmed_anchor_has_no_safe_path",
                self._current_route_step().step_id if self._current_route_step() is not None else "none",
                self._current_target_text(),
            )
            return PolicyStepResult(
                action_name="STOP",
                speed_scale=0.0,
                decision_source="APPROACH_SAFETY_HOLD",
                selected_candidate_id=None,
                local_goal_xy=[0.0, 0.0],
                world_goal_xy=None,
                execution_steps_remaining=0,
                reasoning_summary="confirmed anchor has no safe reachable primitive",
                progress_signal=progress_state.progress_signal,
                replanned=True,
                debug_image_path=None if debug_image_path is None else str(debug_image_path),
                **self._route_result_kwargs(),
            )

        annotated_png_bytes = _clean_rgb_png_bytes(observation.rgb)
        decision = self._reasoner.choose_candidate(
            instruction_context=self._instruction_context,
            candidate_set=candidate_set,
            progress_state=progress_state,
            annotated_image_png_bytes=annotated_png_bytes,
            target_memory_text=self._grounded_target_context(),
            safety_context_text=self._safety_context_text,
            route_context_text=self._route_context_text(robot_pose=observation.pose),
            reacquire_context_text="none",
            obstacle_summary_text=self._obstacle_summary_text,
            semantic_map_summary=self._semantic_map_summary,
            current_step=self._current_route_step(),
        )
        self._safety_context_text = ""
        selected_candidate = _find_candidate(candidate_set, decision.selected_candidate_id)
        if selected_candidate is None or not bool(selected_candidate.valid):
            self._next_replan_time = now + float(self._config.replanning_cooldown_sec)
            self._clear_active_goal()
            self._logger.error(
                "[ERROR] step=%s type=%s reason=%s selected_candidate=%s",
                self._current_route_step().step_id if self._current_route_step() is not None else "none",
                self.current_step_type or "none",
                decision.reason,
                decision.selected_candidate_id,
            )
            return PolicyStepResult(
                action_name="STOP",
                speed_scale=0.0,
                decision_source=f"SEARCH_HOLD({decision.reason})",
                selected_candidate_id=None,
                local_goal_xy=[0.0, 0.0],
                world_goal_xy=None,
                execution_steps_remaining=0,
                reasoning_summary=decision.thought,
                progress_signal=progress_state.progress_signal,
                replanned=True,
                debug_image_path=None if debug_image_path is None else str(debug_image_path),
                **self._route_result_kwargs(),
            )

        action_name, speed_scale = _candidate_to_primitive_action(
            selected_candidate,
            turn_speed_scale=float(self._config.turn_speed_scale),
            forward_speed_scale=float(self._config.forward_speed_scale),
        )
        return self._start_candidate_primitive(
            observation=observation,
            candidate=selected_candidate,
            action_name=action_name,
            speed_scale=speed_scale,
            now=now,
            progress_signal=progress_state.progress_signal,
            decision_source="VLM_SEARCH_PRIMITIVE",
            reasoning_summary=decision.thought,
            debug_image_path=debug_image_path,
        )

    def _verified_visual_target_candidate(
        self,
        *,
        candidate_set: CandidateSet,
    ) -> tuple[WaypointCandidate | None, str | None, float]:
        state = self._landmark_tracker_state
        observation = None if state is None else state.last_observation
        if observation is None:
            return None, None, 0.0
        if not (
            bool(getattr(observation, "visible", False))
            and bool(getattr(observation, "can_confirm", False))
            and bool(getattr(observation, "is_current_frame", True))
            and getattr(observation, "bbox", None) is not None
            and getattr(observation, "bearing_rad", None) is not None
            and str(getattr(observation, "reason", "")).strip().lower() == "ok"
        ):
            return None, None, 0.0

        bearing_deg = float(np.degrees(float(observation.bearing_rad)))
        deadband_deg = max(1.0, float(self._config.visual_target_center_deadband_deg))
        if bearing_deg > deadband_deg:
            desired_action = "TURN_RIGHT"
        elif bearing_deg < -deadband_deg:
            desired_action = "TURN_LEFT"
        else:
            desired_action = "FORWARD"

        valid = [candidate for candidate in candidate_set.candidates if bool(candidate.valid)]
        matching = [
            candidate
            for candidate in valid
            if _candidate_to_primitive_action(
                candidate,
                turn_speed_scale=float(self._config.turn_speed_scale),
                forward_speed_scale=float(self._config.forward_speed_scale),
            )[0]
            == desired_action
        ]
        if matching:
            return max(matching, key=lambda item: float(item.clearance_score)), desired_action, bearing_deg

        # Never turn away from a visible target. If centered FORWARD is unsafe,
        # hold position and let runtime recovery handle the blocked corridor.
        self._logger.info(
            "[VisualTarget] target=%s bearing_deg=%+.1f desired=%s reason=no_safe_matching_candidate",
            self._current_target_text(),
            bearing_deg,
            desired_action,
        )
        return None, desired_action, bearing_deg

    def _start_visual_local_detour(
        self,
        *,
        observation: VLNObservation,
        candidate_set: CandidateSet,
        desired_action: str,
        target_bearing_deg: float,
        progress_signal: str,
        debug_image_path: str | Path | None,
        now: float,
    ) -> PolicyStepResult | None:
        turn_candidate = self._best_detour_turn_candidate(
            candidate_set=candidate_set,
            target_bearing_deg=target_bearing_deg,
        )
        if turn_candidate is None:
            return None
        turn_action, speed_scale = _candidate_to_primitive_action(
            turn_candidate,
            turn_speed_scale=float(self._config.turn_speed_scale),
            forward_speed_scale=float(self._config.forward_speed_scale),
        )
        self._logger.warning(
            "[LocalDetour] target=%s direct=%s blocked=1 detour=%s bearing_deg=%+.1f",
            self._current_target_text(),
            desired_action,
            turn_action,
            float(target_bearing_deg),
        )
        return self._start_candidate_primitive(
            observation=observation,
            candidate=turn_candidate,
            action_name=turn_action,
            speed_scale=speed_scale,
            now=now,
            progress_signal=progress_signal,
            decision_source="VISUAL_TARGET_LOCAL_DETOUR",
            reasoning_summary=(
                f"direct visual action {desired_action} is blocked; execute one "
                f"{turn_action} primitive and re-evaluate the fresh target bearing"
            ),
            debug_image_path=debug_image_path,
        )

    def _best_detour_turn_candidate(
        self,
        *,
        candidate_set: CandidateSet,
        target_bearing_deg: float,
    ) -> WaypointCandidate | None:
        candidates: list[tuple[float, WaypointCandidate]] = []
        for candidate in candidate_set.candidates:
            if not bool(candidate.valid):
                continue
            action_name, _ = _candidate_to_primitive_action(
                candidate,
                turn_speed_scale=float(self._config.turn_speed_scale),
                forward_speed_scale=float(self._config.forward_speed_scale),
            )
            if action_name not in {"TURN_LEFT", "TURN_RIGHT"}:
                continue
            target_side_bonus = 0.0
            if target_bearing_deg < 0.0 and action_name == "TURN_LEFT":
                target_side_bonus = 0.15
            elif target_bearing_deg > 0.0 and action_name == "TURN_RIGHT":
                target_side_bonus = 0.15
            recent_penalty = 0.0
            if self._recent_primitive_actions:
                opposite = "TURN_RIGHT" if action_name == "TURN_LEFT" else "TURN_LEFT"
                if self._recent_primitive_actions[-1] == opposite:
                    recent_penalty = 0.20
            score = (
                float(candidate.clearance_score)
                + 0.20 * float(candidate.depth_support.get("min_clearance_m", 0.0))
                + target_side_bonus
                - recent_penalty
            )
            candidates.append((score, candidate))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def _best_anchor_detour_candidate(
        self,
        *,
        observation: VLNObservation,
        candidate_set: CandidateSet,
        anchor_world_xy: list[float],
    ) -> WaypointCandidate | None:
        target_local = _world_waypoint_to_local_xy(
            robot_pose=observation.pose,
            world_goal_xy=anchor_world_xy,
        )
        target_heading_deg = float(np.degrees(np.arctan2(target_local[1], max(1e-6, target_local[0]))))
        ranked: list[tuple[float, WaypointCandidate]] = []
        for candidate in candidate_set.candidates:
            if not bool(candidate.valid):
                continue
            heading_error = abs(float(candidate.heading_delta_deg) - target_heading_deg)
            heading_score = max(0.0, 1.0 - heading_error / 90.0)
            score = (
                float(candidate.clearance_score)
                + 0.35 * heading_score
                + 0.10 * float(candidate.depth_support.get("min_clearance_m", 0.0))
            )
            ranked.append((score, candidate))
        if not ranked:
            return None
        return max(ranked, key=lambda item: item[0])[1]

    def _start_candidate_primitive(
        self,
        *,
        observation: VLNObservation,
        candidate: WaypointCandidate,
        action_name: str,
        speed_scale: float,
        now: float,
        progress_signal: str,
        decision_source: str,
        reasoning_summary: str,
        debug_image_path: str | Path | None,
    ) -> PolicyStepResult:
        is_turn = str(action_name) in {"TURN_LEFT", "TURN_RIGHT"}
        world_goal_xy = (
            [float(observation.pose.x), float(observation.pose.y)]
            if is_turn
            else _local_waypoint_to_world_xy(
                robot_pose=observation.pose,
                waypoint_local_xy=candidate.local_goal_xy,
            )
        )
        # The internal pose anchor keeps the primitive active across frames;
        # it is not exposed as a navigation target for an in-place turn.
        self._active_world_goal_xy = [float(world_goal_xy[0]), float(world_goal_xy[1])]
        self._active_candidate_id = int(candidate.slot_id)
        self._active_candidate_label = str(candidate.label)
        self._execution_steps_remaining = 1
        self._active_action_name = str(action_name)
        current_step = self._current_route_step()
        self._active_route_step_id = "" if current_step is None else current_step.step_id
        self._active_goal_started_sec = float(now)
        self._active_start_yaw_rad = float(observation.pose.yaw_rad)
        self._active_start_pose = observation.pose
        self._last_decision_summary = str(reasoning_summary)
        self._recent_primitive_actions.append(str(action_name))
        self._logger.info(
            "[PrimitiveDecision] mode=SEARCH target=%s candidate=%s action=%s valid=%s clearance=%.2f reason=%s",
            self._current_target_text(),
            int(candidate.slot_id),
            action_name,
            int(bool(candidate.valid)),
            float(candidate.clearance_score),
            str(reasoning_summary)[:180],
        )
        return PolicyStepResult(
            action_name=str(action_name),
            speed_scale=float(speed_scale),
            decision_source=str(decision_source),
            selected_candidate_id=int(candidate.slot_id),
            local_goal_xy=[float(candidate.local_goal_xy[0]), float(candidate.local_goal_xy[1])],
            world_goal_xy=None if is_turn else list(self._active_world_goal_xy),
            execution_steps_remaining=int(self._execution_steps_remaining),
            reasoning_summary=str(reasoning_summary),
            progress_signal=str(progress_signal),
            replanned=True,
            debug_image_path=None if debug_image_path is None else str(debug_image_path),
            **self._route_result_kwargs(),
        )

    def _annotate_candidate_context(self, *, candidate_set: CandidateSet) -> None:
        for candidate in candidate_set.candidates:
            action_name, _ = _candidate_to_primitive_action(
                candidate,
                turn_speed_scale=float(self._config.turn_speed_scale),
                forward_speed_scale=float(self._config.forward_speed_scale),
            )
            repetition_penalty = 0.0
            if self._recent_primitive_actions and self._recent_primitive_actions[-1] == action_name:
                repetition_penalty = 1.00 if action_name.startswith("TURN_") else 0.10
            candidate.depth_support["route_alignment"] = 0.0
            candidate.depth_support["repetition_penalty"] = float(repetition_penalty)

    def _grounded_target_context(self) -> str:
        parts = [f"current_grounding={self._current_grounding_summary()}"]
        if self._landmark_map_summary:
            parts.append(f"landmark_map={self._landmark_map_summary}")
        return "; ".join(parts)

    def _semantic_planner_step(
        self,
        *,
        observation: VLNObservation,
        candidate_set: CandidateSet,
        progress_signal: str,
        debug_image_path: str | Path | None,
        semantic_plan,
        now: float,
    ) -> PolicyStepResult | None:
        if semantic_plan is None or not bool(getattr(semantic_plan, "valid", False)):
            return None

        local_goal = (
            float(semantic_plan.subgoal_local_xy[0]),
            float(semantic_plan.subgoal_local_xy[1]),
        )
        local_goal_distance_m = float(np.hypot(float(local_goal[0]), float(local_goal[1])))
        path_length_m = float(getattr(semantic_plan, "path_length_m", 0.0))

        action_name = _local_goal_to_primitive_action(local_goal_xy=local_goal)
        if action_name == "STOP":
            return None
        min_executable_forward_m = max(
            0.10,
            0.75 * float(self._config.active_forward_distance_m),
        )
        if (
            local_goal_distance_m < min_executable_forward_m
            or path_length_m < min_executable_forward_m
        ):
            self._log_semantic_skip(
                target_label=str(
                    getattr(
                        semantic_plan,
                        "target_label",
                        self._current_target_text(),
                    )
                ),
                reason="subgoal_too_short_for_primitive",
                action_name=action_name,
                distance_to_target_m=float(
                    getattr(semantic_plan, "distance_to_target_m", 0.0)
                ),
                path_length_m=path_length_m,
                local_goal=local_goal,
                min_executable_m=min_executable_forward_m,
            )
            return None
        speed_scale = float(_primitive_speed_scale(action_name, self._config))

        if (
            self._safety_context_text
            and f"previous_desired={action_name}" in self._safety_context_text
        ):
            self._log_semantic_skip(
                target_label=str(getattr(semantic_plan, "target_label", self._current_target_text())),
                reason="previous_action_safety_blocked",
                action_name=action_name,
                distance_to_target_m=float(getattr(semantic_plan, "distance_to_target_m", 0.0)),
                path_length_m=path_length_m,
                local_goal=local_goal,
                min_executable_m=min_executable_forward_m,
            )
            return None
        selected_candidate = _nearest_candidate_for_local_goal(candidate_set=candidate_set, local_goal_xy=local_goal)
        selected_candidate_id = None if selected_candidate is None else int(selected_candidate.slot_id)
        world_goal_xy_list = (
            None
            if semantic_plan.subgoal_world_xy is None
            else [float(semantic_plan.subgoal_world_xy[0]), float(semantic_plan.subgoal_world_xy[1])]
        )
        self._active_world_goal_xy = None if world_goal_xy_list is None else list(world_goal_xy_list)
        self._active_candidate_id = selected_candidate_id
        self._active_candidate_label = (
            "semantic_subgoal"
            if selected_candidate is None
            else str(selected_candidate.label)
        )
        self._execution_steps_remaining = 1
        self._active_action_name = str(action_name)
        self._active_goal_started_sec = float(now)
        self._active_start_yaw_rad = float(observation.pose.yaw_rad)
        self._active_start_pose = observation.pose
        self._last_decision_summary = (
            f"semantic planner target={semantic_plan.target_label} "
            f"distance_m={float(semantic_plan.distance_to_target_m):.2f} "
            f"bearing_deg={float(semantic_plan.bearing_deg):+.1f} "
            f"path_length_m={float(semantic_plan.path_length_m):.2f} "
            f"subgoal_local_xy=[{local_goal[0]:.2f},{local_goal[1]:.2f}]"
        )
        signature = (
            str(semantic_plan.target_label),
            round(float(semantic_plan.distance_to_target_m), 1),
            round(float(semantic_plan.bearing_deg) / 10.0),
            str(action_name),
            selected_candidate_id,
        )
        if signature != self._last_semantic_plan_signature:
            self._logger.info(
                "[SemanticPlanner] target=%s distance_m=%.2f bearing_deg=%+.1f action=%s candidate=%s subgoal_local_xy=[%.2f,%.2f] goal_world_xy=%s",
                semantic_plan.target_label,
                float(semantic_plan.distance_to_target_m),
                float(semantic_plan.bearing_deg),
                action_name,
                selected_candidate_id,
                float(local_goal[0]),
                float(local_goal[1]),
                world_goal_xy_list,
            )
            self._last_semantic_plan_signature = signature
        return PolicyStepResult(
            action_name=action_name,
            speed_scale=float(speed_scale),
            decision_source="SEMANTIC_LOCAL_PLANNER",
            selected_candidate_id=selected_candidate_id,
            local_goal_xy=[float(local_goal[0]), float(local_goal[1])],
            world_goal_xy=world_goal_xy_list,
            execution_steps_remaining=int(self._execution_steps_remaining),
            reasoning_summary=self._last_decision_summary,
            progress_signal=progress_signal,
            replanned=True,
            debug_image_path=None if debug_image_path is None else str(debug_image_path),
            **self._route_result_kwargs(),
        )

    def _log_semantic_skip(
        self,
        *,
        target_label: str,
        reason: str,
        action_name: str,
        distance_to_target_m: float,
        path_length_m: float,
        local_goal: tuple[float, float],
        min_executable_m: float,
    ) -> None:
        signature = (
            str(target_label),
            str(reason),
            str(action_name),
            round(float(distance_to_target_m), 1),
            round(float(path_length_m), 1),
            round(float(local_goal[0]), 1),
            round(float(local_goal[1]), 1),
        )
        if signature == self._last_semantic_skip_signature:
            return
        self._logger.warning(
            "[SemanticPlanner] skip target=%s reason=%s action=%s distance_m=%.2f path_length_m=%.2f subgoal_local_xy=[%.2f,%.2f] min_executable_m=%.2f",
            target_label,
            reason,
            action_name,
            float(distance_to_target_m),
            float(path_length_m),
            float(local_goal[0]),
            float(local_goal[1]),
            float(min_executable_m),
        )
        self._last_semantic_skip_signature = signature

    def _clear_active_goal(self) -> None:
        self._active_world_goal_xy = None
        self._active_candidate_id = None
        self._active_candidate_label = "none"
        self._execution_steps_remaining = 0
        self._active_goal_started_sec = None
        self._active_action_name = "STOP"
        self._active_route_step_id = ""
        self._active_start_yaw_rad = None
        self._active_start_pose = None

    def cancel_active_goal(self) -> None:
        self._clear_active_goal()
        self._semantic_turn_active_counts_as_scan = False

    def active_primitive_context(self) -> tuple[str, Pose2D] | None:
        action = str(self._active_action_name).strip().upper()
        if action not in {"FORWARD", "BACKWARD", "TURN_LEFT", "TURN_RIGHT"}:
            return None
        if self._active_start_pose is None:
            return None
        return (
            action,
            Pose2D(
                x=float(self._active_start_pose.x),
                y=float(self._active_start_pose.y),
                yaw_rad=float(self._active_start_pose.yaw_rad),
            ),
        )

    def request_immediate_replan(self) -> None:
        self.cancel_active_goal()
        self._next_replan_time = 0.0

    def set_safety_context(self, text: str) -> None:
        self._safety_context_text = str(text).strip()

    def record_safety_result(
        self,
        *,
        overridden: bool,
        desired_action: str,
        applied_action: str,
        reason: str,
    ) -> None:
        desired = str(desired_action).strip().upper()
        applied = str(applied_action).strip().upper()
        reason_text = str(reason).strip() or "unknown"
        if not bool(overridden):
            self._safety_override_streak = 0
            self._last_safety_override_signature = None
            return
        signature = (desired, applied, reason_text)
        if signature == self._last_safety_override_signature:
            self._safety_override_streak += 1
        else:
            self._safety_override_streak = 1
            self._last_safety_override_signature = signature
        if self._safety_override_streak >= 2:
            self._logger.warning(
                "[SAFETY_HOLD] repeated_override streak=%s desired=%s applied=%s reason=%s",
                int(self._safety_override_streak),
                desired,
                applied,
                reason_text,
            )

    @property
    def current_landmark(self) -> str:
        """Current object perception target; empty for non-object route steps."""
        return self.current_perception_target

    @property
    def current_perception_target(self) -> str:
        if self._route_complete:
            return ""
        step = self._current_route_step()
        return "" if step is None else step.perception_target

    @property
    def active_perception_target(self) -> str:
        """Object label that the runtime should perceive at the current keyframe."""
        if self._route_complete:
            return ""
        step = self._current_route_step()
        if step is None:
            return ""
        if step.step_type != "TURN":
            return step.perception_target
        if step.cues:
            return step.perception_target
        if (
            step.trigger is not None
            and step.trigger.trigger_type == "NEAR_OBJECT"
            and step.step_id not in self._satisfied_turn_triggers
        ):
            return step.trigger.label
        if self._semantic_turn_primitives_taken < 1:
            return ""
        return self._next_go_to_target()

    @property
    def current_step_type(self) -> str:
        step = self._current_route_step()
        return "" if step is None else step.step_type

    @property
    def current_behavior_type(self) -> str:
        route_step = self._current_route_step()
        if route_step is None:
            return "none"
        return route_step.step_type.lower()

    @property
    def current_grounding_behavior_type(self) -> str:
        target = self.active_perception_target
        if not target:
            return "none"
        if self.current_step_type == "REGION_TRANSITION":
            return "transition_cue"
        if self.current_step_type == "TURN" and self._current_route_step().cues:
            return "turn_cue"
        if self.current_step_type == "TURN" and target != self.current_perception_target:
            return "turn_lookahead"
        return "go_to"

    @property
    def current_turn_direction(self) -> str:
        step = self._current_route_step()
        if step is None or step.step_type != "TURN":
            return ""
        return str(step.direction).strip().lower()

    @property
    def current_turn_start_yaw_rad(self) -> float | None:
        if self.current_step_type != "TURN":
            return None
        return self._semantic_turn_start_yaw_rad

    @property
    def current_turn_direction_tolerance_deg(self) -> float:
        return float(self._config.semantic_turn_direction_tolerance_deg)

    @property
    def current_arrival_distance_m(self) -> float:
        return self._current_arrival_distance_m()

    @property
    def landmark_anchors(self) -> dict[str, list[float]]:
        return dict(self._landmark_anchors)

    @property
    def current_direction_clause(self) -> str:
        return self._current_direction_clause()

    @property
    def current_semantic_relation_text(self) -> str:
        step = self._current_route_step()
        if step is None or step.step_type != "GO_TO" or not step.relations:
            return ""
        descriptions: list[str] = []
        for item in step.relations:
            if item.relation == "BEHIND_TARGET":
                descriptions.append(
                    f"select the {item.target} that has the {item.reference} behind it"
                )
            elif item.relation == "IN_FRONT_OF":
                descriptions.append(
                    f"select the {item.target} in front of the {item.reference}"
                )
            elif item.relation == "LEFT_OF":
                descriptions.append(
                    f"select the {item.target} left of the {item.reference}"
                )
            elif item.relation == "RIGHT_OF":
                descriptions.append(
                    f"select the {item.target} right of the {item.reference}"
                )
            else:
                descriptions.append(
                    f"select the {item.target} near the {item.reference}"
                )
        return "; ".join(descriptions)

    def set_landmark_state(self, state: LandmarkState | None) -> None:
        self._landmark_tracker_state = state
        if state is None or state.last_observation is None:
            return
        observation = state.last_observation

    def set_landmark_map_entry(self, entry: LandmarkMapEntry | None) -> None:
        self._landmark_map_entry = entry

    def set_all_landmark_map_entries(self, entries: list[LandmarkMapEntry]) -> None:
        self._all_landmark_map_entries = list(entries)

    def set_landmark_map_summary(self, text: str) -> None:
        self._landmark_map_summary = str(text).strip()

    def set_topology_summary(self, text: str) -> None:
        self._topology_summary = str(text).strip()

    def set_obstacle_summary(self, text: str) -> None:
        self._obstacle_summary_text = str(text).strip()

    def set_semantic_map_summary(self, text: str) -> None:
        self._semantic_map_summary = str(text).strip()

    def set_layout_description(self, text: str) -> None:
        self._layout_description = str(text).strip()

    def record_action(self, action: str, result: str, now: float) -> None:
        action_text = str(action).strip()
        result_text = str(result).strip()
        if self._action_history:
            _, last_action, last_result = self._action_history[-1]
            if last_action == action_text and last_result == result_text:
                return
        self._action_history.append((now, action_text, result_text))

    @property
    def current_layout_description(self) -> str:
        return self._layout_description

    @property
    def action_history_text(self) -> str:
        if not self._action_history:
            return ""
        now = self._action_history[-1][0] if self._action_history else 0.0
        lines: list[str] = []
        for t, action, result in self._action_history:
            age = now - t
            if age > 60:
                continue
            lines.append(f"  {age:.0f}s ago: {action} → {result}")
        return "Recent:\n" + "\n".join(lines) if lines else ""

    def set_topology_node_id(self, value: int | None) -> None:
        self._topology_node_id = None if value is None else int(value)

    def update_turn_heading_context(
        self,
        *,
        robot_pose,
        occupancy_snapshot,
    ) -> None:
        if self.current_step_type != "TURN":
            self._semantic_turn_latest_heading = TurnHeadingEstimate(
                valid=False,
                reason="not_turn_step",
            )
            return
        self._semantic_turn_latest_heading = self._turn_heading_estimator.estimate(
            robot_pose=robot_pose,
            occupancy_snapshot=occupancy_snapshot,
        )

    def _semantic_turn_step(
        self,
        *,
        observation: VLNObservation,
        progress_state,
        now: float,
    ) -> PolicyStepResult:
        step = self._current_route_step()
        if step is None or step.step_type != "TURN":
            raise RuntimeError("semantic turn handler called without a TURN step")

        active_result = self._continue_current_step_primitive(
            observation=observation,
            progress_signal=progress_state.progress_signal,
            now=now,
        )
        if active_result is not None:
            if str(active_result.decision_source) == "PRIMITIVE_COMPLETE":
                if self._semantic_turn_active_counts_as_scan:
                    self._semantic_turn_primitives_taken += 1
                self._semantic_turn_active_counts_as_scan = False
            return active_result

        if step.trigger is not None and step.trigger.trigger_type == "NEAR_OBJECT":
            tracker_state = self._landmark_tracker_state
            trigger_reached = (
                tracker_state is not None
                and tracker_state.label.strip().lower() == step.trigger.label.strip().lower()
                and bool(tracker_state.reached)
                and bool(tracker_state.can_complete)
            )
            if trigger_reached:
                self._satisfied_turn_triggers.add(step.step_id)
                self._clear_active_goal()
                self._logger.info(
                    "[STEP_COMPLETE] trigger=NEAR_OBJECT step=%s label=%s",
                    step.step_id,
                    step.trigger.label,
                )
            elif step.step_id not in self._satisfied_turn_triggers:
                return self._clean_decision_step(
                    observation=observation,
                    progress_state=progress_state,
                    now=now,
                )

        if self._semantic_turn_step_id != step.step_id:
            self._semantic_turn_step_id = step.step_id
            self._semantic_turn_start_yaw_rad = float(observation.pose.yaw_rad)
            self._semantic_turn_primitives_taken = 0
            self._semantic_turn_samples.clear()
            self._semantic_turn_return_yaw_rad = None
            self._semantic_turn_target_hypothesis = _TurnTargetHypothesis(
                label=self._next_go_to_target()
            )
            self._semantic_turn_recovery_attempts = 0
            self._semantic_turn_restart_after_recovery = False
            self._semantic_turn_active_counts_as_scan = False
            self._logger.info(
                "[NAV_STATE] turn=start step=%s direction=%s lookahead=%s "
                "yaw_range_deg=[%.1f,%.1f] max_primitives=%s",
                step.step_id,
                step.direction,
                self._next_go_to_target() or "none",
                float(self._config.semantic_turn_min_yaw_deg),
                float(self._config.semantic_turn_max_yaw_deg),
                int(self._config.semantic_turn_max_primitives),
            )
        elif self._semantic_turn_restart_after_recovery:
            self._semantic_turn_start_yaw_rad = float(observation.pose.yaw_rad)
            self._semantic_turn_primitives_taken = 0
            self._semantic_turn_samples.clear()
            self._semantic_turn_return_yaw_rad = None
            self._semantic_turn_target_hypothesis = _TurnTargetHypothesis(
                label=self._next_go_to_target()
            )
            self._semantic_turn_restart_after_recovery = False
            self._semantic_turn_active_counts_as_scan = False
            self._logger.info(
                "[TurnRecovery] restart_scan step=%s attempt=%s/%s yaw_deg=%.1f",
                step.step_id,
                self._semantic_turn_recovery_attempts,
                max(0, int(self._config.semantic_turn_max_recovery_attempts)),
                float(np.degrees(observation.pose.yaw_rad)),
            )

        candidate_set, debug_image_path, _ = self._prepare_typed_step_candidates(observation=observation)
        requested_action = "TURN_LEFT" if step.direction == "left" else "TURN_RIGHT"
        turn_start_yaw = (
            float(observation.pose.yaw_rad)
            if self._semantic_turn_start_yaw_rad is None
            else float(self._semantic_turn_start_yaw_rad)
        )
        signed_yaw_delta = _wrap_angle_rad(float(observation.pose.yaw_rad) - turn_start_yaw)
        directed_yaw_deg = float(np.degrees(signed_yaw_delta))
        if step.direction == "right":
            directed_yaw_deg *= -1.0
        min_yaw_deg = max(1.0, float(self._config.semantic_turn_min_yaw_deg))
        max_yaw_deg = max(min_yaw_deg, float(self._config.semantic_turn_max_yaw_deg))
        sample = self._turn_corridor_sample(
            observation=observation,
            directed_yaw_deg=directed_yaw_deg,
        )
        if sample is not None and (
            not self._semantic_turn_samples
            or abs(float(np.degrees(_wrap_angle_rad(
                sample.robot_yaw_rad - self._semantic_turn_samples[-1].robot_yaw_rad
            ))))
            >= max(1.0, 0.35 * float(self._config.active_turn_angle_deg))
        ):
            self._semantic_turn_samples.append(sample)
            self._logger.info(
                "[TurnScan] step=%s yaw_deg=%.1f corridor_local_deg=%+.1f clearance_m=%.2f score=%.2f samples=%s",
                step.step_id,
                directed_yaw_deg,
                sample.local_heading_deg,
                sample.forward_clearance_m,
                sample.score,
                len(self._semantic_turn_samples),
            )
        elif not bool(self._semantic_turn_latest_heading.valid):
            self._logger.info(
                "[TurnScan] step=%s yaw_deg=%.1f corridor=invalid reason=%s",
                step.step_id,
                directed_yaw_deg,
                self._semantic_turn_latest_heading.reason,
            )

        if (
            self._semantic_turn_primitives_taken >= 1
            and (
                self._turn_cue_is_ready(step=step)
                or self._turn_lookahead_target_is_ready()
            )
        ):
            if self._turn_cue_is_ready(step=step):
                cue = step.cues[0]
                state = self._landmark_tracker_state
                return self._complete_semantic_turn(
                    observation=observation,
                    progress_signal=progress_state.progress_signal,
                    debug_image_path=debug_image_path,
                    directed_yaw_deg=directed_yaw_deg,
                    reason=(
                        f"turn_cue_confirmed label={cue.label} "
                        f"confidence={0.0 if state is None else state.confidence:.2f}"
                    ),
                )
            hypothesis = self._semantic_turn_target_hypothesis
            return self._complete_semantic_turn(
                observation=observation,
                progress_signal=progress_state.progress_signal,
                debug_image_path=debug_image_path,
                directed_yaw_deg=directed_yaw_deg,
                reason=(
                    f"lookahead_target_confirmed label={hypothesis.label} "
                    f"confidence={hypothesis.confidence:.2f} "
                    f"confirmations={hypothesis.confirmation_count}/"
                    f"{hypothesis.required_confirmations} "
                    f"world_anchor={int(hypothesis.has_world_anchor)}"
                ),
            )

        if self._semantic_turn_return_yaw_rad is not None:
            return_error_deg = float(np.degrees(_wrap_angle_rad(
                self._semantic_turn_return_yaw_rad - float(observation.pose.yaw_rad)
            )))
            return_tolerance_deg = max(
                float(self._config.active_turn_tolerance_deg),
                0.5 * float(self._config.active_turn_angle_deg),
            )
            if (
                abs(return_error_deg) <= return_tolerance_deg
                and bool(self._semantic_turn_latest_heading.valid)
                and float(self._semantic_turn_latest_heading.fused_clearance_m)
                >= float(self._config.semantic_turn_min_forward_clearance_m)
            ):
                return self._complete_semantic_turn(
                    observation=observation,
                    progress_signal=progress_state.progress_signal,
                    debug_image_path=debug_image_path,
                    directed_yaw_deg=directed_yaw_deg,
                    reason="best_corridor_heading_recovered",
                )
            if abs(return_error_deg) <= return_tolerance_deg:
                return self._request_turn_recovery(
                    progress_signal=progress_state.progress_signal,
                    debug_image_path=debug_image_path,
                    reason="turn_best_corridor_became_blocked",
                )
            correction_action = "TURN_LEFT" if return_error_deg > 0.0 else "TURN_RIGHT"
            correction_candidate = self._best_candidate_for_action(
                candidate_set=candidate_set,
                action_name=correction_action,
            )
            if correction_candidate is None:
                return self._request_turn_recovery(
                    progress_signal=progress_state.progress_signal,
                    debug_image_path=debug_image_path,
                    reason=(
                        "turn_alignment_candidate_unavailable "
                        f"heading_error_deg={return_error_deg:+.1f}"
                    ),
                )
            self._semantic_turn_active_counts_as_scan = False
            return self._start_candidate_primitive(
                observation=observation,
                candidate=correction_candidate,
                action_name=correction_action,
                speed_scale=float(_primitive_speed_scale(correction_action, self._config)),
                now=now,
                progress_signal=progress_state.progress_signal,
                decision_source="SEMANTIC_TURN_CORRECTION",
                reasoning_summary=f"return to best scanned corridor; heading_error_deg={return_error_deg:+.1f}",
                debug_image_path=debug_image_path,
            )

        if directed_yaw_deg >= min_yaw_deg and self._turn_alignment_is_stable():
            return self._complete_semantic_turn(
                observation=observation,
                progress_signal=progress_state.progress_signal,
                debug_image_path=debug_image_path,
                directed_yaw_deg=directed_yaw_deg,
                reason="stable_corridor_early_exit",
            )

        max_primitives = max(1, int(self._config.semantic_turn_max_primitives))
        scan_exhausted = (
            directed_yaw_deg >= max_yaw_deg
            or self._semantic_turn_primitives_taken >= max_primitives
        )
        if scan_exhausted:
            best_sample = self._best_turn_corridor_sample(
                step_direction=step.direction,
                start_yaw_rad=turn_start_yaw,
                min_yaw_deg=min_yaw_deg,
                max_yaw_deg=max_yaw_deg,
            )
            if best_sample is None:
                return self._request_turn_recovery(
                    progress_signal=progress_state.progress_signal,
                    debug_image_path=debug_image_path,
                    reason="turn_scan_no_safe_corridor",
                )
            self._semantic_turn_return_yaw_rad = float(best_sample.corridor_world_yaw_rad)
            self._logger.info(
                "[TurnScan] step=%s scan_complete=1 best_world_yaw_deg=%.1f best_score=%.2f",
                step.step_id,
                float(np.degrees(best_sample.corridor_world_yaw_rad)),
                best_sample.score,
            )
            return self._semantic_turn_step(
                observation=observation,
                progress_state=progress_state,
                now=now,
            )

        turn_candidate = self._best_candidate_for_action(
            candidate_set=candidate_set,
            action_name=requested_action,
        )
        if turn_candidate is None:
            self._logger.error(
                "[ERROR] step=%s type=TURN reason=no_safe_%s_candidate",
                step.step_id,
                requested_action,
            )
            return self._request_turn_recovery(
                progress_signal=progress_state.progress_signal,
                debug_image_path=debug_image_path,
                reason=f"turn_candidate_unavailable action={requested_action}",
            )

        self._semantic_turn_active_counts_as_scan = True
        return self._start_candidate_primitive(
            observation=observation,
            candidate=turn_candidate,
            action_name=requested_action,
            speed_scale=float(_primitive_speed_scale(requested_action, self._config)),
            now=now,
            progress_signal=progress_state.progress_signal,
            decision_source="SEMANTIC_TURN_PRIMITIVE",
            reasoning_summary=(
                f"typed TURN {step.direction}; primitive "
                f"{self._semantic_turn_primitives_taken + 1}/{max_primitives}; "
                f"directed_yaw_deg={directed_yaw_deg:.1f}"
            ),
            debug_image_path=debug_image_path,
        )

    def _turn_corridor_sample(
        self,
        *,
        observation: VLNObservation,
        directed_yaw_deg: float,
    ) -> _TurnCorridorSample | None:
        estimate = self._semantic_turn_latest_heading
        if not bool(estimate.valid):
            return None
        clearance_m = float(estimate.fused_clearance_m)
        if clearance_m < float(self._config.semantic_turn_min_forward_clearance_m):
            return None
        return _TurnCorridorSample(
            robot_yaw_rad=float(observation.pose.yaw_rad),
            directed_yaw_deg=float(directed_yaw_deg),
            corridor_world_yaw_rad=_wrap_angle_rad(
                float(observation.pose.yaw_rad)
                + float(np.deg2rad(estimate.local_heading_deg))
            ),
            local_heading_deg=float(estimate.local_heading_deg),
            forward_clearance_m=clearance_m,
            score=float(estimate.score),
        )

    def _turn_cue_is_ready(self, *, step: RouteStep) -> bool:
        if not step.cues:
            return False
        cue = step.cues[0]
        state = self._landmark_tracker_state
        observation = None if state is None else state.last_observation
        if state is None or observation is None:
            return False
        if str(state.label).strip().lower() != cue.label:
            return False
        return bool(
            observation.visible
            and observation.is_current_frame
            and observation.can_confirm
            and observation.vlm_verified
            and observation.turn_relation_verified
        )

    def _turn_lookahead_target_is_ready(self) -> bool:
        target_label = self._next_go_to_target().strip().lower()
        if not target_label:
            return False
        hypothesis = self._semantic_turn_target_hypothesis
        if hypothesis.label.strip().lower() != target_label:
            hypothesis = _TurnTargetHypothesis(label=target_label)
            self._semantic_turn_target_hypothesis = hypothesis

        state = self._landmark_tracker_state
        observation = None if state is None else state.last_observation
        if observation is None:
            return bool(hypothesis.ready)
        if str(observation.label).strip().lower() != target_label:
            return bool(hypothesis.ready)
        request_frame_index = int(getattr(observation, "request_frame_index", -1))
        if request_frame_index < 0 or request_frame_index == hypothesis.last_request_frame_index:
            return bool(hypothesis.ready)
        observation_source = str(
            getattr(observation, "source", "")
        ).strip().lower()
        if (
            not observation_source.startswith("grounding_dino")
            or "reused" in observation_source
        ):
            return bool(hypothesis.ready)

        hypothesis.last_request_frame_index = request_frame_index
        confidence = float(getattr(observation, "confidence", 0.0))
        valid_identity = bool(
            getattr(observation, "visible", False)
            and getattr(observation, "bbox", None) is not None
            and getattr(observation, "can_confirm", False)
            and getattr(observation, "vlm_verified", False)
            and getattr(observation, "is_current_frame", True)
        )
        medium_threshold = float(self._config.semantic_turn_target_medium_confidence)
        high_threshold = max(
            medium_threshold,
            float(self._config.semantic_turn_target_high_confidence),
        )
        direction_consistent = getattr(
            observation, "turn_direction_consistent", None
        )
        if (
            not valid_identity
            or confidence < medium_threshold
            or direction_consistent is False
        ):
            hypothesis.confirmation_count = 0
            hypothesis.required_confirmations = 0
            hypothesis.last_world_xy = None
            hypothesis.last_world_bearing_rad = None
            hypothesis.confidence = confidence
            hypothesis.ready = False
            hypothesis.has_world_anchor = False
            self._logger.info(
                "[TurnTarget] target=%s frame=%s accepted=0 confidence=%.2f "
                "identity=%s direction_consistent=%s reason=%s",
                target_label,
                request_frame_index,
                confidence,
                int(valid_identity),
                direction_consistent,
                str(getattr(observation, "reason", "unknown")),
            )
            return False

        request_pose = getattr(observation, "request_robot_pose", None)
        bearing_rad = getattr(observation, "bearing_rad", None)
        world_bearing_rad = None
        if request_pose is not None and bearing_rad is not None:
            # TargetObservation bearing is image/right-positive.
            world_bearing_rad = _wrap_angle_rad(
                float(request_pose.yaw_rad) - float(bearing_rad)
            )
        world_xy = getattr(observation, "estimated_world_xy", None)
        consistent_with_previous = _turn_target_observation_consistent(
            previous_world_xy=hypothesis.last_world_xy,
            current_world_xy=world_xy,
            previous_world_bearing_rad=hypothesis.last_world_bearing_rad,
            current_world_bearing_rad=world_bearing_rad,
        )
        if hypothesis.confirmation_count <= 0 or consistent_with_previous:
            hypothesis.confirmation_count += 1
        else:
            hypothesis.confirmation_count = 1

        relation_extra = int(bool(getattr(observation, "turn_relation_verified", False)))
        base_required = 1 if confidence >= high_threshold else 2
        hypothesis.required_confirmations = max(
            hypothesis.required_confirmations,
            base_required + relation_extra,
        )
        hypothesis.last_world_xy = (
            None if world_xy is None else [float(world_xy[0]), float(world_xy[1])]
        )
        hypothesis.last_world_bearing_rad = world_bearing_rad
        hypothesis.confidence = confidence
        hypothesis.has_world_anchor = world_xy is not None
        hypothesis.ready = bool(
            hypothesis.confirmation_count >= hypothesis.required_confirmations
        )
        self._logger.info(
            "[TurnTarget] target=%s frame=%s accepted=1 confidence=%.2f "
            "confirm=%s/%s consistent=%s relation_review=%s depth=%s world=%s ready=%s",
            target_label,
            request_frame_index,
            confidence,
            hypothesis.confirmation_count,
            hypothesis.required_confirmations,
            int(consistent_with_previous),
            int(bool(getattr(observation, "turn_relation_verified", False))),
            "none"
            if getattr(observation, "depth_m", None) is None
            else f"{float(observation.depth_m):.2f}",
            hypothesis.last_world_xy,
            int(hypothesis.ready),
        )
        return bool(hypothesis.ready)

    def _request_turn_recovery(
        self,
        *,
        progress_signal: str,
        debug_image_path: str | Path | None,
        reason: str,
    ) -> PolicyStepResult:
        maximum = max(0, int(self._config.semantic_turn_max_recovery_attempts))
        if self._semantic_turn_recovery_attempts >= maximum:
            self._logger.error(
                "[TURN_SAFETY_HOLD] reason=%s recovery_attempts=%s/%s",
                reason,
                self._semantic_turn_recovery_attempts,
                maximum,
            )
            return self._hold_result(
                decision_source="TURN_FATAL_SAFETY_HOLD",
                reasoning_summary=(
                    f"bounded TURN recovery exhausted ({maximum} rounds): {reason}"
                ),
                progress_signal=progress_signal,
                debug_image_path=debug_image_path,
                replanned=False,
                last_error="turn_recovery_exhausted",
            )
        self._semantic_turn_recovery_attempts += 1
        self._semantic_turn_restart_after_recovery = True
        self._semantic_turn_primitives_taken = 0
        self._semantic_turn_active_counts_as_scan = False
        self._semantic_turn_samples.clear()
        self._semantic_turn_return_yaw_rad = None
        self._clear_active_goal()
        self._logger.warning(
            "[TurnRecoveryRequest] reason=%s attempt=%s/%s",
            reason,
            self._semantic_turn_recovery_attempts,
            maximum,
        )
        return self._hold_result(
            decision_source="TURN_RECOVERY_REQUEST",
            reasoning_summary=reason,
            progress_signal=progress_signal,
            debug_image_path=debug_image_path,
            replanned=True,
            last_error=reason,
        )

    def _turn_alignment_is_stable(self) -> bool:
        required = max(2, int(self._config.semantic_turn_alignment_confirmations))
        if len(self._semantic_turn_samples) < required:
            return False
        recent = self._semantic_turn_samples[-required:]
        latest = recent[-1]
        if abs(latest.local_heading_deg) > float(
            self._config.semantic_turn_corridor_center_tolerance_deg
        ):
            return False
        reference_yaw = recent[0].corridor_world_yaw_rad
        max_delta_deg = max(
            abs(float(np.degrees(_wrap_angle_rad(item.corridor_world_yaw_rad - reference_yaw))))
            for item in recent[1:]
        )
        return max_delta_deg <= float(self._config.semantic_turn_same_corridor_tolerance_deg)

    def _best_turn_corridor_sample(
        self,
        *,
        step_direction: str,
        start_yaw_rad: float,
        min_yaw_deg: float,
        max_yaw_deg: float,
    ) -> _TurnCorridorSample | None:
        eligible: list[_TurnCorridorSample] = []
        for sample in self._semantic_turn_samples:
            delta_deg = float(np.degrees(_wrap_angle_rad(
                sample.corridor_world_yaw_rad - float(start_yaw_rad)
            )))
            if step_direction == "right":
                delta_deg *= -1.0
            if min_yaw_deg <= delta_deg <= max_yaw_deg:
                eligible.append(sample)
        if not eligible:
            return None
        return max(
            eligible,
            key=lambda item: (item.score, -abs(item.local_heading_deg)),
        )

    def _complete_semantic_turn(
        self,
        *,
        observation: VLNObservation,
        progress_signal: str,
        debug_image_path: str | Path | None,
        directed_yaw_deg: float,
        reason: str,
    ) -> PolicyStepResult:
        step = self._current_route_step()
        self._logger.info(
            "[STEP_COMPLETE] step=%s type=TURN reason=%s yaw_deg=%.1f primitives=%s samples=%s",
            "none" if step is None else step.step_id,
            reason,
            directed_yaw_deg,
            self._semantic_turn_primitives_taken,
            len(self._semantic_turn_samples),
        )
        self._advance_current_step(
            observation=observation,
            source="SemanticTurn",
            reason=reason,
        )
        return self._hold_result(
            decision_source="TURN_COMPLETE",
            reasoning_summary=reason,
            progress_signal=progress_signal,
            debug_image_path=debug_image_path,
            replanned=False,
        )

    def _region_transition_step(
        self,
        *,
        observation: VLNObservation,
        progress_state,
        now: float,
    ) -> PolicyStepResult:
        step = self._current_route_step()
        if step is None or step.step_type != "REGION_TRANSITION":
            raise RuntimeError("region transition handler called without a REGION_TRANSITION step")

        active_result = self._continue_current_step_primitive(
            observation=observation,
            progress_signal=progress_state.progress_signal,
            now=now,
        )
        if active_result is not None:
            return active_result

        candidate_set, debug_image_path, annotated_png_bytes = self._prepare_typed_step_candidates(
            observation=observation
        )
        decision = self._reasoner.choose_candidate(
            instruction_context=self._instruction_context,
            candidate_set=candidate_set,
            progress_state=progress_state,
            annotated_image_png_bytes=annotated_png_bytes,
            target_memory_text="none",
            safety_context_text=self._safety_context_text,
            route_context_text=self._route_context_text(robot_pose=observation.pose),
            reacquire_context_text="none",
            obstacle_summary_text=self._obstacle_summary_text,
            semantic_map_summary=self._semantic_map_summary,
            current_step=step,
        )
        self._safety_context_text = ""
        transition_state = self._region_transition.update(
            step=step,
            robot_pose=observation.pose,
            topology_node_id=self._topology_node_id,
            current_region_label=decision.current_region_label,
            region_confidence=decision.region_confidence,
            region_changed=decision.region_changed,
        )
        if transition_state.completed:
            self._logger.info(
                "[STEP_COMPLETE] step=%s type=REGION_TRANSITION region=%s reason=%s",
                step.step_id,
                transition_state.current_region_label,
                transition_state.completion_reason,
            )
            self._advance_current_step(
                observation=observation,
                source="RegionTransition",
                reason=transition_state.completion_reason,
            )
            return self._hold_result(
                decision_source="REGION_TRANSITION_COMPLETE",
                reasoning_summary=transition_state.completion_reason,
                progress_signal=progress_state.progress_signal,
                debug_image_path=debug_image_path,
                replanned=False,
            )

        selected_candidate = _find_candidate(candidate_set, decision.selected_candidate_id)
        if selected_candidate is None or not bool(selected_candidate.valid):
            self._logger.error(
                "[ERROR] step=%s type=REGION_TRANSITION decision=%s reason=%s",
                step.step_id,
                decision.selected_candidate_id,
                decision.reason,
            )
            return self._hold_result(
                decision_source="SAFETY_HOLD",
                reasoning_summary=decision.thought,
                progress_signal=progress_state.progress_signal,
                debug_image_path=debug_image_path,
                replanned=True,
                last_error=f"region_decision:{decision.reason}",
            )

        action_name, speed_scale = _candidate_to_primitive_action(
            selected_candidate,
            turn_speed_scale=float(self._config.turn_speed_scale),
            forward_speed_scale=float(self._config.forward_speed_scale),
        )
        return self._start_candidate_primitive(
            observation=observation,
            candidate=selected_candidate,
            action_name=action_name,
            speed_scale=speed_scale,
            now=now,
            progress_signal=progress_state.progress_signal,
            decision_source="REGION_TRANSITION_PRIMITIVE",
            reasoning_summary=decision.thought,
            debug_image_path=debug_image_path,
        )

    def _continue_current_step_primitive(
        self,
        *,
        observation: VLNObservation,
        progress_signal: str,
        now: float,
    ) -> PolicyStepResult | None:
        step = self._current_route_step()
        if (
            step is None
            or self._active_route_step_id != step.step_id
            or self._active_world_goal_xy is None
            or self._execution_steps_remaining <= 0
        ):
            return None
        local_goal = _world_waypoint_to_local_xy(
            robot_pose=observation.pose,
            world_goal_xy=self._active_world_goal_xy,
        )
        return self._motion_primitive_step(
            observation=observation,
            local_goal=local_goal,
            progress_signal=progress_signal,
            now=now,
        )

    def _prepare_typed_step_candidates(
        self,
        *,
        observation: VLNObservation,
    ) -> tuple[CandidateSet, str | Path | None, bytes | None]:
        candidate_set = self._candidate_predictor.predict(observation)
        if self._map_candidate_generator is not None:
            candidate_set = self._map_candidate_generator.augment(
                observation=observation,
                base_candidate_set=candidate_set,
                occupancy_snapshot=(
                    None if self._occupancy_grid is None else self._occupancy_grid.snapshot()
                ),
                current_target=self.current_perception_target,
                landmark_state=self._landmark_tracker_state,
                landmark_entry=self._semantic_landmark_entry(),
                progress_signal=self._progress_manager.state.progress_signal,
                safety_context_text=self._safety_context_text,
            )
        self._annotate_candidate_context(candidate_set=candidate_set)
        self._last_candidate_set = candidate_set
        debug_image_path = None
        image_bytes = _clean_rgb_png_bytes(observation.rgb)
        return candidate_set, debug_image_path, image_bytes

    def _best_candidate_for_action(
        self,
        *,
        candidate_set: CandidateSet,
        action_name: str,
    ) -> WaypointCandidate | None:
        matches = []
        for candidate in candidate_set.candidates:
            if not bool(candidate.valid):
                continue
            candidate_action, _ = _candidate_to_primitive_action(
                candidate,
                turn_speed_scale=float(self._config.turn_speed_scale),
                forward_speed_scale=float(self._config.forward_speed_scale),
            )
            if candidate_action == action_name:
                matches.append(candidate)
        if not matches:
            return None
        return max(
            matches,
            key=lambda item: (
                float(item.clearance_score),
                float(item.depth_support.get("min_clearance_m", 0.0)),
                -abs(float(item.heading_delta_deg)),
            ),
        )

    def _hold_result(
        self,
        *,
        decision_source: str,
        reasoning_summary: str,
        progress_signal: str,
        debug_image_path: str | Path | None,
        replanned: bool,
        last_error: str = "",
    ) -> PolicyStepResult:
        return PolicyStepResult(
            action_name="STOP",
            speed_scale=0.0,
            decision_source=decision_source,
            selected_candidate_id=None,
            local_goal_xy=[0.0, 0.0],
            world_goal_xy=None,
            execution_steps_remaining=0,
            reasoning_summary=reasoning_summary,
            progress_signal=progress_signal,
            replanned=replanned,
            debug_image_path=None if debug_image_path is None else str(debug_image_path),
            last_error=last_error,
            **self._route_result_kwargs(),
        )

    def _semantic_landmark_entry(self) -> LandmarkMapEntry | None:
        if self.current_step_type != "GO_TO" or not self.current_perception_target:
            return None
        current_target = self._current_target_text().strip().lower()
        if not current_target or current_target in {"none", "null", "unknown", "n/a", "na", "target", "object", "thing", "place", "location"}:
            return None
        if self._landmark_map_entry is not None and bool(self._landmark_map_entry.world_xy):
            if not bool(getattr(self._landmark_map_entry, "confirmed", False)):
                self._logger.info(
                    "[SemanticPlanner] skip target=%s evidence=%s reason=map_entry_not_confirmed",
                    getattr(self._landmark_map_entry, "label", self._current_target_text()),
                    getattr(self._landmark_map_entry, "evidence_level", "unknown"),
                )
                return None
            return self._landmark_map_entry
        state = self._landmark_tracker_state
        if state is None:
            return None
        # Semantic planner only chases confirmed anchors, not detections/potential
        if str(state.evidence_level) != "confirmed" or not bool(state.can_complete):
            if state.potential_world_xy is not None or state.candidate_world_xy is not None:
                obs_source = "unknown"
                if state.last_observation is not None:
                    obs_source = str(getattr(state.last_observation, "semantic_source", "unknown"))
                self._logger.info(
                    "[SemanticPlanner] skip target=%s evidence=%s can_complete=%s source=%s reason=not_confirmed",
                    state.label, state.evidence_level, int(bool(state.can_complete)), obs_source,
                )
            return None
        world_xy = state.estimated_world_xy
        if world_xy is None:
            return None
        return LandmarkMapEntry(
            label=str(state.label),
            behavior_type=str(state.behavior_type),
            world_xy=[float(world_xy[0]), float(world_xy[1])],
            confidence=float(state.confirmed_confidence),
            first_seen_time_sec=float(state.last_seen_time_sec or 0.0),
            last_seen_time_sec=float(state.last_seen_time_sec or 0.0),
            update_count=0,
            status=f"{state.status}_confirmed",
            source="LandmarkTracker.confirmed",
            depth_m=state.last_confirmed_depth_m,
            bbox=None if state.last_observation is None else state.last_observation.bbox,
            anchor_distance_m=float(state.anchor_distance_m) if state.anchor_distance_m is not None else None,
            robot_forward_m=None,
            robot_left_m=None,
            reached=bool(state.reached),
        )

    def _current_grounding_summary(self) -> str:
        state = self._landmark_tracker_state
        observation = None if state is None else state.last_observation
        if observation is None:
            return "none"
        bbox = getattr(observation, "bbox", None)
        center_uv = getattr(observation, "center_uv", None)
        depth_m = getattr(observation, "depth_m", None)
        bearing_rad = getattr(observation, "bearing_rad", None)
        world_xy = getattr(observation, "estimated_world_xy", None)
        source = str(getattr(observation, "source", "unknown"))
        reason = str(getattr(observation, "reason", "unknown"))
        fresh = bool(
            getattr(observation, "visible", False)
            and bbox is not None
            and reason.strip().lower() == "ok"
            and "reused" not in source.strip().lower()
        )
        proposal_visible = bool(getattr(observation, "visible", False))
        verified_visible = bool(
            proposal_visible
            and getattr(observation, "can_confirm", False)
        )
        return (
            f"label={getattr(observation, 'label', self._current_target_text())} "
            f"verified_visible={int(verified_visible)} proposal_visible={int(proposal_visible)} "
            f"fresh={int(fresh)} reason={reason} source={source} "
            f"bbox={list(bbox) if bbox is not None else 'none'} "
            f"center_uv={_format_xy(center_uv) if center_uv is not None else 'none'} "
            f"depth_m={_format_optional_float(depth_m)} "
            f"bearing_deg={_format_optional_float(None if bearing_rad is None else np.degrees(float(bearing_rad)))} "
            f"world_xy={_format_xy(world_xy) if world_xy is not None else 'none'} "
            f"depth_cluster_area={float(getattr(observation, 'depth_cluster_area_ratio', 0.0)):.4f} "
            f"depth_region={getattr(observation, 'depth_region_source', 'none')} "
            f"evidence={getattr(observation, 'evidence_level', 'none')} "
            f"can_confirm={int(bool(getattr(observation, 'can_confirm', False)))} "
            f"vlm_verified={int(bool(getattr(observation, 'vlm_verified', False)))} "
            f"vlm_conf={float(getattr(observation, 'vlm_verification_confidence', 0.0)):.2f}"
        )

    def _motion_primitive_step(
        self,
        *,
        observation: VLNObservation,
        local_goal: tuple[float, float],
        progress_signal: str,
        now: float,
    ) -> PolicyStepResult:
        action_name = str(self._active_action_name).strip().upper()
        if action_name not in {"FORWARD", "TURN_LEFT", "TURN_RIGHT"}:
            self._clear_active_goal()
            return PolicyStepResult(
                action_name="STOP",
                speed_scale=0.0,
                decision_source="PRIMITIVE_INVALID",
                selected_candidate_id=None,
                local_goal_xy=[0.0, 0.0],
                world_goal_xy=None,
                execution_steps_remaining=0,
                reasoning_summary="invalid primitive action; stopped for replanning",
                progress_signal=progress_signal,
                replanned=False,
                debug_image_path=None,
                **self._route_result_kwargs(),
            )

        if self._active_start_yaw_rad is None:
            self._active_start_yaw_rad = float(observation.pose.yaw_rad)
        if self._active_start_pose is None:
            self._active_start_pose = observation.pose
        self._update_active_action_timer(action_name=action_name, now=now)

        elapsed_sec = self._active_elapsed_sec(now=now)
        max_duration_sec = self._active_action_duration_sec(action_name)
        remaining_after_step = max(0, int(self._execution_steps_remaining) - 1)
        if elapsed_sec < max_duration_sec:
            remaining_after_step = max(1, remaining_after_step)
        self._execution_steps_remaining = remaining_after_step

        yaw_delta_rad = abs(
            _wrap_angle_rad(float(observation.pose.yaw_rad) - float(self._active_start_yaw_rad))
        )
        traveled_m = _pose_distance_m(self._active_start_pose, observation.pose)
        target_yaw_rad = float(np.deg2rad(max(1.0, float(self._config.active_turn_angle_deg))))
        yaw_tolerance_rad = float(np.deg2rad(max(0.1, float(self._config.active_turn_tolerance_deg))))
        forward_target_m = max(0.05, float(self._config.active_forward_distance_m))
        distance_tolerance_m = max(0.0, float(self._config.active_distance_tolerance_m))
        is_turn = action_name in {"TURN_LEFT", "TURN_RIGHT"}
        completion_reason = ""
        if is_turn and yaw_delta_rad >= max(0.0, target_yaw_rad - yaw_tolerance_rad):
            completion_reason = "odom_target_yaw_reached"
        elif action_name == "FORWARD" and traveled_m >= max(0.0, forward_target_m - distance_tolerance_m):
            completion_reason = "odom_forward_distance_reached"
        elif elapsed_sec >= max_duration_sec:
            completion_reason = "timeout_reached"

        if completion_reason:
            completed_candidate_id = self._active_candidate_id
            completed_start_pose = self._active_start_pose
            completed_world_goal_xy = (
                None
                if is_turn or self._active_world_goal_xy is None
                else list(self._active_world_goal_xy)
            )
            self._logger.info(
                "[Primitive] complete action=%s reason=%s elapsed_sec=%.2f max_sec=%.2f odom_distance_m=%.2f target_distance_m=%.2f yaw_delta_deg=%.1f target_yaw_deg=%.1f",
                action_name,
                completion_reason,
                float(elapsed_sec),
                float(max_duration_sec),
                float(traveled_m),
                float(forward_target_m if action_name == "FORWARD" else 0.0),
                float(np.degrees(yaw_delta_rad)),
                float(np.degrees(target_yaw_rad)) if is_turn else 0.0,
            )
            self._clear_active_goal()
            return PolicyStepResult(
                action_name="STOP",
                speed_scale=0.0,
                decision_source="PRIMITIVE_COMPLETE",
                selected_candidate_id=completed_candidate_id,
                local_goal_xy=[float(local_goal[0]), float(local_goal[1])],
                world_goal_xy=completed_world_goal_xy,
                execution_steps_remaining=0,
                reasoning_summary=f"{action_name} primitive completed by {completion_reason}; stopping and replanning",
                progress_signal=progress_signal,
                replanned=False,
                debug_image_path=None,
                completed_primitive_action=action_name,
                completed_primitive_reason=completion_reason,
                completed_primitive_start_pose=(
                    None
                    if completed_start_pose is None
                    else (
                        float(completed_start_pose.x),
                        float(completed_start_pose.y),
                        float(completed_start_pose.yaw_rad),
                    )
                ),
                completed_primitive_end_pose=(
                    float(observation.pose.x),
                    float(observation.pose.y),
                    float(observation.pose.yaw_rad),
                ),
                **self._route_result_kwargs(),
            )

        return PolicyStepResult(
            action_name=action_name,
            speed_scale=float(_primitive_speed_scale(action_name, self._config)),
            decision_source="ACTIVE_PRIMITIVE",
            selected_candidate_id=self._active_candidate_id,
            local_goal_xy=[float(local_goal[0]), float(local_goal[1])],
            world_goal_xy=(
                None
                if is_turn or self._active_world_goal_xy is None
                else list(self._active_world_goal_xy)
            ),
            execution_steps_remaining=int(self._execution_steps_remaining),
            reasoning_summary=self._last_decision_summary,
            progress_signal=progress_signal,
            replanned=False,
            debug_image_path=None,
            **self._route_result_kwargs(),
        )

    def _current_target_text(self) -> str:
        route_step = self._current_route_step()
        return "" if route_step is None else route_step.perception_target

    def _next_go_to_target(self) -> str:
        for step in self._route_steps[self._current_step_index + 1 :]:
            if step.step_type == "GO_TO" and str(step.target).strip():
                return str(step.target).strip()
        return ""

    def _current_route_step(self) -> RouteStep | None:
        if self._current_step_index < len(self._route_steps):
            return self._route_steps[self._current_step_index]
        return None

    def _route_step_count(self) -> int:
        return len(self._route_steps)

    def _route_result_kwargs(self) -> dict[str, Any]:
        current_step = self._current_route_step()
        current_target = ""
        if current_step is not None:
            current_target = (
                current_step.target
                or current_step.to_region
                or (
                    current_step.cues[0].label
                    if current_step.cues
                    else ""
                )
                or (f"turn {current_step.direction}" if current_step.step_type == "TURN" else "")
            )
        return {
            "current_landmark": "" if self._route_complete else self.current_perception_target,
            "completed_landmarks": list(self._completed_landmarks),
            "route_complete": bool(self._route_complete),
            "current_step_index": int(self._current_step_index),
            "route_step_count": int(self._route_step_count()),
            "current_step_id": "" if current_step is None else current_step.step_id,
            "current_step_type": "" if current_step is None else current_step.step_type,
            "current_target": current_target,
            "navigation_status": "ROUTE_COMPLETE" if self._route_complete else "RUNNING",
        }

    def _current_arrival_distance_m(self) -> float:
        if self._route_step_count() > 0 and self._current_step_index >= self._route_step_count() - 1:
            return max(0.05, float(self._config.success_distance_m))
        return max(0.05, float(self._config.arrival_distance_m))

    def _maybe_advance_landmark(self, *, observation: VLNObservation) -> bool:
        if self._route_step_count() <= 0 or self._route_complete:
            self._arrival_confirmation_window.clear()
            return False

        tracker_state = self._landmark_tracker_state
        if (
            tracker_state is not None
            and str(tracker_state.label).strip().lower() == self._current_target_text().strip().lower()
            and bool(tracker_state.reached)
            and bool(tracker_state.can_complete)
        ):
            candidate_anchor = tracker_state.estimated_world_xy or tracker_state.candidate_world_xy
            visible_depth_reached = str(tracker_state.reach_reason).startswith(
                "confirmed_visible_depth"
            )
            if not visible_depth_reached and not self._route_anchor_is_consistent(candidate_anchor):
                return False
            distance_m = tracker_state.anchor_distance_m
            if distance_m is None and tracker_state.last_observation is not None:
                distance_m = tracker_state.last_observation.depth_m
            return self._advance_current_landmark(
                observation=observation,
                anchor_world_xy=candidate_anchor,
                distance_m=distance_m,
                source="LandmarkTracker",
                reason=tracker_state.reach_reason,
                visible_now=(
                    tracker_state.last_observation is not None
                    and bool(tracker_state.last_observation.visible)
                    and int(tracker_state.lost_count) == 0
                ),
                lost_count=int(tracker_state.lost_count),
            )

        map_entry = self._landmark_map_entry
        if (
            map_entry is not None
            and str(map_entry.label).strip().lower() == self._current_target_text().strip().lower()
            and bool(map_entry.confirmed)
            and str(getattr(map_entry, "evidence_level", "confirmed")) == "confirmed"
            and int(getattr(map_entry, "confirmed_observation_count", 0)) >= max(1, int(self._config.confirm_min_observations))
            and float(getattr(map_entry, "confidence", 0.0)) >= float(self._config.confirm_confidence_threshold)
            and map_entry.anchor_distance_m is not None
            and self.current_step_type == "GO_TO"
        ):
            threshold_m = max(0.05, float(self._current_arrival_distance_m()))
            if float(map_entry.anchor_distance_m) <= threshold_m:
                if not self._route_anchor_is_consistent(map_entry.world_xy):
                    return False
                required_confirmations = max(1, int(self._config.landmark_arrival_confirmations))
                self._arrival_confirmation_window.append(True)
                hit_count = int(sum(1 for item in self._arrival_confirmation_window if item))
                if hit_count < required_confirmations:
                    self._logger.info(
                        "[LandmarkCheck] target=%s source=LandmarkMap distance=%.2fm threshold=%.2fm confirm=%s/%s window=%s forward=%s left=%s",
                        self._current_target_text(),
                        float(map_entry.anchor_distance_m),
                        threshold_m,
                        int(hit_count),
                        int(required_confirmations),
                        [int(item) for item in self._arrival_confirmation_window],
                        _format_optional_float(map_entry.robot_forward_m),
                        _format_optional_float(map_entry.robot_left_m),
                    )
                    return False
                return self._advance_current_landmark(
                    observation=observation,
                    anchor_world_xy=map_entry.world_xy,
                    distance_m=float(map_entry.anchor_distance_m),
                    source="LandmarkMap",
                    reason=f"confirmed_anchor_distance<={threshold_m:.2f}m",
                    visible_now=False,
                    lost_count=0,
                )
            self._arrival_confirmation_window.append(False)
            return False

        self._arrival_confirmation_window.append(False)
        return False

    def _route_anchor_is_consistent(self, anchor_world_xy: list[float] | None) -> bool:
        if anchor_world_xy is None or not self._completed_landmarks:
            return True
        previous = self._completed_landmarks[-1]
        previous_anchor = self._landmark_anchors.get(previous)
        previous_yaw = self._landmark_anchor_yaws.get(previous)
        if previous_anchor is None or previous_yaw is None:
            return True
        expected_words = self._expected_direction_words()
        if not expected_words:
            return True
        relative_forward_m, relative_left_m = _relative_xy_from_anchor(
            anchor_world_xy=previous_anchor,
            anchor_yaw_rad=previous_yaw,
            target_world_xy=anchor_world_xy,
        )
        contradiction = False
        if "left" in expected_words and relative_left_m < -0.50:
            contradiction = True
        if "right" in expected_words and relative_left_m > 0.50:
            contradiction = True
        if {"forward", "straight", "ahead"} & expected_words and relative_forward_m < -0.50:
            contradiction = True
        if contradiction:
            reject_key = (previous, self._current_target_text())
            if reject_key != self._last_route_relation_reject_key:
                self._logger.warning(
                    "[RouteRelationReject] previous=%s current=%s expected=%s forward=%.2f left=%.2f anchor=%s",
                    previous,
                    self._current_target_text(),
                    sorted(expected_words),
                    float(relative_forward_m),
                    float(relative_left_m),
                    _format_xy(anchor_world_xy),
                )
                self._last_route_relation_reject_key = reject_key
            return False
        self._last_route_relation_reject_key = None
        return True

    def _advance_current_landmark(
        self,
        *,
        observation: VLNObservation,
        anchor_world_xy: list[float] | None,
        distance_m: float | None,
        source: str,
        reason: str,
        visible_now: bool,
        lost_count: int,
    ) -> bool:
        reached_landmark = self._current_target_text()
        self._last_route_relation_reject_key = None
        self._arrival_confirmation_window.clear()
        if anchor_world_xy is not None:
            self._landmark_anchors[reached_landmark] = list(anchor_world_xy)
            self._landmark_anchor_yaws[reached_landmark] = float(observation.pose.yaw_rad)
        if reached_landmark not in self._completed_landmarks:
            self._completed_landmarks.append(reached_landmark)
        distance_text = "none" if distance_m is None else f"{float(distance_m):.2f}"
        self._logger.info(
            "[LANDMARK_CONFIRMED] target=%s source=%s reason=%s distance=%sm visible_now=%s lost_count=%s",
            reached_landmark,
            source,
            reason,
            distance_text,
            int(bool(visible_now)),
            int(lost_count),
        )
        self._advance_current_step(
            observation=observation,
            source=source,
            reason=reason,
        )
        return True

    def _advance_current_step(
        self,
        *,
        observation: VLNObservation,
        source: str,
        reason: str,
    ) -> None:
        completed_step = self._current_route_step()
        if completed_step is None:
            return
        if completed_step.step_id not in self._completed_step_ids:
            self._completed_step_ids.append(completed_step.step_id)
        if completed_step.step_type == "TURN" and self._completed_landmarks:
            previous = self._completed_landmarks[-1]
            if previous in self._landmark_anchors:
                self._landmark_anchor_yaws[previous] = float(observation.pose.yaw_rad)

        self._current_step_index += 1
        self._arrival_confirmation_window.clear()
        self._semantic_turn_step_id = ""
        self._semantic_turn_start_yaw_rad = None
        self._semantic_turn_primitives_taken = 0
        self._semantic_turn_samples.clear()
        self._semantic_turn_return_yaw_rad = None
        self._semantic_turn_latest_heading = TurnHeadingEstimate(
            valid=False,
            reason="step_advanced",
        )
        self._semantic_turn_target_hypothesis = _TurnTargetHypothesis()
        self._semantic_turn_recovery_attempts = 0
        self._semantic_turn_restart_after_recovery = False
        self._semantic_turn_active_counts_as_scan = False
        self._region_transition.reset()
        self._landmark_tracker_state = None
        self._landmark_map_entry = None
        self._clear_active_goal()

        if self._current_step_index >= self._route_step_count():
            self._route_complete = True
            self._logger.info(
                "[ROUTE_COMPLETE] completed_steps=%s completed_landmarks=%s source=%s reason=%s",
                self._completed_step_ids,
                self._completed_landmarks,
                source,
                reason,
            )
            return

        next_step = self._current_route_step()
        next_target = self.current_perception_target
        self._target_memory.reset_for_target(next_target)
        self._logger.info(
            "[NAV_STATE] step_advance completed=%s next=%s next_type=%s next_target=%s",
            completed_step.step_id,
            next_step.step_id if next_step is not None else "none",
            next_step.step_type if next_step is not None else "none",
            next_target or "none",
        )

    def _route_context_text(self, *, robot_pose) -> str:
        current_index = self._current_step_index
        current_route_step = self._current_route_step()
        expected_words = self._expected_direction_words()
        route_steps = [
            (
                f"{step.step_id}:{step.step_type}:target={step.target or 'none'}:"
                f"from={step.from_region or 'none'}:to={step.to_region or 'none'}:"
                f"direction={step.direction}:completion={step.completion}:"
                f"cues={[f'{cue.role}:{cue.label}:{cue.expected_side}' for cue in step.cues] or ['none']}:"
                f"relations={[f'{relation.target}:{relation.relation}:{relation.reference}' for relation in step.relations] or ['none']}:"
                f"status={'complete' if step.step_id in self._completed_step_ids else ('current' if step == current_route_step else 'pending')}"
            )
            for step in self._route_steps
        ]
        lines = [
            f"route_steps={route_steps or ['none']}",
            f"current_route_step_index={current_index}",
            f"current_step_type={self.current_step_type or 'none'}",
            f"current_perception_target={self.current_perception_target or 'none'}",
            f"completed_landmarks={self._completed_landmarks}",
            f"expected_direction_words={sorted(expected_words) if expected_words else 'none'}",
            f"global_route_direction_hint={self._global_route_direction_hint()}",
        ]
        if self._topology_summary:
            lines.append(self._topology_summary)
        if self._completed_landmarks:
            previous = self._completed_landmarks[-1]
            previous_anchor = self._landmark_anchors.get(previous)
            previous_yaw = self._landmark_anchor_yaws.get(previous)
            lines.append(f"previous_landmark={previous} previous_anchor_world_xy={previous_anchor}")
            if previous_anchor is not None and previous_yaw is not None:
                relative_to_previous = _relative_xy_from_anchor(
                    anchor_world_xy=previous_anchor,
                    anchor_yaw_rad=previous_yaw,
                    target_world_xy=[float(robot_pose.x), float(robot_pose.y)],
                )
                side = _relative_side(local_forward_m=relative_to_previous[0], local_left_m=relative_to_previous[1])
                lines.append(
                    "robot_relative_to_previous_landmark="
                    f"{side}, forward={relative_to_previous[0]:.2f}m, left={relative_to_previous[1]:.2f}m"
                )
        if self._target_memory.state.last_seen_world_xy is not None and self._completed_landmarks:
            previous = self._completed_landmarks[-1]
            previous_anchor = self._landmark_anchors.get(previous)
            previous_yaw = self._landmark_anchor_yaws.get(previous)
            if previous_anchor is not None and previous_yaw is not None:
                relative = _relative_xy_from_anchor(
                    anchor_world_xy=previous_anchor,
                    anchor_yaw_rad=previous_yaw,
                    target_world_xy=self._target_memory.state.last_seen_world_xy,
                )
                side = _relative_side(local_forward_m=relative[0], local_left_m=relative[1])
                consistency = _direction_consistency(expected_words=expected_words, observed_side=side)
                lines.append(
                    "current_target_anchor_relative_to_previous="
                    f"{side}, forward={relative[0]:.2f}m, left={relative[1]:.2f}m, consistency={consistency}"
                )
        return "; ".join(lines)

    def _global_route_direction_hint(self) -> str:
        if not self._route_steps:
            return "none"
        start = max(0, int(self._current_step_index) - 2)
        end = min(len(self._route_steps), int(self._current_step_index) + 1)
        window = self._route_steps[start:end]
        parts: list[str] = []
        for absolute_index, step in enumerate(window, start=start):
            marker = "CURRENT" if absolute_index == self._current_step_index else "PREV"
            parts.append(
                f"{marker}[{absolute_index}]={step.step_type}:"
                f"target={step.target or step.to_region or 'none'}:direction={step.direction}:"
                f"constraints={list(step.constraints) or ['none']}"
            )
        return (
            f"recent_route_window={' | '.join(parts)}; "
            "follow the typed current step while preserving pending global constraints"
        )

    def _current_direction_clause(self) -> str:
        route_step = self._current_route_step()
        return "" if route_step is None else route_step.source_span

    def _expected_direction_words(self) -> set[str]:
        route_step = self._current_route_step()
        if route_step is None or route_step.direction == "none":
            return set()
        return {route_step.direction}

    def _update_active_action_timer(self, *, action_name: str, now: float) -> None:
        if self._active_goal_started_sec is None or action_name != self._active_action_name:
            self._active_goal_started_sec = float(now)
            self._active_action_name = str(action_name)

    def _active_elapsed_sec(self, *, now: float) -> float:
        if self._active_goal_started_sec is None:
            return 0.0
        return max(0.0, float(now) - float(self._active_goal_started_sec))

    def _active_action_duration_sec(self, action_name: str) -> float:
        action = str(action_name).strip().upper()
        if action == "FORWARD":
            return max(0.05, float(self._config.active_forward_duration_sec))
        if action in {"TURN_LEFT", "TURN_RIGHT"}:
            return max(0.05, float(self._config.active_turn_duration_sec))
        return 0.05
