from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .candidate_waypoint_predictor import CandidateSet, CandidateWaypointPredictor, WaypointCandidate
from .instruction_manager import InstructionContext, RouteStep
from .landmark_map import LandmarkMapEntry
from .landmark_tracker import LandmarkState
from .map_candidate_generator import MapCandidateGenerator
from .observation_adapter import VLNObservation
from .progress_manager import ProgressManager
from .target_memory import TargetMemory
from .vln_reasoner import OpenNavReasoner
from .policy_helpers import (
    _INVALID_LANDMARKS,
    _candidate_to_primitive_action,
    _direction_consistency,
    _direction_tokens_from_text,
    _find_candidate,
    _format_optional_float,
    _format_reacquire_prompt_context,
    _format_xy,
    _has_non_forbidden_valid_candidate,
    _is_fresh_grounded_observation,
    _is_valid_landmark,
    _local_goal_to_primitive_action,
    _local_waypoint_to_world_xy,
    _memory_anchor_local_xy,
    _nearest_candidate_for_local_goal,
    _ordered_landmarks_from_steps,
    _pass_action_sequence,
    _pose_distance_m,
    _primitive_speed_scale,
    _relative_side,
    _relative_xy_from_anchor,
    _route_steps,
    _track_target_command,
    _world_waypoint_to_local_xy,
    _wrap_angle_rad,
)


@dataclass
class ExecutionFeedback:
    applied_action: str
    wheel_velocity: float
    steering_angle: float


@dataclass
class OpenNavPolicyConfig:
    execution_steps: int
    stop_distance_m: float
    heading_threshold_rad: float
    turn_speed_scale: float
    forward_speed_scale: float
    replanning_cooldown_sec: float
    active_forward_duration_sec: float
    active_turn_duration_sec: float
    active_turn_angle_deg: float
    active_forward_distance_m: float
    active_distance_tolerance_m: float
    active_turn_tolerance_deg: float
    active_odom_completion_enabled: bool
    active_track_max_yaw_deg: float
    active_track_min_duration_sec: float
    use_discrete_action_primitives: bool
    arrival_distance_m: float
    success_distance_m: float
    recovery_enabled: bool
    recovery_trigger_stalled_steps: int
    recovery_execution_steps: int
    landmark_arrival_confirmations: int
    landmark_arrival_window_size: int
    reacquire_guard_enabled: bool
    reacquire_guard_recent_visible_sec: float
    turn_at_max_turns: int = 4
    turn_at_complete_max_distance_m: float = 3.0
    turn_at_complete_requires_confirmed_anchor: bool = True
    confirm_min_observations: int = 3
    confirm_confidence_threshold: float = 0.50


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


@dataclass
class ReacquireGuardContext:
    previous_semantic_visible: bool
    previous_grounded_visible: bool
    current_grounded_visible: bool
    target_recently_visible: bool
    last_action: str
    last_seen_bearing: str
    lost_after_visible: bool
    forbidden_action: str
    grounded_reason: str


@dataclass
class RouteConsistencyResult:
    consistency: str
    observed_side: str
    relative_forward_m: float
    relative_left_m: float
    expected_words: list[str]
    anchor_source: str


class OpenNavVLNPolicy:
    def __init__(
        self,
        *,
        instruction_context: InstructionContext,
        candidate_predictor: CandidateWaypointPredictor,
        occupancy_grid: Any | None = None,
        map_candidate_generator: MapCandidateGenerator | None = None,
        progress_manager: ProgressManager,
        reasoner: OpenNavReasoner,
        config: OpenNavPolicyConfig,
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
        self._active_start_yaw_rad: float | None = None
        self._active_start_pose = None
        self._last_decision_summary = "startup"
        self._last_candidate_set: CandidateSet | None = None
        self._next_replan_time = 0.0
        self._route_steps = _route_steps(instruction_context)
        self._ordered_landmarks = _ordered_landmarks_from_steps(self._route_steps, instruction_context)
        self._current_landmark_index = 0
        self._completed_landmarks: list[str] = []
        self._route_complete = False
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
        self._previous_semantic_visible = False
        self._previous_grounded_visible = False
        self._last_semantic_visible_time_sec = -float("inf")
        self._last_grounded_visible_time_sec = -float("inf")
        self._last_motion_action = "STOP"
        self._last_semantic_bearing_hint = "unknown"
        self._arrival_confirmation_window: deque[bool] = deque(
            maxlen=max(1, int(self._config.landmark_arrival_window_size))
        )
        self._sync_instruction_target()
        self._safety_context_text = ""
        self._last_applied_action = "STOP"
        self._safety_override_streak = 0
        self._last_safety_override_signature: tuple[str, str, str] | None = None
        self._recovery_action_name: str | None = None
        self._recovery_steps_remaining = 0
        self._recovery_started_sec: float | None = None
        self._recovery_count = 0
        self._last_semantic_plan_signature: tuple | None = None
        self._last_semantic_skip_signature: tuple | None = None
        self._topology_summary = ""
        self._topology_revisit_world_xy: list[float] | None = None
        self._turn_at_landmark_index: int | None = None
        self._turn_at_steps_taken = 0
        self._turn_at_exhausted_landmark_index: int | None = None
        self._pass_landmark_index: int | None = None
        self._pass_actions: list[str] = []
        self._pass_action_index = 0
        self._recent_primitive_actions: deque[str] = deque(maxlen=4)
        self._last_turn_oscillation_signature: tuple | None = None

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
            if self._last_applied_action in {"FORWARD", "TURN_LEFT", "TURN_RIGHT"}:
                self._last_motion_action = self._last_applied_action

        progress_state = self._progress_manager.state
        if self._maybe_advance_landmark(observation=observation):
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
                reasoning_summary="all instruction landmarks reached",
                progress_signal=progress_state.progress_signal,
                replanned=False,
                debug_image_path=(
                    None if self._candidate_predictor.last_saved_debug_path is None
                    else str(self._candidate_predictor.last_saved_debug_path)
                ),
                **self._route_result_kwargs(),
            )

        turn_at_result = self._turn_at_step(
            observation=observation,
            progress_signal=progress_state.progress_signal,
            now=now,
        )
        if turn_at_result is not None:
            return turn_at_result

        pass_result = self._pass_step(
            observation=observation,
            progress_signal=progress_state.progress_signal,
            now=now,
        )
        if pass_result is not None:
            return pass_result

        if self._recovery_action_name is not None:
            return self._recovery_step(observation=observation, progress_signal=progress_state.progress_signal, now=now)

        if self._should_start_safety_recovery():
            return self._start_recovery(
                observation=observation,
                progress_signal=f"SAFETY_BLOCKED:{progress_state.progress_signal}",
                now=now,
                reason="repeated_safety_override",
            )

        if self._should_start_recovery(progress_state=progress_state):
            return self._start_recovery(
                observation=observation,
                progress_signal=progress_state.progress_signal,
                now=now,
            )

        if self._active_world_goal_xy is not None and self._execution_steps_remaining > 0:
            local_goal = _world_waypoint_to_local_xy(
                robot_pose=observation.pose,
                world_goal_xy=self._active_world_goal_xy,
            )
            if bool(self._config.use_discrete_action_primitives) and self._active_action_name in {
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

            action_name, speed_scale = _track_target_command(
                local_goal_xy=local_goal,
                stop_distance_m=float(self._config.stop_distance_m),
            )
            if action_name == "STOP":
                self._clear_active_goal()
            else:
                self._update_active_action_timer(action_name=action_name, now=now)
                active_elapsed_sec = self._active_elapsed_sec(now=now)
                max_active_duration_sec = self._active_action_duration_sec(action_name)
                remaining_after_step = max(0, int(self._execution_steps_remaining) - 1)
                min_track_duration_sec = 0.0
                if action_name == "TRACK_TARGET":
                    min_track_duration_sec = max(0.0, float(self._config.active_track_min_duration_sec))
                if remaining_after_step <= 0 and active_elapsed_sec < min_track_duration_sec:
                    remaining_after_step = 1
                self._execution_steps_remaining = remaining_after_step
                yaw_delta_rad = abs(_wrap_angle_rad(float(observation.pose.yaw_rad) - float(self._active_start_yaw_rad or observation.pose.yaw_rad)))
                max_track_yaw_rad = float(np.deg2rad(max(1.0, float(self._config.active_track_max_yaw_deg))))
                replan_reason = ""
                if int(self._execution_steps_remaining) <= 0:
                    replan_reason = "active step budget completed"
                elif active_elapsed_sec >= max_active_duration_sec:
                    replan_reason = "active duration completed"
                elif action_name == "TRACK_TARGET" and yaw_delta_rad >= max_track_yaw_rad:
                    replan_reason = "active yaw budget completed"

                if replan_reason:
                    completed_candidate_id = self._active_candidate_id
                    completed_world_goal_xy = (
                        list(self._active_world_goal_xy) if self._active_world_goal_xy is not None else None
                    )
                    self._logger.info(
                        "[ActiveGoal] replan action=%s reason=%s elapsed_sec=%.2f max_sec=%.2f yaw_delta_deg=%.1f max_yaw_deg=%.1f",
                        action_name,
                        replan_reason,
                        float(active_elapsed_sec),
                        float(max_active_duration_sec),
                        float(np.degrees(yaw_delta_rad)),
                        float(np.degrees(max_track_yaw_rad)),
                    )
                    self._clear_active_goal()
                    return PolicyStepResult(
                        action_name="STOP",
                        speed_scale=0.0,
                        decision_source="ACTIVE_GOAL_COMPLETE",
                        selected_candidate_id=completed_candidate_id,
                        local_goal_xy=[float(local_goal[0]), float(local_goal[1])],
                        world_goal_xy=completed_world_goal_xy,
                        execution_steps_remaining=0,
                        reasoning_summary=f"{replan_reason}; requesting a fresh visual decision",
                        progress_signal=progress_state.progress_signal,
                        replanned=False,
                        debug_image_path=(
                            None if self._candidate_predictor.last_saved_debug_path is None
                            else str(self._candidate_predictor.last_saved_debug_path)
                        ),
                        **self._route_result_kwargs(),
                    )
                return PolicyStepResult(
                    action_name=action_name,
                    speed_scale=float(speed_scale),
                    decision_source="ACTIVE_GOAL",
                    selected_candidate_id=self._active_candidate_id,
                    local_goal_xy=[float(local_goal[0]), float(local_goal[1])],
                    world_goal_xy=list(self._active_world_goal_xy) if self._active_world_goal_xy is not None else None,
                    execution_steps_remaining=int(self._execution_steps_remaining),
                    reasoning_summary=self._last_decision_summary,
                    progress_signal=progress_state.progress_signal,
                    replanned=False,
                    debug_image_path=(
                        None if self._candidate_predictor.last_saved_debug_path is None
                        else str(self._candidate_predictor.last_saved_debug_path)
                    ),
                    **self._route_result_kwargs(),
                )

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

        reacquire_guard = self._build_reacquire_guard_context(now=now)
        candidate_set = self._candidate_predictor.predict(observation)
        semantic_landmark_entry = self._semantic_landmark_entry()
        if self._map_candidate_generator is not None and self._occupancy_grid is not None:
            candidate_set = self._map_candidate_generator.augment(
                observation=observation,
                base_candidate_set=candidate_set,
                occupancy_snapshot=self._occupancy_grid.snapshot(),
                current_target=self._current_target_text(),
                landmark_state=self._landmark_tracker_state,
                landmark_entry=semantic_landmark_entry,
                progress_signal=progress_state.progress_signal,
                safety_context_text=self._safety_context_text,
            )
        if self._occupancy_grid is not None:
            candidate_set = self._occupancy_grid.filter_candidates(
                observation=observation,
                candidate_set=candidate_set,
            )
        self._last_candidate_set = candidate_set
        semantic_plan = None
        if self._occupancy_grid is not None and semantic_landmark_entry is not None and not bool(semantic_landmark_entry.reached):
            try:
                semantic_plan = self._occupancy_grid.plan_toward_landmark(
                    observation=observation,
                    landmark_entry=semantic_landmark_entry,
                    stop_distance_m=float(self._current_arrival_distance_m()),
                )
            except Exception as exc:
                self._logger.warning("[SemanticPlanner] plan_failed landmark=%s error=%s", self._current_target_text(), exc)
                semantic_plan = None
        if self._occupancy_grid is not None:
            self._occupancy_grid.save_debug(
                observation=observation,
                candidate_set=candidate_set,
                semantic_entries=list(self._all_landmark_map_entries),
                semantic_plan=semantic_plan,
            )
        debug_image_path = self._candidate_predictor.save_debug(
            observation=observation,
            candidate_set=candidate_set,
            target_observation=(
                None
                if self._landmark_tracker_state is None
                else self._landmark_tracker_state.last_observation
            ),
            # The VLM request image stays first-person only. Map state is passed
            # as structured text; occupancy rendering is for human debug.
            left_panel_image_path=None,
        )
        annotated_png_bytes = None
        if debug_image_path is not None and Path(debug_image_path).exists():
            annotated_png_bytes = Path(debug_image_path).read_bytes()

        semantic_plan_result = self._semantic_planner_step(
            observation=observation,
            candidate_set=candidate_set,
            progress_signal=progress_state.progress_signal,
            debug_image_path=debug_image_path,
            semantic_plan=semantic_plan,
            now=now,
        )
        if semantic_plan_result is not None:
            return semantic_plan_result

        if progress_state.progress_signal in {"STALLED", "SPINNING"}:
            topology_revisit_result = self._topology_revisit_step(
                observation=observation,
                candidate_set=candidate_set,
                progress_signal=progress_state.progress_signal,
                debug_image_path=debug_image_path,
                now=now,
                reason="stalled_or_spinning",
            )
            if topology_revisit_result is not None:
                return topology_revisit_result

        safety_context_for_decision = self._safety_context_text
        decision = self._reasoner.choose_candidate(
            instruction_context=self._instruction_context,
            candidate_set=candidate_set,
            progress_state=progress_state,
            annotated_image_png_bytes=annotated_png_bytes,
            annotated_image_path=None if debug_image_path is None else str(debug_image_path),
            target_memory_text=self._combined_target_memory_text(robot_pose=observation.pose),
            safety_context_text=safety_context_for_decision,
            route_context_text=self._route_context_text(robot_pose=observation.pose),
            reacquire_context_text=_format_reacquire_prompt_context(reacquire_guard),
            obstacle_summary_text=self._obstacle_summary_text,
            semantic_map_summary=self._semantic_map_summary,
        )
        self._safety_context_text = ""
        selected_candidate = _find_candidate(candidate_set, decision.selected_candidate_id)
        if selected_candidate is None or not selected_candidate.valid:
            topology_revisit_result = self._topology_revisit_step(
                observation=observation,
                candidate_set=candidate_set,
                progress_signal=progress_state.progress_signal,
                debug_image_path=debug_image_path,
                now=now,
                reason=f"llm_blocked:{decision.reason}",
            )
            if topology_revisit_result is not None:
                return topology_revisit_result
            self._update_semantic_visibility_state(decision=decision, now=now)
            self._log_reacquire_guard(
                guard=reacquire_guard,
                current_semantic_visible=bool(decision.target_visible),
                llm_selected_action=decision.selected_action or "NONE",
                final_selected_action="STOP",
                action_overridden=False,
                override_reason=f"blocked:{decision.reason}",
            )
            self._last_decision_summary = decision.thought
            if (
                int(candidate_set.valid_count) == 0
                and bool(self._config.recovery_enabled)
            ):
                self._logger.warning(
                    "[Recovery] no_valid_candidates signal=%s; starting local escape instead of repeating STOP",
                    progress_state.progress_signal,
                )
                self._safety_context_text = (
                    f"NO_VALID_CANDIDATES; progress_signal={progress_state.progress_signal}; "
                    "runtime will execute local recovery before the next VLM decision."
                )
                return self._start_recovery(
                    observation=observation,
                    progress_signal=progress_state.progress_signal,
                    now=now,
                )
            self._next_replan_time = now + float(self._config.replanning_cooldown_sec)
            self._clear_active_goal()
            return PolicyStepResult(
                action_name="STOP",
                speed_scale=0.0,
                decision_source=f"LLM_BLOCKED({decision.reason})",
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

        decision_summary = str(decision.thought).strip() or "no_reason"
        route_corrected_action_name: str | None = None
        route_consistency = self._evaluate_route_consistency(target_visible_hint=decision.target_visible)
        if route_consistency is not None and route_consistency.consistency == "contradiction":
            correction_candidate = self._select_route_correction_candidate(
                candidate_set=candidate_set,
                route_consistency=route_consistency,
            )
            if correction_candidate is not None and int(correction_candidate.slot_id) != int(selected_candidate.slot_id):
                self._logger.warning(
                    "[RouteCorrection] target=%s expected=%s observed=%s overriding_candidate=%s original_candidate=%s anchor_source=%s",
                    self._current_target_text(),
                    route_consistency.expected_words,
                    route_consistency.observed_side,
                    int(correction_candidate.slot_id),
                    int(selected_candidate.slot_id),
                    route_consistency.anchor_source,
                )
                selected_candidate = correction_candidate
                route_corrected_action_name = _candidate_to_primitive_action(
                    correction_candidate,
                    turn_speed_scale=float(self._config.turn_speed_scale),
                    forward_speed_scale=float(self._config.forward_speed_scale),
                )[0]
                decision_summary = (
                    f"{decision.thought} | route_contradiction_detected; "
                    f"runtime overrode candidate toward expected relation {route_consistency.expected_words}"
                )
            else:
                topology_revisit_result = self._topology_revisit_step(
                    observation=observation,
                    candidate_set=candidate_set,
                    progress_signal=progress_state.progress_signal,
                    debug_image_path=debug_image_path,
                    now=now,
                    reason="route_contradiction",
                )
                if topology_revisit_result is not None:
                    return topology_revisit_result
                self._logger.warning(
                    "[RouteCorrection] target=%s expected=%s observed=%s action=STOP_REPLAN anchor_source=%s",
                    self._current_target_text(),
                    route_consistency.expected_words,
                    route_consistency.observed_side,
                    route_consistency.anchor_source,
                )
                self._next_replan_time = now + float(self._config.replanning_cooldown_sec)
                self._clear_active_goal()
                return PolicyStepResult(
                    action_name="STOP",
                    speed_scale=0.0,
                    decision_source="ROUTE_CONTRADICTION_REPLAN",
                    selected_candidate_id=None,
                    local_goal_xy=[0.0, 0.0],
                    world_goal_xy=None,
                    execution_steps_remaining=0,
                    reasoning_summary=(
                        f"route contradiction detected for current landmark {self._current_target_text()}; "
                        "stopping to request a fresh decision"
                    ),
                    progress_signal=progress_state.progress_signal,
                    replanned=True,
                    debug_image_path=None if debug_image_path is None else str(debug_image_path),
                    **self._route_result_kwargs(),
                )

        world_goal_xy = _local_waypoint_to_world_xy(
            robot_pose=observation.pose,
            waypoint_local_xy=selected_candidate.local_goal_xy,
        )
        world_goal_xy_list = [float(world_goal_xy[0]), float(world_goal_xy[1])]
        memory_anchor_world_xy = _local_waypoint_to_world_xy(
            robot_pose=observation.pose,
            waypoint_local_xy=_memory_anchor_local_xy(
                candidate=selected_candidate,
                bearing_hint=decision.target_bearing_hint,
            ),
        )
        self._target_memory.update_from_decision(
            target_visible=decision.target_visible,
            bearing_hint=decision.target_bearing_hint,
            status=decision.target_status,
            selected_world_goal_xy=[float(memory_anchor_world_xy[0]), float(memory_anchor_world_xy[1])],
        )
        self._log_route_consistency_if_needed(
            current_anchor_world_xy=[float(memory_anchor_world_xy[0]), float(memory_anchor_world_xy[1])],
            target_visible=decision.target_visible,
        )
        self._active_world_goal_xy = [float(world_goal_xy[0]), float(world_goal_xy[1])]
        self._active_candidate_id = int(selected_candidate.slot_id)
        self._active_candidate_label = str(selected_candidate.label)
        self._execution_steps_remaining = max(1, int(self._config.execution_steps))
        self._last_decision_summary = decision_summary
        local_goal = tuple(float(value) for value in selected_candidate.local_goal_xy)
        if bool(self._config.use_discrete_action_primitives):
            action_name, speed_scale = _candidate_to_primitive_action(
                selected_candidate,
                turn_speed_scale=float(self._config.turn_speed_scale),
                forward_speed_scale=float(self._config.forward_speed_scale),
            )
            if route_corrected_action_name in {"TURN_LEFT", "FORWARD", "TURN_RIGHT"}:
                action_name = str(route_corrected_action_name)
                speed_scale = float(_primitive_speed_scale(action_name, self._config))
            elif decision.selected_action in {"TURN_LEFT", "FORWARD", "TURN_RIGHT"}:
                action_name = str(decision.selected_action)
                speed_scale = float(_primitive_speed_scale(action_name, self._config))
            replacement = self._select_candidate_after_turn_guard(
                candidate_set=candidate_set,
                selected_candidate=selected_candidate,
                action_name=action_name,
                decision_reason=decision_summary,
                safety_context_text=safety_context_for_decision,
            )
            if replacement is not None and int(replacement.slot_id) != int(selected_candidate.slot_id):
                selected_candidate = replacement
                local_goal = tuple(float(value) for value in selected_candidate.local_goal_xy)
                world_goal_xy = _local_waypoint_to_world_xy(
                    robot_pose=observation.pose,
                    waypoint_local_xy=selected_candidate.local_goal_xy,
                )
                world_goal_xy_list = [float(world_goal_xy[0]), float(world_goal_xy[1])]
                self._active_world_goal_xy = [float(world_goal_xy[0]), float(world_goal_xy[1])]
                self._active_candidate_id = int(selected_candidate.slot_id)
                self._active_candidate_label = str(selected_candidate.label)
                memory_anchor_world_xy = _local_waypoint_to_world_xy(
                    robot_pose=observation.pose,
                    waypoint_local_xy=_memory_anchor_local_xy(
                        candidate=selected_candidate,
                        bearing_hint=decision.target_bearing_hint,
                    ),
                )
                self._target_memory.update_from_decision(
                    target_visible=decision.target_visible,
                    bearing_hint=decision.target_bearing_hint,
                    status=decision.target_status,
                    selected_world_goal_xy=[float(memory_anchor_world_xy[0]), float(memory_anchor_world_xy[1])],
                )
                action_name, speed_scale = _candidate_to_primitive_action(
                    selected_candidate,
                    turn_speed_scale=float(self._config.turn_speed_scale),
                    forward_speed_scale=float(self._config.forward_speed_scale),
                )
                decision_summary = f"{decision_summary} | TurnOscillationGuard switched to candidate {int(selected_candidate.slot_id)}"
            reacquire_replacement = self._select_reacquire_replacement_candidate(
                candidate_set=candidate_set,
                guard=reacquire_guard,
                selected_candidate=selected_candidate,
                action_name=action_name,
            )
            if reacquire_replacement is not None and int(reacquire_replacement.slot_id) != int(selected_candidate.slot_id):
                previous_action = str(action_name)
                selected_candidate = reacquire_replacement
                local_goal = tuple(float(value) for value in selected_candidate.local_goal_xy)
                world_goal_xy = _local_waypoint_to_world_xy(
                    robot_pose=observation.pose,
                    waypoint_local_xy=selected_candidate.local_goal_xy,
                )
                world_goal_xy_list = [float(world_goal_xy[0]), float(world_goal_xy[1])]
                self._active_world_goal_xy = [float(world_goal_xy[0]), float(world_goal_xy[1])]
                self._active_candidate_id = int(selected_candidate.slot_id)
                self._active_candidate_label = str(selected_candidate.label)
                memory_anchor_world_xy = _local_waypoint_to_world_xy(
                    robot_pose=observation.pose,
                    waypoint_local_xy=_memory_anchor_local_xy(
                        candidate=selected_candidate,
                        bearing_hint=decision.target_bearing_hint,
                    ),
                )
                self._target_memory.update_from_decision(
                    target_visible=decision.target_visible,
                    bearing_hint=decision.target_bearing_hint,
                    status=decision.target_status,
                    selected_world_goal_xy=[float(memory_anchor_world_xy[0]), float(memory_anchor_world_xy[1])],
                )
                action_name, speed_scale = _candidate_to_primitive_action(
                    selected_candidate,
                    turn_speed_scale=float(self._config.turn_speed_scale),
                    forward_speed_scale=float(self._config.forward_speed_scale),
                )
                decision_summary = (
                    f"{decision_summary} | ReacquireGuard switched repeated {previous_action} "
                    f"to candidate {int(selected_candidate.slot_id)} action={action_name}"
                )
        else:
            action_name, speed_scale = _track_target_command(
                local_goal_xy=local_goal,
                stop_distance_m=float(self._config.stop_distance_m),
            )
        final_action_name, guard_overridden, guard_override_reason = self._apply_reacquire_guard(
            guard=reacquire_guard,
            llm_selected_action=action_name,
            current_semantic_visible=bool(decision.target_visible),
            has_non_forbidden_valid_candidate=_has_non_forbidden_valid_candidate(
                candidate_set=candidate_set,
                forbidden_action=reacquire_guard.forbidden_action,
                config=self._config,
            ),
        )
        self._update_semantic_visibility_state(decision=decision, now=now)
        self._log_reacquire_guard(
            guard=reacquire_guard,
            current_semantic_visible=bool(decision.target_visible),
            llm_selected_action=action_name,
            final_selected_action=final_action_name,
            action_overridden=guard_overridden,
            override_reason=guard_override_reason,
        )
        if guard_overridden:
            self._last_decision_summary = (
                f"{decision.thought} | ReacquireGuard override: {guard_override_reason}"
            )
            self._next_replan_time = now + float(self._config.replanning_cooldown_sec)
            self._clear_active_goal()
            return PolicyStepResult(
                action_name=final_action_name,
                speed_scale=0.0,
                decision_source="REACQUIRE_GUARD",
                selected_candidate_id=int(selected_candidate.slot_id),
                local_goal_xy=[0.0, 0.0],
                world_goal_xy=None,
                execution_steps_remaining=0,
                reasoning_summary=self._last_decision_summary,
                progress_signal=progress_state.progress_signal,
                replanned=True,
                debug_image_path=None if debug_image_path is None else str(debug_image_path),
                **self._route_result_kwargs(),
            )
        if action_name != "STOP":
            self._execution_steps_remaining = max(1, int(self._execution_steps_remaining))
            self._active_action_name = action_name
            self._active_goal_started_sec = now
            self._active_start_yaw_rad = float(observation.pose.yaw_rad)
            self._active_start_pose = observation.pose
            if action_name in {"FORWARD", "TURN_LEFT", "TURN_RIGHT"}:
                self._recent_primitive_actions.append(str(action_name))
        self._logger.info(
            "[WaypointDecision] candidate=%s label=%s heading_deg=%.1f semantic_visible=%s grounded_visible=%s target_bearing=%s action=%s",
            int(selected_candidate.slot_id),
            str(selected_candidate.label),
            float(selected_candidate.heading_delta_deg),
            decision.target_visible,
            int(bool(reacquire_guard.current_grounded_visible)),
            decision.target_bearing_hint,
            action_name,
        )
        return PolicyStepResult(
            action_name=action_name,
            speed_scale=float(speed_scale),
            decision_source="LLM_PRIMITIVE" if bool(self._config.use_discrete_action_primitives) else "LLM",
            selected_candidate_id=int(selected_candidate.slot_id),
            local_goal_xy=[float(local_goal[0]), float(local_goal[1])],
            world_goal_xy=world_goal_xy_list,
            execution_steps_remaining=int(self._execution_steps_remaining),
            reasoning_summary=decision.thought,
            progress_signal=progress_state.progress_signal,
            replanned=True,
            debug_image_path=None if debug_image_path is None else str(debug_image_path),
            **self._route_result_kwargs(),
        )

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

        if bool(self._config.use_discrete_action_primitives):
            action_name = _local_goal_to_primitive_action(local_goal_xy=local_goal)
            if action_name == "STOP":
                return None
            min_executable_forward_m = max(0.10, 0.75 * float(self._config.active_forward_distance_m))
            if (
                local_goal_distance_m < min_executable_forward_m
                or path_length_m < min_executable_forward_m
            ):
                self._log_semantic_skip(
                    target_label=str(getattr(semantic_plan, "target_label", self._current_target_text())),
                    reason="subgoal_too_short_for_primitive",
                    action_name=action_name,
                    distance_to_target_m=float(getattr(semantic_plan, "distance_to_target_m", 0.0)),
                    path_length_m=path_length_m,
                    local_goal=local_goal,
                    min_executable_m=min_executable_forward_m,
                )
                return None
            speed_scale = float(_primitive_speed_scale(action_name, self._config))
        else:
            action_name, speed_scale = _track_target_command(
                local_goal_xy=local_goal,
                stop_distance_m=float(self._config.stop_distance_m),
            )
            if action_name == "STOP":
                return None
            min_executable_forward_m = 0.0

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
        self._execution_steps_remaining = max(1, int(self._config.execution_steps))
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

    def _select_candidate_after_turn_guard(
        self,
        *,
        candidate_set: CandidateSet,
        selected_candidate: WaypointCandidate,
        action_name: str,
        decision_reason: str,
        safety_context_text: str,
    ) -> WaypointCandidate | None:
        action = str(action_name).strip().upper()
        if action not in {"TURN_LEFT", "TURN_RIGHT"}:
            return None
        opposite = "TURN_RIGHT" if action == "TURN_LEFT" else "TURN_LEFT"
        recent = [str(item).strip().upper() for item in self._recent_primitive_actions]
        turn_recent = [item for item in recent if item in {"TURN_LEFT", "TURN_RIGHT"}]
        alternating_turns = len(turn_recent) >= 2 and turn_recent[-1] == opposite and turn_recent[-2] == action
        immediate_reverse = len(turn_recent) >= 1 and turn_recent[-1] == opposite
        safety_blocked = bool(safety_context_text and f"previous_desired={action}" in safety_context_text)
        if not safety_blocked and not (alternating_turns and immediate_reverse):
            return None

        alternatives: list[tuple[float, WaypointCandidate, str]] = []
        for candidate in candidate_set.candidates:
            if not bool(candidate.valid):
                continue
            candidate_action, _ = _candidate_to_primitive_action(
                candidate,
                turn_speed_scale=float(self._config.turn_speed_scale),
                forward_speed_scale=float(self._config.forward_speed_scale),
            )
            candidate_action = str(candidate_action).strip().upper()
            if candidate_action == action:
                continue
            if candidate_action == opposite and alternating_turns:
                continue
            action_cost = 0.0 if candidate_action == "FORWARD" else 1.0
            score = (
                action_cost,
                -float(candidate.clearance_score),
                abs(float(candidate.heading_delta_deg)),
                abs(float(candidate.travel_distance_m) - float(self._config.active_forward_distance_m)),
            )
            alternatives.append((float(score[0] * 1000.0 + score[1] + score[2] * 0.01 + score[3] * 0.1), candidate, candidate_action))
        if not alternatives:
            return None
        _, replacement, replacement_action = min(alternatives, key=lambda item: item[0])
        signature = (
            action,
            opposite,
            int(selected_candidate.slot_id),
            int(replacement.slot_id),
            safety_blocked,
            tuple(turn_recent[-3:]),
        )
        if signature != self._last_turn_oscillation_signature:
            self._logger.warning(
                "[TurnOscillationGuard] action=%s replacement=%s selected_candidate=%s replacement_candidate=%s safety_blocked=%s recent_turns=%s reason=%s",
                action,
                replacement_action,
                int(selected_candidate.slot_id),
                int(replacement.slot_id),
                int(bool(safety_blocked)),
                turn_recent[-4:],
                str(decision_reason)[:120],
            )
            self._last_turn_oscillation_signature = signature
        return replacement

    def _clear_active_goal(self) -> None:
        self._active_world_goal_xy = None
        self._active_candidate_id = None
        self._active_candidate_label = "none"
        self._execution_steps_remaining = 0
        self._active_goal_started_sec = None
        self._active_action_name = "STOP"
        self._active_start_yaw_rad = None
        self._active_start_pose = None

    def cancel_active_goal(self) -> None:
        self._clear_active_goal()
        self._clear_recovery()

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
                "[SafetyRecovery] repeated_override streak=%s desired=%s applied=%s reason=%s",
                int(self._safety_override_streak),
                desired,
                applied,
                reason_text,
            )

    @property
    def current_landmark(self) -> str:
        if self._route_complete:
            return ""
        return self._current_target_text()

    @property
    def current_behavior_type(self) -> str:
        route_step = self._current_route_step()
        if route_step is None:
            return "go_to"
        behavior = str(getattr(route_step, "behavior_type", "go_to")).strip().lower()
        return behavior or "go_to"

    @property
    def current_arrival_distance_m(self) -> float:
        return self._current_arrival_distance_m()

    @property
    def landmark_anchors(self) -> dict[str, list[float]]:
        return dict(self._landmark_anchors)

    @property
    def current_direction_clause(self) -> str:
        return self._current_direction_clause()

    def set_landmark_state(self, state: LandmarkState | None) -> None:
        self._landmark_tracker_state = state
        if state is None or state.last_observation is None:
            return
        observation = state.last_observation
        if _is_fresh_grounded_observation(observation):
            self._last_grounded_visible_time_sec = float(observation.timestamp_sec)

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
        self._action_history.append((now, str(action).strip(), str(result).strip()))

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

    def set_topology_revisit_world_xy(self, value: list[float] | None) -> None:
        if value is None:
            self._topology_revisit_world_xy = None
            return
        self._topology_revisit_world_xy = [float(value[0]), float(value[1])]

    def _turn_at_step(
        self,
        *,
        observation: VLNObservation,
        progress_signal: str,
        now: float,
    ) -> PolicyStepResult | None:
        if self.current_behavior_type != "turn_at":
            return None

        if self._turn_at_landmark_index != self._current_landmark_index:
            self._turn_at_landmark_index = int(self._current_landmark_index)
            self._turn_at_steps_taken = 0
            self._turn_at_exhausted_landmark_index = None
            self._logger.info(
                "[TurnAt] start target=%s directive=%s max_turns=%s",
                self._current_target_text(),
                self._current_direction_clause() or "none",
                int(self._config.turn_at_max_turns),
            )

        if (
            self._active_candidate_label == "turn_at"
            and self._active_world_goal_xy is not None
            and self._execution_steps_remaining > 0
        ):
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

        tracker_state = self._landmark_tracker_state
        tracker_observation = None if tracker_state is None else tracker_state.last_observation

        # Check if target is confirmed for turn_at completion
        if bool(self._config.turn_at_complete_requires_confirmed_anchor):
            confirmed_turn_at = (
                tracker_state is not None
                and tracker_state.estimated_world_xy is not None
                and str(tracker_state.evidence_level) == "confirmed"
                and bool(tracker_state.can_complete)
                and int(getattr(tracker_state, "confirmed_observation_count", 0)) >= max(1, int(self._config.confirm_min_observations))
                and float(getattr(tracker_state, "confirmed_confidence", 0.0)) >= float(self._config.confirm_confidence_threshold)
            )
        else:
            confirmed_turn_at = (
                tracker_state is not None
                and tracker_state.estimated_world_xy is not None
            )

        # Distance limit
        if confirmed_turn_at and tracker_state is not None:
            max_dist = max(0.5, float(self._config.turn_at_complete_max_distance_m))
            if tracker_state.anchor_distance_m is not None and float(tracker_state.anchor_distance_m) > max_dist:
                self._logger.info(
                    "[TurnAt] not_complete target=%s reason=too_far distance=%.2f max=%.2f evidence=%s",
                    self._current_target_text(),
                    float(tracker_state.anchor_distance_m),
                    max_dist,
                    str(getattr(tracker_state, "evidence_level", "unknown")),
                )
                confirmed_turn_at = False

        _, grounded_reason = self._current_grounded_visibility()

        # Log potential target (visible but not confirmed)
        if (
            tracker_state is not None
            and tracker_observation is not None
            and not confirmed_turn_at
        ):
            not_confirmed_reasons = []
            if str(getattr(tracker_state, "evidence_level", "none")) != "confirmed":
                not_confirmed_reasons.append(f"evidence={tracker_state.evidence_level}")
            if not bool(getattr(tracker_state, "can_complete", False)):
                not_confirmed_reasons.append("can_complete=false")
            obs_count = int(getattr(tracker_state, "confirmed_observation_count", 0))
            if obs_count < max(1, int(self._config.confirm_min_observations)):
                not_confirmed_reasons.append(f"obs={obs_count}")
            conf = float(getattr(tracker_state, "confirmed_confidence", 0.0))
            if conf < float(self._config.confirm_confidence_threshold):
                not_confirmed_reasons.append(f"conf={conf:.2f}")
            self._logger.info(
                "[TurnAt] not_complete target=%s reason=%s confidence=%.2f source=%s anchor=%s action=continue_search",
                self._current_target_text(),
                ",".join(not_confirmed_reasons) if not_confirmed_reasons else "unknown",
                float(tracker_state.confidence) if tracker_state else 0.0,
                str(getattr(tracker_observation, "semantic_source", "unknown")),
                _format_optional_float(tracker_state.anchor_distance_m) if tracker_state else "none",
            )

        if confirmed_turn_at:
            anchor_world_xy, anchor_source = self._current_route_anchor()
            self._logger.info(
                "[TurnAt] complete target=%s reason=confirmed_anchor confidence=%.2f obs=%s distance=%s turns=%s",
                self._current_target_text(),
                float(tracker_state.confidence) if tracker_state else 0.0,
                int(tracker_state.confirmed_observation_count) if tracker_state else 0,
                _format_optional_float(tracker_state.anchor_distance_m) if tracker_state else "none",
                int(self._turn_at_steps_taken),
            )
            self._advance_current_landmark(
                observation=observation,
                anchor_world_xy=anchor_world_xy,
                distance_m=tracker_state.anchor_distance_m if tracker_state else None,
                source="TurnAt",
                reason="confirmed_anchor",
                visible_now=True,
                lost_count=0,
            )
            self._clear_active_goal()
            return PolicyStepResult(
                action_name="STOP",
                speed_scale=0.0,
                decision_source="TURN_AT_COMPLETE",
                selected_candidate_id=None,
                local_goal_xy=[0.0, 0.0],
                world_goal_xy=anchor_world_xy,
                execution_steps_remaining=0,
                reasoning_summary="turn_at completed because target anchor confirmed",
                progress_signal=progress_signal,
                replanned=False,
                debug_image_path=(
                    None if self._candidate_predictor.last_saved_debug_path is None
                    else str(self._candidate_predictor.last_saved_debug_path)
                ),
                **self._route_result_kwargs(),
            )

        if self._turn_at_exhausted_landmark_index == self._current_landmark_index:
            return None

        max_turns = max(1, int(self._config.turn_at_max_turns))
        if int(self._turn_at_steps_taken) >= max_turns:
            self._logger.warning(
                "[TurnAt] exhausted target=%s reason=max_turns_reached turns=%s grounded_reason=%s next=LLM_SEARCH",
                self._current_target_text(),
                int(self._turn_at_steps_taken),
                grounded_reason,
            )
            self._clear_active_goal()
            self._turn_at_exhausted_landmark_index = int(self._current_landmark_index)
            return None

        action_name = self._turn_at_direction_action()
        self._turn_at_steps_taken += 1
        self._active_world_goal_xy = [float(observation.pose.x), float(observation.pose.y)]
        self._active_candidate_id = None
        self._active_candidate_label = "turn_at"
        self._execution_steps_remaining = max(1, int(self._config.execution_steps))
        self._active_action_name = str(action_name)
        self._active_goal_started_sec = float(now)
        self._active_start_yaw_rad = float(observation.pose.yaw_rad)
        self._active_start_pose = observation.pose
        self._last_decision_summary = (
            f"turn_at primitive {self._turn_at_steps_taken}/{max_turns}; "
            f"directive={self._current_direction_clause() or 'none'}; "
            f"target_visible=0 reason={grounded_reason}"
        )
        self._logger.info(
            "[TurnAt] action=%s target=%s turn=%s/%s directive=%s grounded_reason=%s",
            action_name,
            self._current_target_text(),
            int(self._turn_at_steps_taken),
            int(max_turns),
            self._current_direction_clause() or "none",
            grounded_reason,
        )
        return PolicyStepResult(
            action_name=action_name,
            speed_scale=float(_primitive_speed_scale(action_name, self._config)),
            decision_source="TURN_AT",
            selected_candidate_id=None,
            local_goal_xy=[0.0, 0.0],
            world_goal_xy=list(self._active_world_goal_xy),
            execution_steps_remaining=int(self._execution_steps_remaining),
            reasoning_summary=self._last_decision_summary,
            progress_signal=progress_signal,
            replanned=True,
            debug_image_path=(
                None if self._candidate_predictor.last_saved_debug_path is None
                else str(self._candidate_predictor.last_saved_debug_path)
            ),
            **self._route_result_kwargs(),
        )

    def _turn_at_direction_action(self) -> str:
        text = f"{self._current_direction_clause()} {self._current_target_text()}".lower()
        if "left" in text:
            return "TURN_LEFT"
        if "right" in text:
            return "TURN_RIGHT"
        return self._preferred_search_turn_action()

    def _pass_step(
        self,
        *,
        observation: VLNObservation,
        progress_signal: str,
        now: float,
    ) -> PolicyStepResult | None:
        if self.current_behavior_type != "pass":
            return None

        if self._pass_landmark_index != self._current_landmark_index:
            self._pass_landmark_index = int(self._current_landmark_index)
            self._pass_actions = _pass_action_sequence(self._current_direction_clause())
            self._pass_action_index = 0
            self._logger.info(
                "[Pass] start target=%s directive=%s actions=%s",
                self._current_target_text(),
                self._current_direction_clause() or "none",
                self._pass_actions,
            )

        if (
            self._active_candidate_label == "pass"
            and self._active_world_goal_xy is not None
            and self._execution_steps_remaining > 0
        ):
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

        if self._pass_action_index >= len(self._pass_actions):
            anchor_world_xy, _ = self._current_route_anchor()
            self._logger.info(
                "[Pass] complete target=%s actions=%s next_route_step=%s",
                self._current_target_text(),
                self._pass_actions,
                int(self._current_landmark_index) + 1,
            )
            self._advance_current_landmark(
                observation=observation,
                anchor_world_xy=anchor_world_xy,
                distance_m=None,
                source="PassPrimitive",
                reason="pass_sequence_complete",
                visible_now=False,
                lost_count=0,
            )
            self._clear_active_goal()
            return PolicyStepResult(
                action_name="STOP",
                speed_scale=0.0,
                decision_source="PASS_COMPLETE",
                selected_candidate_id=None,
                local_goal_xy=[0.0, 0.0],
                world_goal_xy=anchor_world_xy,
                execution_steps_remaining=0,
                reasoning_summary="pass behavior completed; advancing to next route step",
                progress_signal=progress_signal,
                replanned=False,
                debug_image_path=(
                    None if self._candidate_predictor.last_saved_debug_path is None
                    else str(self._candidate_predictor.last_saved_debug_path)
                ),
                **self._route_result_kwargs(),
            )

        action_name = str(self._pass_actions[self._pass_action_index]).strip().upper()
        self._pass_action_index += 1
        self._active_world_goal_xy = [float(observation.pose.x), float(observation.pose.y)]
        self._active_candidate_id = None
        self._active_candidate_label = "pass"
        self._execution_steps_remaining = max(1, int(self._config.execution_steps))
        self._active_action_name = action_name
        self._active_goal_started_sec = float(now)
        self._active_start_yaw_rad = float(observation.pose.yaw_rad)
        self._active_start_pose = observation.pose
        self._last_decision_summary = (
            f"pass primitive {self._pass_action_index}/{len(self._pass_actions)}; "
            f"directive={self._current_direction_clause() or 'none'}"
        )
        self._logger.info(
            "[Pass] action=%s target=%s step=%s/%s directive=%s",
            action_name,
            self._current_target_text(),
            int(self._pass_action_index),
            int(len(self._pass_actions)),
            self._current_direction_clause() or "none",
        )
        return PolicyStepResult(
            action_name=action_name,
            speed_scale=float(_primitive_speed_scale(action_name, self._config)),
            decision_source="PASS_PRIMITIVE",
            selected_candidate_id=None,
            local_goal_xy=[0.0, 0.0],
            world_goal_xy=list(self._active_world_goal_xy),
            execution_steps_remaining=int(self._execution_steps_remaining),
            reasoning_summary=self._last_decision_summary,
            progress_signal=progress_signal,
            replanned=True,
            debug_image_path=(
                None if self._candidate_predictor.last_saved_debug_path is None
                else str(self._candidate_predictor.last_saved_debug_path)
            ),
            **self._route_result_kwargs(),
        )

    def _semantic_landmark_entry(self) -> LandmarkMapEntry | None:
        if self.current_behavior_type != "go_to":
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

    def _combined_target_memory_text(self, *, robot_pose) -> str:
        semantic_memory = self._target_memory.summary(robot_pose=robot_pose)
        current_grounding = self._current_grounding_summary()
        parts = [f"current_grounding={current_grounding}", f"semantic_target_memory={semantic_memory}"]
        if self._landmark_map_summary:
            parts.append(f"grounded_landmark_map={self._landmark_map_summary}")
        return "; ".join(parts)

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
        return (
            f"label={getattr(observation, 'label', self._current_target_text())} "
            f"visible={int(bool(getattr(observation, 'visible', False)))} "
            f"fresh={int(fresh)} reason={reason} source={source} "
            f"bbox={list(bbox) if bbox is not None else 'none'} "
            f"center_uv={_format_xy(center_uv) if center_uv is not None else 'none'} "
            f"depth_m={_format_optional_float(depth_m)} "
            f"bearing_deg={_format_optional_float(None if bearing_rad is None else np.degrees(float(bearing_rad)))} "
            f"world_xy={_format_xy(world_xy) if world_xy is not None else 'none'} "
            f"mask_area={float(getattr(observation, 'mask_area_ratio', 0.0)):.4f} "
            f"segmentation={getattr(observation, 'segmentation_source', 'none')} "
            f"evidence={getattr(observation, 'evidence_level', 'none')} "
            f"can_confirm={int(bool(getattr(observation, 'can_confirm', False)))} "
            f"vlm_verified={int(bool(getattr(observation, 'vlm_verified', False)))} "
            f"vlm_conf={float(getattr(observation, 'vlm_verification_confidence', 0.0)):.2f}"
        )

    def _build_reacquire_guard_context(self, *, now: float) -> ReacquireGuardContext:
        current_grounded_visible, grounded_reason = self._current_grounded_visibility()
        recent_window_sec = max(0.0, float(self._config.reacquire_guard_recent_visible_sec))
        semantic_recent = (float(now) - float(self._last_semantic_visible_time_sec)) <= recent_window_sec
        grounded_recent = (float(now) - float(self._last_grounded_visible_time_sec)) <= recent_window_sec
        target_recently_visible = bool(
            semantic_recent
            or grounded_recent
            or self._previous_semantic_visible
            or self._previous_grounded_visible
        )
        last_action = str(self._last_motion_action).strip().upper() or "STOP"
        last_seen_bearing = str(
            self._target_memory.state.last_seen_bearing_hint or self._last_semantic_bearing_hint or "unknown"
        ).strip().lower()
        lost_after_visible = bool(
            bool(self._config.reacquire_guard_enabled)
            and target_recently_visible
            and not current_grounded_visible
            and last_action in {"TURN_LEFT", "TURN_RIGHT"}
        )
        forbidden_action = last_action if lost_after_visible else "NONE"
        return ReacquireGuardContext(
            previous_semantic_visible=bool(self._previous_semantic_visible),
            previous_grounded_visible=bool(self._previous_grounded_visible),
            current_grounded_visible=bool(current_grounded_visible),
            target_recently_visible=bool(target_recently_visible),
            last_action=last_action,
            last_seen_bearing=last_seen_bearing or "unknown",
            lost_after_visible=bool(lost_after_visible),
            forbidden_action=forbidden_action,
            grounded_reason=grounded_reason,
        )

    def _current_grounded_visibility(self) -> tuple[bool, str]:
        state = self._landmark_tracker_state
        if state is None or state.last_observation is None:
            return False, "no_grounding_observation"
        observation = state.last_observation
        if _is_fresh_grounded_observation(observation):
            return True, "grounded_visible"
        reason = str(getattr(observation, "reason", "")).strip().lower()
        source = str(getattr(observation, "source", "")).strip().lower()
        if (
            bool(getattr(observation, "visible", False))
            and "reused" not in source
            and bool(getattr(observation, "is_current_frame", True))
            and reason in {"ok", "depth_invalid", "no_depth", "no_world_xy"}
        ):
            return True, f"image_visible_no_world:{reason or 'visible'}"
        if str(getattr(state, "status", "")).strip().lower() in {"visible", "confirmed", "reached"} and int(
            getattr(state, "lost_count", 0)
        ) == 0:
            return True, f"tracked_{state.status}"
        return False, str(observation.reason).strip() or "grounded_not_visible"

    def _current_route_anchor(self) -> tuple[list[float] | None, str]:
        entry = self._landmark_map_entry
        if (
            entry is not None
            and str(entry.label).strip().lower() == self._current_target_text().strip().lower()
            and bool(entry.world_xy)
            and bool(entry.confirmed)
        ):
            return [float(entry.world_xy[0]), float(entry.world_xy[1])], "LandmarkMap.confirmed"
        state = self._landmark_tracker_state
        if state is None:
            return None, "none"
        observation = state.last_observation
        if observation is not None and bool(observation.visible) and observation.estimated_world_xy is not None:
            return [float(observation.estimated_world_xy[0]), float(observation.estimated_world_xy[1])], "TargetObservation.estimated"
        if state.estimated_world_xy is not None and str(state.status).strip().lower() in {"confirmed", "reached"}:
            return [float(state.estimated_world_xy[0]), float(state.estimated_world_xy[1])], "LandmarkTracker.confirmed"
        return None, "none"

    def _evaluate_route_consistency(self, *, target_visible_hint: bool | None) -> RouteConsistencyResult | None:
        current_grounded_visible, _ = self._current_grounded_visibility()
        if not (bool(target_visible_hint) or bool(current_grounded_visible)) or not self._completed_landmarks:
            return None
        previous = self._completed_landmarks[-1]
        previous_anchor = self._landmark_anchors.get(previous)
        previous_yaw = self._landmark_anchor_yaws.get(previous)
        current_anchor_world_xy, anchor_source = self._current_route_anchor()
        if previous_anchor is None or previous_yaw is None or current_anchor_world_xy is None:
            return None
        relative = _relative_xy_from_anchor(
            anchor_world_xy=previous_anchor,
            anchor_yaw_rad=previous_yaw,
            target_world_xy=current_anchor_world_xy,
        )
        observed_side = _relative_side(local_forward_m=relative[0], local_left_m=relative[1])
        expected_words = sorted(self._expected_direction_words())
        consistency = _direction_consistency(
            expected_words=set(expected_words),
            observed_side=observed_side,
        )
        return RouteConsistencyResult(
            consistency=consistency,
            observed_side=observed_side,
            relative_forward_m=float(relative[0]),
            relative_left_m=float(relative[1]),
            expected_words=expected_words,
            anchor_source=anchor_source,
        )

    def _select_route_correction_candidate(
        self,
        *,
        candidate_set: CandidateSet,
        route_consistency: RouteConsistencyResult,
    ) -> WaypointCandidate | None:
        valid_candidates = [candidate for candidate in candidate_set.candidates if bool(candidate.valid)]
        if not valid_candidates:
            return None
        desired_actions: list[str] = []
        expected_words = set(route_consistency.expected_words)
        if "left" in expected_words:
            desired_actions.append("TURN_LEFT")
        if "right" in expected_words:
            desired_actions.append("TURN_RIGHT")
        if {"forward", "straight", "ahead"} & expected_words:
            desired_actions.append("FORWARD")
        for desired_action in desired_actions:
            matching = [
                candidate
                for candidate in valid_candidates
                if _candidate_to_primitive_action(
                    candidate,
                    turn_speed_scale=float(self._config.turn_speed_scale),
                    forward_speed_scale=float(self._config.forward_speed_scale),
                )[0]
                == desired_action
            ]
            if matching:
                return max(matching, key=lambda candidate: (float(candidate.clearance_score), float(candidate.travel_distance_m)))
        return None

    def _topology_revisit_step(
        self,
        *,
        observation: VLNObservation,
        candidate_set: CandidateSet,
        progress_signal: str,
        debug_image_path: str | Path | None,
        now: float,
        reason: str,
    ) -> PolicyStepResult | None:
        if not bool(self._topology_revisit_world_xy):
            return None
        local_goal = _world_waypoint_to_local_xy(
            robot_pose=observation.pose,
            world_goal_xy=self._topology_revisit_world_xy,
        )
        if float(np.hypot(local_goal[0], local_goal[1])) <= max(0.20, float(self._config.stop_distance_m)):
            return None
        action_name = _local_goal_to_primitive_action(local_goal_xy=local_goal)
        if action_name == "STOP":
            return None
        selected_candidate = _nearest_candidate_for_local_goal(candidate_set=candidate_set, local_goal_xy=local_goal)
        selected_candidate_id = None if selected_candidate is None else int(selected_candidate.slot_id)
        self._active_world_goal_xy = list(self._topology_revisit_world_xy)
        self._active_candidate_id = selected_candidate_id
        self._active_candidate_label = "topology_revisit"
        self._execution_steps_remaining = max(1, int(self._config.execution_steps))
        self._active_action_name = str(action_name)
        self._active_goal_started_sec = float(now)
        self._active_start_yaw_rad = float(observation.pose.yaw_rad)
        self._active_start_pose = observation.pose
        self._last_decision_summary = (
            f"topology revisit active; reason={reason}; "
            f"target_world_xy={_format_xy(self._topology_revisit_world_xy)}; "
            f"local_goal_xy=[{float(local_goal[0]):.2f},{float(local_goal[1]):.2f}]"
        )
        self._logger.warning(
            "[TopologyRevisit] reason=%s target=%s world_xy=%s local_goal_xy=[%.2f,%.2f] action=%s",
            reason,
            self._current_target_text(),
            self._topology_revisit_world_xy,
            float(local_goal[0]),
            float(local_goal[1]),
            action_name,
        )
        return PolicyStepResult(
            action_name=action_name,
            speed_scale=float(_primitive_speed_scale(action_name, self._config)),
            decision_source="TOPOLOGY_REVISIT",
            selected_candidate_id=selected_candidate_id,
            local_goal_xy=[float(local_goal[0]), float(local_goal[1])],
            world_goal_xy=list(self._topology_revisit_world_xy),
            execution_steps_remaining=int(self._execution_steps_remaining),
            reasoning_summary=self._last_decision_summary,
            progress_signal=progress_signal,
            replanned=True,
            debug_image_path=None if debug_image_path is None else str(debug_image_path),
            **self._route_result_kwargs(),
        )

    def _apply_reacquire_guard(
        self,
        *,
        guard: ReacquireGuardContext,
        llm_selected_action: str,
        current_semantic_visible: bool,
        has_non_forbidden_valid_candidate: bool,
    ) -> tuple[str, bool, str]:
        selected = str(llm_selected_action).strip().upper() or "STOP"
        if not bool(guard.lost_after_visible):
            return selected, False, "none"
        if bool(current_semantic_visible):
            return selected, False, "target_visible_again_in_current_image"
        if selected != str(guard.forbidden_action).strip().upper():
            return selected, False, "none"
        if not bool(has_non_forbidden_valid_candidate):
            return selected, False, "no_non_forbidden_valid_candidate"
        return selected, False, "repeat_turn_not_overridden_no_replacement"

    def _select_reacquire_replacement_candidate(
        self,
        *,
        candidate_set: CandidateSet,
        guard: ReacquireGuardContext,
        selected_candidate: WaypointCandidate,
        action_name: str,
    ) -> WaypointCandidate | None:
        if not bool(guard.lost_after_visible):
            return None
        forbidden = str(guard.forbidden_action).strip().upper()
        selected_action = str(action_name).strip().upper()
        if selected_action != forbidden or forbidden not in {"TURN_LEFT", "TURN_RIGHT"}:
            return None
        candidates: list[WaypointCandidate] = []
        for candidate in candidate_set.candidates:
            if not bool(candidate.valid):
                continue
            candidate_action, _ = _candidate_to_primitive_action(
                candidate,
                turn_speed_scale=float(self._config.turn_speed_scale),
                forward_speed_scale=float(self._config.forward_speed_scale),
            )
            if str(candidate_action).strip().upper() == forbidden:
                continue
            candidates.append(candidate)
        if not candidates:
            return None

        def score(candidate: WaypointCandidate) -> tuple[float, float, float, float]:
            candidate_action, _ = _candidate_to_primitive_action(
                candidate,
                turn_speed_scale=float(self._config.turn_speed_scale),
                forward_speed_scale=float(self._config.forward_speed_scale),
            )
            action = str(candidate_action).strip().upper()
            action_priority = 2.0 if action == "FORWARD" else 1.0
            clearance = float(candidate.clearance_score)
            distance = float(candidate.travel_distance_m)
            heading_penalty = -abs(float(candidate.heading_delta_deg)) / 90.0
            return action_priority, clearance, distance, heading_penalty

        replacement = max(candidates, key=score)
        self._logger.warning(
            "[ReacquireGuard] replacing repeated_turn=%s selected_candidate=%s replacement_candidate=%s replacement_action=%s reason=target_lost_after_visible",
            forbidden,
            int(selected_candidate.slot_id),
            int(replacement.slot_id),
            _candidate_to_primitive_action(
                replacement,
                turn_speed_scale=float(self._config.turn_speed_scale),
                forward_speed_scale=float(self._config.forward_speed_scale),
            )[0],
        )
        return replacement

    def _update_semantic_visibility_state(self, *, decision, now: float) -> None:
        semantic_visible = bool(decision.target_visible)
        if semantic_visible:
            self._last_semantic_visible_time_sec = float(now)
        bearing_hint = str(decision.target_bearing_hint or "").strip().lower()
        if bearing_hint in {"left", "right", "center"}:
            self._last_semantic_bearing_hint = bearing_hint
        self._previous_semantic_visible = semantic_visible
        current_grounded_visible, _ = self._current_grounded_visibility()
        self._previous_grounded_visible = bool(current_grounded_visible)

    def _log_reacquire_guard(
        self,
        *,
        guard: ReacquireGuardContext,
        current_semantic_visible: bool,
        llm_selected_action: str,
        final_selected_action: str,
        action_overridden: bool,
        override_reason: str,
    ) -> None:
        if not bool(guard.lost_after_visible) and not bool(action_overridden):
            return
        log_fn = self._logger.warning if action_overridden else self._logger.info
        log_fn(
            "[ReacquireGuard] previous_semantic_visible=%s previous_grounded_visible=%s "
            "current_semantic_visible=%s current_grounded_visible=%s target_recently_visible=%s "
            "last_action=%s last_seen_bearing=%s lost_after_visible=%s forbidden_action=%s "
            "llm_selected_action=%s final_selected_action=%s action_overridden=%s override_reason=%s grounded_reason=%s",
            int(bool(guard.previous_semantic_visible)),
            int(bool(guard.previous_grounded_visible)),
            int(bool(current_semantic_visible)),
            int(bool(guard.current_grounded_visible)),
            int(bool(guard.target_recently_visible)),
            guard.last_action,
            guard.last_seen_bearing,
            int(bool(guard.lost_after_visible)),
            guard.forbidden_action,
            llm_selected_action,
            final_selected_action,
            int(bool(action_overridden)),
            override_reason,
            guard.grounded_reason,
        )

    def _clear_recovery(self) -> None:
        self._recovery_action_name = None
        self._recovery_steps_remaining = 0
        self._recovery_started_sec = None

    def _should_start_recovery(self, *, progress_state) -> bool:
        if not bool(self._config.recovery_enabled):
            return False
        if progress_state.progress_signal not in {"STALLED", "SPINNING"}:
            return False
        if int(progress_state.stalled_steps) < max(1, int(self._config.recovery_trigger_stalled_steps)):
            return False
        return True

    def _should_start_safety_recovery(self) -> bool:
        return bool(self._config.recovery_enabled) and int(self._safety_override_streak) >= 2

    def _start_recovery(
        self,
        *,
        observation: VLNObservation,
        progress_signal: str,
        now: float,
        reason: str = "progress_stalled",
    ) -> PolicyStepResult:
        self._clear_active_goal()
        self._recovery_action_name = self._choose_recovery_action()
        self._recovery_steps_remaining = max(1, int(self._config.recovery_execution_steps))
        self._recovery_started_sec = float(now)
        self._recovery_count += 1
        self._safety_override_streak = 0
        self._last_safety_override_signature = None
        self._logger.warning(
            "[Recovery] start action=%s reason=%s signal=%s count=%s current_landmark=%s",
            self._recovery_action_name,
            reason,
            progress_signal,
            self._recovery_count,
            self._current_target_text(),
        )
        return self._recovery_step(observation=observation, progress_signal=progress_signal, now=now)

    def _recovery_step(
        self,
        *,
        observation: VLNObservation,
        progress_signal: str,
        now: float,
    ) -> PolicyStepResult:
        action_name = str(self._recovery_action_name or "STOP").upper()
        elapsed_sec = 0.0 if self._recovery_started_sec is None else max(0.0, float(now) - float(self._recovery_started_sec))
        max_duration_sec = self._active_action_duration_sec(action_name)
        if self._recovery_steps_remaining <= 0 or elapsed_sec >= max_duration_sec:
            completed_action = action_name
            self._recovery_action_name = None
            self._recovery_steps_remaining = 0
            self._recovery_started_sec = None
            self._next_replan_time = float(now) + float(self._config.replanning_cooldown_sec)
            self._logger.warning("[Recovery] complete action=%s; requesting fresh VLM decision", completed_action)
            return PolicyStepResult(
                action_name="STOP",
                speed_scale=0.0,
                decision_source="RECOVERY_COMPLETE",
                selected_candidate_id=None,
                local_goal_xy=[0.0, 0.0],
                world_goal_xy=None,
                execution_steps_remaining=0,
                reasoning_summary=f"recovery action {completed_action} completed; requesting a fresh visual decision",
                progress_signal=progress_signal,
                replanned=False,
                debug_image_path=(
                    None if self._candidate_predictor.last_saved_debug_path is None
                    else str(self._candidate_predictor.last_saved_debug_path)
                ),
                **self._route_result_kwargs(),
            )

        self._recovery_steps_remaining -= 1
        speed_scale = float(self._config.turn_speed_scale)
        return PolicyStepResult(
            action_name=action_name,
            speed_scale=float(speed_scale),
            decision_source="RECOVERY",
            selected_candidate_id=None,
            local_goal_xy=[0.0, 0.0],
            world_goal_xy=None,
            execution_steps_remaining=int(self._recovery_steps_remaining),
            reasoning_summary=(
                f"stagnation recovery active; executing {action_name} before replanning. "
                f"target_memory={self._combined_target_memory_text(robot_pose=observation.pose)}"
            ),
            progress_signal=progress_signal,
            replanned=False,
            debug_image_path=(
                None if self._candidate_predictor.last_saved_debug_path is None
                else str(self._candidate_predictor.last_saved_debug_path)
            ),
            **self._route_result_kwargs(),
        )

    def _choose_recovery_action(self) -> str:
        cycle = self._recovery_count % 4
        if cycle in {0, 1}:
            return self._preferred_search_turn_action()
        if cycle == 2:
            preferred = self._preferred_search_turn_action()
            return "TURN_LEFT" if preferred == "TURN_RIGHT" else "TURN_RIGHT"
        return "FORWARD"

    def _preferred_search_turn_action(self) -> str:
        bearing = str(self._target_memory.state.last_seen_bearing_hint).lower()
        if bearing == "left":
            return "TURN_LEFT"
        if bearing == "right":
            return "TURN_RIGHT"
        expected_words = self._expected_direction_words()
        if "left" in expected_words:
            return "TURN_LEFT"
        if "right" in expected_words:
            return "TURN_RIGHT"
        return "TURN_RIGHT"

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
                debug_image_path=(
                    None if self._candidate_predictor.last_saved_debug_path is None
                    else str(self._candidate_predictor.last_saved_debug_path)
                ),
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
        odom_completion_enabled = bool(self._config.active_odom_completion_enabled)
        completion_reason = ""
        if odom_completion_enabled and is_turn and yaw_delta_rad >= max(0.0, target_yaw_rad - yaw_tolerance_rad):
            completion_reason = "odom_target_yaw_reached"
        elif odom_completion_enabled and action_name == "FORWARD" and traveled_m >= max(0.0, forward_target_m - distance_tolerance_m):
            completion_reason = "odom_forward_distance_reached"
        elif elapsed_sec >= max_duration_sec:
            completion_reason = "timeout_reached" if odom_completion_enabled and (is_turn or action_name == "FORWARD") else "duration_reached"

        if completion_reason:
            completed_candidate_id = self._active_candidate_id
            completed_world_goal_xy = (
                list(self._active_world_goal_xy) if self._active_world_goal_xy is not None else None
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
                debug_image_path=(
                    None if self._candidate_predictor.last_saved_debug_path is None
                    else str(self._candidate_predictor.last_saved_debug_path)
                ),
                **self._route_result_kwargs(),
            )

        return PolicyStepResult(
            action_name=action_name,
            speed_scale=float(_primitive_speed_scale(action_name, self._config)),
            decision_source="ACTIVE_PRIMITIVE",
            selected_candidate_id=self._active_candidate_id,
            local_goal_xy=[float(local_goal[0]), float(local_goal[1])],
            world_goal_xy=list(self._active_world_goal_xy) if self._active_world_goal_xy is not None else None,
            execution_steps_remaining=int(self._execution_steps_remaining),
            reasoning_summary=self._last_decision_summary,
            progress_signal=progress_signal,
            replanned=False,
            debug_image_path=(
                None if self._candidate_predictor.last_saved_debug_path is None
                else str(self._candidate_predictor.last_saved_debug_path)
            ),
            **self._route_result_kwargs(),
        )

    def _current_target_text(self) -> str:
        route_step = self._current_route_step()
        if route_step is not None:
            landmark = str(getattr(route_step, "landmark", "")).strip().lower()
            if _is_valid_landmark(landmark):
                return landmark
        if self._ordered_landmarks and self._current_landmark_index < len(self._ordered_landmarks):
            return str(self._ordered_landmarks[self._current_landmark_index])
        return self._instruction_context.goal_phrase or self._instruction_context.normalized_text

    def _current_route_step(self) -> RouteStep | None:
        if self._current_landmark_index < len(self._route_steps):
            return self._route_steps[self._current_landmark_index]
        return None

    def _route_step_count(self) -> int:
        if self._route_steps:
            return len(self._route_steps)
        return len(self._ordered_landmarks)

    def _sync_instruction_target(self) -> None:
        self._instruction_context.goal_phrase = self._current_target_text()

    def _route_result_kwargs(self) -> dict[str, Any]:
        return {
            "current_landmark": "" if self._route_complete else self._current_target_text(),
            "completed_landmarks": list(self._completed_landmarks),
            "route_complete": bool(self._route_complete),
        }

    def _current_arrival_distance_m(self) -> float:
        if self._route_step_count() > 0 and self._current_landmark_index >= self._route_step_count() - 1:
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
            distance_m = tracker_state.anchor_distance_m
            if distance_m is None and tracker_state.last_observation is not None:
                distance_m = tracker_state.last_observation.depth_m
            return self._advance_current_landmark(
                observation=observation,
                anchor_world_xy=tracker_state.estimated_world_xy or tracker_state.candidate_world_xy,
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
            and self.current_behavior_type == "go_to"
        ):
            threshold_m = max(0.05, float(self._current_arrival_distance_m()))
            if float(map_entry.anchor_distance_m) <= threshold_m:
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
        self._arrival_confirmation_window.clear()
        if anchor_world_xy is not None:
            self._landmark_anchors[reached_landmark] = list(anchor_world_xy)
            self._landmark_anchor_yaws[reached_landmark] = float(observation.pose.yaw_rad)
        if reached_landmark not in self._completed_landmarks:
            self._completed_landmarks.append(reached_landmark)
        self._current_landmark_index += 1
        self._turn_at_landmark_index = None
        self._turn_at_steps_taken = 0
        self._turn_at_exhausted_landmark_index = None
        self._pass_landmark_index = None
        self._pass_actions = []
        self._pass_action_index = 0

        distance_text = "none" if distance_m is None else f"{float(distance_m):.2f}"
        if self._current_landmark_index >= self._route_step_count():
            self._route_complete = True
            self._logger.info(
                "[Landmark] reached=%s source=%s reason=%s distance=%sm visible_now=%s lost_count=%s route=complete completed=%s",
                reached_landmark,
                source,
                reason,
                distance_text,
                int(bool(visible_now)),
                int(lost_count),
                self._completed_landmarks,
            )
            return True

        next_landmark = self._current_target_text()
        self._target_memory.reset_for_target(next_landmark)
        self._sync_instruction_target()
        self._landmark_tracker_state = None
        self._logger.info(
            "[Landmark] reached=%s source=%s reason=%s distance=%sm visible_now=%s lost_count=%s next=%s completed=%s",
            reached_landmark,
            source,
            reason,
            distance_text,
            int(bool(visible_now)),
            int(lost_count),
            next_landmark,
            self._completed_landmarks,
        )
        return True

    def _route_context_text(self, *, robot_pose) -> str:
        current_target = self._current_target_text()
        current_index = self._current_landmark_index
        current_route_step = self._current_route_step()
        expected_words = self._expected_direction_words()
        route_steps = []
        for step in self._route_steps:
            raw_behavior = str(getattr(step, "behavior_type", "go_to")).strip().lower() or "go_to"
            route_steps.append(
                f"{raw_behavior}:{getattr(step, 'landmark', '')}:{getattr(step, 'directive', '')}"
            )
        lines = [
            f"ordered_landmarks={self._ordered_landmarks}",
            f"route_steps={route_steps or ['none']}",
            f"current_route_step_index={current_index}",
            f"current_behavior_type={self.current_behavior_type}",
            f"current_route_step={current_route_step}",
            f"current_landmark={current_target}",
            f"completed_landmarks={self._completed_landmarks}",
            f"directional_clause_for_current={self._current_direction_clause() or 'none'}",
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
        start = max(0, int(self._current_landmark_index) - 2)
        end = min(len(self._route_steps), int(self._current_landmark_index) + 1)
        window = self._route_steps[start:end]
        parts: list[str] = []
        direction_tokens: list[str] = []
        for absolute_index, step in enumerate(window, start=start):
            directive = str(getattr(step, "directive", "")).strip()
            behavior = str(getattr(step, "behavior_type", "go_to")).strip().lower() or "go_to"
            landmark = str(getattr(step, "landmark", "")).strip().lower()
            tokens = _direction_tokens_from_text(directive)
            direction_tokens.extend(tokens)
            marker = "CURRENT" if absolute_index == self._current_landmark_index else "PREV"
            parts.append(
                f"{marker}[{absolute_index}]={behavior}:{landmark}:directive={directive or 'none'}:directions={tokens or ['none']}"
            )
        latest = direction_tokens[-1] if direction_tokens else "unknown"
        return (
            f"recent_route_window={' | '.join(parts)}; "
            f"latest_direction_bias={latest}; "
            "when current target is not visible, prefer safe frontier/candidate directions that satisfy this recent route direction chain relative to completed landmarks"
        )

    def _current_direction_clause(self) -> str:
        route_step = self._current_route_step()
        if route_step is not None:
            directive = str(getattr(route_step, "directive", "")).strip()
            if directive:
                return directive
        if self._instruction_context.actions and self._current_landmark_index < len(self._instruction_context.actions):
            return str(self._instruction_context.actions[self._current_landmark_index])
        current = self._current_target_text().lower()
        for clause in self._instruction_context.directional_clauses:
            if current and current in str(clause).lower():
                return str(clause)
        return ""

    def _expected_direction_words(self) -> set[str]:
        text = self._current_direction_clause().lower()
        result: set[str] = set()
        for word in ("left", "right", "forward", "straight", "ahead", "back", "backward"):
            if word in text:
                result.add(word)
        return result

    def _log_route_consistency_if_needed(
        self,
        *,
        current_anchor_world_xy: list[float],
        target_visible: bool | None,
    ) -> None:
        route_consistency = self._evaluate_route_consistency(target_visible_hint=target_visible)
        if route_consistency is not None:
            previous = self._completed_landmarks[-1]
            log_fn = self._logger.warning if route_consistency.consistency == "contradiction" else self._logger.info
            log_fn(
                "[RouteCheck] previous=%s current=%s observed=%s forward=%.2f left=%.2f expected=%s consistency=%s anchor_source=%s",
                previous,
                self._current_target_text(),
                route_consistency.observed_side,
                float(route_consistency.relative_forward_m),
                float(route_consistency.relative_left_m),
                route_consistency.expected_words,
                route_consistency.consistency,
                route_consistency.anchor_source,
            )
            return
        if not bool(target_visible) or not self._completed_landmarks:
            return
        previous = self._completed_landmarks[-1]
        previous_anchor = self._landmark_anchors.get(previous)
        previous_yaw = self._landmark_anchor_yaws.get(previous)
        if previous_anchor is None or previous_yaw is None:
            return
        relative = _relative_xy_from_anchor(
            anchor_world_xy=previous_anchor,
            anchor_yaw_rad=previous_yaw,
            target_world_xy=current_anchor_world_xy,
        )
        side = _relative_side(local_forward_m=relative[0], local_left_m=relative[1])
        consistency = _direction_consistency(expected_words=self._expected_direction_words(), observed_side=side)
        log_fn = self._logger.warning if consistency == "contradiction" else self._logger.info
        log_fn(
            "[RouteCheck] previous=%s current=%s observed=%s forward=%.2f left=%.2f expected=%s consistency=%s",
            previous,
            self._current_target_text(),
            side,
            float(relative[0]),
            float(relative[1]),
            sorted(self._expected_direction_words()),
            consistency,
        )

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
        if action == "TRACK_TARGET":
            return max(0.05, float(self._config.active_forward_duration_sec))
        return 0.05
