from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace

from vln_nav.navigation.instruction_manager import RouteStep
from vln_nav.navigation.observation_adapter import Pose2D
from vln_nav.navigation.target_grounding import TargetObservation
from vln_nav.navigation.vln_policy import (
    VLNPolicy,
    _TurnTargetHypothesis,
)


class SemanticTurnTargetTest(unittest.TestCase):
    def test_turn_lookahead_is_exposed_only_after_completed_primitive(self) -> None:
        policy = _policy()
        policy._route_complete = False
        policy._semantic_turn_primitives_taken = 0
        policy._satisfied_turn_triggers = set()
        self.assertEqual(policy.active_perception_target, "")
        policy._semantic_turn_primitives_taken = 1
        self.assertEqual(policy.active_perception_target, "stairs")
        self.assertEqual(policy.current_grounding_behavior_type, "turn_lookahead")

    def test_high_confidence_vlm_verified_target_stops_after_one_keyframe(self) -> None:
        policy = _policy()
        policy._landmark_tracker_state = SimpleNamespace(
            last_observation=_observation(frame_index=10, confidence=0.82)
        )
        self.assertTrue(policy._turn_lookahead_target_is_ready())

    def test_medium_confidence_requires_two_consistent_keyframes(self) -> None:
        policy = _policy()
        policy._landmark_tracker_state = SimpleNamespace(
            last_observation=_observation(frame_index=10, confidence=0.64)
        )
        self.assertFalse(policy._turn_lookahead_target_is_ready())
        policy._landmark_tracker_state = SimpleNamespace(
            last_observation=_observation(
                frame_index=20,
                confidence=0.66,
                world_xy=[2.1, 1.0],
            )
        )
        self.assertTrue(policy._turn_lookahead_target_is_ready())

    def test_missing_keyframe_breaks_medium_confidence_sequence(self) -> None:
        policy = _policy()
        policy._landmark_tracker_state = SimpleNamespace(
            last_observation=_observation(frame_index=10, confidence=0.64)
        )
        self.assertFalse(policy._turn_lookahead_target_is_ready())
        missing = _observation(frame_index=20, confidence=0.0)
        missing.visible = False
        missing.bbox = None
        missing.can_confirm = False
        missing.vlm_verified = False
        missing.semantic_source = "unknown"
        policy._landmark_tracker_state = SimpleNamespace(last_observation=missing)
        self.assertFalse(policy._turn_lookahead_target_is_ready())
        policy._landmark_tracker_state = SimpleNamespace(
            last_observation=_observation(frame_index=30, confidence=0.65)
        )
        self.assertFalse(policy._turn_lookahead_target_is_ready())

    def test_low_confidence_does_not_affect_turn(self) -> None:
        policy = _policy()
        policy._landmark_tracker_state = SimpleNamespace(
            last_observation=_observation(frame_index=10, confidence=0.49)
        )
        self.assertFalse(policy._turn_lookahead_target_is_ready())
        self.assertEqual(policy._semantic_turn_target_hypothesis.confirmation_count, 0)

    def test_direction_conflict_review_requires_an_extra_keyframe(self) -> None:
        policy = _policy()
        policy._landmark_tracker_state = SimpleNamespace(
            last_observation=_observation(
                frame_index=10,
                confidence=0.84,
                turn_relation_verified=True,
            )
        )
        self.assertFalse(policy._turn_lookahead_target_is_ready())
        policy._landmark_tracker_state = SimpleNamespace(
            last_observation=_observation(frame_index=20, confidence=0.83)
        )
        self.assertTrue(policy._turn_lookahead_target_is_ready())

    def test_no_depth_can_end_turn_without_creating_anchor(self) -> None:
        policy = _policy()
        policy._landmark_tracker_state = SimpleNamespace(
            last_observation=_observation(
                frame_index=10,
                confidence=0.86,
                world_xy=None,
                depth_m=None,
            )
        )
        self.assertTrue(policy._turn_lookahead_target_is_ready())
        self.assertFalse(policy._semantic_turn_target_hypothesis.has_world_anchor)

    def test_recovery_is_bounded_then_enters_fatal_hold(self) -> None:
        policy = _policy()
        policy._config.semantic_turn_max_recovery_attempts = 2
        policy._clear_active_goal = lambda: None
        policy._hold_result = lambda **kwargs: kwargs
        first = policy._request_turn_recovery(
            progress_signal="NO_PROGRESS",
            debug_image_path=None,
            reason="no_corridor",
        )
        second = policy._request_turn_recovery(
            progress_signal="NO_PROGRESS",
            debug_image_path=None,
            reason="no_corridor",
        )
        final = policy._request_turn_recovery(
            progress_signal="NO_PROGRESS",
            debug_image_path=None,
            reason="no_corridor",
        )
        self.assertEqual(first["decision_source"], "TURN_RECOVERY_REQUEST")
        self.assertEqual(second["decision_source"], "TURN_RECOVERY_REQUEST")
        self.assertEqual(final["decision_source"], "TURN_FATAL_SAFETY_HOLD")


def _policy() -> VLNPolicy:
    policy = VLNPolicy.__new__(VLNPolicy)
    policy._route_steps = [
        RouteStep(
            step_id="step_1",
            step_type="TURN",
            source_span="turn left",
            direction="left",
        ),
        RouteStep(
            step_id="step_2",
            step_type="GO_TO",
            source_span="find the stairs",
            target="stairs",
        ),
    ]
    policy._current_step_index = 0
    policy._config = SimpleNamespace(
        semantic_turn_target_medium_confidence=0.50,
        semantic_turn_target_high_confidence=0.75,
        semantic_turn_max_recovery_attempts=2,
    )
    policy._semantic_turn_target_hypothesis = _TurnTargetHypothesis(label="stairs")
    policy._semantic_turn_recovery_attempts = 0
    policy._semantic_turn_restart_after_recovery = False
    policy._semantic_turn_samples = []
    policy._semantic_turn_return_yaw_rad = None
    policy._landmark_tracker_state = None
    policy._logger = logging.getLogger("test.semantic_turn")
    return policy


def _observation(
    *,
    frame_index: int,
    confidence: float,
    world_xy: list[float] | None = None,
    depth_m: float | None = 2.0,
    turn_relation_verified: bool = False,
) -> TargetObservation:
    if world_xy is None and depth_m is not None:
        world_xy = [2.0, 1.0]
    return TargetObservation(
        label="stairs",
        visible=True,
        bbox=(100, 80, 220, 260),
        center_uv=(160.0, 170.0),
        confidence=float(confidence),
        depth_m=depth_m,
        bearing_rad=-0.20,
        estimated_world_xy=world_xy,
        depth_valid_ratio=0.8 if depth_m is not None else 0.0,
        area_ratio=0.1,
        source="grounding_dino:original_pixels",
        reason="ok",
        timestamp_sec=float(frame_index),
        request_frame_index=int(frame_index),
        request_sim_time_sec=float(frame_index),
        request_robot_pose=Pose2D(
            x=0.0,
            y=0.0,
            yaw_rad=0.20 + 0.01 * float(frame_index),
        ),
        is_current_frame=True,
        evidence_level="confirmed_candidate",
        semantic_source="grounding_dino",
        can_confirm=True,
        vlm_verified=True,
        vlm_verification_confidence=0.9,
        requires_viewpoint_confirmation=confidence < 0.75,
        turn_direction_consistent=True,
        turn_relation_verified=bool(turn_relation_verified),
        turn_relation_reason="verified" if turn_relation_verified else "not_required",
    )


if __name__ == "__main__":
    unittest.main()
