from __future__ import annotations

import logging
import unittest
from collections import deque
from types import SimpleNamespace

from vln_nav.navigation.candidate_waypoint_predictor import (
    CandidateSet,
    WaypointCandidate,
)
from vln_nav.navigation.observation_adapter import Pose2D
from vln_nav.navigation.progress_manager import ProgressManager
from vln_nav.navigation.vln_policy import OpenNavVLNPolicy


class TraversePolicyConstraintTest(unittest.TestCase):
    def setUp(self) -> None:
        progress = ProgressManager(
            history_length=6,
            stall_distance_m=0.08,
            spin_yaw_deg=85.0,
        )
        progress.begin_traverse(
            step_id="step_1",
            robot_pose=Pose2D(0.0, 0.0, 0.0),
            cue_label="wall-mounted tv",
            cue_expected_side="right",
        )
        policy = object.__new__(OpenNavVLNPolicy)
        policy._progress_manager = progress
        policy._config = SimpleNamespace(
            traverse_heading_correction_limit_deg=30.0,
            traverse_candidate_heading_tolerance_deg=22.0,
            turn_speed_scale=1.0,
            forward_speed_scale=1.0,
        )
        policy._recent_primitive_actions = deque(maxlen=4)
        policy._landmark_tracker_state = None
        policy._safety_context_text = ""
        policy._logger = logging.getLogger("test_traverse_policy_constraints")
        self.policy = policy

    def test_locked_heading_rejects_large_lateral_candidate(self) -> None:
        candidates = CandidateSet(
            candidates=[
                _candidate(0, 0.0),
                _candidate(1, 18.0),
                _candidate(2, 60.0),
            ],
            valid_count=3,
        )
        self.policy._apply_traverse_candidate_constraints(
            observation=SimpleNamespace(pose=Pose2D(0.0, 0.0, 0.0)),
            candidate_set=candidates,
        )

        self.assertTrue(candidates.candidates[0].valid)
        self.assertFalse(candidates.candidates[2].valid)
        self.assertEqual(
            candidates.candidates[2].depth_support["traverse_filter_reason"],
            "outside_locked_heading_corridor",
        )

    def test_immediate_opposite_turn_is_rejected_when_forward_is_safe(self) -> None:
        self.policy._recent_primitive_actions.append("TURN_LEFT")
        candidates = CandidateSet(
            candidates=[
                _candidate(0, 0.0),
                _candidate(1, -18.0),
            ],
            valid_count=2,
        )
        self.policy._apply_traverse_candidate_constraints(
            observation=SimpleNamespace(pose=Pose2D(0.0, 0.0, 0.0)),
            candidate_set=candidates,
        )

        self.assertTrue(candidates.candidates[0].valid)
        self.assertFalse(candidates.candidates[1].valid)
        self.assertEqual(
            candidates.candidates[1].depth_support["traverse_filter_reason"],
            "reverse_turn_without_new_evidence",
        )


def _candidate(slot_id: int, heading_deg: float) -> WaypointCandidate:
    return WaypointCandidate(
        slot_id=slot_id,
        label=f"candidate_{slot_id}",
        valid=True,
        local_goal_xy=[1.0, 0.0],
        heading_delta_deg=heading_deg,
        travel_distance_m=1.0,
        clearance_score=0.8,
        depth_support={"min_clearance_m": 1.0},
        image_uv=(100, 100),
    )


if __name__ == "__main__":
    unittest.main()
