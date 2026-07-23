from __future__ import annotations

import unittest

from vln_nav.navigation.observation_adapter import Pose2D
from vln_nav.navigation.progress_manager import ProgressManager


class TraverseProgressTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = ProgressManager(
            history_length=6,
            stall_distance_m=0.08,
            spin_yaw_deg=85.0,
        )

    def test_confirmed_right_cue_is_passed_after_crossing_anchor_plane(self) -> None:
        self.manager.begin_traverse(
            step_id="step_1",
            robot_pose=Pose2D(0.0, 0.0, 0.0),
            cue_label="wall-mounted tv",
            cue_expected_side="right",
        )
        self.manager.update_traverse_motion(robot_pose=Pose2D(1.0, 0.0, 0.0))
        before = self.manager.update_traverse_cue(
            robot_pose=Pose2D(1.0, 0.0, 0.0),
            anchor_world_xy=[2.0, -1.0],
            confirmed=True,
            pass_margin_m=0.20,
            min_progress_m=0.50,
        )
        self.assertFalse(before.cue_passed)

        self.manager.update_traverse_motion(robot_pose=Pose2D(2.3, 0.0, 0.0))
        after = self.manager.update_traverse_cue(
            robot_pose=Pose2D(2.3, 0.0, 0.0),
            anchor_world_xy=[2.0, -1.0],
            confirmed=True,
            pass_margin_m=0.20,
            min_progress_m=0.50,
        )
        self.assertTrue(after.cue_passed)
        self.assertAlmostEqual(after.measured_progress_m, 2.3, places=3)

    def test_vlm_review_requires_two_distinct_progress_positions(self) -> None:
        self.manager.begin_traverse(
            step_id="step_1",
            robot_pose=Pose2D(0.0, 0.0, 0.0),
            cue_label="wall-mounted tv",
            cue_expected_side="right",
        )
        self.manager.update_traverse_motion(robot_pose=Pose2D(1.0, 0.0, 0.0))
        first = self.manager.record_traverse_review(
            likely_complete=True,
            min_translation_m=0.20,
        )
        self.assertEqual(first.review_confirmations, 1)

        same_pose = self.manager.record_traverse_review(
            likely_complete=True,
            min_translation_m=0.20,
        )
        self.assertEqual(same_pose.review_confirmations, 1)

        self.manager.update_traverse_motion(robot_pose=Pose2D(1.3, 0.0, 0.0))
        second = self.manager.record_traverse_review(
            likely_complete=True,
            min_translation_m=0.20,
        )
        self.assertEqual(second.review_confirmations, 2)
        self.assertEqual(second.review_stage, "CONFIRMED")

    def test_late_confirmed_anchor_can_complete_after_robot_already_passed_it(self) -> None:
        self.manager.begin_traverse(
            step_id="step_1",
            robot_pose=Pose2D(0.0, 0.0, 0.0),
            cue_label="wall-mounted tv",
            cue_expected_side="right",
        )
        self.manager.update_traverse_motion(robot_pose=Pose2D(2.4, 0.0, 0.0))
        state = self.manager.update_traverse_cue(
            robot_pose=Pose2D(2.4, 0.0, 0.0),
            anchor_world_xy=[2.0, -1.0],
            confirmed=True,
            pass_margin_m=0.20,
            min_progress_m=0.50,
        )

        self.assertTrue(state.cue_passed)


if __name__ == "__main__":
    unittest.main()
