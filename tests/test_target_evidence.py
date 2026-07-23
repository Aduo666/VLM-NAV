from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace

import numpy as np

from vln_nav.navigation.grounding_dino_detector import (
    GroundingDinoDetection,
    _nms_detections,
)
from vln_nav.navigation.landmark_tracker import LandmarkTracker, LandmarkTrackerConfig
from vln_nav.navigation.observation_adapter import Pose2D
from vln_nav.navigation.target_grounding import (
    TargetObservation,
    _select_grounding_detection,
)
from vln_nav.navigation.vln_policy import OpenNavVLNPolicy, _TurnCorridorSample


def _logger() -> logging.Logger:
    logger = logging.getLogger("test_target_evidence")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


class TargetEvidenceTest(unittest.TestCase):
    def test_nms_keeps_distinct_instances(self) -> None:
        detections = [
            GroundingDinoDetection((10, 10, 50, 50), 0.90, "chair"),
            GroundingDinoDetection((12, 12, 49, 49), 0.80, "chair"),
            GroundingDinoDetection((100, 10, 140, 50), 0.70, "chair"),
        ]
        kept = _nms_detections(detections, iou_threshold=0.50, max_detections=5)
        self.assertEqual([item.bbox for item in kept], [(10, 10, 50, 50), (100, 10, 140, 50)])

    def test_instance_continuity_outranks_new_high_confidence_box(self) -> None:
        detections = [
            GroundingDinoDetection((100, 20, 150, 80), 0.92, "chair"),
            GroundingDinoDetection((12, 12, 52, 52), 0.70, "chair"),
        ]
        request = SimpleNamespace(
            preferred_bbox=(10, 10, 50, 50),
            rgb_shape=(100, 200),
            depth=np.full((100, 200), 2.0, dtype=np.float32),
        )
        config = SimpleNamespace(
            min_confidence=0.50,
            min_depth_m=0.20,
            max_depth_m=10.0,
            depth_statistic="median",
            center_patch_ratio=0.25,
            depth_mask_tolerance_ratio=0.35,
            depth_mask_min_pixels=16,
            min_depth_valid_ratio=0.12,
        )
        selected, index, reason = _select_grounding_detection(
            detections=detections,
            request=request,
            config=config,
        )
        self.assertEqual(index, 1)
        self.assertEqual(selected.bbox, (12, 12, 52, 52))
        self.assertEqual(reason, "temporal_instance_continuity")

    def test_unlocked_selection_prefers_high_confidence_evidence_tier(self) -> None:
        detections = [
            GroundingDinoDetection((10, 10, 50, 50), 0.74, "chair"),
            GroundingDinoDetection((100, 20, 150, 80), 0.76, "chair"),
        ]
        depth = np.full((100, 200), 6.0, dtype=np.float32)
        depth[10:50, 10:50] = 0.5
        request = SimpleNamespace(
            preferred_bbox=None,
            preferred_world_xy=None,
            rgb_shape=(100, 200),
            depth=depth,
        )
        config = SimpleNamespace(
            min_confidence=0.50,
            semantic_verification_auto_accept_confidence=0.75,
            min_depth_m=0.20,
            max_depth_m=10.0,
            depth_statistic="median",
            center_patch_ratio=0.25,
            depth_mask_tolerance_ratio=0.35,
            depth_mask_min_pixels=16,
            min_depth_valid_ratio=0.12,
        )
        selected, index, _ = _select_grounding_detection(
            detections=detections,
            request=request,
            config=config,
        )
        self.assertEqual(index, 1)
        self.assertEqual(selected.bbox, (100, 20, 150, 80))

    def test_low_confidence_confirmation_requires_distinct_viewpoint(self) -> None:
        tracker = LandmarkTracker(
            config=LandmarkTrackerConfig(
                arrival_visible_depth_m=1.0,
                arrival_anchor_radius_m=1.0,
                min_visible_count=1,
                arrival_confirmations=1,
                max_lost_count=3,
                anchor_update_min_consistent_observations=2,
                anchor_update_max_jump_m=1.0,
                arrival_window_size=3,
                confirm_confidence_threshold=0.75,
                confirm_min_observations=2,
                confirm_max_anchor_jump_m=0.5,
                confirm_max_depth_jump_m=0.8,
                potential_confidence_threshold=0.5,
                low_confidence_min_translation_m=0.2,
                low_confidence_min_yaw_deg=10.0,
            ),
            logger=_logger(),
        )

        first_pose = Pose2D(0.0, 0.0, 0.0)
        first = tracker.update(
            target_label="chair",
            behavior_type="go_to",
            observation=_low_confidence_observation(first_pose, timestamp_sec=1.0),
            robot_pose=first_pose,
            timestamp_sec=1.0,
        )
        self.assertIsNone(first.estimated_world_xy)

        same_pose = tracker.update(
            target_label="chair",
            behavior_type="go_to",
            observation=_low_confidence_observation(first_pose, timestamp_sec=2.0),
            robot_pose=first_pose,
            timestamp_sec=2.0,
        )
        self.assertIsNone(same_pose.estimated_world_xy)

        second_pose = Pose2D(0.25, 0.0, 0.0)
        confirmed = tracker.update(
            target_label="chair",
            behavior_type="go_to",
            observation=_low_confidence_observation(second_pose, timestamp_sec=3.0),
            robot_pose=second_pose,
            timestamp_sec=3.0,
        )
        self.assertEqual(confirmed.evidence_level, "confirmed")
        self.assertEqual(confirmed.confirmed_observation_count, 2)

    def test_turn_early_exit_requires_two_consistent_corridor_samples(self) -> None:
        policy = object.__new__(OpenNavVLNPolicy)
        policy._config = SimpleNamespace(
            semantic_turn_alignment_confirmations=2,
            semantic_turn_corridor_center_tolerance_deg=12.0,
            semantic_turn_same_corridor_tolerance_deg=20.0,
        )
        policy._semantic_turn_samples = [
            _TurnCorridorSample(0.52, 30.0, 0.78, 15.0, 1.5, 1.2),
            _TurnCorridorSample(0.78, 45.0, 0.82, 2.0, 1.6, 1.3),
        ]
        self.assertTrue(policy._turn_alignment_is_stable())

        policy._semantic_turn_samples[-1] = _TurnCorridorSample(
            0.78, 45.0, 1.30, 30.0, 1.6, 1.3
        )
        self.assertFalse(policy._turn_alignment_is_stable())

    def test_depth_invalid_target_can_be_triangulated_from_two_viewpoints(self) -> None:
        tracker = LandmarkTracker(
            config=LandmarkTrackerConfig(
                arrival_visible_depth_m=1.0,
                arrival_anchor_radius_m=1.0,
                min_visible_count=1,
                arrival_confirmations=1,
                max_lost_count=3,
                anchor_update_min_consistent_observations=2,
                anchor_update_max_jump_m=1.0,
                arrival_window_size=3,
                confirm_confidence_threshold=0.75,
                confirm_min_observations=2,
                confirm_max_anchor_jump_m=0.5,
                confirm_max_depth_jump_m=0.8,
                potential_confidence_threshold=0.5,
                low_confidence_min_translation_m=0.2,
                low_confidence_min_yaw_deg=10.0,
            ),
            logger=_logger(),
        )
        target_xy = np.asarray([2.0, 1.0], dtype=np.float64)
        first_pose = Pose2D(0.0, 0.0, 0.0)
        first = tracker.update(
            target_label="tv",
            behavior_type="pass_cue",
            observation=_bearing_only_observation(
                label="tv",
                pose=first_pose,
                target_xy=target_xy,
                timestamp_sec=1.0,
            ),
            robot_pose=first_pose,
            timestamp_sec=1.0,
        )
        self.assertIsNone(first.estimated_world_xy)

        second_pose = Pose2D(0.3, 0.0, 0.0)
        second = tracker.update(
            target_label="tv",
            behavior_type="pass_cue",
            observation=_bearing_only_observation(
                label="tv",
                pose=second_pose,
                target_xy=target_xy,
                timestamp_sec=2.0,
            ),
            robot_pose=second_pose,
            timestamp_sec=2.0,
        )
        self.assertEqual(second.evidence_level, "confirmed")
        np.testing.assert_allclose(second.estimated_world_xy, target_xy, atol=0.05)


def _low_confidence_observation(pose: Pose2D, *, timestamp_sec: float) -> TargetObservation:
    return TargetObservation(
        label="chair",
        visible=True,
        bbox=(20, 20, 80, 90),
        center_uv=(50.0, 55.0),
        confidence=0.60,
        depth_m=2.0,
        bearing_rad=0.0,
        estimated_world_xy=[2.0, 0.0],
        depth_valid_ratio=1.0,
        area_ratio=0.10,
        source="grounding_dino:original_pixels",
        reason="ok",
        timestamp_sec=timestamp_sec,
        request_sim_time_sec=timestamp_sec,
        request_robot_pose=pose,
        is_current_frame=True,
        evidence_level="confirmed_candidate",
        semantic_source="grounding_dino",
        can_confirm=True,
        vlm_verified=True,
        vlm_verification_confidence=0.90,
        vlm_verification_reason="match",
        requires_viewpoint_confirmation=True,
    )


def _bearing_only_observation(
    *,
    label: str,
    pose: Pose2D,
    target_xy: np.ndarray,
    timestamp_sec: float,
) -> TargetObservation:
    delta = np.asarray(target_xy, dtype=np.float64) - np.asarray(
        [pose.x, pose.y],
        dtype=np.float64,
    )
    world_heading = float(np.arctan2(delta[1], delta[0]))
    image_right_positive_bearing = float(pose.yaw_rad) - world_heading
    return TargetObservation(
        label=label,
        visible=True,
        bbox=(20, 20, 80, 90),
        center_uv=(50.0, 55.0),
        confidence=0.85,
        depth_m=None,
        bearing_rad=image_right_positive_bearing,
        estimated_world_xy=None,
        depth_valid_ratio=0.0,
        area_ratio=0.10,
        source="grounding_dino:original_pixels",
        reason="depth_invalid",
        timestamp_sec=timestamp_sec,
        request_sim_time_sec=timestamp_sec,
        request_robot_pose=pose,
        is_current_frame=True,
        evidence_level="confirmed_candidate",
        semantic_source="grounding_dino",
        can_confirm=True,
        vlm_verified=True,
        vlm_verification_confidence=0.90,
        vlm_verification_reason="match",
    )


if __name__ == "__main__":
    unittest.main()
