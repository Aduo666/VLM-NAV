from __future__ import annotations

import unittest

import numpy as np

from vln_nav.navigation.depth_safety import DepthSafetyConfig, DepthSafetyMonitor
from vln_nav.navigation.observation_adapter import (
    MotionFeedback,
    Pose2D,
    VLNObservation,
)


class DepthSafetyMonitorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.monitor = DepthSafetyMonitor(
            config=DepthSafetyConfig(
                enabled=True,
                stop_distance_m=0.6,
                roi_vertical_range=(0.5, 0.95),
                center_width_ratio=0.34,
                side_width_ratio=0.30,
                near_field_memory_enabled=False,
            )
        )

    def test_forward_stops_for_close_front_obstacle(self) -> None:
        depth = np.full((100, 100), 4.0, dtype=np.float32)
        depth[50:95, 40:60] = 0.45

        decision = self.monitor.apply(
            observation=self._observation(depth=depth),
            action_name="FORWARD",
            speed_scale=1.0,
        )

        self.assertTrue(decision.overridden)
        self.assertEqual(decision.action_name, "STOP")
        self.assertEqual(decision.reason, "front_blocked_stop")

    def test_left_turn_checks_left_side(self) -> None:
        depth = np.full((100, 100), 4.0, dtype=np.float32)
        depth[50:95, :30] = 0.50

        left = self.monitor.apply(
            observation=self._observation(depth=depth),
            action_name="TURN_LEFT",
            speed_scale=1.0,
        )
        right = self.monitor.apply(
            observation=self._observation(depth=depth),
            action_name="TURN_RIGHT",
            speed_scale=1.0,
        )

        self.assertTrue(left.overridden)
        self.assertFalse(right.overridden)

    def test_lidar_sector_can_stop_forward_without_depth(self) -> None:
        observation = self._observation(depth=None)
        observation.meta["lidar"] = {
            "enabled": True,
            "front_depth_m": 0.40,
            "left_depth_m": 3.0,
            "right_depth_m": 3.0,
        }

        decision = self.monitor.apply(
            observation=observation,
            action_name="FORWARD",
            speed_scale=1.0,
        )

        self.assertTrue(decision.overridden)
        self.assertAlmostEqual(decision.front_depth_m, 0.40)

    @staticmethod
    def _observation(*, depth: np.ndarray | None) -> VLNObservation:
        return VLNObservation(
            rgb=None,
            depth=depth,
            pose=Pose2D(x=0.0, y=0.0, yaw_rad=0.0),
            camera={"width": 100, "height": 100, "hfov_deg": 90.0},
            motion_feedback=MotionFeedback(
                last_action="STOP",
                wheel_velocity=0.0,
                steering_angle=0.0,
            ),
            meta={"timestamp_sec": 1.0},
        )


if __name__ == "__main__":
    unittest.main()
