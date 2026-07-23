from __future__ import annotations

import unittest

import numpy as np

from vln_nav.navigation.observation_adapter import Pose2D
from vln_nav.navigation.occupancy_grid import OccupancyGridSnapshot
from vln_nav.navigation.turn_heading_estimator import (
    TurnHeadingEstimator,
    TurnHeadingEstimatorConfig,
)


class TurnHeadingEstimatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.estimator = TurnHeadingEstimator(
            TurnHeadingEstimatorConfig(
                heading_half_fov_deg=60.0,
                heading_step_deg=5.0,
                lidar_bin_half_width_deg=2.5,
                min_corridor_width_deg=15.0,
                min_clearance_m=0.60,
                max_clearance_m=4.0,
            )
        )
        self.pose = Pose2D(x=0.0, y=0.0, yaw_rad=0.0)

    def test_lidar_corridor_center_is_estimated(self) -> None:
        endpoints = _synthetic_scan(
            open_ranges=[(-15.0, 25.0)],
        )
        estimate = self.estimator.estimate(
            robot_pose=self.pose,
            beam_endpoints_local_xy=endpoints,
            occupancy_snapshot=_empty_free_snapshot(),
        )
        self.assertTrue(estimate.valid)
        self.assertAlmostEqual(estimate.local_heading_deg, 5.0, delta=5.0)
        self.assertGreaterEqual(estimate.corridor_width_deg, 35.0)

    def test_occupancy_rejects_lidar_open_but_blocked_branch(self) -> None:
        endpoints = _synthetic_scan(
            open_ranges=[(-42.0, -18.0), (18.0, 42.0)],
        )
        snapshot = _empty_free_snapshot()
        for distance_m in np.arange(0.25, 3.5, 0.05):
            world_x = float(distance_m * np.cos(np.deg2rad(30.0)))
            world_y = float(distance_m * np.sin(np.deg2rad(30.0)))
            row = int(np.floor((world_y - snapshot.origin_y) / snapshot.resolution_m))
            col = int(np.floor((world_x - snapshot.origin_x) / snapshot.resolution_m))
            snapshot.inflated_occupied[row, col] = True
            snapshot.free_mask[row, col] = False

        estimate = self.estimator.estimate(
            robot_pose=self.pose,
            beam_endpoints_local_xy=endpoints,
            occupancy_snapshot=snapshot,
        )
        self.assertTrue(estimate.valid)
        self.assertLess(estimate.local_heading_deg, -10.0)

    def test_missing_sensor_and_map_is_invalid(self) -> None:
        estimate = self.estimator.estimate(
            robot_pose=self.pose,
            beam_endpoints_local_xy=None,
            occupancy_snapshot=None,
        )
        self.assertFalse(estimate.valid)
        self.assertEqual(estimate.reason, "no_lidar_or_occupancy")


def _synthetic_scan(
    *,
    open_ranges: list[tuple[float, float]],
) -> np.ndarray:
    angles_deg = np.arange(-90.0, 90.1, 1.0, dtype=np.float32)
    ranges_m = np.full_like(angles_deg, 0.35)
    for minimum, maximum in open_ranges:
        ranges_m[(angles_deg >= minimum) & (angles_deg <= maximum)] = 4.0
    angles_rad = np.deg2rad(angles_deg)
    return np.stack(
        [ranges_m * np.cos(angles_rad), ranges_m * np.sin(angles_rad)],
        axis=1,
    ).astype(np.float32)


def _empty_free_snapshot() -> OccupancyGridSnapshot:
    resolution_m = 0.05
    cell_count = 400
    return OccupancyGridSnapshot(
        resolution_m=resolution_m,
        size_m=cell_count * resolution_m,
        log_odds=np.full((cell_count, cell_count), -1.0, dtype=np.float32),
        inflated_occupied=np.zeros((cell_count, cell_count), dtype=bool),
        free_mask=np.ones((cell_count, cell_count), dtype=bool),
        origin_x=-10.0,
        origin_y=-10.0,
    )


if __name__ == "__main__":
    unittest.main()
