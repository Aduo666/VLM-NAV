from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from vln_nav.app.navigation import _create_coarse_tracker, _join_path_from_current
from vln_nav.control.trajectory import PathTracker
from vln_nav.navigation.navdp_client import NavDPClient, NavDPClientConfig, normalize_navdp_response
from vln_nav.runtime.agx_profile import load_agx_profile


ROOT = Path(__file__).resolve().parents[1]


class CoarseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "configs/navigation.yaml").open(encoding="utf-8") as handle:
            cls.settings = yaml.safe_load(handle)
        cls.client_config = NavDPClientConfig.from_mapping(cls.settings["server"])
        cls.tracking_config = load_agx_profile(project_root=ROOT).agx_controller.continuous

    def response(self, path):
        points = np.asarray(path, dtype=float)
        return normalize_navdp_response(
            {"trajectory": [points.tolist()], "all_trajectory": [[points.tolist()]], "all_values": [[1.0]]},
            latency_sec=5.0, config=self.client_config,
        )

    def test_short_path_accepted(self):
        result = self.response([[0, 0], [0.01, 0]])
        np.testing.assert_array_equal(result.model_choice.trajectory_xy, [[0, 0], [0.01, 0]])

    def test_large_coordinates_and_jumps_accepted(self):
        result = self.response([[0, 0], [20, 3]])
        np.testing.assert_array_equal(result.model_choice.trajectory_xy[-1], [20, 3])

    def test_nonfinite_coordinates_rejected(self):
        with self.assertRaises(RuntimeError):
            self.response([[0, 0], [np.nan, 0]])

    def test_malformed_coordinates_rejected(self):
        with self.assertRaises(RuntimeError):
            self.response([[0], [1]])

    def test_low_depth_coverage_sent_to_server(self):
        client = NavDPClient(self.client_config)
        path = [[[0.0, 0.0], [0.1, 0.0]]]
        response = {"trajectory": path, "all_trajectory": [path], "all_values": [[1.0]]}
        with patch.object(client, "_post_multipart", return_value=response) as request:
            result = client.pointgoal_step(
                rgb=np.zeros((20, 20, 3), dtype=np.uint8), depth_m=np.zeros((20, 20), dtype=np.float32),
                goal_forward_m=0.05, goal_left_m=0.0,
            )
        request.assert_called_once()
        self.assertEqual(len(result.candidates), 1)

    def test_connection_distance_does_not_reject_tracker(self):
        path = np.asarray([[2, 0], [3, 0]], dtype=float)
        joined, error, remaining = _join_path_from_current(path, current_xy=(0, 0))
        tracker = _create_coarse_tracker(path, joined_path=joined, config=self.tracking_config, initial_speed_mps=0)
        self.assertGreater(error, self.settings["execution"]["max_path_join_error_m"])
        self.assertGreater(remaining, 0)
        self.assertIsNotNone(tracker)
        self.assertEqual(tracker.step([0, 0, 0], 1 / 60).status, "TRACKING")

    def test_short_moving_path_has_tracker(self):
        path = np.asarray([[0, 0], [0.01, 0]], dtype=float)
        joined, _, _ = _join_path_from_current(path, current_xy=(0, 0))
        self.assertIsNotNone(_create_coarse_tracker(path, joined_path=joined,
                                                  config=self.tracking_config, initial_speed_mps=0))

    def test_zero_motion_path_has_no_tracker(self):
        path = self.response([[0, 0], [0, 0]]).model_choice.trajectory_xy
        for current in ((0, 0), (0.2, 0)):
            joined, _, _ = _join_path_from_current(path, current_xy=current)
            self.assertIsNone(_create_coarse_tracker(path, joined_path=joined,
                                                    config=self.tracking_config, initial_speed_mps=0))

    def test_tracking_error_does_not_stop(self):
        tracker = PathTracker([[0, 0], [2, 0]], self.tracking_config)
        result = tracker.step([0, 2, 0], 1 / 60)
        self.assertGreater(result.cross_track_m, self.tracking_config.max_path_error_m)
        self.assertEqual(result.status, "TRACKING")
        self.assertGreater(result.command.linear_mps, 0)

    def test_tracking_timeout_does_not_stop(self):
        tracker = PathTracker([[0, 0], [2, 0]], replace(self.tracking_config, timeout_sec=0.01))
        self.assertEqual(tracker.step([0, 0, 0], 0.02).status, "TRACKING")

    def test_target_behind_does_not_stop(self):
        tracker = PathTracker([[0, 0], [-2, 0]], self.tracking_config)
        self.assertEqual(tracker.step([0, 0, 0], 1 / 60).status, "TRACKING")

    def test_arrival_still_stops(self):
        tracker = PathTracker([[0, 0], [0.1, 0]], self.tracking_config)
        result = tracker.step([0.1, 0, 0], 1 / 60)
        self.assertEqual(result.status, "ARRIVED")
        self.assertEqual(result.command.linear_mps, 0)
        self.assertEqual(result.command.angular_radps, 0)


if __name__ == "__main__":
    unittest.main()
