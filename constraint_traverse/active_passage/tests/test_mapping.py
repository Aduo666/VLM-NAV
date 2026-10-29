from __future__ import annotations

from contextlib import redirect_stderr
import io
import unittest

import numpy as np

from constraint_traverse.active_passage.__main__ import parse_args
from constraint_traverse.active_passage.contracts import CameraFrame, PredictedObservation, RealObservation, RobotState
from constraint_traverse.active_passage.mapping import ObservationMapFusion
from constraint_traverse.rgbd_map import RGBDMapConfig, RGBDPlanarMap, SOURCE_BITS


def observation(source="arm", timestamp=1.0, *, identifier=None, camera_xy=(0.025, 0.025),
                camera_yaw=0.0, distance=1.0, robot_xy=(0.0, 0.0)):
    cosine, sine = np.cos(camera_yaw), np.sin(camera_yaw)
    yaw = np.asarray([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
    optical = np.asarray([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    world_from_camera = np.eye(4)
    world_from_camera[:3, :3] = yaw @ optical
    world_from_camera[:3, 3] = [*camera_xy, 0.3]
    world_from_base = np.eye(4)
    world_from_base[:3, 3] = [*robot_xy, 0.3]
    robot = RobotState(timestamp, world_from_base, np.zeros(6), ("arm_joint",), [camera_yaw], [0])
    frame = CameraFrame(source, timestamp, np.zeros((1, 1, 3), dtype=np.uint8),
                        np.asarray([[distance]], dtype=np.float32), np.eye(3), world_from_camera)
    return RealObservation(identifier or f"{source}:{timestamp}", frame, robot)


class FusionTests(unittest.TestCase):
    def setUp(self):
        config = RGBDMapConfig(size_x_m=8, size_y_m=8, resolution_m=0.05, pixel_stride=1)
        self.map = RGBDPlanarMap(config, obstacle_min_height_m=0.05, obstacle_max_height_m=1.5)
        self.fusion = ObservationMapFusion(self.map)

    def integrate(self, *observations):
        return self.fusion.integrate(observations, floor_offset_m=0.3)

    def cell(self, x=1.025, y=0.025):
        row, column, inside = self.map.snapshot().grid.world_to_cell(x, y)
        self.assertTrue(inside)
        return row, column

    def test_default_fusion_uses_front_and_base_down(self):
        result = self.integrate(observation("front"), observation("arm"), observation("down"))
        self.assertEqual({row["source"] for row in result["observations"]}, {"front", "down"})
        self.assertEqual(self.map.source_mask[self.cell()], 3)
        self.assertEqual(self.fusion.source_cell_counts()["arm"], 0)

    def test_arm_only_builds_obstacle_and_records_source(self):
        self.fusion.set_sources(("arm",), timestamp_sec=0)
        result = self.integrate(observation())
        self.assertTrue(self.map.snapshot().grid.occupied_mask[self.cell()])
        self.assertEqual(self.map.source_mask[self.cell()], SOURCE_BITS["arm"])
        self.assertEqual(dict(result["source_ray_counts"]), {"arm": 1})
        self.assertGreater(self.fusion.source_cell_counts()["arm"], 0)

    def test_three_sources_share_map_with_independent_source_bits(self):
        self.fusion.set_sources(("front", "arm", "down"), timestamp_sec=0)
        self.integrate(*(observation(source) for source in ("front", "arm", "down")))
        self.assertEqual(self.map.source_mask[self.cell()], 7)
        self.assertEqual(int(self.map.snapshot().grid.occupied_mask.sum()), 1)

    def test_arm_obstacle_wins_against_free_ray_from_other_camera(self):
        self.fusion.set_sources(("front", "arm"), timestamp_sec=0)
        self.integrate(observation("front", distance=2), observation("arm", distance=1))
        self.assertTrue(self.map.snapshot().grid.occupied_mask[self.cell()])
        self.assertFalse(self.map.snapshot().grid.free_mask[self.cell()])

    def test_camera_rotation_translation_and_robot_motion_keep_world_landmark(self):
        self.fusion.set_sources(("front", "arm"), timestamp_sec=0)
        self.integrate(observation("front", camera_xy=(-0.025, 0.025), distance=1.05))
        result = self.integrate(observation("arm", timestamp=2, camera_xy=(1.025, -1.025),
                                            camera_yaw=np.pi / 2, distance=1.05, robot_xy=(0.3, 0.3)))
        self.assertEqual(int(self.map.snapshot().grid.occupied_mask.sum()), 1)
        self.assertEqual(self.map.source_mask[self.cell()], SOURCE_BITS["front"] | SOURCE_BITS["arm"])
        self.assertEqual(result["observations"][0]["timestamp_sec"], 2)

    def test_different_capture_times_use_each_observations_pose(self):
        self.fusion.set_sources(("front", "arm"), timestamp_sec=0)
        result = self.integrate(observation("arm", timestamp=2, camera_xy=(1.025, -1.025),
                                            camera_yaw=np.pi / 2, distance=1.05),
                                observation("front", timestamp=1, camera_xy=(-0.025, 0.025), distance=1.05))
        self.assertEqual([row["timestamp_sec"] for row in result["observations"]], [1, 2])
        self.assertEqual(int(self.map.snapshot().grid.occupied_mask.sum()), 1)

    def test_repeated_frame_does_not_reinforce_evidence(self):
        frame = observation("front")
        self.integrate(frame)
        before = self.map.log_odds.copy()
        version = self.map.version
        self.assertIsNone(self.integrate(frame))
        self.assertEqual(self.map.version, version)
        np.testing.assert_array_equal(self.map.log_odds, before)

    def test_repeated_frame_within_batch_is_fused_once(self):
        frame = observation("front")
        result = self.integrate(frame, frame)
        self.assertEqual(len(result["observations"]), 1)

    def test_separate_captures_at_same_time_are_individually_tracked(self):
        first = observation("front", identifier="first")
        second = observation("front", identifier="second")
        self.integrate(first)
        self.integrate(second)
        self.assertIsNone(self.integrate(first, second))

    def test_enabling_source_rejects_older_cached_capture(self):
        self.fusion.set_sources(("front", "arm"), timestamp_sec=2)
        self.assertIsNone(self.integrate(observation(timestamp=1.5)))
        self.assertIsNotNone(self.integrate(observation(timestamp=2.1)))

    def test_out_of_order_capture_is_not_fused(self):
        self.integrate(observation("front", timestamp=2))
        self.assertIsNone(self.integrate(observation("front", timestamp=1)))

    def test_disabling_source_keeps_history_and_stops_new_evidence(self):
        self.fusion.set_sources(("arm",), timestamp_sec=0)
        self.integrate(observation())
        before, version = self.map.source_mask.copy(), self.map.version
        self.fusion.set_sources((), timestamp_sec=2)
        self.assertIsNone(self.integrate(observation(timestamp=3, distance=2)))
        np.testing.assert_array_equal(self.map.source_mask, before)
        self.assertEqual(self.map.version, version)

    def test_no_sources_leave_map_unknown(self):
        self.fusion.set_sources((), timestamp_sec=0)
        self.assertIsNone(self.integrate(observation("front"), observation("arm"), observation("down")))
        self.assertEqual(self.map.version, 0)
        self.assertTrue(self.map.snapshot().grid.unknown_mask.all())

    def test_prediction_is_rejected_before_any_real_frame_is_fused(self):
        real = observation("front")
        predicted = PredictedObservation("prediction", "hypothesis", real.frame)
        with self.assertRaises(TypeError):
            self.integrate(real, predicted)
        self.assertEqual(self.map.version, 0)
        self.assertEqual(self.fusion.snapshot()["last"], {})

    def test_unknown_source_is_rejected(self):
        with self.assertRaises(ValueError):
            self.integrate(observation("other"))
        self.assertEqual(self.map.version, 0)

    def test_source_and_history_snapshot_restores_without_aliasing(self):
        self.fusion.set_sources(("front", "arm"), timestamp_sec=0)
        frame = observation()
        self.integrate(frame)
        saved = self.fusion.snapshot()
        self.fusion.set_sources(("down",), timestamp_sec=2)
        self.fusion.reset_history()
        self.fusion.restore(saved)
        self.assertEqual(self.fusion.sources, ("front", "arm"))
        self.assertIsNone(self.integrate(frame))
        self.fusion.reset_history()
        self.assertTrue(saved["last"])

    def test_reset_history_accepts_new_map_initial_frame(self):
        self.fusion.set_sources(("arm",), timestamp_sec=0)
        frame = observation()
        self.integrate(frame)
        self.map.reset([0, 0])
        self.fusion.reset_history()
        self.assertIsNotNone(self.integrate(frame))
        self.assertEqual(self.map.source_mask[self.cell()], SOURCE_BITS["arm"])

    def test_invalid_source_selection_leaves_state_unchanged(self):
        before = self.fusion.snapshot()
        for sources in (("arm", "arm"), ("other",), "arm"):
            with self.subTest(sources=sources), self.assertRaises(ValueError):
                self.fusion.set_sources(sources, timestamp_sec=2)
        self.assertEqual(self.fusion.snapshot(), before)


class FusionCommandTests(unittest.TestCase):
    def test_command_line_selects_three_sources(self):
        self.assertEqual(parse_args(["run", "--map-sources", "front", "arm", "down"]).map_sources,
                         ("front", "arm", "down"))

    def test_command_line_can_disable_all_sources(self):
        self.assertEqual(parse_args(["run", "--map-sources"]).map_sources, ())

    def test_command_line_rejects_duplicate_sources(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["run", "--map-sources", "arm", "arm"])


if __name__ == "__main__":
    unittest.main()
