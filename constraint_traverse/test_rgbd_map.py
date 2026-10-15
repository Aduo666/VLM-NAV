from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from constraint_traverse.rgbd_map import RGBDMapConfig, RGBDPlanarMap
from vln_nav.active_perception.bev_goal_map import render_planning_goal_map
from vln_nav.active_perception.contracts import TimedDepthObservation


def observation(*, source="front", wall=False, floor=False, x=0.0, height=0.30, wall_distance=1.0):
    depth = np.full((5, 5), np.nan, dtype=np.float32)
    if wall:
        depth[2, 2] = wall_distance
    if floor:
        depth[3, 2] = 3.0
    intrinsics = np.asarray([[10, 0, 2], [0, 10, 2], [0, 0, 1]], dtype=float)
    transform = np.asarray([[0, 0, 1, x], [-1, 0, 0, 0], [0, -1, 0, height], [0, 0, 0, 1]], dtype=float)
    return TimedDepthObservation(source, 1.0, depth, intrinsics, transform)


class DirectMapTests(unittest.TestCase):
    def setUp(self):
        self.config = RGBDMapConfig(size_x_m=8, size_y_m=8, resolution_m=0.10, pixel_stride=1)
        self.map = RGBDPlanarMap(self.config, obstacle_min_height_m=0.05, obstacle_max_height_m=1.50)

    def integrate(self, *frames, center=(0, 0)):
        self.map.integrate_batch(list(frames), center_world_xy=center, floor_z_m=0.0)
        return self.map.snapshot().grid

    def cell(self, grid, x, y=0):
        row, column, inside = grid.world_to_cell(x, y)
        self.assertTrue(inside)
        return row, column

    def test_floor_ray_marks_free_without_obstacle(self):
        grid = self.integrate(observation(floor=True))
        self.assertTrue(grid.free_mask[self.cell(grid, 1)])
        self.assertTrue(grid.free_mask[self.cell(grid, 3)])
        self.assertFalse(grid.occupied_mask.any())
        self.assertTrue(grid.unknown_mask[self.cell(grid, 3.5)])
        self.assertEqual(self.map.log_odds.ndim, 2)

    def test_nearest_obstacle_stops_floor_ray(self):
        grid = self.integrate(observation(wall=True, floor=True))
        self.assertTrue(grid.occupied_mask[self.cell(grid, 1)])
        self.assertTrue(grid.free_mask[self.cell(grid, 0.5)])
        self.assertTrue(grid.unknown_mask[self.cell(grid, 2)])

    def test_source_fusion_obstacle_wins(self):
        grid = self.integrate(observation(wall=True), observation(source="down", floor=True))
        snapshot = self.map.snapshot()
        self.assertTrue(grid.occupied_mask[self.cell(grid, 1)])
        self.assertEqual(snapshot.source_mask[self.cell(grid, 1)], 1)
        self.assertTrue(grid.unknown_mask[self.cell(grid, 2)])
        self.assertEqual(snapshot.source_mask[self.cell(grid, 0.5)], 3)

    def test_down_camera_keeps_obstacles_in_same_column(self):
        depth = np.full((50, 50), np.nan, dtype=np.float32)
        depth[15, 45], depth[35, 45] = 0.4, 0.8
        intrinsics = np.asarray([[20, 0, 25], [0, 20, 25], [0, 0, 1]], dtype=float)
        transform = np.diag([1.0, -1.0, -1.0, 1.0])
        transform[2, 3] = 1.0
        frame = TimedDepthObservation("down", 1.0, depth, intrinsics, transform)
        grid = self.integrate(frame)
        self.assertTrue(grid.occupied_mask[self.cell(grid, 0.400001, 0.200001)])
        self.assertTrue(grid.occupied_mask[self.cell(grid, 0.800001, -0.400001)])

    def test_invalid_depth_retains_unknown(self):
        frame = observation()
        frame.depth_m[0, :] = [0, -1, np.inf, np.nan, 10]
        grid = self.integrate(frame)
        self.assertTrue(grid.unknown_mask.all())
        self.assertFalse(self.map.source_mask.any())

    def test_points_above_obstacle_band_not_projected(self):
        grid = self.integrate(observation(wall=True, height=2.0))
        self.assertTrue(grid.unknown_mask.all())

    def test_clipped_obstacle_does_not_create_endpoint_hit(self):
        self.map = RGBDPlanarMap(replace(self.config, max_ray_length_m=2),
                                 obstacle_min_height_m=0.05, obstacle_max_height_m=1.50)
        grid = self.integrate(observation(wall=True, wall_distance=4))
        self.assertFalse(grid.occupied_mask.any())
        self.assertTrue(grid.free_mask[self.cell(grid, 2)])
        self.assertTrue(grid.unknown_mask[self.cell(grid, 3)])

    def test_camera_translation_in_world_coordinates(self):
        grid = self.integrate(observation(wall=True, x=0.4))
        self.assertTrue(grid.occupied_mask[self.cell(grid, 1.4)])
        self.assertTrue(grid.free_mask[self.cell(grid, 1)])

    def test_sliding_preserves_world_evidence(self):
        initial = self.integrate(observation(wall=True))
        shifted = self.integrate(observation(), center=(0.5, 0.3))
        self.assertNotEqual(initial.origin_x_m, shifted.origin_x_m)
        self.assertTrue(shifted.occupied_mask[self.cell(shifted, 1)])
        self.assertTrue(shifted.unknown_mask[:, -1].all())
        self.assertEqual(self.map.source_mask[self.cell(shifted, 1)], 1)

    def test_snapshot_independent_and_reset(self):
        grid = self.integrate(observation(wall=True))
        snapshot = self.map.snapshot()
        self.map.reset([0, 0])
        self.assertTrue(grid.occupied_mask.any())
        self.assertTrue(snapshot.source_mask.any())
        self.assertTrue(self.map.snapshot().grid.unknown_mask.all())

    def test_map_display_and_click_coordinates(self):
        grid = self.integrate(observation(floor=True))
        result = render_planning_goal_map(
            grid, source_mask=self.map.source_mask, robot_world_xyz=[0, 0, 0.3],
            robot_yaw_rad=0, view_size_m=4, rear_view_m=1, image_size_px=200,
            goal_local_xy=(1, 0),
        )
        self.assertEqual(result.rgba.shape, (200, 200, 4))
        self.assertTrue(result.selectable_mask[99, 99])
        np.testing.assert_allclose(result.pixel_to_local(99, 99), [1.01, 0.01])


if __name__ == "__main__":
    unittest.main()
