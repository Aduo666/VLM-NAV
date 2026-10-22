from __future__ import annotations

import math
import unittest

import numpy as np

from constraint_traverse.reference_path import ReferencePath


def reference(points, headings=None, directions=None):
    metadata = dict(goal_world_xy=np.asarray(points[-1]), source="FINE", request_id=1,
                    created_sim_time_sec=0.0, map_version=3)
    if headings is None:
        return ReferencePath.from_positions(points, initial_yaw_rad=0, **metadata)
    return ReferencePath(np.asarray(points), np.asarray(headings), np.asarray(directions), **metadata)


class ReferenceTests(unittest.TestCase):
    def test_inputs_are_immutable_copies(self):
        points = np.asarray([[0., 0.], [1., 0.]])
        path = reference(points)
        points[1] = 5
        np.testing.assert_array_equal(path.world_xy[1], [1, 0])
        self.assertFalse(path.world_xy.flags.writeable)

    def test_ordered_lookahead_follows_turn(self):
        path = reference([[0, 0], [1, 0], [1, 1]])
        window = path.window([0.5, 0, 0], lookahead_m=1)
        np.testing.assert_allclose(window.world_xy, [[0.5, 0], [1, 0], [1, 0.5]])
        self.assertFalse(window.reaches_path_end)

    def test_direction_change_ends_window(self):
        path = reference([[0, 0], [1, 0], [0, 0]], [0, 0, 0], [0, 1, -1])
        window = path.window([0.5, 0, 0], lookahead_m=3)
        self.assertEqual(window.direction, 1)
        np.testing.assert_array_equal(window.world_xy[-1], [1, 0])
        self.assertEqual(path.window([1, 0, 0], lookahead_m=3).direction, -1)

    def test_spin_keeps_heading_and_blocks_later_translation(self):
        path = reference([[0, 0], [0, 0], [0, 1]], [0, math.pi / 2, math.pi / 2], [0, 0, 1])
        self.assertEqual(path.window([0, 0, 0], lookahead_m=2).direction, 0)
        self.assertEqual(path.window([0, 0, math.pi / 2], lookahead_m=2).direction, 1)

    def test_reference_end_is_separate_from_high_level_goal(self):
        path = reference([[0, 0], [0.2, 0]])
        self.assertIsNone(path.window([0.2, 0, 0], lookahead_m=1))

    def test_reverse_positions_preserve_body_heading(self):
        path = reference([[0, 0], [-0.2, 0]])
        self.assertEqual(path.motion_direction[1], -1)
        self.assertAlmostEqual(float(path.world_yaw_rad[-1] % (2 * math.pi)), 0)

    def test_position_match_does_not_skip_spin(self):
        path = reference([[0, 0], [0, 0], [1, 0]], [0, 1, 0], [0, 0, 1])
        self.assertEqual(path.window([0, 0, 0], lookahead_m=2).cursor, 1)

    def test_reuse_reference_until_lookahead_becomes_short(self):
        path = reference([[0, 0], [1.2, 0]])
        self.assertFalse(path.needs_extension([0, 0, 0], lookahead_m=1.2))
        self.assertTrue(path.needs_extension([0.8, 0, 0], lookahead_m=1.2))

    def test_map_signature_ignores_far_changes(self):
        from constraint_traverse.test_navigation import make_grid
        path = reference([[0, 0], [0.4, 0]])
        grid = make_grid()
        original = path.map_signature(grid, radius_m=0.3)
        grid.occupied_mask[0, 0] = True
        self.assertEqual(path.map_signature(grid, radius_m=0.3), original)
        grid.occupied_mask[40, 43] = True
        self.assertNotEqual(path.map_signature(grid, radius_m=0.3), original)


if __name__ == "__main__":
    unittest.main()
