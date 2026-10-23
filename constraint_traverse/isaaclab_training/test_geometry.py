from __future__ import annotations

import unittest

import numpy as np

from .geometry import DoorTracker


class GeometryTests(unittest.TestCase):
    def make_door(self, *, start=(-1, 0, 0), width=0.8):
        points = np.array([[-0.3, -0.2], [-0.3, 0.2], [0.3, -0.2], [0.3, 0.2]])
        annotation = dict(center_world_xy=[0, 0], normal_world_xy=[1, 0], width_m=width, thickness_m=0.2)
        return DoorTracker.create(annotation, points, start)

    def test_complete_body_exit(self):
        door = self.make_door()
        door.update([0.1, 0, 0])
        self.assertTrue(door.crossed)
        self.assertFalse(door.passed)
        door.update([0.5, 0, 0])
        self.assertTrue(door.passed)

    def test_retreat_and_bypass(self):
        door = self.make_door()
        for pose in ([0.05, 0, 0], [-0.6, 0, 0], [-0.6, 2, 0], [0.6, 2, 0]):
            door.update(pose)
        self.assertFalse(door.passed)
        self.assertFalse(door.crossed)

    def test_retreat_then_reenter(self):
        door = self.make_door()
        door.update([0.05, 0, 0])
        door.update([-0.6, 0, 0])
        self.assertFalse(door.crossed)
        self.assertTrue(door.armed)
        door.update([0.6, 0, 0])
        self.assertTrue(door.passed)

    def test_leaving_aperture_invalidates_attempt(self):
        door = self.make_door()
        for pose in ([0.05, 0, 0], [0.05, 2, 0], [0.6, 2, 0]):
            door.update(pose)
        self.assertFalse(door.passed)
        for pose in ([-0.6, 2, 0], [-0.6, 0, 0], [0.6, 0, 0]):
            door.update(pose)
        self.assertTrue(door.passed)

    def test_body_width_must_fit(self):
        door = self.make_door(width=0.3)
        door.update([0.6, 0, 0])
        self.assertFalse(door.passed)

    def test_initial_pose_inside_door_requires_entry(self):
        door = self.make_door(start=(0.05, 0, 0))
        door.update([0.6, 0, 0])
        self.assertFalse(door.passed)
        door.update([-0.6, 0, 0])
        door.update([0.6, 0, 0])
        self.assertTrue(door.passed)


if __name__ == "__main__":
    unittest.main()
