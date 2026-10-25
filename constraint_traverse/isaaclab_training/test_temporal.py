from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from constraint_traverse.collect.recorder import quantize_depth_m
from constraint_traverse.collect.sampling import PhysicsState
from vln_nav.active_perception.local_geometry import LocalVoxelMap, LocalVoxelMapConfig

from .temporal import InvalidDepthFrame, MapSampler


class MockCamera:
    def __init__(self, timestamp, depth=None):
        self.frame = dict(rendering_time=float(timestamp), rgb=np.full((3, 3, 4), 127, dtype=np.uint8),
                          distance_to_image_plane=np.full((3, 3), 0.7004, dtype=np.float32) if depth is None else depth)

    def get_current_frame(self, clone=False):
        return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in self.frame.items()} if clone else self.frame

    def get_intrinsics_matrix(self):
        return np.array([[2., 0., 1.], [0., 2., 1.], [0., 0., 1.]])


def physics_state(timestamp, step, x=0.):
    position = np.array([x, 0., 0.32])
    front = np.eye(4)
    front[:3, 3] = position + np.array([0.1, 0., 0.1])
    down = np.eye(4)
    down[:3, 3] = position + np.array([-0.1, 0., 0.05])
    return PhysicsState(timestamp, step, position, np.array([1., 0., 0., 0.]),
        np.array([x, 0., 0.]), np.zeros(2), np.zeros(4), np.zeros(4), np.zeros(4), front, down)


def sampler(initial_time=0.0):
    local_map = LocalVoxelMap(LocalVoxelMapConfig(size_x_m=2., size_y_m=2.,
        min_z_m=-0.2, max_z_m=1.8, resolution_m=0.1, pixel_stride=1))
    local_map.reset(np.array([0., 0., 0.32]))
    local_map.restore(replace(local_map.snapshot(), timestamp_sec=initial_time))
    result = MapSampler(local_map, max_skew_sec=0.08, min_valid_ratio=0.5, integration_period_sec=0.25)
    result.reset(start_time_sec=max(0., initial_time), map_timestamp_sec=initial_time)
    return result


class TemporalTests(unittest.TestCase):
    def test_zero_and_one_step_delay_use_capture_pose(self):
        for delay in (0, 1):
            with self.subTest(delay=delay):
                item = sampler()
                captured = physics_state(1., 59, x=0.1)
                item.append_state(captured)
                now = 1. + delay / 59
                if delay:
                    item.append_state(physics_state(now, 60, x=0.5))
                front, down = MockCamera(1.), MockCamera(1.)
                with patch.object(item.local_map, "integrate_batch", wraps=item.local_map.integrate_batch) as integrate:
                    self.assertTrue(item.poll(front, down, now=now))
                    observations = integrate.call_args.args[0]
                    np.testing.assert_array_equal(observations[0].camera_to_world, captured.front_world_from_camera)
                    np.testing.assert_array_equal(observations[1].camera_to_world, captured.down_world_from_camera)
                    np.testing.assert_array_equal(integrate.call_args.kwargs["center_world_xyz"], captured.position)
                    np.testing.assert_array_equal(observations[0].depth_m, quantize_depth_m(front.frame["distance_to_image_plane"]))
                    self.assertEqual(observations[0].timestamp_sec, 1.)
                self.assertEqual(item.last_capture_sec, dict(front=1., down=1.))
                self.assertEqual(item.last_available_sec, dict(front=now, down=now))
                self.assertEqual(item.last_integration_sec, 1.)
                self.assertEqual(item.last_integration_available_sec, now)
                np.testing.assert_array_equal(item.valid_ratios, [1., 1.])
                self.assertFalse(item.poll(front, down, now=now + 1 / 59))
                self.assertEqual(item.fusion_count, 1)

    def test_reset_discards_history_and_previous_episode_frames(self):
        item = sampler()
        item.append_state(physics_state(1., 59))
        self.assertTrue(item.poll(MockCamera(1.), MockCamera(1.), now=1.))
        item.reset(start_time_sec=2., map_timestamp_sec=1.8)
        self.assertIsNone(item.history.at(1.))
        self.assertEqual(item.fusion_count, 0)
        np.testing.assert_array_equal(item.valid_ratios, [0., 0.])
        item.append_state(physics_state(1.99, 117))
        item.append_state(physics_state(2., 118))
        self.assertFalse(item.poll(MockCamera(1.99), MockCamera(1.99), now=2.))
        self.assertEqual(item.last_capture_sec, dict(front=-np.inf, down=-np.inf))
        self.assertFalse(item.poll(MockCamera(2.), MockCamera(2.), now=2.))
        item.append_state(physics_state(2.1, 124))
        self.assertTrue(item.poll(MockCamera(2.1), MockCamera(2.1), now=2.1))
        self.assertEqual(item.fusion_count, 1)

    def test_integration_period_uses_capture_time(self):
        item = sampler()
        fused = []
        for index in range(1, 10):
            step = 4 * index
            capture = step / 59
            now = (step + index % 2) / 59
            item.append_state(physics_state(capture, step))
            if now > capture:
                item.append_state(physics_state(now, step + 1))
            if item.poll(MockCamera(capture), MockCamera(capture), now=now):
                fused.append(capture)
        np.testing.assert_allclose(fused, [16 / 59, 32 / 59], atol=1e-12)
        self.assertEqual(item.fusion_count, 2)
        item.append_state(physics_state(0.78, 46))
        self.assertFalse(item.poll(MockCamera(0.78), MockCamera(0.78), now=1.2))
        self.assertAlmostEqual(item.last_integration_sec, 32 / 59)

    def test_pair_keeps_each_source_time_and_rejects_future(self):
        item = sampler()
        item.append_state(physics_state(1., 59))
        item.append_state(physics_state(1. + 1 / 59, 60, x=0.2))
        front, down = MockCamera(1.), MockCamera(1. + 1 / 59)
        self.assertFalse(item.poll(front, down, now=1.))
        self.assertEqual(item.last_capture_sec["down"], -np.inf)
        with patch.object(item.local_map, "integrate_batch", wraps=item.local_map.integrate_batch) as integrate:
            self.assertTrue(item.poll(front, down, now=1. + 1 / 59))
            observations = integrate.call_args.args[0]
            self.assertEqual(observations[0].timestamp_sec, 1.)
            self.assertEqual(observations[1].timestamp_sec, 1. + 1 / 59)
        self.assertEqual(item.last_available_sec["front"], 1.)
        self.assertEqual(item.last_available_sec["down"], 1. + 1 / 59)

    def test_insufficient_valid_depth_is_rejected(self):
        item = sampler()
        item.append_state(physics_state(1., 59))
        with self.assertRaisesRegex(RuntimeError, "有效深度比例不足"):
            item.poll(MockCamera(1., np.full((3, 3), np.nan)), MockCamera(1.), now=1.)
        self.assertEqual(item.fusion_count, 0)

    def test_invalid_pair_can_be_retried_at_same_physics_time(self):
        item = sampler()
        item.append_state(physics_state(0.5, 29))
        item.poll(MockCamera(0.5), MockCamera(0.5), now=0.5)
        version = item.local_map.snapshot().version
        item.append_state(physics_state(1.0, 59))
        front, down = MockCamera(1.0), MockCamera(1.0, np.zeros((3, 3), dtype=np.float32))
        with self.assertRaises(InvalidDepthFrame):
            item.poll(front, down, now=1.0)
        self.assertEqual(item.last_capture_sec, dict(front=0.5, down=0.5))
        self.assertEqual(item.local_map.snapshot().version, version)
        self.assertEqual(item.last_integration_sec, 0.5)
        down.frame["distance_to_image_plane"][:] = 0.7
        self.assertTrue(item.poll(front, down, now=1.0))
        self.assertEqual(item.last_capture_sec, dict(front=1.0, down=1.0))
        self.assertEqual(item.fusion_count, 2)

    def test_failed_pair_discards_partially_accepted_frames(self):
        item = sampler()
        item.append_state(physics_state(1.0, 59))
        front, down = MockCamera(1.0), MockCamera(1.01)
        self.assertFalse(item.poll(front, down, now=1.0))
        item.append_state(physics_state(1.01, 60))
        down.frame["distance_to_image_plane"][:] = 0
        with self.assertRaises(InvalidDepthFrame):
            item.poll(front, down, now=1.01)
        front.frame["rendering_time"] = 1.01
        down.frame["distance_to_image_plane"][:] = 0.7
        self.assertTrue(item.poll(front, down, now=1.01))


if __name__ == "__main__":
    unittest.main()
