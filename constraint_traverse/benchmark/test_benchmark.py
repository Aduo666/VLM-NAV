from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from vln_nav.runtime.agx_profile import load_agx_profile

from .geometry import DoorScene, PassageGeometry
from .model import FourWheelModel, MotionLimits, STATE_SIZE
from .planner import PassageCost, SamplingPlanner
from .record import TrialSnapshot
from .ui import MapRenderer


ROOT = Path(__file__).resolve().parents[2]


def setUpModule():
    torch.set_num_threads(1)


def settings():
    with (ROOT / "constraint_traverse/configs/benchmark.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def fixture():
    cfg = settings()
    hardware = load_agx_profile(project_root=ROOT).agx_controller.continuous
    model = FourWheelModel(hardware, MotionLimits(**cfg["motion"]), cfg["planner"]["dt_sec"])
    geometry = PassageGeometry([[-0.4, 0.4, -0.3, 0.3]], DoorScene(1.0))
    cost = PassageCost(geometry, cfg["cost"])
    cost.set_target([1.5, 0])
    return cfg, model, geometry, cost


class GeometryTests(unittest.TestCase):
    def test_map_goal_coordinate_roundtrip(self):
        _, _, geometry, _ = fixture()
        renderer = MapRenderer(geometry)
        for point in ([1.6, 0], [-1.2, 0.5], [0.6, -1.3]):
            column, row = renderer.world_to_pixel([point])[0]
            np.testing.assert_allclose(renderer.pixel_to_world(column, row), point, atol=1.0e-12)

    def test_subcell_overlap_detected(self):
        geometry = PassageGeometry([[-0.01, 0.01, -0.01, 0.01]], DoorScene(0.10), margin_m=0)
        self.assertFalse(geometry.validate_poses([[0, 0.045, 0]]).safe)

    def test_collision_between_endpoints(self):
        geometry = PassageGeometry([[-0.1, 0.1, -0.1, 0.1]], DoorScene(0.5), margin_m=0)
        self.assertFalse(geometry.validate_poses([[-0.5, 0.6, 0], [0.5, 0.6, 0]]).safe)

    def test_margin_and_physical_separate(self):
        geometry = PassageGeometry([[-0.1, 0.1, -0.1, 0.1]], DoorScene(0.24), margin_m=0.03)
        self.assertGreater(geometry.clearance([0, 0, 0]), 0)
        self.assertFalse(geometry.validate_poses([[0, 0, 0]]).safe)

    def test_tail_defines_passage(self):
        _, _, geometry, _ = fixture()
        self.assertFalse(geometry.completely_passed([0.3, 0, 0]))
        self.assertTrue(geometry.completely_passed([0.7, 0, 0]))
        self.assertFalse(geometry.completely_passed([0.3, 0, np.pi / 2]))

    def test_torch_collision_matches_shapely(self):
        _, _, geometry, _ = fixture()
        generator = np.random.default_rng(42)
        poses = generator.uniform([-1, -1, -np.pi], [1, 1, np.pi], (500, 3))
        gaps = geometry.torch_clearance(torch.tensor(poses, dtype=torch.float32)).numpy()
        for pose, gap in zip(poses, gaps):
            overlap = geometry.body_at(pose).intersects(geometry.obstacle_geometry)
            self.assertEqual(bool(gap <= 0), overlap)


class ModelTests(unittest.TestCase):
    def test_straight_and_reverse(self):
        _, model, _, _ = fixture()
        for direction in (-1, 1):
            result = model.rollout(torch.zeros(STATE_SIZE), torch.tensor([[direction * 0.3, 0.0]] * 20))
            self.assertGreater(direction * float(result[-1, 0]), 0.3)
            self.assertLess(abs(float(result[-1, 1])), 1.0e-5)

    def test_speed_change_bounded(self):
        _, model, _, _ = fixture()
        state = torch.zeros(STATE_SIZE)
        accelerating = model.step(state, [0.5, 0])
        self.assertLessEqual(float(accelerating[14]), model.limits.linear_accel_mps2 * model.dt_sec + 1.0e-6)
        braking = model.step(accelerating, [0, 0])
        self.assertLessEqual(float(accelerating[14] - braking[14]), model.limits.linear_decel_mps2 * model.dt_sec + 1.0e-6)

    def test_turning_and_spin(self):
        _, model, _, _ = fixture()
        for action in ([0.3, 0.3], [0, 0.3]):
            states = model.rollout(torch.zeros(STATE_SIZE), torch.tensor([action] * 20))
            self.assertGreater(float(states[-1, 2]), 0.1)
            self.assertFalse(bool(torch.any(states[:, 16] > 0.5)))

    def test_braking_holds_wheel_orientation(self):
        _, model, _, _ = fixture()
        state = torch.zeros(STATE_SIZE)
        state[6:10] = torch.tensor([0.2, 0.3, -0.2, -0.3])
        state[17:21] = state[6:10]
        stopped = model.step(state, [0, 0])
        torch.testing.assert_close(stopped[6:10], state[6:10])

    def test_motion_constraint_reported(self):
        _, model, _, _ = fixture()
        state = torch.zeros(STATE_SIZE)
        state[14:16] = torch.tensor([0.1, 0.5])
        result = model.step(state, [0.1, 0.5])
        self.assertEqual(float(result[16]), 1)


class PlannerTests(unittest.TestCase):
    def test_same_physical_cost(self):
        _, model, _, cost = fixture()
        controls = torch.tensor([[0.3, 0]] * 10)
        states = model.rollout(torch.zeros(STATE_SIZE), controls)[1:]
        expected = cost.running(states, controls).sum() + cost.terminal(states, controls)
        torch.testing.assert_close(cost.trajectory(states, controls), expected)

    def test_reset_seed_and_both_algorithms(self):
        cfg, model, _, cost = fixture()
        cfg["planner"].update(samples=32, horizon_steps=10, kernel_support_points=4, iterations=1)
        initial = np.zeros(STATE_SIZE, dtype=np.float32)
        initial[0] = -1.5
        for method in ("MPPI", "KMPPI"):
            first = SamplingPlanner(method, model, cost, cfg["planner"], cfg["safety"]).plan(initial)
            repeated = SamplingPlanner(method, model, cost, cfg["planner"], cfg["safety"]).plan(initial)
            self.assertTrue(first.valid, first.reason)
            np.testing.assert_array_equal(first.controls, repeated.controls)
            self.assertEqual(first.candidates.shape, (32, 10, 3))

    def test_unsafe_candidates_report_failure(self):
        cfg, model, geometry, cost = fixture()
        cfg["planner"].update(samples=16, horizon_steps=10, iterations=1)
        initial = np.zeros(STATE_SIZE, dtype=np.float32)
        initial[:2] = [0, 0.7]
        for method in ("MPPI", "KMPPI"):
            with self.subTest(method=method):
                result = SamplingPlanner(method, model, cost, cfg["planner"], cfg["safety"]).plan(initial)
                self.assertFalse(result.valid)
                self.assertEqual(result.reason, "NO_SAFE_CANDIDATE")
                np.testing.assert_array_equal(result.command, [0, 0])


class SnapshotTests(unittest.TestCase):
    def test_snapshot_arrays_are_independent(self):
        class Robot:
            def __init__(self):
                self.position = np.asarray([-1.5, 0, 0.3])
                self.orientation = np.asarray([1, 0, 0, 0])
                self.linear_velocity = np.asarray([0.1, 0, 0])
                self.angular_velocity = np.asarray([0, 0, 0.2])
                self.joint_positions = np.arange(8, dtype=float)
                self.joint_velocities = np.arange(8, dtype=float) * 0.1

            def get_world_pose(self):
                return self.position, self.orientation

            def get_linear_velocity(self):
                return self.linear_velocity

            def get_angular_velocity(self):
                return self.angular_velocity

            def get_joint_positions(self):
                return self.joint_positions

            def get_joint_velocities(self):
                return self.joint_velocities

            def set_world_pose(self, *, position, orientation):
                self.position, self.orientation = position.copy(), orientation.copy()

            def set_linear_velocity(self, value):
                self.linear_velocity = value.copy()

            def set_angular_velocity(self, value):
                self.angular_velocity = value.copy()

            def set_joint_positions(self, value):
                self.joint_positions = value.copy()

            def set_joint_velocities(self, value):
                self.joint_velocities = value.copy()

        robot = Robot()
        saved = TrialSnapshot.capture(robot, [0.1, 0.2], [1.5, 0], 42)
        robot.position[:] = 20
        robot.joint_positions[:] = 10
        saved.restore(robot)
        self.assertEqual(max(saved.errors(robot).values()), 0)
        self.assertEqual(saved.seed, 42)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
