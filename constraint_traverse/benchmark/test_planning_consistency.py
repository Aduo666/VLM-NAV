from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from constraint_traverse.navigation import PlanningGridGeometry
from vln_nav.navigation.trajectory_clearance import FootprintConfig, FootprintPartConfig, PlanningGrid
from .model import STATE_SIZE
from .geometry import sample_body_points
from .planner import PassageCost, SamplingPlanner
from .test_benchmark import fixture


def setUpModule():
    torch.set_num_threads(1)


class ConsistencyTests(unittest.TestCase):
    def test_prediction_equals_physical_substeps(self):
        _, model, _, _ = fixture()
        for action in ([0.5, 0], [0.3, 0.3], [0, 0.3], [-0.3, 0]):
            initial = torch.zeros(STATE_SIZE)
            expected = initial
            for _ in range(12):
                expected = model.step(expected, action, 1 / 60)
            actual = model.step(initial, action, 0.2)
            torch.testing.assert_close(actual, expected, atol=1.0e-6, rtol=1.0e-6)

    def test_braking_prediction_equals_physical_substeps(self):
        _, model, _, _ = fixture()
        initial = model.rollout(torch.zeros(STATE_SIZE), [[0.3, 0]] * 5)[-1]
        expected = initial
        for _ in range(12):
            expected = model.step(expected, [0, 0], 1 / 60)
        actual = model.step(initial, [0, 0], 0.2)
        torch.testing.assert_close(actual, expected, atol=1.0e-6, rtol=1.0e-6)

    def test_null_candidate_updates_null_support_points(self):
        cfg, model, _, cost = fixture()
        planner = SamplingPlanner("KMPPI", model, cost, cfg["planner"], cfg["safety"])
        controller = planner.controller

        def select_null(costs):
            controller.omega = torch.zeros_like(costs)
            controller.omega[0] = 1
            return controller.omega

        initial = torch.zeros(STATE_SIZE)
        initial[0] = -1.5
        with patch.object(controller, "_compute_weighting", side_effect=select_null):
            result = planner.plan(initial)
        np.testing.assert_array_equal(result.candidate_controls[0], 0)
        np.testing.assert_allclose(result.controls, 0, atol=1.0e-7)
        np.testing.assert_allclose(result.command, 0, atol=1.0e-7)

    def test_wall_candidates_can_adjust_without_collision(self):
        for method in ("MPPI", "KMPPI"):
            cfg, model, _, cost = fixture()
            cost.set_target([1.6, 0.9])
            state = np.zeros(STATE_SIZE, dtype=np.float32)
            state[:2] = [-0.6, 0.9]
            planner = SamplingPlanner(method, model, cost, cfg["planner"], cfg["safety"])
            result = planner.plan(state)
            self.assertEqual(result.diagnostics["candidate_motion_violation_fraction"], 0)
            self.assertGreaterEqual(result.min_clearance_m, 0)

    def test_near_goal_commands_make_forward_progress(self):
        from constraint_traverse.test_navigation import make_grid, make_geometry
        for method in ("MPPI", "KMPPI"):
            for distance in (0.40, 0.23, 0.15):
                cfg, model, _, cost = fixture()
                cost.geometry = make_geometry()
                cost.geometry.update_grid(make_grid())
                cost.set_target([distance, 0])
                state = np.zeros(STATE_SIZE, dtype=np.float32)
                planner = SamplingPlanner(method, model, cost, cfg["planner"], cfg["safety"])
                result = planner.plan(state)
                self.assertGreater(float(result.command[0]), 0.005, (method, distance, result.command))
                self.assertEqual(result.diagnostics["candidate_motion_violation_fraction"], 0)

    def test_kmppi_bounds_and_rolling_consistency(self):
        cfg, model, _, cost = fixture()
        planner = SamplingPlanner("KMPPI", model, cost, cfg["planner"], cfg["safety"])
        controller = planner.controller
        controller.theta[:, 0] = torch.tensor([0.5, -0.5, 0.5, 0.5, -0.5, 0.5])
        controller.synchronize_controls()
        self.assertGreater(float(controller.interpolation_excess), 0.2)
        for _ in range(5):
            reconstructed, _ = controller.deparameterize_to_trajectory_single(controller.theta)
            torch.testing.assert_close(controller.U, controller._bound_action(reconstructed))
            self.assertLessEqual(float(controller.U.abs().max()), 0.5)
            controller.shift_nominal_trajectory()

    def test_obstacle_penetration_cost_increases(self):
        cfg, _, _, _ = fixture()
        geometry = PlanningGridGeometry(FootprintConfig((FootprintPartConfig("body", -0.4, 0.4, -0.3, 0.3),), 0.03))
        rows, columns = np.indices((100, 100))
        x = -2.5 + (columns + 0.5) * 0.05
        occupied = (x >= 0.5) & (x <= 1.5)
        grid = PlanningGrid(occupied, ~occupied, np.zeros_like(occupied), np.zeros_like(x), -2.5, -2.5, 0.05, 1)
        geometry.update_grid(grid)
        cost = PassageCost(geometry, cfg["cost"])
        cost.set_target([2, 0])
        states, actions = torch.zeros(2, STATE_SIZE), torch.zeros(2, 2)
        states[:, 0] = torch.tensor([0.11, 1])
        terms = cost.components(states, actions)
        self.assertGreater(float(terms["penetration"][1]), float(terms["penetration"][0]))
        self.assertGreater(float(terms["occupied"][1]), float(terms["occupied"][0]))
        self.assertGreater(float(cost.running(states, actions)[1]), float(cost.running(states, actions)[0]))

    def test_cost_uses_intermediate_motion(self):
        cfg, model, _, cost = fixture()
        state = torch.zeros(1, STATE_SIZE)
        state[0, :2] = torch.tensor([-1.0, 0.9])
        intermediate = torch.stack((state.clone(), state.clone()))
        intermediate[0, 0, 0] = 0
        endpoint_cost = cost.running(state, torch.zeros(1, 2))
        motion_cost = cost.running(state, torch.zeros(1, 2), motion_states=intermediate)
        self.assertGreater(float(motion_cost[0]), float(endpoint_cost[0]))

    def test_overlap_optimization_preserves_area_metrics(self):
        _, _, geometry, _ = fixture()
        rng = np.random.default_rng(31)
        poses = torch.tensor(rng.uniform([-3.3, -3.3, -np.pi], [3.3, 3.3, np.pi], (10, 10, 3)), dtype=torch.float32)
        points = torch.as_tensor(sample_body_points(geometry.parts, 0.05))
        obstacles = torch.tensor(geometry.scene.obstacles, dtype=torch.float32)
        c, s = poses[..., 2, None].cos(), poses[..., 2, None].sin()
        x = poses[..., 0, None] + c * points[:, 0] - s * points[:, 1]
        y = poses[..., 1, None] + s * points[:, 0] + c * points[:, 1]
        occupied = ((x[..., None] >= obstacles[:, 0]) & (x[..., None] <= obstacles[:, 1])
                    & (y[..., None] >= obstacles[:, 2]) & (y[..., None] <= obstacles[:, 3])).any(dim=-1)
        outside = (x.abs() > geometry.scene.half_size_m) | (y.abs() > geometry.scene.half_size_m)
        metrics = geometry.torch_metrics(poses)
        torch.testing.assert_close(metrics["occupied_fraction"], occupied.float().mean(dim=-1))
        torch.testing.assert_close(metrics["outside_fraction"], outside.float().mean(dim=-1))


if __name__ == "__main__":
    unittest.main()
