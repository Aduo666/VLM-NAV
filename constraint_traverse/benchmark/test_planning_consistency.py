from __future__ import annotations

import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np
import torch

from constraint_traverse.navigation import PlanningGridGeometry
from vln_nav.navigation.trajectory_clearance import FootprintConfig, FootprintPartConfig, PlanningGrid
from .model import STATE_SIZE
from .geometry import sample_body_points
from .planner import PassageCost, SamplingPlanner
from .test_benchmark import fixture
from constraint_traverse.reference_path import ReferencePath


def setUpModule():
    torch.set_num_threads(1)


class ConsistencyTests(unittest.TestCase):
    def reference_cost(self, points, headings=None, directions=None):
        from constraint_traverse.test_navigation import make_grid, make_geometry
        cfg, model, _, cost = fixture()
        cost.geometry = make_geometry()
        cost.geometry.update_grid(make_grid())
        metadata = dict(goal_world_xy=np.asarray(points[-1]), source="FINE", request_id=1,
                        created_sim_time_sec=0, map_version=1)
        path = (ReferencePath.from_positions(points, initial_yaw_rad=0, **metadata) if headings is None
                else ReferencePath(np.asarray(points), np.asarray(headings), np.asarray(directions), **metadata))
        window = path.window([0, 0, 0], lookahead_m=3)
        cost.set_target(window.world_xy[-1])
        cost.set_reference_window(window)
        return cfg, model, cost

    def test_route_progress_rewards_required_detour(self):
        _, _, cost = self.reference_cost([[0, 0], [-0.4, 0], [-0.4, 0.8], [0.8, 0.8]])
        states = torch.zeros(2, STATE_SIZE)
        states[1, 0] = -0.3
        components = cost.components(states, torch.zeros(2, 2))
        self.assertLess(float(components["goal"][1]), float(components["goal"][0]))

    def test_spin_reference_has_heading_objective(self):
        _, _, cost = self.reference_cost([[0, 0], [0, 0]], [0, np.pi / 2], [0, 0])
        states = torch.zeros(2, 1, STATE_SIZE)
        states[1, 0, 2] = np.pi / 2
        terminal = cost.terminal(states, torch.zeros(2, 1, 2))
        self.assertGreater(float(terminal[0]), float(terminal[1]))

    def test_reference_commands_reverse_and_spin(self):
        for method in ("MPPI", "KMPPI"):
            for spin in (False, True):
                cfg, model, cost = (self.reference_cost([[0, 0], [0, 0]], [0, np.pi / 2], [0, 0]) if spin
                                    else self.reference_cost([[0, 0], [-0.4, 0]]))
                planner = SamplingPlanner(method, model, cost, cfg["planner"], cfg["safety"])
                result = planner.plan(np.zeros(STATE_SIZE, dtype=np.float32))
                if spin:
                    self.assertGreater(float(result.command[1]), 0.02, (method, result.command))
                else:
                    self.assertLess(float(result.command[0]), -0.005, (method, result.command))

    def test_reference_direction_is_enforced_over_full_horizon(self):
        for method in ("MPPI", "KMPPI"):
            for direction in (-1, 1):
                cfg, model, cost = self.reference_cost([[0, 0], [direction * 1.0, 0]])
                planner = SamplingPlanner(method, model, cost, cfg["planner"], cfg["safety"])
                result = planner.plan(np.zeros(STATE_SIZE, dtype=np.float32))
                self.assertTrue(result.valid)
                self.assertTrue((result.controls[:, 0] * direction >= 0).all())
                self.assertTrue((result.candidate_controls[..., 0] * direction >= 0).all())
                self.assertTrue((np.abs(result.controls[:, 1]) <= model.max_curvature * np.abs(result.controls[:, 0]) + 1e-6).all())
                self.assertGreaterEqual(result.diagnostics["ess"], result.diagnostics["safe_candidates"] * 0.1)

    def test_steering_target_and_actual_response_are_separate(self):
        _, model, _, _ = fixture()
        state = torch.zeros(STATE_SIZE)
        after = model.step(state, [0, 0.3], 1 / 60)
        self.assertGreater(float(after[17:21].abs().max()), float(after[6:10].abs().max()))
        self.assertLessEqual(float(after[17:21].abs().max()), model.limits.steering_rate_radps / 60 + 1e-6)
        self.assertEqual(float(after[10:14].abs().max()), 0)

    def test_wheel_overspeed_does_not_raise_velocity_target(self):
        _, model, _, _ = fixture()
        state = torch.zeros(STATE_SIZE)
        state[3] = 0.8
        state[10:14] = 0.8
        state[14] = 0.3
        state[21:25] = 0.3
        rims, _, _ = model.actuator_targets(state, [0.3, 0])
        torch.testing.assert_close(rims, torch.full((4,), 0.3))

    def test_correlated_noise_uses_matching_covariance(self):
        cfg, model, _, cost = fixture()
        planner = SamplingPlanner("MPPI", model, cost, cfg["planner"], cfg["safety"])
        controller = planner.controller
        noise = torch.linspace(-0.2, 0.2, controller.T * 2).reshape(1, controller.T, 2)
        index = torch.arange(controller.T)
        covariance = controller.noise_correlation ** torch.abs(index[:, None] - index[None, :])
        expected = controller.lambda_ * torch.einsum("ij,bjk,kl->bil", torch.linalg.inv(covariance), noise,
                                                    controller.noise_sigma_inv)
        torch.testing.assert_close(controller._compute_action_cost(noise), expected, atol=0.003, rtol=1e-4)

    def test_warm_start_uses_prediction_time_interval(self):
        from constraint_traverse.navigation import _PlanningWorker, _SamplingJob
        from constraint_traverse.test_navigation import make_grid
        cfg, model, _, _ = fixture()
        owner = SimpleNamespace(geometry=Mock(), config=cfg, physics_dt=1 / 60)
        worker = _PlanningWorker(owner)
        worker.model, worker.cost, worker.planner = model, Mock(), Mock()
        worker.goal, worker.epoch, worker.previous_time = np.asarray([1, 0]), 1, 1.0
        worker.planner.plan.return_value = SimpleNamespace(diagnostics={})
        job = _SamplingJob(1, np.zeros(STATE_SIZE, dtype=np.float32), make_grid(), np.asarray([1, 0]), 1.2, None,
                           np.zeros((6, 2), dtype=np.float32), 7.0)
        worker._run(job)
        self.assertIsNone(worker.results.get_nowait()[4])
        self.assertAlmostEqual(worker.planner.plan.call_args.kwargs["elapsed_sec"], 0.3)
        self.assertAlmostEqual(worker.previous_time, 1.3)

    def test_reference_guidance_tracks_s_curve_in_prediction_model(self):
        angle = np.linspace(0, np.pi / 2, 41)
        radius = 0.5
        first = np.column_stack((radius * np.sin(angle), radius * (1 - np.cos(angle))))
        second = np.column_stack((radius + radius * (1 - np.cos(angle[1:])),
                                  radius + radius * np.sin(angle[1:])))
        points = np.vstack((first, second))
        headings = np.r_[angle, np.pi / 2 - angle[1:]]
        cfg, model, cost = self.reference_cost(points, headings, np.r_[0, np.ones(len(points) - 1)])
        planner = SamplingPlanner("MPPI", model, cost, cfg["planner"], cfg["safety"])
        initial = torch.zeros(STATE_SIZE)
        planner._prepare_sampling(initial, None)
        controls = planner.controller.get_action_sequence()
        states = model.rollout(initial, controls)
        _, errors, _ = cost.route_errors(states)
        self.assertLess(float(errors.sqrt().max()), 0.04)
        self.assertTrue(bool((controls[:, 0] >= 0).all()))

    def test_narrow_reference_sets_attainable_soft_clearance(self):
        from constraint_traverse.test_navigation import make_grid
        cfg, _, _, _ = fixture()
        geometry = PlanningGridGeometry(FootprintConfig((FootprintPartConfig("body", -0.4, 0.4, -0.3, 0.3),), 0.03))
        grid = make_grid()
        rows, _ = np.indices(grid.shape)
        y = grid.origin_y_m + (rows + 0.5) * grid.resolution_m
        grid.occupied_mask[:] = abs(y) > 0.36
        grid.free_mask[:] = ~grid.occupied_mask
        geometry.update_grid(grid)
        cost = PassageCost(geometry, cfg["cost"])
        path = ReferencePath.from_positions([[0, 0], [0.6, 0]], initial_yaw_rad=0,
            goal_world_xy=np.asarray([0.6, 0]), source="FINE", request_id=1, created_sim_time_sec=0, map_version=1)
        cost.set_target([0.6, 0])
        cost.set_reference_window(path.window([0, 0, 0], lookahead_m=1.2))
        self.assertLess(cost.reference_clearance_target, 0.10)
        self.assertGreaterEqual(cost.reference_clearance_target, geometry.margin_m)

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
            controller.valid_candidates = torch.ones_like(costs, dtype=torch.bool)
            controller.weight_temperature = controller.lambda_
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
