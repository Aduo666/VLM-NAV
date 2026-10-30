from __future__ import annotations

from pathlib import Path
import logging
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.ndimage import distance_transform_edt

from constraint_traverse.active_passage.traversal.controller import AutomaticTraversal, load_navigation_config
from constraint_traverse.active_passage.traversal.planner import PlanningFailure, SmacLatticePlanner
from constraint_traverse.benchmark.model import FourWheelModel, MotionLimits
from constraint_traverse.benchmark.run import load_config
from vln_nav.navigation.trajectory_clearance import FootprintConfig, FootprintPartConfig, PlanningGrid
from vln_nav.runtime.agx_profile import load_agx_profile
from vln_nav.control.trajectory import VelocityCommand


ROOT = Path(__file__).resolve().parents[3]


def grid(occupied=None, unknown=None):
    occupied = np.zeros((160, 160), dtype=bool) if occupied is None else occupied
    unknown = np.zeros_like(occupied) if unknown is None else unknown
    return PlanningGrid(occupied, ~(occupied | unknown), unknown,
        distance_transform_edt(~occupied).astype(np.float32) * 0.05, -4.0, -4.0, 0.05, 1)


class SmacTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings = load_navigation_config(ROOT / "constraint_traverse/configs/active_passage_navigation.yaml", ROOT)
        cls.footprint = FootprintConfig((FootprintPartConfig("body", -0.35, 0.35, -0.22, 0.22),), 0.03)
        cls.planner = SmacLatticePlanner(cls.footprint, settings["path"])

    def plan(self, mapping, start=(-1.5, 0, 0), goal=(1.5, 0)):
        return self.planner.plan(mapping, start, goal, request_id=3, timestamp=0)[0]

    def test_native_search_returns_position_goal_path(self):
        path = self.plan(grid())
        self.assertEqual(path.source, "SMAC_STATE_LATTICE")
        self.assertLess(np.linalg.norm(path.world_xy[-1] - [1.5, 0]), 0.11)
        self.assertTrue((path.motion_direction[1:] == 1).any())

    def test_offset_door_requires_detour(self):
        occupied = np.zeros((160, 160), dtype=bool)
        occupied[:, 79:82] = True
        occupied[66:95, 79:82] = False
        path = self.plan(grid(occupied), start=(-1.5, 1, 0), goal=(1.5, 1))
        crossing = np.abs(path.world_xy[:, 0]) < 0.2
        self.assertTrue(crossing.any())
        self.assertLess(np.max(np.abs(path.world_xy[crossing, 1])), 0.5)

    def test_unknown_start_is_rejected_without_changing_evidence(self):
        unknown = np.zeros((160, 160), dtype=bool)
        unknown[80, 50] = True
        mapping = grid(unknown=unknown)
        before = mapping.unknown_mask.copy()
        with self.assertRaises(PlanningFailure) as error:
            self.plan(mapping)
        self.assertEqual(error.exception.reason, "OBSERVATION_REQUIRED")
        np.testing.assert_array_equal(mapping.unknown_mask, before)

    def test_unknown_wall_is_impassable(self):
        unknown = np.zeros((160, 160), dtype=bool)
        unknown[:, 79:82] = True
        with self.assertRaises(PlanningFailure):
            self.plan(grid(unknown=unknown))

    def test_known_obstacle_start_is_rejected(self):
        occupied = np.zeros((160, 160), dtype=bool)
        occupied[80, 50] = True
        with self.assertRaises(PlanningFailure) as error:
            self.plan(grid(occupied))
        self.assertEqual(error.exception.reason, "KNOWN_OBSTACLE")

    def test_reverse_motion_supported(self):
        path = self.plan(grid(), start=(1.5, 0, 0), goal=(-1.5, 0))
        self.assertGreater(len(path.world_xy), 2)
        self.assertTrue(np.isin(path.motion_direction, [-1, 0, 1]).all())


class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = load_navigation_config(ROOT / "constraint_traverse/configs/active_passage_navigation.yaml", ROOT)
        hardware = load_agx_profile(project_root=ROOT).agx_controller.continuous
        config = load_config(ROOT / "constraint_traverse/configs/benchmark.yaml")
        cls.model = FourWheelModel(hardware, MotionLimits(**config["motion"]), 1 / 60)
        footprint = FootprintConfig((FootprintPartConfig("body", -0.35, 0.35, -0.22, 0.22),), 0.03)
        cls.validator = AutomaticTraversal.__new__(AutomaticTraversal)
        cls.validator.settings = cls.settings
        cls.validator.sampling = SimpleNamespace(execution_model=cls.model, config=config)
        cls.validator.path_planner = SmacLatticePlanner(footprint, cls.settings["path"])

    def test_executed_prefix_and_braking_must_be_known(self):
        state = np.zeros(17, dtype=np.float32)
        state[3] = state[14] = 0.4
        state[10:14] = 0.4
        unknown = np.zeros((160, 160), dtype=bool)
        unknown[:, 90:] = True
        with self.assertRaises(PlanningFailure) as error:
            self.validator.check_control(grid(unknown=unknown), state, [0.4, 0])
        self.assertEqual(error.exception.reason, "OBSERVATION_REQUIRED")

    def test_free_control_prefix_is_accepted(self):
        self.validator.check_control(grid(), np.zeros(17, dtype=np.float32), [0.3, 0])

    def test_zero_command_preserves_rest(self):
        state = np.zeros(17, dtype=np.float32)
        np.testing.assert_allclose(self.model.step(state, [0, 0]).numpy(), state, atol=1e-6)

    def controller(self, state):
        manager = AutomaticTraversal.__new__(AutomaticTraversal)
        manager.settings, manager.logger = self.settings, logging.getLogger("test")
        manager.active, manager.status = True, "TRACKING"
        manager.goal = np.asarray([0.1, 0])
        manager.actual, manager.events = [], []
        manager.stopped_since, manager.busy_since, manager.seen_plan = None, None, None
        manager.path_planner = self.validator.path_planner
        manager.sampling = SimpleNamespace(execution_model=self.model, config=self.validator.sampling.config,
            read_state=lambda robot, drive: state,
            apply_command=lambda **kwargs: VelocityCommand(*map(float, kwargs["command"])))
        return manager

    def test_arrival_requires_actual_stop_and_hold(self):
        state = np.zeros(17, dtype=np.float32)
        state[3] = 0.2
        manager = self.controller(state)
        manager.step(robot=None, drive=None, grid=grid(), now=0)
        self.assertTrue(manager.active)
        self.assertEqual(manager.status, "STOPPING_AT_GOAL")
        state[3] = 0
        manager.step(robot=None, drive=None, grid=grid(), now=0.1)
        self.assertTrue(manager.active)
        manager.step(robot=None, drive=None, grid=grid(), now=0.5)
        self.assertFalse(manager.active)
        self.assertEqual(manager.status, "SUCCESS")

    def test_received_plan_starts_new_response_timeout(self):
        manager = self.controller(np.zeros(17, dtype=np.float32))
        manager.goal = np.asarray([2, 0])
        manager.path = SimpleNamespace(request_id=7)
        manager._update_path = lambda *args: None
        manager.busy_since = 0
        manager.sampling.last_plan = SimpleNamespace(diagnostics={"request_timestamp_sec": 0})
        manager.sampling.status = "TRACKING"
        manager.sampling.requested = np.asarray([0.2, 0])
        manager.sampling.step = lambda **kwargs: None
        manager.sampling._worker = SimpleNamespace(busy=True)
        with patch("constraint_traverse.active_passage.traversal.controller.time.monotonic", return_value=30):
            manager.step(robot=None, drive=None, grid=grid(), now=0)
        self.assertTrue(manager.active)
        self.assertEqual(manager.status, "TRACKING")
        self.assertEqual(manager.busy_since, 30)


if __name__ == "__main__":
    unittest.main()
