from __future__ import annotations

from dataclasses import replace
import io
import logging
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from constraint_traverse.benchmark.run import load_config
from constraint_traverse.navigation import PlanningGridGeometry, SamplingNavigationController
from vln_nav.app import navigation
from vln_nav.navigation.trajectory_clearance import FootprintConfig, FootprintPartConfig, PlanningGrid
from vln_nav.runtime.agx_profile import load_agx_profile


ROOT = Path(__file__).resolve().parents[1]


def setUpModule():
    torch.set_num_threads(1)


def make_grid(*, origin=(-2.0, -2.0), resolution=0.05, known=True, version=1):
    shape = (80, 80)
    return PlanningGrid(np.zeros(shape, bool), np.full(shape, known), np.full(shape, not known),
                        np.full(shape, 4, dtype=np.float32), *origin, resolution, version)


def make_geometry():
    footprint = FootprintConfig((FootprintPartConfig("body", -0.15, 0.15, -0.1, 0.1),), 0.03)
    return PlanningGridGeometry(footprint, 0.01)


class GridTests(unittest.TestCase):
    def test_free_grid(self):
        geometry = make_geometry()
        geometry.update_grid(make_grid())
        self.assertTrue(geometry.validate_poses([[0, 0, 0], [0.5, 0, 0]]).safe)

    def test_unknown_blocked_and_visualized(self):
        geometry = make_geometry()
        grid = make_grid()
        grid.unknown_mask[40, 49] = True
        grid.free_mask[40, 49] = False
        geometry.update_grid(grid)
        result = geometry.validate_poses([[0, 0, 0], [0.5, 0, 0]])
        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "UNKNOWN")
        self.assertGreater(len(geometry.rejected_unknown_world_xy), 0)

    def test_occupied_between_endpoints(self):
        geometry = make_geometry()
        grid = make_grid()
        grid.occupied_mask[40, 47] = True
        grid.free_mask[40, 47] = False
        geometry.update_grid(grid)
        self.assertEqual(geometry.validate_poses([[0, 0, 0], [0.7, 0, 0]]).reason, "OBSTACLE")

    def test_grid_corner_overlap_detected(self):
        footprint = FootprintConfig((FootprintPartConfig("tiny", -0.01, 0.01, -0.01, 0.01),), 0)
        geometry = PlanningGridGeometry(footprint, 0.01)
        grid = make_grid()
        grid.occupied_mask[40, 41] = True
        grid.free_mask[40, 41] = False
        geometry.update_grid(grid)
        self.assertFalse(geometry.validate_poses([[0.045, 0.025, 0]]).safe)

    def test_shifted_map_world_coordinates(self):
        geometry = make_geometry()
        geometry.update_grid(make_grid(origin=(10, 20)))
        self.assertTrue(geometry.validate_poses([[11, 21, 0]]).safe)
        self.assertEqual(geometry.validate_poses([[0, 0, 0]]).reason, "OUTSIDE_MAP")

    def test_latest_map_invalidates_cache(self):
        geometry = make_geometry()
        first = make_grid(version=1)
        geometry.update_grid(first)
        self.assertTrue(geometry.validate_poses([[0.5, 0, 0]]).safe)
        second = make_grid(version=2)
        second.unknown_mask[40, 49] = True
        second.free_mask[40, 49] = False
        geometry.update_grid(second)
        self.assertFalse(geometry.validate_poses([[0.5, 0, 0]]).safe)

    def test_cost_field_unknown_and_outside(self):
        geometry = make_geometry()
        geometry.update_grid(make_grid(known=False))
        gaps = geometry.torch_clearance(torch.tensor([[0, 0, 0], [10, 10, 0]], dtype=torch.float32))
        self.assertTrue(bool(torch.all(gaps < 0)))


class ModeTests(unittest.TestCase):
    def test_all_modes_valid_in_configuration(self):
        with navigation.SETTINGS_PATH.open(encoding="utf-8") as handle:
            settings = yaml.safe_load(handle)
        for mode in navigation.LOW_LEVEL_MODES:
            settings["execution"]["default_mode"] = mode
            with patch.object(Path, "open", return_value=io.StringIO(yaml.safe_dump(settings))):
                result = navigation._load_settings()
            self.assertEqual(result[0]["execution"]["default_mode"], mode)

    def test_invalid_mode_rejected(self):
        with navigation.SETTINGS_PATH.open(encoding="utf-8") as handle:
            settings = yaml.safe_load(handle)
        settings["execution"]["default_mode"] = "BAD"
        with patch.object(Path, "open", return_value=io.StringIO(yaml.safe_dump(settings))):
            with self.assertRaises(ValueError):
                navigation._load_settings()


class ControllerTests(unittest.TestCase):
    def test_both_methods_can_command_free_space(self):
        hardware = load_agx_profile(project_root=ROOT).agx_controller.continuous

        class Robot:
            def get_world_pose(self):
                return np.asarray([0, 0, 0.3]), np.asarray([1, 0, 0, 0])

            def get_linear_velocity(self):
                return np.zeros(3)

            def get_angular_velocity(self):
                return np.zeros(3)

        class Drive:
            def readback(self):
                return {"steering_position": {w.steering_joint: 0 for w in hardware.wheels},
                        "wheel_velocity": {w.drive_joint: 0 for w in hardware.wheels}}

            def apply_continuous_joint_targets(self, motors, steering):
                self.motors, self.steering = motors, steering

        footprint = FootprintConfig((FootprintPartConfig("body", -0.15, 0.15, -0.1, 0.1),), 0.03)
        for method in navigation.SAMPLING_MODES:
            controller = SamplingNavigationController(method=method, hardware=hardware, footprint=footprint, physics_dt=1 / 60,
                                                       config_path=ROOT / "constraint_traverse/configs/benchmark.yaml", logger=logging.getLogger("test"))
            drive = Drive()
            applied = controller.step(robot=Robot(), drive=drive, grid=make_grid(), goal_world_xy=[1, 0], now=0)
            self.assertEqual(controller.status, "TRACKING")
            self.assertGreater(applied.linear_mps, 0)
            self.assertEqual(len(drive.motors), 4)
            self.assertEqual(len(drive.steering), 4)
            controller.reset()
            self.assertIsNone(controller.planner)
            self.assertIsNone(controller.goal)


if __name__ == "__main__":
    unittest.main()
