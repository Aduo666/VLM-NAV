from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import unittest

import numpy as np
import yaml

from vln_nav.navigation.fine_local_planner import FineLocalPlanner, FinePlannerConfig
from vln_nav.navigation.trajectory_clearance import (
    FootprintCollisionChecker, FootprintConfig, FootprintPartConfig,
    GeometryStatus, PlanningGrid,
)
from vln_nav.runtime.agx_geometry import build_agx_footprint, rotation_envelope
from vln_nav.runtime.agx_profile import load_agx_profile


class FootprintLayerTests(unittest.TestCase):
    def make_grid(self, *, known=True):
        shape = (60, 60)
        return PlanningGrid(
            occupied_mask=np.zeros(shape, dtype=bool),
            free_mask=np.full(shape, known, dtype=bool),
            unknown_mask=np.full(shape, not known, dtype=bool),
            clearance_m=np.full(shape, 5.0),
            origin_x_m=-1.5, origin_y_m=-1.5, resolution_m=0.05, map_version=1,
        )

    def make_checker(self):
        return FootprintCollisionChecker(
            FootprintConfig((FootprintPartConfig("body", -0.1, 0.1, -0.1, 0.1),), 0.1),
            yaw_resolution_deg=5.0,
        )

    def test_obstacle_in_safety_margin_still_blocks(self):
        grid = self.make_grid()
        row, column, _ = grid.world_to_cell(0.22, 0.0)
        grid.occupied_mask[row, column] = True
        grid.free_mask[row, column] = False
        checker = self.make_checker()
        metrics = checker.metrics(grid, x_m=0.0, y_m=0.0, yaw_rad=0.0)
        self.assertTrue(metrics.collision)
        self.assertFalse(metrics.physical_collision)
        self.assertTrue(metrics.safety_margin_collision)
        self.assertEqual(checker.classify_pose(grid, x_m=0.0, y_m=0.0, yaw_rad=0.0).status, GeometryStatus.BLOCKED)

    def test_new_unknown_separates_physical_and_margin(self):
        grid = self.make_grid()
        for x in (0.32, 0.42):
            row, column, _ = grid.world_to_cell(x, 0.0)
            grid.unknown_mask[row, column] = True
            grid.free_mask[row, column] = False
        unknown = self.make_checker().incremental_unknown_metrics(
            grid, initial_pose=np.zeros(3), swept_poses=np.asarray([[0.2, 0.0, 0.0]]),
        )
        self.assertEqual(unknown.cell_count, 2)
        self.assertEqual(unknown.physical_unknown_linear_indices.size, 1)
        self.assertEqual(unknown.safety_margin_unknown_linear_indices.size, 1)

    def test_safety_margin_always_contains_physical_cells(self):
        checker = FootprintCollisionChecker(
            FootprintConfig((FootprintPartConfig("body", -0.13, 0.1783, -0.1567, 0.2553),), 0.03),
            yaw_resolution_deg=5.0,
        )
        grid = self.make_grid()
        for yaw in np.linspace(-np.pi, np.pi, 73):
            physical, _ = checker.covered_linear_indices(grid, x_m=0.012, y_m=0.018, yaw_rad=yaw, margin_m=0.0)
            padded, _ = checker.covered_linear_indices(grid, x_m=0.012, y_m=0.018, yaw_rad=yaw)
            self.assertTrue(set(physical) <= set(padded))

    def test_steering_envelope_does_not_clear_unvisited_space(self):
        checker = FootprintCollisionChecker(
            FootprintConfig((FootprintPartConfig(
                "wheel", -0.2, 0.2, -0.2, 0.2, (-0.02, 0.02, -0.02, 0.02),
            ),), 0.03), yaw_resolution_deg=5.0,
        )
        grid = checker.clear_pose(self.make_grid(known=False), x_m=0.0, y_m=0.0, yaw_rad=0.0)
        row, column, _ = grid.world_to_cell(0.17, 0.0)
        self.assertTrue(grid.unknown_mask[row, column])
        unknown = checker.incremental_unknown_metrics(
            grid, initial_pose=np.zeros(3), swept_poses=np.zeros((1, 3)),
        )
        self.assertGreater(unknown.physical_unknown_linear_indices.size, 0)

    def test_safety_margin_unknown_still_rejects_forward(self):
        root = Path(__file__).resolve().parents[2]
        with (root / "configs/navigation.yaml").open(encoding="utf-8") as handle:
            settings = yaml.safe_load(handle)["fine_planner"]
        config = FinePlannerConfig.from_mapping({item.name: settings[item.name] for item in fields(FinePlannerConfig)})
        grid = self.make_grid()
        row, column, _ = grid.world_to_cell(0.32, 0.0)
        grid.unknown_mask[row, column] = True
        grid.free_mask[row, column] = False
        actions = FineLocalPlanner(config, self.make_checker()).inspect_initial_actions(
            grid=grid, start_pose_world=(0.0, 0.0, 0.0),
        )
        forward = next(item for item in actions if item.action_name == "forward_2")
        self.assertEqual(forward.result, "UNKNOWN")
        self.assertEqual(forward.physical_unknown_count, 0)
        self.assertEqual(forward.safety_margin_unknown_count, 1)

    def test_continuous_steering_is_inside_analytic_envelope(self):
        points = np.asarray([[x, y, z] for x in (-0.09, 0.09) for y in (-0.04, 0.04) for z in (-0.09, 0.09)])
        pivot = np.asarray([0.01, -0.02, 0.03])
        axis = np.asarray([0.1, 0.2, 1.0])
        axis /= np.linalg.norm(axis)
        minimum, maximum = rotation_envelope(points, pivot, axis)
        vectors = points - pivot
        parallel = np.outer(vectors @ axis, axis)
        for angle in np.linspace(-np.pi, np.pi, 301):
            rotated = pivot + parallel + (vectors - parallel) * np.cos(angle) + np.cross(axis, vectors) * np.sin(angle)
            self.assertTrue(np.all(rotated >= minimum - 1.0e-12))
            self.assertTrue(np.all(rotated <= maximum + 1.0e-12))


class UsdFootprintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from pxr import Usd
        except ImportError:
            raise unittest.SkipTest("USD 核查测试需要 Isaac Sim 的 pxr 环境")
        cls.Usd = Usd
        cls.root = Path(__file__).resolve().parents[2]
        cls.profile = load_agx_profile(project_root=cls.root)

    def make_stage(self):
        stage = self.Usd.Stage.CreateInMemory()
        stage.DefinePrim(self.profile.asset.prim_path, "Xform").GetReferences().AddReference(str(self.profile.asset.usd_path))
        base = self.profile.asset.prim_path + "/" + self.profile.asset.articulation_path
        return stage, base

    def test_all_23_colliders_and_four_independent_wheels(self):
        stage, base = self.make_stage()
        footprint, audit = build_agx_footprint(stage, self.profile, base, margin_m=0.03)
        self.assertEqual(audit["collider_count"], 23)
        self.assertEqual(len(footprint.parts), 23)
        names = {part.name for part in footprint.parts}
        self.assertTrue({"front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel"} <= names)
        self.assertTrue(all(item["geometry_descendants"] for item in audit["colliders"]))

    def test_stale_cube_extent_uses_actual_shape(self):
        stage, base = self.make_stage()
        _, audit = build_agx_footprint(stage, self.profile, base, margin_m=0.03)
        cubes = [item for item in audit["colliders"] if item["usd_type"] == "Cube"]
        self.assertEqual(len(cubes), 2)
        self.assertTrue(all(item["geometry_descendants"][0]["authored_extent_matches_shape"] is False for item in cubes))
        self.assertGreater(cubes[0]["current_max_base_xyz_m"][0] - cubes[0]["current_min_base_xyz_m"][0], 0.10)

    def test_base_frame_is_independent_of_world_placement(self):
        from pxr import Gf, UsdGeom
        stage, base = self.make_stage()
        first, _ = build_agx_footprint(stage, self.profile, base, margin_m=0.03)
        root = stage.GetPrimAtPath(self.profile.asset.prim_path)
        transform = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), -90))
        transform.SetTranslateOnly(Gf.Vec3d(1.5, 4.0, 0.32))
        UsdGeom.Xformable(root).MakeMatrixXform().Set(transform)
        second, _ = build_agx_footprint(stage, self.profile, base, margin_m=0.03)
        np.testing.assert_allclose(
            [(p.min_x_m, p.max_x_m, p.min_y_m, p.max_y_m) for p in first.parts],
            [(p.min_x_m, p.max_x_m, p.min_y_m, p.max_y_m) for p in second.parts], atol=1.0e-8,
        )

    def test_missing_collision_geometry_stops_building(self):
        stage, base = self.make_stage()
        mesh = stage.GetPrimAtPath(self.profile.asset.prim_path + "/base_link/collisions/ranger_mini3/Scene/mesh")
        instance = mesh.GetParent()
        while instance.IsInstanceProxy():
            instance = instance.GetParent()
        instance.SetInstanceable(False)
        mesh = stage.GetPrimAtPath(mesh.GetPath())
        mesh.SetActive(False)
        with self.assertRaises(ValueError):
            build_agx_footprint(stage, self.profile, base, margin_m=0.03)


if __name__ == "__main__":
    unittest.main()
