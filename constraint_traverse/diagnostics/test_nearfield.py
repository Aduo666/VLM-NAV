from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
import unittest

import numpy as np
import yaml

from constraint_traverse.collect.bev import local_voxel_config_from_mapping
from constraint_traverse.diagnostics.nearfield import (
    build_stride_comparison,
    project_cell_to_depth,
)
from vln_nav.active_perception.camera_geometry import CameraFrameGeometry
from vln_nav.navigation.fine_local_planner import FinePlannerConfig
from vln_nav.navigation.trajectory_clearance import (
    FootprintCollisionChecker,
    FootprintConfig,
)


class NearfieldDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[2]
        with (root / "configs/navigation.yaml").open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        mapping = dict(config["mapping"]["map"])
        mapping.pop("integration_period_sec")
        cls.map_config = replace(
            local_voxel_config_from_mapping(mapping),
            size_x_m=2.0,
            size_y_m=2.0,
        )
        fine = config["fine_planner"]
        cls.fine_config = FinePlannerConfig.from_mapping({
            field.name: fine[field.name]
            for field in fields(FinePlannerConfig)
        })
        cls.checker = FootprintCollisionChecker(
            FootprintConfig.from_mapping(fine["footprint"]),
            yaw_resolution_deg=cls.fine_config.yaw_resolution_deg,
        )

    def make_frame(self, source: str) -> CameraFrameGeometry:
        rgb = np.zeros((80, 80, 3), dtype=np.uint8)
        depth = np.full((80, 80), 1.0, dtype=np.float32)
        intrinsics = np.asarray([
            [40.0, 0.0, 39.5],
            [0.0, 40.0, 39.5],
            [0.0, 0.0, 1.0],
        ])
        pose = np.eye(4)
        pose[2, 3] = 0.32
        return CameraFrameGeometry(
            source=source,
            frame_index=0,
            timestamp_sec=0.1,
            rgb=rgb,
            depth_m=depth,
            intrinsics=intrinsics,
            base_from_camera=np.eye(4),
            world_from_camera=pose,
            world_from_base=pose,
        )

    def test_stride_comparison_uses_identical_frames(self) -> None:
        front = self.make_frame("front")
        down = self.make_frame("down")
        results, comparison = build_stride_comparison(
            frame_pairs=[(front, down)],
            map_config=self.map_config,
            fine_config=self.fine_config,
            collision_checker=self.checker,
            robot_world_xyz=np.asarray([0.0, 0.0, 0.32]),
            robot_yaw_rad=0.0,
            floor_z_m=0.0,
            obstacle_min_height_m=0.05,
            obstacle_max_height_m=1.50,
        )
        self.assertEqual(set(results), {5, 10})
        self.assertEqual(results[10]["updates"][0]["rays"]["front"], 64)
        self.assertEqual(results[5]["updates"][0]["rays"]["front"], 256)
        self.assertEqual(
            comparison["physical_unknown_count"]
            + comparison["margin_only_unknown_count"],
            len(comparison["cells"]),
        )

    def test_cell_projection_reports_behind_camera(self) -> None:
        frame = self.make_frame("down")
        result = project_cell_to_depth(
            frame,
            world_x=0.0,
            world_y=0.0,
            floor_z_m=0.0,
        )
        self.assertEqual(result[0]["visibility"], "BEHIND_CAMERA")


if __name__ == "__main__":
    unittest.main()
