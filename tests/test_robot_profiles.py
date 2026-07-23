from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from vln_nav.navigation.observation_adapter import Pose2D
from vln_nav.navigation.lidar_sensor import _filter_level_frame_beams_by_height
from vln_nav.navigation.occupancy_grid import GlobalOccupancyMap, OccupancyGridConfig
from vln_nav.navigation.spatial_transform import estimate_target_world_xy
from vln_nav.runtime.config import PROJECT_ROOT, load_config
from vln_nav.runtime.robot_profile import _validate_profile_files, load_robot_profile


class RobotProfileTest(unittest.TestCase):
    def test_both_profiles_validate_local_assets(self) -> None:
        agx = load_robot_profile(profile_name="agx", project_root=PROJECT_ROOT)
        g1 = load_robot_profile(profile_name="unitree_g1_29dof", project_root=PROJECT_ROOT)
        self.assertEqual(agx.robot_type, "agx_4ws")
        self.assertFalse(agx.is_g1)
        self.assertEqual(g1.robot_type, "unitree_g1_policy")
        self.assertTrue(g1.is_g1)
        self.assertEqual(len(g1.g1_controller.sdk_joint_names), 29)
        self.assertEqual(g1.simulation.policy_decimation, 4)
        self.assertAlmostEqual(g1.primitives.turn_command, 1.0)
        self.assertEqual(g1.sensors.observer_camera_prim_path, "pelvis/observeCamera")
        self.assertEqual(g1.sensors.lidar_prim_path, "pelvis/Lidar")

    def test_minimal_config_switches_to_g1_without_robot_overrides(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8") as handle:
            handle.write('robot:\n  profile: "unitree_g1_29dof"\n')
            handle.flush()
            config = load_config(handle.name)
        self.assertEqual(config.robot_profile_name, "unitree_g1_29dof")
        self.assertEqual(config.robot_init_position[2], 0.8)
        self.assertAlmostEqual(config.physics_dt, 0.005)
        self.assertEqual(config.robot_policy_decimation, 4)
        self.assertEqual(config.robot_prim_path, "/World/Robot/g1_29dof_rev_1_0")
        self.assertEqual(config.front_camera_prim_path, "torso_link/d435_link/Camera")
        self.assertEqual(
            f"{config.robot_prim_path}/{config.front_camera_prim_path}",
            "/World/Robot/g1_29dof_rev_1_0/torso_link/d435_link/Camera",
        )

    def test_calibrated_projection_respects_robot_left_sign(self) -> None:
        intrinsics = np.eye(3, dtype=np.float64)
        world_from_camera = np.asarray(
            [
                [0.0, 0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        geometry = estimate_target_world_xy(
            center_uv=(-0.5, 0.0),
            depth_m=2.0,
            image_width=2,
            image_height=2,
            horizontal_fov_deg=90.0,
            robot_pose=Pose2D(x=0.0, y=0.0, yaw_rad=0.0),
            camera_intrinsics=intrinsics,
            camera_view_matrix_ros=np.linalg.inv(world_from_camera),
            robot_world_position_xyz=np.zeros(3, dtype=np.float64),
            robot_world_orientation_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        )
        self.assertGreater(geometry.robot_frame_xy[1], 0.0)
        self.assertAlmostEqual(geometry.world_xy[0], 2.0, places=5)

    def test_g1_rejects_track_target_execution_mode(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8") as handle:
            handle.write(
                'robot:\n  profile: "unitree_g1_29dof"\n'
                "control:\n  use_discrete_action_primitives: false\n"
            )
            handle.flush()
            with self.assertRaisesRegex(ValueError, "TRACK_TARGET is not supported"):
                load_config(handle.name)

    def test_g1_rejects_spawn_height_below_fall_threshold(self) -> None:
        profile = load_robot_profile(profile_name="unitree_g1_29dof", project_root=PROJECT_ROOT)
        invalid_profile = replace(profile, spawn_z_m=0.0)
        with self.assertRaisesRegex(ValueError, "pelvis/root height"):
            _validate_profile_files(invalid_profile)

    def test_g1_height_filter_rejects_floor_and_ceiling_returns(self) -> None:
        endpoints, hits, rejected_count = _filter_level_frame_beams_by_height(
            points_robot_xyz=np.asarray(
                [
                    [1.0, -0.2, -0.80],
                    [2.0, 0.0, -0.20],
                    [3.0, 0.2, 1.40],
                ],
                dtype=np.float32,
            ),
            beam_hit_mask=np.asarray([True, True, True]),
            min_height_m=-0.62,
            max_height_m=1.20,
        )
        np.testing.assert_allclose(endpoints, np.asarray([[2.0, 0.0]], dtype=np.float32))
        np.testing.assert_array_equal(hits, np.asarray([True]))
        self.assertEqual(rejected_count, 2)

    def test_g1_static_obstacle_requires_cross_frame_hits(self) -> None:
        occupancy, pose = self._make_g1_occupancy()
        adjacent_hits = np.asarray([[1.0, 0.0], [1.0, 0.05]], dtype=np.float32)
        occupancy.update_from_lidar(
            robot_pose=pose,
            points_local_xy=adjacent_hits,
            beam_endpoints_local_xy=adjacent_hits,
            beam_hit_mask=np.asarray([True, True]),
            timestamp_sec=0.1,
            static_obstacle_confirmation=True,
        )
        self.assertFalse(np.any(occupancy._confirmed_static_occupied))

    def test_g1_confirmed_static_obstacle_survives_intermittent_free_rays(self) -> None:
        occupancy, pose = self._make_g1_occupancy()
        hit_endpoint = np.asarray([[1.0, 0.0]], dtype=np.float32)
        for timestamp_sec in (0.1, 0.2, 0.3):
            self._update_g1_occupancy(
                occupancy=occupancy,
                pose=pose,
                endpoint=hit_endpoint,
                hit=True,
                timestamp_sec=timestamp_sec,
            )

        miss_endpoint = np.asarray([[2.0, 0.0]], dtype=np.float32)
        for update_index in range(6):
            self._update_g1_occupancy(
                occupancy=occupancy,
                pose=pose,
                endpoint=miss_endpoint,
                hit=False,
                timestamp_sec=0.4 + 0.2 * update_index,
            )
            self._update_g1_occupancy(
                occupancy=occupancy,
                pose=pose,
                endpoint=hit_endpoint,
                hit=True,
                timestamp_sec=0.5 + 0.2 * update_index,
            )

        snapshot = occupancy.snapshot()
        row, col = self._cell_for(snapshot=snapshot, world_xy=(1.0, 0.0))
        self.assertTrue(occupancy._confirmed_static_occupied[row, col])
        self.assertGreater(snapshot.log_odds[row, col], occupancy._config.occupied_threshold)

    def test_g1_confirmed_false_obstacle_releases_after_consistent_free_rays(self) -> None:
        occupancy, pose = self._make_g1_occupancy()
        hit_endpoint = np.asarray([[1.0, 0.0]], dtype=np.float32)
        for timestamp_sec in (0.1, 0.2, 0.3):
            self._update_g1_occupancy(
                occupancy=occupancy,
                pose=pose,
                endpoint=hit_endpoint,
                hit=True,
                timestamp_sec=timestamp_sec,
            )

        miss_endpoint = np.asarray([[2.0, 0.0]], dtype=np.float32)
        for update_index in range(8):
            self._update_g1_occupancy(
                occupancy=occupancy,
                pose=pose,
                endpoint=miss_endpoint,
                hit=False,
                timestamp_sec=0.4 + 0.1 * update_index,
            )

        snapshot = occupancy.snapshot()
        row, col = self._cell_for(snapshot=snapshot, world_xy=(1.0, 0.0))
        self.assertFalse(occupancy._confirmed_static_occupied[row, col])
        self.assertLess(snapshot.log_odds[row, col], occupancy._config.occupied_threshold)

    @staticmethod
    def _make_g1_occupancy() -> tuple[GlobalOccupancyMap, Pose2D]:
        import logging

        config = OccupancyGridConfig(
            resolution_m=0.1,
            size_m=4.0,
            update_period_sec=0.02,
            occupied_min_neighbors=3,
        )
        occupancy = GlobalOccupancyMap(
            config=config,
            logger=logging.getLogger("test_g1_occupancy"),
        )
        return occupancy, Pose2D(x=0.0, y=0.0, yaw_rad=0.0)

    @staticmethod
    def _update_g1_occupancy(
        *,
        occupancy: GlobalOccupancyMap,
        pose: Pose2D,
        endpoint: np.ndarray,
        hit: bool,
        timestamp_sec: float,
    ) -> None:
        occupancy.update_from_lidar(
            robot_pose=pose,
            points_local_xy=endpoint if hit else np.zeros((0, 2), dtype=np.float32),
            beam_endpoints_local_xy=endpoint,
            beam_hit_mask=np.asarray([hit]),
            timestamp_sec=timestamp_sec,
            static_obstacle_confirmation=True,
        )

    @staticmethod
    def _cell_for(*, snapshot, world_xy: tuple[float, float]) -> tuple[int, int]:
        row = int(np.floor((world_xy[1] - snapshot.origin_y) / snapshot.resolution_m))
        col = int(np.floor((world_xy[0] - snapshot.origin_x) / snapshot.resolution_m))
        return row, col


if __name__ == "__main__":
    unittest.main()
