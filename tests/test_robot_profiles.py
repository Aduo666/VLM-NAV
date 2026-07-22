from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from vln_nav.navigation.observation_adapter import Pose2D
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


if __name__ == "__main__":
    unittest.main()
