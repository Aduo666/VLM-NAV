from __future__ import annotations

import logging
import unittest

import numpy as np

from vln_nav.navigation.observation_adapter import Pose2D
from vln_nav.navigation.occupancy_grid import OccupancyGridSnapshot
from vln_nav.navigation.primitive_safety import PrimitiveSafetyEvaluator
from vln_nav.runtime.config import PROJECT_ROOT
from vln_nav.runtime.robot_profile import load_robot_profile


class PrimitiveSafetyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test_primitive_safety")
        self.logger.handlers.clear()
        self.logger.addHandler(logging.NullHandler())
        self.pose = Pose2D(x=0.0, y=0.0, yaw_rad=0.0)

    def test_front_obstacle_blocks_forward_but_not_in_place_turn(self) -> None:
        evaluator = self._evaluator("agx", lidar_points=[[0.90, 0.0]])

        self.assertFalse(evaluator.evaluate(action_name="FORWARD").valid)
        self.assertTrue(evaluator.evaluate(action_name="TURN_LEFT").valid)
        self.assertTrue(evaluator.evaluate(action_name="TURN_RIGHT").valid)

    def test_unknown_occupancy_cells_do_not_block_turn(self) -> None:
        evaluator = self._evaluator(
            "unitree_g1_29dof",
            lidar_points=[],
            snapshot=_snapshot(),
        )

        decision = evaluator.evaluate(action_name="TURN_LEFT")
        self.assertTrue(decision.valid)
        self.assertEqual(decision.source, "none")

    def test_robot_profiles_produce_different_turn_envelopes(self) -> None:
        point = [[0.0, 0.40]]
        agx = self._evaluator("agx", lidar_points=point)
        g1 = self._evaluator("unitree_g1_29dof", lidar_points=point)

        self.assertFalse(agx.evaluate(action_name="TURN_LEFT").valid)
        self.assertTrue(g1.evaluate(action_name="TURN_LEFT").valid)

    def test_raw_occupied_map_blocks_sweep_without_inflated_mask(self) -> None:
        snapshot = _snapshot(occupied_world_xy=(0.90, 0.0))
        evaluator = self._evaluator(
            "agx",
            lidar_points=[],
            snapshot=snapshot,
        )

        decision = evaluator.evaluate(action_name="FORWARD")
        self.assertFalse(decision.valid)
        self.assertEqual(decision.source, "occupancy")

    def _evaluator(
        self,
        profile_name: str,
        *,
        lidar_points,
        snapshot: OccupancyGridSnapshot | None = None,
    ) -> PrimitiveSafetyEvaluator:
        evaluator = PrimitiveSafetyEvaluator(
            profile=load_robot_profile(
                profile_name=profile_name,
                project_root=PROJECT_ROOT,
            ),
            logger=self.logger,
        )
        evaluator.update_environment(
            robot_pose=self.pose,
            current_lidar_points_local_xy=np.asarray(
                lidar_points,
                dtype=np.float32,
            ).reshape((-1, 2)),
            occupancy_snapshot=snapshot,
        )
        return evaluator


def _snapshot(
    *,
    occupied_world_xy: tuple[float, float] | None = None,
) -> OccupancyGridSnapshot:
    resolution = 0.10
    origin = -2.0
    shape = (40, 40)
    log_odds = np.zeros(shape, dtype=np.float32)
    occupied = np.zeros(shape, dtype=bool)
    if occupied_world_xy is not None:
        col = int(np.floor((occupied_world_xy[0] - origin) / resolution))
        row = int(np.floor((occupied_world_xy[1] - origin) / resolution))
        occupied[row, col] = True
        log_odds[row, col] = 2.0
    return OccupancyGridSnapshot(
        resolution_m=resolution,
        size_m=4.0,
        log_odds=log_odds,
        occupied_mask=occupied,
        inflated_occupied=occupied.copy(),
        free_mask=np.zeros(shape, dtype=bool),
        origin_x=origin,
        origin_y=origin,
    )


if __name__ == "__main__":
    unittest.main()
