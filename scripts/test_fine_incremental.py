from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from vln_nav.app.navigation import _FineFailureInput, _repeated_partial_path
from vln_nav.control.fine_path_tracker import FinePathTracker
from vln_nav.control.trajectory import TrajectoryConfig
from vln_nav.navigation.fine_local_planner import (
    FineLocalPlanner,
    FinePlannerConfig,
    FinePlanningFailure,
    FineUnknownBlockedError,
)
from vln_nav.navigation.trajectory_clearance import (
    FootprintCollisionChecker,
    FootprintConfig,
    PlanningGrid,
    assess_known_swept_path,
    clear_current_footprint,
    translation_distance_until_direction_change,
)


class FineIncrementalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        with (root / "configs/navigation.yaml").open(encoding="utf-8") as handle:
            cls.navigation = yaml.safe_load(handle)
        with (root / "configs/robots/agx.yaml").open(encoding="utf-8") as handle:
            robot = yaml.safe_load(handle)
        fine = cls.navigation["fine_planner"]
        cls.config = FinePlannerConfig.from_mapping({
            field.name: fine[field.name] for field in fields(FinePlannerConfig)
        })
        cls.checker = FootprintCollisionChecker(
            FootprintConfig.from_mapping(fine["footprint"]),
            yaw_resolution_deg=cls.config.yaw_resolution_deg,
        )
        cls.tracker_config = replace(
            TrajectoryConfig.from_mapping(robot["controller"]["continuous"]),
            **{
                name: float(value)
                for name, value in cls.navigation["fine_tracking"].items()
                if name in {field.name for field in fields(TrajectoryConfig)}
            },
        )

    def make_grid(self, known: np.ndarray, *, version: int = 1) -> PlanningGrid:
        shape = known.shape
        return PlanningGrid(
            occupied_mask=np.zeros(shape, dtype=bool),
            free_mask=known.copy(),
            unknown_mask=~known,
            clearance_m=np.full(shape, 5.0, dtype=np.float32),
            origin_x_m=-1.5,
            origin_y_m=-1.5,
            resolution_m=0.05,
            map_version=version,
        )

    def corridor_grid(self) -> PlanningGrid:
        rows, columns = np.indices((60, 60))
        world_x = -1.5 + (columns + 0.5) * 0.05
        world_y = -1.5 + (rows + 0.5) * 0.05
        known = (world_x > -0.65) & (world_x < 0.65) & (abs(world_y) < 0.7)
        return clear_current_footprint(
            self.make_grid(known),
            x_m=0.0,
            y_m=0.0,
            yaw_rad=0.0,
            collision_checker=self.checker,
        )

    def test_safe_short_path_uses_reference_detour(self) -> None:
        reference = np.asarray([
            [0.0, 0.0], [-0.3, 0.0], [-0.3, 0.5], [1.0, 0.5], [1.0, 0.0],
        ])
        grid = self.corridor_grid()
        plan = FineLocalPlanner(self.config, self.checker).plan(
            grid=grid,
            start_pose_world=(0.0, 0.0, 0.0),
            goal_world_xy=(1.0, 0.0),
            reference_path_world_xy=reference,
        )
        self.assertEqual(plan.status, "SAFE_PARTIAL_PATH")
        self.assertLess(plan.path_world_xy[-1, 0], -0.05)
        clearance, unknown = assess_known_swept_path(
            grid,
            plan.path_world_xy,
            plan.path_world_yaw_rad,
            collision_checker=self.checker,
            max_step_m=0.05,
            max_yaw_step_deg=5.0,
        )
        self.assertFalse(clearance.collision)
        self.assertEqual(unknown.cell_count, 0)

    def test_complete_path_stays_available(self) -> None:
        grid = self.make_grid(np.ones((60, 60), dtype=bool))
        plan = FineLocalPlanner(self.config, self.checker).plan(
            grid=grid,
            start_pose_world=(0.0, 0.0, 0.0),
            goal_world_xy=(0.4, 0.0),
        )
        self.assertEqual(plan.status, "FULL_PATH")
        self.assertLessEqual(
            np.linalg.norm(plan.path_world_xy[-1] - np.asarray([0.4, 0.0])),
            self.config.goal_tolerance_m + 1.0e-9,
        )

    def test_unusable_partial_candidate_keeps_searching(self) -> None:
        planner = FineLocalPlanner(replace(self.config, partial_search_max_expansions=1), self.checker)
        grid = self.make_grid(np.ones((60, 60), dtype=bool))
        with patch.object(planner, "_has_safe_partial_continuation", return_value=False):
            plan = planner.plan(grid=grid, start_pose_world=(0, 0, 0), goal_world_xy=(0.6, 0))
        self.assertEqual(plan.status, "FULL_PATH")
        self.assertGreater(plan.expanded_node_count, 1)

    def test_cancel_request_stops_search(self) -> None:
        planner = FineLocalPlanner(self.config, self.checker)
        with self.assertRaises(FinePlanningFailure) as context:
            planner.plan(grid=self.make_grid(np.ones((60, 60), dtype=bool)),
                         start_pose_world=(0, 0, 0), goal_world_xy=(1, 0), cancel_requested=lambda: True)
        self.assertEqual(context.exception.reason, "CANCELLED")

    def test_unknown_actions_report_vehicle_parts(self) -> None:
        grid = clear_current_footprint(
            self.make_grid(np.zeros((60, 60), dtype=bool)),
            x_m=0.0,
            y_m=0.0,
            yaw_rad=0.0,
            collision_checker=self.checker,
        )
        with self.assertRaises(FineUnknownBlockedError) as context:
            FineLocalPlanner(self.config, self.checker).plan(
                grid=grid,
                start_pose_world=(0.0, 0.0, 0.0),
                goal_world_xy=(1.0, 0.0),
            )
        actions = context.exception.action_diagnostics
        self.assertEqual(len(actions), 12)
        straight = next(item for item in actions if item.action_name == "forward_2")
        self.assertEqual(straight.result, "UNKNOWN")
        self.assertGreater(len(straight.unknown_linear_indices), 0)
        self.assertGreater(sum(count for _, count in straight.part_unknown_counts), 0)

    def test_forward_safe_while_spin_unknown_executes_forward(self) -> None:
        rows, columns = np.indices((80, 80))
        world_x = -2.0 + (columns + 0.5) * 0.05
        world_y = -2.0 + (rows + 0.5) * 0.05
        known = (
            (world_x > -0.7)
            & (world_x < 0.7)
            & (abs(world_y) < 0.39)
        )
        grid = PlanningGrid(
            occupied_mask=np.zeros((80, 80), dtype=bool),
            free_mask=known,
            unknown_mask=~known,
            clearance_m=np.full((80, 80), 5.0, dtype=np.float32),
            origin_x_m=-2.0,
            origin_y_m=-2.0,
            resolution_m=0.05,
            map_version=1,
        )
        grid = clear_current_footprint(
            grid,
            x_m=0.0,
            y_m=0.0,
            yaw_rad=0.0,
            collision_checker=self.checker,
        )
        planner = FineLocalPlanner(self.config, self.checker)
        actions = {
            item.action_name: item
            for item in planner.inspect_initial_actions(
                grid=grid,
                start_pose_world=(0.0, 0.0, 0.0),
            )
        }
        self.assertEqual(actions["forward_2"].result, "SAFE")
        self.assertEqual(actions["spin_left"].result, "UNKNOWN")
        plan = planner.plan(
            grid=grid,
            start_pose_world=(0.0, 0.0, 0.0),
            goal_world_xy=(1.0, 0.0),
        )
        self.assertEqual(plan.status, "SAFE_PARTIAL_PATH")
        self.assertEqual(int(plan.path_motion_direction[1]), 1)
        self.assertGreaterEqual(plan.path_world_xy[-1, 0], 0.1)

    def test_short_path_requires_stop_distance(self) -> None:
        config = replace(self.config, max_expansions=100)
        with self.assertRaises(FinePlanningFailure):
            FineLocalPlanner(config, self.checker).plan(
                grid=self.corridor_grid(),
                start_pose_world=(0.0, 0.0, 0.0),
                goal_world_xy=(1.0, 0.0),
                initial_speed_mps=0.6,
            )

    def test_known_wall_blocks_complete_path(self) -> None:
        grid = self.make_grid(np.ones((60, 60), dtype=bool))
        grid.occupied_mask[:, 41] = True
        grid.free_mask[:, 41] = False
        with self.assertRaises(FinePlanningFailure):
            FineLocalPlanner(self.config, self.checker).plan(
                grid=grid,
                start_pose_world=(0.0, 0.0, 0.0),
                goal_world_xy=(1.0, 0.0),
            )

    def test_unknown_beyond_near_field_blocks_full_path(self) -> None:
        grid = self.corridor_grid()
        clearance, unknown = assess_known_swept_path(
            grid,
            np.asarray([[0.0, 0.0], [1.0, 0.0]]),
            np.zeros(2),
            collision_checker=self.checker,
            max_step_m=0.05,
            max_yaw_step_deg=5.0,
        )
        self.assertFalse(clearance.collision)
        self.assertGreater(unknown.cell_count, 0)

    def test_new_obstacle_or_unknown_invalidates_active_path(self) -> None:
        grid = self.make_grid(np.ones((60, 60), dtype=bool))
        plan = FineLocalPlanner(self.config, self.checker).plan(
            grid=grid,
            start_pose_world=(0.0, 0.0, 0.0),
            goal_world_xy=(0.4, 0.0),
        )
        obstacle_grid = self.make_grid(np.ones((60, 60), dtype=bool))
        obstacle_grid.occupied_mask[30, 43] = True
        obstacle_grid.free_mask[30, 43] = False
        collision, _ = assess_known_swept_path(
            obstacle_grid,
            plan.path_world_xy,
            plan.path_world_yaw_rad,
            collision_checker=self.checker,
            max_step_m=0.05,
            max_yaw_step_deg=5.0,
        )
        self.assertTrue(collision.collision)
        unknown_grid = self.make_grid(np.ones((60, 60), dtype=bool))
        unknown_grid.unknown_mask[30, 43] = True
        unknown_grid.free_mask[30, 43] = False
        _, unknown = assess_known_swept_path(
            unknown_grid,
            plan.path_world_xy,
            plan.path_world_yaw_rad,
            collision_checker=self.checker,
            max_step_m=0.05,
            max_yaw_step_deg=5.0,
        )
        self.assertGreater(unknown.cell_count, 0)

    def test_repeated_short_path_requires_actual_motion(self) -> None:
        plan = FineLocalPlanner(self.config, self.checker).plan(
            grid=self.corridor_grid(),
            start_pose_world=(0.0, 0.0, 0.0),
            goal_world_xy=(1.0, 0.0),
        )
        endpoint = plan.path_world_xy[-1].copy()
        self.assertTrue(_repeated_partial_path(
            plan,
            current_world_xy=np.zeros(2),
            last_switch_world_xy=np.zeros(2),
            last_endpoint_world_xy=endpoint,
            resolution_m=0.05,
        ))
        self.assertFalse(_repeated_partial_path(
            plan,
            current_world_xy=np.asarray([0.06, 0.0]),
            last_switch_world_xy=np.zeros(2),
            last_endpoint_world_xy=endpoint,
            resolution_m=0.05,
        ))

    def test_rolling_short_paths_keep_moving_forward(self) -> None:
        planner = FineLocalPlanner(self.config, self.checker)
        history = []
        start = np.asarray([0.0, 0.0, 0.0])
        for _ in range(4):
            rows, columns = np.indices((100, 100))
            world_x = -1.5 + (columns + 0.5) * 0.05
            world_y = -1.5 + (rows + 0.5) * 0.05
            known = (
                (world_x > start[0] - 0.65)
                & (world_x < start[0] + 0.65)
                & (abs(world_y) < 0.7)
            )
            grid = clear_current_footprint(
                self.make_grid(known),
                x_m=float(start[0]),
                y_m=float(start[1]),
                yaw_rad=float(start[2]),
                collision_checker=self.checker,
            )
            plan = planner.plan(
                grid=grid,
                start_pose_world=tuple(map(float, start)),
                goal_world_xy=(2.0, 0.0),
                recent_world_poses=np.asarray(history).reshape(-1, 3),
            )
            self.assertEqual(plan.status, "SAFE_PARTIAL_PATH")
            self.assertGreater(plan.path_world_xy[-1, 0], start[0] + 0.05)
            history.append(start.copy())
            start = np.asarray([
                plan.path_world_xy[-1, 0],
                plan.path_world_xy[-1, 1],
                plan.path_world_yaw_rad[-1],
            ])
        self.assertGreater(start[0], 0.3)

    def test_short_tracker_requires_actual_endpoint(self) -> None:
        points = np.asarray([[0.0, 0.0], [0.05, 0.0], [0.1, 0.0]])
        yaw = np.zeros(3)
        direction = np.asarray([0, 1, 1], dtype=np.int8)
        def tracker() -> FinePathTracker:
            return FinePathTracker(
                points, yaw, direction, self.tracker_config
            )
        self.assertEqual(tracker().step((0.0, 0.0, 0.0), 1 / 60).status, "TRACKING")
        self.assertEqual(tracker().step((0.11, 0.19, 0.0), 1 / 60).status, "PATH_ERROR")
        self.assertEqual(tracker().step((0.1, 0.0, 0.0), 1 / 60).status, "ARRIVED")

    def test_short_tracker_caps_speed_for_braking(self) -> None:
        tracker = FinePathTracker(
            np.asarray([[0.0, 0.0], [0.05, 0.0], [0.1, 0.0]]),
            np.zeros(3),
            np.asarray([0, 1, 1], dtype=np.int8),
            self.tracker_config,
            initial_speed_mps=0.6,
            safety_braking_deceleration_mps2=0.3,
            safety_reaction_time_sec=0.3,
            safety_reserve_m=0.05,
        )
        result = tracker.step((0.0, 0.0, 0.0), 1 / 60)
        self.assertEqual(result.status, "TRACKING")
        self.assertGreater(result.command.linear_mps, 0.0)
        self.assertLess(result.command.linear_mps, 0.16)
        middle = tracker.step((0.05, 0.0, 0.0), 1 / 60)
        self.assertEqual(middle.status, "TRACKING")
        self.assertGreater(middle.command.linear_mps, 0.0)

    def test_stop_space_ends_at_direction_change(self) -> None:
        points = np.asarray([
            [0.0, 0.0],
            [0.05, 0.0],
            [0.05, 0.0],
            [0.05, 0.4],
        ])
        directions = np.asarray([0, 1, 0, 1], dtype=np.int8)
        self.assertAlmostEqual(
            translation_distance_until_direction_change(points, directions),
            0.05,
        )
        tracker = FinePathTracker(
            points,
            np.asarray([0.0, 0.0, np.pi / 2, np.pi / 2]),
            directions,
            self.tracker_config,
            initial_speed_mps=0.6,
            safety_braking_deceleration_mps2=0.3,
            safety_reaction_time_sec=0.3,
            safety_reserve_m=0.0,
        )
        result = tracker.step((0.0, 0.0, 0.0), 1 / 60)
        self.assertEqual(result.status, "TRACKING")
        self.assertLess(result.command.linear_mps, 0.11)

    def test_reverse_leg_does_not_extend_forward_stop_space(self) -> None:
        points = np.asarray([[0.0, 0.0], [0.1, 0.0], [-0.3, 0.0]])
        directions = np.asarray([0, 1, -1], dtype=np.int8)
        self.assertAlmostEqual(
            translation_distance_until_direction_change(points, directions),
            0.1,
        )

    def test_lateral_error_does_not_increase_stop_space(self) -> None:
        points = np.asarray([[0.05, 0.1], [0.1, 0.0]])
        directions = np.asarray([0, 1], dtype=np.int8)
        self.assertAlmostEqual(
            translation_distance_until_direction_change(
                points,
                directions,
                first_segment_start_xy=np.asarray([0.0, 0.0]),
            ),
            0.05,
        )

    def test_far_map_update_does_not_repeat_failed_search(self) -> None:
        grid = self.make_grid(np.zeros((60, 60), dtype=bool))
        job = type("Job", (), {
            "start_pose_world": (0.0, 0.0, 0.0),
            "goal_world_xy": (1.0, 0.0),
        })()
        failed = _FineFailureInput.from_job(
            grid, job, relevant_radius_m=0.6
        )
        changed = self.make_grid(np.zeros((60, 60), dtype=bool), version=2)
        changed.unknown_mask[0, 0] = False
        self.assertTrue(failed.matches(changed, job.start_pose_world, job.goal_world_xy))
        changed.unknown_mask[30, 30] = False
        self.assertFalse(failed.matches(changed, job.start_pose_world, job.goal_world_xy))


if __name__ == "__main__":
    unittest.main()
