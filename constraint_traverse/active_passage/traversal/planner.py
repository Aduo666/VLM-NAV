from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
from shapely.geometry import box, Point
from shapely.ops import unary_union

from constraint_traverse.reference_path import ReferencePath, angle_difference
from vln_nav.navigation.trajectory_clearance import FootprintCollisionChecker, sample_swept_path


class PlanningFailure(RuntimeError):
    def __init__(self, reason, detail):
        self.reason, self.detail = reason, detail
        super().__init__(f"{reason}: {detail}")


class SmacLatticePlanner:
    def __init__(self, footprint, settings):
        try:
            from . import _smac_lattice
        except ImportError as error:
            raise RuntimeError("Smac binding unavailable. Run python -m constraint_traverse.active_passage.traversal.build") from error
        self.footprint, self.settings = footprint, settings
        if settings["unknown_policy"] != "blocked":
            raise ValueError("Unknown cells must remain blocked")
        path = Path(settings["primitive_file"])
        with path.open(encoding="utf-8") as handle:
            self.primitives = json.load(handle)
        metadata = self.primitives["lattice_metadata"]
        self.resolution = float(metadata["grid_resolution"])
        self.backend = _smac_lattice.LatticePlanner(str(path))
        body = unary_union([box(p.min_x_m, p.min_y_m, p.max_x_m, p.max_y_m) for p in footprint.parts]).convex_hull
        body = body.buffer(footprint.margin_m, join_style=2)
        self.polygon = np.asarray(body.exterior.coords[:-1], dtype=float)
        self.inscribed_radius = float(body.exterior.distance(Point(0, 0)))
        self.checker = FootprintCollisionChecker(footprint, yaw_resolution_deg=5)

    def _check_pose(self, grid, pose):
        metrics = self.checker.metrics(grid, x_m=pose[0], y_m=pose[1], yaw_rad=pose[2])
        if metrics.collision:
            raise PlanningFailure("KNOWN_OBSTACLE", "Robot footprint intersects an observed obstacle")
        if metrics.outside_ratio > 0:
            raise PlanningFailure("OUTSIDE_MAP", "Robot footprint extends outside the local map")
        if metrics.unknown_ratio > 0:
            raise PlanningFailure("OBSERVATION_REQUIRED", "Robot footprint includes unknown cells")

    def plan(self, grid, start, goal, *, request_id, timestamp):
        started = time.perf_counter()
        start, goal = np.asarray(start, dtype=float).reshape(3), np.asarray(goal, dtype=float).reshape(2)
        if not np.isfinite(start).all() or not np.isfinite(goal).all():
            raise ValueError("Planner coordinates must be finite")
        if not np.isclose(grid.resolution_m, self.resolution):
            raise ValueError("Map and Smac primitive resolutions differ")
        self._check_pose(grid, start)
        row, column, inside = grid.world_to_cell(*goal)
        if not inside:
            raise PlanningFailure("OUTSIDE_MAP", "Goal is outside the local map")
        if grid.unknown_mask[row, column]:
            raise PlanningFailure("OBSERVATION_REQUIRED", "Goal has not been observed")
        if grid.occupied_mask[row, column]:
            raise PlanningFailure("KNOWN_OBSTACLE", "Goal lies in an observed obstacle")
        costs = np.zeros(grid.shape, dtype=np.uint8)
        if grid.occupied_mask.any():
            costs[:] = np.clip(180 * np.exp(-grid.clearance_m / 0.3), 0, 252).astype(np.uint8)
            costs[grid.clearance_m <= self.inscribed_radius] = 253
        costs[grid.occupied_mask] = 254
        costs[grid.unknown_mask] = 255
        output = self.backend.plan(costs, grid.resolution_m, grid.origin_x_m, grid.origin_y_m,
            start.tolist(), goal.tolist(), self.polygon, float(self.settings["search_time_sec"]),
            float(self.settings["goal_tolerance_m"]))
        poses = np.asarray(output["poses"])
        if len(poses) < 2:
            reason = output["reason"] if output["reason"] in {"SEARCH_TIMEOUT", "OUTSIDE_MAP"} else "NO_PATH"
            raise PlanningFailure(reason, output["reason"])
        poses = np.vstack((start, poses))
        difference = np.diff(poses, axis=0)
        difference[:, 2] = angle_difference(poses[1:, 2], poses[:-1, 2])
        poses = poses[np.r_[True, (np.linalg.norm(difference[:, :2], axis=1) + np.abs(difference[:, 2])) > 1e-6]]
        swept = sample_swept_path(poses[:, :2], poses[:, 2], max_step_m=0.02, max_yaw_step_deg=2)
        for pose in swept:
            self._check_pose(grid, pose)
        directions = np.zeros(len(poses), dtype=np.int8)
        for index in range(1, len(poses)):
            delta = poses[index, :2] - poses[index - 1, :2]
            if np.linalg.norm(delta) > 1e-5:
                yaw = poses[index - 1, 2] + 0.5 * float(angle_difference(poses[index, 2], poses[index - 1, 2]))
                directions[index] = 1 if delta @ [np.cos(yaw), np.sin(yaw)] >= 0 else -1
        path = ReferencePath(poses[:, :2], poses[:, 2], directions, goal,
                             "SMAC_STATE_LATTICE", request_id, float(timestamp), int(grid.map_version))
        return path, dict(native_sec=float(output["planning_sec"]), total_sec=time.perf_counter() - started,
                           iterations=int(output["iterations"]), unknown_policy="blocked")
