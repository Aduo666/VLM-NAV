from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
import shapely
import torch

from constraint_traverse.benchmark.geometry import SweepResult, swept_part_polygons
from constraint_traverse.benchmark.model import FourWheelModel, MotionLimits, STATE_SIZE
from constraint_traverse.benchmark.planner import PassageCost, SamplingPlanner
from constraint_traverse.benchmark.run import load_config
from constraint_traverse.collect.transforms import pose_xyyaw
from vln_nav.control.trajectory import VelocityCommand


class PlanningGridGeometry:
    def __init__(self, footprint, sweep_step_m):
        self.parts = np.asarray([[p.min_x_m, p.max_x_m, p.min_y_m, p.max_y_m] for p in footprint.parts])
        self.margin_m = float(footprint.margin_m)
        self.sweep_step_m = float(sweep_step_m)
        self.radius_m = float(np.max(np.hypot(np.abs(self.parts[:, :2]).max(axis=1), np.abs(self.parts[:, 2:]).max(axis=1))))
        self.grid = None
        self._version_key = None
        self._tensor_cache = {}
        self.rejected_unknown_world_xy = None

    def update_grid(self, grid):
        key = (grid.map_version, grid.origin_x_m, grid.origin_y_m, grid.resolution_m, grid.shape)
        if key == self._version_key:
            return
        self.grid = grid
        self._version_key = key
        self._tensor_cache.clear()
        rows, columns = grid.shape
        r = grid.resolution_m
        self.x_edges = grid.origin_x_m + np.arange(columns + 1) * r
        self.y_edges = grid.origin_y_m + np.arange(rows + 1) * r
        self.bounds = (self.x_edges[0], self.y_edges[0], self.x_edges[-1], self.y_edges[-1])
        self.occupied = self._mask_geometry(grid.occupied_mask)
        self.unknown = self._mask_geometry(grid.unknown_mask & ~grid.occupied_mask)
        blocked = grid.occupied_mask | grid.unknown_mask
        self.clearance_field = distance_transform_edt(~blocked, sampling=r).astype(np.float32)
        points = []
        for xmin, xmax, ymin, ymax in self.parts:
            xs = np.linspace(xmin, xmax, max(2, math.ceil((xmax - xmin) / r) + 1))
            ys = np.linspace(ymin, ymax, max(2, math.ceil((ymax - ymin) / r) + 1))
            xx, yy = np.meshgrid(xs, ys)
            points.append(np.column_stack((xx.ravel(), yy.ravel())))
        self.sample_points = np.unique(np.round(np.concatenate(points), 7), axis=0).astype(np.float32)

    def _mask_geometry(self, mask):
        rows, columns = np.nonzero(mask)
        cells = shapely.box(self.x_edges[columns], self.y_edges[rows], self.x_edges[columns + 1], self.y_edges[rows + 1])
        return shapely.coverage_union_all(cells)

    def torch_clearance(self, poses):
        if self.grid is None:
            return torch.full(poses.shape[:-1], -1.0, device=poses.device, dtype=poses.dtype)
        key = (poses.device, poses.dtype)
        if key not in self._tensor_cache:
            self._tensor_cache[key] = (torch.as_tensor(self.sample_points, device=poses.device, dtype=poses.dtype),
                                       torch.as_tensor(self.clearance_field, device=poses.device, dtype=poses.dtype))
        points, field = self._tensor_cache[key]
        c, s = torch.cos(poses[..., 2, None]), torch.sin(poses[..., 2, None])
        x = poses[..., 0, None] + c * points[:, 0] - s * points[:, 1]
        y = poses[..., 1, None] + s * points[:, 0] + c * points[:, 1]
        r = self.grid.resolution_m
        columns = torch.floor((x - self.grid.origin_x_m) / r).long()
        rows = torch.floor((y - self.grid.origin_y_m) / r).long()
        height, width = self.grid.shape
        inside = (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
        values = field[rows.clamp(0, height - 1), columns.clamp(0, width - 1)]
        values = torch.where(inside, values, torch.zeros_like(values))
        return values.amin(dim=-1) - math.sqrt(2) * r / 2

    def validate_poses(self, poses):
        self.rejected_unknown_world_xy = None
        if self.grid is None:
            return SweepResult(False, "NO_MAP", 0.0)
        poses = np.asarray(poses, dtype=float).reshape(-1, 3)
        if len(poses) == 0 or not np.isfinite(poses).all():
            return SweepResult(False, "INVALID_PREDICTION", 0.0)
        polygons, margins, x, y = swept_part_polygons(
            self.parts, poses, radius_m=self.radius_m, margin_m=self.margin_m, step_m=self.sweep_step_m,
        )
        xmin, ymin, xmax, ymax = self.bounds
        room_gap = np.minimum.reduce(((x - xmin).min(axis=(1, 2)), (y - ymin).min(axis=(1, 2)),
                                      (xmax - x).min(axis=(1, 2)), (ymax - y).min(axis=(1, 2))))
        occupied_distances = self._distances(polygons, self.occupied, len(margins))
        unknown_distances = self._distances(polygons, self.unknown, len(margins))
        outside = room_gap < margins
        collision = occupied_distances <= margins + 1.0e-9
        unresolved = unknown_distances <= margins + 1.0e-9
        bad = np.flatnonzero(outside | collision | unresolved)
        minimum = float(np.minimum(occupied_distances, room_gap).min())
        if len(bad):
            index = int(bad[0])
            reason = "OUTSIDE_MAP" if outside[index] else "OBSTACLE" if collision[index] else "UNKNOWN"
            if reason == "UNKNOWN":
                relevant = polygons.reshape(len(margins), -1)[index]
                region = shapely.union_all(shapely.buffer(relevant, margins[index]))
                rows, columns = np.nonzero(self.grid.unknown_mask)
                cells = shapely.box(self.x_edges[columns], self.y_edges[rows], self.x_edges[columns + 1], self.y_edges[rows + 1])
                touched = shapely.intersects(cells, region)
                self.rejected_unknown_world_xy = np.column_stack(((self.x_edges[columns[touched]] + self.x_edges[columns[touched] + 1]) / 2,
                                                                 (self.y_edges[rows[touched]] + self.y_edges[rows[touched] + 1]) / 2))
            return SweepResult(False, reason, minimum)
        return SweepResult(True, "SAFE", minimum)

    @staticmethod
    def _distances(polygons, geometry, sample_count):
        if geometry.is_empty:
            return np.full(sample_count, 1.0e6)
        return shapely.distance(polygons, geometry).reshape(sample_count, -1).min(axis=1)


class SamplingNavigationController:
    def __init__(self, *, method, hardware, footprint, physics_dt, config_path, logger):
        self.config = load_config(Path(config_path))
        self.method, self.hardware, self.logger = method, hardware, logger
        torch.set_num_threads(1)
        limits = MotionLimits(**self.config["motion"])
        self.model = FourWheelModel(hardware, limits, self.config["planner"]["dt_sec"], device=self.config["planner"]["device"])
        self.execution_model = FourWheelModel(hardware, limits, physics_dt)
        sampling_footprint = replace(footprint, margin_m=float(self.config["safety"]["margin_m"]))
        sampling_footprint.validate()
        self.geometry = PlanningGridGeometry(sampling_footprint, self.config["safety"]["sweep_step_m"])
        self.cost = PassageCost(self.geometry, self.config["cost"], device=self.model.device)
        self.reset()

    def reset(self):
        self.planner = None
        self.goal = None
        self.last_plan = None
        self.last_command = np.zeros(2, dtype=np.float32)
        self.requested = np.zeros(2, dtype=np.float32)
        self.last_planning_time = -float("inf")
        self.last_checked_map_key = None
        self.status = "IDLE"
        self.geometry.rejected_unknown_world_xy = None

    def read_state(self, robot, drive):
        position, orientation = robot.get_world_pose()
        pose = pose_xyyaw(position, orientation)
        linear, angular = np.asarray(robot.get_linear_velocity()), np.asarray(robot.get_angular_velocity())
        readback = drive.readback()
        c, s = math.cos(pose[2]), math.sin(pose[2])
        state = np.zeros(STATE_SIZE, dtype=np.float32)
        state[:3], state[3:6] = pose, [c * linear[0] + s * linear[1], -s * linear[0] + c * linear[1], angular[2]]
        for index, wheel in enumerate(self.hardware.wheels):
            state[6 + index] = readback["steering_position"][wheel.steering_joint] * wheel.steering_sign
            state[10 + index] = readback["wheel_velocity"][wheel.drive_joint] * wheel.radius_m * wheel.drive_sign
        state[14:16] = self.last_command
        return state

    def step(self, *, robot, drive, grid, goal_world_xy, now):
        state = self.read_state(robot, drive)
        goal = np.asarray(goal_world_xy, dtype=np.float32).reshape(2)
        if self.goal is None or np.linalg.norm(goal - self.goal) > 0.01:
            self.goal = goal.copy()
            self.cost.set_target(goal)
            self.planner = SamplingPlanner(self.method, self.model, self.cost, self.config["planner"], self.config["safety"])
            self.requested[:] = 0
            self.last_planning_time = -float("inf")
        if grid is None:
            self.requested[:] = 0
            self.status = "NO_MAP"
        else:
            self.geometry.update_grid(grid)
            if now - self.last_planning_time >= self.config["planner"]["dt_sec"]:
                self.last_planning_time = now
                self.last_plan = self.planner.plan(state)
                self.requested = self.last_plan.command.copy()
                self.status = "TRACKING" if self.last_plan.safe else self.last_plan.reason
                if self.status == "UNKNOWN":
                    self.geometry.validate_poses(self.last_plan.prediction)
                self.logger.info("[SamplingPlan] mode=%s map_version=%d status=%s latency=%.3fs clearance=%.3fm goal_world=%s",
                                 self.method, grid.map_version, self.status, self.last_plan.planning_sec,
                                 self.last_plan.min_clearance_m, goal.tolist())
                self.last_checked_map_key = self.geometry._version_key
            elif self.geometry._version_key != self.last_checked_map_key:
                future = self.execution_model.rollout(state, [self.requested])
                braking = self.execution_model.braking_rollout(future[-1], self.config["safety"]["reaction_time_sec"])
                check = self.geometry.validate_poses(np.concatenate((future[:, :3].numpy(), braking[:, :3].numpy())))
                self.last_checked_map_key = self.geometry._version_key
                if not check.safe:
                    self.requested[:] = 0
                    self.status = check.reason
                    self.last_planning_time = -float("inf")
                    self.logger.warning("[SamplingSafetyStop] mode=%s reason=%s map_version=%d", self.method, check.reason, grid.map_version)
        with torch.no_grad():
            predicted = self.execution_model.step(state, self.requested).numpy()
        motors, steering = {}, {}
        for index, wheel in enumerate(self.hardware.wheels):
            motors[wheel.drive_joint] = float(predicted[10 + index] / wheel.radius_m / wheel.drive_sign)
            steering[wheel.steering_joint] = float(predicted[6 + index] / wheel.steering_sign)
        drive.apply_continuous_joint_targets(motors, steering)
        self.last_command = predicted[14:16].copy()
        return VelocityCommand(float(self.last_command[0]), float(self.last_command[1]))
