from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch

from constraint_traverse.benchmark.model import FourWheelModel, MotionLimits, STATE_SIZE
from constraint_traverse.benchmark.geometry import sample_body_points
from constraint_traverse.benchmark.planner import PassageCost, SamplingPlanner
from constraint_traverse.benchmark.run import load_config
from constraint_traverse.collect.transforms import pose_xyyaw
from vln_nav.control.trajectory import VelocityCommand


class PlanningGridGeometry:
    def __init__(self, footprint):
        self.parts = np.asarray([[p.min_x_m, p.max_x_m, p.min_y_m, p.max_y_m] for p in footprint.parts])
        self.margin_m = float(footprint.margin_m)
        self.grid = None
        self._version_key = None
        self._tensor_cache = {}

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
        if grid.occupied_mask.any():
            self.clearance_field = (distance_transform_edt(~grid.occupied_mask, sampling=r)
                                    - distance_transform_edt(grid.occupied_mask, sampling=r)).astype(np.float32)
        else:
            self.clearance_field = np.full(grid.shape, math.hypot(rows, columns) * r, dtype=np.float32)
        self.sample_points = sample_body_points(self.parts, r)

    def torch_clearance(self, poses):
        return self.torch_metrics(poses)["clearance"]

    def torch_metrics(self, poses):
        if self.grid is None:
            raise RuntimeError("规划代价需要二维地图")
        key = (poses.device, poses.dtype)
        if key not in self._tensor_cache:
            self._tensor_cache[key] = (torch.as_tensor(self.sample_points, device=poses.device, dtype=poses.dtype),
                                       torch.as_tensor(self.clearance_field, device=poses.device, dtype=poses.dtype),
                                       torch.as_tensor(self.grid.occupied_mask, device=poses.device),
                                       torch.as_tensor(self.grid.unknown_mask & ~self.grid.occupied_mask, device=poses.device))
        points, field, occupied_field, unknown_field = self._tensor_cache[key]
        c, s = torch.cos(poses[..., 2, None]), torch.sin(poses[..., 2, None])
        x = poses[..., 0, None] + c * points[:, 0] - s * points[:, 1]
        y = poses[..., 1, None] + s * points[:, 0] + c * points[:, 1]
        r = self.grid.resolution_m
        columns = torch.floor((x - self.grid.origin_x_m) / r).long()
        rows = torch.floor((y - self.grid.origin_y_m) / r).long()
        height, width = self.grid.shape
        inside = (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
        rr, cc = rows.clamp(0, height - 1), columns.clamp(0, width - 1)
        values = field[rr, cc] - math.sqrt(2) * r / 2
        xmin, ymin, xmax, ymax = self.bounds
        boundary = torch.minimum(torch.minimum(x - xmin, xmax - x), torch.minimum(y - ymin, ymax - y))
        clearance = torch.minimum(values, boundary).amin(dim=-1)
        return {"clearance": clearance, "penetration": torch.relu(-clearance),
                "occupied_fraction": (inside & occupied_field[rr, cc]).to(poses.dtype).mean(dim=-1),
                "unknown_fraction": (inside & unknown_field[rr, cc]).to(poses.dtype).mean(dim=-1),
                "outside_fraction": (~inside).to(poses.dtype).mean(dim=-1)}


class SamplingNavigationController:
    def __init__(self, *, method, hardware, footprint, physics_dt, config_path, logger):
        self.config = load_config(Path(config_path))
        self.method, self.hardware, self.logger = method, hardware, logger
        torch.set_num_threads(1)
        limits = MotionLimits(**self.config["motion"])
        period_steps = max(1, math.ceil(self.config["planner"]["dt_sec"] / physics_dt - 1.0e-8))
        self.planning_period = period_steps * physics_dt
        self.model = FourWheelModel(hardware, limits, self.planning_period, device=self.config["planner"]["device"], integration_dt_sec=physics_dt)
        self.execution_model = FourWheelModel(hardware, limits, physics_dt, integration_dt_sec=physics_dt)
        sampling_footprint = replace(footprint, margin_m=float(self.config["safety"]["margin_m"]))
        sampling_footprint.validate()
        self.geometry = PlanningGridGeometry(sampling_footprint)
        self.cost = PassageCost(self.geometry, self.config["cost"], device=self.model.device)
        self.reset()

    def reset(self):
        self.planner = None
        self.goal = None
        self.last_plan = None
        self.last_command = np.zeros(2, dtype=np.float32)
        self.requested = np.zeros(2, dtype=np.float32)
        self.last_planning_time = -float("inf")
        self.status = "IDLE"

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
            if now - self.last_planning_time >= self.planning_period - 1.0e-8:
                self.last_planning_time = now
                self.last_plan = self.planner.plan(state)
                self.requested = self.last_plan.command.copy()
                self.status = "TRACKING" if self.last_plan.valid else self.last_plan.reason
                self.logger.info("[SamplingPlan] mode=%s map_version=%d status=%s latency=%.3fs clearance=%.3fm goal_world=%s diagnostics=%s",
                                 self.method, grid.map_version, self.status, self.last_plan.planning_sec,
                                 self.last_plan.min_clearance_m, goal.tolist(), self.last_plan.diagnostics)
        with torch.no_grad():
            predicted = self.execution_model.step(state, self.requested).numpy()
        motors, steering = {}, {}
        for index, wheel in enumerate(self.hardware.wheels):
            motors[wheel.drive_joint] = float(predicted[10 + index] / wheel.radius_m / wheel.drive_sign)
            steering[wheel.steering_joint] = float(predicted[6 + index] / wheel.steering_sign)
        drive.apply_continuous_joint_targets(motors, steering)
        self.last_command = predicted[14:16].copy()
        return VelocityCommand(float(self.last_command[0]), float(self.last_command[1]))
