from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import queue
import threading
import time

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
        self._point_tensor_cache = {}
        self._sample_resolution = None

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
        if self._sample_resolution != r:
            self.sample_points = sample_body_points(self.parts, r)
            self._sample_resolution = r
            self._point_tensor_cache.clear()

    def torch_clearance(self, poses):
        return self.torch_metrics(poses)["clearance"]

    def torch_metrics(self, poses):
        if self.grid is None:
            raise RuntimeError("规划代价需要二维地图")
        key = (poses.device, poses.dtype)
        if key not in self._tensor_cache:
            if key not in self._point_tensor_cache:
                self._point_tensor_cache[key] = torch.as_tensor(self.sample_points, device=poses.device, dtype=poses.dtype)
            self._tensor_cache[key] = (self._point_tensor_cache[key],
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


class _PlanningWorker:
    def __init__(self, owner):
        self.owner = owner
        self.model = self.cost = self.planner = None
        self.goal = self.epoch = self.previous_time = None
        self.results = queue.Queue()
        self.lock = threading.Lock()
        self.busy = False
        self.closed = False
        self.thread = None

    def submit(self, job):
        with self.lock:
            if self.busy or self.closed:
                return False
            self.busy = True
        self.thread = threading.Thread(target=self._run, args=(job,), daemon=True)
        self.thread.start()
        return True

    def _run(self, job):
        epoch, state, grid, goal, timestamp = job
        try:
            owner = self.owner
            if self.model is None:
                requested = owner.config["planner"]["device"]
                if requested == "auto":
                    requested = f"cuda:{1 if torch.cuda.device_count() > 1 else 0}" if torch.cuda.is_available() else "cpu"
                self.model = FourWheelModel(owner.hardware, owner.limits, owner.planning_period,
                                           device=requested, integration_dt_sec=owner.physics_dt)
                self.cost = PassageCost(owner.geometry, owner.config["cost"], device=self.model.device)
                owner.logger.info("[SamplingDevice] mode=%s device=%s", owner.method, self.model.device)
            if self.epoch != epoch or self.goal is None or np.linalg.norm(goal - self.goal) > 0.01:
                self.goal, self.epoch, self.previous_time = goal.copy(), epoch, None
                self.cost.set_target(goal)
                self.planner = SamplingPlanner(owner.method, self.model, self.cost, owner.config["planner"], owner.config["safety"])
            owner.geometry.update_grid(grid)
            elapsed = None if self.previous_time is None else timestamp - self.previous_time
            result = self.planner.plan(state, elapsed_sec=elapsed)
            self.previous_time = timestamp
            self.results.put((epoch, timestamp, grid.map_version, result, None))
        except Exception as error:
            self.results.put((epoch, timestamp, grid.map_version, None, error))
        finally:
            with self.lock:
                self.busy = False

    def close(self, wait=False):
        with self.lock:
            self.closed = True
        if wait and self.thread is not None:
            self.thread.join()


class SamplingNavigationController:
    def __init__(self, *, method, hardware, footprint, physics_dt, config_path, logger):
        self.config = load_config(Path(config_path))
        self.method, self.hardware, self.logger = method, hardware, logger
        torch.set_num_threads(1)
        self.limits = MotionLimits(**self.config["motion"])
        self.physics_dt = float(physics_dt)
        period_steps = max(1, math.ceil(self.config["planner"]["dt_sec"] / physics_dt - 1.0e-8))
        self.planning_period = period_steps * physics_dt
        self.execution_model = FourWheelModel(hardware, self.limits, physics_dt, integration_dt_sec=physics_dt)
        sampling_footprint = replace(footprint, margin_m=float(self.config["safety"]["margin_m"]))
        sampling_footprint.validate()
        self.geometry = PlanningGridGeometry(sampling_footprint)
        self._worker = _PlanningWorker(self)
        self._epoch = 0
        self.reset()

    def reset(self):
        self._epoch += 1
        self.planner = None
        self.goal = None
        self.last_plan = None
        self.last_command = np.zeros(2, dtype=np.float32)
        self.requested = np.zeros(2, dtype=np.float32)
        self.last_planning_time = -float("inf")
        self.plan_timestamp = None
        self.status = "IDLE"

    def close(self, wait=False):
        self.reset()
        self._worker.close(wait=wait)

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
            self.reset()
            self.goal = goal.copy()
        while not self._worker.results.empty():
            epoch, timestamp, version, result, error = self._worker.results.get_nowait()
            if epoch != self._epoch:
                continue
            if error is not None:
                self.logger.error("[SamplingError] mode=%s error=%s", self.method, error, exc_info=(type(error), error, error.__traceback__))
                self.status = "PLANNING_ERROR"
                continue
            self.last_plan, self.plan_timestamp = result, timestamp
            self.planner = self._worker.planner
            self.logger.info("[SamplingPlan] mode=%s map_version=%d latency=%.3fs age=%.3fs diagnostics=%s",
                             self.method, version, result.planning_sec, now - timestamp, result.diagnostics)
        if grid is None:
            self.requested[:] = 0
            self.status = "NO_MAP"
        else:
            if not self._worker.busy and now - self.last_planning_time >= self.planning_period - 1.0e-8:
                snapshot = replace(grid, occupied_mask=grid.occupied_mask.copy(), free_mask=grid.free_mask.copy(),
                                   unknown_mask=grid.unknown_mask.copy(), clearance_m=grid.clearance_m.copy())
                if self._worker.submit((self._epoch, state.copy(), snapshot, goal.copy(), float(now))):
                    self.last_planning_time = now
            self.requested[:] = 0
            self.status = "PLANNING"
            if self.last_plan is not None and self.last_plan.valid:
                index = math.floor((now - self.plan_timestamp) / self.planning_period + 1.0e-8)
                if 0 <= index < len(self.last_plan.controls):
                    self.requested = self.last_plan.controls[index].copy()
                    self.status = "TRACKING"
                else:
                    self.status = "WAITING_FOR_PLAN"
        with torch.no_grad():
            predicted = self.execution_model.step(state, self.requested).numpy()
        motors, steering = {}, {}
        for index, wheel in enumerate(self.hardware.wheels):
            motors[wheel.drive_joint] = float(predicted[10 + index] / wheel.radius_m / wheel.drive_sign)
            steering[wheel.steering_joint] = float(predicted[6 + index] / wheel.steering_sign)
        drive.apply_continuous_joint_targets(motors, steering)
        self.last_command = predicted[14:16].copy()
        return VelocityCommand(float(self.last_command[0]), float(self.last_command[1]))
