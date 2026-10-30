from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import math
import multiprocessing
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch

from constraint_traverse.benchmark.model import FourWheelModel, MotionLimits, STATE_SIZE, read_steering_targets, read_wheel_targets
from constraint_traverse.benchmark.geometry import sample_body_points
from constraint_traverse.benchmark.planner import PassageCost, SamplingPlanner
from constraint_traverse.benchmark.run import load_config
from constraint_traverse.collect.transforms import pose_xyyaw
from constraint_traverse.reference_path import ReferencePath, ReferenceWindow
from constraint_traverse.lifecycle import exit_with_parent
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


@dataclass(frozen=True)
class _SamplingJob:
    epoch: int
    state: np.ndarray
    grid: object
    goal: np.ndarray
    timestamp: float
    window: ReferenceWindow | None
    committed_controls: np.ndarray
    previous_activation_time: float | None


_PROCESS_WORKER = None


def _initialize_sampling_process(spec):
    global _PROCESS_WORKER
    torch.set_num_threads(1)
    owner = SimpleNamespace(**spec)
    owner.geometry = PlanningGridGeometry(owner.footprint)
    owner.logger = logging.getLogger("constraint_traverse.sampling")
    _PROCESS_WORKER = _PlanningWorker(owner)


def _sampling_process_main(spec, requests, replies, stop_requested):
    exit_with_parent()
    replies.cancel_join_thread()
    if stop_requested.is_set():
        return
    _initialize_sampling_process(spec)
    while not stop_requested.is_set():
        try:
            job = requests.get(timeout=0.1)
        except queue.Empty:
            continue
        if job is None:
            return
        _PROCESS_WORKER._run(job)
        result = _PROCESS_WORKER.results.get_nowait()
        if not stop_requested.is_set():
            replies.put(result)


class _PlanningWorker:
    def __init__(self, owner):
        self.owner = owner
        self.model = self.cost = self.planner = None
        self.goal = self.epoch = self.previous_time = None
        self.results = queue.Queue()
        self.lock = threading.Lock()
        self.busy = False
        self.closed = False
        self.process = None
        self.requests = self.replies = self.stop_requested = None

    def submit(self, job):
        with self.lock:
            if self.busy or self.closed:
                return False
            self.busy = True
            try:
                if self.process is None:
                    spec = {name: getattr(self.owner, name) for name in
                            ("method", "hardware", "config", "limits", "planning_period", "physics_dt", "footprint")}
                    context = multiprocessing.get_context("spawn")
                    self.requests, self.replies = context.Queue(), context.Queue()
                    self.stop_requested = context.Event()
                    self.process = context.Process(target=_sampling_process_main,
                        args=(spec, self.requests, self.replies, self.stop_requested),
                        name=f"navigation-{self.owner.method}", daemon=True)
                    self.process.start()
                self.requests.put_nowait(job)
            except Exception as error:
                self.results.put((job.epoch, job.timestamp, job.grid.map_version, None, error, job.state))
                self.busy = False
        return True

    def poll(self):
        if self.replies is None or self.closed:
            return
        while True:
            try:
                result = self.replies.get_nowait()
            except queue.Empty:
                break
            self.results.put(result)
            with self.lock:
                self.busy = False

    def _run(self, job):
        epoch, state, grid, goal, timestamp = job.epoch, job.state, job.grid, job.goal, job.timestamp
        predicted_start = state.copy()
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
            with torch.no_grad():
                predicted = torch.tensor(state, device=self.model.device)
                for control in job.committed_controls:
                    predicted = self.model._physical_step(predicted, torch.tensor(control, device=self.model.device), owner.physics_dt)
                predicted_start = predicted.cpu().numpy()
            prediction_time = timestamp + len(job.committed_controls) * owner.physics_dt
            if job.window is not None:
                self.cost.set_target(job.window.world_xy[-1])
                self.cost.set_reference_window(job.window)
            else:
                self.cost.set_target(goal)
            elapsed = None if self.previous_time is None else max(0.0, prediction_time - self.previous_time)
            result = self.planner.plan(predicted_start, elapsed_sec=elapsed)
            result.diagnostics["prediction_timestamp_sec"] = float(prediction_time)
            self.previous_time = prediction_time
            self.results.put((epoch, timestamp, grid.map_version, result, None, predicted_start))
        except Exception as error:
            self.results.put((epoch, timestamp, grid.map_version, None, error, predicted_start))
        finally:
            with self.lock:
                self.busy = False

    def close(self, wait=False):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            process = self.process
            if self.stop_requested is not None:
                self.stop_requested.set()
            if self.requests is not None:
                self.requests.put_nowait(None)
        try:
            if process is not None and process.pid is not None:
                process.join(timeout=2.0 if wait else 0.1)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1.0)
                if process.is_alive():
                    raise RuntimeError(f"规划子进程未能退出：{process.pid}")
                process.close()
        finally:
            for channel in (self.requests, self.replies):
                if channel is not None:
                    channel.cancel_join_thread()
                    channel.close()
            self.process = None
            self.requests = self.replies = self.stop_requested = None
            self.busy = False


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
        self.footprint = sampling_footprint
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
        self.reference = None
        self.reference_cursor = 1
        self._segment_key = None
        self._delay_estimate = 0.0
        self.max_result_age_sec = float(self.config["planner"].get("max_result_age_sec", float("inf")))
        self.command_blocked = False
        self.response_serial = 0
        self.status = "IDLE"

    def _control_for_time(self, now):
        if self.last_plan is None or not self.last_plan.valid or self.plan_timestamp is None or self.command_blocked:
            return np.zeros(2, dtype=np.float32)
        requested_at = self.last_plan.diagnostics.get("request_timestamp_sec", self.plan_timestamp)
        if now - requested_at > self.max_result_age_sec:
            return np.zeros(2, dtype=np.float32)
        index = math.floor((now - self.plan_timestamp) / self.planning_period + 1e-8)
        return self.last_plan.controls[index].copy() if 0 <= index < len(self.last_plan.controls) else np.zeros(2, dtype=np.float32)

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
        state[17:21] = read_steering_targets(drive, self.hardware, state[6:10])
        state[21:25] = read_wheel_targets(drive, self.hardware, state[10:14])
        return state

    def step(self, *, robot, drive, grid, goal_world_xy, now, reference_path: ReferencePath | None = None):
        state = self.read_state(robot, drive)
        goal = np.asarray(goal_world_xy, dtype=np.float32).reshape(2)
        if self.goal is None or np.linalg.norm(goal - self.goal) > 0.01:
            self.reset()
            self.goal = goal.copy()
        window = None
        if reference_path is not None:
            if np.linalg.norm(reference_path.goal_world_xy - goal) > 0.01:
                raise ValueError("参考路径所属目标与当前导航目标不一致")
            if reference_path is not self.reference:
                self.reference, self.reference_cursor = reference_path, 1
            window = reference_path.window(state[:3], lookahead_m=self.config["planner"]["reference_lookahead_m"],
                cursor=self.reference_cursor, stopped=(np.linalg.norm(state[3:5]) < 0.025
                    and abs(state[5]) < 0.04 and np.max(np.abs(state[10:14])) < 0.025))
            if window is not None:
                self.reference_cursor = window.cursor
                segment_key = (reference_path.source, window.direction)
                if self._segment_key is not None and segment_key != self._segment_key:
                    self._epoch += 1
                    self.last_plan = self.plan_timestamp = None
                    self.last_planning_time = -float("inf")
                self._segment_key = segment_key
        self._worker.poll()
        while not self._worker.results.empty():
            epoch, timestamp, version, result, error, predicted_start = self._worker.results.get_nowait()
            self.response_serial += 1
            if epoch != self._epoch:
                continue
            if error is not None:
                self.logger.error("[SamplingError] mode=%s error=%s", self.method, error, exc_info=(type(error), error, error.__traceback__))
                self.status = "PLANNING_ERROR"
                continue
            observed_delay = max(0.0, now - timestamp)
            self._delay_estimate = min(self.planning_period * len(result.controls) / 2,
                                       min(observed_delay, self.planning_period) if self._delay_estimate == 0
                                       else 0.5 * (self._delay_estimate + observed_delay))
            result.diagnostics["connection_position_error_m"] = float(np.linalg.norm(state[:2] - predicted_start[:2]))
            result.diagnostics["connection_speed_error_mps"] = float(abs(state[3] - predicted_start[3]))
            result.diagnostics["connection_steering_error_rad"] = float(np.abs(state[6:10] - predicted_start[6:10]).max())
            result.diagnostics["connection_yaw_error_rad"] = abs(float((state[2] - predicted_start[2] + math.pi) % (2 * math.pi) - math.pi))
            result.diagnostics["request_timestamp_sec"] = float(timestamp)
            result.diagnostics["map_version"] = int(version)
            limits = self.config["planner"]
            disconnected = (result.diagnostics["connection_position_error_m"] > limits.get("connection_position_m", 0.05)
                or result.diagnostics["connection_yaw_error_rad"] > limits.get("connection_yaw_rad", math.radians(5))
                or result.diagnostics["connection_speed_error_mps"] > limits.get("connection_speed_mps", 0.15)
                or result.diagnostics["connection_steering_error_rad"] > limits.get("connection_steering_rad", 0.15))
            if disconnected:
                self.logger.warning("[SamplingConnectionRejected] diagnostics=%s", result.diagnostics)
                self.last_planning_time = -float("inf")
                continue
            self.last_plan, self.plan_timestamp = result, now
            self.command_blocked = False
            self.planner = self._worker.planner
            result.diagnostics["request_timestamp_sec"] = float(timestamp)
            result.diagnostics["map_version"] = int(version)
            self.logger.info("[SamplingPlan] mode=%s map_version=%d latency=%.3fs age=%.3fs diagnostics=%s",
                             self.method, version, result.planning_sec, now - timestamp, result.diagnostics)
        if grid is None:
            self.requested[:] = 0
            self.status = "NO_MAP"
        elif reference_path is not None and window is None:
            self.requested[:] = 0
            self.status = "REFERENCE_DONE"
            if self.last_plan is not None:
                self._epoch += 1
                self.last_plan = self.plan_timestamp = None
        else:
            if not self._worker.busy and now - self.last_planning_time >= self.planning_period - 1.0e-8:
                snapshot = replace(grid, occupied_mask=grid.occupied_mask.copy(), free_mask=grid.free_mask.copy(),
                                   unknown_mask=grid.unknown_mask.copy(), clearance_m=grid.clearance_m.copy())
                controls = []
                for index in range(math.ceil(self._delay_estimate / self.physics_dt)):
                    controls.append(self._control_for_time(now + index * self.physics_dt))
                job = _SamplingJob(self._epoch, state.copy(), snapshot, goal.copy(), float(now), window,
                                   np.asarray(controls, dtype=np.float32).reshape(-1, 2), self.plan_timestamp)
                if self._worker.submit(job):
                    self.last_planning_time = now
            self.requested[:] = 0
            self.status = "PLANNING"
            if self.last_plan is not None and self.last_plan.valid:
                index = math.floor((now - self.plan_timestamp) / self.planning_period + 1.0e-8)
                if 0 <= index < len(self.last_plan.controls):
                    self.requested = self._control_for_time(now)
                    self.status = "CONTROL_BLOCKED" if self.command_blocked else "TRACKING"
                else:
                    self.status = "WAITING_FOR_PLAN"
            elif self.last_plan is not None:
                self.status = self.last_plan.reason
        return self.apply_command(robot=robot, drive=drive, command=self.requested, state=state)

    def apply_command(self, *, robot, drive, command, state=None, blocked=False):
        if state is None:
            state = self.read_state(robot, drive)
        self.requested = np.asarray(command, dtype=np.float32).reshape(2).copy()
        if blocked:
            self.command_blocked = True
        with torch.no_grad():
            rims, angles, filtered = self.execution_model.actuator_targets(state, self.requested)
            rims, angles, filtered = rims.numpy(), angles.numpy(), filtered.numpy()
        motors, steering = {}, {}
        for index, wheel in enumerate(self.hardware.wheels):
            motors[wheel.drive_joint] = float(rims[index] / wheel.radius_m / wheel.drive_sign)
            steering[wheel.steering_joint] = float(angles[index] / wheel.steering_sign)
        drive.apply_continuous_joint_targets(motors, steering)
        self.motor_targets, self.steering_targets = motors, steering
        self.last_command = filtered.copy()
        return VelocityCommand(float(self.last_command[0]), float(self.last_command[1]))
