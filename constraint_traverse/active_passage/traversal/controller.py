from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from constraint_traverse.lifecycle import ThreadWorker
from constraint_traverse.navigation import SamplingNavigationController
from vln_nav.navigation.trajectory_clearance import sample_swept_path

from .planner import PlanningFailure, SmacLatticePlanner


class PathWorker(ThreadWorker):
    def __init__(self, planner):
        super().__init__()
        self.planner = planner

    def _run(self, job):
        epoch, grid, pose, goal, request, timestamp = job
        try:
            path, diagnostics = self.planner.plan(grid, pose, goal, request_id=request, timestamp=timestamp)
            self._results.put((epoch, path, diagnostics, None))
        except Exception as error:
            self._results.put((epoch, None, {}, error))
        finally:
            with self._lock:
                self._busy = False

    def close(self, wait=False):
        super().close(wait=False)
        if wait and self._thread is not None:
            self._thread.join(timeout=float(self.planner.settings["search_time_sec"]) + 2)
            if self._thread.is_alive():
                raise RuntimeError("Smac 搜索线程没有按时退出")


def load_navigation_config(path, root):
    with Path(path).open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    if set(settings) != {"path", "control", "arrival"}:
        raise ValueError("导航配置需要包含 path、control 和 arrival")
    for section in settings.values():
        for name, value in section.items():
            if name.endswith(("_sec", "_m", "_mps", "_radps")):
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"导航参数必须为有限正数：{name}")
    for section, name in (("path", "primitive_file"), ("control", "config_path")):
        target = Path(settings[section][name]).expanduser()
        if not target.is_absolute():
            target = Path(root) / target
        if not target.is_file():
            raise ValueError(f"导航依赖文件不存在：{target}")
        settings[section][name] = str(target.resolve())
    return settings


class AutomaticTraversal:
    def __init__(self, *, footprint, hardware, physics_dt, settings, seed, gpu, logger):
        self.settings, self.logger = settings, logger
        self.sampling = SamplingNavigationController(method="MPPI", hardware=hardware, footprint=footprint,
            physics_dt=physics_dt, config_path=settings["control"]["config_path"], logger=logger)
        self.sampling.config["planner"]["seed"] = int(seed)
        self.sampling.config["planner"]["device"] = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        self.sampling.config["planner"]["max_result_age_sec"] = settings["control"]["max_result_age_sec"]
        self.sampling.max_result_age_sec = settings["control"]["max_result_age_sec"]
        self.path_planner = SmacLatticePlanner(self.sampling.footprint, settings["path"])
        self.worker = PathWorker(self.path_planner)
        self.epoch = self.request = 0
        self.events = []
        self.active = False
        self.status = "IDLE"
        self.path = None
        self.actual = []
        self.reset()

    def reset(self, last_command=None):
        self.epoch += 1
        self.active, self.path, self.goal = False, None, None
        self.sampling.reset()
        if last_command is not None:
            self.sampling.last_command = np.asarray(last_command, dtype=np.float32).copy()
        self.signature = self.failed_key = None
        self.last_path_time = -math.inf
        self.stopped_since = self.busy_since = None
        self.reply_serial = 0
        self.seen_plan = None
        self.actual, self.events = [], []
        self.status = "IDLE"

    def begin(self, goal):
        previous = self.sampling.last_command.copy()
        self.reset(previous)
        self.goal = np.asarray(goal, dtype=float).reshape(2).copy()
        if not np.isfinite(self.goal).all():
            raise ValueError("导航目标必须包含有限坐标")
        self.active, self.status = True, "PATH_PLANNING"

    def _status(self, name, now, **details):
        if name != self.status:
            self.status = name
            self.events.append(("NAVIGATION_STATUS", dict(status=name, **details)))
            self.logger.info("[Traversal] t=%.3f status=%s details=%s", now, name, details)

    def _check_poses(self, grid, poses):
        if len(poses) >= 2:
            poses = sample_swept_path(poses[:, :2], poses[:, 2], max_step_m=0.01, max_yaw_step_deg=1)
        for pose in poses:
            self.path_planner._check_pose(grid, pose)

    def check_control(self, grid, state, command):
        model = self.sampling.execution_model
        count = math.ceil(self.settings["control"]["validation_period_sec"] / model.dt_sec)
        with torch.no_grad():
            prefix = model.rollout(state, np.tile(command, (count, 1)))
            stop = model.braking_rollout(prefix[-1], self.sampling.config["safety"]["reaction_time_sec"])
            poses = torch.cat((prefix[:, :3], stop[1:, :3])).numpy()
        self._check_poses(grid, poses)

    def _update_path(self, grid, state, now):
        result = self.worker.poll()
        if result is not None and result[0] == self.epoch:
            _, path, diagnostics, error = result
            if error is None:
                self.path = path
                self.signature = path.map_signature(grid, radius_m=self.path_planner.checker.bounding_radius_m)
                self.events.append(("REFERENCE_PATH", dict(path=path, diagnostics=diagnostics)))
                self.failed_key = None
            elif isinstance(error, PlanningFailure):
                self._status(error.reason, now, detail=error.detail)
            else:
                self._status("PLANNING_ERROR", now, detail=str(error))
                self.logger.error("Smac 搜索失败：%s", error)
        if self.path is not None:
            signature = self.path.map_signature(grid, radius_m=self.path_planner.checker.bounding_radius_m)
            if signature != self.signature:
                window = self.path.window(state[:3], lookahead_m=1000, cursor=self.sampling.reference_cursor)
                try:
                    if window is not None:
                        self._check_poses(grid, np.column_stack((window.world_xy, window.world_yaw_rad)))
                    self.signature = signature
                except PlanningFailure as error:
                    self.path = None
                    self.sampling.reset()
                    self._status(error.reason, now, detail=error.detail)
        if self.path is None and not self.worker.busy:
            # 失败搜索只在地图证据或机器人位姿变化后重试；预算耗尽可以继续搜索。
            key = (grid.map_version, *np.round(state[:3], 2))
            retry = self.status in {"NO_PATH", "SEARCH_TIMEOUT", "PATH_PLANNING"}
            if (key != self.failed_key or retry) and now - self.last_path_time >= self.settings["path"]["replan_period_sec"]:
                self.request += 1
                copied = replace(grid, occupied_mask=grid.occupied_mask.copy(), free_mask=grid.free_mask.copy(),
                                 unknown_mask=grid.unknown_mask.copy(), clearance_m=grid.clearance_m.copy())
                if self.worker.submit((self.epoch, copied, state[:3].copy(), self.goal.copy(), self.request, now)):
                    self.last_path_time, self.failed_key = now, key

    def step(self, *, robot, drive, grid, now):
        state = self.sampling.read_state(robot, drive)
        blocked = False
        self.actual.append(state[:3].copy())
        arrival = self.settings["arrival"]
        reached = np.linalg.norm(state[:2] - self.goal) <= arrival["position_tolerance_m"]
        if reached:
            command = np.zeros(2)
            stopped = (np.linalg.norm(state[3:5]) <= arrival["linear_velocity_mps"]
                       and abs(state[5]) <= arrival["angular_velocity_radps"])
            self.stopped_since = (now if self.stopped_since is None else self.stopped_since) if stopped else None
            self._status("STOPPING_AT_GOAL", now)
            if self.stopped_since is not None and now - self.stopped_since >= arrival["stopped_hold_sec"]:
                self.active = False
                self._status("SUCCESS", now)
        else:
            self.stopped_since = None
            self._update_path(grid, state, now)
            command = np.zeros(2)
            if self.path is not None:
                # 关节目标在本物理周期末统一提交，安全检查前不会推进仿真。
                self.sampling.step(robot=robot, drive=drive, grid=grid, goal_world_xy=self.goal,
                                   now=now, reference_path=self.path)
                serial = getattr(self.sampling, "response_serial", 0)
                if serial != getattr(self, "reply_serial", 0):
                    self.reply_serial, self.busy_since = serial, None
                command = self.sampling.requested.copy()
                if self.sampling.last_plan is not None:
                    plan = self.sampling.last_plan
                    if plan is not self.seen_plan:
                        self.seen_plan = plan
                        self.busy_since = None
                        self.events.append(("CONTROL_PLAN", dict(plan=plan, activation_timestamp_sec=now,
                            reference_request_id=self.path.request_id, map_version=grid.map_version)))
                    age = now - plan.diagnostics.get("request_timestamp_sec", now)
                    if age > self.settings["control"]["max_result_age_sec"]:
                        command[:] = 0
                        blocked = True
                        self._status("STALE_CONTROL", now, age_sec=age)
                    else:
                        self._status(self.sampling.status, now)
                elif self.sampling.status == "PLANNING_ERROR":
                    self._status("PLANNING_ERROR", now)
                else:
                    self._status("CONTROL_PLANNING", now)
                if self.sampling._worker.busy:
                    self.busy_since = time.monotonic() if self.busy_since is None else self.busy_since
                    limit = ("startup_timeout_sec" if self.seen_plan is None else "response_timeout_sec")
                    if time.monotonic() - self.busy_since > self.settings["control"][limit]:
                        command[:] = 0
                        blocked = True
                        self.active = False
                        self._status("CONTROL_TIMEOUT", now)
                else:
                    self.busy_since = None
        try:
            self.check_control(grid, state, command)
        except PlanningFailure as error:
            command[:] = 0
            blocked = True
            self._status(error.reason, now, detail=error.detail)
        # 使用提交前的实测状态，保持模型中的命令历史与最终提交动作一致。
        applied = self.sampling.apply_command(robot=robot, drive=drive, command=command, state=state, blocked=blocked)
        return command, applied

    def close(self, wait=False):
        self.active = False
        try:
            self.worker.close(wait=wait)
        finally:
            self.sampling.close(wait=wait)
