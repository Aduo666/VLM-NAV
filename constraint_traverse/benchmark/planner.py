from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch
from pytorch_mppi import mppi

from .geometry import PassageGeometry
from .model import FourWheelModel, STATE_SIZE


class PassageCost:
    def __init__(self, geometry: PassageGeometry, settings: dict, *, device="cpu"):
        self.geometry, self.settings = geometry, settings
        self.device = torch.device(device)
        self.goal = torch.zeros(2, device=self.device)
        self.reference = None

    def set_target(self, goal, reference=None):
        self.goal = torch.as_tensor(goal, dtype=torch.float32, device=self.device)
        if self.goal.shape != (2,) or not torch.isfinite(self.goal).all():
            raise ValueError("目标必须为有效的世界二维坐标")
        self.reference = None if reference is None else torch.as_tensor(reference, dtype=torch.float32, device=self.device)

    def running(self, state, action):
        cfg = self.settings
        delta = self.goal - state[..., :2]
        distance_squared = delta.square().sum(dim=-1)
        bearing = torch.atan2(delta[..., 1], delta[..., 0])
        heading = torch.atan2(torch.sin(bearing - state[..., 2]), torch.cos(bearing - state[..., 2]))
        gap = self.geometry.torch_clearance(state[..., :3])
        invalid = (gap <= self.geometry.margin_m) | (state[..., 16] > 0.5)
        cost = cfg["goal"] * distance_squared
        cost = cost + cfg["heading"] * heading.square() * (distance_squared > 0.30 ** 2)
        cost = cost + cfg["control"] * action.square().sum(dim=-1)
        cost = cost + cfg["obstacle"] * torch.relu(cfg["obstacle_soft_distance_m"] + self.geometry.margin_m - gap).square()
        cost = cost + cfg["invalid"] * invalid.float()
        if self.reference is not None:
            lateral = torch.cdist(state[..., :2], self.reference).amin(dim=-1)
            cost = cost + cfg["reference_path"] * lateral.square()
        return cost

    def terminal(self, states, actions):
        return self.settings["terminal_goal"] * (states[..., -1, :2] - self.goal).square().sum(dim=-1)

    def trajectory(self, states, actions):
        return self.running(states, actions).sum(dim=-1) + self.terminal(states, actions)


@dataclass
class PlanResult:
    command: np.ndarray
    controls: np.ndarray
    prediction: np.ndarray
    candidates: np.ndarray
    candidate_controls: np.ndarray
    candidate_costs: np.ndarray
    safe: bool
    reason: str
    min_clearance_m: float
    planning_sec: float
    optimizer_sec: float
    validation_sec: float


class SamplingPlanner:
    def __init__(self, method, model: FourWheelModel, cost: PassageCost, settings: dict, safety: dict):
        if method not in {"MPPI", "KMPPI"}:
            raise ValueError(f"当前交互程序支持 MPPI/KMPPI：{method}")
        self.method, self.model, self.cost, self.settings, self.safety = method, model, cost, settings, safety
        self.seed = int(settings["seed"])
        self._cuda_indices = [model.device.index or 0] if model.device.type == "cuda" else []
        with torch.random.fork_rng(devices=self._cuda_indices):
            torch.manual_seed(self.seed)
            horizon = int(settings["horizon_steps"])
            initial = torch.zeros(horizon, 2, device=model.device)
            initial[:, 0] = min(0.35, model.limits.max_linear_mps)
            kwargs = dict(
                dynamics=model.step, running_cost=cost.running, nx=STATE_SIZE,
                noise_sigma=torch.diag(torch.tensor([
                    float(settings["noise_linear_std"]) ** 2,
                    float(settings["noise_angular_std"]) ** 2,
                ], device=model.device)),
                num_samples=int(settings["samples"]), horizon=horizon, device=model.device,
                lambda_=float(settings["temperature"]), terminal_state_cost=cost.terminal,
                u_min=torch.tensor([-model.limits.max_linear_mps, -model.limits.max_angular_radps], device=model.device),
                u_max=torch.tensor([model.limits.max_linear_mps, model.limits.max_angular_radps], device=model.device),
                U_init=initial, sample_null_action=True,
            )
            if method == "MPPI":
                self.controller = mppi.MPPI(**kwargs)
            else:
                self.controller = mppi.KMPPI(
                    **kwargs, num_support_pts=int(settings["kernel_support_points"]),
                    kernel=mppi.RBFKernel(sigma=float(settings["kernel_sigma"])),
                )
                self.controller.theta[:, 0] = initial[0, 0]
            self.rng_cpu = torch.get_rng_state().clone()
            self.rng_cuda = torch.cuda.get_rng_state(model.device).clone() if self._cuda_indices else None
        self.first_command = True

    def _synchronize(self):
        if self._cuda_indices:
            torch.cuda.synchronize(self.model.device)

    def plan(self, state) -> PlanResult:
        self._synchronize()
        started = time.perf_counter()
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.model.device)
        with torch.no_grad(), torch.random.fork_rng(devices=self._cuda_indices):
            torch.set_rng_state(self.rng_cpu)
            if self.rng_cuda is not None:
                torch.cuda.set_rng_state(self.rng_cuda, self.model.device)
            for iteration in range(int(self.settings["iterations"])):
                self.controller.command(tensor, shift_nominal_trajectory=(iteration == 0 and not self.first_command))
            self.first_command = False
            self.rng_cpu = torch.get_rng_state().clone()
            self.rng_cuda = torch.cuda.get_rng_state(self.model.device).clone() if self._cuda_indices else None
            controls = self.controller.get_action_sequence().clone()
            prediction = self.model.rollout(tensor, controls)
            candidates = self.controller.states[0, :, :, :3].cpu().numpy()
            candidate_controls = self.controller.perturbed_action.cpu().numpy()
            candidate_costs = self.controller.cost_total.cpu().numpy()
            self._synchronize()
            optimized = time.perf_counter()
            poses = prediction[:, :3].cpu().numpy()
            validation = self.cost.geometry.validate_poses(poses)
            reason = validation.reason
            safe = validation.safe and bool(torch.all(prediction[:, 16] < 0.5))
            if not safe and reason == "SAFE":
                reason = "MOTION_CONSTRAINT"
            # 完整预测序列和本次短动作后的制动区域分别复核。
            braking = self.model.braking_rollout(prediction[1], self.safety["reaction_time_sec"])
            brake_check = self.cost.geometry.validate_poses(braking[:, :3].cpu().numpy())
            if safe and not brake_check.safe:
                safe, reason = False, "UNSAFE_BRAKING"
            command = controls[0].cpu().numpy() if safe else np.zeros(2, dtype=np.float32)
        finished = time.perf_counter()
        return PlanResult(
            command, controls.cpu().numpy(), poses, candidates, candidate_controls, candidate_costs,
            safe, reason, min(validation.min_clearance_m, brake_check.min_clearance_m),
            finished - started, optimized - started, finished - optimized,
        )
