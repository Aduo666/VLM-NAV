from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time

import numpy as np
import torch
from pytorch_mppi import mppi

from .geometry import PassageGeometry
from .model import FourWheelModel, STATE_SIZE
from .kmppi import ConsistentKMPPI
from .feasible_mppi import FeasibleMPPI


RNG_LOCK = threading.Lock()


class PassageCost:
    def __init__(self, geometry: PassageGeometry, settings: dict, *, device="cpu"):
        self.geometry, self.settings = geometry, settings
        self.device = torch.device(device)
        self.goal = torch.zeros(2, device=self.device)
        self.reference = None
        self.dt_sec = 0.20
        self.action_limits = torch.tensor([0.5, 0.5], device=self.device)

    def set_target(self, goal, reference=None):
        self.goal = torch.as_tensor(goal, dtype=torch.float32, device=self.device)
        if self.goal.shape != (2,) or not torch.isfinite(self.goal).all():
            raise ValueError("目标必须为有效的世界二维坐标")
        self.reference = None if reference is None else torch.as_tensor(reference, dtype=torch.float32, device=self.device)

    def components(self, state, action, *, motion_states=None):
        cfg = self.settings
        delta = self.goal - state[..., :2]
        distance_squared = (delta / cfg["goal_distance_scale_m"]).square().sum(dim=-1)
        bearing = torch.atan2(delta[..., 1], delta[..., 0])
        heading = torch.atan2(torch.sin(bearing - state[..., 2]), torch.cos(bearing - state[..., 2]))
        checked = state if motion_states is None else motion_states
        metrics = self.geometry.torch_metrics(checked[..., :3])
        reduce = (lambda value: value) if motion_states is None else (lambda value: value.mean(dim=0))
        gap = metrics["clearance"]
        soft_distance = cfg["obstacle_soft_distance_m"]
        components = {
            "goal": cfg["goal"] * distance_squared,
            "heading": cfg["heading"] * (heading / torch.pi).square() * (delta.square().sum(dim=-1) > 0.30 ** 2),
            "control": cfg["control"] * (action / self.action_limits).square().sum(dim=-1),
            "clearance": cfg["obstacle"] * reduce(torch.relu((soft_distance + self.geometry.margin_m - gap) / soft_distance).square()),
            "penetration": cfg["penetration"] * reduce((metrics["penetration"] / cfg["penetration_scale_m"]).square()),
            "occupied": cfg["occupied"] * reduce(metrics["occupied_fraction"]),
            "unknown": cfg["unknown"] * reduce(metrics["unknown_fraction"]),
            "outside": cfg["outside"] * reduce(metrics["outside_fraction"]),
            "motion": cfg["motion_constraint"] * reduce(checked[..., 16]),
        }
        if self.reference is not None:
            lateral = torch.cdist(state[..., :2], self.reference).amin(dim=-1)
            components["reference"] = cfg["reference_path"] * (lateral / cfg["goal_distance_scale_m"]).square()
        return {name: value * self.dt_sec / cfg["reference_dt_sec"] for name, value in components.items()}

    def running(self, state, action, *, motion_states=None):
        return sum(self.components(state, action, motion_states=motion_states).values())

    def terminal(self, states, actions):
        return self.settings["terminal_goal"] * ((states[..., -1, :2] - self.goal) / self.settings["goal_distance_scale_m"]).square().sum(dim=-1)

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
    valid: bool
    reason: str
    min_clearance_m: float
    planning_sec: float
    optimizer_sec: float
    diagnostics: dict[str, float]


class SamplingPlanner:
    def __init__(self, method, model: FourWheelModel, cost: PassageCost, settings: dict, safety: dict):
        if method not in {"MPPI", "KMPPI"}:
            raise ValueError(f"当前交互程序支持 MPPI/KMPPI：{method}")
        self.method, self.model, self.cost, self.settings, self.safety = method, model, cost, settings, safety
        cost.dt_sec = model.dt_sec
        cost.action_limits = torch.tensor([model.limits.max_linear_mps, model.limits.max_angular_radps], device=model.device)
        self.last_output = np.zeros(2, dtype=np.float32)
        self.seed = int(settings["seed"])
        self._cuda_indices = [model.device.index or 0] if model.device.type == "cuda" else []
        with torch.random.fork_rng(devices=self._cuda_indices):
            torch.manual_seed(self.seed)
            horizon = int(settings["horizon_steps"])
            initial = torch.zeros(horizon, 2, device=model.device)
            kwargs = dict(
                dynamics=model.step,
                running_cost=lambda state, action: cost.running(state, action, motion_states=model.last_substeps),
                nx=STATE_SIZE,
                noise_sigma=torch.diag(torch.tensor([
                    float(settings["noise_linear_std"]) ** 2,
                    float(settings["noise_angular_std"]) ** 2,
                ], device=model.device)),
                num_samples=int(settings["samples"]), horizon=horizon, device=model.device,
                lambda_=float(settings["temperature"]), terminal_state_cost=cost.terminal,
                u_min=torch.tensor([-model.limits.max_linear_mps, -model.limits.max_angular_radps], device=model.device),
                u_max=torch.tensor([model.limits.max_linear_mps, model.limits.max_angular_radps], device=model.device),
                U_init=initial, sample_null_action=True,
                project_action=model.project_actions,
            )
            if method == "MPPI":
                self.controller = FeasibleMPPI(**kwargs)
            else:
                self.controller = ConsistentKMPPI(
                    **kwargs, num_support_pts=int(settings["kernel_support_points"]),
                    kernel=mppi.RBFKernel(sigma=float(settings["kernel_sigma"])),
                )
            self.rng_cpu = torch.get_rng_state().clone()
            self.rng_cuda = torch.cuda.get_rng_state(model.device).clone() if self._cuda_indices else None
        self.first_command = True

    def _prepare_sampling(self, tensor, elapsed_sec):
        controller = self.controller
        delta = self.cost.goal - tensor[:2]
        distance = float(torch.linalg.vector_norm(delta))
        speed_limit = min(self.model.limits.max_linear_mps, max(0.025, distance * 0.8))
        controller.u_max = torch.tensor([speed_limit, self.model.limits.max_angular_radps], device=self.model.device)
        controller.u_min = -controller.u_max
        scale = min(1.0, max(0.15, distance / self.settings.get("near_goal_distance_m", 0.8)))
        controller.lambda_ = float(self.settings["temperature"]) * scale * scale
        sigma = torch.tensor([self.settings["noise_linear_std"], self.settings["noise_angular_std"]], device=self.model.device) * scale
        controller.noise_sigma = torch.diag(sigma.square())
        controller._noise_sigma_inv_diag = sigma.reciprocal().square()
        controller._noise_sigma_sqrt_diag = sigma
        controller.noise_sigma_inv = torch.diag(controller._noise_sigma_inv_diag)
        controller._setup_action_cost_fn()
        if self.first_command:
            initial = self.model.project_actions(tensor[14:16])
            if self.method == "KMPPI":
                controller.theta[:] = initial
            else:
                controller.U[:] = initial
        else:
            steps = max(0.0, float(self.model.dt_sec if elapsed_sec is None else elapsed_sec)) / self.model.dt_sec
            if self.method == "KMPPI":
                controller.advance_nominal(steps)
            else:
                indices = torch.arange(controller.T, device=self.model.device) + steps
                lower, upper = indices.floor().long(), indices.ceil().long()
                extended = torch.cat((controller.U, controller.u_init[None]), dim=0)
                controller.U = torch.lerp(extended[lower.clamp(max=controller.T)],
                                          extended[upper.clamp(max=controller.T)],
                                          (indices - lower)[:, None])
        if self.method == "KMPPI":
            controller.synchronize_controls()
        else:
            controller.U = controller._bound_action(controller.U)
        yaw = float(tensor[2])
        dx, dy = float(delta[0]), float(delta[1])
        forward = math.cos(yaw) * dx + math.sin(yaw) * dy
        left = -math.sin(yaw) * dx + math.cos(yaw) * dy
        bearing = math.atan2(left, forward)
        duration = controller.T * self.model.dt_sec
        base_speed = min(speed_limit, distance / max(duration - 0.5, self.model.dt_sec))
        direction = -1.0 if forward < 0 else 1.0
        curvature = 2 * left / max(distance * distance, 1.0e-6)
        priors = torch.zeros(6, controller.T, 2, device=self.model.device)
        for index, factor in enumerate((0.5, 1.0, 1.5)):
            v = direction * min(speed_limit, base_speed * factor)
            priors[index, :, 0] = v
            priors[index, :, 1] = v * curvature
        priors[3, :, 0] = direction * min(speed_limit, base_speed)
        priors[4, :, 1] = float(np.sign(bearing)) * min(self.model.limits.max_angular_radps, abs(bearing) / duration)
        turn_steps = min(controller.T - 1, math.ceil(abs(bearing) / (self.model.limits.max_angular_radps * self.model.dt_sec)))
        priors[5, :turn_steps, 1] = float(np.sign(bearing)) * self.model.limits.max_angular_radps
        priors[5, turn_steps:, 0] = min(speed_limit, distance / max((controller.T - turn_steps) * self.model.dt_sec, self.model.dt_sec))
        controller.action_priors = controller._bound_action(priors)

    def _synchronize(self):
        if self._cuda_indices:
            torch.cuda.synchronize(self.model.device)

    def plan(self, state, elapsed_sec=None) -> PlanResult:
        self._synchronize()
        started = time.perf_counter()
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.model.device)
        with RNG_LOCK, torch.no_grad(), torch.random.fork_rng(devices=self._cuda_indices):
            torch.set_rng_state(self.rng_cpu)
            if self.rng_cuda is not None:
                torch.cuda.set_rng_state(self.rng_cuda, self.model.device)
            self._prepare_sampling(tensor, elapsed_sec)
            for iteration in range(int(self.settings["iterations"])):
                self.controller.command(tensor, shift_nominal_trajectory=False)
            self.first_command = False
            self.rng_cpu = torch.get_rng_state().clone()
            self.rng_cuda = torch.cuda.get_rng_state(self.model.device).clone() if self._cuda_indices else None
            controls = self.controller.get_action_sequence().clone()
            prediction, motion_states = self.model.rollout(tensor, controls, return_substeps=True)
            candidates = self.controller.states[0, :, :, :3].cpu().numpy()
            candidate_controls = self.controller.perturbed_action.cpu().numpy()
            candidate_costs = self.controller.cost_total.cpu().numpy()
            self._synchronize()
            optimized = time.perf_counter()
            poses = prediction[:, :3].cpu().numpy()
            valid = bool(torch.isfinite(controls).all() and torch.isfinite(prediction).all())
            reason = "OPTIMIZED" if valid else "INVALID_OUTPUT"
            command = controls[0].cpu().numpy() if valid else np.zeros(2, dtype=np.float32)
            clearance = float(self.cost.geometry.torch_clearance(motion_states[..., :3]).amin()) if valid else 0.0
            diagnostics = {
                "prediction_dt_sec": self.model.dt_sec,
                "integration_dt_sec": self.model.integration_dt_sec,
                "ess": float(1 / self.controller.omega.square().sum()),
                "max_weight": float(self.controller.omega.max()),
                "temperature": float(self.controller.lambda_),
                "best_cost": float(self.controller.cost_total.min()),
                "control_bound_excess": float(torch.relu(controls.abs() - self.cost.action_limits).max()),
                "command_change": float(np.linalg.norm(command - self.last_output)),
                "candidate_motion_violation_fraction": float((self.controller.states[0, :, :, 16] > 0.5).any(dim=-1).float().mean()),
            }
            if valid:
                diagnostics.update({f"cost_{name}": float(value.sum()) for name, value in
                                    self.cost.components(prediction[1:], controls, motion_states=motion_states).items()})
                diagnostics["cost_terminal"] = float(self.cost.terminal(prediction[1:], controls))
            if self.method == "KMPPI":
                reconstructed, _ = self.controller.deparameterize_to_trajectory_single(self.controller.theta)
                diagnostics["interpolation_excess"] = float(self.controller.interpolation_excess)
                diagnostics["nominal_consistency_error"] = float((controls - self.controller._bound_action(reconstructed)).abs().max())
            self.last_output = command.copy()
        finished = time.perf_counter()
        return PlanResult(
            command, controls.cpu().numpy(), poses, candidates, candidate_controls, candidate_costs,
            valid, reason, clearance, finished - started, optimized - started, diagnostics,
        )
