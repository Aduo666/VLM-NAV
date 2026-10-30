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
        self.reference_window = None
        self.reference_clearance_target = settings["obstacle_soft_distance_m"] + geometry.margin_m
        self.dt_sec = 0.20
        self.action_limits = torch.tensor([0.5, 0.5], device=self.device)

    def set_target(self, goal, reference=None):
        self.goal = torch.tensor(np.asarray(goal).copy(), dtype=torch.float32, device=self.device)
        if self.goal.shape != (2,) or not torch.isfinite(self.goal).all():
            raise ValueError("目标必须为有效的世界二维坐标")
        self.reference = None if reference is None else torch.as_tensor(reference, dtype=torch.float32, device=self.device)
        self.reference_window = None
        self.reference_clearance_target = self.settings["obstacle_soft_distance_m"] + self.geometry.margin_m

    def set_reference_window(self, window):
        self.reference_window = window
        self.route_xy = torch.tensor(window.world_xy, dtype=torch.float32, device=self.device)
        self.route_yaw = torch.tensor(window.world_yaw_rad, dtype=torch.float32, device=self.device)
        self.route_delta = torch.diff(self.route_xy, dim=0)
        self.route_lengths = torch.linalg.vector_norm(self.route_delta, dim=-1)
        self.route_s = torch.cat((torch.zeros(1, device=self.device), self.route_lengths.cumsum(0)))
        # 柔性间距使用参考路线能够提供的空间，障碍侵入代价独立计算。
        samples = []
        spacing = self.geometry.grid.resolution_m if getattr(self.geometry, "grid", None) is not None else 0.05
        for index in range(1, len(window.world_xy)):
            delta_yaw = float((window.world_yaw_rad[index] - window.world_yaw_rad[index - 1] + math.pi) % (2 * math.pi) - math.pi)
            count = max(1, math.ceil(np.linalg.norm(window.world_xy[index] - window.world_xy[index - 1]) / spacing),
                        math.ceil(abs(delta_yaw) / math.radians(5)))
            fractions = np.linspace(0, 1, count + 1)
            xy = window.world_xy[index - 1] + fractions[:, None] * (window.world_xy[index] - window.world_xy[index - 1])
            samples.append(np.column_stack((xy, window.world_yaw_rad[index - 1] + fractions * delta_yaw)))
        metrics = self.geometry.torch_metrics(torch.tensor(np.concatenate(samples), dtype=torch.float32, device=self.device))
        if all(float(metrics[name].max()) == 0 for name in ("occupied_fraction", "unknown_fraction", "outside_fraction")):
            self.reference_clearance_target = min(self.reference_clearance_target,
                max(self.geometry.margin_m, float(metrics["clearance"].min())))

    def route_errors(self, state):
        delta = state[..., None, :2] - self.route_xy[:-1]
        fraction = (delta * self.route_delta).sum(-1) / self.route_lengths.square().clamp_min(1.0e-9)
        fraction = fraction.clamp(0, 1)
        projected = self.route_xy[:-1] + fraction[..., None] * self.route_delta
        distances = (state[..., None, :2] - projected).square().sum(-1)
        index = distances.argmin(-1)
        selected_fraction = fraction.gather(-1, index[..., None]).squeeze(-1)
        progress = self.route_s[index] + selected_fraction * self.route_lengths[index]
        yaw_delta = torch.atan2(torch.sin(torch.diff(self.route_yaw)), torch.cos(torch.diff(self.route_yaw)))
        yaw = self.route_yaw[index] + selected_fraction * yaw_delta[index]
        if self.reference_window.direction == 0:
            yaw = self.route_yaw[-1].expand_as(progress)
        heading = torch.atan2(torch.sin(state[..., 2] - yaw), torch.cos(state[..., 2] - yaw))
        return self.route_s[-1] - progress, distances.gather(-1, index[..., None]).squeeze(-1), heading

    def components(self, state, action, *, motion_states=None, previous_state=None):
        cfg = self.settings
        delta = self.goal - state[..., :2]
        distance_squared = (delta / cfg["goal_distance_scale_m"]).square().sum(dim=-1)
        bearing = torch.atan2(delta[..., 1], delta[..., 0])
        heading = torch.atan2(torch.sin(bearing - state[..., 2]), torch.cos(bearing - state[..., 2]))
        checked = state if motion_states is None else motion_states
        metrics = self.geometry.torch_metrics(checked[..., :3])
        self.last_geometry_metrics = metrics
        reduce = (lambda value: value) if motion_states is None else (lambda value: value.mean(dim=0))
        gap = metrics["clearance"]
        soft_distance = cfg["obstacle_soft_distance_m"]
        components = {
            "goal": cfg["goal"] * distance_squared,
            "heading": cfg["heading"] * (heading / torch.pi).square() * (delta.square().sum(dim=-1) > 0.30 ** 2),
            "control": cfg["control"] * (action / self.action_limits).square().sum(dim=-1),
            "clearance": cfg["obstacle"] * reduce(torch.relu((self.reference_clearance_target - gap) / soft_distance).square()),
            "penetration": cfg["penetration"] * reduce((metrics["penetration"] / cfg["penetration_scale_m"]).square()),
            "occupied": cfg["occupied"] * reduce(metrics["occupied_fraction"]),
            "unknown": cfg["unknown"] * reduce(metrics["unknown_fraction"]),
            "outside": cfg["outside"] * reduce(metrics["outside_fraction"]),
            "motion": cfg["motion_constraint"] * reduce(checked[..., 16]),
        }
        if self.reference_window is not None:
            remaining, lateral_squared, route_heading = self.route_errors(state)
            components["goal"] = cfg["goal"] * (remaining / cfg["goal_distance_scale_m"]).square()
            components["reference"] = cfg["reference_path"] * lateral_squared / cfg["goal_distance_scale_m"] ** 2
            components["heading"] = cfg["heading"] * route_heading.square()
            curvature = torch.abs(torch.atan2(torch.sin(torch.diff(self.route_yaw)),
                torch.cos(torch.diff(self.route_yaw)))) / self.route_lengths.clamp_min(1e-6)
            if self.reference_window.direction == 0:
                desired = torch.zeros_like(remaining)
            else:
                limit = torch.minimum(torch.sqrt(cfg.get("lateral_accel_mps2", 0.20) / curvature.clamp_min(1e-3)),
                                      self.action_limits[1] / curvature.clamp_min(1e-3)).amin()
                cruise = min(float(self.action_limits[0]), cfg.get("cruise_speed_mps", 0.35), float(limit))
                desired = self.reference_window.direction * torch.minimum(torch.full_like(remaining, cruise),
                    torch.sqrt(2 * cfg.get("braking_decel_mps2", 0.30) * remaining.clamp_min(0)))
            components["speed"] = cfg.get("speed", 4.0) * ((state[..., 3] - desired) / self.action_limits[0]).square()
        elif self.reference is not None:
            lateral = torch.cdist(state[..., :2], self.reference).amin(dim=-1)
            components["reference"] = cfg["reference_path"] * (lateral / cfg["goal_distance_scale_m"]).square()
        if previous_state is not None:
            components["control_change"] = cfg.get("control_change", 3.0) * (
                (action - previous_state[..., 14:16]) / self.action_limits).square().sum(-1)
            components["steering_change"] = cfg.get("steering_change", 0.25) * (
                state[..., 17:21] - previous_state[..., 17:21]).square().sum(-1)
        return {name: value * self.dt_sec / cfg["reference_dt_sec"] for name, value in components.items()}

    def running(self, state, action, *, motion_states=None, previous_state=None):
        return sum(self.components(state, action, motion_states=motion_states, previous_state=previous_state).values())

    def terminal(self, states, actions):
        if self.reference_window is not None:
            remaining, lateral_squared, heading = self.route_errors(states[..., -1, :])
            yaw_cost = heading.square() if self.reference_window.direction == 0 else 0
            return self.settings["terminal_goal"] * (remaining.square() + lateral_squared + yaw_cost)
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
        with RNG_LOCK, torch.random.fork_rng(devices=self._cuda_indices):
            torch.manual_seed(self.seed)
            horizon = int(settings["horizon_steps"])
            initial = torch.zeros(horizon, 2, device=model.device)
            kwargs = dict(
                dynamics=model.step,
                running_cost=self._running,
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
                project_action=self._project_actions,
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
        self.controller.candidate_validity = self._candidate_validity
        self.controller.ess_fraction = float(settings.get("ess_fraction", 0.12))
        self.controller.noise_correlation = float(settings.get("noise_correlation", 0.85))

    def _project_actions(self, actions):
        window = self.cost.reference_window
        if window is None:
            return self.model.project_actions(actions)
        v, w = actions[..., 0], actions[..., 1]
        if window.direction == 0:
            return torch.stack((torch.zeros_like(v), w), dim=-1)
        v = v.clamp_min(0) if window.direction > 0 else v.clamp_max(0)
        bound = self.model.max_curvature * v.abs()
        return torch.stack((v, w.clamp(min=-bound, max=bound)), dim=-1)

    def _running(self, state, action):
        value = self.cost.running(state, action, motion_states=self.model.last_substeps,
                                  previous_state=self.model.last_initial_state)
        if self._cost_step == 0:
            metrics = self.cost.last_geometry_metrics
            self._prefix_valid = torch.stack([metrics[name].amax(dim=0) == 0
                for name in ("occupied_fraction", "unknown_fraction", "outside_fraction")]).all(dim=0)
        self._cost_step += 1
        return value

    def _candidate_validity(self):
        states = self.controller.states[0, :, 0]
        stop = self.model.braking_rollout(states, reaction_sec=self.safety["reaction_time_sec"])
        valid = self._prefix_valid.clone()
        for batch in stop.split(8):
            metrics = self.cost.geometry.torch_metrics(batch[..., :3])
            for name in ("occupied_fraction", "unknown_fraction", "outside_fraction"):
                valid &= metrics[name].amax(dim=0) == 0
        return valid

    def _output_prefix_valid(self, first_state, substeps):
        metrics = self.cost.geometry.torch_metrics(substeps[..., :3])
        if any(bool((metrics[name] > 0).any()) for name in ("occupied_fraction", "unknown_fraction", "outside_fraction")):
            return False
        stopping = self.model.braking_rollout(first_state, reaction_sec=self.safety["reaction_time_sec"])
        metrics = self.cost.geometry.torch_metrics(stopping[..., :3])
        return all(bool((metrics[name] == 0).all()) for name in ("occupied_fraction", "unknown_fraction", "outside_fraction"))

    def _reference_controls(self, state):
        window = self.cost.reference_window
        predicted = state.clone()
        output = np.zeros((self.controller.T, 2), dtype=np.float32)
        linear_limit = max(abs(float(self.controller.u_min[0])), abs(float(self.controller.u_max[0])))
        points = window.world_xy
        delta = np.diff(points, axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        cumulative = np.r_[0, np.cumsum(lengths)]
        yaw_delta = (np.diff(window.world_yaw_rad) + math.pi) % (2 * math.pi) - math.pi
        cursor = 0
        for index in range(len(output)):
            pose = predicted[:3].detach().cpu().numpy().astype(np.float64)
            if window.direction == 0:
                error = float((window.world_yaw_rad[-1] - pose[2] + math.pi) % (2 * math.pi) - math.pi)
                v, w = 0.0, 3.0 * error
            else:
                fraction = np.clip(((pose[:2] - points[:-1]) * delta).sum(-1)
                    / np.maximum(lengths ** 2, 1e-12), 0, 1)
                projected = points[:-1] + fraction[:, None] * delta
                distance = np.linalg.norm(pose[:2] - projected, axis=1)
                distance[:cursor] = np.inf
                cursor = int(distance.argmin())
                reference_yaw = window.world_yaw_rad[cursor] + fraction[cursor] * yaw_delta[cursor]
                remaining = cumulative[-1] - cumulative[cursor] - fraction[cursor] * lengths[cursor]
                speed = min(linear_limit, math.sqrt(max(0, 2 * self.model.limits.linear_decel_mps2 * remaining)))
                v = window.direction * speed
                heading_error = (reference_yaw - pose[2] + math.pi) % (2 * math.pi) - math.pi
                error_xy = pose[:2] - projected[cursor]
                lateral = -math.sin(reference_yaw) * error_xy[0] + math.cos(reference_yaw) * error_xy[1]
                curvature = yaw_delta[cursor] / max(lengths[cursor], 1e-6)
                w = speed * curvature + 3.0 * heading_error - window.direction * 4.0 * lateral
            control = self.controller._bound_action(torch.tensor([v, w], device=self.model.device))
            output[index] = control.cpu().numpy()
            predicted = self.model.step(predicted, control)
        return torch.tensor(output, device=self.model.device)

    def _prepare_sampling(self, tensor, elapsed_sec):
        controller = self.controller
        delta = self.cost.goal - tensor[:2]
        distance = float(torch.linalg.vector_norm(delta))
        speed_limit = min(self.model.limits.max_linear_mps, max(0.025, distance * 0.8))
        if self.cost.reference_window is not None and self.cost.reference_window.direction != 0:
            curvature = np.abs((np.diff(self.cost.reference_window.world_yaw_rad) + math.pi) % (2 * math.pi) - math.pi)
            curvature = curvature / np.maximum(np.linalg.norm(np.diff(self.cost.reference_window.world_xy, axis=0), axis=1), 1e-6)
            maximum = max(float(curvature.max()), 1e-3)
            speed_limit = min(speed_limit, self.cost.settings.get("cruise_speed_mps", 0.35),
                self.model.limits.max_angular_radps / maximum,
                math.sqrt(self.cost.settings.get("lateral_accel_mps2", 0.20) / maximum))
        controller.u_max = torch.tensor([speed_limit, self.model.limits.max_angular_radps], device=self.model.device)
        controller.u_min = -controller.u_max
        if self.cost.reference_window is not None:
            direction = self.cost.reference_window.direction
            if direction >= 0:
                controller.u_min[0] = 0
            if direction <= 0:
                controller.u_max[0] = 0
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
        if self.cost.reference_window is not None:
            reference_controls = controller._bound_action(self._reference_controls(tensor))
            controller.action_priors[:3] = torch.stack((reference_controls * 0.7, reference_controls, reference_controls * 0.9))
            controller.action_priors = controller._bound_action(controller.action_priors)
            blend = 1.0 if self.first_command else float(self.settings.get("reference_warm_start_blend", 0.5))
            if self.method == "KMPPI":
                indices = controller.Tk[0].round().long()
                controller.theta = torch.lerp(controller.theta, reference_controls[indices], blend)
                controller.synchronize_controls()
            else:
                controller.U = controller._bound_action(torch.lerp(controller.U, reference_controls, blend))

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
                self._cost_step = 0
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
            valid = bool(torch.isfinite(controls).all() and torch.isfinite(prediction).all()
                         and self.controller.valid_candidates.any())
            prefix_valid = valid and self._output_prefix_valid(prediction[1], motion_states[:, 0])
            valid = bool(valid and prefix_valid)
            reason = "OPTIMIZED" if valid else "NO_SAFE_CANDIDATE"
            command = controls[0].cpu().numpy() if valid else np.zeros(2, dtype=np.float32)
            clearance = float(self.cost.geometry.torch_clearance(motion_states[..., :3]).amin()) if valid else 0.0
            diagnostics = {
                "prediction_dt_sec": self.model.dt_sec,
                "integration_dt_sec": self.model.integration_dt_sec,
                "ess": float(1 / self.controller.omega.square().sum()) if valid else 0.0,
                "max_weight": float(self.controller.omega.max()),
                "temperature": float(self.controller.weight_temperature),
                "safe_candidates": int(self.controller.valid_candidates.sum()),
                "output_prefix_safe": bool(prefix_valid),
                "best_cost": float(self.controller.cost_total.min()),
                "control_bound_excess": float(torch.relu(controls.abs() - self.cost.action_limits).max()),
                "command_change": float(np.linalg.norm(command - self.last_output)),
                "candidate_motion_violation_fraction": float((self.controller.states[0, :, :, 16] > 0.5).any(dim=-1).float().mean()),
            }
            if self.cost.reference_window is not None:
                diagnostics["reference_length_m"] = self.cost.reference_window.length_m
                diagnostics["reference_direction"] = self.cost.reference_window.direction
                diagnostics["reference_clearance_target_m"] = self.cost.reference_clearance_target
            if valid:
                diagnostics.update({f"cost_{name}": float(value.sum()) for name, value in
                                    self.cost.components(prediction[1:], controls, motion_states=motion_states,
                                                         previous_state=prediction[:-1]).items()})
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
