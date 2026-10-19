from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .motion_kernel import DynamicsKernel, rate_limit, project_twist

STATE_SIZE = 17


@dataclass(frozen=True)
class MotionLimits:
    max_linear_mps: float
    max_angular_radps: float
    linear_accel_mps2: float
    linear_decel_mps2: float
    angular_accel_radps2: float
    angular_decel_radps2: float
    steering_rate_radps: float
    steering_response_sec: float

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in self.__dict__.values()):
            raise ValueError("运动模型参数必须为有限正数")


class FourWheelModel:
    def __init__(self, hardware, limits: MotionLimits, dt_sec: float, *, device="cpu", integration_dt_sec=1 / 60):
        if not math.isfinite(dt_sec) or dt_sec <= 0:
            raise ValueError("积分周期必须为有限正数")
        if limits.max_linear_mps > hardware.max_linear_mps or limits.max_angular_radps > hardware.max_angular_radps:
            raise ValueError("实验速度限制超过机器人配置上限")
        if limits.linear_accel_mps2 > hardware.max_acceleration_mps2:
            raise ValueError("实验加速度超过机器人配置上限")
        self.hardware, self.limits, self.dt_sec = hardware, limits, float(dt_sec)
        if not math.isfinite(integration_dt_sec) or integration_dt_sec <= 0:
            raise ValueError("物理积分周期必须为有限正数")
        self.integration_dt_sec = float(integration_dt_sec)
        self.last_substeps = None
        if device == "auto":
            device = f"cuda:{1 if torch.cuda.device_count() > 1 else 0}" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        wheels = hardware.wheels
        self.xy = torch.tensor([[w.x_m, w.y_m] for w in wheels], dtype=torch.float32, device=self.device)
        self.radii = torch.tensor([w.radius_m for w in wheels], dtype=torch.float32, device=self.device)
        self.steering_signs = np.asarray([w.steering_sign for w in wheels])
        self.drive_signs = np.asarray([w.drive_sign for w in wheels])
        matrix = torch.zeros((8, 3), dtype=torch.float32, device=self.device)
        matrix[0::2, 0], matrix[1::2, 1] = 1, 1
        matrix[0::2, 2], matrix[1::2, 2] = -self.xy[:, 1], self.xy[:, 0]
        self.velocity_inverse = torch.linalg.pinv(matrix)
        self.kernel = torch.jit.script(DynamicsKernel(self.xy, self.radii, self.velocity_inverse, limits, hardware))
        self.max_curvature = float(self.kernel.max_curvature)

    def project_actions(self, actions):
        return project_twist(actions, self.limits.max_linear_mps, self.limits.max_angular_radps, self.max_curvature)

    def step(self, state, action, dt_sec=None):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=state.dtype, device=self.device)
        dt = self.dt_sec if dt_sec is None else float(dt_sec)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("动作持续时间必须为有限正数")
        count = max(1, math.ceil(dt / self.integration_dt_sec - 1.0e-9))
        substeps = []
        for index in range(count):
            step_dt = min(self.integration_dt_sec, dt - index * self.integration_dt_sec)
            state = self._physical_step(state, action, step_dt)
            substeps.append(state)
        self.last_substeps = torch.stack(substeps)
        result = state.clone()
        result[..., 16] = self.last_substeps[..., 16].amax(dim=0)
        return result

    def _physical_step(self, state, action, dt):
        return self.kernel(state, action, float(dt))

    def rollout(self, initial, actions, dt_sec=None, *, return_substeps=False):
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        state = torch.as_tensor(initial, dtype=torch.float32, device=self.device)
        states = [state]
        substeps = []
        for action in actions:
            state = self.step(state, action, dt_sec)
            states.append(state)
            if return_substeps:
                substeps.append(self.last_substeps)
        result = torch.stack(states)
        return (result, torch.stack(substeps, dim=1)) if return_substeps else result

    def braking_rollout(self, initial, reaction_sec=0.10):
        state = torch.as_tensor(initial, dtype=torch.float32, device=self.device)
        states = [state]
        dt = min(self.dt_sec, 0.05)
        count = math.ceil(reaction_sec / dt)
        for _ in range(count):
            state = self.step(state, state[14:16], dt)
            states.append(state)
        for _ in range(math.ceil(10 / dt)):
            state = self.step(state, torch.zeros(2, device=self.device), dt)
            states.append(state)
            if float(state[3:6].abs().max()) < 0.005 and float(state[10:14].abs().max()) < 0.005:
                break
        return torch.stack(states)
