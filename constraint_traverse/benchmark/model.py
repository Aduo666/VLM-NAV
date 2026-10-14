from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


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


def rate_limit(current, target, acceleration, deceleration, dt):
    increasing = (current * target >= 0) & (target.abs() > current.abs())
    rate = torch.where(increasing, acceleration, deceleration)
    return current + torch.maximum(torch.minimum(target - current, rate * dt), -rate * dt)


class FourWheelModel:
    def __init__(self, hardware, limits: MotionLimits, dt_sec: float, *, device="cpu"):
        if not math.isfinite(dt_sec) or dt_sec <= 0:
            raise ValueError("积分周期必须为有限正数")
        if limits.max_linear_mps > hardware.max_linear_mps or limits.max_angular_radps > hardware.max_angular_radps:
            raise ValueError("实验速度限制超过机器人配置上限")
        if limits.linear_accel_mps2 > hardware.max_acceleration_mps2:
            raise ValueError("实验加速度超过机器人配置上限")
        self.hardware, self.limits, self.dt_sec = hardware, limits, float(dt_sec)
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

    def step(self, state, action, dt_sec=None):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=state.dtype, device=self.device)
        dt = self.dt_sec if dt_sec is None else float(dt_sec)
        lim = self.limits
        v = rate_limit(state[..., 14], action[..., 0].clamp(-lim.max_linear_mps, lim.max_linear_mps),
                       torch.full_like(action[..., 0], lim.linear_accel_mps2),
                       torch.full_like(action[..., 0], lim.linear_decel_mps2), dt)
        w = rate_limit(state[..., 15], action[..., 1].clamp(-lim.max_angular_radps, lim.max_angular_radps),
                       torch.full_like(action[..., 1], lim.angular_accel_radps2),
                       torch.full_like(action[..., 1], lim.angular_decel_radps2), dt)
        vx, vy = v[..., None] - w[..., None] * self.xy[:, 1], w[..., None] * self.xy[:, 0]
        angles = torch.atan2(vy, vx)
        speeds = torch.hypot(vx, vy)
        reverse = angles.abs() > math.pi / 2
        angles = torch.where(reverse, angles - torch.sign(angles) * math.pi, angles)
        speeds = torch.where(reverse, -speeds, speeds)
        stopping = (v.abs() + w.abs()) < 1.0e-6
        angles = torch.where(stopping[..., None], state[..., 6:10], angles)
        invalid = (angles.abs() > self.hardware.max_steering_rad + 1.0e-6).any(dim=-1)
        invalid = invalid | ((v * w).abs() > self.hardware.max_lateral_acceleration_mps2 + 1.0e-6)
        scale = (self.hardware.max_wheel_radps / (speeds.abs() / self.radii).amax(dim=-1).clamp_min(1.0e-6)).clamp(max=1)
        speeds, v, w = speeds * scale[..., None], v * scale, w * scale
        angles = torch.where(invalid[..., None], state[..., 6:10], angles)
        speeds = torch.where(invalid[..., None], torch.zeros_like(speeds), speeds)
        steer_delta = (angles - state[..., 6:10]) * (1 - math.exp(-dt / lim.steering_response_sec))
        steering = state[..., 6:10] + steer_delta.clamp(-lim.steering_rate_radps * dt, lim.steering_rate_radps * dt)
        wheel_speeds = rate_limit(state[..., 10:14], speeds,
                                 torch.full_like(speeds, lim.linear_accel_mps2),
                                 torch.full_like(speeds, lim.linear_decel_mps2), dt)
        vectors = torch.stack((wheel_speeds * torch.cos(steering), wheel_speeds * torch.sin(steering)), dim=-1)
        twist = vectors.flatten(start_dim=-2) @ self.velocity_inverse.T
        yaw_mid = state[..., 2] + 0.5 * twist[..., 2] * dt
        c, s = torch.cos(yaw_mid), torch.sin(yaw_mid)
        result = state.clone()
        result[..., 0] += (c * twist[..., 0] - s * twist[..., 1]) * dt
        result[..., 1] += (s * twist[..., 0] + c * twist[..., 1]) * dt
        result[..., 2] += twist[..., 2] * dt
        result[..., 3:6], result[..., 6:10], result[..., 10:14] = twist, steering, wheel_speeds
        result[..., 14], result[..., 15], result[..., 16] = v, w, invalid.float()
        return result

    def rollout(self, initial, actions, dt_sec=None):
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        state = torch.as_tensor(initial, dtype=torch.float32, device=self.device)
        states = [state]
        for action in actions:
            state = self.step(state, action, dt_sec)
            states.append(state)
        return torch.stack(states)

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
