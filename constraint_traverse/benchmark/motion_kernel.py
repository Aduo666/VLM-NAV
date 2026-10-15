from __future__ import annotations

import math
import torch


@torch.jit.script
def rate_limit(current: torch.Tensor, target: torch.Tensor, acceleration: torch.Tensor,
               deceleration: torch.Tensor, dt: float):
    increasing = (current * target >= 0) & (target.abs() > current.abs())
    rate = torch.where(increasing, acceleration, deceleration)
    return current + torch.maximum(torch.minimum(target - current, rate * dt), -rate * dt)


class DynamicsKernel(torch.nn.Module):
    def __init__(self, xy, radii, inverse, limits, hardware):
        super().__init__()
        self.register_buffer("xy", xy)
        self.register_buffer("radii", radii)
        self.register_buffer("velocity_inverse", inverse)
        self.max_linear_mps = float(limits.max_linear_mps)
        self.max_angular_radps = float(limits.max_angular_radps)
        self.linear_accel_mps2 = float(limits.linear_accel_mps2)
        self.linear_decel_mps2 = float(limits.linear_decel_mps2)
        self.angular_accel_radps2 = float(limits.angular_accel_radps2)
        self.angular_decel_radps2 = float(limits.angular_decel_radps2)
        self.steering_rate_radps = float(limits.steering_rate_radps)
        self.steering_response_sec = float(limits.steering_response_sec)
        self.max_steering_rad = float(hardware.max_steering_rad)
        self.max_lateral_acceleration_mps2 = float(hardware.max_lateral_acceleration_mps2)
        self.max_wheel_radps = float(hardware.max_wheel_radps)

    def forward(self, state: torch.Tensor, action: torch.Tensor, dt: float):
        v = rate_limit(state[..., 14], action[..., 0].clamp(-self.max_linear_mps, self.max_linear_mps),
                       torch.full_like(action[..., 0], self.linear_accel_mps2),
                       torch.full_like(action[..., 0], self.linear_decel_mps2), dt)
        w = rate_limit(state[..., 15], action[..., 1].clamp(-self.max_angular_radps, self.max_angular_radps),
                       torch.full_like(action[..., 1], self.angular_accel_radps2),
                       torch.full_like(action[..., 1], self.angular_decel_radps2), dt)
        vx, vy = v[..., None] - w[..., None] * self.xy[:, 1], w[..., None] * self.xy[:, 0]
        angles = torch.atan2(vy, vx)
        speeds = torch.hypot(vx, vy)
        reverse = angles.abs() > math.pi / 2
        angles = torch.where(reverse, angles - torch.sign(angles) * math.pi, angles)
        speeds = torch.where(reverse, -speeds, speeds)
        stopping = (v.abs() + w.abs()) < 1.0e-6
        angles = torch.where(stopping[..., None], state[..., 6:10], angles)
        invalid = (angles.abs() > self.max_steering_rad + 1.0e-6).any(dim=-1)
        invalid = invalid | ((v * w).abs() > self.max_lateral_acceleration_mps2 + 1.0e-6)
        scale = (self.max_wheel_radps / (speeds.abs() / self.radii).amax(dim=-1).clamp_min(1.0e-6)).clamp(max=1)
        speeds, v, w = speeds * scale[..., None], v * scale, w * scale
        angles = torch.where(invalid[..., None], state[..., 6:10], angles)
        speeds = torch.where(invalid[..., None], torch.zeros_like(speeds), speeds)
        steer_delta = (angles - state[..., 6:10]) * (1 - math.exp(-dt / self.steering_response_sec))
        steering = state[..., 6:10] + steer_delta.clamp(-self.steering_rate_radps * dt, self.steering_rate_radps * dt)
        wheel_speeds = rate_limit(state[..., 10:14], speeds,
                                 torch.full_like(speeds, self.linear_accel_mps2),
                                 torch.full_like(speeds, self.linear_decel_mps2), dt)
        vectors = torch.stack((wheel_speeds * torch.cos(steering), wheel_speeds * torch.sin(steering)), dim=-1)
        twist = vectors.flatten(start_dim=-2) @ self.velocity_inverse.T
        yaw_mid = state[..., 2] + 0.5 * twist[..., 2] * dt
        c, s = torch.cos(yaw_mid), torch.sin(yaw_mid)
        result = state.clone()
        result[..., 0] += (c * twist[..., 0] - s * twist[..., 1]) * dt
        result[..., 1] += (s * twist[..., 0] + c * twist[..., 1]) * dt
        result[..., 2] += twist[..., 2] * dt
        result[..., 3:6] = twist
        result[..., 6:10] = steering
        result[..., 10:14] = wheel_speeds
        result[..., 14] = v
        result[..., 15] = w
        result[..., 16] = invalid.float()
        return result
