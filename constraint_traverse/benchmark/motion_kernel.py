from __future__ import annotations

import math
import torch


@torch.jit.script
def rate_limit(current: torch.Tensor, target: torch.Tensor, acceleration: torch.Tensor,
               deceleration: torch.Tensor, dt: float):
    increasing = (current * target >= 0) & (target.abs() > current.abs())
    rate = torch.where(increasing, acceleration, deceleration)
    return current + torch.maximum(torch.minimum(target - current, rate * dt), -rate * dt)


@torch.jit.script
def project_twist(action: torch.Tensor, max_linear: float, max_angular: float, max_curvature: float):
    v = action[..., 0].clamp(-max_linear, max_linear)
    w = action[..., 1].clamp(-max_angular, max_angular)
    turning_limit = max_curvature * v.abs()
    arc_w = torch.maximum(torch.minimum(w, turning_limit), -turning_limit)
    spin = ((w - arc_w).abs() / max_angular > v.abs() / max_linear) | (v.abs() < 1.0e-6)
    v = torch.where(spin, torch.zeros_like(v), v)
    w = torch.where(spin, w, arc_w)
    return torch.stack((v, w), dim=-1)


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
        self.spin_yaw_scale = float(limits.spin_yaw_scale)
        self.wheel_response_sec = float(limits.wheel_response_sec)
        self.max_steering_rad = float(hardware.max_steering_rad)
        self.max_lateral_acceleration_mps2 = float(hardware.max_lateral_acceleration_mps2)
        self.max_wheel_radps = float(hardware.max_wheel_radps)
        self.steering_alignment_rad = float(hardware.steering_alignment_rad)
        self.max_curvature = float(1 / max(float(xy[:, 1].abs().max())
                                           + float(xy[:, 0].abs().max()) / math.tan(hardware.max_steering_rad), 1.0e-6))

    def forward(self, state: torch.Tensor, action: torch.Tensor, dt: float):
        target = project_twist(action, self.max_linear_mps, self.max_angular_radps, self.max_curvature)
        previous = state[..., 14:16]
        previous_spin = (previous[..., 0].abs() < 1.0e-6) & (previous[..., 1].abs() >= 1.0e-6)
        target_spin = (target[..., 0].abs() < 1.0e-6) & (target[..., 1].abs() >= 1.0e-6)
        changing_mode = (previous_spin != target_spin) | (previous[..., 0] * target[..., 0] < 0)
        changing_mode = changing_mode & (previous.abs().sum(dim=-1) >= 1.0e-6)
        target = torch.where(changing_mode[..., None], torch.zeros_like(target), target)
        delta = target - previous
        accelerating = (previous * target >= 0) & (target.abs() > previous.abs())
        linear_rate = torch.where(accelerating[..., 0], self.linear_accel_mps2, self.linear_decel_mps2)
        angular_rate = torch.where(accelerating[..., 1], self.angular_accel_radps2, self.angular_decel_radps2)
        rates = torch.stack((linear_rate, angular_rate), dim=-1)
        fraction = (rates * dt / delta.abs().clamp_min(1.0e-9)).amin(dim=-1).clamp(max=1.0)
        commanded = previous + fraction[..., None] * delta
        v, w = commanded[..., 0], commanded[..., 1]
        vx, vy = v[..., None] - w[..., None] * self.xy[:, 1], w[..., None] * self.xy[:, 0]
        angles = torch.atan2(vy, vx)
        speeds = torch.hypot(vx, vy)
        reverse = angles.abs() > math.pi / 2
        angles = torch.where(reverse, angles - torch.sign(angles) * math.pi, angles)
        speeds = torch.where(reverse, -speeds, speeds)
        stopping = (v.abs() + w.abs()) < 1.0e-6
        angles = torch.where(stopping[..., None], state[..., 17:21], angles)
        invalid = (angles.abs() > self.max_steering_rad + 1.0e-6).any(dim=-1)
        invalid = invalid | ((v * w).abs() > self.max_lateral_acceleration_mps2 + 1.0e-6)
        scale = (self.max_wheel_radps / (speeds.abs() / self.radii).amax(dim=-1).clamp_min(1.0e-6)).clamp(max=1)
        speeds, v, w = speeds * scale[..., None], v * scale, w * scale
        angles = torch.where(invalid[..., None], state[..., 17:21], angles)
        speeds = torch.where(invalid[..., None], torch.zeros_like(speeds), speeds)
        steering_target = state[..., 17:21] + (angles - state[..., 17:21]).clamp(
            -self.steering_rate_radps * dt, self.steering_rate_radps * dt)
        steering = steering_target + (state[..., 6:10] - steering_target) * math.exp(-dt / self.steering_response_sec)
        aligning = ((angles - state[..., 6:10]).abs().amax(dim=-1) > self.steering_alignment_rad)
        aligning = aligning & (state[..., 10:14].abs().amax(dim=-1) < 0.025)
        speeds = torch.where(aligning[..., None], torch.zeros_like(speeds), speeds)
        wheel_target = rate_limit(state[..., 21:25], speeds,
                                 torch.full_like(speeds, self.linear_accel_mps2),
                                 torch.full_like(speeds, self.linear_decel_mps2), dt)
        wheel_speeds = wheel_target + (state[..., 10:14] - wheel_target) * math.exp(-dt / self.wheel_response_sec)
        vectors = torch.stack((wheel_speeds * torch.cos(steering), wheel_speeds * torch.sin(steering)), dim=-1)
        twist = vectors.flatten(start_dim=-2) @ self.velocity_inverse.T
        spinning = (v.abs() < 1e-6) & (w.abs() > 1e-6)
        twist[..., 2] = twist[..., 2] * torch.where(spinning,
            torch.full_like(v, self.spin_yaw_scale), torch.ones_like(v))
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
        result[..., 17:21] = steering_target
        result[..., 21:25] = wheel_target
        return result
