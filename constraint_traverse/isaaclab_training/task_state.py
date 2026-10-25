from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import torch


@dataclass(frozen=True)
class RewardConfig:
    progress_weight: float = 5.0
    success_reward: float = 10.0
    collision_penalty: float = 10.0
    timeout_penalty: float = 2.0
    time_penalty_per_sec: float = 0.05
    action_change_penalty: float = 0.01
    linear_speed_penalty_per_sec: float = 5.0
    angular_speed_penalty_per_sec: float = 5.0
    hold_progress_reward: float = 2.0

    def __post_init__(self):
        if any(not math.isfinite(value) or value < 0 for value in asdict(self).values()):
            raise ValueError("奖励参数必须是非负的有限数值")


class TaskState:
    goal_radius_m = 0.15
    goal_linear_speed_mps = 0.05
    goal_angular_speed_radps = 0.08
    goal_hold_sec = 0.3

    def __init__(self, num_envs, device, *, timeout_sec=60.0):
        self.timeout_sec = float(timeout_sec)
        self.goal = torch.zeros(num_envs, 2, device=device)
        self.elapsed = torch.zeros(num_envs, device=device)
        self.hold_since = torch.full_like(self.elapsed, -1.0)
        self.distance = torch.zeros_like(self.elapsed)
        self.previous_distance = torch.zeros_like(self.elapsed)
        self.length = torch.zeros_like(self.elapsed)
        self.previous_xy = torch.zeros_like(self.goal)
        self.previous_action = torch.zeros_like(self.goal)
        self.velocity = torch.zeros_like(self.goal)
        self.hold_fraction = torch.zeros_like(self.elapsed)
        self.previous_hold_fraction = torch.zeros_like(self.elapsed)
        self.success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.collision = torch.zeros_like(self.success)
        self.truncated = torch.zeros_like(self.success)
        self.motion_xy = torch.zeros_like(self.goal)
        self.motion_yaw = torch.zeros_like(self.elapsed)
        self.last_motion = torch.zeros_like(self.elapsed)
        self.stalled = torch.zeros_like(self.success)
        self.action_variation = torch.zeros_like(self.elapsed)

    def reset(self, ids, goal, xy, yaw=None):
        self.goal[ids] = goal
        self.elapsed[ids] = 0
        self.hold_since[ids] = -1
        self.distance[ids] = torch.linalg.vector_norm(goal - xy, dim=-1)
        self.previous_distance[ids] = self.distance[ids]
        self.length[ids] = 0
        self.previous_xy[ids] = xy
        self.previous_action[ids] = 0
        self.velocity[ids] = 0
        self.hold_fraction[ids] = 0
        self.previous_hold_fraction[ids] = 0
        self.success[ids] = False
        self.collision[ids] = False
        self.truncated[ids] = False
        self.motion_xy[ids] = xy
        self.motion_yaw[ids] = 0 if yaw is None else yaw
        self.last_motion[ids] = 0
        self.stalled[ids] = False
        self.action_variation[ids] = 0

    def update(self, xy, velocity, collision, dt, yaw=None):
        self.elapsed += float(dt)
        self.velocity.copy_(velocity)
        self.previous_hold_fraction.copy_(self.hold_fraction)
        self.previous_distance.copy_(self.distance)
        self.distance.copy_(torch.linalg.vector_norm(self.goal - xy, dim=-1))
        self.length += torch.linalg.vector_norm(xy - self.previous_xy, dim=-1)
        self.previous_xy.copy_(xy)
        yaw = torch.zeros_like(self.motion_yaw) if yaw is None else yaw
        yaw_change = torch.atan2(torch.sin(yaw - self.motion_yaw), torch.cos(yaw - self.motion_yaw)).abs()
        moving = (torch.linalg.vector_norm(xy - self.motion_xy, dim=-1) >= 0.02) | (yaw_change >= 0.02)
        self.last_motion[moving] = self.elapsed[moving]
        self.motion_xy[moving] = xy[moving]
        self.motion_yaw[moving] = yaw[moving]
        self.stalled |= self.elapsed - self.last_motion >= 12.0
        eligible = ((self.distance <= self.goal_radius_m)
                    & (velocity[:, 0].abs() <= self.goal_linear_speed_mps)
                    & (velocity[:, 1].abs() <= self.goal_angular_speed_radps) & ~collision)
        self.hold_since.copy_(torch.where(eligible, torch.where(self.hold_since < 0, self.elapsed, self.hold_since), -1.0))
        hold_sec = torch.where(eligible, self.elapsed - self.hold_since, 0.0)
        self.hold_fraction.copy_((hold_sec / self.goal_hold_sec).clamp(0.0, 1.0))
        reached = eligible & (hold_sec >= self.goal_hold_sec - 1e-7)
        self.collision.copy_(collision)
        self.success.copy_(reached & ~collision)
        terminated = collision | self.success
        self.truncated.copy_((self.elapsed >= self.timeout_sec - 1e-6) & ~terminated)
        return terminated, self.truncated.clone()

    def rewards(self, command, *, dt, action_scale, braking_acceleration, config: RewardConfig):
        scale = torch.as_tensor(action_scale, device=self.goal.device, dtype=self.goal.dtype)
        action = command / scale
        change = torch.square(action - self.previous_action).sum(dim=-1)
        remaining = (self.distance - self.goal_radius_m).clamp_min(0.0)
        target_speed = torch.sqrt(2 * float(braking_acceleration) * remaining).clamp_max(scale[0])
        excess_speed = ((self.velocity[:, 0].abs() - target_speed).clamp_min(0.0) / scale[0]).square()
        turning_at_goal = (self.distance <= self.goal_radius_m) * (self.velocity[:, 1] / scale[1]).square()
        terms = dict(
            progress=config.progress_weight * (self.previous_distance - self.distance),
            success=config.success_reward * self.success.float(),
            collision=-config.collision_penalty * self.collision.float(),
            timeout=-config.timeout_penalty * self.truncated.float(),
            time=torch.full_like(self.distance, -config.time_penalty_per_sec * float(dt)),
            action_change=-config.action_change_penalty * change,
            braking=-config.linear_speed_penalty_per_sec * float(dt) * excess_speed,
            goal_turning=-config.angular_speed_penalty_per_sec * float(dt) * turning_at_goal,
            hold=config.hold_progress_reward * (self.hold_fraction - self.previous_hold_fraction))
        self.previous_action.copy_(action)
        return sum(terms.values()), terms

    def terminal(self, ids, pose):
        return dict(env_ids=ids.clone(), success=self.success[ids].clone(),
            collision=self.collision[ids].clone(), truncated=self.truncated[ids].clone(),
            elapsed_sec=self.elapsed[ids].clone(), endpoint_error_m=self.distance[ids].clone(),
            path_length_m=self.length[ids].clone(), pose_world_xyyaw=pose[ids].clone(),
            stalled=self.stalled[ids].clone(), action_variation=self.action_variation[ids].clone())


def validate_batches(num_envs, num_mini_batches):
    if not 1 <= num_mini_batches <= num_envs or num_envs % num_mini_batches:
        raise ValueError("循环 PPO 要求环境数量能够被批次数量整除，且批次数量在 1 到环境数量之间")
