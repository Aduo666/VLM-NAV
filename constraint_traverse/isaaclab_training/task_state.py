from __future__ import annotations

import torch


class TaskState:
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
        eligible = (self.distance <= 0.15) & (velocity[:, 0].abs() <= 0.05) & (velocity[:, 1].abs() <= 0.08)
        self.hold_since.copy_(torch.where(eligible, torch.where(self.hold_since < 0, self.elapsed, self.hold_since), -1.0))
        reached = eligible & (self.elapsed - self.hold_since >= 0.3 - 1e-7)
        self.collision.copy_(collision)
        self.success.copy_(reached & ~collision)
        terminated = collision | self.success
        self.truncated.copy_((self.elapsed >= self.timeout_sec - 1e-6) & ~terminated)
        return terminated, self.truncated.clone()

    def terminal(self, ids, pose):
        return dict(env_ids=ids.clone(), success=self.success[ids].clone(),
            collision=self.collision[ids].clone(), truncated=self.truncated[ids].clone(),
            elapsed_sec=self.elapsed[ids].clone(), endpoint_error_m=self.distance[ids].clone(),
            path_length_m=self.length[ids].clone(), pose_world_xyyaw=pose[ids].clone(),
            stalled=self.stalled[ids].clone(), action_variation=self.action_variation[ids].clone())


def validate_batches(num_envs, num_mini_batches):
    if not 1 <= num_mini_batches <= num_envs or num_envs % num_mini_batches:
        raise ValueError("循环 PPO 要求环境数量能够被批次数量整除，且批次数量在 1 到环境数量之间")
