from __future__ import annotations

from contextlib import redirect_stdout
import io
import numpy as np
import torch
from torch import nn
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent

from .data import BEV_SHAPE, BEV_SIZE, OBS_SIZE, STATE_SIZE


class ObservationEncoder(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("state_mean", torch.as_tensor(mean, dtype=torch.float32).reshape(STATE_SIZE))
        self.register_buffer("state_std", torch.as_tensor(std, dtype=torch.float32).reshape(STATE_SIZE))
        if not torch.isfinite(self.state_mean).all() or not torch.isfinite(self.state_std).all() or (self.state_std <= 0).any():
            raise ValueError("状态归一化参数无效")
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2), nn.GroupNorm(4, 16), nn.ELU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.GroupNorm(4, 32), nn.ELU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.GroupNorm(4, 32), nn.ELU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(512, 128), nn.ELU())
        self.fusion = nn.Sequential(nn.Linear(128 + STATE_SIZE, 128), nn.ELU())

    def forward(self, raw):
        if raw.shape[-1] != OBS_SIZE:
            raise ValueError(f"观测长度应为 {OBS_SIZE}")
        shape = raw.shape[:-1]
        flat = raw.reshape(-1, OBS_SIZE).float()
        bev = flat[:, :BEV_SIZE].reshape(-1, *BEV_SHAPE) / 255.0
        state = (flat[:, BEV_SIZE:] - self.state_mean) / self.state_std
        encoded = self.fusion(torch.cat((self.cnn(bev), state), dim=-1))
        return encoded.reshape(*shape, 128)


class EncodedObservations:
    def get_actor_obs(self, obs):
        return self.actor_encoder(torch.cat([obs[key] for key in self.obs_groups["policy"]], dim=-1))

    def get_critic_obs(self, obs):
        return self.critic_encoder(torch.cat([obs[key] for key in self.obs_groups["critic"]], dim=-1))

    def update_normalization(self, obs):
        # 固定的 BC 归一化参数由编码器缓冲区管理。
        return None

    def get_actions_log_prob(self, actions):
        values = super().get_actions_log_prob(actions)
        # RSL-RL 3.1.2 在循环更新中会去掉旧概率的单维度。
        return values.squeeze() if actions.ndim == 3 else values

    def actor_modules(self):
        result = dict(actor_encoder=self.actor_encoder, actor=self.actor)
        if self.is_recurrent:
            result["memory_a"] = self.memory_a
        return result

    def actor_weights(self):
        return {name: module.state_dict() for name, module in self.actor_modules().items()}

    def load_actor_weights(self, weights):
        if set(weights) != set(self.actor_modules()):
            raise ValueError("Actor 检查点的模块名称不匹配")
        for name, module in self.actor_modules().items():
            module.load_state_dict(weights[name], strict=True)
        self.critic_encoder.state_mean.copy_(self.actor_encoder.state_mean)
        self.critic_encoder.state_std.copy_(self.actor_encoder.state_std)


def parent_arguments(obs, obs_groups, num_actions, kwargs):
    if num_actions != 2 or any(len(obs_groups[name]) != 1 for name in ("policy", "critic")):
        raise ValueError("窄门策略要求两维动作以及单组策略、价值观测")
    if any(obs[key].shape[-1] != OBS_SIZE for name in ("policy", "critic") for key in obs_groups[name]):
        raise ValueError("策略观测维度错误")
    if kwargs.pop("actor_obs_normalization", False) or kwargs.pop("critic_obs_normalization", False):
        raise ValueError("固定观测处理要求关闭 RSL-RL 自动归一化")
    batch = obs.batch_size[0]
    encoded = TensorDict({key: torch.zeros(batch, 128, device=obs.device) for key in obs.keys()}, batch_size=[batch])
    arguments = dict(actor_hidden_dims=[128], critic_hidden_dims=[128], actor_obs_normalization=False,
        critic_obs_normalization=False, init_noise_std=0.1, noise_std_type="log", state_dependent_std=False)
    arguments.update(kwargs)
    if arguments["state_dependent_std"]:
        raise ValueError("当前动作头使用独立的高斯噪声参数")
    return encoded, arguments


class BevActorCriticRecurrent(EncodedObservations, ActorCriticRecurrent):
    def __init__(self, obs, obs_groups, num_actions, state_mean=None, state_std=None, **kwargs):
        encoded, arguments = parent_arguments(obs, obs_groups, num_actions, kwargs)
        with redirect_stdout(io.StringIO()):
            super().__init__(encoded, obs_groups, num_actions, rnn_hidden_dim=128, rnn_num_layers=1, **arguments)
        self.actor = nn.Linear(128, 2)
        self.critic = nn.Linear(128, 1)
        self.actor_encoder = ObservationEncoder(np.zeros(STATE_SIZE) if state_mean is None else state_mean,
                                                 np.ones(STATE_SIZE) if state_std is None else state_std)
        self.critic_encoder = ObservationEncoder(self.actor_encoder.state_mean.clone(), self.actor_encoder.state_std.clone())

    def act(self, obs, masks=None, hidden_state=None):
        output = self.memory_a(self.get_actor_obs(obs), masks, hidden_state)
        if masks is None:
            output = output.squeeze(0)
        self._update_distribution(output)
        return self.distribution.sample()

    def evaluate(self, obs, masks=None, hidden_state=None):
        output = self.memory_c(self.get_critic_obs(obs), masks, hidden_state)
        return self.critic(output.squeeze(0) if masks is None else output)


class BevActorCritic(EncodedObservations, ActorCritic):
    def __init__(self, obs, obs_groups, num_actions, state_mean=None, state_std=None, **kwargs):
        encoded, arguments = parent_arguments(obs, obs_groups, num_actions, kwargs)
        with redirect_stdout(io.StringIO()):
            super().__init__(encoded, obs_groups, num_actions, **arguments)
        self.actor = nn.Linear(128, 2)
        self.critic = nn.Linear(128, 1)
        self.actor_encoder = ObservationEncoder(np.zeros(STATE_SIZE) if state_mean is None else state_mean,
                                                 np.ones(STATE_SIZE) if state_std is None else state_std)
        self.critic_encoder = ObservationEncoder(self.actor_encoder.state_mean.clone(), self.actor_encoder.state_std.clone())


def make_policy(manifest, *, model="lstm", device="cpu", num_envs=1):
    obs = TensorDict({"policy": torch.zeros(num_envs, OBS_SIZE, device=device)}, batch_size=[num_envs])
    cls = {"lstm": BevActorCriticRecurrent, "mlp": BevActorCritic}[model]
    return cls(obs, {"policy": ["policy"], "critic": ["policy"]}, 2,
        state_mean=manifest["state_mean"], state_std=manifest["state_std"]).to(device)
