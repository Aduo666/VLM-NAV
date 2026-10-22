from __future__ import annotations

from copy import deepcopy

from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner

from .policy import BevActorCritic, BevActorCriticRecurrent
from .task_state import validate_batches


class BevOnPolicyRunner(OnPolicyRunner):
    def _construct_algorithm(self, obs):
        validate_batches(self.env.num_envs, self.alg_cfg["num_mini_batches"])
        policy_cfg = deepcopy(self.policy_cfg)
        cls = {"BevActorCriticRecurrent": BevActorCriticRecurrent, "BevActorCritic": BevActorCritic}[policy_cfg.pop("class_name")]
        policy = cls(obs, self.cfg["obs_groups"], self.env.num_actions, **policy_cfg).to(self.device)
        algorithm = deepcopy(self.alg_cfg)
        if algorithm.pop("class_name") != "PPO":
            raise ValueError("当前 Runner 使用官方 PPO")
        algorithm = PPO(policy, device=self.device, multi_gpu_cfg=self.multi_gpu_cfg, **algorithm)
        algorithm.init_storage("rl", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions])
        return algorithm


def runner_config(manifest, *, model="lstm", steps=16, mini_batches=1, noise=0.1):
    return dict(num_steps_per_env=steps, save_interval=100, logger="tensorboard",
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
        policy=dict(class_name="BevActorCriticRecurrent" if model == "lstm" else "BevActorCritic",
            state_mean=manifest["state_mean"], state_std=manifest["state_std"],
            actor_obs_normalization=False, critic_obs_normalization=False, init_noise_std=noise),
        algorithm=dict(class_name="PPO", num_learning_epochs=1, num_mini_batches=mini_batches,
            clip_param=0.2, gamma=0.99, lam=0.95, value_loss_coef=1.0, entropy_coef=0.001,
            learning_rate=1e-4, max_grad_norm=1.0, use_clipped_value_loss=True,
            schedule="fixed", desired_kl=None, rnd_cfg=None, symmetry_cfg=None))
