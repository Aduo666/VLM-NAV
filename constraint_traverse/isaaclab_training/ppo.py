from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml

from .bc import capture_rng_state, load_checkpoint, restore_rng_state, save_checkpoint, seed_everything
from .data import Episode, discover, fingerprint
from .runner import BevOnPolicyRunner, runner_config
from .task_state import RewardConfig, validate_batches


@dataclass(frozen=True)
class PpoConfig:
    num_envs: int = 1
    rollout_steps: int = 32
    scene_iterations: int = 64
    num_learning_epochs: int = 4
    num_mini_batches: int = 1
    learning_rate: float = 3e-5
    initial_noise_std: float = 0.1
    clip_param: float = 0.1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.001
    value_loss_coef: float = 1.0
    max_grad_norm: float = 1.0
    save_interval: int = 50
    timeout_sec: float = 60.0
    collision_force_threshold_n: float = 1.0
    sensor_render_retries: int = 3
    max_consecutive_sensor_failures: int = 3
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self):
        for name in ("num_envs", "rollout_steps", "scene_iterations", "num_learning_epochs", "num_mini_batches", "save_interval", "max_consecutive_sensor_failures"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} 必须是正整数")
        if type(self.sensor_render_retries) is not int or self.sensor_render_retries < 0:
            raise ValueError("sensor_render_retries 必须是非负整数")
        validate_batches(self.num_envs, self.num_mini_batches)
        for name in ("learning_rate", "initial_noise_std", "clip_param", "gamma", "gae_lambda", "max_grad_norm",
                     "timeout_sec", "collision_force_threshold_n"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须是大于零的有限数值")
        if self.rollout_steps * self.num_envs < 2:
            raise ValueError("每次 PPO 更新至少需要两个采样")
        if not 0 < self.gamma < 1 or not 0 < self.gae_lambda <= 1 or not 0 < self.clip_param < 1:
            raise ValueError("gamma、gae_lambda 或 clip_param 超出允许范围")
        if any(not math.isfinite(value) or value < 0 for value in (self.entropy_coef, self.value_loss_coef)):
            raise ValueError("熵奖励和价值损失系数必须是非负有限数值")

    @classmethod
    def from_mapping(cls, values):
        values = dict(values)
        values["reward"] = RewardConfig(**values.get("reward", {}))
        return cls(**values)


def load_config(path):
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict) or set(values) - {"ppo", "reward"}:
        raise ValueError("PPO 配置需要包含 ppo 和 reward 配置组")
    return PpoConfig.from_mapping(dict(values.get("ppo", {}), reward=values.get("reward", {})))


def training_runner_config(manifest, model, config):
    values = runner_config(manifest, model=model, steps=config.rollout_steps,
        mini_batches=config.num_mini_batches, noise=config.initial_noise_std)
    values["algorithm"].update(num_learning_epochs=config.num_learning_epochs,
        learning_rate=config.learning_rate, clip_param=config.clip_param, gamma=config.gamma,
        lam=config.gae_lambda, entropy_coef=config.entropy_coef,
        value_loss_coef=config.value_loss_coef, max_grad_norm=config.max_grad_norm)
    return values


def compute_returns(algorithm, observations):
    if not algorithm.policy.is_recurrent:
        algorithm.compute_returns(observations)
        return
    memory = algorithm.policy.memory_c
    hidden = memory.hidden_state
    try:
        algorithm.compute_returns(observations)
    finally:
        # 末尾观测仅用于估计回报，下一次真正执行时再推进 Critic 状态。
        memory.reset(hidden_state=hidden)


class PpoTrainer:
    def __init__(self, *, source, source_path, resume, config, groups, dataset, signature,
                 environment_factory, output, seed, device):
        self.source = source
        self.resume = bool(resume)
        self.config = config
        self.groups = groups
        self.dataset = str(Path(dataset).resolve())
        self.signature = signature
        self.environment_factory = environment_factory
        self.output = Path(output)
        self.seed, self.device = int(seed), str(device)
        self.model, self.manifest = source["model"], source["manifest"]
        if not groups or any(not group for group in groups):
            raise ValueError("PPO 训练需要至少一个包含 Episode 的场景")
        if resume:
            if source.get("training_stage") != "ppo" or source.get("training_state") is None:
                raise ValueError("恢复 PPO 训练需要使用 PPO 检查点")
            if source.get("optimizer_state_dict") is None:
                raise ValueError("PPO 检查点缺少优化器状态")
            if PpoConfig.from_mapping(source["training_config"]) != config or source["seed"] != self.seed:
                raise ValueError("恢复 PPO 时的配置和随机种子必须与检查点一致")
            self.state = deepcopy(source["training_state"])
            if self.state["dataset"] != self.dataset or self.state["dataset_signature"] != signature:
                raise ValueError("PPO 恢复训练的数据路径或内容已经变化")
        else:
            if source.get("training_stage", "bc") != "bc":
                raise ValueError("新的 PPO 微调需要使用 BC 检查点")
            self.state = dict(completed_iterations=0, total_steps=0, scene_index=0,
                scene_iterations=0, scene_cursors=[list(range(config.num_envs)) for _ in groups],
                dataset=self.dataset, dataset_signature=signature,
                source_bc_checkpoint=str(Path(source_path).resolve()), episodes_total=0,
                recent_episodes=[], learning_rate=config.learning_rate)
        if (not 0 <= self.state["scene_index"] < len(groups)
                or len(self.state["scene_cursors"]) != len(groups)
                or any(len(cursors) != config.num_envs or any(type(value) is not int or value < 0 for value in cursors)
                       for cursors in self.state["scene_cursors"])):
            raise ValueError("PPO 检查点的场景采样进度无效")
        if not resume and self.output.exists() and any(self.output.iterdir()):
            raise ValueError(f"新训练需要使用空的输出目录：{self.output}")
        latest = self.output / "last.pt"
        if resume and latest.exists():
            if Path(source_path).resolve() != latest.resolve():
                raise ValueError("在已有输出目录中恢复训练需要使用该目录的 last.pt；其他检查点需要指定空的输出目录")
        elif resume and self.output.exists() and any(self.output.iterdir()):
            raise ValueError("恢复训练需要使用对应检查点的输出目录，或者新的空目录")
        self.recent = deque(self.state["recent_episodes"], maxlen=100)
        self.env = None
        self.runner = None
        self.obs = None
        self.writer = None
        seed_everything(self.seed)

    def _open_scene(self):
        index = self.state["scene_index"]
        rng = capture_rng_state()
        self.env = self.environment_factory(self.groups[index], self.state["scene_cursors"][index])
        restore_rng_state(rng)
        self.obs = self.env.get_observations().to(self.device)
        if self.runner is None:
            self.runner = BevOnPolicyRunner(self.env,
                training_runner_config(self.manifest, self.model, self.config), device=self.device)
            if self.resume:
                self.runner.alg.policy.load_state_dict(self.source["model_state_dict"], strict=True)
                self.runner.alg.optimizer.load_state_dict(self.source["optimizer_state_dict"])
                self.runner.alg.learning_rate = self.state["learning_rate"]
                restore_rng_state(self.source["rng"])
            else:
                self.runner.alg.policy.load_actor_weights(self.source["actor_state_dict"])
        else:
            self.runner.env = self.env
        self.runner.current_learning_iteration = self.state["completed_iterations"]
        self.runner.writer = self.writer
        self.runner.logger_type = "tensorboard"
        self.runner.tot_timesteps = self.state["total_steps"]
        self.episode_rewards = torch.zeros(self.config.num_envs, device=self.device)
        self.episode_lengths = torch.zeros(self.config.num_envs, dtype=torch.long, device=self.device)
        self.runner.train_mode()
        with torch.inference_mode():
            actor_critic = self.runner.alg.policy
            if actor_critic.is_recurrent:
                for memory in (actor_critic.memory_a, actor_critic.memory_c):
                    shape = (memory.rnn.num_layers, self.config.num_envs, memory.rnn.hidden_size)
                    zeros = memory.rnn.weight_ih_l0.new_zeros(shape)
                    memory.reset(hidden_state=(zeros, torch.zeros_like(zeros)))
            else:
                actor_critic.reset()
        print(f"PPO 场景：{self.groups[index][0].meta['usd_path']}，包含 {len(self.groups[index])} 条录制起点", flush=True)

    def _close_scene(self):
        if self.env is not None:
            self.env.close()
            self.env = None
            self.obs = None
            if self.runner is not None:
                self.runner.env = None

    def _save(self, *, periodic=False):
        self.state["recent_episodes"] = list(self.recent)
        self.state["learning_rate"] = self.runner.alg.learning_rate
        arguments = dict(model=self.model, optimizer=self.runner.alg.optimizer, seed=self.seed,
            training_config=asdict(self.config), training_stage="ppo", training_state=deepcopy(self.state))
        save_checkpoint(self.output / "last.pt", self.runner.alg.policy, self.manifest, **arguments)
        if periodic:
            name = f"iteration_{self.state['completed_iterations']:06d}.pt"
            save_checkpoint(self.output / name, self.runner.alg.policy, self.manifest, **arguments)

    def _record_episodes(self, extras):
        terminal = extras.get("terminal")
        if terminal is None:
            return []
        records = []
        arrays = {key: value.detach().cpu().tolist() for key, value in terminal.items()}
        for row, env_id in enumerate(arrays["env_ids"]):
            record = {key: value[row] for key, value in arrays.items() if key != "env_ids"}
            record.update(env_id=env_id, iteration=self.state["completed_iterations"] + 1,
                episode=extras["episode_paths"][row], collision_contact=extras["collision_contacts"][row],
                sensor_diagnostic=extras.get("sensor_diagnostics", [None] * len(arrays["env_ids"]))[row],
                episode_return=float(self.episode_rewards[env_id]), episode_steps=int(self.episode_lengths[env_id]))
            records.append(record)
            self.recent.append({key: record.get(key, False) for key in
                ("success", "collision", "truncated", "endpoint_error_m", "sensor_failure", "episode_return", "episode_steps")})
            self.episode_rewards[env_id] = 0
            self.episode_lengths[env_id] = 0
            self.state["episodes_total"] += 1
        return records

    def _log_metrics(self, record, *, iterations, start_iteration):
        self.runner.writer = self.writer
        valid = [item for item in self.recent if not item.get("sensor_failure", False)]
        values = {"Train/" + name: record[name] for name in
            ("reward_mean", "training_success_rate", "training_collision_rate", "training_timeout_rate",
             "training_sensor_failure_rate", "training_endpoint_error_m") if record[name] is not None}
        for axis, name in enumerate(("linear", "angular")):
            values["Policy/" + name + "_clipped_fraction"] = record["action_clipped_fraction"][axis]
            self.writer.add_scalar("Policy/" + name + "_noise_std", record["action_noise_std"][axis], record["iteration"])
        self.runner.log(dict(it=record["iteration"], tot_iter=iterations,
            start_iter=start_iteration, num_learning_iterations=iterations - start_iteration,
            collection_time=record["collection_sec"],
            learn_time=record["iteration_sec"] - record["collection_sec"], loss_dict=record["losses"],
            ep_infos=[values], rewbuffer=[item["episode_return"] for item in valid if "episode_return" in item],
            lenbuffer=[item["episode_steps"] for item in valid if "episode_steps" in item]))
        for name, value in record["reward_terms"].items():
            self.writer.add_scalar("Reward/" + name, value, record["iteration"])

    def run(self, iterations):
        if type(iterations) is not int or iterations <= self.state["completed_iterations"]:
            raise ValueError("目标迭代次数必须大于已经完成的次数")
        self.output.mkdir(parents=True, exist_ok=True)
        description = dict(ppo={key: value for key, value in asdict(self.config).items() if key != "reward"},
            reward=asdict(self.config.reward))
        (self.output / "config.yaml").write_text(yaml.safe_dump(description, allow_unicode=True, sort_keys=False), encoding="utf-8")
        self.writer = SummaryWriter(str(self.output / "tensorboard"),
            purge_step=self.state["completed_iterations"] + 1 if self.resume else None)
        start_iteration = self.state["completed_iterations"] + 1
        try:
            with (self.output / "metrics.jsonl").open("a", encoding="utf-8", buffering=1) as metrics, \
                 (self.output / "episodes.jsonl").open("a", encoding="utf-8", buffering=1) as episodes:
                while self.state["completed_iterations"] < iterations:
                    if self.env is None:
                        self._open_scene()
                        if self.state["completed_iterations"] == 0:
                            self._save()
                    started = time.perf_counter()
                    reward_sum = torch.zeros((), device=self.device)
                    clipped = torch.zeros(2, device=self.device)
                    terms = {}
                    completed_episodes = []
                    algorithm = self.runner.alg
                    with torch.inference_mode():
                        for _ in range(self.config.rollout_steps):
                            actions = algorithm.act(self.obs)
                            clipped += (actions.abs() > 1).sum(dim=0)
                            self.obs, rewards, dones, extras = self.env.step(actions)
                            self.obs = self.obs.to(self.device)
                            if not bool(torch.isfinite(rewards).all()):
                                raise RuntimeError("PPO 奖励包含非有限数值")
                            self.episode_rewards += rewards
                            self.episode_lengths += 1
                            algorithm.process_env_step(self.obs, rewards, dones, extras)
                            reward_sum += rewards.mean()
                            for name, value in extras.get("reward_terms", {}).items():
                                terms[name] = terms.get(name, torch.zeros((), device=self.device)) + value.mean()
                            completed_episodes.extend(self._record_episodes(extras))
                        compute_returns(algorithm, self.obs)
                    collection_sec = time.perf_counter() - started
                    losses = algorithm.update()
                    if (not all(math.isfinite(value) for value in losses.values())
                            or any(not bool(torch.isfinite(parameter).all()) for parameter in algorithm.policy.parameters())):
                        raise RuntimeError("PPO 更新产生了非有限的损失或参数")
                    index = self.state["scene_index"]
                    scene = self.groups[index][0].meta["usd_path"]
                    self.state["completed_iterations"] += 1
                    self.state["total_steps"] += self.config.num_envs * self.config.rollout_steps
                    self.state["scene_iterations"] += 1
                    self.state["scene_cursors"][index] = (self.env.unwrapped.reset_sequence - 1).tolist()
                    # 已完成批次的回报使用最后观测进行价值估计，切换房间后清空循环状态。
                    switch_scene = len(self.groups) > 1 and self.state["scene_iterations"] >= self.config.scene_iterations
                    if switch_scene:
                        self.state["scene_index"] = (index + 1) % len(self.groups)
                        self.state["scene_iterations"] = 0
                    iteration = self.state["completed_iterations"]
                    record = dict(iteration=iteration, total_steps=self.state["total_steps"], scene=scene,
                        reward_mean=float(reward_sum / self.config.rollout_steps),
                        losses={key: float(value) for key, value in losses.items()},
                        action_clipped_fraction=(clipped / (self.config.num_envs * self.config.rollout_steps)).cpu().tolist(),
                        action_noise_std=algorithm.policy.log_std.detach().exp().cpu().tolist(),
                        episodes_total=self.state["episodes_total"], recent_episodes=len(self.recent),
                        collection_sec=collection_sec, iteration_sec=time.perf_counter() - started,
                        reward_terms={key: float(value / self.config.rollout_steps) for key, value in terms.items()})
                    valid = [item for item in self.recent if not item.get("sensor_failure", False)]
                    record["valid_recent_episodes"] = len(valid)
                    record["training_sensor_failure_rate"] = float(np.mean([item.get("sensor_failure", False) for item in self.recent])) if self.recent else None
                    for key, name in (("success", "training_success_rate"), ("collision", "training_collision_rate"),
                                      ("truncated", "training_timeout_rate"), ("endpoint_error_m", "training_endpoint_error_m")):
                        record[name] = float(np.mean([item[key] for item in valid])) if valid else None
                    self.runner.current_learning_iteration = iteration
                    self._save(periodic=iteration % self.config.save_interval == 0 or iteration == iterations)
                    for episode_record in completed_episodes:
                        episodes.write(json.dumps(episode_record, ensure_ascii=False, allow_nan=False) + "\n")
                    line = json.dumps(record, ensure_ascii=False, allow_nan=False)
                    metrics.write(line + "\n")
                    self._log_metrics(record, iterations=iterations, start_iteration=start_iteration)
                    self.writer.flush()
                    if switch_scene:
                        self._close_scene()
        finally:
            self._close_scene()
            self.writer.close()
        return deepcopy(self.state)


def run_cli(args):
    source_path = args.resume if args.resume is not None else args.checkpoint
    policy, source = load_checkpoint(source_path)
    del policy
    if args.resume is not None and source.get("training_stage") != "ppo":
        raise ValueError("--resume 需要填写 PPO 检查点")
    if args.resume is not None:
        config = PpoConfig.from_mapping(source["training_config"]) if args.config is None else load_config(args.config)
        seed = source["seed"] if args.seed is None else args.seed
        dataset = Path(source["training_state"]["dataset"]) if args.dataset is None else args.dataset
    else:
        config_path = Path(__file__).resolve().parents[1] / "configs/ppo.yaml" if args.config is None else args.config
        config = load_config(config_path)
        seed = 0 if args.seed is None else args.seed
        if args.dataset is None:
            raise ValueError("新的 PPO 训练需要通过 --dataset 指定录制数据集")
        dataset = args.dataset
    groups = defaultdict(list)
    signature = []
    for path in discover(dataset):
        episode = Episode(path)
        groups[episode.meta["usd_path"]].append(episode)
        signature.append(dict(path=str(episode.path), fingerprint=fingerprint(episode.path)))
    trainer = PpoTrainer(source=source, source_path=source_path, resume=args.resume is not None,
        config=config, groups=list(groups.values()), dataset=dataset, signature=signature,
        environment_factory=None, output=args.output, seed=seed, device=args.device)
    if args.iterations <= trainer.state["completed_iterations"]:
        raise ValueError("目标迭代次数必须大于检查点中已经完成的次数")
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=not args.gui, enable_cameras=True, device=args.device,
        experience="isaacsim.exp.base.python.kit", multi_gpu=False)
    app = launcher.app
    try:
        from isaacsim.core.utils.extensions import enable_extension
        from isaacsim.core.utils.stage import create_new_stage
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        enable_extension("isaacsim.sensors.camera")
        from .environment import TraverseEnv

        def environment_factory(group, cursors):
            create_new_stage()
            env = TraverseEnv(group, num_envs=config.num_envs, device=args.device, seed=seed,
                timeout_sec=config.timeout_sec, collision_force_threshold_n=config.collision_force_threshold_n,
                reward_config=config.reward, sensor_render_retries=config.sensor_render_retries,
                max_consecutive_sensor_failures=config.max_consecutive_sensor_failures,
                sensor_diagnostic_dir=args.output / "sensor_failures")
            try:
                env.reset_sequence[:] = cursors
                return RslRlVecEnvWrapper(env, clip_actions=1.0)
            except BaseException:
                env.close()
                raise

        trainer.environment_factory = environment_factory
        trainer.run(args.iterations)
        print(f"PPO 训练完成，检查点：{args.output / 'last.pt'}", flush=True)
    except KeyboardInterrupt:
        latest = args.output / "last.pt"
        if latest.is_file():
            print(f"PPO 训练已中断，检查点：{latest}", flush=True)
        else:
            print("PPO 初始化已中断", flush=True)
        raise
    except BaseException:
        traceback.print_exc()
        import omni.kit.app
        omni.kit.app.get_app().post_quit(1)
        raise
    finally:
        app.close(wait_for_replicator=False)
