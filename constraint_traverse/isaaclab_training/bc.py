from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import random

import numpy as np
import torch
from tensordict import TensorDict

from .data import BEV_SIZE, OBS_SIZE, load_manifest, read_npz, validate_manifest_sources
from .policy import make_policy


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def observation(arrays, start=0, end=None, device="cpu"):
    bev = arrays["bev"][start:end].reshape(-1, BEV_SIZE).astype(np.float32)
    raw = np.concatenate((bev, arrays["state"][start:end]), axis=-1)
    return TensorDict({"policy": torch.as_tensor(raw, device=device)}, batch_size=[len(raw)])


def save_checkpoint(path, policy, manifest, *, model, optimizer=None, epoch=0, seed=0,
                    training_config=None):
    payload = dict(format_version=1, model=model, model_state_dict=policy.state_dict(),
        actor_state_dict=policy.actor_weights(), manifest=manifest, epoch=epoch, seed=seed,
        training_config=None if training_config is None else dict(training_config),
        optimizer_state_dict=None if optimizer is None else optimizer.state_dict(),
        rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                 cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        versions={key: importlib.metadata.version(key) for key in ("torch", "rsl-rl-lib", "tensordict", "numpy")},
        action_scale=[0.8, 0.5], network_config=dict(bev_shape=[3, 80, 80], state_size=15,
            feature_dim=128, lstm_hidden_dim=128, lstm_layers=1, truncate_steps=16),
        framework=dict(isaaclab_tag="v2.3.2", isaaclab_commit="37ddf626871758333d6ed89cf64ad702aef127d0",
                       isaacsim="5.1.0"))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)


def load_checkpoint(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["format_version"] != 1 or saved["action_scale"] != [0.8, 0.5]:
        raise ValueError("检查点版本或动作定义不一致")
    policy = make_policy(saved["manifest"], model=saved["model"], device=device)
    policy.load_state_dict(saved["model_state_dict"], strict=True)
    policy.eval()
    return policy, saved


def masked_huber(prediction, target, weight, normalization):
    pointwise = torch.nn.functional.huber_loss(prediction, target, reduction="none").mean(dim=-1)
    return (pointwise * weight / normalization).sum() / len(normalization)


def train(cache: Path, output: Path, *, model="lstm", epochs=200, batch_size=None,
          learning_rate=None, seed=None, device="cpu", resume=None):
    manifest = load_manifest(cache)
    if "root" in manifest:
        validate_manifest_sources(manifest)
    supplied = dict(batch_size=batch_size, learning_rate=learning_rate, seed=seed)
    first_epoch = 0
    if resume:
        policy, saved = load_checkpoint(resume, device)
        if saved["manifest"] != manifest or saved["model"] != model:
            raise ValueError("恢复训练的数据或模型配置不同")
        training_config = saved.get("training_config")
        if training_config is None or set(training_config) != set(supplied):
            raise ValueError("检查点缺少完整训练配置，无法恢复训练")
        for name, value in supplied.items():
            if value is not None and value != training_config[name]:
                raise ValueError(f"恢复训练参数 {name} 与检查点不同：{value}，检查点为 {training_config[name]}")
        if saved["optimizer_state_dict"] is None:
            raise ValueError("检查点缺少优化器状态，无法恢复训练")
    else:
        defaults = dict(batch_size=8, learning_rate=3e-4, seed=0)
        training_config = {name: defaults[name] if value is None else value for name, value in supplied.items()}
    batch_size = training_config["batch_size"]
    learning_rate = training_config["learning_rate"]
    seed = training_config["seed"]
    if epochs < 1 or batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("训练轮数、批次大小和学习率必须为正数")
    if not resume:
        seed_everything(seed)
        policy = make_policy(manifest, model=model, device=device)
    parameters = [parameter for module in policy.actor_modules().values() for parameter in module.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    if resume:
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        first_epoch = saved["epoch"]
        random.setstate(saved["rng"]["python"])
        np.random.set_state(saved["rng"]["numpy"])
        torch.set_rng_state(saved["rng"]["torch"])
        if saved["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
    if first_epoch >= epochs:
        raise ValueError("目标训练轮数必须大于检查点中已完成的轮数")
    trajectories = [read_npz(cache / item["cache"]) for item in manifest["episodes"]]
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(first_epoch, epochs):
        policy.train()
        order = np.random.permutation(len(trajectories))
        epoch_loss = 0.0
        for offset in range(0, len(order), batch_size):
            batch = [trajectories[i] for i in order[offset:offset + batch_size]]
            policy.reset()
            total = torch.tensor([item["weight"].sum() for item in batch], device=device)
            for start in range(0, max(len(item["action"]) for item in batch), 16):
                optimizer.zero_grad(set_to_none=True)
                predictions, targets, weights = [], [], []
                for index in range(start, min(start + 16, max(len(item["action"]) for item in batch))):
                    raw = torch.zeros(len(batch), OBS_SIZE, device=device)
                    target = torch.zeros(len(batch), 2, device=device)
                    weight = torch.zeros(len(batch), device=device)
                    for j, item in enumerate(batch):
                        if index < len(item["action"]):
                            raw[j, :BEV_SIZE] = torch.as_tensor(item["bev"][index].reshape(-1), device=device)
                            raw[j, BEV_SIZE:] = torch.as_tensor(item["state"][index], device=device)
                            target[j] = torch.as_tensor(item["action"][index], device=device)
                            weight[j] = float(item["weight"][index])
                    predictions.append(policy.act_inference(TensorDict({"policy": raw}, batch_size=[len(batch)])))
                    targets.append(target)
                    weights.append(weight)
                loss = masked_huber(torch.stack(predictions), torch.stack(targets), torch.stack(weights), total)
                if not torch.isfinite(loss):
                    raise FloatingPointError("BC 损失出现非有限值")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                if policy.is_recurrent:
                    policy.memory_a.detach_hidden_state()
                epoch_loss += float(loss.detach()) * len(batch)
        policy.reset()
        save_checkpoint(output / "last.pt", policy, manifest, model=model, optimizer=optimizer,
                        epoch=epoch + 1, seed=seed, training_config=training_config)
        if (epoch + 1) % 10 == 0:
            save_checkpoint(output / f"epoch_{epoch + 1:04d}.pt", policy, manifest,
                            model=model, optimizer=optimizer, epoch=epoch + 1, seed=seed,
                            training_config=training_config)
        print(json.dumps(dict(epoch=epoch + 1, weighted_huber=epoch_loss / len(trajectories)), ensure_ascii=False), flush=True)
