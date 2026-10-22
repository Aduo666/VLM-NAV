from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import tempfile
import traceback
import time

import numpy as np
import torch
from tensordict import TensorDict

from .bc import load_checkpoint, observation, save_checkpoint, seed_everything
from .data import ACTION_SCALE, BEV_SIZE, Episode, discover, ensure_disjoint, fingerprint, load_manifest, read_npz, trajectory_fingerprint
from .policy import make_policy
from .runner import BevOnPolicyRunner, runner_config
from .geometry import DoorTracker


def compare_actors(first, second, arrays, device):
    first.eval()
    second.eval()
    first.reset()
    second.reset()
    maximum = 0.0
    with torch.inference_mode():
        for i in range(len(arrays["action"])):
            obs = observation(arrays, i, i + 1, device)
            a, b = first.act_inference(obs), second.act_inference(obs)
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(a.clamp(-1, 1) * torch.tensor(ACTION_SCALE, device=device),
                                       b.clamp(-1, 1) * torch.tensor(ACTION_SCALE, device=device), atol=2e-6, rtol=2e-5)
            maximum = max(maximum, float((a - b).abs().max()))
            if first.is_recurrent:
                for x, y in zip(first.memory_a.hidden_state, second.memory_a.hidden_state):
                    torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5)
            if i == len(arrays["action"]) // 2:
                first.reset()
                second.reset()
    first.reset()
    second.reset()
    return maximum


def check_sim(env, wrapped, manifest, cache, device):
    print("正在检查初始地图和车辆状态", flush=True)
    episode = env.episodes[0]
    entry = next(item for item in manifest["episodes"] if Path(item["path"]) == episode.path)
    arrays = read_npz(cache / entry["cache"])
    initial = wrapped.get_observations()
    restored_bev = initial["policy"][0, :BEV_SIZE].cpu().numpy().reshape(3, 80, 80).astype(np.uint8)
    differences = int(np.count_nonzero(restored_bev != arrays["bev"][0]))
    if differences:
        raise AssertionError(f"初始地图 BEV 有 {differences} 个元素不一致")
    if float(env.reset_errors.max()) > 1e-4:
        raise AssertionError(f"机器人恢复误差过大：{env.reset_errors.cpu().tolist()}")
    with torch.inference_mode():
        before = env.camera_world_transform(0, "front").copy()
        pose = env.robot.data.root_link_state_w[:, :7].clone()
        pose[:, 0] += 0.1
        env.robot.write_root_pose_to_sim(pose)
        env.sim.forward()
        after = env.camera_world_transform(0, "front")
        np.testing.assert_allclose(after[:3, 3] - before[:3, 3], [0.1, 0, 0], atol=1e-4)
    wrapped.reset()
    policy = make_policy(manifest, device=device)
    optimizer = torch.optim.AdamW([p for m in policy.actor_modules().values() for p in m.parameters()], lr=3e-4)
    optimizer.zero_grad(set_to_none=True)
    loss = 0
    for i in range(min(8, len(arrays["action"]))):
        mean = policy.act_inference(observation(arrays, i, i + 1, device))
        target = torch.tensor(arrays["action"][i:i + 1], device=device)
        loss = loss + torch.nn.functional.huber_loss(mean, target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    policy.reset()
    with tempfile.TemporaryDirectory(prefix="constraint-traverse-sim-check-") as temporary:
        checkpoint = Path(temporary) / "check.pt"
        save_checkpoint(checkpoint, policy, manifest, model="lstm", optimizer=optimizer)
        restored, saved = load_checkpoint(checkpoint, device)
        reload_error = compare_actors(policy, restored, arrays, device)
        runner = BevOnPolicyRunner(wrapped, runner_config(manifest), device=device)
        runner.alg.policy.load_actor_weights(saved["actor_state_dict"])
        transfer_error = compare_actors(policy, runner.alg.policy, arrays, device)
    print("检查点恢复与连续 Actor 推理一致性通过，正在检查 PPO 采样和更新", flush=True)
    algorithm = runner.alg
    actor = algorithm.policy
    norm = {key: value.clone() for key, value in actor.state_dict().items() if "state_mean" in key or "state_std" in key}
    before_actor = actor.actor_encoder.cnn[0].weight.detach().clone()
    before_critic = actor.critic_encoder.cnn[0].weight.detach().clone()
    env.task.timeout_sec = 8 * env.step_dt
    obs, _ = wrapped.reset()
    actor.reset()
    actor.train()
    clipped = torch.zeros(2, device=device)
    resets = 0
    sensor_check = None
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    sample_started = time.perf_counter()
    with torch.inference_mode():
        for index in range(16):
            actions = algorithm.act(obs)
            original = actions.clone()
            clipped += (actions.abs() > 1).sum(dim=0)
            obs, rewards, dones, extras = wrapped.step(actions)
            sampler = env.samplers[0]
            if sampler.fusion_count > 0:
                sensor_check = dict(fusion_count=sampler.fusion_count, capture_sec=dict(sampler.last_capture_sec),
                    available_sec=dict(sampler.last_available_sec), valid_ratio=sampler.valid_ratios.tolist(),
                    start_time_sec=sampler.start_time_sec)
            expected = rewards + algorithm.gamma * algorithm.transition.values[:, 0] * extras["time_outs"]
            if dones.any():
                terminal = extras["terminal"]
                if not bool(terminal["truncated"].all()) or not bool(extras["time_outs"].all()):
                    raise AssertionError("纯超时结束信息错误")
                if float(terminal["elapsed_sec"].min()) < env.task.timeout_sec - 1e-5:
                    raise AssertionError("超时计时提前结束")
                resets += int(dones.sum())
            algorithm.process_env_step(obs, rewards, dones, extras)
            torch.testing.assert_close(algorithm.storage.actions[index], original)
            torch.testing.assert_close(algorithm.storage.rewards[index, :, 0], expected)
        algorithm.compute_returns(obs)
    sample_elapsed = time.perf_counter() - sample_started
    losses = algorithm.update()
    print("PPO 采样和更新完成，正在检查到达与碰撞终止", flush=True)
    if not all(np.isfinite(value) for value in losses.values()):
        raise AssertionError("PPO 损失存在非有限值")
    if any(not bool(torch.isfinite(parameter).all()) or
           (parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()))
           for parameter in actor.parameters()):
        raise AssertionError("PPO 参数或梯度存在非有限值")
    if torch.equal(before_actor, actor.actor_encoder.cnn[0].weight) or torch.equal(before_critic, actor.critic_encoder.cnn[0].weight):
        raise AssertionError("PPO 更新未传递至 Actor 或 Critic")
    for key, value in norm.items():
        torch.testing.assert_close(value, actor.state_dict()[key], rtol=0, atol=0)
    if sensor_check is None or min(sensor_check["capture_sec"].values()) <= sensor_check["start_time_sec"]:
        raise AssertionError(f"实时地图没有收到新的相机融合数据：{sensor_check}")
    env.task.timeout_sec = 60.0
    wrapped.reset()
    pose, _ = env._pose_velocity()
    env.task.goal.copy_(pose[:, :2])
    success_record = None
    for _ in range(20):
        _, _, dones, extras = wrapped.step(torch.zeros(1, 2, device=device))
        if dones.any():
            success_record = extras["terminal"]
            break
    if success_record is None or not bool(success_record["success"].all()) or float(success_record["elapsed_sec"].min()) < 0.3:
        raise AssertionError("连续停稳到达验收失败")
    if min(env.camera_valid_ratio[0]) < 0.02:
        raise AssertionError("相机数据验收失败")
    from pxr import UsdGeom, UsdPhysics, Usd
    wrapped.reset()
    wall = next(prim for prim in env.sim.stage.Traverse()
                if "/Room/Meshes/wall/" in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.CollisionAPI))
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(wall).ComputeAlignedRange()
    center = (np.asarray(box.GetMin()) + np.asarray(box.GetMax())) / 2
    with torch.inference_mode():
        root = env.robot.data.root_link_state_w[:, :7].clone()
        root[:, :2] = torch.tensor(center[:2], dtype=torch.float32, device=device)
        env.robot.write_root_pose_to_sim(root)
        env.robot.write_root_link_velocity_to_sim(torch.zeros(1, 6, device=device))
    env.last_sensor_time = float(env.sim.current_time)
    collision_record = None
    for _ in range(5):
        _, _, dones, extras = wrapped.step(torch.zeros(1, 2, device=device))
        if dones.any():
            collision_record = extras["terminal"]
            break
    if collision_record is None or not bool(collision_record["collision"].all()):
        raise AssertionError(f"真实障碍物接触未触发碰撞终止，接触记录数：{env.contact_count.tolist()}")
    env.perturb = True
    wrapped.reset()
    perturbed_pose = env.start_pose[0].cpu().tolist()
    return dict(initial_bev_differences=differences, reset_errors=env.reset_errors.cpu().tolist(),
        checkpoint_max_error=reload_error, transfer_max_error=transfer_error, ppo_losses=losses,
        timeout_resets=resets, action_clipped_fraction=(clipped / 16).cpu().tolist(),
        camera_valid_ratio=sensor_check["valid_ratio"], map_updates=sensor_check["fusion_count"],
        stopping_success=True, obstacle_collision=True,
        sensor_time_check=sensor_check, rollout_steps_per_sec=16 / sample_elapsed,
        perturbed_start=perturbed_pose,
        torch_peak_memory_mb=(torch.cuda.max_memory_allocated(device) / 1024 ** 2
                              if str(device).startswith("cuda") else None))


def terminal_json(record):
    return {key: value.detach().cpu().tolist() for key, value in record.items()}


def evaluate_group(env, wrapped, policy, *, perturbations, annotations):
    results = []
    for perturbation in range(perturbations + 1):
        env.perturb = perturbation > 0
        env.reset_sequence[:] = 0
        obs, _ = wrapped.reset()
        policy.reset()
        completed = 0
        initial_pose = env.start_pose[0].cpu().numpy().copy()
        initial_map_age = float(obs["policy"][0, -1])

        def trackers(index, pose):
            path = env.episodes[index].path
            data = annotations.get(str(path), annotations.get(f"{path.parent.name}/{path.name}", []))
            return [DoorTracker.create(item, env.body_points, pose) for item in data]

        doors = trackers(0, initial_pose)
        with torch.inference_mode():
            while completed < len(env.episodes):
                action = policy.act_inference(obs)
                obs, _, dones, extras = wrapped.step(action)
                policy.reset(dones)
                pose = (extras["terminal"]["pose_world_xyyaw"][0] if dones.any() else env._pose_velocity()[0][0]).cpu().numpy()
                for door in doors:
                    door.update(pose)
                if dones.any():
                    result = terminal_json(extras["terminal"])
                    result.update(episode=str(env.episodes[completed].path), perturbation=perturbation,
                        initial_map="recorded_history", initial_map_age_sec=initial_map_age,
                        initial_pose_world_xyyaw=initial_pose.tolist(), doors_annotated=len(doors),
                        doors_passed=sum(d.passed and not bool(extras["terminal"]["collision"][0]) for d in doors))
                    results.append(result)
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                    completed += 1
                    if completed < len(env.episodes):
                        initial_pose = env.start_pose[0].cpu().numpy().copy()
                        initial_map_age = float(obs["policy"][0, -1])
                        doors = trackers(completed, initial_pose)
    return results


def run(args):
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=not args.gui, enable_cameras=True, device=args.device)
    app = launcher.app
    env = None
    try:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaacsim.core.utils.stage import create_new_stage
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("isaacsim.sensors.camera")
        from .environment import TraverseEnv
        seed_everything(args.seed)
        manifest = load_manifest(args.cache)
        episodes = [Episode(path) for path in discover(args.dataset)]
        if args.command == "check-sim":
            env = TraverseEnv([episodes[0]], device=args.device, seed=args.seed)
            wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
            result = check_sim(env, wrapped, manifest, args.cache, args.device)
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        else:
            policy, saved = load_checkpoint(args.checkpoint, args.device)
            other = dict(episodes=[dict(fingerprint=fingerprint(e.path), trajectory_fingerprint=trajectory_fingerprint(e.path)) for e in episodes])
            ensure_disjoint(saved["manifest"], other)
            groups = defaultdict(list)
            annotations = {} if args.doors is None else json.loads(args.doors.read_text(encoding="utf-8"))
            for episode in episodes:
                groups[episode.meta["usd_path"]].append(episode)
            results = []
            for group in groups.values():
                create_new_stage()
                env = TraverseEnv(group, device=args.device, seed=args.seed)
                wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
                results.extend(evaluate_group(env, wrapped, policy, perturbations=args.perturbations, annotations=annotations))
                env.close()
                env = None
            summary = dict(episodes=len(results), success_rate=float(np.mean([x["success"][0] for x in results])),
                collision_rate=float(np.mean([x["collision"][0] for x in results])),
                timeout_rate=float(np.mean([x["truncated"][0] for x in results])),
                stall_rate=float(np.mean([x["stalled"][0] for x in results])),
                mean_elapsed_sec=float(np.mean([x["elapsed_sec"][0] for x in results])),
                mean_path_length_m=float(np.mean([x["path_length_m"][0] for x in results])),
                doors_annotated=sum(x["doors_annotated"] for x in results),
                doors_passed=sum(x["doors_passed"] for x in results),
                endpoint_error_m=float(np.mean([x["endpoint_error_m"][0] for x in results])), results=results)
            summary["door_pass_rate"] = (summary["doors_passed"] / summary["doors_annotated"]
                                        if summary["doors_annotated"] else None)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({key: value for key, value in summary.items() if key != "results"}, ensure_ascii=False))
    except BaseException:
        traceback.print_exc()
        import omni.kit.app
        omni.kit.app.get_app().post_quit(1)
        raise
    finally:
        if env is not None:
            env.close()
        app.close(wait_for_replicator=False)
