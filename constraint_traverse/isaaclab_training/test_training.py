from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import torch
from tensordict import TensorDict

from .bc import load_checkpoint, save_checkpoint, seed_everything, train
from .data import BEV_SIZE, OBS_SIZE, STATE_SIZE
from .policy import make_policy
from .runner import BevOnPolicyRunner, runner_config
from .task_state import TaskState, validate_batches
from .geometry import DoorTracker


MANIFEST = dict(state_mean=[0.0] * STATE_SIZE, state_std=[1.0] * STATE_SIZE, episodes=[])


def inputs(batch=2):
    raw = torch.zeros(batch, OBS_SIZE)
    raw[:, :BEV_SIZE] = torch.randint(0, 2, (batch, BEV_SIZE)).float() * 255
    return TensorDict({"policy": raw}, batch_size=[batch])


class TinyEnv:
    num_envs = 2
    num_actions = 2
    max_episode_length = 10
    cfg = None
    device = "cpu"

    def __init__(self):
        self.obs = inputs()
        self.step_index = 0

    def get_observations(self):
        return self.obs

    def step(self, actions):
        self.step_index += 1
        executed = actions.clamp(-1, 1)
        self.obs = self.obs.clone()
        self.obs["policy"][:, -3:-1] = executed
        dones = torch.tensor([self.step_index % 3 == 0, self.step_index % 5 == 0])
        return self.obs, -executed.square().sum(-1), dones, {"time_outs": dones.clone()}


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        seed_everything(7)

    def test_sequences_and_singleton(self):
        for batch, length in ((1, 1), (1, 4), (2, 4)):
            policy = make_policy(MANIFEST, num_envs=batch)
            sequence = torch.stack([inputs(batch)["policy"] for _ in range(length)])
            sequential = []
            for raw in sequence:
                sequential.append(policy.act_inference(TensorDict({"policy": raw}, batch_size=[batch])))
            policy.reset()
            hidden = (torch.zeros(1, batch, 128), torch.zeros(1, batch, 128))
            obs = TensorDict({"policy": sequence}, batch_size=[length, batch])
            policy.act(obs, masks=torch.ones(length, batch, dtype=torch.bool), hidden_state=hidden)
            torch.testing.assert_close(policy.action_mean, torch.stack(sequential), atol=2e-6, rtol=2e-5)

    def test_bev_dtype_and_actor_gradient(self):
        policy = make_policy(MANIFEST)
        obs = inputs()
        a = policy.get_actor_obs(obs)
        b = policy.get_actor_obs(TensorDict({"policy": obs["policy"].byte()}, batch_size=[2]))
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        prediction = policy.act_inference(obs)
        prediction.square().mean().backward()
        self.assertGreater(float(policy.actor_encoder.cnn[0].weight.grad.abs().sum()), 0)
        self.assertIsNone(policy.critic_encoder.cnn[0].weight.grad)
        self.assertIsNone(policy.log_std.grad)
        policy.memory_a.detach_hidden_state()
        self.assertIsNone(policy.memory_a.hidden_state[0].grad_fn)

    def test_partial_reset(self):
        policy = make_policy(MANIFEST, num_envs=2)
        obs = inputs()
        policy.act(obs)
        policy.evaluate(obs)
        before = tuple(tuple(value.detach().clone() for value in state) for state in policy.get_hidden_states())
        policy.reset(torch.tensor([True, False]))
        for old, current in zip(before, policy.get_hidden_states()):
            for old_tensor, current_tensor in zip(old, current):
                self.assertEqual(float(current_tensor[:, 0].abs().sum()), 0)
                torch.testing.assert_close(old_tensor[:, 1], current_tensor[:, 1])

    def test_checkpoint_and_actor_transfer(self):
        for model in ("lstm", "mlp"):
            policy = make_policy(MANIFEST, model=model)
            with tempfile.TemporaryDirectory(prefix="constraint-traverse-check-") as folder:
                path = Path(folder) / "model.pt"
                save_checkpoint(path, policy, MANIFEST, model=model)
                restored, _ = load_checkpoint(path)
                transferred = make_policy(MANIFEST, model=model)
                transferred.load_actor_weights(policy.actor_weights())
                for _ in range(6):
                    obs = inputs(1)
                    expected = policy.act_inference(obs)
                    torch.testing.assert_close(restored.act_inference(obs), expected, rtol=0, atol=0)
                    torch.testing.assert_close(transferred.act_inference(obs), expected, rtol=0, atol=0)

    def test_bc_resume_across_truncation(self):
        with tempfile.TemporaryDirectory(prefix="constraint-traverse-bc-test-") as folder:
            root = Path(folder)
            manifest = dict(MANIFEST, version=1, episodes=[])
            for i, count in enumerate((17, 3, 5)):
                filename = f"episode_{i}.npz"
                bev = np.zeros((count, 3, 80, 80), dtype=np.uint8)
                bev[:, 2] = 255
                np.savez(root / filename, bev=bev, state=np.random.randn(count, STATE_SIZE).astype(np.float32),
                    action=np.full((count, 2), 0.3, dtype=np.float32), weight=np.ones(count, dtype=np.float32))
                manifest["episodes"].append(dict(cache=filename))
            (root / "manifest.json").write_text(json.dumps(manifest))
            configuration = dict(batch_size=2, learning_rate=2e-4, seed=11)
            train(root, root / "full", epochs=2, **configuration)
            train(root, root / "resumed", epochs=1, **configuration)
            train(root, root / "resumed", epochs=2, resume=root / "resumed/last.pt")
            full, _ = load_checkpoint(root / "full/last.pt")
            resumed, saved = load_checkpoint(root / "resumed/last.pt")
            self.assertEqual(saved["training_config"], configuration)
            self.assertEqual(saved["seed"], configuration["seed"])
            self.assertEqual(saved["optimizer_state_dict"]["param_groups"][0]["lr"], configuration["learning_rate"])
            for key, value in full.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
            for name, value in dict(batch_size=8, learning_rate=3e-4, seed=0).items():
                with self.subTest(parameter=name), self.assertRaisesRegex(ValueError, name):
                    train(root, root / "resumed", epochs=3, resume=root / "resumed/last.pt", **{name: value})

    def test_terminal_priority_and_hold(self):
        for goal in (False, True):
            for collision in (False, True):
                for timeout in (False, True):
                    state = TaskState(1, "cpu")
                    state.reset(torch.tensor([0]), torch.zeros(1, 2), torch.ones(1, 2))
                    state.elapsed[:] = 59.99 if timeout else 1.0
                    state.hold_since[:] = state.elapsed - 1.0
                    xy = torch.zeros(1, 2) if goal else torch.ones(1, 2)
                    terminated, truncated = state.update(xy, torch.zeros(1, 2), torch.tensor([collision]), 0.02)
                    self.assertEqual(bool(terminated[0]), collision or goal)
                    self.assertEqual(bool(truncated[0]), timeout and not (collision or goal))
                    self.assertEqual(bool(state.success[0]), goal and not collision)
        state = TaskState(1, "cpu")
        state.reset(torch.tensor([0]), torch.zeros(1, 2), torch.zeros(1, 2))
        for _ in range(5):
            terminated, _ = state.update(torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor([False]), 0.06)
        self.assertFalse(bool(terminated[0]))
        state.update(torch.ones(1, 2), torch.zeros(1, 2), torch.tensor([False]), 0.01)
        self.assertEqual(float(state.hold_since[0]), -1)
        for _ in range(6):
            terminated, _ = state.update(torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor([False]), 0.06)
        self.assertTrue(bool(terminated[0]))

    def test_terminal_snapshot(self):
        state = TaskState(2, "cpu", timeout_sec=0.01)
        ids = torch.tensor([0, 1])
        state.reset(ids, torch.zeros(2, 2), torch.ones(2, 2))
        state.update(torch.ones(2, 2), torch.zeros(2, 2), torch.tensor([True, False]), 0.02)
        result = state.terminal(ids, torch.ones(2, 3))
        state.reset(torch.tensor([0]), torch.zeros(1, 2), torch.zeros(1, 2))
        self.assertTrue(bool(result["collision"][0]))
        self.assertTrue(bool(state.truncated[1]))
        self.assertEqual(float(result["pose_world_xyyaw"][0, 0]), 1.0)
        with torch.inference_mode():
            state.update(torch.ones(2, 2), torch.zeros(2, 2), torch.tensor([False, False]), 0.02)
        state.reset(ids, torch.zeros(2, 2), torch.zeros(2, 2))

    def test_batch_configuration(self):
        for envs, batches in ((1, 1), (4, 2), (8, 4)):
            validate_batches(envs, batches)
        for envs, batches in ((1, 4), (3, 2), (1, 0)):
            with self.assertRaises(ValueError):
                validate_batches(envs, batches)

    def test_door_geometry(self):
        annotation = dict(center_world_xy=[0, 0], normal_world_xy=[1, 0], width_m=0.8, thickness_m=0.1)
        points = np.array([[-0.4, -0.3], [-0.4, 0.3], [0.4, -0.3], [0.4, 0.3]])
        door = DoorTracker.create(annotation, points, [-1, 0, 0])
        door.update([0.1, 0, 0])
        self.assertFalse(door.passed)
        door.update([0.6, 0, 0])
        self.assertTrue(door.passed)
        bypass = DoorTracker.create(annotation, points, [-1, 1, 0])
        bypass.update([1, 1, 0])
        self.assertFalse(bypass.passed)

    def test_ppo_update_and_timeout(self):
        env = TinyEnv()
        runner = BevOnPolicyRunner(env, runner_config(MANIFEST, steps=10), device="cpu")
        policy, algorithm = runner.alg.policy, runner.alg
        with torch.no_grad():
            policy.actor.bias[0] = 2.0
        normalization = {key: value.clone() for key, value in policy.state_dict().items() if "state_mean" in key or "state_std" in key}
        before = policy.actor_encoder.cnn[0].weight.detach().clone()
        obs = env.get_observations()
        with torch.inference_mode():
            for index in range(10):
                actions = algorithm.act(obs)
                original = actions.clone()
                obs, rewards, dones, extras = env.step(actions)
                if index == 0:
                    self.assertTrue(bool((original[:, 0] > 1).all()))
                    torch.testing.assert_close(obs["policy"][:, -3], torch.ones(2))
                expected = rewards + algorithm.gamma * algorithm.transition.values[:, 0] * extras["time_outs"]
                algorithm.process_env_step(obs, rewards, dones, extras)
                torch.testing.assert_close(algorithm.storage.actions[index], original)
                torch.testing.assert_close(algorithm.storage.rewards[index, :, 0], expected)
            algorithm.compute_returns(obs)
        losses = algorithm.update()
        self.assertTrue(all(np.isfinite(value) for value in losses.values()))
        self.assertFalse(torch.equal(before, policy.actor_encoder.cnn[0].weight))
        for key, value in normalization.items():
            torch.testing.assert_close(value, policy.state_dict()[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
