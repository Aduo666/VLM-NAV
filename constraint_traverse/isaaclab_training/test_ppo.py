from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from tensordict import TensorDict

from .bc import load_checkpoint, save_checkpoint, seed_everything
from .data import ACTION_SCALE, BEV_SIZE, OBS_SIZE
from .policy import make_policy
from .ppo import PpoConfig, PpoTrainer, compute_returns, load_config, training_runner_config
from .runner import BevOnPolicyRunner
from .task_state import RewardConfig, TaskState
from .test_training import MANIFEST


class RecordedStartEnv:
    def __init__(self, group, cursors, config):
        self.config = config
        self.num_envs, self.num_actions, self.device = config.num_envs, 2, "cpu"
        self.unwrapped = self
        self.cfg = None
        self.max_episode_length = 2
        self.reset_sequence = np.array(cursors, dtype=np.int64) + 1
        self.group = group
        self.task = TaskState(self.num_envs, "cpu", timeout_sec=0.1)
        self.closed = False
        self._reset(torch.arange(self.num_envs))

    def _reset(self, ids):
        self.task.reset(ids, torch.zeros(len(ids), 2), torch.ones(len(ids), 2))

    def get_observations(self):
        raw = torch.zeros(self.num_envs, OBS_SIZE)
        raw[:, 2 * BEV_SIZE // 3:BEV_SIZE] = 255
        raw[:, BEV_SIZE:BEV_SIZE + 2] = self.task.previous_xy
        raw[:, -3:-1] = self.task.previous_action * torch.tensor(ACTION_SCALE)
        return TensorDict({"policy": raw}, batch_size=[self.num_envs])

    def step(self, actions):
        command = actions.clamp(-1, 1) * torch.tensor(ACTION_SCALE)
        xy = self.task.previous_xy - 0.01
        ended, timed_out = self.task.update(xy, command, torch.zeros(self.num_envs, dtype=torch.bool), 0.05)
        rewards, terms = self.task.rewards(command, dt=0.05, action_scale=ACTION_SCALE,
            braking_acceleration=0.6, config=self.config.reward)
        done = ended | timed_out
        extras = dict(time_outs=timed_out, reward_terms=terms)
        ids = done.nonzero().flatten()
        if len(ids):
            extras["terminal"] = self.task.terminal(ids, torch.cat((xy, torch.zeros(self.num_envs, 1)), dim=-1))
            extras["episode_paths"] = [str(self.group[int(self.reset_sequence[i] - 1) % len(self.group)].path) for i in ids]
            extras["collision_contacts"] = [None] * len(ids)
            self.reset_sequence[ids.cpu().numpy()] += 1
            self._reset(ids)
        return self.get_observations(), rewards, done.long(), extras

    def close(self):
        self.closed = True


class PpoTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        seed_everything(17)

    def reward(self, *, distance, velocity, dt=0.1):
        task = TaskState(1, "cpu")
        xy = torch.tensor([[distance, 0.0]])
        task.reset(torch.tensor([0]), torch.zeros(1, 2), xy)
        task.update(xy, torch.tensor([velocity]), torch.tensor([False]), dt)
        return task.rewards(torch.zeros(1, 2), dt=dt, action_scale=ACTION_SCALE,
            braking_acceleration=0.6, config=RewardConfig())

    def test_stopping_reward_and_distance_gate(self):
        stopped, _ = self.reward(distance=0.1, velocity=[0.0, 0.0])
        moving, terms = self.reward(distance=0.1, velocity=[0.6, 0.0])
        turning, _ = self.reward(distance=0.1, velocity=[0.0, 0.4])
        self.assertGreater(float(stopped), float(moving))
        self.assertGreater(float(stopped), float(turning))
        self.assertLess(float(terms["braking"]), 0)
        _, far = self.reward(distance=2.0, velocity=[0.8, 0.4])
        self.assertEqual(float(far["braking"]), 0)
        self.assertEqual(float(far["goal_turning"]), 0)
        self.assertEqual(float(far["hold"]), 0)
        _, short = self.reward(distance=0.1, velocity=[0.6, 0], dt=0.02)
        self.assertAlmostEqual(float(short["braking"] / 0.02), float(terms["braking"] / 0.1), places=5)

    def test_hold_reward_cannot_accumulate_by_restarting(self):
        task = TaskState(1, "cpu")
        xy = torch.zeros(1, 2)
        task.reset(torch.tensor([0]), xy, xy)
        total = 0.0
        for speed in (0.0, 0.0, 0.0, 0.1):
            task.update(xy, torch.tensor([[speed, 0.0]]), torch.tensor([False]), 0.08)
            _, terms = task.rewards(torch.zeros(1, 2), dt=0.08, action_scale=ACTION_SCALE,
                braking_acceleration=0.6, config=RewardConfig())
            total += float(terms["hold"])
        self.assertAlmostEqual(total, 0.0, places=6)
        self.assertEqual(float(task.hold_fraction[0]), 0)

    def test_configuration_validation(self):
        path = Path(__file__).resolve().parents[1] / "configs/ppo.yaml"
        self.assertEqual(load_config(path), PpoConfig())
        for settings in (dict(num_envs=1, num_mini_batches=2), dict(rollout_steps=1),
                         dict(initial_noise_std=0), dict(gamma=1), dict(num_learning_epochs=0)):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                PpoConfig(**settings)
        with self.assertRaises(ValueError):
            RewardConfig(hold_progress_reward=-1)

    def trainer(self, root, *, output, resume=None, groups=1, config=None):
        config = config or PpoConfig(rollout_steps=4, num_learning_epochs=1, save_interval=1, scene_iterations=1)
        source_path = root / "bc.pt" if resume is None else resume
        if not (root / "bc.pt").exists():
            actor = make_policy(MANIFEST)
            with torch.no_grad():
                actor.actor.bias[0] = 1.5
            save_checkpoint(root / "bc.pt", actor, MANIFEST, model="lstm")
        _, source = load_checkpoint(source_path)
        scenes = [[SimpleNamespace(path=root / f"room_{i}" / f"episode_{j}",
            meta={"usd_path": f"room_{i}.usd"}) for j in range(3)] for i in range(groups)]
        opened = []

        def factory(group, cursors):
            result = RecordedStartEnv(group, cursors, config)
            opened.append((group[0].meta["usd_path"], list(cursors), result))
            return result

        trainer = PpoTrainer(source=source, source_path=source_path, resume=resume is not None,
            config=config, groups=scenes, dataset=root / "dataset", signature=["unchanged"],
            environment_factory=factory, output=output, seed=0, device="cpu")
        return trainer, opened

    def test_actor_initialization_and_ppo_checkpoint_inference(self):
        with tempfile.TemporaryDirectory(prefix="constraint-traverse-ppo-test-") as folder:
            root = Path(folder)
            trainer, _ = self.trainer(root, output=root / "run")
            with redirect_stdout(io.StringIO()):
                trainer._open_scene()
            source, _ = load_checkpoint(root / "bc.pt")
            source.reset()
            with torch.inference_mode():
                for _ in range(4):
                    torch.testing.assert_close(source.act_inference(trainer.obs),
                        trainer.runner.alg.policy.act_inference(trainer.obs), rtol=0, atol=0)
            trainer._close_scene()
            with redirect_stdout(io.StringIO()):
                trainer.run(1)
            restored, saved = load_checkpoint(root / "run/last.pt")
            self.assertEqual(saved["training_stage"], "ppo")
            self.assertEqual(saved["training_state"]["completed_iterations"], 1)
            with torch.inference_mode():
                restored.reset()
                trainer.runner.alg.policy.reset()
                obs = RecordedStartEnv([SimpleNamespace(path=root / "episode")], [0], trainer.config).get_observations()
                torch.testing.assert_close(restored.act_inference(obs), trainer.runner.alg.policy.act_inference(obs), rtol=0, atol=0)

    def test_sampling_logging_and_resume(self):
        with tempfile.TemporaryDirectory(prefix="constraint-traverse-ppo-test-") as folder:
            root = Path(folder)
            output = root / "run"
            trainer, _ = self.trainer(root, output=output)
            with redirect_stdout(io.StringIO()):
                first = trainer.run(2)
            self.assertEqual(first["total_steps"], 8)
            self.assertTrue((output / "iteration_000002.pt").is_file())
            _, previous = load_checkpoint(output / "last.pt")
            resumed, opened = self.trainer(root, output=output, resume=output / "last.pt")
            with redirect_stdout(io.StringIO()):
                resumed._open_scene()
            for key, value in resumed.runner.alg.policy.state_dict().items():
                torch.testing.assert_close(value, previous["model_state_dict"][key], rtol=0, atol=0)
            optimizer = resumed.runner.alg.optimizer.state_dict()
            self.assertEqual(optimizer["param_groups"], previous["optimizer_state_dict"]["param_groups"])
            for index, state in optimizer["state"].items():
                for key, value in state.items():
                    torch.testing.assert_close(value, previous["optimizer_state_dict"]["state"][index][key], rtol=0, atol=0)
            self.assertEqual(opened[0][1], previous["training_state"]["scene_cursors"][0])
            with redirect_stdout(io.StringIO()):
                final = resumed.run(3)
            self.assertEqual(final["completed_iterations"], 3)
            records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([record["iteration"] for record in records], [1, 2, 3])
            self.assertGreater(records[0]["action_clipped_fraction"][0], 0)
            self.assertEqual(records[-1]["training_timeout_rate"], 1.0)
            self.assertEqual(load_config(output / "config.yaml"), trainer.config)
            self.assertTrue(all(env.closed for _, _, env in opened))

    def test_scene_rotation_preserves_optimizer_and_episode_progress(self):
        with tempfile.TemporaryDirectory(prefix="constraint-traverse-ppo-test-") as folder:
            root = Path(folder)
            trainer, opened = self.trainer(root, output=root / "run", groups=2)
            with redirect_stdout(io.StringIO()):
                trainer.run(3)
            self.assertEqual([name for name, _, _ in opened], ["room_0.usd", "room_1.usd", "room_0.usd"])
            self.assertGreater(opened[2][1][0], opened[0][1][0])
            self.assertTrue(all(env.closed for _, _, env in opened))
            steps = [float(state["step"]) for state in trainer.runner.alg.optimizer.state.values()]
            self.assertTrue(steps and all(step == 3 for step in steps))

    def test_recurrent_probability_matches_recorded_samples(self):
        for count in (1, 2):
            config = PpoConfig(num_envs=count, rollout_steps=4, num_learning_epochs=1)
            group = [SimpleNamespace(path=Path("episode"))]
            env = RecordedStartEnv(group, list(range(count)), config)
            runner = BevOnPolicyRunner(env, training_runner_config(MANIFEST, "lstm", config))
            obs = env.get_observations()
            with torch.inference_mode():
                for _ in range(config.rollout_steps):
                    actions = runner.alg.act(obs)
                    obs, rewards, dones, extras = env.step(actions)
                    runner.alg.process_env_step(obs, rewards, dones, extras)
                critic_state = tuple(value.clone() for value in runner.alg.policy.memory_c.hidden_state)
                compute_returns(runner.alg, obs)
                for expected, current in zip(critic_state, runner.alg.policy.memory_c.hidden_state):
                    torch.testing.assert_close(current, expected, rtol=0, atol=0)
            batch = next(runner.alg.storage.recurrent_mini_batch_generator(1, 1))
            observations, actions, _, _, _, old_log_prob, _, _, hidden, masks = batch
            runner.alg.policy.act(observations, masks=masks, hidden_state=hidden[0])
            current = runner.alg.policy.get_actions_log_prob(actions)
            self.assertEqual(current.shape, old_log_prob.squeeze().shape)
            torch.testing.assert_close(current, old_log_prob.squeeze(), atol=2e-5, rtol=2e-5)

    def test_scene_rotation_clears_stored_recurrent_state(self):
        with tempfile.TemporaryDirectory(prefix="constraint-traverse-ppo-test-") as folder:
            root = Path(folder)
            config = PpoConfig(rollout_steps=4, scene_iterations=2, num_learning_epochs=1)
            trainer, _ = self.trainer(root, output=root / "run", groups=2, config=config)
            original = trainer.environment_factory

            def factory(group, cursors):
                result = original(group, cursors)
                result.task.timeout_sec = 100.0
                return result

            trainer.environment_factory = factory
            with redirect_stdout(io.StringIO()):
                trainer.run(3)
            for saved in (trainer.runner.alg.storage.saved_hidden_state_a,
                          trainer.runner.alg.storage.saved_hidden_state_c):
                for value in saved:
                    self.assertEqual(float(value[0].abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
