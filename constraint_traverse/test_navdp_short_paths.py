from __future__ import annotations

import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch


SOURCE = Path("/Data/qiang/NavDP/baselines/navdp/policy_network.py")


class NavDPShortPathTests(unittest.TestCase):
    def test_real_pointgoal_method_preserves_short_trajectories(self):
        if not SOURCE.exists():
            self.skipTest("本机没有 NavDP 服务端源码")
        scheduler_module = types.ModuleType("diffusers.schedulers.scheduling_ddpm")
        scheduler_module.DDPMScheduler = object
        imports = {"diffusers": types.ModuleType("diffusers"),
                   "diffusers.schedulers": types.ModuleType("diffusers.schedulers"),
                   "diffusers.schedulers.scheduling_ddpm": scheduler_module,
                   "policy_backbone": types.ModuleType("policy_backbone")}
        spec = importlib.util.spec_from_file_location("verified_navdp_policy", SOURCE)
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", imports):
            spec.loader.exec_module(module)
        # 网络与调度器使用固定输出，验证真实 pointgoal 方法的轨迹后处理。
        for distance in (0.15, 0.23, 0.40, 0.60):
            action = torch.zeros(16, 24, 3)
            action[:, :, 0] = distance * 4 / 24
            scheduler = types.SimpleNamespace(config=types.SimpleNamespace(num_train_timesteps=1),
                                              timesteps=torch.tensor([0]), set_timesteps=lambda count: None,
                                              step=lambda **kwargs: types.SimpleNamespace(prev_sample=action))
            model = types.SimpleNamespace(
                device="cpu", predict_size=24, noise_scheduler=scheduler,
                rgbd_encoder=lambda images, depths: torch.zeros(1, 1, 4),
                point_encoder=lambda goals: torch.zeros(len(goals), 4),
                predict_noise=lambda *args: torch.zeros_like(action),
                predict_critic=lambda actions, rgbd: torch.arange(16, dtype=torch.float32),
            )
            paths, scores, selected, _ = module.NavDP_Policy.predict_pointgoal_action(
                model, np.asarray([[distance, 0, 0]]), None, None)
            np.testing.assert_allclose(paths[:, :, -1, 0], distance, atol=1.0e-6)
            np.testing.assert_allclose(selected[:, :, -1, 0], distance, atol=1.0e-6)
            self.assertEqual(float(scores.max()), 15)


if __name__ == "__main__":
    unittest.main()
