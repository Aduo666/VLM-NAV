from __future__ import annotations

import torch
from pytorch_mppi import mppi


class ConsistentKMPPI(mppi.KMPPI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.specific_action_sampler is not None:
            raise ValueError("KMPPI 自定义候选需要同时提供控制支撑点")
        indices = self.Tk[0]
        lower = indices.floor().long()
        upper = indices.ceil().long().clamp(max=self.T - 1)
        self.theta = torch.lerp(self.U[lower], self.U[upper], (indices - lower)[:, None])
        self.synchronize_controls()

    def synchronize_controls(self):
        self.theta = self._bound_action(self.theta)
        raw, _ = self.deparameterize_to_trajectory_single(self.theta)
        self.U = self._bound_action(raw)
        self.interpolation_excess = (raw - self.U).abs().amax()

    def shift_nominal_trajectory(self):
        self.theta, _ = self.do_kernel_interpolation(self.Tk[0] + 1, self.Tk[0], self.theta)
        self.synchronize_controls()

    def reset(self):
        super().reset()
        self.synchronize_controls()

    def _compute_perturbed_action_and_noise(self):
        points = self._bound_action(self.theta + self._sample_noise((self.K, self.num_support_pts)))
        # 停车候选在支撑点层生成，候选评分与参数更新使用同一组动作。
        if self.sample_null_action:
            points[0] = 0
        self.noise_theta = points - self.theta
        actions, _ = self.deparameterize_to_trajectory_batch(points)
        self.perturbed_action = self._bound_action(actions)
        self.noise = self.perturbed_action - self.U

    def _command(self, state):
        super()._command(state)
        self.synchronize_controls()
        action = self.U[:self.u_per_command]
        return action[0] if self.u_per_command == 1 else action
