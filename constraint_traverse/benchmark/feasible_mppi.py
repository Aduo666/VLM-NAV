from __future__ import annotations

import torch
from pytorch_mppi import mppi
from .weighting import AdaptiveWeights


class FeasibleMPPI(AdaptiveWeights, mppi.MPPI):
    def __init__(self, *args, project_action, **kwargs):
        self.project_action = project_action
        self.action_priors = None
        super().__init__(*args, **kwargs)

    def _setup_action_cost_fn(self):
        super()._setup_action_cost_fn()
        independent_cost = self._compute_action_cost

        def correlated_cost(noise):
            correlation = getattr(self, "noise_correlation", 0.85)
            if noise.shape[-2] < 2 or correlation == 0:
                return independent_cost(noise)
            precision = noise.clone()
            precision[:, 0] = noise[:, 0] - correlation * noise[:, 1]
            precision[:, -1] = noise[:, -1] - correlation * noise[:, -2]
            precision[:, 1:-1] = ((1 + correlation ** 2) * noise[:, 1:-1]
                                  - correlation * (noise[:, :-2] + noise[:, 2:]))
            return independent_cost(precision / (1 - correlation ** 2))

        self._compute_action_cost = correlated_cost

    def _compute_perturbed_action_and_noise(self):
        noise = self._sample_noise((self.K, self.T))
        correlation = getattr(self, "noise_correlation", 0.85)
        for index in range(1, self.T):
            noise[:, index] = correlation * noise[:, index - 1] + (1 - correlation ** 2) ** 0.5 * noise[:, index]
        action = self._sample_specific_actions(self.U + noise)
        self.perturbed_action = self._bound_action(action)
        self.noise = self.perturbed_action - self.U

    def _bound_action(self, action):
        return self.project_action(super()._bound_action(action))

    def _sample_specific_actions(self, action):
        action = super()._sample_specific_actions(action)
        if self.action_priors is not None:
            count = min(len(self.action_priors), len(action) - 1)
            action[1:1 + count] = self.action_priors[:count]
        return action

    def _command(self, state):
        super()._command(state)
        self.U = self._bound_action(self.U)
        action = self.U[:self.u_per_command]
        return action[0] if self.u_per_command == 1 else action
