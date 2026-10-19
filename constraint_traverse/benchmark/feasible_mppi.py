from __future__ import annotations

import torch
from pytorch_mppi import mppi


class FeasibleMPPI(mppi.MPPI):
    def __init__(self, *args, project_action, **kwargs):
        self.project_action = project_action
        self.action_priors = None
        super().__init__(*args, **kwargs)

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
