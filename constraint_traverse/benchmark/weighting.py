from __future__ import annotations

import torch


class AdaptiveWeights:
    def _compute_weighting(self, cost_total):
        valid = torch.isfinite(cost_total)
        if getattr(self, "candidate_validity", None) is not None:
            valid = valid & self.candidate_validity()
        self.valid_candidates = valid
        count = int(valid.sum())
        if count == 0:
            self.omega = torch.zeros_like(cost_total)
            self.weight_temperature = float(self.lambda_)
            return self.omega
        shifted = cost_total - cost_total[valid].min()
        target = min(count, max(2.0, count * getattr(self, "ess_fraction", 0.12)))

        def weights(temperature):
            logits = torch.where(valid, -shifted / temperature, -torch.inf)
            return torch.softmax(logits, dim=0)

        low = max(float(self.lambda_), 1e-6)
        high = max(low, float(shifted[valid].max()) * 2 + 1)
        for _ in range(18):
            middle = (low + high) / 2
            omega = weights(middle)
            if float(1 / omega.square().sum()) < target:
                low = middle
            else:
                high = middle
        self.weight_temperature = high
        self.omega = weights(high)
        return self.omega
