from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .contracts import ExecutionOutcome, ExecutionStatus, LossSamples, ObservationValue, RiskSummary


@dataclass(frozen=True)
class ExecutionLossConfig:
    collision_cost: float
    incomplete_cost: float
    time_cost_per_sec: float
    clearance_cost: float
    clearance_target_m: float

    def __post_init__(self):
        values = (self.collision_cost, self.incomplete_cost, self.time_cost_per_sec, self.clearance_cost)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("执行损失权重必须为非负有限数值")
        if not math.isfinite(self.clearance_target_m) or self.clearance_target_m <= 0:
            raise ValueError("间距目标必须为正数")


@dataclass(frozen=True)
class LossBreakdown:
    collision: float
    incomplete: float
    time: float
    clearance: float

    @property
    def total(self):
        return self.collision + self.incomplete + self.time + self.clearance


def _risk_parameters(alpha, beta):
    if not math.isfinite(alpha) or not 0 <= alpha < 1:
        raise ValueError("CVaR 置信水平 alpha 需要满足 0 <= alpha < 1")
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("CVaR 权重 beta 必须为非负有限数值")


def empirical_risk(losses, *, alpha, beta, weights=None) -> RiskSummary:
    _risk_parameters(alpha, beta)
    values = np.asarray(losses, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("风险评价需要非空的一维有限损失样本")
    probabilities = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=np.float64)
    if probabilities.shape != values.shape or not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("样本权重必须与损失对应，且为非负有限数值")
    total = float(probabilities.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("样本权重之和必须为正数")
    probabilities = probabilities / total
    expectation = float(probabilities @ values)
    order = np.argsort(-values, kind="stable")
    sorted_weights = probabilities[order]
    tail_mass = 1.0 - alpha
    mass_before = np.r_[0.0, np.cumsum(sorted_weights[:-1])]
    # 分位数落在离散样本内部时，只取进入最坏尾部的那部分概率质量。
    tail_weights = np.minimum(sorted_weights, np.maximum(0.0, tail_mass - mass_before))
    cvar = float(tail_weights @ values[order] / tail_mass)
    if not math.isfinite(expectation) or not math.isfinite(cvar) or not math.isfinite(expectation + beta * cvar):
        raise ValueError("风险计算结果超出有限数值范围")
    return RiskSummary(expectation, cvar, float(alpha), float(beta))


def _marginal_weights(samples: LossSamples):
    marginal = {}
    for environment, noise, weight in zip(samples.environment_ids, samples.execution_noise_ids, samples.weights):
        if weight > 0:
            key = (environment, noise)
            marginal[key] = marginal.get(key, 0.0) + float(weight)
    return marginal


class RiskEvaluator:
    def __init__(self, config: ExecutionLossConfig, *, alpha: float, beta: float):
        if not isinstance(config, ExecutionLossConfig):
            raise TypeError("执行损失配置类型无效")
        _risk_parameters(alpha, beta)
        self.config, self.alpha, self.beta = config, float(alpha), float(beta)

    def loss_components(self, outcome: ExecutionOutcome) -> LossBreakdown:
        if not isinstance(outcome, ExecutionOutcome):
            raise TypeError("执行损失需要 ExecutionOutcome")
        cfg = self.config
        distance = outcome.minimum_clearance_m
        if distance is None:
            if cfg.clearance_cost > 0 and (outcome.completed or outcome.traversal_duration_sec > 0):
                raise ValueError("已执行的通行过程缺少最小间距记录")
            clearance = 0.0
        else:
            shortage = max(0.0, (cfg.clearance_target_m - distance) / cfg.clearance_target_m)
            clearance = cfg.clearance_cost * shortage ** 2
        result = LossBreakdown(
            cfg.collision_cost * outcome.collided,
            cfg.incomplete_cost * (outcome.status is not ExecutionStatus.COMPLETED),
            cfg.time_cost_per_sec * outcome.traversal_duration_sec,
            clearance,
        )
        if not math.isfinite(result.total):
            raise ValueError("执行损失超出有限数值范围")
        return result

    def execution_loss(self, outcome: ExecutionOutcome) -> float:
        return self.loss_components(outcome).total

    def risk(self, samples: LossSamples) -> RiskSummary:
        if not isinstance(samples, LossSamples):
            raise TypeError("风险评价需要 LossSamples")
        return empirical_risk(samples.losses, alpha=self.alpha, beta=self.beta, weights=samples.weights)

    def observation_value(self, before: LossSamples, after: LossSamples, *, observation_cost: float) -> ObservationValue:
        if not isinstance(before, LossSamples) or not isinstance(after, LossSamples):
            raise TypeError("观察价值需要观察前后的 LossSamples")
        if not math.isfinite(observation_cost) or observation_cost < 0:
            raise ValueError("观察成本必须为非负有限数值")
        prior, posterior_executions = _marginal_weights(before), _marginal_weights(after)
        if prior.keys() != posterior_executions.keys() or any(
            not math.isclose(weight, posterior_executions[key], rel_tol=1e-9, abs_tol=0)
            for key, weight in prior.items()
        ):
            raise ValueError("观察前后的风险必须使用相同的事前环境与执行误差分布")
        return ObservationValue(self.risk(before), self.risk(after), float(observation_cost))
