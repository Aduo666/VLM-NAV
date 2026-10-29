from __future__ import annotations

import math
import subprocess
import sys
import unittest

import numpy as np

from constraint_traverse.active_passage.contracts import ExecutionOutcome, ExecutionStatus, LossSamples
from constraint_traverse.active_passage.evaluation import ExecutionLossConfig, RiskEvaluator, empirical_risk


def evaluator():
    return RiskEvaluator(ExecutionLossConfig(100, 20, 0.1, 10, 0.2), alpha=0.5, beta=1)


def samples(values, *, weights=None, environments=("A", "B"), noises=("seed0", "seed0")):
    return LossSamples(np.asarray(values), environments, noises, weights)


class RiskTests(unittest.TestCase):
    def test_upper_tail(self):
        risk = empirical_risk([0, 1, 2, 3], alpha=0.5, beta=2)
        self.assertAlmostEqual(risk.expectation, 1.5)
        self.assertAlmostEqual(risk.cvar, 2.5)
        self.assertAlmostEqual(risk.total, 6.5)

    def test_weighted_fractional_tail(self):
        risk = empirical_risk([0, 100], weights=[0.99, 0.01], alpha=0.95, beta=1)
        self.assertAlmostEqual(risk.expectation, 1)
        self.assertAlmostEqual(risk.cvar, 20)

    def test_quantile_inside_tied_samples(self):
        self.assertAlmostEqual(empirical_risk([0, 1, 1, 10], alpha=0.6, beta=0).cvar, 6.625)

    def test_zero_weight_unsafe_sample_is_ignored(self):
        risk = empirical_risk([2, 1e6], weights=[1, 0], alpha=0.99, beta=0)
        self.assertAlmostEqual(risk.cvar, 2)

    def test_zero_alpha_is_full_distribution_average(self):
        risk = empirical_risk([0, 100], weights=[0.9, 0.1], alpha=0, beta=0)
        self.assertAlmostEqual(risk.cvar, risk.expectation)

    def test_constant_loss(self):
        self.assertAlmostEqual(empirical_risk([7, 7, 7], alpha=0.99, beta=1).cvar, 7)

    def test_weights_are_normalized(self):
        self.assertAlmostEqual(empirical_risk([0, 10], weights=[3, 1], alpha=0.5, beta=0).cvar, 5)

    def test_invalid_risk_parameters(self):
        for alpha, beta in ((1, 0), (-0.1, 0), (math.nan, 0), (0.5, -1), (0.5, math.inf)):
            with self.subTest(alpha=alpha, beta=beta), self.assertRaises(ValueError):
                empirical_risk([0, 1], alpha=alpha, beta=beta)

    def test_invalid_samples(self):
        for values, weights in (([], None), ([math.nan], None), ([1, 2], [1]),
                                ([1, 2], [1, -1]), ([1, 2], [0, 0]), ([1], [math.inf])):
            with self.subTest(values=values, weights=weights), self.assertRaises(ValueError):
                empirical_risk(values, weights=weights, alpha=0.5, beta=1)


class ObservationValueTests(unittest.TestCase):
    def test_same_execution_distribution_only_costs_observation(self):
        result = evaluator().observation_value(samples([0, 100]), samples([0, 100]), observation_cost=3)
        self.assertAlmostEqual(result.value, -3)

    def test_conditional_branches_do_not_create_artificial_benefit(self):
        before = samples([0, 100])
        after = samples([0, 0, 100, 100], environments=("A", "A", "B", "B"), noises=("seed0",) * 4)
        self.assertAlmostEqual(evaluator().observation_value(before, after, observation_cost=3).value, -3)

    def test_improved_trajectory_has_positive_value(self):
        value = evaluator().observation_value(samples([0, 100]), samples([0, 20]), observation_cost=3)
        self.assertAlmostEqual(value.value, 117)

    def test_worse_trajectory_has_negative_value(self):
        value = evaluator().observation_value(samples([0, 100]), samples([0, 200]), observation_cost=3)
        self.assertAlmostEqual(value.value, -153)

    def test_changed_prior_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluator().observation_value(samples([0, 100]), samples([0, 100], weights=[0.9, 0.1]), observation_cost=0)

    def test_changed_execution_error_distribution_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluator().observation_value(samples([0, 100]), samples([0, 100], noises=("seed1", "seed1")), observation_cost=0)

    def test_weighted_observation_branches_preserve_prior(self):
        before = samples([10, 100], weights=[0.8, 0.2])
        after = samples([10, 10, 100], environments=("A", "A", "B"), noises=("seed0",) * 3,
                        weights=[0.3, 0.5, 0.2])
        self.assertAlmostEqual(evaluator().observation_value(before, after, observation_cost=2).value, -2)

    def test_invalid_observation_cost(self):
        for cost in (-1, math.inf, math.nan):
            with self.subTest(cost=cost), self.assertRaises(ValueError):
                evaluator().observation_value(samples([0, 1]), samples([0, 1]), observation_cost=cost)


class ExecutionLossTests(unittest.TestCase):
    def test_unfinished_states_receive_incomplete_cost(self):
        for status in (ExecutionStatus.STOPPED, ExecutionStatus.TIMEOUT, ExecutionStatus.PLANNING_FAILED):
            with self.subTest(status=status):
                result = evaluator().loss_components(ExecutionOutcome(status, False, 0, None))
                self.assertEqual(result.incomplete, 20)

    def test_collision_and_incomplete_are_both_counted(self):
        outcome = ExecutionOutcome(ExecutionStatus.COLLISION, True, 5, 0.1)
        self.assertAlmostEqual(evaluator().execution_loss(outcome), 123)

    def test_completion_and_collision_are_separate_metrics(self):
        result = evaluator().loss_components(ExecutionOutcome(ExecutionStatus.COMPLETED, True, 0, 0.2))
        self.assertEqual(result.incomplete, 0)
        self.assertEqual(result.collision, 100)

    def test_missing_executed_clearance_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluator().execution_loss(ExecutionOutcome(ExecutionStatus.COMPLETED, False, 1, None))

    def test_no_obstacle_has_no_clearance_cost(self):
        outcome = ExecutionOutcome(ExecutionStatus.COMPLETED, False, 1, math.inf)
        self.assertAlmostEqual(evaluator().execution_loss(outcome), 0.1)

    def test_import_has_no_simulator_dependency(self):
        code = ("import sys; import constraint_traverse.active_passage.evaluation; "
                "assert not any(name in sys.modules for name in ('isaacsim', 'isaaclab', 'omni', 'pxr', 'torch', 'vln_nav'))")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
